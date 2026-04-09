"""
image_pipeline.py — Post-centric image pipeline orchestrator (v2).

Architecture:
  post_text → visual intent → search queries → collect candidates →
  → score each (post-centric) → top-N rerank → final decision → outcome

Key principles:
  1. IMAGE = f(POST), not f(CHANNEL). Channel is weak fallback only.
  2. Post-centric score is PRIMARY. Provider/meta score is a secondary bonus.
  3. Hard reject gates fire BEFORE any scoring.
  4. Two modes: AUTOPOST (strict, prefers no-image over junk) and
     EDITOR (lenient, returns multiple tolerable candidates).
  5. Every candidate gets a CandidateTrace for runtime explainability.

Decision flow (per candidate):
  ┌─ hard_reject? → REJECT (wrong_sense / blocked_visual)
  │
  ├─ post_centric_score (subject + sense + scene + query tokens)
  │   └─ + provider_bonus (capped secondary signal)
  │
  ├─ generic_stock_penalty
  │
  ├─ positive_affirmation_check (must have ≥1 real positive signal)
  │
  └─ final_score → compare to mode threshold → ACCEPT / REJECT / NO_IMAGE
"""
from __future__ import annotations

import logging
import re
import time as _time
from collections import deque
from dataclasses import dataclass, field
from urllib.parse import urlparse as _urlparse

from visual_intent import (
    VisualIntent,
    extract_visual_intent,
    VISUALITY_HIGH,
    VISUALITY_MEDIUM,
    VISUALITY_LOW,
    VISUALITY_NONE,
)
from topic_utils import (
    TOPIC_FAMILY_TERMS,
    detect_topic_family,
    STRICT_IMAGE_FAMILIES,
    get_family_allowed_visuals,
    get_family_blocked_visuals,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Named constants (no magic numbers)
# ---------------------------------------------------------------------------

# Scoring weights — post-centric, NOT channel-centric
W_SUBJECT = 14          # Per subject-word hit (highest weight)
W_SUBJECT_STRONG = 15   # Bonus when ≥3 subject words match
W_SENSE = 8             # Per sense-word hit
W_SCENE = 7             # Per scene-word hit
W_QUERY_TOKEN = 10      # Per relevant query-token hit
W_FAMILY_TERM = 5       # Per family-term hit (weak signal)
W_ALLOWED_VISUAL = 10   # Per allowed visual class hit
P_BLOCKED_VISUAL = -35  # Per blocked visual class hit
P_CROSS_FAMILY = -25    # Cross-family mismatch
P_GENERIC_STOCK = -20   # Per generic stock signal (max 3 hits counted)
P_GENERIC_STOCK_MILD = -5  # Per stock signal when subject matches strongly

# Provider score handling — provider score is SECONDARY, capped
PROVIDER_BONUS_WEIGHT = 0.25   # Provider score contributes only 25% as bonus
PROVIDER_BONUS_CAP = 15        # Max points provider can add

# Mode-specific thresholds
AUTOPOST_MIN_SCORE = 25        # Strict: must have real confidence
EDITOR_MIN_SCORE = 4           # Very lenient: show weak-but-relevant candidates
EDITOR_SOFT_MIN = 8            # Preferred editor score; below = marked tolerable

# Affirmation thresholds — at least one must be met
AFFIRMATION_MIN_SUBJECT_HITS = 1
AFFIRMATION_MIN_ALLOWED_HITS = 1
AFFIRMATION_MIN_SCENE_HITS = 2   # Scene needs 2; single scene words too generic

# Score cap when no positive affirmation is found. Prevents low-confidence
# candidates from accumulating points purely from weak signals.
MAX_SCORE_WITHOUT_AFFIRMATION = 10

# Top-N reranking
TOP_N_CANDIDATES = 8    # Collect up to N candidates before final pick
EDITOR_TOP_N = 10       # Editor returns more candidates for user to pick from

# ---------------------------------------------------------------------------
# Generic filler patterns — autopost rejects these unless post explicitly
# matches the visual subject (e.g. "AI chip" image only for AI chip posts)
# ---------------------------------------------------------------------------
_AUTOPOST_GENERIC_FILLER = [
    "ai chip", "ai processor", "artificial intelligence chip",
    "code screen", "code on screen", "programming code",
    "binary code", "matrix code", "circuit board abstract",
    "random stock product", "generic product photo",
    "abstract technology", "digital transformation",
    "futuristic city", "robot hand", "robot face",
    "hologram", "virtual reality abstract",
    "neon lights abstract", "cyber background",
]


# ---------------------------------------------------------------------------
# Pipeline modes
# ---------------------------------------------------------------------------
MODE_AUTOPOST = "autopost"
MODE_EDITOR = "editor"


# ---------------------------------------------------------------------------
# Image diversity / anti-repeat memory (B)
# ---------------------------------------------------------------------------

# Maximum items to remember
_DIVERSITY_CACHE_SIZE = 50
_DIVERSITY_TTL_SECONDS = 3600 * 6  # 6 hours

# Penalty weights for diversity
P_REPEAT_URL = -200          # same exact URL → hard reject
P_REPEAT_VISUAL_CLASS = -15  # same visual class recently
P_REPEAT_DOMAIN = -8         # same provider domain recently
P_REPEAT_SUBJECT = -10       # same subject template recently


class _DiversityEntry:
    __slots__ = ("value", "ts")

    def __init__(self, value: str):
        self.value = value
        self.ts = _time.monotonic()


class ImageDiversityCache:
    """In-memory cache for recent image selections to prevent repetition.

    Thread-safe enough for single-process asyncio usage.
    """

    def __init__(self, maxlen: int = _DIVERSITY_CACHE_SIZE, ttl: float = _DIVERSITY_TTL_SECONDS):
        self.recent_urls: deque[_DiversityEntry] = deque(maxlen=maxlen)
        self.recent_visual_classes: deque[_DiversityEntry] = deque(maxlen=maxlen)
        self.recent_domains: deque[_DiversityEntry] = deque(maxlen=maxlen)
        self.recent_subjects: deque[_DiversityEntry] = deque(maxlen=maxlen)
        self._ttl = ttl

    def _alive(self, entry: _DiversityEntry) -> bool:
        return (_time.monotonic() - entry.ts) < self._ttl

    def _has(self, q: deque[_DiversityEntry], val: str) -> int:
        """Count how many alive entries match val."""
        return sum(1 for e in q if self._alive(e) and e.value == val)

    def compute_penalty(self, url: str, visual_class: str, domain: str, subject: str) -> int:
        """Compute diversity penalty for a candidate image."""
        penalty = 0
        url_norm = (url or "").strip().lower()
        if url_norm and self._has(self.recent_urls, url_norm) > 0:
            penalty += P_REPEAT_URL
        vc = (visual_class or "").strip().lower()
        if vc and self._has(self.recent_visual_classes, vc) > 0:
            penalty += P_REPEAT_VISUAL_CLASS
        dom = (domain or "").strip().lower()
        if dom:
            dom_count = self._has(self.recent_domains, dom)
            if dom_count >= 2:
                penalty += P_REPEAT_DOMAIN * 2
            elif dom_count >= 1:
                penalty += P_REPEAT_DOMAIN
        subj = (subject or "").strip().lower()
        if subj and self._has(self.recent_subjects, subj) > 0:
            penalty += P_REPEAT_SUBJECT
        return penalty

    def record(self, url: str, visual_class: str, domain: str, subject: str) -> None:
        """Record a selected image into the diversity cache."""
        if url:
            self.recent_urls.append(_DiversityEntry(url.strip().lower()))
        if visual_class:
            self.recent_visual_classes.append(_DiversityEntry(visual_class.strip().lower()))
        if domain:
            self.recent_domains.append(_DiversityEntry(domain.strip().lower()))
        if subject:
            self.recent_subjects.append(_DiversityEntry(subject.strip().lower()))

    def prune(self) -> None:
        """Remove expired entries."""
        now = _time.monotonic()
        for q in (self.recent_urls, self.recent_visual_classes, self.recent_domains, self.recent_subjects):
            while q and (now - q[0].ts) >= self._ttl:
                q.popleft()


# Global singleton diversity cache
_diversity_cache = ImageDiversityCache()


def get_diversity_cache() -> ImageDiversityCache:
    """Return the global image diversity cache."""
    return _diversity_cache


# ---------------------------------------------------------------------------
# Outcome types — explicit, semantic decision labels
# ---------------------------------------------------------------------------
OUTCOME_ACCEPT_BEST = "ACCEPT_BEST"
OUTCOME_ACCEPT_FOR_EDITOR = "ACCEPT_FOR_EDITOR_ONLY"
OUTCOME_REJECT_NO_MATCH = "REJECT_NO_MATCH"
OUTCOME_REJECT_WRONG_SENSE = "REJECT_WRONG_SENSE"
OUTCOME_REJECT_GENERIC_STOCK = "REJECT_GENERIC_STOCK"
OUTCOME_REJECT_CROSS_FAMILY = "REJECT_CROSS_FAMILY"
OUTCOME_REJECT_LOW_CONFIDENCE = "REJECT_LOW_CONFIDENCE"
OUTCOME_REJECT_REPEAT = "REJECT_REPEAT_IMAGE"
OUTCOME_REJECT_GENERIC_FILLER = "REJECT_GENERIC_FILLER"
OUTCOME_NO_IMAGE_SAFE = "NO_IMAGE_SAFE_FALLBACK"
OUTCOME_NO_IMAGE_LOW_VISUALITY = "NO_IMAGE_LOW_VISUALITY"
OUTCOME_NO_IMAGE_NO_CANDIDATES = "NO_IMAGE_NO_CANDIDATES"


# ---------------------------------------------------------------------------
# CandidateTrace — per-candidate debug/explainability record
# ---------------------------------------------------------------------------

@dataclass
class CandidateTrace:
    """Runtime trace for a single image candidate. Compact but sufficient
    for root-cause analysis of accept/reject decisions."""
    url: str = ""
    source: str = ""                # unsplash / pexels / pixabay
    query: str = ""                 # Search query that found this candidate
    meta_snippet: str = ""          # First 120 chars of metadata
    # Scoring breakdown
    subject_hits: int = 0
    sense_hits: int = 0
    scene_hits: int = 0
    query_token_hits: int = 0
    family_term_hits: int = 0
    allowed_visual_hits: int = 0
    generic_stock_hits: int = 0
    # Scores
    post_centric_score: int = 0
    provider_score: int = 0
    provider_bonus: int = 0
    diversity_penalty: int = 0      # Penalty from anti-repeat cache
    final_score: int = 0
    # Decision
    hard_reject: str = ""           # Non-empty if hard-rejected
    reject_reason: str = ""         # Soft reject reason
    outcome: str = ""               # One of OUTCOME_* constants

    def as_log_dict(self) -> dict:
        """Compact dict for structured logging."""
        return {
            "url": self.url[:80],
            "src": self.source,
            "q": self.query[:40],
            "subj": self.subject_hits,
            "sense": self.sense_hits,
            "scene": self.scene_hits,
            "qtok": self.query_token_hits,
            "stock": self.generic_stock_hits,
            "pc": self.post_centric_score,
            "prov": self.provider_score,
            "bonus": self.provider_bonus,
            "div": self.diversity_penalty,
            "final": self.final_score,
            "hard": self.hard_reject[:40] if self.hard_reject else "",
            "rej": self.reject_reason[:40] if self.reject_reason else "",
            "out": self.outcome,
        }


# ---------------------------------------------------------------------------
# Pipeline result
# ---------------------------------------------------------------------------

@dataclass
class ImagePipelineResult:
    """Result of the image pipeline."""
    image_url: str = ""
    score: int = 0
    source_provider: str = ""       # unsplash / pexels / pixabay
    matched_query: str = ""
    no_image_reason: str = ""       # Why no image was selected
    outcome: str = ""               # One of OUTCOME_* constants
    mode: str = MODE_AUTOPOST
    visual_intent: VisualIntent | None = None
    candidates_evaluated: int = 0
    candidates_rejected: int = 0
    reject_reasons: list[str] = field(default_factory=list)
    # For editor mode: multiple tolerable candidates
    editor_candidates: list[tuple[str, int]] = field(default_factory=list)
    # Full trace for debugging
    traces: list[CandidateTrace] = field(default_factory=list)

    @property
    def has_image(self) -> bool:
        return bool(self.image_url)

    def trace_summary(self) -> dict:
        """Compact runtime trace dict for logging/debugging image requests."""
        intent = self.visual_intent
        return {
            "mode": self.mode,
            "outcome": self.outcome,
            "image_url": (self.image_url or "")[:80],
            "score": self.score,
            "source": self.source_provider,
            "query": self.matched_query[:60] if self.matched_query else "",
            "no_image_reason": self.no_image_reason,
            "candidates_evaluated": self.candidates_evaluated,
            "candidates_rejected": self.candidates_rejected,
            "reject_reasons": self.reject_reasons[:5],
            "visual_subject": (intent.main_subject if intent else "")[:40],
            "visual_sense": (intent.sense if intent else "")[:30],
            "visuality": intent.visuality if intent else "",
            "search_queries": [q[:40] for q in (intent.search_queries if intent else [])[:3]],
            "editor_candidates_count": len(self.editor_candidates),
        }


# ---------------------------------------------------------------------------
# Cross-family detection (decoupled from image_search.py)
# ---------------------------------------------------------------------------
# Lightweight meta-family detection that does NOT depend on image_search.py.
# This allows score_candidate to be tested without httpx installed.

_META_FAMILY_KEYWORDS: dict[str, list[str]] = {
    "food": ["food", "dish", "recipe", "cooking", "restaurant", "kitchen", "meal",
             "chef", "cuisine", "baking", "dessert", "salad", "pizza", "coffee"],
    "cars": ["car", "vehicle", "automobile", "automotive", "engine", "sedan",
             "suv", "truck", "driving", "highway", "garage", "tire"],
    "massage": ["massage", "therapy", "spa", "relaxation", "bodywork",
                "wellness", "physiotherapy"],
    "beauty": ["beauty", "makeup", "cosmetic", "skincare", "manicure",
               "nail art", "hairstyle", "salon"],
    "health": ["health", "medical", "doctor", "hospital", "clinic",
               "fitness", "workout", "exercise", "gym"],
    "tech": ["technology", "software", "programming", "computer", "laptop",
             "smartphone", "server", "coding", "developer"],
    "finance": ["finance", "banking", "investment", "trading", "stock market",
                "cryptocurrency", "budget"],
    "education": ["education", "school", "university", "learning", "teaching",
                  "classroom", "student", "library"],
    "marketing": ["marketing", "advertising", "brand", "campaign", "social media",
                  "analytics", "content"],
    "local_business": ["repair", "plumbing", "electrician", "cleaning",
                       "construction", "renovation", "handyman"],
    "lifestyle": ["lifestyle", "travel", "fashion", "interior", "design",
                  "photography"],
}


def detect_meta_family(text: str) -> str:
    """Detect topic family from image metadata text.

    Lightweight version decoupled from image_search.py.
    Requires ≥2 keyword hits to declare a confident match.
    Returns family name or 'generic'.
    """
    if not text:
        return "generic"
    t = text.strip().lower()

    best_family = "generic"
    best_hits = 0
    for family, keywords in _META_FAMILY_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in t)
        if hits >= 2 and hits > best_hits:
            best_hits = hits
            best_family = family

    return best_family


# ---------------------------------------------------------------------------
# Wrong-sense hard reject
# ---------------------------------------------------------------------------

def check_wrong_sense(meta_text: str, intent: VisualIntent) -> str | None:
    """Check if image metadata contains a forbidden meaning.

    Returns the reject reason string if wrong sense detected, None otherwise.
    This is a HARD reject — candidate is immediately disqualified.
    """
    if not intent.forbidden_meanings or not meta_text:
        return None

    meta_lower = meta_text.lower()
    for forbidden in intent.forbidden_meanings:
        forbidden_lower = forbidden.lower()
        words = forbidden_lower.split()
        if len(words) == 1:
            if re.search(rf"\b{re.escape(forbidden_lower)}\b", meta_lower):
                return f"wrong_sense:{forbidden}"
        else:
            if all(w in meta_lower for w in words):
                return f"wrong_sense:{forbidden}"

    return None


# ---------------------------------------------------------------------------
# Generic stock detection (enhanced)
# ---------------------------------------------------------------------------

_GENERIC_STOCK_SIGNALS = [
    "stock photo", "shutterstock", "istockphoto", "getty images",
    "abstract background", "business team meeting", "happy people",
    "teamwork concept", "success concept", "idea concept",
    "generic office", "diverse group", "people working",
    "handshake", "brainstorm", "motivation", "inspiration concept",
    "copy space", "banner template", "mock up",
    "blank space", "placeholder", "presentation template",
    "growth chart", "puzzle pieces", "light bulb idea", "target goal",
    "thumbs up", "high five", "fist bump",
    "smiling office people", "smiling businesswoman", "smiling businessman",
    "business success", "corporate success", "team celebration",
    "abstract dashboard", "abstract data", "abstract digital",
    "unrelated workshop portrait", "concept image",
    "business handshake", "partnership agreement",
    "innovation concept", "creativity concept",
    "global business", "world map business",
    "arrows growth", "rocket launch business",
]


def compute_generic_stock_penalty(meta_text: str, intent: VisualIntent) -> tuple[int, int]:
    """Compute generic stock penalty for a candidate.

    Returns (penalty_score, stock_hit_count).
    penalty_score is 0 or negative.
    """
    if not meta_text:
        return 0, 0

    meta_lower = meta_text.lower()
    stock_hits = sum(1 for gs in _GENERIC_STOCK_SIGNALS if gs in meta_lower)

    if stock_hits == 0:
        return 0, 0

    # Check if there's strong subject alignment that overrides stock penalty
    if intent.main_subject:
        subject_words = intent.main_subject.lower().split()
        subject_hits = sum(1 for w in subject_words if w in meta_lower)
        if subject_hits >= 2:
            return P_GENERIC_STOCK_MILD * stock_hits, stock_hits

    return P_GENERIC_STOCK * min(stock_hits, 3), stock_hits


# ---------------------------------------------------------------------------
# Post-centric candidate scoring (v2 — no image_search dependency)
# ---------------------------------------------------------------------------

def score_candidate(
    meta_text: str,
    intent: VisualIntent,
    query: str = "",
) -> tuple[int, str, CandidateTrace]:
    """Score an image candidate based purely on post-centric intent.

    Returns (score, reject_reason, trace).

    This is the PRIMARY scoring function. Provider score is handled
    separately as a capped bonus by the pipeline orchestrator.

    Decision flow:
    1. Hard reject gates (wrong sense, empty meta)
    2. Subject match (highest weight)
    3. Sense/context match
    4. Scene match
    5. Query token match
    6. Family term match (weak)
    7. Allowed visual class match
    8. Blocked visual class penalty
    9. Cross-family penalty (using local detection, no image_search import)
    10. Generic stock penalty
    11. Positive affirmation requirement
    """
    trace = CandidateTrace()

    if not meta_text:
        trace.reject_reason = "empty_meta"
        return 0, "empty_meta", trace

    text = meta_text.strip().lower().replace("ё", "е")
    text = re.sub(r"\s+", " ", text)
    trace.meta_snippet = text[:120]

    score = 0
    reject_reason = ""

    # --- Hard reject: wrong sense (FIRST — before any scoring) ---
    wrong_sense = check_wrong_sense(text, intent)
    if wrong_sense:
        trace.hard_reject = wrong_sense
        trace.post_centric_score = -100
        return -100, wrong_sense, trace

    # --- 1. Subject match (highest weight) ---
    subject_hits = 0
    if intent.main_subject:
        subject_words = [w for w in intent.main_subject.lower().split() if len(w) >= 3]
        subject_hits = sum(1 for w in subject_words if w in text)
        score += subject_hits * W_SUBJECT
        if subject_hits >= 3:
            score += W_SUBJECT_STRONG
    trace.subject_hits = subject_hits

    # --- 2. Sense/context match ---
    sense_hits = 0
    if intent.sense:
        sense_words = [w for w in intent.sense.lower().replace("_", " ").split() if len(w) >= 3]
        sense_hits = sum(1 for w in sense_words if w in text)
        score += sense_hits * W_SENSE
    trace.sense_hits = sense_hits

    # --- 3. Scene match ---
    scene_hits = 0
    if intent.scene:
        scene_words = [w for w in intent.scene.lower().split() if len(w) >= 3]
        scene_hits = sum(1 for w in scene_words if w in text)
        score += scene_hits * W_SCENE
    trace.scene_hits = scene_hits

    # --- 4. Query token match ---
    query_token_hits = 0
    if query:
        q_words = [w for w in query.lower().split() if len(w) >= 3]
        _stopwords = {
            "photo", "editorial", "realistic", "professional", "close",
            "fresh", "modern", "creative", "natural", "light",
        }
        q_relevant = [w for w in q_words if w not in _stopwords]
        query_token_hits = sum(1 for w in q_relevant if w in text)
        score += query_token_hits * W_QUERY_TOKEN
    trace.query_token_hits = query_token_hits

    # --- 5. Family term match (weak signal) ---
    family_term_hits = 0
    if intent.post_family in TOPIC_FAMILY_TERMS:
        terms = TOPIC_FAMILY_TERMS[intent.post_family].get("en", [])[:10]
        family_term_hits = sum(1 for t in terms if t in text)
        score += family_term_hits * W_FAMILY_TERM
    trace.family_term_hits = family_term_hits

    # --- 6. Allowed visual class match ---
    allowed = get_family_allowed_visuals(intent.post_family)
    allowed_hits = 0
    if allowed:
        allowed_hits = sum(1 for cls in allowed if cls in text)
        score += allowed_hits * W_ALLOWED_VISUAL
    trace.allowed_visual_hits = allowed_hits

    # --- 7. Blocked visual class penalty ---
    # Multiple blocked visuals CAN accumulate: if an image matches several
    # off-topic categories, each adds a penalty. This is intentional to
    # strongly reject candidates that are clearly from the wrong domain.
    blocked = get_family_blocked_visuals(intent.post_family)
    if blocked:
        for cls in blocked:
            if re.search(rf"\b{re.escape(cls)}\b", text, re.I):
                score += P_BLOCKED_VISUAL
                if not reject_reason:
                    reject_reason = f"blocked_visual:{cls}"

    # --- 8. Cross-family detection (local, no image_search import) ---
    meta_family = detect_meta_family(text)
    if meta_family != "generic" and meta_family != intent.post_family:
        score += P_CROSS_FAMILY
        if not reject_reason:
            reject_reason = f"cross_family:{meta_family}"

    # --- 9. Generic stock penalty ---
    stock_penalty, stock_hits = compute_generic_stock_penalty(text, intent)
    score += stock_penalty
    trace.generic_stock_hits = stock_hits
    if stock_penalty <= -40 and not reject_reason:
        reject_reason = "generic_stock"

    # --- 10. Positive affirmation requirement ---
    has_affirmation = (
        (subject_hits >= AFFIRMATION_MIN_SUBJECT_HITS)
        or (allowed_hits >= AFFIRMATION_MIN_ALLOWED_HITS)
        or (scene_hits >= AFFIRMATION_MIN_SCENE_HITS)
    )
    if not has_affirmation and score > 0:
        score = min(score, MAX_SCORE_WITHOUT_AFFIRMATION)
        if not reject_reason:
            reject_reason = "no_positive_affirmation"

    # --- 11. Autopost generic filler detection ---
    # Detect generic tech/AI filler images (ai chip, code screen, etc.)
    # These are only penalized if the post subject does NOT explicitly match.
    filler_hits = sum(1 for f in _AUTOPOST_GENERIC_FILLER if f in text)
    if filler_hits > 0 and subject_hits < 2:
        score -= filler_hits * 15
        if not reject_reason:
            reject_reason = "generic_filler"

    trace.post_centric_score = score
    trace.reject_reason = reject_reason
    return score, reject_reason, trace


# ---------------------------------------------------------------------------
# Compute final score: post-centric primary + provider bonus (capped)
# ---------------------------------------------------------------------------

def compute_final_score(pc_score: int, provider_score: int) -> tuple[int, int]:
    """Compute final score with provider as capped secondary signal.

    The post-centric score is the PRIMARY signal.
    Provider score contributes a CAPPED bonus on top.

    Returns (final_score, provider_bonus_applied).
    """
    provider_bonus = min(
        int(max(provider_score, 0) * PROVIDER_BONUS_WEIGHT),
        PROVIDER_BONUS_CAP,
    )
    final = pc_score + provider_bonus
    return final, provider_bonus


# ---------------------------------------------------------------------------
# Determine outcome for a scored candidate
# ---------------------------------------------------------------------------

def determine_candidate_outcome(
    trace: CandidateTrace,
    mode: str = MODE_AUTOPOST,
) -> str:
    """Determine the outcome type for a scored candidate.

    Uses mode-specific thresholds.
    Editor mode: high recall — show weak-but-tolerable candidates.
    Autopost mode: high precision — reject generic filler, repeats.
    """
    if trace.hard_reject:
        if "wrong_sense" in trace.hard_reject:
            return OUTCOME_REJECT_WRONG_SENSE
        return OUTCOME_REJECT_CROSS_FAMILY

    # Hard reject repeated images in autopost
    if mode == MODE_AUTOPOST and trace.diversity_penalty <= P_REPEAT_URL:
        return OUTCOME_REJECT_REPEAT

    min_score = AUTOPOST_MIN_SCORE if mode == MODE_AUTOPOST else EDITOR_MIN_SCORE
    score = trace.final_score

    if score < min_score:
        if trace.generic_stock_hits >= 2:
            return OUTCOME_REJECT_GENERIC_STOCK
        if trace.reject_reason and "cross_family" in trace.reject_reason:
            return OUTCOME_REJECT_CROSS_FAMILY
        if trace.reject_reason and "wrong_sense" in trace.reject_reason:
            return OUTCOME_REJECT_WRONG_SENSE
        if trace.reject_reason and "generic_filler" in trace.reject_reason:
            return OUTCOME_REJECT_GENERIC_FILLER
        if score < EDITOR_MIN_SCORE:
            return OUTCOME_REJECT_NO_MATCH
        if mode == MODE_AUTOPOST:
            return OUTCOME_REJECT_LOW_CONFIDENCE
        return OUTCOME_ACCEPT_FOR_EDITOR

    # Score meets threshold
    if mode == MODE_AUTOPOST:
        return OUTCOME_ACCEPT_BEST
    if score >= AUTOPOST_MIN_SCORE:
        return OUTCOME_ACCEPT_BEST
    return OUTCOME_ACCEPT_FOR_EDITOR


# ---------------------------------------------------------------------------
# Validation for autopost (post-centric version)
# ---------------------------------------------------------------------------

def validate_image_post_centric(
    image_ref: str,
    *,
    intent: VisualIntent,
    image_meta: str = "",
    mode: str = MODE_AUTOPOST,
) -> tuple[bool, str]:
    """Validate an image against the post's visual intent.

    Returns (is_valid, reject_reason).
    """
    if not image_ref or not image_ref.startswith("http"):
        return True, ""  # No image or local — nothing to reject

    if image_meta:
        wrong_sense = check_wrong_sense(image_meta, intent)
        if wrong_sense:
            logger.warning(
                "VALIDATE_REJECT reason=%s url=%r meta=%r",
                wrong_sense, image_ref[:80], image_meta[:120],
            )
            return False, wrong_sense

        blocked = get_family_blocked_visuals(intent.post_family)
        meta_lower = image_meta.lower()
        if blocked:
            for cls in blocked:
                if re.search(rf"\b{re.escape(cls)}\b", meta_lower, re.I):
                    reason = f"blocked_visual:{cls}"
                    logger.warning(
                        "VALIDATE_REJECT reason=%s url=%r meta=%r",
                        reason, image_ref[:80], image_meta[:120],
                    )
                    return False, reason

        min_score = AUTOPOST_MIN_SCORE if mode == MODE_AUTOPOST else EDITOR_MIN_SCORE
        score, reason, _trace = score_candidate(image_meta, intent)
        if score < min_score:
            logger.warning(
                "VALIDATE_REJECT score=%d min=%d reason=%s url=%r",
                score, min_score, reason, image_ref[:80],
            )
            return False, reason or "low_score"

    return True, ""


# ---------------------------------------------------------------------------
# Main pipeline entry point (v2)
# ---------------------------------------------------------------------------

async def run_image_pipeline(
    *,
    title: str = "",
    body: str = "",
    channel_topic: str = "",
    used_refs: set[str] | None = None,
    mode: str = MODE_AUTOPOST,
) -> ImagePipelineResult:
    """Run the full post-centric image pipeline (v2).

    Flow:
    1. Extract visual intent from post text
    2. Check visuality — skip if too abstract
    3. Build search queries from intent
    4. Collect ALL candidates from providers (broad retrieval)
    5. Score each candidate with post-centric scoring
    6. Add provider bonus (capped secondary signal)
    7. Top-N reranking by final score
    8. Apply mode-specific threshold → pick best or no-image
    9. Log full trace for explainability

    Channel topic is used ONLY as a weak fallback (step 1 in extract_visual_intent).
    Provider score is SECONDARY — cannot rescue a low post-centric score.
    """
    result = ImagePipelineResult(mode=mode)

    # --- Step 1: Extract visual intent ---
    intent = extract_visual_intent(
        title=title,
        body=body,
        channel_topic=channel_topic,
    )
    result.visual_intent = intent

    # --- Step 2: Check visuality ---
    if intent.visuality == VISUALITY_NONE:
        result.no_image_reason = "low_visuality"
        result.outcome = OUTCOME_NO_IMAGE_LOW_VISUALITY
        logger.info(
            "IMAGE_PIPELINE outcome=%s reason=low_visuality mode=%s title=%r",
            result.outcome, mode, (title or "")[:60],
        )
        return result

    if intent.visuality == VISUALITY_LOW and not intent.main_subject:
        result.no_image_reason = "low_visuality_no_subject"
        result.outcome = OUTCOME_NO_IMAGE_LOW_VISUALITY
        logger.info(
            "IMAGE_PIPELINE outcome=%s reason=low_visuality_no_subject mode=%s title=%r",
            result.outcome, mode, (title or "")[:60],
        )
        return result

    # --- Step 3: Get search queries ---
    queries = intent.search_queries
    if not queries:
        result.no_image_reason = "no_search_queries"
        result.outcome = OUTCOME_NO_IMAGE_NO_CANDIDATES
        return result

    # --- Step 4: Collect candidates from providers ---
    from image_search import (
        _search_unsplash,
        _search_pexels,
        _search_pixabay,
        _make_client,
        _prepare_used_refs,
    )

    used_fps = _prepare_used_refs(used_refs)
    family = intent.post_family

    candidates: list[CandidateTrace] = []

    logger.info(
        "IMAGE_PIPELINE_START mode=%s family=%s subject=%r queries=%s visuality=%s",
        mode, family, (intent.main_subject or "")[:40],
        [q[:40] for q in queries[:3]], intent.visuality,
    )

    async with _make_client() as client:
        for source_name, search_fn in [
            ("unsplash", _search_unsplash),
            ("pexels", _search_pexels),
            ("pixabay", _search_pixabay),
        ]:
            for q in queries:
                url, provider_score, meta = await search_fn(
                    client, q, used_fps, family, f"{title} {body}",
                )
                result.candidates_evaluated += 1

                if not url:
                    continue

                # --- Step 5: Score with post-centric scoring ---
                pc_score, pc_reason, trace = score_candidate(meta, intent, q)
                trace.url = url
                trace.source = source_name
                trace.query = q
                trace.provider_score = provider_score

                # --- Step 6: Add provider bonus (CAPPED) ---
                final_score, provider_bonus = compute_final_score(pc_score, provider_score)
                trace.provider_bonus = provider_bonus
                trace.final_score = final_score

                # Determine outcome
                trace.outcome = determine_candidate_outcome(trace, mode)

                candidates.append(trace)

                if trace.hard_reject or trace.outcome.startswith("REJECT"):
                    result.candidates_rejected += 1
                    if pc_reason:
                        result.reject_reasons.append(pc_reason)

    result.traces = candidates

    # --- Step 7: Apply diversity penalties from anti-repeat cache ---
    div_cache = get_diversity_cache()
    for c in candidates:
        if c.hard_reject:
            continue
        # Extract domain from URL
        url_domain = ""
        if c.url:
            try:
                url_domain = _urlparse(c.url).netloc
            except Exception:
                pass
        # Detect visual class from meta snippet
        vc = detect_meta_family(c.meta_snippet)
        div_penalty = div_cache.compute_penalty(
            url=c.url,
            visual_class=vc,
            domain=url_domain,
            subject=intent.main_subject or "",
        )
        if div_penalty != 0:
            c.diversity_penalty = div_penalty
            c.final_score += div_penalty
            # Re-determine outcome with updated score
            c.outcome = determine_candidate_outcome(c, mode)
            if c.outcome.startswith("REJECT"):
                result.candidates_rejected += 1
                if not c.reject_reason:
                    c.reject_reason = "repeat_image"
                result.reject_reasons.append(c.reject_reason)

    # --- Step 8: Top-N reranking ---
    accepted = [
        c for c in candidates
        if not c.hard_reject and not c.outcome.startswith("REJECT")
    ]
    accepted.sort(key=lambda c: c.final_score, reverse=True)
    top_n_limit = EDITOR_TOP_N if mode == MODE_EDITOR else TOP_N_CANDIDATES
    top_n = accepted[:top_n_limit]

    # --- Step 9: Mode-specific final decision ---
    if top_n:
        best = top_n[0]
        min_score = AUTOPOST_MIN_SCORE if mode == MODE_AUTOPOST else EDITOR_MIN_SCORE

        if best.final_score >= min_score:
            result.image_url = best.url
            result.score = best.final_score
            result.source_provider = best.source
            result.matched_query = best.query
            result.outcome = best.outcome

            if mode == MODE_EDITOR:
                result.editor_candidates = [
                    (c.url, c.final_score) for c in top_n[:10]
                    if c.final_score >= EDITOR_MIN_SCORE
                ]

            # Record selection in diversity cache
            _record_selection(best, intent)

            logger.info(
                "IMAGE_PIPELINE_SELECTED outcome=%s mode=%s source=%s "
                "final_score=%d pc=%d prov=%d+%d div=%d query=%r "
                "subject=%r subj_hits=%d sense_hits=%d scene_hits=%d "
                "stock_hits=%d url=%r",
                result.outcome, mode, best.source,
                best.final_score, best.post_centric_score,
                best.provider_score, best.provider_bonus,
                best.diversity_penalty,
                best.query[:40],
                (intent.main_subject or "")[:40],
                best.subject_hits, best.sense_hits, best.scene_hits,
                best.generic_stock_hits, best.url[:80],
            )
            return result

        if mode == MODE_EDITOR and best.final_score >= EDITOR_MIN_SCORE:
            result.image_url = best.url
            result.score = best.final_score
            result.outcome = OUTCOME_ACCEPT_FOR_EDITOR
            result.source_provider = best.source
            result.matched_query = best.query
            result.editor_candidates = [
                (c.url, c.final_score) for c in top_n[:10]
                if c.final_score >= EDITOR_MIN_SCORE
            ]
            _record_selection(best, intent)
            return result

    # --- No acceptable candidate found ---

    # Editor mode fallback: if strict scoring rejected everyone but there are
    # non-hard-rejected candidates, surface the best ones as tolerable options.
    # This prevents the common "картинка не найдена" for typical queries.
    if mode == MODE_EDITOR and candidates:
        soft_rejected = [
            c for c in candidates
            if not c.hard_reject and c.final_score > 0
        ]
        soft_rejected.sort(key=lambda c: c.final_score, reverse=True)
        if soft_rejected:
            best_soft = soft_rejected[0]
            result.image_url = best_soft.url
            result.score = best_soft.final_score
            result.source_provider = best_soft.source
            result.matched_query = best_soft.query
            result.outcome = OUTCOME_ACCEPT_FOR_EDITOR
            result.editor_candidates = [
                (c.url, c.final_score) for c in soft_rejected[:10]
            ]
            _record_selection(best_soft, intent)
            logger.info(
                "IMAGE_PIPELINE_EDITOR_FALLBACK score=%d source=%s query=%r url=%r",
                best_soft.final_score, best_soft.source,
                best_soft.query[:40], best_soft.url[:80],
            )
            return result

    result.no_image_reason = _determine_no_image_reason(result, intent)
    result.outcome = _no_image_reason_to_outcome(result.no_image_reason)
    result.reject_reasons = list(set(result.reject_reasons))

    # Log trace summary for top rejected candidates
    for trace in candidates[:5]:
        logger.info("IMAGE_PIPELINE_TRACE %s", trace.as_log_dict())

    logger.info(
        "IMAGE_PIPELINE_NO_IMAGE outcome=%s reason=%s mode=%s "
        "evaluated=%d rejected=%d reject_reasons=%s",
        result.outcome, result.no_image_reason, mode,
        result.candidates_evaluated, result.candidates_rejected,
        result.reject_reasons[:5],
    )

    return result


def _record_selection(trace: CandidateTrace, intent: VisualIntent) -> None:
    """Record an accepted image in the diversity cache to prevent repetition."""
    cache = get_diversity_cache()
    url_domain = ""
    if trace.url:
        try:
            url_domain = _urlparse(trace.url).netloc
        except Exception:
            pass
    vc = detect_meta_family(trace.meta_snippet)
    cache.record(
        url=trace.url,
        visual_class=vc,
        domain=url_domain,
        subject=intent.main_subject or "",
    )


def _determine_no_image_reason(result: ImagePipelineResult, intent: VisualIntent) -> str:
    """Determine the most specific no-image reason from pipeline state."""
    if intent.no_image_reason:
        return intent.no_image_reason

    reasons = result.reject_reasons
    if not reasons and result.candidates_evaluated == 0:
        return "no_candidates"

    wrong_sense_count = sum(1 for r in reasons if r.startswith("wrong_sense"))
    generic_stock_count = sum(1 for r in reasons if r == "generic_stock")
    blocked_count = sum(1 for r in reasons if r.startswith("blocked_visual"))
    cross_family_count = sum(1 for r in reasons if r.startswith("cross_family"))
    repeat_count = sum(1 for r in reasons if r == "repeat_image")
    filler_count = sum(1 for r in reasons if r == "generic_filler")

    if wrong_sense_count > 0:
        return "wrong_sense"
    if repeat_count > 0:
        return "repeat_image"
    if filler_count > 0:
        return "generic_filler"
    if generic_stock_count > 0:
        return "generic_stock"
    if blocked_count > 0:
        return "blocked_visual"
    if cross_family_count > 0:
        return "cross_family"
    if result.candidates_rejected > 0:
        return "low_subject_match"
    return "no_candidates"


def _no_image_reason_to_outcome(reason: str) -> str:
    """Map a no-image reason string to an OUTCOME_* constant."""
    mapping = {
        "low_visuality": OUTCOME_NO_IMAGE_LOW_VISUALITY,
        "low_visuality_no_subject": OUTCOME_NO_IMAGE_LOW_VISUALITY,
        "no_search_queries": OUTCOME_NO_IMAGE_NO_CANDIDATES,
        "no_candidates": OUTCOME_NO_IMAGE_NO_CANDIDATES,
        "wrong_sense": OUTCOME_REJECT_WRONG_SENSE,
        "generic_stock": OUTCOME_REJECT_GENERIC_STOCK,
        "generic_filler": OUTCOME_REJECT_GENERIC_FILLER,
        "repeat_image": OUTCOME_REJECT_REPEAT,
        "blocked_visual": OUTCOME_REJECT_CROSS_FAMILY,
        "cross_family": OUTCOME_REJECT_CROSS_FAMILY,
        "low_subject_match": OUTCOME_REJECT_LOW_CONFIDENCE,
        "weak_subject": OUTCOME_NO_IMAGE_SAFE,
        "no_visual_subject": OUTCOME_NO_IMAGE_SAFE,
    }
    return mapping.get(reason, OUTCOME_NO_IMAGE_SAFE)
