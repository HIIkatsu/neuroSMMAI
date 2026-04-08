"""
image_pipeline.py — Post-centric image pipeline orchestrator.

New architecture:
  post_text -> visual intent -> search queries -> candidate ranking -> accept or no-image

Key principle: IMAGE = f(POST), not f(CHANNEL).
Channel context is only used as a very weak fallback.

This module orchestrates the full flow and delegates to:
  - visual_intent.py for intent extraction
  - image_search.py for provider search and scoring
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

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
    visual_intent: VisualIntent | None = None
    candidates_evaluated: int = 0
    candidates_rejected: int = 0
    reject_reasons: list[str] = field(default_factory=list)

    @property
    def has_image(self) -> bool:
        return bool(self.image_url)


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
        # Check each word in the forbidden phrase
        words = forbidden_lower.split()
        if len(words) == 1:
            # Single word: use word boundary to avoid false positives
            if re.search(rf"\b{re.escape(forbidden_lower)}\b", meta_lower):
                return f"wrong_sense:{forbidden}"
        else:
            # Multi-word: check if all words are present
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
    # Enhanced: more generic stock patterns
    "smiling office people", "smiling businesswoman", "smiling businessman",
    "business success", "corporate success", "team celebration",
    "abstract dashboard", "abstract data", "abstract digital",
    "unrelated workshop portrait", "concept image",
    "business handshake", "partnership agreement",
    "innovation concept", "creativity concept",
    "global business", "world map business",
    "arrows growth", "rocket launch business",
]


def compute_generic_stock_penalty(meta_text: str, intent: VisualIntent) -> int:
    """Compute generic stock penalty for a candidate.

    Returns a negative penalty score (0 or negative).
    Only applies if the candidate has weak subject alignment.
    """
    if not meta_text:
        return 0

    meta_lower = meta_text.lower()
    stock_hits = sum(1 for gs in _GENERIC_STOCK_SIGNALS if gs in meta_lower)

    if stock_hits == 0:
        return 0

    # Check if there's strong subject alignment that overrides stock penalty
    if intent.main_subject:
        subject_words = intent.main_subject.lower().split()
        subject_hits = sum(1 for w in subject_words if w in meta_lower)
        if subject_hits >= 2:
            # Strong subject match overrides stock penalty
            return -5 * stock_hits  # Mild penalty only

    # No strong subject match -> full stock penalty
    return -20 * min(stock_hits, 3)


# ---------------------------------------------------------------------------
# Post-centric candidate scoring
# ---------------------------------------------------------------------------

def score_candidate(
    meta_text: str,
    intent: VisualIntent,
    query: str = "",
) -> tuple[int, str]:
    """Score an image candidate based on post-centric intent.

    Returns (score, reject_reason).
    reject_reason is non-empty when candidate should be hard-rejected.

    Scoring factors (post-centric, NOT channel-centric):
    1. Subject match: does the image match the post's main subject?
    2. Context/sense match: does the image match the disambiguated meaning?
    3. Scene match: does the image match the expected scene?
    4. Post-specific token match: direct word overlap with post content
    5. Generic stock penalty: penalize obvious generic stock images
    6. Wrong-sense hard reject: immediately reject wrong meaning
    """
    if not meta_text:
        return 0, "empty_meta"

    text = meta_text.strip().lower().replace("ё", "е")
    text = re.sub(r"\s+", " ", text)
    score = 0
    reject_reason = ""

    # --- Wrong-sense hard reject (FIRST — before any scoring) ---
    wrong_sense = check_wrong_sense(text, intent)
    if wrong_sense:
        return -100, wrong_sense

    # --- 1. Subject match (highest weight) ---
    subject_hits = 0
    if intent.main_subject:
        subject_words = [w for w in intent.main_subject.lower().split() if len(w) >= 3]
        subject_hits = sum(1 for w in subject_words if w in text)
        score += subject_hits * 14

        # Bonus for strong subject alignment
        if subject_hits >= 3:
            score += 15  # Strong subject bonus

    # --- 2. Context/sense match ---
    if intent.sense:
        sense_words = [w for w in intent.sense.lower().replace("_", " ").split() if len(w) >= 3]
        sense_hits = sum(1 for w in sense_words if w in text)
        score += sense_hits * 8

    # --- 3. Scene match ---
    scene_hits = 0
    if intent.scene:
        scene_words = [w for w in intent.scene.lower().split() if len(w) >= 3]
        scene_hits = sum(1 for w in scene_words if w in text)
        score += scene_hits * 7

    # --- 4. Query token match (search query alignment) ---
    if query:
        q_words = [w for w in query.lower().split() if len(w) >= 3]
        _stopwords = {
            "photo", "editorial", "realistic", "professional", "close",
            "fresh", "modern", "creative", "natural", "light",
        }
        q_relevant = [w for w in q_words if w not in _stopwords]
        query_hits = sum(1 for w in q_relevant if w in text)
        score += query_hits * 10

    # --- 5. Family term match (weak positive signal) ---
    family_hit_score = 0
    if intent.post_family in TOPIC_FAMILY_TERMS:
        terms = TOPIC_FAMILY_TERMS[intent.post_family].get("en", [])[:10]
        family_hits = sum(1 for t in terms if t in text)
        family_hit_score = family_hits * 5
        score += family_hit_score

    # --- 6. Allowed visual class match ---
    allowed = get_family_allowed_visuals(intent.post_family)
    allowed_hits = 0
    if allowed:
        allowed_hits = sum(1 for cls in allowed if cls in text)
        score += allowed_hits * 10

    # --- 7. Blocked visual class penalty ---
    blocked = get_family_blocked_visuals(intent.post_family)
    if blocked:
        for cls in blocked:
            if re.search(rf"\b{re.escape(cls)}\b", text, re.I):
                score -= 35
                if not reject_reason:
                    reject_reason = f"blocked_visual:{cls}"

    # --- 8. Cross-family detection ---
    # Detect what family the image metadata actually belongs to
    from image_search import _detect_meta_family
    meta_family = _detect_meta_family(text)
    if meta_family != "generic" and meta_family != intent.post_family:
        score -= 25
        if not reject_reason:
            reject_reason = f"cross_family:{meta_family}"

    # --- 9. Generic stock penalty ---
    stock_penalty = compute_generic_stock_penalty(text, intent)
    score += stock_penalty
    if stock_penalty <= -40 and not reject_reason:
        reject_reason = "generic_stock"

    # --- 10. Positive affirmation requirement ---
    # At least one real positive signal must be present to accept an image.
    # Thresholds: ≥1 subject word, ≥1 allowed class, or ≥2 scene words
    # (scene needs 2 because individual scene words like "setting" are too generic).
    _MIN_SUBJECT_HITS = 1
    _MIN_ALLOWED_HITS = 1
    _MIN_SCENE_HITS = 2
    has_affirmation = (
        (subject_hits >= _MIN_SUBJECT_HITS)
        or (allowed_hits >= _MIN_ALLOWED_HITS)
        or (scene_hits >= _MIN_SCENE_HITS)
    )
    if not has_affirmation and score > 0:
        score = min(score, 10)  # Cap score without affirmation
        if not reject_reason:
            reject_reason = "no_positive_affirmation"

    return score, reject_reason


# ---------------------------------------------------------------------------
# Validation for autopost (post-centric version)
# ---------------------------------------------------------------------------

def validate_image_post_centric(
    image_ref: str,
    *,
    intent: VisualIntent,
    image_meta: str = "",
) -> tuple[bool, str]:
    """Validate an image against the post's visual intent.

    Returns (is_valid, reject_reason).
    """
    if not image_ref or not image_ref.startswith("http"):
        return True, ""  # No image or local — nothing to reject

    # Check wrong sense
    if image_meta:
        wrong_sense = check_wrong_sense(image_meta, intent)
        if wrong_sense:
            logger.warning(
                "VALIDATE_REJECT_WRONG_SENSE reason=%s url=%r meta=%r",
                wrong_sense, image_ref[:80], image_meta[:120],
            )
            return False, wrong_sense

    # Check blocked visual classes
    if image_meta:
        blocked = get_family_blocked_visuals(intent.post_family)
        meta_lower = image_meta.lower()
        if blocked:
            for cls in blocked:
                if re.search(rf"\b{re.escape(cls)}\b", meta_lower, re.I):
                    reason = f"blocked_visual:{cls}"
                    logger.warning(
                        "VALIDATE_REJECT_BLOCKED reason=%s url=%r meta=%r",
                        reason, image_ref[:80], image_meta[:120],
                    )
                    return False, reason

    # Check minimum score
    if image_meta:
        score, reason = score_candidate(image_meta, intent)
        if score < 15:
            logger.warning(
                "VALIDATE_REJECT_LOW_SCORE score=%d reason=%s url=%r meta=%r",
                score, reason, image_ref[:80], image_meta[:120],
            )
            return False, reason or "low_score"

    return True, ""


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

async def run_image_pipeline(
    *,
    title: str = "",
    body: str = "",
    channel_topic: str = "",
    used_refs: set[str] | None = None,
) -> ImagePipelineResult:
    """Run the full post-centric image pipeline.

    Flow:
    1. Extract visual intent from post text
    2. Check visuality — skip if too abstract
    3. Build search queries from intent
    4. Search providers + rank candidates with post-centric scoring
    5. Apply wrong-sense hard reject
    6. Return best candidate or no-image with reason

    Channel topic is used ONLY as a weak fallback (step 1 in extract_visual_intent).
    """
    result = ImagePipelineResult()

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
        logger.info(
            "IMAGE_PIPELINE_SKIP reason=low_visuality title=%r",
            (title or "")[:60],
        )
        return result

    if intent.visuality == VISUALITY_LOW and not intent.main_subject:
        result.no_image_reason = "low_visuality_no_subject"
        logger.info(
            "IMAGE_PIPELINE_SKIP reason=low_visuality_no_subject title=%r",
            (title or "")[:60],
        )
        return result

    # --- Step 3: Get search queries ---
    queries = intent.search_queries
    if not queries:
        result.no_image_reason = "no_search_queries"
        return result

    # --- Step 4: Search and rank ---
    # Import search functions from image_search (they handle provider calls)
    from image_search import (
        _search_unsplash,
        _search_pexels,
        _search_pixabay,
        _make_client,
        _prepare_used_refs,
        _clean_text,
        MIN_RELEVANCE_SCORE,
    )

    used_fps = _prepare_used_refs(used_refs)

    # Use post family for provider queries (NOT channel family)
    family = intent.post_family

    best_url = ""
    best_score = -1
    best_source = ""
    best_query = ""
    best_meta = ""
    all_reject_reasons: list[str] = []

    logger.info(
        "IMAGE_PIPELINE_START family=%s subject=%r queries=%s visuality=%s",
        family, (intent.main_subject or "")[:40],
        [q[:40] for q in queries[:3]], intent.visuality,
    )

    # Use post-centric scoring threshold
    min_score = MIN_RELEVANCE_SCORE

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

                # Re-score with post-centric scoring
                pc_score, pc_reason = score_candidate(meta, intent, q)

                # Use the higher of provider score and post-centric score.
                # Rationale: the legacy provider scoring (_meta_score) is well-tested
                # and tuned for family-based matching; the new post-centric scoring
                # adds subject/sense/scene awareness. Taking max() ensures that a
                # candidate accepted by either system proceeds, preventing regressions
                # where the new scoring is stricter than the old one. Both systems
                # share the same hard-reject gates (wrong-sense, blocked visuals).
                effective_score = max(provider_score, pc_score)

                if effective_score < min_score:
                    result.candidates_rejected += 1
                    if pc_reason:
                        all_reject_reasons.append(pc_reason)
                    logger.info(
                        "IMAGE_PIPELINE_REJECT score=%d/%d source=%s query=%r "
                        "reason=%s meta=%r",
                        effective_score, min_score, source_name, q[:40],
                        pc_reason or "low_score", (meta or "")[:80],
                    )
                    continue

                # Wrong-sense hard reject
                wrong_sense = check_wrong_sense(meta or "", intent)
                if wrong_sense:
                    result.candidates_rejected += 1
                    all_reject_reasons.append(wrong_sense)
                    logger.info(
                        "IMAGE_PIPELINE_WRONG_SENSE source=%s query=%r "
                        "reason=%s meta=%r",
                        source_name, q[:40], wrong_sense, (meta or "")[:80],
                    )
                    continue

                if effective_score > best_score:
                    best_score = effective_score
                    best_url = url
                    best_source = source_name
                    best_query = q
                    best_meta = meta

                    # Early exit for very strong matches
                    if best_score >= min_score * 3:
                        break

    result.reject_reasons = list(set(all_reject_reasons))

    if best_url:
        result.image_url = best_url
        result.score = best_score
        result.source_provider = best_source
        result.matched_query = best_query
        logger.info(
            "IMAGE_PIPELINE_SELECTED source=%s score=%d query=%r "
            "subject=%r visuality=%s url=%r",
            best_source, best_score, best_query[:40],
            (intent.main_subject or "")[:40],
            intent.visuality, best_url[:80],
        )
    else:
        result.no_image_reason = _determine_no_image_reason(result, intent)
        logger.info(
            "IMAGE_PIPELINE_NO_IMAGE reason=%s evaluated=%d rejected=%d "
            "reject_reasons=%s",
            result.no_image_reason,
            result.candidates_evaluated,
            result.candidates_rejected,
            result.reject_reasons[:5],
        )

    return result


def _determine_no_image_reason(result: ImagePipelineResult, intent: VisualIntent) -> str:
    """Determine the most specific no-image reason from pipeline state."""
    if intent.no_image_reason:
        return intent.no_image_reason

    reasons = result.reject_reasons
    if not reasons and result.candidates_evaluated == 0:
        return "no_candidates"

    # Count reason categories
    wrong_sense_count = sum(1 for r in reasons if r.startswith("wrong_sense"))
    generic_stock_count = sum(1 for r in reasons if r == "generic_stock")
    blocked_count = sum(1 for r in reasons if r.startswith("blocked_visual"))
    cross_family_count = sum(1 for r in reasons if r.startswith("cross_family"))

    if wrong_sense_count > 0:
        return "wrong_sense"
    if generic_stock_count > 0:
        return "generic_stock"
    if blocked_count > 0:
        return "blocked_visual"
    if cross_family_count > 0:
        return "cross_family"
    if result.candidates_rejected > 0:
        return "low_subject_match"
    return "no_candidates"
