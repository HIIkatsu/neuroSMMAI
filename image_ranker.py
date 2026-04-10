"""
image_ranker.py — Candidate scoring and ranking for image pipeline v3.

Scores each candidate against the post's VisualIntentV2.
Produces transparent, explainable per-candidate score breakdowns.

Scoring dimensions:
  1. subject_match   — Does the image match the post's main subject?
  2. sense_match     — Does it match the disambiguated meaning?
  3. scene_match     — Does it match the expected visual scene?
  4. generic_stock_penalty — Is it generic stock filler?
  5. repeat_penalty   — Has this image/type been used recently?
  6. provider_bonus   — Small bonus for provider confidence (CAPPED)

Key rules:
  - provider_bonus cannot rescue a weak candidate
  - hard reject gates fire BEFORE scoring
  - affirmation requirement: ≥1 real positive signal needed
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from visual_intent_v2 import VisualIntentV2, IMAGEABILITY_LOW, IMAGEABILITY_NONE
from image_history import ImageHistory, extract_domain, url_content_hash
from topic_utils import (
    TOPIC_FAMILY_TERMS,
    get_family_allowed_visuals,
    get_family_blocked_visuals,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scoring weights (named constants, no magic numbers)
# ---------------------------------------------------------------------------

# Post-centric weights (PRIMARY)
W_SUBJECT = 14              # Per subject-word hit (highest weight)
W_SUBJECT_STRONG = 15       # Bonus when ≥3 subject words match
W_SENSE = 8                 # Per sense-word hit
W_SCENE = 7                 # Per scene-word hit
W_QUERY_TOKEN = 10          # Per relevant query-token hit
W_FAMILY_TERM = 5           # Per family-term hit (weak signal)
W_ALLOWED_VISUAL = 10       # Per allowed visual class hit

# Penalties
P_BLOCKED_VISUAL = -35      # Per blocked visual class hit
P_CROSS_FAMILY = -25        # Cross-family mismatch
P_GENERIC_STOCK = -20       # Per generic stock signal (max 3 counted)
P_GENERIC_STOCK_MILD = -5   # Per stock signal when subject matches strongly
P_GENERIC_FILLER_PER = -15  # Per generic filler hit when subject doesn't match

# Provider bonus — SECONDARY, CAPPED
PROVIDER_BONUS_WEIGHT = 0.25
PROVIDER_BONUS_CAP = 15

# Affirmation requirements
AFFIRMATION_MIN_SUBJECT_HITS = 1
AFFIRMATION_MIN_ALLOWED_HITS = 1
AFFIRMATION_MIN_SCENE_HITS = 2
MAX_SCORE_WITHOUT_AFFIRMATION = 10

# Mode-specific thresholds
AUTOPOST_MIN_SCORE = 25     # Strict: real confidence required
EDITOR_MIN_SCORE = 4        # Lenient: show weak-but-relevant
EDITOR_SOFT_MIN = 8         # Preferred editor score

# Top-N reranking limits
TOP_N_AUTOPOST = 8
TOP_N_EDITOR = 10


# Scene mismatch penalty
P_SCENE_MISMATCH = -22          # Per scene mismatch hit
P_SCENE_MISMATCH_STRONG = -30   # Strong mismatch (completely wrong context)


# ---------------------------------------------------------------------------
# Subject-specific accept/reject rules (scene-level precision)
# ---------------------------------------------------------------------------
_SUBJECT_SCENE_RULES: dict[str, dict[str, list[str]]] = {
    "scooter": {
        "accept": ["scooter", "kick scooter", "electric scooter", "wheel", "tire",
                    "brake", "repair", "urban", "riding", "rider", "sidewalk",
                    "handlebar", "deck", "folding"],
        "reject": ["hiking", "backpack", "forest", "mountain trail", "camping",
                    "trekking", "wilderness", "hiker", "nature walk"],
    },
    "car": {
        "accept": ["car", "engine", "fuel", "gasoline", "petrol", "diesel",
                    "dashboard", "mechanic", "garage", "traffic", "highway",
                    "motor", "automotive", "vehicle", "sedan", "tire", "brake",
                    "transmission", "exhaust", "oil change", "repair shop"],
        "reject": ["beach postcard", "seaside retro", "vintage postcard",
                    "tropical car", "sunset car scenic",
                    "soup", "vegetables", "salad", "food plate", "cooking pot",
                    "kitchen interior", "recipe", "baking"],
        "deprioritize": ["retro car", "vintage car", "classic car", "antique car",
                         "old car", "postcard car", "retro automobile"],
    },
    "investment": {
        "accept": ["investment", "investor", "stock", "portfolio", "finance",
                    "chart", "graph", "trading", "money", "capital", "fund",
                    "dividend", "market", "bull", "bear", "financial",
                    "business meeting", "office", "conference"],
        "reject": ["cosmetics", "beauty", "makeup", "skincare", "cream",
                    "serum", "lipstick", "mascara", "nail polish", "beauty salon",
                    "spa treatment", "facial", "product flatlay", "flatlay",
                    "beauty product", "hair care", "perfume"],
    },
    "business": {
        "accept": ["business", "entrepreneur", "startup", "office", "meeting",
                    "strategy", "marketing", "deal", "handshake", "presentation",
                    "conference", "workplace", "laptop work", "business plan"],
        "reject": ["clinic", "treatment room", "empty clinic", "medical room",
                    "examination room", "hospital", "dental", "cosmetics",
                    "beauty", "makeup", "skincare", "spa", "product flatlay",
                    "flatlay", "beauty product"],
    },
    "entrepreneur": {
        "accept": ["entrepreneur", "business owner", "startup", "office",
                    "meeting", "strategy", "innovation", "pitch", "investor",
                    "coworking", "workspace", "laptop work", "brainstorm"],
        "reject": ["clinic", "treatment room", "empty clinic", "medical room",
                    "examination room", "hospital", "dental", "cosmetics",
                    "beauty", "makeup", "skincare", "spa", "product flatlay"],
    },
    "crocodile": {
        "accept": ["crocodile", "alligator", "reptile", "swamp", "river",
                    "wildlife", "danger", "teeth", "jaw", "predator",
                    "road danger", "animal crossing", "warning sign"],
        "reject": ["soup", "vegetables", "salad", "cooking", "kitchen",
                    "recipe", "food plate", "baking", "grocery", "market",
                    "fruit", "dinner", "breakfast", "meal prep"],
    },
    "carbonara": {
        "accept": ["carbonara", "spaghetti carbonara", "pasta carbonara",
                    "plated pasta", "creamy pasta", "guanciale", "pecorino",
                    "italian pasta dish"],
        "weak_accept": ["eggs", "flour", "ingredient", "raw pasta"],
        "reject": ["bakery", "deli", "food market", "grocery store",
                    "food shop", "supermarket", "food stall", "market stand"],
    },
    "pasta": {
        "accept": ["pasta", "spaghetti", "penne", "fettuccine", "plated dish",
                    "italian food", "sauce", "bolognese", "alfredo"],
        "weak_accept": ["flour", "dough", "ingredient"],
        "reject": ["bakery shop", "grocery", "food market", "supermarket"],
    },
    "fuel": {
        "accept": ["fuel", "gasoline", "petrol", "gas station", "pump",
                    "diesel", "engine", "car", "vehicle", "tank",
                    "refueling", "nozzle", "gas price"],
        "reject": ["beach", "seaside", "ocean", "scenic road trip",
                    "vintage car postcard", "soup", "vegetables",
                    "cooking", "kitchen", "recipe"],
        "deprioritize": ["retro car", "vintage car", "classic car", "antique car",
                         "old car", "postcard car", "retro automobile"],
    },
    "engine": {
        "accept": ["engine", "motor", "cylinder", "piston", "mechanical",
                    "car engine", "automotive", "repair", "mechanic", "garage",
                    "under hood", "oil", "spark plug"],
        "reject": ["beach", "scenic", "landscape", "fashion", "portrait",
                    "lifestyle outdoor", "hiking"],
    },
    "brake": {
        "accept": ["brake", "braking", "brake pad", "disc brake", "caliper",
                    "brake fluid", "repair", "mechanic", "wheel", "rotor"],
        "reject": ["hiking", "forest", "nature", "backpack", "lifestyle",
                    "portrait", "fashion"],
    },
    "massage": {
        "accept": ["massage", "spa", "therapy", "hands", "bodywork",
                    "relaxation", "wellness", "treatment table", "oil massage"],
        "reject": ["tech", "code", "server", "gaming", "car", "food"],
    },
    "repair": {
        "accept": ["repair", "tool", "wrench", "screwdriver", "workshop",
                    "mechanic", "fixing", "maintenance", "broken", "service"],
        "reject": ["hiking", "nature", "lifestyle", "portrait", "fashion",
                    "beach", "scenic"],
    },
    "kitchen": {
        "accept": ["kitchen", "cabinet", "countertop", "facade", "kitchen set",
                    "kitchen interior", "installed kitchen", "measuring", "install",
                    "kitchen design", "modern kitchen", "kitchen furniture",
                    "kitchen renovation", "cupboard", "sink", "faucet",
                    "kitchen appliance", "kitchen island"],
        "reject": ["children eating", "family breakfast", "family dinner",
                    "kids at table", "children at table", "family meal",
                    "birthday party", "baby eating", "toddler", "school lunch",
                    "playground", "daycare"],
    },
    "furniture": {
        "accept": ["furniture", "cabinet", "wardrobe", "shelf", "desk",
                    "interior", "room design", "wood", "crafted", "assembled",
                    "showroom", "workshop", "upholstery"],
        "reject": ["children eating", "family breakfast", "kids playing",
                    "playground", "birthday", "pet", "animal"],
    },
    "chinese car": {
        "accept": ["modern car", "new car", "sedan", "suv", "crossover",
                    "dealership", "showroom", "dashboard", "car interior",
                    "test drive", "chinese automobile", "chinese brand",
                    "geely", "chery", "haval", "byd", "changan", "great wall"],
        "deprioritize": ["retro car", "vintage car", "classic car", "antique car",
                         "old car", "postcard car", "beach car", "decorative car",
                         "car photography art", "rustic car"],
        "reject": ["bicycle", "motorcycle", "horse cart", "train", "airplane"],
    },
}

# Scene mismatch rules: if post-family + subject indicate one context,
# penalise images from a completely different scene
_SCENE_MISMATCH_RULES: list[tuple[list[str], list[str], int]] = [
    # (post_signals, image_reject_signals, penalty)
    # Person/lifestyle when post is about mechanics/object/repair
    (
        ["repair", "engine", "brake", "mechanic", "tool", "fix", "maintenance"],
        ["person outdoor", "lifestyle", "girl", "boy", "woman walking",
         "man walking", "portrait outdoor", "casual person", "people park"],
        P_SCENE_MISMATCH,
    ),
    # Nature/hiking when topic is urban transport
    (
        ["scooter", "car", "vehicle", "transport", "traffic", "urban",
         "driving", "fuel", "gasoline", "highway"],
        ["hiking", "forest trail", "mountain", "camping", "backpack",
         "trekking", "wilderness", "nature walk", "countryside"],
        P_SCENE_MISMATCH_STRONG,
    ),
    # Shop/market when topic is specific dish
    (
        ["carbonara", "pasta dish", "spaghetti", "plated", "bolognese",
         "risotto", "lasagna", "ramen", "pho", "sushi plate"],
        ["food market", "grocery", "supermarket", "food shop", "deli counter",
         "market stall", "bakery shop", "food stand"],
        P_SCENE_MISMATCH,
    ),
    # Concept/abstract when post is factual/news
    (
        ["news", "report", "fact", "statistics", "data", "study", "research"],
        ["abstract concept", "idea concept", "motivation concept",
         "creativity concept", "innovation concept"],
        P_SCENE_MISMATCH,
    ),
    # Children/family when topic is furniture/kitchen/interior
    (
        ["kitchen", "furniture", "cabinet", "countertop", "interior",
         "facade", "wardrobe", "shelf", "renovation"],
        ["children eating", "family breakfast", "family dinner",
         "kids at table", "children at table", "family meal",
         "birthday party", "baby eating", "toddler", "school lunch",
         "playground", "daycare", "kids playing"],
        P_SCENE_MISMATCH,
    ),
    # Retro/vintage when topic is modern/chinese cars
    (
        ["chinese car", "new car", "modern car", "crossover", "suv",
         "electric car", "chinese brand", "chinese automobile"],
        ["retro car", "vintage car", "classic car", "antique car",
         "old car", "postcard car", "rustic car", "decorative car",
         "car photography art"],
        P_SCENE_MISMATCH,
    ),
    # Beauty/cosmetics when topic is investment/business/finance
    (
        ["investment", "investor", "finance", "business", "entrepreneur",
         "startup", "capital", "portfolio", "stock", "trading", "fund"],
        ["cosmetics", "beauty product", "makeup", "skincare", "cream",
         "serum", "lipstick", "nail polish", "beauty salon", "product flatlay",
         "flatlay", "beauty flatlay", "spa treatment", "facial mask"],
        P_SCENE_MISMATCH_STRONG,
    ),
    # Clinic/medical when topic is business/entrepreneur (not medical)
    (
        ["entrepreneur", "business owner", "startup", "marketing",
         "strategy", "brand", "pitch", "deal"],
        ["clinic", "treatment room", "empty clinic", "medical room",
         "examination room", "hospital ward", "dental chair",
         "medical equipment", "operating room"],
        P_SCENE_MISMATCH,
    ),
    # Kitchen/food/soup when topic is crocodile/car/automotive danger
    (
        ["crocodile", "alligator", "reptile", "predator",
         "road danger", "animal crossing"],
        ["soup", "vegetables", "salad", "cooking pot", "kitchen",
         "recipe", "food plate", "baking", "grocery", "market",
         "fruit basket", "dinner table", "meal prep"],
        P_SCENE_MISMATCH_STRONG,
    ),
    # Retro car when topic is fuel/gasoline/modern automotive
    (
        ["fuel", "gasoline", "petrol", "gas station", "diesel",
         "refueling", "gas price", "oil price"],
        ["retro car", "vintage car", "classic car", "antique car",
         "old car", "postcard car"],
        P_SCENE_MISMATCH,
    ),
]


# ---------------------------------------------------------------------------
# Generic filler patterns
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


# ---------------------------------------------------------------------------
# Outcome constants — explicit, semantic decision labels
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
OUTCOME_NO_IMAGE_LOW_IMAGEABILITY = "NO_IMAGE_LOW_IMAGEABILITY"
OUTCOME_NO_IMAGE_NO_CANDIDATES = "NO_IMAGE_NO_CANDIDATES"


# ---------------------------------------------------------------------------
# Meta-family detection (lightweight, no image_search dependency)
# ---------------------------------------------------------------------------
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

    Requires ≥2 keyword hits for confident match.
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
# CandidateScore — per-candidate trace/score breakdown
# ---------------------------------------------------------------------------
@dataclass
class CandidateScore:
    """Per-candidate scoring breakdown with full explainability."""
    url: str = ""
    provider: str = ""
    query: str = ""
    meta_snippet: str = ""

    # Score components
    subject_match: int = 0
    sense_match: int = 0
    scene_match: int = 0
    query_token_hits: int = 0
    family_term_hits: int = 0
    allowed_visual_hits: int = 0
    generic_stock_hits: int = 0
    generic_filler_hits: int = 0
    scene_mismatch_hits: int = 0
    subject_scene_reject_hits: int = 0

    # Aggregate scores
    post_centric_score: int = 0
    exact_subject_score: int = 0
    scene_match_score: int = 0
    provider_bonus: int = 0
    repeat_penalty: int = 0
    final_score: int = 0

    # Decision
    hard_reject: str = ""
    reject_reason: str = ""
    outcome: str = ""
    fallback_level: str = ""     # "exact", "near", "family", "weak"
    final_accept_reason: str = ""

    def as_log_dict(self) -> dict:
        """Compact dict for structured logging."""
        return {
            "url": self.url[:80],
            "prov": self.provider,
            "q": self.query[:40],
            "subj": self.subject_match,
            "sense": self.sense_match,
            "scene": self.scene_match,
            "qtok": self.query_token_hits,
            "stock": self.generic_stock_hits,
            "filler": self.generic_filler_hits,
            "scene_mis": self.scene_mismatch_hits,
            "subj_rej": self.subject_scene_reject_hits,
            "exact_subj": self.exact_subject_score,
            "scene_sc": self.scene_match_score,
            "pc": self.post_centric_score,
            "bonus": self.provider_bonus,
            "repeat": self.repeat_penalty,
            "final": self.final_score,
            "hard": self.hard_reject[:40] if self.hard_reject else "",
            "rej": self.reject_reason[:40] if self.reject_reason else "",
            "out": self.outcome,
            "fb_level": self.fallback_level,
            "accept_reason": self.final_accept_reason[:40] if self.final_accept_reason else "",
        }


# ---------------------------------------------------------------------------
# Wrong-sense hard reject
# ---------------------------------------------------------------------------
def check_wrong_sense(meta_text: str, intent: VisualIntentV2) -> str | None:
    """Check for forbidden meaning in metadata. Returns reject reason or None."""
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
# Generic stock penalty
# ---------------------------------------------------------------------------
def compute_generic_stock_penalty(
    meta_text: str,
    intent: VisualIntentV2,
) -> tuple[int, int]:
    """Returns (penalty, hit_count). Penalty is 0 or negative."""
    if not meta_text:
        return 0, 0
    meta_lower = meta_text.lower()
    stock_hits = sum(1 for gs in _GENERIC_STOCK_SIGNALS if gs in meta_lower)
    if stock_hits == 0:
        return 0, 0

    # Strong subject alignment softens the penalty
    if intent.subject:
        subject_words = intent.subject.lower().split()
        subject_meta_hits = sum(1 for w in subject_words if w in meta_lower)
        if subject_meta_hits >= 2:
            return P_GENERIC_STOCK_MILD * stock_hits, stock_hits

    return P_GENERIC_STOCK * min(stock_hits, 3), stock_hits


# ---------------------------------------------------------------------------
# Subject-specific hard filter check
# ---------------------------------------------------------------------------
def _check_subject_scene_rules(
    meta_text: str,
    intent: VisualIntentV2,
) -> tuple[int, int, int]:
    """Check subject-specific accept/reject rules.

    Returns (score_adjustment, reject_hits, weak_accept_hits).
    """
    if not intent.subject or not meta_text:
        return 0, 0, 0

    subject_lower = intent.subject.lower()
    meta_lower = meta_text.lower()
    total_adjustment = 0
    reject_hits = 0
    weak_accept_hits = 0

    for subj_key, rules in _SUBJECT_SCENE_RULES.items():
        if subj_key not in subject_lower:
            continue
        # Check reject terms
        for term in rules.get("reject", ()):
            if term in meta_lower:
                total_adjustment += P_SCENE_MISMATCH
                reject_hits += 1
        # Check deprioritize terms (softer than reject, but still penalised)
        for term in rules.get("deprioritize", ()):
            if term in meta_lower:
                total_adjustment += P_SCENE_MISMATCH // 2  # Half penalty
                weak_accept_hits += 1
        # Check weak_accept (reduce score for ingredients-only matches)
        for term in rules.get("weak_accept", ()):
            if term in meta_lower:
                weak_accept_hits += 1
        # Check accept terms (bonus for exact subject match)
        accept_hits = sum(1 for t in rules.get("accept", ()) if t in meta_lower)
        if accept_hits > 0:
            total_adjustment += accept_hits * 3  # Small bonus for subject-rule match

    return total_adjustment, reject_hits, weak_accept_hits


# ---------------------------------------------------------------------------
# Scene mismatch penalty check
# ---------------------------------------------------------------------------
def _check_scene_mismatch(
    meta_text: str,
    intent: VisualIntentV2,
) -> tuple[int, int]:
    """Check for scene mismatch using global rules.

    Returns (penalty, hit_count).
    """
    if not meta_text:
        return 0, 0
    meta_lower = meta_text.lower()
    subject_lower = (intent.subject or "").lower()
    scene_lower = (intent.scene or "").lower()
    combined = f"{subject_lower} {scene_lower}"

    total_penalty = 0
    total_hits = 0
    for post_signals, image_reject_signals, penalty in _SCENE_MISMATCH_RULES:
        # Check if any post signal matches the intent
        post_match = any(sig in combined for sig in post_signals)
        if not post_match:
            continue
        # Check if any reject signal matches the image metadata
        for rej_sig in image_reject_signals:
            if rej_sig in meta_lower:
                total_penalty += penalty
                total_hits += 1
    return total_penalty, total_hits


# ---------------------------------------------------------------------------
# Determine fallback level for scoring hierarchy
# ---------------------------------------------------------------------------
def _determine_fallback_level(
    subject_hits: int,
    scene_hits: int,
    family_term_hits: int,
    allowed_visual_hits: int,
) -> str:
    """Classify match quality into hierarchical level.

    Returns one of: "exact", "near", "family", "weak".
    """
    if subject_hits >= 2 and scene_hits >= 1:
        return "exact"
    if subject_hits >= 1:
        return "near"
    if family_term_hits >= 2 or allowed_visual_hits >= 1:
        return "family"
    return "weak"


# ---------------------------------------------------------------------------
# Score a single candidate (post-centric, no provider influence)
# ---------------------------------------------------------------------------
def score_candidate(
    meta_text: str,
    intent: VisualIntentV2,
    query: str = "",
) -> tuple[int, str, CandidateScore]:
    """Score candidate purely on post-centric intent.

    Returns (score, reject_reason, candidate_score).
    Provider bonus is handled separately by the pipeline.
    """
    cs = CandidateScore()

    if not meta_text:
        cs.reject_reason = "empty_meta"
        return 0, "empty_meta", cs

    text = meta_text.strip().lower().replace("ё", "е")
    text = re.sub(r"\s+", " ", text)
    cs.meta_snippet = text[:120]

    score = 0
    reject_reason = ""

    # --- Hard reject: wrong sense ---
    wrong_sense = check_wrong_sense(text, intent)
    if wrong_sense:
        cs.hard_reject = wrong_sense
        cs.post_centric_score = -100
        return -100, wrong_sense, cs

    # --- 1. Subject match (highest weight) ---
    subject_hits = 0
    if intent.subject:
        subject_words = [w for w in intent.subject.lower().split() if len(w) >= 3]
        subject_hits = sum(1 for w in subject_words if w in text)
        score += subject_hits * W_SUBJECT
        if subject_hits >= 3:
            score += W_SUBJECT_STRONG
    cs.subject_match = subject_hits
    cs.exact_subject_score = subject_hits * W_SUBJECT

    # --- 2. Sense match ---
    sense_hits = 0
    if intent.sense:
        sense_words = [w for w in intent.sense.lower().replace("_", " ").split() if len(w) >= 3]
        sense_hits = sum(1 for w in sense_words if w in text)
        score += sense_hits * W_SENSE
    cs.sense_match = sense_hits

    # --- 3. Scene match ---
    scene_hits = 0
    if intent.scene:
        scene_words = [w for w in intent.scene.lower().split() if len(w) >= 3]
        scene_hits = sum(1 for w in scene_words if w in text)
        score += scene_hits * W_SCENE
    cs.scene_match = scene_hits
    cs.scene_match_score = scene_hits * W_SCENE

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
    cs.query_token_hits = query_token_hits

    # --- 5. Family term match (weak) ---
    family_term_hits = 0
    if intent.post_family in TOPIC_FAMILY_TERMS:
        terms = TOPIC_FAMILY_TERMS[intent.post_family].get("en", [])[:10]
        family_term_hits = sum(1 for t in terms if t in text)
        score += family_term_hits * W_FAMILY_TERM
    cs.family_term_hits = family_term_hits

    # --- 6. Allowed visual class match ---
    allowed = get_family_allowed_visuals(intent.post_family)
    allowed_hits = 0
    if allowed:
        allowed_hits = sum(1 for cls in allowed if cls in text)
        score += allowed_hits * W_ALLOWED_VISUAL
    cs.allowed_visual_hits = allowed_hits

    # --- 7. Blocked visual class penalty ---
    blocked = get_family_blocked_visuals(intent.post_family)
    if blocked:
        for cls in blocked:
            if re.search(rf"\b{re.escape(cls)}\b", text, re.I):
                score += P_BLOCKED_VISUAL
                if not reject_reason:
                    reject_reason = f"blocked_visual:{cls}"

    # --- 8. Cross-family detection ---
    meta_family = detect_meta_family(text)
    if meta_family != "generic" and meta_family != intent.post_family:
        score += P_CROSS_FAMILY
        if not reject_reason:
            reject_reason = f"cross_family:{meta_family}"

    # --- 9. Generic stock penalty ---
    stock_penalty, stock_hits = compute_generic_stock_penalty(text, intent)
    score += stock_penalty
    cs.generic_stock_hits = stock_hits
    if stock_penalty <= -40 and not reject_reason:
        reject_reason = "generic_stock"

    # --- 10. Generic filler detection ---
    filler_hits = sum(1 for f in _AUTOPOST_GENERIC_FILLER if f in text)
    cs.generic_filler_hits = filler_hits
    if filler_hits > 0 and subject_hits < 2:
        score += filler_hits * P_GENERIC_FILLER_PER
        if not reject_reason:
            reject_reason = "generic_filler"

    # --- 11. Subject-specific scene rules (B: hard filters) ---
    subj_adj, subj_reject_hits, weak_accept_hits = _check_subject_scene_rules(text, intent)
    score += subj_adj
    cs.subject_scene_reject_hits = subj_reject_hits
    if subj_reject_hits > 0 and not reject_reason:
        reject_reason = "subject_scene_reject"

    # --- 12. Scene mismatch penalties (D) ---
    scene_penalty, scene_mis_hits = _check_scene_mismatch(text, intent)
    score += scene_penalty
    cs.scene_mismatch_hits = scene_mis_hits
    if scene_mis_hits > 0 and not reject_reason:
        reject_reason = "scene_mismatch"

    # --- 13. Determine fallback level (C: scoring hierarchy) ---
    fb_level = _determine_fallback_level(
        subject_hits, scene_hits, family_term_hits, allowed_hits,
    )
    cs.fallback_level = fb_level

    # Apply hierarchy penalty: generic family match gets reduced score
    # for dish/object-specific posts
    if fb_level == "family" and subject_hits == 0:
        # Family-only match: reduce score significantly for specific subjects
        if intent.subject:
            score = min(score, MAX_SCORE_WITHOUT_AFFIRMATION + 5)
            # Stronger penalty: if exact candidates exist for this subject,
            # broad family should not win
            if subj_reject_hits > 0 or scene_mis_hits > 0:
                score = min(score, MAX_SCORE_WITHOUT_AFFIRMATION)
    if fb_level == "weak":
        score = min(score, MAX_SCORE_WITHOUT_AFFIRMATION)

    # Weak-accept penalty: ingredients-only for dish posts
    if weak_accept_hits > 0 and subject_hits == 0:
        score = min(score, MAX_SCORE_WITHOUT_AFFIRMATION + 3)
        if not reject_reason:
            reject_reason = "weak_accept_only"

    # --- 14. Positive affirmation requirement ---
    has_affirmation = (
        (subject_hits >= AFFIRMATION_MIN_SUBJECT_HITS)
        or (allowed_hits >= AFFIRMATION_MIN_ALLOWED_HITS)
        or (scene_hits >= AFFIRMATION_MIN_SCENE_HITS)
    )
    if not has_affirmation and score > 0:
        score = min(score, MAX_SCORE_WITHOUT_AFFIRMATION)
        if not reject_reason:
            reject_reason = "no_positive_affirmation"

    cs.post_centric_score = score
    cs.reject_reason = reject_reason
    return score, reject_reason, cs


# ---------------------------------------------------------------------------
# Provider bonus (CAPPED secondary signal)
# ---------------------------------------------------------------------------
def compute_provider_bonus(provider_score: int) -> int:
    """Compute capped provider bonus. Cannot rescue a weak candidate."""
    return min(
        int(max(provider_score, 0) * PROVIDER_BONUS_WEIGHT),
        PROVIDER_BONUS_CAP,
    )


# ---------------------------------------------------------------------------
# Determine candidate outcome (mode-specific)
# ---------------------------------------------------------------------------
def determine_outcome(
    cs: CandidateScore,
    mode: str = "autopost",
) -> str:
    """Determine outcome for a scored candidate."""
    if cs.hard_reject:
        if "wrong_sense" in cs.hard_reject:
            return OUTCOME_REJECT_WRONG_SENSE
        return OUTCOME_REJECT_CROSS_FAMILY

    # Hard reject repeated images in autopost
    if mode == "autopost" and cs.repeat_penalty <= -200:
        return OUTCOME_REJECT_REPEAT

    min_score = AUTOPOST_MIN_SCORE if mode == "autopost" else EDITOR_MIN_SCORE
    score = cs.final_score

    if score < min_score:
        if cs.generic_stock_hits >= 2:
            return OUTCOME_REJECT_GENERIC_STOCK
        if cs.reject_reason and "cross_family" in cs.reject_reason:
            return OUTCOME_REJECT_CROSS_FAMILY
        if cs.reject_reason and "wrong_sense" in cs.reject_reason:
            return OUTCOME_REJECT_WRONG_SENSE
        if cs.reject_reason and "generic_filler" in cs.reject_reason:
            return OUTCOME_REJECT_GENERIC_FILLER
        if score < EDITOR_MIN_SCORE:
            return OUTCOME_REJECT_NO_MATCH
        if mode == "autopost":
            return OUTCOME_REJECT_LOW_CONFIDENCE
        return OUTCOME_ACCEPT_FOR_EDITOR

    if mode == "autopost":
        return OUTCOME_ACCEPT_BEST
    if score >= AUTOPOST_MIN_SCORE:
        return OUTCOME_ACCEPT_BEST
    return OUTCOME_ACCEPT_FOR_EDITOR


# ---------------------------------------------------------------------------
# Rank candidates
# ---------------------------------------------------------------------------
def rank_candidates(
    candidates: list[CandidateScore],
    *,
    intent: VisualIntentV2,
    history: ImageHistory,
    mode: str = "autopost",
) -> list[CandidateScore]:
    """Score, apply diversity penalties, and rank all candidates.

    Modifies candidates in-place and returns them sorted by final_score desc.
    """
    for cs in candidates:
        if cs.hard_reject:
            continue

        # Apply provider bonus (capped)
        cs.provider_bonus = compute_provider_bonus(cs.post_centric_score)
        cs.final_score = cs.post_centric_score + cs.provider_bonus

        # Apply anti-repeat penalty from history (E: strengthened)
        domain = extract_domain(cs.url)
        visual_class = detect_meta_family(cs.meta_snippet)
        # Compute scene_class from metadata for dedup
        scene_class = visual_class
        # Coarse pattern for visual pattern-level dedup
        coarse_pattern = f"{visual_class}_{intent.subject or 'generic'}".lower().replace(" ", "_")
        cs.repeat_penalty = history.compute_penalty(
            url=cs.url,
            content_hash=url_content_hash(cs.url),
            visual_class=visual_class,
            subject_bucket=intent.subject or "",
            domain=domain,
            scene_class=scene_class,
            coarse_pattern=coarse_pattern,
        )
        cs.final_score += cs.repeat_penalty

        # Determine outcome
        cs.outcome = determine_outcome(cs, mode)

        # Set final_accept_reason for traceable logs (F)
        if not cs.outcome.startswith("REJECT"):
            reasons = []
            if cs.subject_match > 0:
                reasons.append(f"subject_hit={cs.subject_match}")
            if cs.scene_match > 0:
                reasons.append(f"scene_hit={cs.scene_match}")
            if cs.allowed_visual_hits > 0:
                reasons.append(f"allowed_visual={cs.allowed_visual_hits}")
            if cs.family_term_hits > 0:
                reasons.append(f"family_term={cs.family_term_hits}")
            reasons.append(f"level={cs.fallback_level}")
            cs.final_accept_reason = "; ".join(reasons)

            # Production accept log
            logger.info(
                "IMAGE_ACCEPT_REASON url=%s score=%d reason=%s",
                cs.url[:80], cs.final_score, cs.final_accept_reason,
            )
            logger.info(
                "IMAGE_SUBJECT=%s IMAGE_SCENE=%s IMAGE_FALLBACK_LEVEL=%s",
                (intent.subject or "")[:40], (intent.scene or "")[:40], cs.fallback_level,
            )
        else:
            # Production reject log
            logger.info(
                "IMAGE_REJECT_REASON url=%s score=%d reason=%s outcome=%s",
                cs.url[:80], cs.final_score, cs.reject_reason or cs.hard_reject, cs.outcome,
            )
            if cs.scene_mismatch_hits > 0:
                logger.info(
                    "IMAGE_SCENE_MISMATCH url=%s hits=%d",
                    cs.url[:80], cs.scene_mismatch_hits,
                )
            if cs.repeat_penalty < 0:
                logger.info(
                    "IMAGE_REPEAT_PENALTY url=%s penalty=%d",
                    cs.url[:80], cs.repeat_penalty,
                )

    # Sort by final score descending
    candidates.sort(key=lambda c: c.final_score, reverse=True)
    return candidates
