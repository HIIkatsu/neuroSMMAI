from __future__ import annotations

import logging
import os
import re
import time
from urllib.parse import urlparse

import httpx

from topic_utils import (
    TOPIC_FAMILY_TERMS,
    detect_topic_family,
    detect_subfamily,
    classify_visual_type,
    STRICT_IMAGE_FAMILIES,
    IRRELEVANT_IMAGE_CLASSES,
    VISUAL_TYPE_TEXT_ONLY,
    get_family_image_queries,
    get_subfamily_image_queries,
    get_family_irrelevant_classes,
    get_family_allowed_visuals,
    get_family_blocked_visuals,
    get_family_bad_url_signals,
)

logger = logging.getLogger(__name__)

PEXELS_API_KEY = (os.getenv("PEXELS_API_KEY") or "").strip()
UNSPLASH_ACCESS_KEY = (os.getenv("UNSPLASH_ACCESS_KEY") or "").strip()
PIXABAY_API_KEY = (os.getenv("PIXABAY_API_KEY") or "").strip()

IMAGE_HTTP_PROXY_URL = (
    os.getenv("IMAGE_HTTP_PROXY_URL")
    or os.getenv("OPENROUTER_PROXY_URL")
    or os.getenv("HTTPS_PROXY")
    or os.getenv("HTTP_PROXY")
    or os.getenv("ALL_PROXY")
    or ""
).strip() or None

CONNECT_TIMEOUT = float(os.getenv("IMAGE_CONNECT_TIMEOUT_SEC", "4.0"))
READ_TIMEOUT = float(os.getenv("IMAGE_READ_TIMEOUT_SEC", "7.0"))
MAX_QUERY_LEN = int(os.getenv("IMAGE_SEARCH_QUERY_MAX_LEN", "140"))
BAD_URL_PARTS = ("avatar", "icon", "logo", "sprite", "thumb", "placeholder")

# Minimum semantic relevance score to accept an image.
# If no candidate meets this threshold, the system prefers no image over a bad one.
# Raised from 12 to 15 to further reduce off-topic/generic image acceptance.
MIN_RELEVANCE_SCORE = int(os.getenv("IMAGE_MIN_RELEVANCE_SCORE", "15"))


# ---------------------------------------------------------------------------
# Provider circuit breaker: skip providers that are consistently failing
# ---------------------------------------------------------------------------
_PROVIDER_FAILURE_THRESHOLD = 3  # consecutive failures to trip
_PROVIDER_COOLDOWN_SECONDS = 120  # seconds to skip after tripped
_provider_failures: dict[str, int] = {}  # provider_name -> consecutive failure count
_provider_tripped_at: dict[str, float] = {}  # provider_name -> monotonic timestamp


def _provider_available(name: str) -> bool:
    """Check if a provider is available (not in cooldown from repeated failures)."""
    tripped = _provider_tripped_at.get(name)
    if tripped is None:
        return True
    if time.monotonic() - tripped > _PROVIDER_COOLDOWN_SECONDS:
        # Cooldown expired, reset and allow
        _provider_failures.pop(name, None)
        _provider_tripped_at.pop(name, None)
        return True
    return False


def _provider_success(name: str) -> None:
    """Record a successful call, resetting failure count."""
    _provider_failures.pop(name, None)
    _provider_tripped_at.pop(name, None)


def _provider_failure(name: str) -> None:
    """Record a failed call; trip the breaker if threshold is exceeded."""
    count = _provider_failures.get(name, 0) + 1
    _provider_failures[name] = count
    if count >= _PROVIDER_FAILURE_THRESHOLD:
        _provider_tripped_at[name] = time.monotonic()
        logger.warning("Image provider %s circuit-breaker tripped after %d failures (cooldown %ds)",
                        name, count, _PROVIDER_COOLDOWN_SECONDS)

_STOPWORDS = {
    "и", "или", "для", "как", "что", "это", "без", "под", "над", "при", "про", "если", "чтобы", "после", "перед", "когда",
    "the", "and", "for", "with", "from", "that", "this", "into", "your", "about", "after", "before",
    "post", "telegram", "editorial", "photo", "image", "visual", "brief", "post", "news", "trend", "latest",
    "новость", "новости", "свежий", "свежее", "канал", "канала", "пост", "разбор", "обзор", "важно", "сейчас",
}

_NEGATIVE_HINTS: dict[str, list[str]] = {
    "food": ["tech device", "circuit board", "abstract neon", "corporate meeting", "finance chart"],
    "health": ["fashion model", "luxury product", "gaming setup", "circuit board", "stock chart"],
    "beauty": ["circuit board", "server rack", "gaming setup", "generic office", "finance chart"],
    "local_business": ["abstract art", "neon wallpaper", "cartoon", "fashion model", "gaming"],
    "education": ["fashion model", "gaming poster", "abstract neon", "luxury product"],
    "finance": ["fashion model", "food dish", "beauty product", "gaming poster", "cartoon"],
    "marketing": ["cartoon", "gaming poster", "abstract neon wallpaper", "food dish"],
    "lifestyle": ["server rack", "circuit board", "abstract tech", "gaming setup"],
    "expert_blog": ["cartoon", "abstract neon", "gaming poster", "food flat lay", "beauty product"],
    "massage": ["illustration", "cartoon", "promo flyer", "spa candles", "cream jar", "beauty salon ad"],
    "hardware": ["3d render", "mockup", "cartoon", "rgb wallpaper", "logo"],
    "gaming": ["poster", "cover art", "fan art", "mascot"],
    "cars": ["logo", "dealership logo", "render", "toy car"],
    "tech": ["fashion model", "food dish", "beauty product", "cartoon"],
    "business": ["cartoon", "food flat lay", "beauty product", "gaming poster"],
    "generic": ["logo", "icon", "infographic"],
}

# Broad irrelevant-image class penalties — applies to ALL families via channel-aware policy
_IRRELEVANT_PENALTIES: list[str] = IRRELEVANT_IMAGE_CLASSES

# Anti-repeat motifs: penalize overly generic / repetitive imagery
_ANTI_REPEAT_MOTIFS: list[str] = [
    "neon abstract", "neon light", "abstract blue",
    "generic desk laptop", "laptop coffee",
    "person typing laptop", "woman laptop",
    "man laptop", "girl laptop", "smiling woman",
    "business man suit", "business woman suit",
    "handshake", "thumbs up",
]

# Compact Russian→English translation table for literal-topic-first image queries.
# NOT a full taxonomy — only high-frequency topic words that lack Latin equivalents
# in queries and where subfamily detection does not already produce an English term.
_LITERAL_TRANSLATIONS: list[tuple[str, str]] = [
    # Food items not covered by food subfamily detection
    ("гриб", "mushrooms"),
    ("ягод", "berries"),
    ("мясо", "meat"),
    ("рыб", "fish seafood"),
    ("овощ", "vegetables"),
    ("фрукт", "fruits"),
    ("молок", "dairy milk"),
    ("мёд", "honey"),
    ("мед", "honey"),
    ("орех", "nuts"),
    # Vehicles / automotive
    ("машин", "cars automobile"),
    ("автомобил", "automobile car"),
    ("авто", "car automotive"),
    ("грузовик", "truck cargo"),
    ("мотоцикл", "motorcycle"),
    # Professional / service roles
    ("бухгалтер", "accountant office"),
    ("юрист", "lawyer legal"),
    ("нотариус", "notary office"),
    ("стоматолог", "dentist dental"),
    ("психолог", "psychologist therapy"),
    ("репетитор", "tutor teaching"),
    ("архитектор", "architect design"),
    ("ветеринар", "veterinarian animal"),
    ("агроном", "agronomist farming"),
    ("логопед", "speech therapist"),
    # Tech / IT
    ("сервер", "server data center"),
    ("дата центр", "data center server room"),
    ("дата-центр", "data center server room"),
    ("программист", "programmer developer"),
    ("разработчик", "software developer"),
    # Business / operations
    ("склад", "warehouse storage"),
    ("логистик", "logistics supply chain"),
    ("транспорт", "transportation logistics"),
    ("доставк", "delivery courier"),
    ("импорт", "import export trade"),
    ("производств", "manufacturing production"),
    ("промышленност", "industry manufacturing"),
    # Finance topics not covered by finance subfamily
    ("страховани", "insurance"),
    ("ипотек", "mortgage real estate"),
    ("кредит", "credit loan finance"),
    # Other high-frequency topics
    ("туризм", "tourism travel"),
    ("недвижимост", "real estate property"),
    ("свадьб", "wedding ceremony"),
    ("детск", "children kids"),
    ("домашн", "home domestic"),
    ("сад", "garden gardening"),
    ("огород", "garden vegetable garden"),
    ("животн", "animals wildlife"),
    ("питомц", "pets animals"),
    ("кош", "cat feline"),
    ("соб", "dog canine"),
]

# ── Context-aware query expansion rules ──
# For complex multi-word Russian topics, map context modifiers → English expansion terms.
# These produce subject-focused and context-focused candidate queries when the raw topic
# contains multiple semantic dimensions (e.g. "производство китайских машин" has
# both a process dimension — manufacturing — and a subject dimension — Chinese cars).
_CONTEXT_MODIFIERS: list[tuple[str, str, str]] = [
    # (russian_keyword, subject_expansion, context_expansion)
    # Manufacturing / production contexts
    ("производств", "factory assembly line", "manufacturing industrial"),
    ("промышленност", "factory heavy equipment", "industrial plant"),
    ("завод", "factory production", "industrial facility"),
    ("фабрик", "factory workshop", "manufacturing process"),
    # Origin / geography modifiers
    ("китайск", "chinese", "china market"),
    ("немецк", "german", "germany engineering"),
    ("японск", "japanese", "japan technology"),
    ("корейск", "korean", "korea innovation"),
    ("американск", "american", "usa market"),
    ("европейск", "european", "europe market"),
    # Process / activity modifiers
    ("ремонт", "repair service", "workshop maintenance"),
    ("обслуживан", "maintenance service", "professional care"),
    ("тестирован", "testing quality", "quality control"),
    ("разработк", "development design", "engineering process"),
    ("продаж", "sales showroom", "commercial retail"),
    ("обучен", "training education", "learning course"),
    ("доставк", "delivery logistics", "courier shipping"),
    ("сравнен", "comparison review", "versus analysis"),
]


# Minimum word count in a query to trigger context-aware expansion.
# Queries with fewer words are simple enough that literal translation suffices.
_MIN_WORDS_FOR_EXPANSION = 3

# Maximum number of expansion parts (subject/context) to include per candidate query.
_MAX_EXPANSION_PARTS = 2


def _expand_query_candidates(query: str, family: str, literal_en: str) -> list[str]:
    """Build 2-3 expanded candidate queries for complex multi-word topics.

    Produces controlled query expansion by building:
    1. subject-focused query  — emphasizes the OBJECT of the topic
    2. context-focused query  — emphasizes the PROCESS or SETTING

    Example: "производство китайских машин"
      literal_en = "cars automobile" (from family)
      subject-focused = "chinese cars automobile factory"
      context-focused = "manufacturing industrial china market automotive"

    Returns empty list for simple queries that don't need expansion.
    """
    if not query or not literal_en:
        return []

    src = _clean_text(query)
    word_count = len(src.split())

    if word_count < _MIN_WORDS_FOR_EXPANSION:
        return []

    subject_parts: list[str] = []
    context_parts: list[str] = []

    for ru_kw, subj_expansion, ctx_expansion in _CONTEXT_MODIFIERS:
        if ru_kw in src:
            subject_parts.append(subj_expansion)
            context_parts.append(ctx_expansion)

    # No modifiers matched — nothing to expand
    if not subject_parts and not context_parts:
        return []

    candidates: list[str] = []

    # Subject-focused: literal topic + subject modifiers
    subj_query = f"{literal_en} {' '.join(subject_parts[:_MAX_EXPANSION_PARTS])} realistic photo"
    candidates.append(_strip_non_latin(subj_query)[:MAX_QUERY_LEN])

    # Context-focused: context modifiers + literal topic
    ctx_query = f"{' '.join(context_parts[:_MAX_EXPANSION_PARTS])} {literal_en} editorial photo"
    candidates.append(_strip_non_latin(ctx_query)[:MAX_QUERY_LEN])

    # Combined: if both subject and context are available, fuse top tokens
    if subject_parts and context_parts:
        combined = f"{literal_en} {subject_parts[0]} {context_parts[0]} professional photo"
        candidates.append(_strip_non_latin(combined)[:MAX_QUERY_LEN])

    return [c for c in candidates if c.strip()]


def _clean_text(text: str) -> str:
    text = (text or "").strip().lower().replace("ё", "е")
    text = re.sub(r"\s+", " ", text)
    return text


# Pattern matching only Latin/ASCII tokens (no Cyrillic) — used when building stock photo API queries
_LATIN_TOKEN_RE = re.compile(r"^[A-Za-z0-9][\w.+-]*$")


def _strip_non_latin(text: str) -> str:
    """Remove non-ASCII/non-Latin words from a query string so stock photo APIs receive only English tokens."""
    words = (text or "").split()
    latin_words = [w for w in words if _LATIN_TOKEN_RE.match(w)]
    return re.sub(r"\s+", " ", " ".join(latin_words)).strip()


def _query_tokens(text: str) -> list[str]:
    raw = _clean_text(text)
    words = re.findall(r"[a-zа-яё0-9][a-zа-яё0-9+-]{2,}", raw)
    out: list[str] = []
    seen: set[str] = set()
    for word in words:
        if len(word) < 4 or word in _STOPWORDS:
            continue
        if word in seen:
            continue
        seen.add(word)
        out.append(word)
    return out[:18]


def _meaningful_query_core(query: str, family: str) -> str:
    tokens = _query_tokens(query)
    # Only keep Latin/ASCII tokens so non-English words don't leak into stock API queries
    latin_tokens = [t for t in tokens if _LATIN_TOKEN_RE.match(t)]
    if family in TOPIC_FAMILY_TERMS:
        family_terms = TOPIC_FAMILY_TERMS[family]["en"][:2]
    else:
        family_terms = []
    out = []
    for item in latin_tokens[:5] + family_terms:
        if item not in out:
            out.append(item)
    return " ".join(out[:6]).strip()


def _specialized_queries(query: str, family: str) -> list[str]:
    tokens = _query_tokens(query)
    src = _clean_text(query)
    # Only use Latin tokens in query templates — Russian tokens break stock photo API search
    latin_tokens = [t for t in tokens if _LATIN_TOKEN_RE.match(t)]
    core = _strip_non_latin(" ".join(latin_tokens[:5]).strip())
    out: list[str] = []
    if family == "massage":
        area = "body"
        if any(x in src for x in ("шея", "neck", "плеч")):
            area = "neck shoulder"
        elif any(x in src for x in ("спина", "поясниц", "back", "lower back")):
            area = "back lower back"
        elif any(x in src for x in ("лицо", "face", "jaw")):
            area = "face jaw"
        out.extend([
            f"therapeutic massage therapist hands {area} realistic clinic photo {core}",
            f"recovery treatment physiotherapy {area} realistic photo {core}",
            f"wellness session therapist {area} studio realistic editorial photo",
        ])
    elif family == "food":
        if any(x in src for x in ("кофе", "coffee", "капучино", "latte")):
            out.extend([
                f"coffee cup cafe barista editorial photography {core}",
                f"coffee shop morning mood realistic photo {core}",
            ])
        elif any(x in src for x in ("рецепт", "recipe", "готовить", "cooking")):
            out.extend([
                f"kitchen cooking ingredients flat lay editorial photography {core}",
                f"homemade food preparation close-up natural light photo {core}",
            ])
        elif any(x in src for x in ("ресторан", "restaurant", "кафе", "cafe")):
            out.extend([
                f"restaurant dish gourmet plating professional editorial photo {core}",
                f"dining table food atmosphere restaurant realistic photo {core}",
            ])
        else:
            out.extend([
                f"beautifully plated food dish close-up editorial photography {core}",
                f"fresh ingredients natural light food editorial photo {core}",
                f"food flat lay overhead professional photo {core}",
            ])
    elif family == "health":
        if any(x in src for x in ("фитнес", "fitness", "тренировк", "workout", "спорт")):
            out.extend([
                f"fitness workout exercise healthy lifestyle realistic editorial photo {core}",
                f"sport training motivation realistic photo {core}",
            ])
        elif any(x in src for x in ("медитац", "meditation", "йог", "yoga", "mindfulness")):
            out.extend([
                f"meditation mindfulness peaceful nature editorial photo {core}",
                f"yoga practice calm atmosphere realistic photo {core}",
            ])
        elif any(x in src for x in ("питани", "nutrition", "диет", "diet")):
            out.extend([
                f"healthy food nutrition meal prep clean editorial photo {core}",
                f"nutritious meal fresh vegetables realistic photo {core}",
            ])
        else:
            out.extend([
                f"wellness healthy lifestyle natural light editorial photo {core}",
                f"health wellbeing calm realistic editorial photo {core}",
                f"healthy living active lifestyle realistic photo {core}",
            ])
    elif family == "beauty":
        if any(x in src for x in ("маникюр", "nail", "гель", "shellac")):
            out.extend([
                f"nail art manicure creative beauty editorial photo {core}",
                f"nail studio professional close-up realistic photo {core}",
            ])
        elif any(x in src for x in ("парикмахер", "стриж", "hair", "окраш")):
            out.extend([
                f"hair salon professional styling editorial photo {core}",
                f"hair beauty treatment close-up realistic photo {core}",
            ])
        elif any(x in src for x in ("косметолог", "clinic", "инъекц", "ботокс")):
            out.extend([
                f"cosmetology clinic professional treatment editorial photo {core}",
                f"beauty procedure professional realistic photo {core}",
            ])
        else:
            out.extend([
                f"beauty skincare product flat lay editorial photography {core}",
                f"cosmetics makeup artistic editorial close-up photo {core}",
                f"beauty routine skincare realistic editorial photo {core}",
            ])
    elif family == "local_business":
        if any(x in src for x in ("ремонт", "repair", "сервис", "service")):
            out.extend([
                f"repair workshop technician tools professional realistic photo {core}",
                f"local service professional craftsman at work editorial photo {core}",
            ])
        elif any(x in src for x in ("строительств", "renovation", "ремонт квартир")):
            out.extend([
                f"home renovation construction interior realistic editorial photo {core}",
                f"renovation worker professional tools realistic photo {core}",
            ])
        elif any(x in src for x in ("клининг", "cleaning", "уборк")):
            out.extend([
                f"professional cleaning service realistic editorial photo {core}",
                f"clean home cleaning professional realistic photo {core}",
            ])
        else:
            out.extend([
                f"local small business service professional realistic editorial photo {core}",
                f"craftsman workshop process realistic photo {core}",
            ])
    elif family == "education":
        if any(x in src for x in ("онлайн", "online", "курс", "course", "edtech")):
            out.extend([
                f"online learning laptop education realistic editorial photo {core}",
                f"e-learning course study digital realistic photo {core}",
            ])
        elif any(x in src for x in ("книг", "book", "библиотек", "library")):
            out.extend([
                f"books library study reading realistic editorial photo {core}",
                f"education book learning realistic photo {core}",
            ])
        else:
            out.extend([
                f"education learning study classroom realistic editorial photo {core}",
                f"student study focused realistic photo {core}",
                f"teaching learning process realistic editorial photo {core}",
            ])
    elif family == "finance":
        if any(x in src for x in ("крипт", "crypto", "биткоин", "bitcoin")):
            out.extend([
                f"cryptocurrency digital finance realistic editorial photo {core}",
                f"blockchain digital investment realistic photo {core}",
            ])
        elif any(x in src for x in ("аналитик", "analytics", "график", "chart", "dashboard")):
            out.extend([
                f"financial analytics dashboard chart realistic editorial photo {core}",
                f"data analytics finance business realistic photo {core}",
            ])
        else:
            out.extend([
                f"finance investment professional business realistic editorial photo {core}",
                f"financial planning money realistic photo {core}",
                f"investment strategy business realistic editorial photo {core}",
            ])
    elif family == "marketing":
        if any(x in src for x in ("smm", "соцсет", "social media", "контент")):
            out.extend([
                f"social media marketing content creation realistic editorial photo {core}",
                f"digital marketing strategy realistic photo {core}",
            ])
        elif any(x in src for x in ("аналитик", "analytics", "метрик", "metric")):
            out.extend([
                f"marketing analytics dashboard metrics realistic editorial photo {core}",
                f"data metrics marketing realistic photo {core}",
            ])
        else:
            out.extend([
                f"marketing strategy team professional realistic editorial photo {core}",
                f"creative marketing workspace realistic photo {core}",
                f"digital marketing campaign realistic editorial photo {core}",
            ])
    elif family == "lifestyle":
        if any(x in src for x in ("путешеств", "travel", "поездк")):
            out.extend([
                f"travel lifestyle destination editorial photo {core}",
                f"travel adventure realistic editorial photo {core}",
            ])
        elif any(x in src for x in ("утро", "morning", "рутин", "routine")):
            out.extend([
                f"morning routine lifestyle cozy realistic editorial photo {core}",
                f"morning coffee calm lifestyle realistic photo {core}",
            ])
        else:
            out.extend([
                f"lifestyle everyday authentic realistic editorial photo {core}",
                f"lifestyle moment natural realistic photo {core}",
                f"everyday life calm editorial photo {core}",
            ])
    elif family == "expert_blog":
        out.extend([
            f"professional expert thoughtful workspace realistic editorial photo {core}",
            f"specialist focused work professional realistic photo {core}",
            f"expert knowledge professional environment realistic editorial photo {core}",
        ])
    elif family == "hardware":
        if any(x in src for x in ("видеокарт", "gpu", "graphics")):
            out.extend([
                f"graphics card pc hardware closeup realistic editorial photo {core}",
                f"computer hardware bench gpu components realistic photo {core}",
            ])
        elif any(x in src for x in ("ноут", "laptop")):
            out.extend([
                f"laptop workspace realistic editorial photo {core}",
                f"portable computer desk setup realistic photo {core}",
            ])
        else:
            out.extend([
                f"computer hardware workstation realistic editorial photo {core}",
                f"pc setup desk technology realistic photo {core}",
            ])
    elif family == "cars":
        if any(x in src for x in ("салон", "interior", "cockpit")):
            out.extend([
                f"car interior dashboard cockpit realistic editorial photo {core}",
                f"vehicle interior steering wheel realistic photo {core}",
            ])
        elif any(x in src for x in ("двигат", "engine", "ремонт", "service")):
            out.extend([
                f"car engine service workshop realistic photo {core}",
                f"automotive repair garage editorial photo {core}",
            ])
        else:
            out.extend([
                f"automotive exterior road realistic editorial photo {core}",
                f"car vehicle realistic photo {core}",
            ])
    elif family == "gaming":
        out.extend([
            f"gaming setup monitor controller desk realistic editorial photo {core}",
            f"esports room desk setup realistic photo {core}",
            f"video game hardware controller screen realistic editorial photo",
        ])
    elif family == "tech":
        out.extend([
            f"software development code screen monitor workspace {core}",
            f"technology server data center infrastructure {core}",
            f"programming developer office professional workspace {core}",
        ])
    elif family == "business":
        out.extend([
            f"business office meeting strategy analytics dashboard {core}",
            f"marketing team workspace professional office {core}",
            f"corporate presentation boardroom professional {core}",
        ])
    else:
        base = _meaningful_query_core(query, family)
        if base:
            out.extend([
                f"{base} realistic editorial photo",
                f"{base} realistic photo",
                f"{base} professional environment photo",
            ])
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in out:
        item = re.sub(r"\s+", " ", item).strip()[:MAX_QUERY_LEN]
        if item and item not in seen:
            seen.add(item)
            cleaned.append(item)
    return cleaned[:5]


def _topic_to_english(query: str) -> str:
    family = detect_topic_family(query)
    if family in TOPIC_FAMILY_TERMS:
        return " ".join(TOPIC_FAMILY_TERMS[family]["en"][:3])
    return _clean_text(query)


def _short_topic_nouns(query: str) -> str:
    """Extract exactly 1-2 highly relevant English nouns for the topic query.

    Stock photo APIs return much more relevant results when given short, precise
    English noun phrases instead of long, mixed-language sentences.
    """
    family = detect_topic_family(query)
    # Use the most representative English terms for known families
    _FAMILY_SHORT: dict[str, str] = {
        "massage": "massage therapist",
        "food": "food dish restaurant",
        "health": "wellness healthy lifestyle",
        "beauty": "beauty skincare cosmetics",
        "local_business": "local business workshop",
        "education": "education learning study",
        "finance": "finance money investment",
        "marketing": "marketing analytics digital",
        "lifestyle": "lifestyle morning routine",
        "expert_blog": "professional expert workspace",
        "cars": "car automotive",
        "gaming": "gaming controller",
        "hardware": "computer hardware",
        "tech": "technology software",
        "business": "business office",
    }
    if family in _FAMILY_SHORT:
        return _FAMILY_SHORT[family]
    # For generic topics: take up to 2 latin tokens from the query that are not stopwords
    latin_tokens = [t for t in _query_tokens(query) if _LATIN_TOKEN_RE.match(t)]
    if latin_tokens:
        return " ".join(latin_tokens[:2])
    return ""


def _translate_literal_topic(query: str, family: str, subfamily: str) -> str:
    """Return an English translation of the literal topic for use as the primary image query.

    Priority order:
    1. Detected subfamily name (already English, e.g. "coffee", "nails", "crypto")
    2. Latin tokens already present in the query
    3. Compact Russian→English lookup table (_LITERAL_TRANSLATIONS)
    4. First 2 English terms for the detected family (soft fallback)
    """
    # 1. Subfamily name is already English and maximally specific
    if subfamily and subfamily != "generic":
        return subfamily.replace("_", " ")

    # 2. Latin tokens present in the query — use them directly
    latin_tokens = [t for t in _query_tokens(query) if _LATIN_TOKEN_RE.match(t)]
    if latin_tokens:
        return " ".join(latin_tokens[:3])

    # 3. Russian→English lookup for common topic words
    q_lower = _clean_text(query)
    for ru_prefix, en_translation in _LITERAL_TRANSLATIONS:
        if ru_prefix in q_lower:
            return en_translation

    # 4. Family-level fallback (least specific)
    family_terms = TOPIC_FAMILY_TERMS.get(family, {}).get("en", [])[:2]
    return " ".join(family_terms) if family_terms else ""


def _generic_queries(query: str) -> list[str]:
    family = detect_topic_family(query)
    eng = _topic_to_english(query)
    core = _meaningful_query_core(query, family) or eng
    out = []
    if core:
        if family in STRICT_IMAGE_FAMILIES:
            # For strict families: use topical queries only
            out.extend([
                f"{core} professional editorial photo",
                f"{core} professional environment photo",
            ])
        else:
            out.extend([
                f"{core} realistic editorial photo",
                f"{core} realistic professional photo",
            ])
    # Use family-specific image queries as fallback (covers all 15+ families)
    family_queries = get_family_image_queries(family)
    out.extend(family_queries[:2])
    return out


def build_best_visual_queries(query: str) -> list[str]:
    """Build image search queries with the literal topic as the primary term.

    New order (Objective 2 fix — literal topic first):
    1. Literal translated topic as the PRIMARY query (most specific)
    2. 2 variants of the literal topic with context/scene/angle
    3. Controlled query expansion candidates (subject-focused + context-focused)
    4. Existing specialized queries (family+subtype aware)
    5. Subfamily-specific queries
    6. Family image queries as soft boost/fallback (appended at end)
    7. Generic queries as final fallback
    """
    family = detect_topic_family(query)
    subfamily = detect_subfamily(family, query)

    # Step 1: Translate literal topic to English — PRIMARY query
    literal_en = _translate_literal_topic(query, family, subfamily)
    # family_boost_applied: True when literal translation falls back to family terms
    family_boost_applied = bool(
        not literal_en
        or literal_en in " ".join(TOPIC_FAMILY_TERMS.get(family, {}).get("en", []))
    )

    queries: list[str] = []
    if literal_en:
        # Primary: bare literal topic (most relevant for stock photo API)
        queries.append(literal_en)
        # Step 2: 2 context/scene variants of the literal topic
        queries.append(f"fresh {literal_en} close-up editorial photo")
        queries.append(f"{literal_en} professional realistic photo")

    # Step 3: Controlled query expansion — subject-focused + context-focused candidates
    # For complex multi-word topics (e.g. "производство китайских машин"),
    # generates candidates that separately emphasize the subject and the context.
    expansion_candidates = _expand_query_candidates(query, family, literal_en)
    queries.extend(expansion_candidates)

    # Step 4: Existing family-aware specialized queries (sub-type detection)
    queries.extend(_specialized_queries(query, family))

    # Step 5: Subfamily-specific queries when available
    if subfamily:
        sub_queries = get_subfamily_image_queries(family, subfamily)
        if sub_queries:
            queries.extend(sub_queries)

    # Step 6 & 7: Family queries as soft boost / fallback (at the END, not start)
    queries.extend(get_family_image_queries(family))
    queries.extend(_generic_queries(query))

    logger.info(
        "IMAGE_QUERY_BUILD literal_topic=%r literal_en=%r family=%s subfamily=%s "
        "family_boost_applied=%s expanded=%d queries=%s",
        query[:80], literal_en, family, subfamily,
        family_boost_applied, len(expansion_candidates),
        [q[:60] for q in queries[:6]],
    )

    cleaned: list[str] = []
    seen = set()
    for item in queries:
        item = _strip_non_latin(re.sub(r"\s+", " ", item).strip())[:MAX_QUERY_LEN]
        if item and item not in seen:
            seen.add(item)
            cleaned.append(item)
    return cleaned[:10]


def build_visual_fallback(query: str) -> str:
    queries = build_best_visual_queries(query)
    return queries[0] if queries else _clean_text(query)


def _resolved_proxy_url() -> str | None:
    proxy = (IMAGE_HTTP_PROXY_URL or "").strip()
    if not proxy:
        return None
    if proxy.lower().startswith("socks"):
        try:
            import socksio  # type: ignore # noqa: F401
        except Exception:
            return None
    return proxy


def _make_client() -> httpx.AsyncClient:
    kwargs = dict(
        timeout=httpx.Timeout(connect=CONNECT_TIMEOUT, read=READ_TIMEOUT, write=READ_TIMEOUT, pool=READ_TIMEOUT),
        follow_redirects=True,
        trust_env=False,
        headers={"User-Agent": "NeuroSMM/1.0", "Accept": "application/json, */*"},
    )
    proxy = _resolved_proxy_url()
    if proxy:
        kwargs["proxy"] = proxy
    return httpx.AsyncClient(**kwargs)


def _url_ok(url: str) -> bool:
    if not url or not url.startswith("http"):
        return False
    lowered = url.lower()
    if any(part in lowered for part in BAD_URL_PARTS):
        return False
    try:
        return bool(urlparse(url).netloc)
    except Exception:
        return False


def _image_fingerprint(url: str) -> str:
    raw = str(url or "").strip().lower()
    if not raw:
        return ""
    m = re.search(r"photo-([\w-]+)", raw)
    if m and "unsplash" in raw:
        return f"unsplash:{m.group(1)}"
    m = re.search(r"/photos/(\d+)/", raw)
    if m and "pexels" in raw:
        return f"pexels:{m.group(1)}"
    m = re.search(r"pixabay\.com.*?[_/](\d{6,})", raw)
    if m:
        return f"pixabay:{m.group(1)}"
    return raw.split("?", 1)[0]


def _prepare_used_refs(used_refs: set[str] | None) -> set[str]:
    return {_image_fingerprint(x) for x in (used_refs or set()) if _image_fingerprint(x)}


def _not_used(url: str, used_fps: set[str]) -> bool:
    fp = _image_fingerprint(url)
    return bool(fp) and fp not in used_fps


# Pre-compiled word-boundary patterns per family for _detect_meta_family.
# Built once at import time to avoid re-compiling on every call.
_META_FAMILY_PATTERNS: dict[str, list[tuple[str, re.Pattern]]] = {}
_META_FAMILY_PRIORITY = [
    "massage", "food", "beauty", "local_business",
    "cars", "gaming", "hardware",
    "health",
    "education", "finance", "marketing", "lifestyle", "expert_blog",
    "tech", "business",
]
for _fam in _META_FAMILY_PRIORITY:
    _block = TOPIC_FAMILY_TERMS.get(_fam, {})
    # Deduplicate terms across ru/en to avoid double-counting
    _terms = list(dict.fromkeys(_block.get("ru", []) + _block.get("en", [])))
    _META_FAMILY_PATTERNS[_fam] = [
        (t, re.compile(rf"\b{re.escape(t)}")) for t in _terms
    ]


def _detect_meta_family(text: str) -> str:
    """Detect topic family of an image from its metadata text.

    Unlike detect_topic_family() which trusts user-provided short topic strings,
    image metadata is longer and noisier.  We use word-boundary matching and
    require at least 2 distinct term hits before declaring a confident match.

    This avoids false positives like "spa" ⊂ "workspace", "pr" ⊂ "professional",
    "diet" ⊂ "editorial" that plague naive substring search on metadata.

    Returns a family name or "generic" when no confident match.
    """
    if not text:
        return "generic"
    q = text.strip().lower().replace("ё", "е")
    q = re.sub(r"\s+", " ", q)

    for family in _META_FAMILY_PRIORITY:
        patterns = _META_FAMILY_PATTERNS.get(family, [])
        hits = sum(1 for _term, pat in patterns if pat.search(q))
        if hits >= 2:
            return family
    return "generic"


def _meta_score(family: str, meta_text: str, query: str = "", post_text: str = "") -> tuple[int, str]:
    """Compute semantic relevance score for an image candidate.

    Returns (score, reject_reason) — reject_reason is non-empty when
    the candidate is explicitly off-topic.

    Systemic relevance gate:
    1. Positive signals: family term matches, query token matches, allowed visual class matches
    2. Cross-family penalty: if image metadata belongs to a DIFFERENT family → hard penalty
    3. Blocked visual veto: per-family blocked classes → hard penalty
    4. Negative hints & anti-repeat penalties
    5. Positive affirmation: tracked via allowed_hits + query_hits for caller
    """
    text = _clean_text(meta_text)
    if not text:
        return 0, "empty_meta"
    score = 0
    reject_reason = ""

    # --- Positive signals ---

    # Family term matches
    family_term_hits = 0
    if family in TOPIC_FAMILY_TERMS:
        terms = TOPIC_FAMILY_TERMS[family].get("ru", []) + TOPIC_FAMILY_TERMS[family].get("en", [])
        family_term_hits = sum(1 for token in terms if token in text)
        score += family_term_hits * 7

    # Query token matches (direct relevance)
    query_tokens = _query_tokens(query)
    query_hits = sum(1 for token in query_tokens[:8] if token in text)
    score += query_hits * 11

    # Post-text relevance boost: reward candidates whose metadata aligns with
    # actual nouns/subjects from the generated post body (not just the search query).
    post_hits = 0
    if post_text:
        post_tokens = _query_tokens(post_text)[:16]
        post_hits = sum(1 for token in post_tokens if token in text)
        score += post_hits * 6
        # Strong post-text alignment bonus: if multiple post-specific terms match,
        # this image is specifically relevant to THIS post, not just the family
        if post_hits >= 3:
            score += 12  # Extra bonus for strong post-specific alignment

    # Allowed visual class matches (strong positive signal)
    allowed = get_family_allowed_visuals(family)
    allowed_hits = 0
    if allowed:
        allowed_hits = sum(1 for cls in allowed if cls in text)
        score += allowed_hits * 14

    # --- Cross-family coherence penalty (SYSTEMIC) ---
    # Detect what family the image metadata itself belongs to.
    # If the image is clearly from a DIFFERENT family, it's likely off-topic.
    # This catches ALL cross-family leaks generically (food→massage, beauty→finance, etc.)
    meta_family = _detect_meta_family(text)
    cross_family_mismatch = False
    if meta_family != "generic" and meta_family != family:
        cross_family_mismatch = True
        score -= 30
        reject_reason = f"cross_family:{meta_family}"

    # --- Negative signals ---

    # Family-specific bad hints
    for bad in _NEGATIVE_HINTS.get(family, []) + _NEGATIVE_HINTS["generic"]:
        if bad in text:
            score -= 18

    # Hard veto: blocked_visual_classes — categorically off-topic for this family.
    # Uses word-boundary matching to avoid false positives (e.g. 'food' ≠ 'seafood').
    blocked = get_family_blocked_visuals(family)
    if blocked:
        for cls in blocked:
            if re.search(rf"\b{re.escape(cls)}\b", text, re.I):
                score -= 40
                if not reject_reason:
                    reject_reason = f"blocked_visual:{cls}"

    # Broad irrelevant-image class penalties for strict families
    if family in STRICT_IMAGE_FAMILIES:
        family_irrelevant = get_family_irrelevant_classes(family)
        for cls in family_irrelevant:
            if cls in text:
                score -= 20

    # Anti-repeat motifs (generic/overused imagery)
    for motif in _ANTI_REPEAT_MOTIFS:
        if motif in text:
            score -= 10

    # Generic stock image penalty (heuristic): images with generic metadata
    # but no specific alignment to the post content are likely filler
    _generic_stock_signals = [
        "stock photo", "shutterstock", "istockphoto", "getty images",
        "abstract background", "business team meeting", "happy people",
        "teamwork concept", "success concept", "idea concept",
        "generic office", "diverse group", "people working",
        "handshake", "brainstorm", "motivation", "inspiration concept",
        "copy space", "banner template", "flat lay", "mock up",
        "blank space", "placeholder", "presentation template",
        "growth chart", "puzzle pieces", "light bulb idea", "target goal",
        "thumbs up", "high five", "fist bump",
    ]
    generic_stock_hits = sum(1 for gs in _generic_stock_signals if gs in text)
    if generic_stock_hits >= 1:
        # Only penalize if there's no strong post-specific alignment
        if post_hits < 3 and query_hits < 3:
            score -= 20
            if not reject_reason:
                reject_reason = "generic_stock_image"

    # --- Positive affirmation requirement ---
    # A candidate must have at least SOME positive signal from the target family.
    # Without any allowed_visual hit AND without meaningful query token matches,
    # the image is likely generic/random even if it doesn't trigger negatives.
    has_affirmation = (allowed_hits >= 1) or (query_hits >= 2) or (family_term_hits >= 2) or (post_hits >= 3)
    if not has_affirmation and score > 0:
        # Cap score: candidate has no clear positive signal for target family
        score = min(score, MIN_RELEVANCE_SCORE - 1)
        if not reject_reason:
            reject_reason = "no_positive_affirmation"

    return score, reject_reason


async def _search_pexels(client: httpx.AsyncClient, query: str, used_fps: set[str], family: str, post_text: str = "") -> tuple[str, int, str]:
    if not PEXELS_API_KEY or not _provider_available("pexels"):
        return "", 0, ""
    try:
        r = await client.get(
            "https://api.pexels.com/v1/search",
            params={"query": query, "per_page": 12, "orientation": "landscape"},
            headers={"Authorization": PEXELS_API_KEY},
        )
        r.raise_for_status()
        data = r.json()
        _provider_success("pexels")
    except httpx.TimeoutException as e:
        logger.warning("pexels search timeout query=%r err=%s", query, e)
        _provider_failure("pexels")
        return "", 0, ""
    except httpx.HTTPStatusError as e:
        logger.warning("pexels search HTTP %s query=%r", e.response.status_code, query)
        if e.response.status_code in (401, 403):
            logger.error("pexels: API key invalid or forbidden")
        _provider_failure("pexels")
        return "", 0, ""
    except (httpx.ConnectError, OSError) as e:
        logger.warning("pexels search network error query=%r err=%s", query, e)
        _provider_failure("pexels")
        return "", 0, ""
    except Exception as e:
        logger.error("pexels search unexpected error query=%r err=%s", query, e, exc_info=True)
        _provider_failure("pexels")
        return "", 0, ""
    best_url = ""
    best_score = -1
    best_meta = ""
    for item in data.get("photos") or []:
        meta = " ".join([str(item.get("alt") or ""), str(item.get("photographer") or ""), str(item.get("avg_color") or "")])
        for key in ("large2x", "large", "original", "medium"):
            url = (item.get("src") or {}).get(key) or ""
            if _url_ok(url) and _not_used(url, used_fps):
                score, _reason = _meta_score(family, meta, query, post_text)
                if score > best_score:
                    best_score = score
                    best_url = url.strip()
                    best_meta = meta
    return best_url, best_score, best_meta


async def _search_unsplash(client: httpx.AsyncClient, query: str, used_fps: set[str], family: str, post_text: str = "") -> tuple[str, int, str]:
    if not UNSPLASH_ACCESS_KEY or not _provider_available("unsplash"):
        return "", 0, ""
    try:
        r = await client.get(
            "https://api.unsplash.com/search/photos",
            params={"query": query, "per_page": 12, "orientation": "landscape", "content_filter": "high"},
            headers={"Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"},
        )
        r.raise_for_status()
        data = r.json()
        _provider_success("unsplash")
    except httpx.TimeoutException as e:
        logger.warning("unsplash search timeout query=%r err=%s", query, e)
        _provider_failure("unsplash")
        return "", 0, ""
    except httpx.HTTPStatusError as e:
        logger.warning("unsplash search HTTP %s query=%r", e.response.status_code, query)
        if e.response.status_code in (401, 403):
            logger.error("unsplash: API key invalid or forbidden")
        _provider_failure("unsplash")
        return "", 0, ""
    except (httpx.ConnectError, OSError) as e:
        logger.warning("unsplash search network error query=%r err=%s", query, e)
        _provider_failure("unsplash")
        return "", 0, ""
    except Exception as e:
        logger.error("unsplash search unexpected error query=%r err=%s", query, e, exc_info=True)
        _provider_failure("unsplash")
        return "", 0, ""
    best_url = ""
    best_score = -1
    best_meta = ""
    for item in data.get("results") or []:
        meta = " ".join([
            str(item.get("description") or ""),
            str(item.get("alt_description") or ""),
            " ".join((tag.get("title") or "") for tag in (item.get("tags") or []) if isinstance(tag, dict)),
            str((item.get("user") or {}).get("name") or ""),
        ])
        for key in ("regular", "full", "raw", "small"):
            url = (item.get("urls") or {}).get(key) or ""
            if _url_ok(url) and _not_used(url, used_fps):
                score, _reason = _meta_score(family, meta, query, post_text)
                if score > best_score:
                    best_score = score
                    best_url = url.strip()
                    best_meta = meta
    return best_url, best_score, best_meta


async def _search_pixabay(client: httpx.AsyncClient, query: str, used_fps: set[str], family: str, post_text: str = "") -> tuple[str, int, str]:
    if not PIXABAY_API_KEY or not _provider_available("pixabay"):
        return "", 0, ""
    try:
        r = await client.get(
            "https://pixabay.com/api/",
            params={
                "key": PIXABAY_API_KEY,
                "q": query,
                "image_type": "photo",
                "orientation": "horizontal",
                "per_page": 12,
                "safesearch": "true",
                "min_width": 800,
            },
        )
        r.raise_for_status()
        data = r.json()
        _provider_success("pixabay")
    except httpx.TimeoutException as e:
        logger.warning("pixabay search timeout query=%r err=%s", query, e)
        _provider_failure("pixabay")
        return "", 0, ""
    except httpx.HTTPStatusError as e:
        logger.warning("pixabay search HTTP %s query=%r", e.response.status_code, query)
        if e.response.status_code in (401, 403):
            logger.error("pixabay: API key invalid or forbidden")
        _provider_failure("pixabay")
        return "", 0, ""
    except (httpx.ConnectError, OSError) as e:
        logger.warning("pixabay search network error query=%r err=%s", query, e)
        _provider_failure("pixabay")
        return "", 0, ""
    except Exception as e:
        logger.error("pixabay search unexpected error query=%r err=%s", query, e, exc_info=True)
        _provider_failure("pixabay")
        return "", 0, ""
    best_url = ""
    best_score = -1
    best_meta = ""
    for item in data.get("hits") or []:
        meta = " ".join([str(item.get("tags") or ""), str(item.get("user") or ""), str(item.get("type") or "")])
        url = str(item.get("largeImageURL") or item.get("webformatURL") or "").strip()
        if not url:
            continue
        if _url_ok(url) and _not_used(url, used_fps):
            score, _reason = _meta_score(family, meta, query, post_text)
            if score > best_score:
                best_score = score
                best_url = url
                best_meta = meta
    return best_url, best_score, best_meta


async def find_image(
    query: str,
    used_refs: set[str] | None = None,
    *,
    topic: str = "",
    post_text: str = "",
    title: str = "",
    mode: str = "",
) -> str:
    """Find the best semantically relevant image for the given query.

    Uses the new POST-CENTRIC pipeline when post_text or title is available.
    Falls back to the legacy query-based flow otherwise.

    Args:
        mode: "autopost" (strict, prefers no-image over junk) or
              "editor" (lenient, tolerable for user selection).
              Empty string defaults to "autopost".

    New pipeline flow:
    1. Extract visual intent from post text (title + body)
    2. Word-sense disambiguation to avoid wrong-meaning images
    3. Post-centric scoring (PRIMARY) + provider bonus (CAPPED secondary)
    4. Top-N reranking → mode-specific threshold → accept or no-image
    5. Channel topic used ONLY as weak fallback

    Legacy flow (backward-compatible):
    Uses family detection from query + topic, build_best_visual_queries, _meta_score.
    """
    # --- Pipeline v3 (new core) ---
    from image_pipeline_v3 import run_pipeline_v3, MODE_AUTOPOST as V3_AUTOPOST, MODE_EDITOR as V3_EDITOR
    pipeline_mode = mode if mode in (V3_AUTOPOST, V3_EDITOR) else V3_AUTOPOST

    actual_post_text = post_text or ""
    actual_title = title or query or ""
    if actual_post_text or actual_title:
        result = await run_pipeline_v3(
            title=actual_title,
            body=actual_post_text,
            channel_topic=topic,
            used_refs=used_refs,
            mode=pipeline_mode,
        )
        if result.has_image:
            logger.info(
                "IMAGE_V3_ACCEPT url=%r score=%d source=%s mode=%s query=%r",
                result.image_url[:80], result.score, result.source_provider,
                pipeline_mode, (result.matched_query or "")[:60],
            )
            return result.image_url
        if result.no_image_reason:
            logger.info(
                "IMAGE_V3_NO_MATCH reason=%s outcome=%s mode=%s query=%r",
                result.no_image_reason, result.outcome, pipeline_mode, query[:60],
            )
        else:
            logger.info(
                "IMAGE_V3_REJECT mode=%s query=%r reasons=%s",
                pipeline_mode, query[:60], result.reject_reasons[:5],
            )
        # If v3 pipeline found nothing, try legacy as last resort
        # but only if there's query text distinct from post text
        if not query or query == actual_title:
            return ""
        logger.info(
            "IMAGE_LEGACY_FALLBACK_USED reason=v3_no_result mode=%s query=%r",
            pipeline_mode, query[:60],
        )

    # --- LEGACY: Query-based flow (backward-compatible fallback) ---
    queries = build_best_visual_queries(query)
    if not queries:
        return ""
    combined_text = query + " " + topic
    family = detect_topic_family(combined_text)
    subfamily = detect_subfamily(family, combined_text)
    visual_type = classify_visual_type(topic, query, post_text)
    used_fps = _prepare_used_refs(used_refs)

    # For text-only visual types, skip image search entirely
    if visual_type == VISUAL_TYPE_TEXT_ONLY:
        logger.info(
            "IMAGE_SKIPPED_TEXT_ONLY family=%s subfamily=%s query=%r",
            family, subfamily, query,
        )
        return ""

    logger.info(
        "IMAGE_POLICY_START family=%s subfamily=%s visual_type=%s queries=%s "
        "used=%s min_score=%s",
        family, subfamily, visual_type, queries, len(used_fps), MIN_RELEVANCE_SCORE,
    )

    best_url = ""
    best_score = -1
    best_source = ""
    best_query = ""
    best_meta = ""
    candidates_evaluated = 0
    candidates_rejected = 0

    async with _make_client() as client:
        # Source priority: unsplash > pexels > pixabay
        for source_name, search_fn in [
            ("unsplash", _search_unsplash),
            ("pexels", _search_pexels),
            ("pixabay", _search_pixabay),
        ]:
            for q in queries:
                url, score, meta = await search_fn(client, q, used_fps, family, post_text)
                candidates_evaluated += 1
                if not url:
                    continue
                if score < MIN_RELEVANCE_SCORE:
                    candidates_rejected += 1
                    # Compute reject reason for logging
                    _score, reject_reason = _meta_score(family, meta, q, post_text)
                    meta_family = _detect_meta_family(_clean_text(meta))
                    logger.info(
                        "IMAGE_CANDIDATE_REJECT score=%d min=%d source=%s query=%r "
                        "family=%s meta_family=%s reason=%s meta=%r url=%r",
                        score, MIN_RELEVANCE_SCORE, source_name, q,
                        family, meta_family, reject_reason or "low_score",
                        meta[:120], url[:80],
                    )
                    continue
                if score > best_score:
                    best_score = score
                    best_url = url
                    best_source = source_name
                    best_query = q
                    best_meta = meta
                    # Early exit: if we found a clearly good candidate, stop searching
                    # weaker fallback queries for this source to save API calls
                    if best_score >= MIN_RELEVANCE_SCORE * 2:
                        break

    # Final coherence gate on the winning candidate
    if best_url and best_meta:
        meta_family = _detect_meta_family(_clean_text(best_meta))
        if meta_family != "generic" and meta_family != family:
            logger.warning(
                "IMAGE_FINAL_GATE_REJECT source=%s score=%d family=%s meta_family=%s "
                "query=%r meta=%r url=%r — cross-family mismatch at final gate",
                best_source, best_score, family, meta_family,
                best_query, best_meta[:120], best_url[:80],
            )
            best_url = ""

    if best_url:
        meta_family = _detect_meta_family(_clean_text(best_meta)) if best_meta else "unknown"
        logger.info(
            "IMAGE_SELECTED source=%s score=%d family=%s meta_family=%s subfamily=%s "
            "visual_type=%s query=%r meta=%r url=%r evaluated=%d rejected=%d",
            best_source, best_score, family, meta_family, subfamily, visual_type,
            best_query, best_meta[:120], best_url[:80],
            candidates_evaluated, candidates_rejected,
        )
        return best_url

    logger.info(
        "IMAGE_SKIPPED_NO_RELEVANT_CANDIDATE min_threshold=%d family=%s subfamily=%s "
        "visual_type=%s queries=%s evaluated=%d rejected=%d",
        MIN_RELEVANCE_SCORE, family, subfamily, visual_type, queries,
        candidates_evaluated, candidates_rejected,
    )
    return ""


async def trigger_unsplash_download(download_location: str) -> bool:
    if not download_location or not UNSPLASH_ACCESS_KEY:
        return False
    try:
        kwargs: dict = {
            "headers": {"Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}", "Accept-Version": "v1"},
            "timeout": httpx.Timeout(8.0, connect=5.0),
            "follow_redirects": True,
        }
        proxy_url = _resolved_proxy_url()
        if proxy_url:
            kwargs["proxy"] = proxy_url
        async with httpx.AsyncClient(**kwargs) as client:
            resp = await client.get(download_location)
            return 200 <= resp.status_code < 400
    except Exception:
        return False


# Known CDN/asset URL patterns where the path is a hash/ID, not a semantic filename.
# These URLs never contain topic keywords and must not be rejected by URL keyword matching.
_CDN_ASSET_PATTERNS: list[re.Pattern] = [
    re.compile(r"pixabay\.com/get/", re.I),
    re.compile(r"images\.pexels\.com/photos/\d+/", re.I),
    re.compile(r"cdn\.pixabay\.com/", re.I),
    re.compile(r"images\.unsplash\.com/photo-", re.I),
    re.compile(r"live\.staticflickr\.com/", re.I),
    re.compile(r"openverse\.org/.*/thumb/", re.I),
]


def _is_cdn_asset_url(url: str) -> bool:
    """Return True if the URL is a known CDN/asset URL with non-semantic path."""
    for pattern in _CDN_ASSET_PATTERNS:
        if pattern.search(url):
            return True
    return False


def _provider_from_url(url: str) -> str:
    """Extract a short provider name from the image URL."""
    u = (url or "").lower()
    if "pixabay" in u:
        return "pixabay"
    if "pexels" in u:
        return "pexels"
    if "unsplash" in u:
        return "unsplash"
    if "openverse" in u:
        return "openverse"
    if "flickr" in u:
        return "flickr"
    return "unknown"


def _log_image_decision_trace(
    *,
    url: str,
    mode: str,
    subject: str,
    family: str,
    provider: str,
    has_meta: bool,
    has_page_url: bool,
    reject_reason: str,
    accept_reason: str,
    final_score: int,
    legacy_fallback_used: bool,
) -> None:
    """Log a compact decision trace for each image candidate evaluation."""
    logger.info(
        "IMAGE_DECISION_TRACE mode=%s subject=%r family=%s provider=%s "
        "meta=%s page_url=%s reject=%s accept=%s score=%d "
        "legacy_fallback=%s url=%r",
        mode, (subject or "")[:40], family, provider,
        "yes" if has_meta else "no",
        "yes" if has_page_url else "no",
        reject_reason or "-",
        accept_reason or "-",
        final_score,
        "yes" if legacy_fallback_used else "no",
        (url or "")[:80],
    )


def validate_image_for_autopost(
    image_ref: str,
    *,
    topic: str = "",
    prompt: str = "",
    post_text: str = "",
    image_meta: str = "",
    title: str = "",
    mode: str = "",
) -> bool:
    """Final quality gate for autopost: decide if an image is safe to publish.

    Returns True if the image passes, False if it should be rejected.

    Args:
        mode: "autopost" (strict) or "editor" (lenient).
              Empty string defaults to "autopost".

    Uses the new post-centric validation when post_text/title available:
    1. Wrong-sense hard reject via visual intent
    2. Subject match validation
    3. Generic stock detection

    Legacy checks (always applied):
    1. URL-based signals (bad URL parts for the family)
    2. Meta-based cross-family coherence (if image_meta provided)
    """
    if not image_ref or not image_ref.startswith("http"):
        return True  # no image or local media — nothing to reject

    url_lower = image_ref.lower()

    # --- Post-centric validation (v3) ---
    actual_post_text = post_text or ""
    actual_title = title or prompt or ""
    if (actual_post_text or actual_title) and image_meta:
        from image_pipeline_v3 import validate_image_post_centric_v3, MODE_AUTOPOST as V3_AUTO, MODE_EDITOR as V3_ED
        from visual_intent_v2 import extract_visual_intent_v2
        pipeline_mode = mode if mode in (V3_AUTO, V3_ED) else V3_AUTO
        intent = extract_visual_intent_v2(
            title=actual_title,
            body=actual_post_text,
            channel_topic=topic,
        )
        is_valid, reject_reason = validate_image_post_centric_v3(
            image_ref, intent=intent, image_meta=image_meta, mode=pipeline_mode,
        )
        if not is_valid:
            logger.warning(
                "VALIDATE_V3_REJECT reason=%s mode=%s url=%r",
                reject_reason, pipeline_mode, image_ref[:80],
            )
            return False

    # --- LEGACY: Channel-based checks (always run for safety) ---
    # Prioritize post subject for family detection: post_text/title contain the
    # actual subject, while channel topic is only a weak prior.
    _family_source = ""
    if post_text:
        _family_source = post_text[:400]
    if title:
        _family_source = (title + " " + _family_source).strip()
    if not _family_source:
        _family_source = topic + " " + prompt
    family = detect_topic_family(_family_source)

    # Check 1: URL-based off-topic signals
    if family in STRICT_IMAGE_FAMILIES:
        bad_signals = get_family_bad_url_signals(family)
        for signal in bad_signals:
            if signal in url_lower:
                logger.warning(
                    "VALIDATE_REJECT_URL signal=%r family=%s url=%r",
                    signal, family, image_ref[:80],
                )
                return False

    # Check 2: Meta-based cross-family coherence
    if image_meta:
        meta_text = _clean_text(image_meta)
        meta_family = _detect_meta_family(meta_text)
        if meta_family != "generic" and meta_family != family:
            logger.warning(
                "VALIDATE_REJECT_CROSS_FAMILY family=%s meta_family=%s "
                "meta=%r url=%r",
                family, meta_family, image_meta[:120], image_ref[:80],
            )
            return False

        # Also check blocked_visual_classes against meta
        blocked = get_family_blocked_visuals(family)
        if blocked:
            for cls in blocked:
                if re.search(rf"\b{re.escape(cls)}\b", meta_text, re.I):
                    logger.warning(
                        "VALIDATE_REJECT_BLOCKED_VISUAL cls=%r family=%s "
                        "meta=%r url=%r",
                        cls, family, image_meta[:120], image_ref[:80],
                    )
                    return False

    # Check 3: Minimum relevance score for images with metadata
    if image_meta:
        combined_query = " ".join(filter(None, [topic, prompt]))
        score, reason = _meta_score(family, _clean_text(image_meta), combined_query, post_text)
        if score < MIN_RELEVANCE_SCORE:
            logger.warning(
                "VALIDATE_REJECT_LOW_SCORE score=%d min=%d family=%s reason=%s meta=%r url=%r",
                score, MIN_RELEVANCE_SCORE, family, reason, image_meta[:120], image_ref[:80],
            )
            return False

    # Check 4: URL-path keyword sanity for images WITHOUT metadata.
    # SKIP for CDN/asset URLs where path is a hash/ID and never contains topic words.
    if not image_meta and (topic or prompt):
        if _is_cdn_asset_url(image_ref):
            logger.debug(
                "VALIDATE_SKIP_CDN_URL family=%s url=%r topic=%r — "
                "CDN/asset URL has no semantic path, skipping URL keyword check",
                family, image_ref[:80], (topic or "")[:40],
            )
        else:
            url_path = urlparse(image_ref).path.lower().replace("-", " ").replace("_", " ")
            _check_tokens = set()
            for src in (topic, prompt):
                for tok in re.findall(r"[a-z]{4,}", (src or "").lower()):
                    _check_tokens.add(tok)
            if post_text:
                for tok in re.findall(r"[a-z]{4,}", post_text.lower()):
                    _check_tokens.add(tok)
            if len(_check_tokens) >= 2:
                url_hits = sum(1 for tok in _check_tokens if tok in url_path)
                if url_hits == 0 and family in STRICT_IMAGE_FAMILIES:
                    logger.warning(
                        "VALIDATE_REJECT_NO_META_URL family=%s url=%r topic=%r — "
                        "no keyword match in URL for strict family",
                        family, image_ref[:80], (topic or "")[:40],
                    )
                    return False

    logger.debug(
        "VALIDATE_ACCEPT family=%s url=%r meta=%r",
        family, image_ref[:80], (image_meta or "")[:80],
    )
    return True
