"""
visual_intent.py — Post-centric visual intent extraction for the image pipeline.

This module extracts a structured VisualIntent from post text (title + body),
performing:
  1. Main subject identification
  2. Word-sense disambiguation (WSD) via context
  3. Forbidden-meaning detection
  4. Visuality / imageability scoring
  5. Scene description for image search

The key principle: IMAGE = f(POST), not f(CHANNEL).
Channel context is used ONLY as a weak fallback when post text is too empty.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from topic_utils import detect_topic_family, TOPIC_FAMILY_TERMS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Visuality levels
# ---------------------------------------------------------------------------
VISUALITY_HIGH = "high"
VISUALITY_MEDIUM = "medium"
VISUALITY_LOW = "low"
VISUALITY_NONE = "none"


# ---------------------------------------------------------------------------
# VisualIntent — the structured output of intent extraction
# ---------------------------------------------------------------------------
@dataclass
class VisualIntent:
    """Compact visual brief extracted from post text."""
    main_subject: str = ""          # Primary visual object/concept (English)
    sense: str = ""                 # Disambiguated meaning context
    scene: str = ""                 # Expected visual scene description
    forbidden_meanings: list[str] = field(default_factory=list)
    visuality: str = VISUALITY_MEDIUM
    search_queries: list[str] = field(default_factory=list)
    post_family: str = "generic"    # Family detected from POST text (not channel)
    no_image_reason: str = ""       # If visuality is low/none, explains why
    source: str = "post"            # "post" or "fallback" (channel)


# ---------------------------------------------------------------------------
# Word-sense disambiguation rules
# ---------------------------------------------------------------------------
# Each entry: (stem, list_of_senses)
# Each sense: (sense_name, context_keywords, english_subject, forbidden_senses)

@dataclass
class WordSense:
    """One possible meaning of an ambiguous word."""
    name: str                       # e.g. "car", "industrial_machine"
    context_keywords: list[str]     # Words that indicate this sense
    english_subject: str            # What to search for
    forbidden_subjects: list[str]   # What NOT to search for when this sense is active


# Disambiguation dictionary: stem -> list of possible senses
# The first sense is the default if no context keywords match.
_WSD_RULES: dict[str, list[WordSense]] = {
    "машин": [
        WordSense(
            name="car",
            context_keywords=[
                "авто", "автомоб", "водител", "дорог", "руль", "двигател",
                "бензин", "дизель", "шин", "колес", "парков", "гараж",
                "трасс", "пробк", "тест-драйв", "каршеринг", "такси",
                "салон авто", "кузов", "коробк", "передач",
            ],
            english_subject="car automobile vehicle",
            forbidden_subjects=["industrial machine", "factory machine", "machinery equipment", "manufacturing machine"],
        ),
        WordSense(
            name="industrial_machine",
            context_keywords=[
                "производств", "промышлен", "завод", "фабрик", "станок",
                "оборудован", "конвейер", "цех", "агрегат", "мощност",
                "пресс", "токарн", "фрезер", "сварочн", "линия сборки",
            ],
            english_subject="industrial machine factory equipment manufacturing",
            forbidden_subjects=["car", "automobile", "vehicle", "sedan", "suv"],
        ),
    ],
    "ремень": [
        WordSense(
            name="timing_belt",
            context_keywords=[
                "грм", "двигател", "мотор", "распредвал", "коленвал",
                "замена ремн", "привод", "натяжител", "ролик", "зубчат",
                "engine", "timing",
            ],
            english_subject="car engine timing belt automotive",
            forbidden_subjects=["clothing belt", "fashion belt", "leather belt", "scooter belt"],
        ),
        WordSense(
            name="clothing_belt",
            context_keywords=[
                "одежд", "мод", "стиль", "кожан", "пряжк",
                "аксессуар", "гардероб", "fashion", "outfit",
            ],
            english_subject="fashion leather belt accessory",
            forbidden_subjects=["timing belt", "engine belt", "drive belt", "industrial belt"],
        ),
        WordSense(
            name="drive_belt",
            context_keywords=[
                "производств", "станок", "привод", "конвейер",
                "промышлен", "equipment", "industrial",
            ],
            english_subject="industrial drive belt conveyor",
            forbidden_subjects=["clothing belt", "car timing belt", "fashion belt"],
        ),
    ],
    "ремня": [
        WordSense(
            name="timing_belt",
            context_keywords=[
                "грм", "двигател", "мотор", "распредвал", "коленвал",
                "привод", "натяжител", "ролик", "зубчат",
                "engine", "timing",
            ],
            english_subject="car engine timing belt automotive",
            forbidden_subjects=["clothing belt", "fashion belt", "leather belt", "scooter belt"],
        ),
        WordSense(
            name="clothing_belt",
            context_keywords=[
                "одежд", "мод", "стиль", "кожан", "пряжк",
                "аксессуар", "гардероб", "fashion", "outfit",
            ],
            english_subject="fashion leather belt accessory",
            forbidden_subjects=["timing belt", "engine belt", "drive belt", "industrial belt"],
        ),
    ],
    "батаре": [
        WordSense(
            name="vehicle_battery",
            context_keywords=[
                "авто", "автомоб", "машин", "аккумулятор авто", "стартер",
                "зарядк авто", "электромоб", "tesla", "ev", "electric vehicle",
            ],
            english_subject="car battery vehicle accumulator automotive",
            forbidden_subjects=["aa battery", "battery icon", "phone battery", "power bank"],
        ),
        WordSense(
            name="generic_battery",
            context_keywords=[
                "телефон", "смартфон", "ноутбук", "гаджет", "зарядк",
                "power bank", "пауэрбанк", "литий", "литиев",
            ],
            english_subject="battery charging power bank device",
            forbidden_subjects=["car battery", "vehicle battery", "industrial battery"],
        ),
    ],
    "кран": [
        WordSense(
            name="faucet",
            context_keywords=[
                "вод", "кухн", "ванн", "сантехник", "смесител",
                "водопровод", "трубы", "раковин", "plumbing", "faucet",
            ],
            english_subject="faucet water tap plumbing",
            forbidden_subjects=["crane construction", "tower crane", "lifting crane"],
        ),
        WordSense(
            name="construction_crane",
            context_keywords=[
                "строительств", "стройк", "подъем", "грузоподъем",
                "башен", "монтаж", "construction", "crane lifting",
            ],
            english_subject="construction crane tower crane building site",
            forbidden_subjects=["faucet", "water tap", "kitchen tap", "bathroom faucet"],
        ),
    ],
    "мышь": [
        WordSense(
            name="computer_mouse",
            context_keywords=[
                "компьют", "ноутбук", "клавиатур", "монитор", "дисплей",
                "рабоч стол", "mouse pad", "геймерск", "беспроводн", "usb",
                "клик", "скролл",
            ],
            english_subject="computer mouse peripheral device",
            forbidden_subjects=["animal mouse", "rodent", "mouse trap"],
        ),
        WordSense(
            name="animal_mouse",
            context_keywords=[
                "животн", "грызун", "хвост", "сыр", "ловушк",
                "мышелов", "нор", "зверек", "rodent",
            ],
            english_subject="mouse rodent animal",
            forbidden_subjects=["computer mouse", "mouse device", "gaming mouse"],
        ),
    ],
    "лист": [
        WordSense(
            name="paper_sheet",
            context_keywords=[
                "бумаг", "документ", "печат", "принтер", "формат",
                "a4", "лист бумаг", "страниц", "тетрад",
            ],
            english_subject="paper sheet document",
            forbidden_subjects=["tree leaf", "plant leaf", "foliage"],
        ),
        WordSense(
            name="tree_leaf",
            context_keywords=[
                "дерев", "растен", "осен", "зелен", "природ",
                "лес", "парк", "ботаник", "сад", "garden",
            ],
            english_subject="tree leaf plant foliage nature",
            forbidden_subjects=["paper sheet", "document", "printer paper"],
        ),
    ],
    "плит": [
        WordSense(
            name="cooking_stove",
            context_keywords=[
                "кухн", "готовк", "варк", "кастрюл", "сковород",
                "газов", "электрическ", "индукцион", "духовк", "cooking",
            ],
            english_subject="kitchen stove cooktop cooking",
            forbidden_subjects=["concrete slab", "floor tile", "building slab"],
        ),
        WordSense(
            name="building_slab",
            context_keywords=[
                "строительств", "фундамент", "бетон", "перекрыт",
                "пол", "тротуар", "облицовк", "плитк",
            ],
            english_subject="concrete slab tile construction building",
            forbidden_subjects=["cooking stove", "kitchen stove", "cooktop"],
        ),
    ],
    "замок": [
        WordSense(
            name="lock",
            context_keywords=[
                "дверн", "ключ", "безопасност", "охран", "сейф",
                "замочн", "кодов", "электронн", "security", "lock",
            ],
            english_subject="door lock security key",
            forbidden_subjects=["castle", "medieval castle", "fortress"],
        ),
        WordSense(
            name="castle",
            context_keywords=[
                "средневеков", "истори", "крепост", "башн", "рыцар",
                "замок-дворец", "дворец", "туризм", "экскурс",
            ],
            english_subject="castle medieval fortress architecture",
            forbidden_subjects=["door lock", "padlock", "security lock"],
        ),
    ],
}


# ---------------------------------------------------------------------------
# Visuality assessment — how "imageable" is a given post?
# ---------------------------------------------------------------------------
_HIGH_VISUALITY_SIGNALS = [
    # Objects/items that are highly visual
    "фото", "photo", "картинк", "изображен", "image",
    "еда", "блюд", "рецепт", "food", "dish", "recipe",
    "машин", "автомоб", "car", "vehicle",
    "массаж", "massage",
    "маникюр", "nail", "стриж", "hair",
    "ремонт", "repair", "workshop",
    "кухн", "kitchen", "ресторан", "restaurant", "кафе",
    "интерьер", "interior", "дизайн",
    "животн", "animal", "питом", "pet",
    "пейзаж", "landscape", "природ", "nature",
    "архитектур", "architecture", "здани", "building",
    "спорт", "sport", "тренировк", "workout", "спортзал", "gym", "фитнес",
    "гаджет", "gadget", "устройств", "device", "смартфон", "smartphone",
    "ноутбук", "laptop", "телефон",
    "тест-драйв", "обзор авто", "седан", "кроссовер",
]

_LOW_VISUALITY_SIGNALS = [
    # Abstract/conceptual topics that are hard to visualize
    "контент-план", "content plan", "стратег", "strategy",
    "алгоритм", "algorithm", "формул", "formula",
    "абстракц", "abstraction", "концепц", "concept",
    "филосов", "philosophy", "теори", "theory",
    "мнение", "opinion", "размышлен", "reflection",
    "подборк", "список", "list", "цитат", "quote",
    "мем", "meme", "юмор", "humor", "анекдот", "joke",
    "новост", "news", "анонс", "announce",
    "kpi", "roi", "метрик", "metric",
    "конверс", "conversion", "воронк", "funnel",
    "аналитик", "analytics", "data",
]

_NONE_VISUALITY_SIGNALS = [
    # Posts that should almost never have images
    "текстов пост", "text only",
    "без картинк", "no image",
    "голосован", "опрос", "poll", "survey",
]


def _normalize(text: str) -> str:
    text = (text or "").strip().lower().replace("ё", "е")
    return re.sub(r"\s+", " ", text)


def _assess_visuality(text: str) -> str:
    """Assess how visual/imageable the post content is."""
    src = _normalize(text)
    if not src or len(src.strip()) < 10:
        return VISUALITY_NONE

    none_hits = sum(1 for s in _NONE_VISUALITY_SIGNALS if s in src)
    if none_hits >= 1:
        return VISUALITY_NONE

    high_hits = sum(1 for s in _HIGH_VISUALITY_SIGNALS if s in src)
    low_hits = sum(1 for s in _LOW_VISUALITY_SIGNALS if s in src)

    # Count total words as proxy for content richness
    word_count = len(src.split())

    if high_hits >= 3:
        return VISUALITY_HIGH
    if high_hits >= 1 and low_hits == 0:
        return VISUALITY_HIGH
    if high_hits >= 1 and low_hits >= 1:
        return VISUALITY_MEDIUM
    if low_hits >= 3:
        return VISUALITY_LOW
    if low_hits >= 1 and high_hits == 0:
        return VISUALITY_LOW

    # Default: medium if the post has decent content
    if word_count >= 15:
        return VISUALITY_MEDIUM
    if word_count >= 5:
        return VISUALITY_LOW
    return VISUALITY_NONE


# ---------------------------------------------------------------------------
# Subject extraction — get main English subject from Russian post text
# ---------------------------------------------------------------------------

# Quick translation table for common visual subjects (Russian stem -> English)
_SUBJECT_TRANSLATIONS: list[tuple[str, str]] = [
    # Food
    ("кофе", "coffee"), ("чай", "tea"), ("торт", "cake"), ("пицц", "pizza"),
    ("суш", "sushi"), ("бургер", "burger"), ("салат", "salad"), ("паст", "pasta"),
    ("рецепт", "recipe food dish"),
    ("завтрак", "breakfast"), ("обед", "lunch"), ("ужин", "dinner"),
    ("выпечк", "bakery pastry"), ("хлеб", "bread"),
    ("десерт", "dessert"), ("мороженое", "ice cream"),
    ("гриб", "mushrooms"), ("ягод", "berries"), ("мясо", "meat"),
    ("рыб", "fish seafood"), ("овощ", "vegetables"), ("фрукт", "fruits"),
    # Vehicles
    ("автомобил", "automobile car"), ("авто ", "car automotive"),
    ("грузовик", "truck"), ("мотоцикл", "motorcycle"),
    ("электромоб", "electric vehicle"),
    ("седан", "sedan car"), ("кроссовер", "crossover suv car"),
    ("тест-драйв", "test drive car"), ("внедорожник", "suv offroad"),
    # Tech
    ("ноутбук", "laptop"), ("компьют", "computer"), ("сервер", "server"),
    ("видеокарт", "graphics card gpu"), ("процессор", "processor cpu"),
    ("смартфон", "smartphone"), ("планшет", "tablet"),
    # Body/wellness
    ("массаж", "massage therapy"), ("спина", "back body"),
    ("шея", "neck shoulder"), ("лицо", "face facial"),
    # Beauty
    ("маникюр", "manicure nail art"), ("педикюр", "pedicure"),
    ("стриж", "haircut hairstyle"), ("окраш", "hair coloring"),
    # Home/building
    ("кухн", "kitchen"), ("ванн", "bathroom"), ("спальн", "bedroom"),
    ("гостин", "living room"), ("балкон", "balcony"),
    ("ремонт", "repair renovation"), ("строительств", "construction"),
    # Nature
    ("сад", "garden"), ("огород", "vegetable garden"),
    ("парк", "park"), ("лес", "forest"), ("озер", "lake"),
    # Business
    ("офис", "office"), ("склад", "warehouse"), ("магазин", "shop store"),
    ("ресторан", "restaurant dining"),
    # Professions
    ("стоматолог", "dentist"), ("психолог", "psychologist"),
    ("ветеринар", "veterinarian"), ("юрист", "lawyer"),
    ("бухгалтер", "accountant"), ("архитектор", "architect"),
    # Education
    ("школ", "school"), ("университет", "university"),
    ("библиотек", "library"), ("книг", "books"),
    # Other
    ("свадьб", "wedding"), ("детск", "children kids"),
    ("животн", "animals"), ("собак", "dog"), ("кошк", "cat"),
    ("путешеств", "travel"), ("туризм", "tourism"),
    ("недвижимост", "real estate"), ("ипотек", "mortgage"),
    ("спортзал", "gym fitness"), ("фитнес", "fitness workout"),
    ("телефон", "phone smartphone"), ("iphone", "iphone smartphone"),
    ("galaxy", "galaxy smartphone"), ("samsung", "samsung smartphone"),
]


def _extract_subject(text: str) -> str:
    """Extract the main visual subject from post text as English terms."""
    src = _normalize(text)
    subjects: list[str] = []

    for ru_stem, en_subject in _SUBJECT_TRANSLATIONS:
        if ru_stem in src:
            subjects.append(en_subject)
            if len(subjects) >= 3:
                break

    # Also grab any existing English words from the text
    # Pattern: start with letter, then letters/digits/hyphens (for compound words like "no-code")
    latin_words = re.findall(r"\b[a-z][a-z0-9-]{3,}\b", src)
    _stopwords = {
        "the", "and", "for", "with", "from", "that", "this", "into",
        "your", "about", "after", "before", "post", "telegram", "http",
        "https", "photo", "image",
    }
    for w in latin_words[:5]:
        if w not in _stopwords and w not in " ".join(subjects):
            subjects.append(w)
            if len(subjects) >= 4:
                break

    return " ".join(subjects[:4])


# ---------------------------------------------------------------------------
# Scene extraction — build a scene description from post context
# ---------------------------------------------------------------------------

_SCENE_PATTERNS: list[tuple[str, str]] = [
    # Location/setting indicators
    ("кухн", "kitchen cooking environment"),
    ("ресторан", "restaurant dining setting"),
    ("кафе", "cafe interior"),
    ("офис", "office workspace"),
    ("клиник", "clinic medical setting"),
    ("салон", "salon professional setting"),
    ("мастерск", "workshop craftsman"),
    ("завод", "factory industrial setting"),
    ("фабрик", "factory production"),
    ("склад", "warehouse storage"),
    ("дом", "home domestic"),
    ("квартир", "apartment interior"),
    ("студи", "studio creative space"),
    ("гараж", "garage workspace"),
    ("спортзал", "gym fitness"),
    ("парк", "park outdoor nature"),
    ("улиц", "street urban outdoor"),
    ("магазин", "retail shop"),
    ("школ", "school classroom"),
    ("библиотек", "library books"),
    # Process/action indicators
    ("готовк", "cooking preparation process"),
    ("ремонт", "repair maintenance work"),
    ("тренировк", "workout training exercise"),
    ("обучен", "learning teaching process"),
    ("лечен", "treatment medical process"),
    ("массаж сеанс", "massage therapy session"),
]


def _extract_scene(text: str) -> str:
    """Extract scene/setting description from post text."""
    src = _normalize(text)
    scenes: list[str] = []
    for ru_hint, en_scene in _SCENE_PATTERNS:
        if ru_hint in src:
            scenes.append(en_scene)
            if len(scenes) >= 2:
                break
    return " ".join(scenes[:2])


# ---------------------------------------------------------------------------
# Word-sense disambiguation
# ---------------------------------------------------------------------------

def _disambiguate(text: str, title: str = "") -> tuple[str, str, list[str]]:
    """Perform word-sense disambiguation on the post text.

    Uses word-boundary-aware stem matching to avoid false positives
    (e.g., "ремен" inside "современная").

    The title is given extra priority: if the ambiguous word appears in the
    title, we treat it as the main subject. If it only appears in the body,
    we require at least one context keyword to match before overriding
    subject extraction (avoids hijacking subjects for incidental mentions
    like "время работы от батареи" in a smartphone review).

    Returns: (resolved_subject, sense_name, forbidden_meanings)
    """
    src = _normalize(text)
    title_norm = _normalize(title) if title else ""

    for stem, senses in _WSD_RULES.items():
        # Use word-start boundary matching
        pattern = re.compile(
            r"(?:^|(?<=\s)|(?<=[^а-яё]))" + re.escape(stem),
            re.IGNORECASE,
        )
        if not pattern.search(src):
            continue

        in_title = bool(title_norm and pattern.search(title_norm))

        # Score each sense by counting context keyword matches
        best_sense: WordSense | None = None
        best_score = -1
        for sense in senses:
            score = sum(1 for kw in sense.context_keywords if kw in src)
            if score > best_score:
                best_score = score
                best_sense = sense

        if best_sense and best_score > 0:
            return best_sense.english_subject, best_sense.name, best_sense.forbidden_subjects

        # No context match: only default if the word is in the title
        # (i.e., it's likely the main subject, not an incidental mention)
        if in_title and senses:
            default = senses[0]
            return default.english_subject, f"{default.name}_default", default.forbidden_subjects

    return "", "", []


# ---------------------------------------------------------------------------
# Build search queries from visual intent
# ---------------------------------------------------------------------------

_LATIN_TOKEN_RE = re.compile(r"^[A-Za-z0-9][\w.+-]*$")
MAX_QUERY_LEN = 140


def _build_search_queries(intent: VisualIntent) -> list[str]:
    """Build ranked image search queries from the visual intent."""
    queries: list[str] = []

    # Priority 1: Main subject (most specific)
    if intent.main_subject:
        queries.append(intent.main_subject)
        # With scene context
        if intent.scene:
            queries.append(f"{intent.main_subject} {intent.scene} realistic photo")
        else:
            queries.append(f"{intent.main_subject} professional realistic photo")
        queries.append(f"{intent.main_subject} editorial close-up photo")

    # Priority 2: Scene-only queries if subject is weak
    if intent.scene and not intent.main_subject:
        queries.append(f"{intent.scene} realistic editorial photo")

    # Priority 3: Family-based queries as soft fallback
    family_terms = TOPIC_FAMILY_TERMS.get(intent.post_family, {}).get("en", [])[:3]
    if family_terms:
        family_str = " ".join(family_terms)
        if intent.main_subject:
            queries.append(f"{intent.main_subject} {family_terms[0]} editorial photo")
        queries.append(f"{family_str} realistic editorial photo")

    # Clean: only Latin tokens, dedupe, max length
    cleaned: list[str] = []
    seen: set[str] = set()
    for q in queries:
        words = q.split()
        latin_words = [w for w in words if _LATIN_TOKEN_RE.match(w)]
        q_clean = re.sub(r"\s+", " ", " ".join(latin_words)).strip()[:MAX_QUERY_LEN]
        if q_clean and q_clean not in seen:
            seen.add(q_clean)
            cleaned.append(q_clean)

    return cleaned[:8]


# ---------------------------------------------------------------------------
# Main extraction entry point
# ---------------------------------------------------------------------------

def extract_visual_intent(
    *,
    title: str = "",
    body: str = "",
    channel_topic: str = "",
) -> VisualIntent:
    """Extract visual intent from post text.

    Primary source: title + body (the actual post content).
    Channel topic is used ONLY as a weak fallback when post text is empty.

    Returns a VisualIntent with all fields populated.
    """
    # Combine post text sources — title gets higher priority by appearing first
    post_text = f"{title} {body}".strip()
    post_normalized = _normalize(post_text)

    intent = VisualIntent()

    # --- Step 1: Assess visuality ---
    intent.visuality = _assess_visuality(post_text)

    # If post is empty but we have channel_topic, don't bail early.
    # Let the fallback path on step 6 handle it.
    post_is_empty = len(post_normalized.strip()) < 5
    if intent.visuality == VISUALITY_NONE and not (post_is_empty and channel_topic):
        intent.no_image_reason = "low_visuality"
        intent.source = "post"
        logger.info(
            "VISUAL_INTENT visuality=none reason=low_visuality title=%r",
            (title or "")[:60],
        )
        return intent

    # --- Step 2: Detect post family (from POST text, not channel) ---
    intent.post_family = detect_topic_family(post_text)

    # --- Step 3: Word-sense disambiguation ---
    wsd_subject, sense_name, forbidden = _disambiguate(post_text, title=title)
    if wsd_subject:
        intent.main_subject = wsd_subject
        intent.sense = sense_name
        intent.forbidden_meanings = forbidden

    # --- Step 4: Subject extraction (if WSD didn't provide one) ---
    if not intent.main_subject:
        intent.main_subject = _extract_subject(post_text)

    # --- Step 5: Scene extraction ---
    intent.scene = _extract_scene(post_text)

    # --- Step 6: Fallback to channel topic if post is too empty ---
    if not intent.main_subject and not intent.scene:
        if channel_topic:
            intent.source = "fallback"
            fallback_subject = _extract_subject(channel_topic)
            if fallback_subject:
                intent.main_subject = fallback_subject
            # Also try WSD on channel topic
            if not intent.main_subject:
                wsd_subj, sense, forb = _disambiguate(channel_topic, title=channel_topic)
                if wsd_subj:
                    intent.main_subject = wsd_subj
                    intent.sense = sense
                    intent.forbidden_meanings = forb
            intent.scene = _extract_scene(channel_topic)
            intent.post_family = detect_topic_family(channel_topic)
            # Re-assess visuality with fallback content
            if intent.main_subject and intent.visuality == VISUALITY_NONE:
                intent.visuality = VISUALITY_LOW
                intent.no_image_reason = ""
        else:
            intent.no_image_reason = "no_visual_subject"
            intent.visuality = VISUALITY_LOW

    # --- Step 7: If still no subject, downgrade visuality ---
    if not intent.main_subject:
        if intent.visuality in (VISUALITY_HIGH, VISUALITY_MEDIUM):
            intent.visuality = VISUALITY_LOW
        intent.no_image_reason = "weak_subject"

    # --- Step 8: Build search queries ---
    intent.search_queries = _build_search_queries(intent)

    logger.info(
        "VISUAL_INTENT subject=%r sense=%r scene=%r forbidden=%s visuality=%s "
        "family=%s queries=%s source=%s no_image_reason=%s title=%r",
        intent.main_subject[:50] if intent.main_subject else "",
        intent.sense,
        intent.scene[:50] if intent.scene else "",
        intent.forbidden_meanings[:3] if intent.forbidden_meanings else [],
        intent.visuality,
        intent.post_family,
        [q[:40] for q in intent.search_queries[:3]],
        intent.source,
        intent.no_image_reason,
        (title or "")[:60],
    )

    return intent
