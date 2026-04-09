"""
visual_intent_v2.py — Post-centric visual intent extraction for image pipeline v3.

Core principle: IMAGE = f(POST), not f(CHANNEL).
Channel topic is used ONLY as a very weak fallback when post is too empty.

Extracts from title + body:
  - subject: primary visual object/concept (English)
  - sense/context: disambiguated meaning
  - scene: expected visual scene
  - forbidden_meanings: words indicating wrong sense
  - imageability: how visual the post is
  - query_terms: positive search terms
  - negative_terms: terms to avoid in search
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from topic_utils import detect_topic_family, TOPIC_FAMILY_TERMS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Imageability levels
# ---------------------------------------------------------------------------
IMAGEABILITY_HIGH = "high"
IMAGEABILITY_MEDIUM = "medium"
IMAGEABILITY_LOW = "low"
IMAGEABILITY_NONE = "none"


# ---------------------------------------------------------------------------
# VisualIntentV2 — structured output
# ---------------------------------------------------------------------------
@dataclass
class VisualIntentV2:
    """Structured visual brief extracted from post text (v2).

    All fields are populated by extract_visual_intent_v2().
    """
    subject: str = ""                       # Primary visual object (English)
    sense: str = ""                         # Disambiguated meaning context
    scene: str = ""                         # Expected visual scene
    forbidden_meanings: list[str] = field(default_factory=list)
    imageability: str = IMAGEABILITY_MEDIUM # How visual/imageable the post is
    query_terms: list[str] = field(default_factory=list)   # Positive search queries
    negative_terms: list[str] = field(default_factory=list) # Terms to avoid
    post_family: str = "generic"            # Family detected from POST text
    no_image_reason: str = ""               # Reason if image not recommended
    source: str = "post"                    # "post" or "channel_fallback"


# ---------------------------------------------------------------------------
# Word-sense disambiguation rules
# ---------------------------------------------------------------------------
@dataclass
class WordSense:
    """One possible meaning of an ambiguous word."""
    name: str
    context_keywords: list[str]
    english_subject: str
    forbidden_subjects: list[str]


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
            forbidden_subjects=["industrial machine", "factory machine", "machinery equipment"],
        ),
        WordSense(
            name="industrial_machine",
            context_keywords=[
                "производств", "промышлен", "завод", "фабрик", "станок",
                "оборудован", "конвейер", "цех", "агрегат", "мощност",
                "пресс", "токарн", "фрезер", "сварочн",
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
            ],
            english_subject="car engine timing belt automotive",
            forbidden_subjects=["clothing belt", "fashion belt", "leather belt"],
        ),
        WordSense(
            name="clothing_belt",
            context_keywords=[
                "одежд", "мод", "стиль", "кожан", "пряжк",
                "аксессуар", "гардероб", "fashion", "outfit",
            ],
            english_subject="fashion leather belt accessory",
            forbidden_subjects=["timing belt", "engine belt", "drive belt"],
        ),
    ],
    "ремня": [
        WordSense(
            name="timing_belt",
            context_keywords=[
                "грм", "двигател", "мотор", "распредвал", "коленвал",
                "привод", "натяжител", "ролик", "зубчат",
            ],
            english_subject="car engine timing belt automotive",
            forbidden_subjects=["clothing belt", "fashion belt", "leather belt"],
        ),
        WordSense(
            name="clothing_belt",
            context_keywords=[
                "одежд", "мод", "стиль", "кожан", "пряжк",
                "аксессуар", "гардероб", "fashion", "outfit",
            ],
            english_subject="fashion leather belt accessory",
            forbidden_subjects=["timing belt", "engine belt", "drive belt"],
        ),
    ],
    "батаре": [
        WordSense(
            name="vehicle_battery",
            context_keywords=[
                "авто", "автомоб", "машин", "аккумулятор авто", "стартер",
                "зарядк авто", "электромоб", "tesla",
            ],
            english_subject="car battery vehicle accumulator automotive",
            forbidden_subjects=["aa battery", "battery icon", "phone battery", "power bank"],
        ),
        WordSense(
            name="generic_battery",
            context_keywords=[
                "телефон", "смартфон", "ноутбук", "гаджет", "зарядк",
                "power bank", "литий",
            ],
            english_subject="battery charging power bank device",
            forbidden_subjects=["car battery", "vehicle battery"],
        ),
    ],
    "кран": [
        WordSense(
            name="faucet",
            context_keywords=[
                "вод", "кухн", "ванн", "сантехник", "смесител",
                "водопровод", "трубы", "раковин",
            ],
            english_subject="faucet water tap plumbing",
            forbidden_subjects=["crane construction", "tower crane", "lifting crane"],
        ),
        WordSense(
            name="construction_crane",
            context_keywords=[
                "строительств", "стройк", "подъем", "грузоподъем",
                "башен", "монтаж",
            ],
            english_subject="construction crane tower crane building site",
            forbidden_subjects=["faucet", "water tap", "kitchen tap"],
        ),
    ],
    "мыш": [
        WordSense(
            name="computer_mouse",
            context_keywords=[
                "компьют", "ноутбук", "клавиатур", "монитор", "дисплей",
                "mouse pad", "геймерск", "беспроводн", "usb", "клик",
            ],
            english_subject="computer mouse peripheral device",
            forbidden_subjects=["animal mouse", "rodent", "mouse trap"],
        ),
        WordSense(
            name="animal_mouse",
            context_keywords=[
                "животн", "грызун", "хвост", "сыр", "ловушк",
                "мышелов", "нор", "зверек",
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
                "a4", "лист бумаг", "страниц",
            ],
            english_subject="paper sheet document",
            forbidden_subjects=["tree leaf", "plant leaf", "foliage"],
        ),
        WordSense(
            name="tree_leaf",
            context_keywords=[
                "дерев", "растен", "осен", "зелен", "природ",
                "лес", "парк", "ботаник", "сад",
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
                "газов", "электрическ", "индукцион", "духовк",
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
                "замочн", "кодов", "электронн",
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
    "замк": [
        WordSense(
            name="lock",
            context_keywords=[
                "дверн", "ключ", "безопасност", "охран", "сейф",
                "замочн", "кодов", "электронн",
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
# Imageability assessment
# ---------------------------------------------------------------------------
_HIGH_IMAGEABILITY_SIGNALS = [
    "фото", "photo", "картинк", "изображен", "image",
    "еда", "блюд", "рецепт", "food", "dish", "recipe",
    "машин", "автомоб", "car", "vehicle",
    "массаж", "massage", "маникюр", "nail", "стриж", "hair",
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

_LOW_IMAGEABILITY_SIGNALS = [
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

_NONE_IMAGEABILITY_SIGNALS = [
    "текстов пост", "text only",
    "без картинк", "no image",
    "голосован", "опрос", "poll", "survey",
]


# ---------------------------------------------------------------------------
# Subject translation table (Russian stem → English)
# ---------------------------------------------------------------------------
_SUBJECT_TRANSLATIONS: list[tuple[str, str]] = [
    # Food
    ("карбонар", "carbonara pasta spaghetti"),
    ("кофе", "coffee"), ("чай", "tea"), ("торт", "cake"), ("пицц", "pizza"),
    ("суш", "sushi"), ("бургер", "burger"), ("салат", "salad"), ("паст", "pasta"),
    ("рецепт", "recipe food dish"),
    ("завтрак", "breakfast"), ("обед", "lunch"), ("ужин", "dinner"),
    ("выпечк", "bakery pastry"), ("хлеб", "bread"),
    ("десерт", "dessert"), ("мороженое", "ice cream"),
    ("гриб", "mushrooms"), ("ягод", "berries"), ("мясо", "meat"),
    ("рыб", "fish seafood"), ("овощ", "vegetables"), ("фрукт", "fruits"),
    # Transport / micro-mobility (BEFORE vehicles to catch specific terms)
    ("самокат", "scooter electric scooter kick scooter"),
    ("велосипед", "bicycle bike cycling"),
    ("тормоз", "brake braking"),
    ("колодк", "brake pad"),
    ("колес", "wheel tire"),
    ("бензин", "gasoline fuel petrol"),
    ("дизел", "diesel fuel"),
    ("двигател", "engine motor"),
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
    ("кож", "skincare skin face"),
    # Home/building
    ("кухн", "kitchen"), ("ванн", "bathroom"), ("спальн", "bedroom"),
    ("гостин", "living room"), ("балкон", "balcony"),
    ("ремонт", "repair renovation"), ("строительств", "construction"),
    # Nature
    ("сад", "garden"), ("огород", "vegetable garden"),
    ("парк", "park"),
    (" лес ", "forest"), (" лесу", "forest"), (" лесн", "forest"),
    (" леса", "forest"), (" лесом", "forest"), (" лесе", "forest"),
    ("озер", "lake"),
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
    # Sport
    ("спорт", "sport athletics"), ("футбол", "football soccer"),
    ("баскетбол", "basketball"), ("теннис", "tennis"),
    ("хоккей", "hockey"), ("бокс", "boxing"),
    ("чемпионат", "championship competition sport"),
    # Other
    ("свадьб", "wedding ceremony"), ("детск", "children kids"),
    ("ребен", "children kids"), ("дет", "children kids"),
    ("животн", "animals"), ("собак", "dog"), ("кошк", "cat"),
    ("путешеств", "travel"), ("туризм", "tourism"),
    ("недвижимост", "real estate"), ("ипотек", "mortgage"),
    ("спортзал", "gym fitness"), ("фитнес", "fitness workout"),
    ("телефон", "phone smartphone"), ("iphone", "iphone smartphone"),
    ("galaxy", "galaxy smartphone"), ("samsung", "samsung smartphone"),
    ("учител", "teacher education"), ("преподав", "teacher lecturer"),
    ("педагог", "teacher educator"),
]


# ---------------------------------------------------------------------------
# Scene extraction patterns
# ---------------------------------------------------------------------------
_SCENE_PATTERNS: list[tuple[str, str]] = [
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
    ("готовк", "cooking preparation process"),
    ("ремонт", "repair maintenance work"),
    ("тренировк", "workout training exercise"),
    ("обучен", "learning teaching process"),
    ("лечен", "treatment medical process"),
    ("массаж сеанс", "massage therapy session"),
]


# ---------------------------------------------------------------------------
# Negative terms by family (terms to AVOID in image search)
# ---------------------------------------------------------------------------
_FAMILY_NEGATIVE_TERMS: dict[str, list[str]] = {
    "food": ["tech", "code", "circuit", "corporate", "finance", "abstract"],
    "health": ["junk food", "gaming", "circuit board", "finance"],
    "beauty": ["circuit", "server", "gaming", "finance", "code"],
    "cars": ["food", "cooking", "recipe", "beauty", "makeup"],
    "tech": ["food", "cooking", "massage", "beauty salon"],
    "massage": ["tech", "code", "car", "finance", "gaming"],
    "education": ["gaming", "nightclub", "food photography"],
    "finance": ["food", "cooking", "gaming", "massage"],
    "marketing": ["food", "medical", "gaming", "construction"],
    "local_business": ["abstract", "gaming", "corporate stock"],
    "lifestyle": ["code", "circuit", "server rack", "medical"],
    "generic": ["abstract corporate", "random stock", "generic background"],
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _normalize(text: str) -> str:
    text = (text or "").strip().lower().replace("ё", "е")
    return re.sub(r"\s+", " ", text)


def _assess_imageability(text: str) -> str:
    """Assess how visual/imageable the post content is."""
    src = _normalize(text)
    if not src or len(src.strip()) < 10:
        return IMAGEABILITY_NONE

    none_hits = sum(1 for s in _NONE_IMAGEABILITY_SIGNALS if s in src)
    if none_hits >= 1:
        return IMAGEABILITY_NONE

    high_hits = sum(1 for s in _HIGH_IMAGEABILITY_SIGNALS if s in src)
    low_hits = sum(1 for s in _LOW_IMAGEABILITY_SIGNALS if s in src)
    word_count = len(src.split())

    if high_hits >= 3:
        return IMAGEABILITY_HIGH
    if high_hits >= 1 and low_hits == 0:
        return IMAGEABILITY_HIGH
    if high_hits >= 1 and low_hits >= 1:
        return IMAGEABILITY_MEDIUM
    if low_hits >= 3:
        return IMAGEABILITY_LOW
    if low_hits >= 1 and high_hits == 0:
        return IMAGEABILITY_LOW
    if word_count >= 15:
        return IMAGEABILITY_MEDIUM
    if word_count >= 5:
        return IMAGEABILITY_LOW
    return IMAGEABILITY_NONE


def _extract_subject(text: str) -> str:
    """Extract main visual subject from post text as English terms."""
    src = _normalize(text)
    subjects: list[str] = []

    for ru_stem, en_subject in _SUBJECT_TRANSLATIONS:
        if ru_stem in src:
            subjects.append(en_subject)
            if len(subjects) >= 3:
                break

    # Grab existing English words from text
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


def _disambiguate(text: str, title: str = "") -> tuple[str, str, list[str]]:
    """Word-sense disambiguation using context keywords.

    Title gets priority: if ambiguous word is in title, treat as main subject.

    Returns: (resolved_subject, sense_name, forbidden_meanings)
    """
    src = _normalize(text)
    title_norm = _normalize(title) if title else ""

    for stem, senses in _WSD_RULES.items():
        pattern = re.compile(
            r"(?:^|(?<=\s)|(?<=[^а-яё]))" + re.escape(stem),
            re.IGNORECASE,
        )
        if not pattern.search(src):
            continue

        in_title = bool(title_norm and pattern.search(title_norm))

        best_sense: WordSense | None = None
        best_score = -1
        for sense in senses:
            score = sum(1 for kw in sense.context_keywords if kw in src)
            if score > best_score:
                best_score = score
                best_sense = sense

        if best_sense and best_score > 0:
            return best_sense.english_subject, best_sense.name, best_sense.forbidden_subjects

        if in_title and senses:
            default = senses[0]
            return default.english_subject, f"{default.name}_default", default.forbidden_subjects

    return "", "", []


_LATIN_TOKEN_RE = re.compile(r"^[A-Za-z0-9][\w.+-]*$")
MAX_QUERY_LEN = 140


def _build_query_terms(intent: VisualIntentV2) -> list[str]:
    """Build ranked image search query terms from intent."""
    queries: list[str] = []

    if intent.subject:
        queries.append(intent.subject)
        if intent.scene:
            queries.append(f"{intent.subject} {intent.scene} realistic photo")
        else:
            queries.append(f"{intent.subject} professional realistic photo")
        queries.append(f"{intent.subject} editorial close-up photo")

    if intent.scene and not intent.subject:
        queries.append(f"{intent.scene} realistic editorial photo")

    family_terms = TOPIC_FAMILY_TERMS.get(intent.post_family, {}).get("en", [])[:3]
    if family_terms:
        family_str = " ".join(family_terms)
        if intent.subject:
            queries.append(f"{intent.subject} {family_terms[0]} editorial photo")
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


def _build_negative_terms(intent: VisualIntentV2) -> list[str]:
    """Build negative terms (what to avoid) from intent."""
    negatives: list[str] = list(intent.forbidden_meanings)
    family_negatives = _FAMILY_NEGATIVE_TERMS.get(intent.post_family, [])
    negatives.extend(family_negatives)
    return negatives


# ---------------------------------------------------------------------------
# Main extraction entry point
# ---------------------------------------------------------------------------
def extract_visual_intent_v2(
    *,
    title: str = "",
    body: str = "",
    channel_topic: str = "",
) -> VisualIntentV2:
    """Extract visual intent from post text (v2).

    Primary: title + body. Channel topic = very weak fallback only.
    """
    post_text = f"{title} {body}".strip()
    post_normalized = _normalize(post_text)

    intent = VisualIntentV2()

    # Step 1: Imageability assessment
    intent.imageability = _assess_imageability(post_text)

    post_is_empty = len(post_normalized.strip()) < 5
    if intent.imageability == IMAGEABILITY_NONE and not (post_is_empty and channel_topic):
        intent.no_image_reason = "low_imageability"
        intent.source = "post"
        return intent

    # Step 2: Post family detection (from POST text, not channel)
    intent.post_family = detect_topic_family(post_text)

    # Step 3: Word-sense disambiguation
    wsd_subject, sense_name, forbidden = _disambiguate(post_text, title=title)
    if wsd_subject:
        intent.subject = wsd_subject
        intent.sense = sense_name
        intent.forbidden_meanings = forbidden

    # Step 4: Subject extraction (if WSD didn't resolve)
    if not intent.subject:
        intent.subject = _extract_subject(post_text)

    # Step 5: Scene extraction
    intent.scene = _extract_scene(post_text)

    # Step 6: Fallback to channel topic only if post too empty
    if not intent.subject and not intent.scene:
        if channel_topic:
            intent.source = "channel_fallback"
            fallback_subject = _extract_subject(channel_topic)
            if fallback_subject:
                intent.subject = fallback_subject
            if not intent.subject:
                wsd_subj, sense, forb = _disambiguate(channel_topic, title=channel_topic)
                if wsd_subj:
                    intent.subject = wsd_subj
                    intent.sense = sense
                    intent.forbidden_meanings = forb
            intent.scene = _extract_scene(channel_topic)
            intent.post_family = detect_topic_family(channel_topic)
            if intent.subject and intent.imageability == IMAGEABILITY_NONE:
                intent.imageability = IMAGEABILITY_LOW
                intent.no_image_reason = ""
        else:
            intent.no_image_reason = "no_visual_subject"
            intent.imageability = IMAGEABILITY_LOW

    # Step 7: Downgrade if still no subject
    if not intent.subject:
        if intent.imageability in (IMAGEABILITY_HIGH, IMAGEABILITY_MEDIUM):
            intent.imageability = IMAGEABILITY_LOW
        intent.no_image_reason = "weak_subject"

    # Step 8: Build query and negative terms
    intent.query_terms = _build_query_terms(intent)
    intent.negative_terms = _build_negative_terms(intent)

    logger.info(
        "VISUAL_INTENT_V2 subject=%r sense=%r scene=%r forbidden=%s "
        "imageability=%s family=%s queries=%s negatives=%s source=%s title=%r",
        intent.subject[:50] if intent.subject else "",
        intent.sense,
        intent.scene[:50] if intent.scene else "",
        intent.forbidden_meanings[:3],
        intent.imageability,
        intent.post_family,
        [q[:40] for q in intent.query_terms[:3]],
        intent.negative_terms[:3],
        intent.source,
        (title or "")[:60],
    )

    return intent
