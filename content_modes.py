"""
content_modes.py — Explicit content mode detection and mode-aware generation rules.

Every generated post is classified into a content mode BEFORE generation.
The mode determines:
  - text generation rules (creativity, hedging, what is allowed/forbidden)
  - image prompt rules (what scene must be depicted)
  - anti-hallucination strictness level
  - fallback behavior when verified data is absent

Content modes are orthogonal to topic families:
  - topic family  = WHAT the channel is about (food, cars, tech, …)
  - content mode  = HOW this specific post should be generated (practical guide, factual news, opinion, …)

Both are used together: family provides visual and tonal context,
mode provides generation safety and structure rules.
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Content mode constants
# ---------------------------------------------------------------------------

MODE_HOWTO = "howto"                     # пошаговая инструкция / practical guide
MODE_LIFESTYLE = "lifestyle_utility"     # бытовой совет / lifestyle utility
MODE_FOOD = "food_cooking"               # food / cooking
MODE_AUTO = "auto_advice"               # auto / basic car advice
MODE_OPINION = "opinion"                # opinion / commentary
MODE_NEWS = "news_factual"              # news / factual update
MODE_CITY_TRANSPORT = "city_transport"  # city / transport / local topic
MODE_GENERIC = "generic"                # fallback when mode is unclear

ALL_MODES = (
    MODE_HOWTO, MODE_LIFESTYLE, MODE_FOOD, MODE_AUTO,
    MODE_OPINION, MODE_NEWS, MODE_CITY_TRANSPORT, MODE_GENERIC,
)

# ---------------------------------------------------------------------------
# Factual risk classification
# ---------------------------------------------------------------------------

# Modes where fabricating data is UNACCEPTABLE
FACTUAL_STRICT_MODES = frozenset({MODE_NEWS, MODE_CITY_TRANSPORT})

# Modes where caution is needed but light creativity is OK
FACTUAL_CAUTIOUS_MODES = frozenset({MODE_AUTO, MODE_HOWTO, MODE_LIFESTYLE})

# Modes where sensory/descriptive creativity is allowed
FACTUAL_PERMISSIVE_MODES = frozenset({MODE_FOOD, MODE_OPINION, MODE_GENERIC})


# ---------------------------------------------------------------------------
# Detection keywords per mode (Russian + English)
# ---------------------------------------------------------------------------

_MODE_KEYWORDS: dict[str, list[str]] = {
    MODE_HOWTO: [
        "как сделать", "как ", "пошагов", "инструкци", "чеклист", "гайд",
        "руководств", "how to", "guide", "step by step", "checklist",
        "совет", "лайфхак", "lifehack", "tip", "способ ",
        "шаг 1", "шаг 2", "шаг 3", "пункт 1",
    ],
    MODE_LIFESTYLE: [
        "бытов", "домашн", "уборк", "стирк", "хранен", "порядок",
        "организац дом", "ремонт дом", "дача", "сад ", "огород",
        "полезн привычк", "полезный совет", "экономи", "бюджет",
        "everyday", "home tip", "household", "routine",
        "утренн", "вечерн", "ritual",
    ],
    MODE_FOOD: [
        "рецепт", "блюд", "приготов", "готов", "выпечк", "десерт",
        "ингредиент", "кухн", "повар", "шеф", "гастроном",
        "recipe", "dish", "cooking", "baking", "ingredient",
        "завтрак", "обед", "ужин", "перекус",
    ],
    MODE_AUTO: [
        "автомобил", "машин", "двигател", "тормоз", "масл мотор",
        "бензин", "дизель", "колес", "шин", "подвеск", "коробк передач",
        "автосервис", "то автомобил", "техосмотр", "car ", "vehicle",
        "engine", "brake", "tire", "oil change",
        "самокат", "мотоцикл",
    ],
    MODE_OPINION: [
        "мнение", "считаю", "по-моему", "на мой взгляд", "позиция",
        "колонк", "комментар", "размышлен", "рефлекси", "дискусс",
        "opinion", "commentary", "editorial", "take", "perspective",
        "debate", "hot take",
    ],
    MODE_NEWS: [
        "новост", "анонс", "запуск", "релиз", "обновлен",
        "вышел", "выпуск", "запущен", "объявил", "сообщил",
        "news", "announce", "release", "launch", "update",
        "решени правительств", "закон принят", "законопроект",
        "центробанк", "минздрав", "роспотребнадзор", "фас ", "фнс",
    ],
    MODE_CITY_TRANSPORT: [
        "автобус", "маршрут", "остановк", "метро", "трамвай", "троллейбус",
        "электричк", " поезд ", "аэропорт", "вокзал", "пробк", "светофор",
        "парковк", "штраф", "гибдд", "гаи ", "дтп", "пдд",
        "транспорт", "такси", "каршеринг",
        "муниципальн", "администрац города", "мэри", "жкх",
        "аренд", "ипотек", "коммунальн", "тариф",
        "bus", "subway", "metro", "tram", "airport", "parking",
        "traffic", "taxi", "rideshare", "municipal",
        "цены на продукт", "цены на бензин", "средняя цена",
        "rent", "grocery price", "municipal service",
    ],
}


def detect_content_mode(
    title: str = "",
    body: str = "",
    channel_topic: str = "",
    generation_mode: str = "manual",
) -> str:
    """Detect the content mode for a post.

    Priority:
      1. If generation_mode == "news" → MODE_NEWS
      2. Title + body keyword matching (strongest signal)
      3. Channel topic as weak fallback

    Returns one of ALL_MODES.
    """
    # News generation mode always maps to factual news
    if generation_mode == "news":
        return MODE_NEWS

    combined = " ".join(filter(None, [title, body])).strip()
    if not combined:
        combined = channel_topic or ""

    q = combined.lower().replace("ё", "е")

    # Score each mode
    scores: dict[str, int] = {m: 0 for m in ALL_MODES}

    for mode, keywords in _MODE_KEYWORDS.items():
        for kw in keywords:
            if kw in q:
                scores[mode] += 1

    # Pick the mode with the highest score (if tied, prefer more specific modes)
    best_mode = MODE_GENERIC
    best_score = 0
    # Check in specificity order: most specific first
    specificity_order = [
        MODE_CITY_TRANSPORT, MODE_NEWS, MODE_FOOD, MODE_AUTO,
        MODE_HOWTO, MODE_LIFESTYLE, MODE_OPINION, MODE_GENERIC,
    ]
    for mode in specificity_order:
        if scores[mode] > best_score:
            best_score = scores[mode]
            best_mode = mode

    # Weak channel fallback: if nothing matched from title/body, try channel topic
    if best_score == 0 and channel_topic:
        ch_q = channel_topic.lower().replace("ё", "е")
        for mode in specificity_order:
            for kw in _MODE_KEYWORDS.get(mode, []):
                if kw in ch_q:
                    best_mode = mode
                    best_score = 1
                    break
            if best_score > 0:
                break

    logger.info(
        "CONTENT_MODE_DETECTED mode=%s score=%d title=%r",
        best_mode, best_score, (title or "")[:60],
    )
    return best_mode


def is_factual_strict(mode: str) -> bool:
    """Return True if the mode requires strict factual grounding."""
    return mode in FACTUAL_STRICT_MODES


def is_factual_cautious(mode: str) -> bool:
    """Return True if the mode needs moderate caution."""
    return mode in FACTUAL_CAUTIOUS_MODES


def is_factual_permissive(mode: str) -> bool:
    """Return True if the mode allows creative/sensory language."""
    return mode in FACTUAL_PERMISSIVE_MODES


# ---------------------------------------------------------------------------
# Mode-aware text generation rules
# ---------------------------------------------------------------------------

_MODE_TEXT_RULES: dict[str, dict] = {
    MODE_HOWTO: {
        "tone": "concrete, step-by-step, checklist, practical",
        "allowed": "concrete steps, checklist tone, practical examples, common-sense advice",
        "forbidden": "fake statistics, fake studies, fake percentages, invented research claims",
        "prompt_hint": (
            "Пиши как пошаговую инструкцию или чеклист. Конкретные действия, а не абстрактные рассуждения. "
            "ЗАПРЕЩЕНО: выдуманная статистика, проценты, исследования. "
            "Используй здравый смысл и практический опыт."
        ),
        "creativity": 0.7,
        "reject_threshold": 4,
    },
    MODE_LIFESTYLE: {
        "tone": "warm, practical, relatable, human",
        "allowed": "practical tips, relatable scenarios, common-sense advice",
        "forbidden": "fake statistics, fake authorities, medical claims, legal claims",
        "prompt_hint": (
            "Пиши как полезный бытовой совет. Тёплый, практичный тон. "
            "Без выдуманных цифр и ссылок на исследования. Здравый смысл."
        ),
        "creativity": 0.75,
        "reject_threshold": 4,
    },
    MODE_FOOD: {
        "tone": "sensory, appetizing, descriptive, warm",
        "allowed": "sensory descriptions (taste, smell, texture), practical cooking advice, ingredient details",
        "forbidden": "fake scientific claims about food, fake health statistics, invented nutrition research",
        "prompt_hint": (
            "Пиши чувственно и аппетитно. Описывай вкусы, запахи, текстуры. "
            "ЗАПРЕЩЕНО: выдуманные научные утверждения о еде, фальшивая нутрициология, "
            "выдуманные исследования о пользе/вреде продуктов."
        ),
        "creativity": 0.85,
        "reject_threshold": 5,
    },
    MODE_AUTO: {
        "tone": "practical, informed, driver-focused, no-hype",
        "allowed": "practical checks, maintenance tips, general car advice, common-sense recommendations",
        "forbidden": "invented service statistics, fake failure rates, fake brand comparisons with numbers",
        "prompt_hint": (
            "Пиши практично для водителя. Проверки, обслуживание, реальные сценарии. "
            "ЗАПРЕЩЕНО: выдуманная статистика поломок, фальшивые процентные показатели, "
            "выдуманные данные по сервисным центрам."
        ),
        "creativity": 0.7,
        "reject_threshold": 4,
    },
    MODE_OPINION: {
        "tone": "personal, thoughtful, honest, engaging",
        "allowed": "personal perspective, commentary, analysis, rhetorical questions",
        "forbidden": "presenting opinions as facts, fake data to support opinions",
        "prompt_hint": (
            "Пиши как авторский комментарий. Мнение, рефлексия, дискуссия. "
            "Если цитируешь факты — будь осторожен и используй формулировки «по ощущениям», «кажется», «многие считают». "
            "ЗАПРЕЩЕНО: подавать мнение как доказанный факт."
        ),
        "creativity": 0.85,
        "reject_threshold": 5,
    },
    MODE_NEWS: {
        "tone": "neutral, grounded, factual, cautious",
        "allowed": "only grounded claims from source data, general observations, cautious wording",
        "forbidden": (
            "fake numbers, fake percentages, fake studies, fake 'according to department/laboratory/statistics', "
            "fake local laws/taxes/transport rules, fake public decisions, invented precision"
        ),
        "prompt_hint": (
            "СТРОГИЙ ФАКТИЧЕСКИЙ РЕЖИМ.\n"
            "Пиши ТОЛЬКО на основе предоставленных фактов из источника.\n"
            "Если фактов нет — пиши как общее наблюдение или комментарий.\n"
            "КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО:\n"
            "- Выдуманные цифры, проценты, статистика\n"
            "- «По данным...», «Согласно исследованиям...», «Статистика показывает...» — без реального источника\n"
            "- Выдуманные ведомства, лаборатории, департаменты\n"
            "- Конкретные законы, налоги, правила — если их нет в фактах\n"
            "- Любая выдуманная точность\n"
            "Если нет подтверждённых данных — используй: «по наблюдениям», «часто бывает», «как правило»."
        ),
        "creativity": 0.5,
        "reject_threshold": 3,
    },
    MODE_CITY_TRANSPORT: {
        "tone": "practical, local, grounded, neutral",
        "allowed": "general transport guidance, common-sense city advice, cautious wording",
        "forbidden": (
            "fake municipal data, fake department references, invented bus routes/schedules, "
            "fake rent prices, fake grocery prices, fake local regulations, fake government decisions"
        ),
        "prompt_hint": (
            "СТРОГИЙ РЕЖИМ ДЛЯ ГОРОДСКОЙ/ТРАНСПОРТНОЙ ТЕМАТИКИ.\n"
            "Пиши ТОЛЬКО то, что можешь подтвердить.\n"
            "КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО:\n"
            "- Выдуманные номера маршрутов, расписания, остановки\n"
            "- Фальшивые цены на проезд, аренду, продукты\n"
            "- Выдуманные решения мэрии, администрации, департаментов\n"
            "- Конкретные муниципальные данные без подтверждения\n"
            "- «По данным департамента транспорта...» — если нет реального источника\n"
            "Если точных данных нет — используй общие формулировки: «как правило», «обычно», «в большинстве городов»."
        ),
        "creativity": 0.5,
        "reject_threshold": 3,
    },
    MODE_GENERIC: {
        "tone": "concrete, human, practical",
        "allowed": "general advice, common-sense observations, practical tips",
        "forbidden": "fake statistics, fake authorities, invented precision",
        "prompt_hint": (
            "Пиши конкретно и по-человечески. Без выдуманных цифр и исследований."
        ),
        "creativity": 0.75,
        "reject_threshold": 5,
    },
}


def get_mode_text_rules(mode: str) -> dict:
    """Return text generation rules for a content mode."""
    return _MODE_TEXT_RULES.get(mode, _MODE_TEXT_RULES[MODE_GENERIC])


def get_mode_prompt_hint(mode: str) -> str:
    """Return the prompt safety hint for a content mode."""
    rules = get_mode_text_rules(mode)
    return rules.get("prompt_hint", "")


def get_mode_creativity(mode: str) -> float:
    """Return the creativity level (temperature hint) for a content mode."""
    rules = get_mode_text_rules(mode)
    return rules.get("creativity", 0.75)


def get_mode_reject_threshold(mode: str) -> int:
    """Return the text validation reject threshold for a content mode."""
    rules = get_mode_text_rules(mode)
    return rules.get("reject_threshold", 5)


# ---------------------------------------------------------------------------
# Mode-aware image prompt rules
# ---------------------------------------------------------------------------

_MODE_IMAGE_RULES: dict[str, dict] = {
    MODE_HOWTO: {
        "scene_hint": "Show the real object or action described in the title. Hands-on, process, tools, result.",
        "forbidden_scenes": "abstract mood, generic office, random lifestyle, unrelated glamour",
        "style": "clear practical photography showing the process or result",
    },
    MODE_LIFESTYLE: {
        "scene_hint": "Show the practical home/daily scene from the title. Relatable, warm, real.",
        "forbidden_scenes": "corporate office, tech abstract, random glamour stock",
        "style": "warm authentic lifestyle photography, natural light, relatable scene",
    },
    MODE_FOOD: {
        "scene_hint": "Show the dish, ingredients, or kitchen process from the title. Appetizing, real food.",
        "forbidden_scenes": "corporate office, tech abstract, random person portrait, unrelated scene",
        "style": "appetizing food photography, natural lighting, editorial composition",
    },
    MODE_AUTO: {
        "scene_hint": "Show the car part, problem context, or maintenance scene from the title. Not generic luxury car glamour unless title fits.",
        "forbidden_scenes": "random luxury car glamour (unless title is about luxury cars), abstract mood, corporate office",
        "style": "practical automotive photography, real car context, detail or process shot",
    },
    MODE_OPINION: {
        "scene_hint": "Show a relevant scene that matches the opinion topic. Thoughtful, editorial.",
        "forbidden_scenes": "generic handshake, thumbs up, random corporate stock",
        "style": "thoughtful editorial photography, relevant to the topic being discussed",
    },
    MODE_NEWS: {
        "scene_hint": "Show a relevant neutral scene that matches the news topic. Documentary, factual, not mood.",
        "forbidden_scenes": "random mood image, abstract art, generic lifestyle, unrelated glamour",
        "style": "neutral documentary photography, relevant to the news topic, journalistic style",
    },
    MODE_CITY_TRANSPORT: {
        "scene_hint": "Show actual buses, stops, streets, traffic, city infrastructure. Not abstract office or workshop.",
        "forbidden_scenes": "abstract office, workshop, random lifestyle mood, unrelated tech image",
        "style": "urban documentary photography, real city transport/infrastructure, street-level view",
    },
    MODE_GENERIC: {
        "scene_hint": "Show a scene relevant to the post title.",
        "forbidden_scenes": "completely unrelated generic stock photo",
        "style": "professional editorial photography",
    },
}


def get_mode_image_rules(mode: str) -> dict:
    """Return image prompt rules for a content mode."""
    return _MODE_IMAGE_RULES.get(mode, _MODE_IMAGE_RULES[MODE_GENERIC])


# ---------------------------------------------------------------------------
# Semantic mismatch guard for images
# ---------------------------------------------------------------------------

# Core objects/scenes that MUST appear in image prompts for certain topics
_TOPIC_REQUIRED_VISUALS: list[tuple[list[str], list[str]]] = [
    # (topic keywords, required visual elements)
    (["автобус", "маршрут", "остановк", "bus", "route", "stop"], ["bus", "stop", "street", "transport", "road"]),
    (["метро", "subway", "metro", "подземк"], ["metro", "subway", "station", "underground", "train"]),
    (["аэропорт", "airport", "самолёт", "самолет", "рейс"], ["airport", "terminal", "airplane", "flight"]),
    (["аренд", "квартир", "жиль", "rent", "apartment"], ["apartment", "housing", "building", "interior", "home"]),
    (["продукт", "магазин", "grocery", "цены на еду"], ["grocery", "store", "supermarket", "shopping", "food"]),
    (["лекарств", "аптек", "medicine", "pharmacy"], ["pharmacy", "medicine", "medical", "health"]),
    (["рецепт", "блюд", "готовк", "recipe", "dish", "cook"], ["food", "dish", "kitchen", "cooking", "ingredient"]),
    (["машин", "двигател", "тормоз", "car ", "engine", "brake"], ["car", "engine", "automotive", "vehicle", "mechanic"]),
    (["парковк", "parking", "стоянк"], ["parking", "car", "lot", "street"]),
    (["пробк", "traffic", "светофор"], ["traffic", "road", "street", "cars", "intersection"]),
    (["такси", "taxi", "каршеринг"], ["taxi", "car", "ride", "street"]),
]


def check_image_prompt_relevance(
    title: str,
    image_prompt: str,
    content_mode: str = MODE_GENERIC,
) -> tuple[bool, str]:
    """Check if the image prompt is semantically relevant to the title/topic.

    Returns (is_relevant, reason).
    A lightweight guard that catches obvious mismatches.
    """
    if not title or not image_prompt:
        return True, "no_data_to_check"

    title_lower = title.lower().replace("ё", "е")
    prompt_lower = image_prompt.lower()

    for topic_keywords, required_visuals in _TOPIC_REQUIRED_VISUALS:
        # Check if the title matches this topic
        topic_match = any(kw in title_lower for kw in topic_keywords)
        if topic_match:
            # Check if at least one required visual element is in the prompt
            has_relevant = any(vis in prompt_lower for vis in required_visuals)
            if not has_relevant:
                return False, f"mismatch:title_about_{topic_keywords[0]}_but_prompt_lacks_relevant_visual"

    return True, "ok"


def fix_image_prompt_for_mode(
    image_prompt: str,
    title: str,
    content_mode: str,
) -> str:
    """Enhance or fix an image prompt to match the content mode.

    Adds mode-specific scene hints and strips irrelevant elements.
    """
    if not image_prompt:
        return image_prompt

    rules = get_mode_image_rules(content_mode)
    scene_hint = rules.get("scene_hint", "")

    # Check relevance
    is_relevant, reason = check_image_prompt_relevance(title, image_prompt, content_mode)

    if not is_relevant:
        # The prompt is semantically mismatched — rebuild it
        logger.info(
            "IMAGE_PROMPT_MODE_FIX reason=%s mode=%s title=%r",
            reason, content_mode, (title or "")[:60],
        )
        # Build a new prompt from the title + mode scene hint
        clean_title = re.sub(r"[^\w\s.,!?-]", "", title or "").strip()
        style = rules.get("style", "professional editorial photography")
        return f"A professional photograph depicting: {clean_title}. {scene_hint}. {style}."

    return image_prompt
