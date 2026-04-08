"""Prompt builder — modular prompt construction for planner → writer pipeline.

Instead of a single monolithic prompt, this module builds separate, focused
prompt blocks that are composed differently for the planner and writer stages.

Key design principles:
  - Each block has a single responsibility
  - Blocks are short and focused
  - Voice block never mentions post subject
  - Channel context is soft, not dominant
  - Manual request always takes priority
"""
from __future__ import annotations

import re
import logging
from typing import Any

from generation_spec import (
    GenerationSpec,
    PlannerOutput,
    VOICE_MODES,
    OPENING_ARCHETYPES,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual prompt blocks (kept short and focused)
# ---------------------------------------------------------------------------

def _build_voice_block(spec: GenerationSpec) -> str:
    """Voice constraints from author role. Affects ONLY tone, not subject."""
    role = spec.author_role_type
    if not role:
        return ""

    vm = VOICE_MODES.get(role, {})
    if not vm:
        return ""

    parts = [f"ГОЛОС АВТОРА (только тон и подача, НЕ тема текста):"]
    parts.append(f"- Стиль: {vm.get('style', 'нейтральный')}")
    parts.append(f"- Тон: {vm.get('tone', 'спокойный')}")

    if vm.get("allow_first_person"):
        parts.append("- Допускается первое лицо («я», «мой опыт»), но НЕ обязательно")
    elif vm.get("person") == "1st_plural":
        parts.append("- Используй «мы», «наш». НЕ используй «я», «мой опыт»")
    else:
        parts.append("- НЕ используй первое лицо. Пиши от третьего лица")

    if spec.author_role_description:
        parts.append(f"- Кто автор: {spec.author_role_description}")
    if spec.author_activities:
        parts.append(f"- Деятельность: {spec.author_activities}")

    # Critical: voice does NOT determine subject
    parts.append("- ВАЖНО: роль автора задаёт ТОЛЬКО тон и подачу. Тема поста определяется отдельно.")

    return "\n".join(parts)


def _build_factual_safety_block(spec: GenerationSpec) -> str:
    """Factual safety constraints — universal, not niche-specific."""
    parts = ["ФАКТИЧЕСКАЯ БЕЗОПАСНОСТЬ:"]
    parts.append("- Если не уверен в факте — пиши осторожнее: «часто бывает», «по опыту», а НЕ «доказано» или «всегда»")
    parts.append("- ЗАПРЕЩЕНО: выдуманные проценты, статистика, исследования, кейсы клиентов")
    parts.append("- ЗАПРЕЩЕНО: точные технические/медицинские/юридические/финансовые диагнозы без явного основания")
    parts.append("- ЗАПРЕЩЕНО: «учёные доказали», «по статистике», «исследования показывают» без конкретного источника")
    parts.append("- Если тема незнакомая — пиши на уровне здравого смысла, без деталей, в которых не уверен")
    parts.append("- Лучше общее верное утверждение, чем конкретное выдуманное")

    if spec.author_forbidden_claims:
        parts.append(f"- НЕЛЬЗЯ приписывать автору: {spec.author_forbidden_claims}")

    # Anti-hallucination: commerce/location claims
    parts.append("- ЗАПРЕЩЕНО: «можно купить в ...», «продаётся в ...», «есть в Связном/DNS/МВидео» — если это НЕ указано во входных данных")
    parts.append("- ЗАПРЕЩЕНО: конкретные цены, магазины, локации, бренды — если их нет в исходных фактах или запросе")

    return "\n".join(parts)


def _build_anecdote_guard_block(spec: GenerationSpec) -> str:
    """Block fabricated personal/service anecdotes when not in input data."""
    # Check if input explicitly contains personal case signals
    input_text = f"{spec.source_prompt} {spec.primary_topic}".lower()
    has_personal_input = any(kw in input_text for kw in [
        "клиент", "обратил", "сервис", "мастерск", "из практик",
        "из опыт", "кейс", "случай из", "история из",
    ])
    if has_personal_input:
        return ""  # Input explicitly mentions personal cases — allow

    return (
        "ЗАПРЕТ НА ВЫДУМАННЫЕ ИСТОРИИ (ОБЯЗАТЕЛЬНО):\n"
        "- Во входных данных НЕТ упоминаний клиентов, сервиса, практики.\n"
        "- Поэтому НЕЛЬЗЯ писать: «клиент пришёл», «ко мне обратились», «в моём сервисе»,\n"
        "  «мы часто видим», «из практики», «недавно ко мне», «мой последний клиент».\n"
        "- Роль автора влияет ТОЛЬКО на тон, но НЕ даёт права выдумывать истории из практики.\n"
        "- Если хочешь дать пример — используй обезличенную ситуацию или общеизвестный факт."
    )


def _build_source_grounding_block(spec: GenerationSpec) -> str:
    """Source grounding for news mode — text must be built from source facts."""
    if spec.generation_mode != "news" or not spec.source_facts:
        return ""

    facts_text = "\n".join(f"  - {f}" for f in spec.source_facts[:10])
    parts = [
        "ПРИВЯЗКА К ИСТОЧНИКУ (ОБЯЗАТЕЛЬНО для новостного режима):",
        f"- Используй ТОЛЬКО эти факты из источника:\n{facts_text}",
        "- НЕ добавляй факты, которых нет в этом списке.",
        "- Главные сущности/субъекты в тексте ДОЛЖНЫ совпадать с фактами из источника.",
        "- Если не можешь написать по этим фактам — так и скажи, НЕ выдумывай.",
    ]

    if spec.forbidden_facts:
        forbidden_text = "\n".join(f"  - {f}" for f in spec.forbidden_facts[:5])
        parts.append(f"- НЕ упоминай (запрещённые факты):\n{forbidden_text}")

    return "\n".join(parts)


def _build_brevity_block(spec: GenerationSpec) -> str:
    """Length and density constraints."""
    return (
        f"КРАТКОСТЬ (ОБЯЗАТЕЛЬНО):\n"
        f"- Целевой объём body: {spec.target_length_words}-{spec.max_length_words} слов. Это жёсткая цель.\n"
        f"- Минимум: {spec.min_length_words} слов.\n"
        f"- Telegram — читают на ходу. Каждое предложение обязано добавлять новое.\n"
        f"- Если фразу можно удалить без потери смысла — удали.\n"
        f"- Максимум 2 абзаца основного текста.\n"
        f"- НЕТ вступительной воды, НЕТ разжёвывания, НЕТ повторения одной мысли."
    )


def _build_topic_priority_block(spec: GenerationSpec) -> str:
    """Explicit topic priority instructions based on generation mode."""
    if spec.generation_mode == "manual" and spec.source_prompt and spec.source_prompt.lower() != spec.channel_topic.lower():
        return (
            f"ПРИОРИТЕТ ТЕМЫ:\n"
            f"- Главная тема поста: «{spec.primary_topic}» (запрос пользователя) — {int(spec.request_priority * 100)}% текста\n"
            f"- Тема канала «{spec.channel_topic}» — только мягкий контекст ({int(spec.channel_priority * 100)}%), НЕ основная тема\n"
            f"- Если запрос пользователя отличается от темы канала — пиши про запрос пользователя\n"
            f"- НЕ подменяй тему поста темой канала. НЕ тащи текст обратно в тему канала."
        )
    return (
        f"ТЕМА ПОСТА:\n"
        f"- Основная тема: «{spec.primary_topic}»\n"
        f"- Тема канала: «{spec.channel_topic}»\n"
        f"- Пиши непосредственно о теме поста."
    )


def _build_must_not_force_block(spec: GenerationSpec) -> str:
    """Things that should NOT be forced into the text."""
    if not spec.must_not_force:
        return ""
    items = "\n".join(f"- {x}" for x in spec.must_not_force)
    return f"НЕ НАВЯЗЫВАЙ в тексте (даже если кажется уместным):\n{items}"


def _build_opener_dedup_block(spec: GenerationSpec) -> str:
    """Opener novelty instructions."""
    if not spec.forbidden_opener_types:
        return ""
    forbidden = ", ".join(spec.forbidden_opener_types)
    return (
        f"НАЧАЛО ТЕКСТА — РАЗНООБРАЗИЕ:\n"
        f"- Недавно использованные типы начала (ИЗБЕГАЙ): {forbidden}\n"
        f"- Выбери другой тип: один из [{', '.join(a for a in OPENING_ARCHETYPES if a not in spec.forbidden_opener_types)}]"
    )


# ---------------------------------------------------------------------------
# Planner prompt — builds structured plan, not final text
# ---------------------------------------------------------------------------

def build_planner_prompt(
    spec: GenerationSpec,
    *,
    strategy_mode: dict[str, str] | None = None,
    angle: dict[str, str] | None = None,
    recent_posts: list[str] | None = None,
    recent_plan: list[str] | None = None,
) -> str:
    """Build a planner prompt that produces a structured JSON plan.

    The planner does NOT write the final text — it produces a validated plan.
    """
    angle = angle or {}
    recent_posts = recent_posts or []
    recent_plan = recent_plan or []

    strategy_hint = ""
    if strategy_mode:
        strategy_hint = f"Стратегия: {strategy_mode.get('label', '')} — {strategy_mode.get('prompt_hint', '')}"

    recent_posts_block = ""
    if recent_posts:
        items = "\n".join(f"- {p[:100]}" for p in recent_posts[:5])
        recent_posts_block = f"\nНедавние посты (НЕ повторяй):\n{items}"

    recent_plan_block = ""
    if recent_plan:
        items = "\n".join(f"- {p[:100]}" for p in recent_plan[:5])
        recent_plan_block = f"\nНедавние темы из плана (НЕ повторяй):\n{items}"

    return f"""Ты — планировщик контента. Составь ПЛАН для одного короткого Telegram-поста.
НЕ пиши сам текст. Верни только JSON с планом.

{_build_topic_priority_block(spec)}

{_build_voice_block(spec)}

{_build_anecdote_guard_block(spec)}

{_build_source_grounding_block(spec)}

{strategy_hint}

Угол поста:
- anchor: {angle.get('opening', 'конкретное наблюдение')}
- focus: {angle.get('focus', 'практическая польза')}

{_build_opener_dedup_block(spec)}
{recent_posts_block}
{recent_plan_block}

{_build_factual_safety_block(spec)}

Верни ТОЛЬКО JSON:
{{
  "resolved_topic": "о чём конкретно этот пост (одно предложение)",
  "angle": "под каким углом подать тему",
  "opening_type": "один из: {', '.join(OPENING_ARCHETYPES)}",
  "outline": ["пункт 1 плана", "пункт 2 плана", "пункт 3 плана"],
  "claims": ["ключевое утверждение 1", "ключевое утверждение 2"],
  "confidence": "high / medium / low",
  "forbidden_directions": ["чего НЕ должно быть в тексте"],
  "voice_mode": "{spec.author_role_type or 'neutral'}"
}}""".strip()


# ---------------------------------------------------------------------------
# Writer prompt — writes final text from validated plan
# ---------------------------------------------------------------------------

def build_writer_prompt(
    spec: GenerationSpec,
    plan: PlannerOutput,
    *,
    today: str = "",
    family_guardrails: str = "",
    family_style: str = "",
    title_guardrails: str = "",
    recent_openings: str = "",
    blocked_phrases: str = "",
    extra_rules: str = "",
) -> str:
    """Build a writer prompt from a validated PlannerOutput.

    The writer receives a VALIDATED plan and produces the final text.
    This prompt is shorter and more focused than the old monolithic prompt.
    """
    topic_block = _build_topic_priority_block(spec)
    voice_block = _build_voice_block(spec)
    safety_block = _build_factual_safety_block(spec)
    brevity_block = _build_brevity_block(spec)
    must_not_block = _build_must_not_force_block(spec)
    anecdote_guard = _build_anecdote_guard_block(spec)
    source_grounding = _build_source_grounding_block(spec)

    plan_block = f"""ПЛАН ПОСТА (уже проверен, следуй ему):
- Тема: {plan.resolved_topic}
- Угол: {plan.angle}
- Тип начала: {plan.opening_type}
- Структура: {' → '.join(plan.outline) if plan.outline else 'свободная'}
- Уровень уверенности: {plan.confidence}"""

    if plan.forbidden_directions:
        plan_block += "\n- НЕ включать: " + ", ".join(plan.forbidden_directions)

    openings_block = ""
    if recent_openings:
        openings_block = f"\nНельзя начинать примерно так же:\n{recent_openings}"

    phrases_block = ""
    if blocked_phrases:
        phrases_block = f"\nНельзя повторять эти формулировки:\n{blocked_phrases}"

    return f"""Напиши один короткий сильный пост для Telegram по готовому плану.

Дата: {today}

{topic_block}

{plan_block}

{voice_block}

{anecdote_guard}

{source_grounding}

{safety_block}

{brevity_block}

{must_not_block}

{family_style}

- guardrails: {family_guardrails}
- title rules: {title_guardrails}
{openings_block}
{phrases_block}

Требования:
- title: 4-9 слов, конкретный и цепляющий
- body: {spec.target_length_words}-{spec.max_length_words} слов, 2 абзаца максимум
- Первое предложение — конкретный факт, наблюдение или ситуация. НЕ абстракция.
- ЗАПРЕЩЕНЫ: «В наше время», «Давайте разберёмся», «Важно понимать», «Не секрет» и подобные шаблоны
- Завершай конкретным вопросом по теме поста (не «Что думаете?»)
- ЗАПРЕТ на выдуманные @упоминания, каналы, ссылки, URL
- image_prompt: краткий английский промпт для фотореалистичного изображения по теме поста

{extra_rules}

Верни только JSON: {{title, body, cta, short, button_text, image_prompt}}""".strip()


# ---------------------------------------------------------------------------
# Rewrite prompt (enhanced with new dimensions)
# ---------------------------------------------------------------------------

def build_targeted_rewrite_prompt(
    title: str,
    body: str,
    cta: str,
    weak_dims: dict[str, int],
    spec: GenerationSpec,
) -> str | None:
    """Build a targeted rewrite prompt for near-miss texts.

    Enhanced with new dimensions: request_fit, claim_risk, opener_novelty, length_fit.
    """
    rewritable = {
        "hook", "naturalness", "role_fit", "specificity",
        "publish_ready", "topic_fit", "request_fit",
        "opener_novelty", "length_fit", "claim_risk",
    }

    fixable = {k: v for k, v in weak_dims.items() if k in rewritable and v <= 4}
    if not fixable:
        return None

    instructions: list[str] = []

    if "hook" in fixable:
        instructions.append(
            "ПЕРЕДЕЛАЙ НАЧАЛО: первое предложение — конкретный факт или ситуация. "
            "Никаких «В наше время...» или абстрактных рассуждений."
        )
    if "naturalness" in fixable:
        instructions.append(
            "УБЕРИ AI-ЗВУЧАНИЕ: замени канцелярит и шаблоны на живые формулировки."
        )
    if "request_fit" in fixable:
        instructions.append(
            f"ВЕРНИ К ТЕМЕ ЗАПРОСА: текст должен быть о «{spec.primary_topic}», "
            f"а не о «{spec.channel_topic}». Убери навязанную привязку к теме канала."
        )
    if "topic_fit" in fixable and "request_fit" not in fixable:
        instructions.append(
            f"ВЕРНИ К ТЕМЕ: перефокусируй текст на тему «{spec.primary_topic}»."
        )
    if "role_fit" in fixable:
        instructions.append("ГОЛОС: приведи тон в соответствие с ролью автора канала.")
    if "claim_risk" in fixable:
        instructions.append(
            "СМЯГЧИ УТВЕРЖДЕНИЯ: замени слишком уверенные/конкретные заявления "
            "на более осторожные формулировки. Убери выдуманную статистику и диагнозы."
        )
    if "opener_novelty" in fixable:
        instructions.append(
            "СМЕНИ ТИП НАЧАЛА: начало слишком похоже на недавние посты. "
            "Используй другой тип открытия."
        )
    if "length_fit" in fixable:
        instructions.append(
            f"СОКРАТИ: текст слишком длинный. Цель: {spec.target_length_words}-{spec.max_length_words} слов. "
            "Убери воду, повторы, лишние объяснения."
        )
    if "specificity" in fixable:
        instructions.append("ДОБАВЬ КОНКРЕТИКУ: замени водные фразы на факты и примеры.")
    if "publish_ready" in fixable:
        instructions.append("ПОДГОТОВЬ К ПУБЛИКАЦИИ: убери мета-комментарии и артефакты.")

    if not instructions:
        return None

    return f"""Перепиши этот пост, исправив ТОЛЬКО указанные проблемы.

Тема поста: {spec.primary_topic}
Тема канала: {spec.channel_topic}

ТЕКУЩИЙ ТЕКСТ:
Заголовок: {title}
Текст: {body}
CTA: {cta}

ЧТО ИСПРАВИТЬ:
{chr(10).join('- ' + i for i in instructions)}

ВАЖНО:
- Исправь ТОЛЬКО указанные проблемы
- Сохрани тему и основные мысли
- НЕ добавляй новых фактов
- НЕ делай текст длиннее
- Целевой объём body: {spec.target_length_words}-{spec.max_length_words} слов

Верни только JSON: {{title, body, cta, short, button_text, image_prompt}}""".strip()
