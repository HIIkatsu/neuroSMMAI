"""Generation Spec — structured intermediate representation for text generation.

Separates four previously conflated concerns:
  1. channel_topic  — what the channel is about (soft context)
  2. author_role    — voice/tone constraints only, NOT subject matter
  3. post_topic     — the actual topic of THIS specific post
  4. confidence     — factual risk / fabrication risk level

The GenerationSpec is built BEFORE any prompt is constructed and serves as
the single source of truth for the planner → writer pipeline.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Voice mode constants — role affects ONLY voice, not content subject
# ---------------------------------------------------------------------------
VOICE_MODES = {
    "media": {
        "label": "редакционный",
        "person": "3rd",
        "allow_first_person": False,
        "tone": "нейтральный информационный",
        "style": "как редакция / новостная лента",
    },
    "expert": {
        "label": "экспертный",
        "person": "1st_optional",
        "allow_first_person": True,
        "tone": "уверенный профессиональный",
        "style": "как практикующий специалист",
    },
    "master": {
        "label": "мастер-практик",
        "person": "1st_optional",
        "allow_first_person": True,
        "tone": "спокойный уверенный",
        "style": "как опытный мастер-практик",
    },
    "business_owner": {
        "label": "бизнес / команда",
        "person": "1st_plural",
        "allow_first_person": False,
        "tone": "деловой командный",
        "style": "от лица бизнеса/команды",
    },
    "brand": {
        "label": "бренд",
        "person": "1st_plural",
        "allow_first_person": False,
        "tone": "уверенный заботливый",
        "style": "от лица бренда",
    },
    "blogger": {
        "label": "блогер",
        "person": "1st_optional",
        "allow_first_person": True,
        "tone": "живой личный",
        "style": "как живой автор с личным голосом",
    },
    "educator": {
        "label": "преподаватель",
        "person": "1st_optional",
        "allow_first_person": True,
        "tone": "ясный структурированный",
        "style": "как опытный наставник",
    },
}


# ---------------------------------------------------------------------------
# Opening archetypes for deduplication
# ---------------------------------------------------------------------------
OPENING_ARCHETYPES = [
    "observation",       # конкретное наблюдение или факт
    "mistake",           # типичная ошибка
    "myth_busting",      # развенчание мифа
    "checklist",         # пошаговый чеклист
    "practical_advice",  # прямой совет
    "contrast",          # сравнение / противопоставление
    "trend",             # тренд или изменение
    "mini_case",         # короткий кейс / пример
    "question",          # вопрос читателю
    "warning",           # предупреждение / что избегать
    "statistic",         # цифра / статистика
    "scenario",          # узнаваемая ситуация
]


# Markers that help classify recent openers into archetypes
_ARCHETYPE_MARKERS: dict[str, list[str]] = {
    "observation": ["замечаю", "заметил", "обратил внимание", "вижу", "наблюда"],
    "mistake": ["ошибк", "ошибочно", "неправильно", "не стоит", "распространённая"],
    "myth_busting": ["миф", "заблужден", "на самом деле", "считается что", "принято думать"],
    "checklist": ["шаг", "пункт", "чеклист", "список", "порядок"],
    "practical_advice": ["совет", "рекомендаци", "попробуй", "сделай", "начни с"],
    "contrast": ["в отличие", "но", "однако", "зато", "сравни"],
    "trend": ["тренд", "всё чаще", "в последнее время", "набирает", "меняется"],
    "mini_case": ["клиент", "пришёл", "обратился", "случай", "история", "недавно", "вчера"],
    "question": ["?", "знаете ли", "задумывались", "почему", "как часто"],
    "warning": ["осторожн", "опасн", "избегай", "не допускай", "внимание"],
    "statistic": ["%", "процент", "цифр", "данн", "статистик"],
    "scenario": ["представьте", "утро", "вечер", "ситуация", "когда вы"],
}


@dataclass
class GenerationSpec:
    """Structured intermediate representation for one post generation.

    Built before any prompt and validated before the writer stage.
    """

    # --- Generation context ---
    generation_mode: str = "manual"     # "manual" | "autopost" | "news"

    # --- Topic separation ---
    primary_topic: str = ""             # resolved topic for THIS post
    source_prompt: str = ""             # raw user request (manual mode)
    channel_topic: str = ""             # channel's general topic (soft context)
    channel_family: str = "generic"     # detected topic family

    # --- Priority control ---
    channel_priority: float = 0.15      # how much channel context influences (0.0-1.0)
    request_priority: float = 0.85      # how much user request dominates (0.0-1.0)

    # --- Voice (role affects ONLY voice, not subject) ---
    author_role_type: str = ""
    author_role_description: str = ""
    author_activities: str = ""
    author_forbidden_claims: str = ""
    voice_mode: dict[str, Any] = field(default_factory=dict)

    # --- Factual safety ---
    factual_mode: str = "cautious"      # "cautious" | "confident" | "hedged"

    # --- Target constraints ---
    target_length_words: int = 70       # target body length in words (reduced from 90)
    max_length_words: int = 100         # hard max (reduced from 130)
    min_length_words: int = 40          # minimum viable length

    # --- Content control ---
    must_include: list[str] = field(default_factory=list)
    must_not_force: list[str] = field(default_factory=list)  # things NOT to force into text

    # --- Style ---
    channel_style: str = ""
    channel_audience: str = ""

    # --- Opener tracking ---
    recent_opener_types: list[str] = field(default_factory=list)
    forbidden_opener_types: list[str] = field(default_factory=list)

    # --- Strategy ---
    strategy_mode: str = ""
    strategy_label: str = ""
    strategy_hint: str = ""


@dataclass
class PlannerOutput:
    """Structured plan produced by the planner stage, validated before writing."""
    resolved_topic: str = ""
    angle: str = ""
    opening_type: str = ""              # one of OPENING_ARCHETYPES
    outline: list[str] = field(default_factory=list)
    claims: list[str] = field(default_factory=list)
    confidence: str = "medium"          # "high" | "medium" | "low"
    forbidden_directions: list[str] = field(default_factory=list)
    voice_mode: str = ""                # e.g. "expert", "media"
    target_words: int = 70


# ---------------------------------------------------------------------------
# Build GenerationSpec from settings
# ---------------------------------------------------------------------------

def build_generation_spec(
    *,
    channel_topic: str = "",
    requested: str = "",
    generation_mode: str = "manual",
    channel_family: str = "generic",
    owner_settings: dict[str, Any] | None = None,
    recent_opener_types: list[str] | None = None,
) -> GenerationSpec:
    """Build a GenerationSpec from channel settings and user request.

    Key logic:
    - For MANUAL mode: user request is primary_topic, channel_topic is soft context
    - For AUTOPOST mode: channel_topic dominates, request is secondary
    - Role ONLY affects voice constraints, never the subject matter
    """
    settings = owner_settings or {}
    ct = (channel_topic or "").strip()
    rq = (requested or "").strip()

    # --- Resolve primary topic ---
    if generation_mode == "manual" and rq and rq.lower() != ct.lower():
        primary_topic = rq
        channel_priority = 0.15
        request_priority = 0.85
    elif generation_mode == "autopost":
        primary_topic = rq if rq and rq.lower() != ct.lower() else ct
        channel_priority = 0.5 if primary_topic == ct else 0.3
        request_priority = 1.0 - channel_priority
    else:
        primary_topic = rq or ct
        channel_priority = 0.3
        request_priority = 0.7

    # --- Resolve voice from role (voice ONLY, not subject) ---
    role_type = str(settings.get("author_role_type") or "").strip().lower()
    voice_mode = VOICE_MODES.get(role_type, {})

    # --- Length targets (reduced 30-40%) ---
    if generation_mode == "news":
        target_words = 50
        max_words = 75
    elif generation_mode == "autopost":
        target_words = 60
        max_words = 90
    else:  # manual
        target_words = 70
        max_words = 100

    # --- Factual mode ---
    factual_mode = "cautious"  # default safe

    # --- Opener dedup ---
    recent_openers = recent_opener_types or []
    # Forbid archetypes used in last 3 posts
    from collections import Counter
    archetype_counts = Counter(recent_openers[-6:])
    forbidden_openers = [a for a, c in archetype_counts.items() if c >= 2]

    # --- Must not force ---
    must_not_force = []
    if generation_mode == "manual" and rq and rq.lower() != ct.lower():
        # When manual request differs from channel topic, don't force channel elements
        must_not_force.append(f"тема канала «{ct}» как основной предмет текста")
        must_not_force.append("клиентские кейсы и истории из практики автора")

    return GenerationSpec(
        generation_mode=generation_mode,
        primary_topic=primary_topic,
        source_prompt=rq,
        channel_topic=ct,
        channel_family=channel_family,
        channel_priority=channel_priority,
        request_priority=request_priority,
        author_role_type=role_type,
        author_role_description=str(settings.get("author_role_description") or ""),
        author_activities=str(settings.get("author_activities") or ""),
        author_forbidden_claims=str(settings.get("author_forbidden_claims") or ""),
        voice_mode=voice_mode,
        factual_mode=factual_mode,
        target_length_words=target_words,
        max_length_words=max_words,
        min_length_words=40,
        must_not_force=must_not_force,
        channel_style=str(settings.get("channel_style") or ""),
        channel_audience=str(settings.get("channel_audience") or ""),
        recent_opener_types=recent_openers,
        forbidden_opener_types=forbidden_openers,
    )


# ---------------------------------------------------------------------------
# Planner validation
# ---------------------------------------------------------------------------

def validate_planner_output(plan: PlannerOutput, spec: GenerationSpec) -> list[str]:
    """Validate planner output against GenerationSpec.

    Returns list of validation errors. Empty list = plan is valid.
    """
    errors: list[str] = []

    if not plan.resolved_topic:
        errors.append("planner_validation: resolved_topic is empty")
        return errors

    resolved_lower = plan.resolved_topic.lower()
    source_lower = (spec.source_prompt or spec.primary_topic).lower()
    channel_lower = spec.channel_topic.lower()

    # 1. For manual mode: resolved_topic must match user request, not channel topic
    if spec.generation_mode == "manual" and spec.source_prompt:
        source_words = set(re.findall(r"[а-яёa-z]{3,}", source_lower))
        resolved_words = set(re.findall(r"[а-яёa-z]{3,}", resolved_lower))
        channel_words = set(re.findall(r"[а-яёa-z]{3,}", channel_lower))

        if source_words and resolved_words:
            source_overlap = len(source_words & resolved_words) / max(len(source_words), 1)
            channel_overlap = len(channel_words & resolved_words) / max(len(channel_words), 1) if channel_words else 0

            # If resolved topic is closer to channel than to user request = topic hijack
            if source_overlap < 0.2 and channel_overlap > 0.5:
                errors.append(
                    f"planner_validation: resolved_topic hijacked by channel topic "
                    f"(request overlap={source_overlap:.0%}, channel overlap={channel_overlap:.0%})"
                )

    # 2. Check if role turned into post subject
    role_desc_lower = spec.author_role_description.lower()
    if role_desc_lower and spec.generation_mode == "manual" and spec.source_prompt:
        role_words = set(re.findall(r"[а-яёa-z]{4,}", role_desc_lower))
        if role_words and resolved_words:
            role_overlap = len(role_words & resolved_words) / max(len(role_words), 1)
            source_overlap_for_role = len(source_words & resolved_words) / max(len(source_words), 1) if source_words else 0
            if role_overlap > 0.6 and source_overlap_for_role < 0.3:
                errors.append(
                    "planner_validation: resolved_topic looks like author role description, not user request"
                )

    # 3. Check forbidden opener types
    if plan.opening_type in spec.forbidden_opener_types:
        errors.append(
            f"planner_validation: opening_type '{plan.opening_type}' was used recently, pick another"
        )

    # 4. Check overconfident claims
    high_risk_patterns = [
        r"\d{2,3}\s*%",                    # specific percentages
        r"доказано",                         # "proven"
        r"гарантирован",                     # "guaranteed"
        r"всегда\s+(?:работает|помогает)",   # "always works"
        r"никогда\s+не",                     # "never"
        r"единственный\s+способ",            # "the only way"
        r"точный\s+диагноз",                # "exact diagnosis"
    ]
    for claim in plan.claims:
        claim_lower = claim.lower()
        for pattern in high_risk_patterns:
            if re.search(pattern, claim_lower):
                errors.append(f"planner_validation: risky claim detected: '{claim[:80]}'")
                break

    return errors


# ---------------------------------------------------------------------------
# Classify opener archetype from text
# ---------------------------------------------------------------------------

def classify_opener_archetype(text: str) -> str:
    """Classify the opening of a text into one of the OPENING_ARCHETYPES.

    Uses keyword matching heuristic. Returns best-matching archetype
    or 'observation' as default.
    """
    if not text:
        return "observation"

    first_line = text.split("\n", 1)[0].strip()
    first_sentence = re.split(r"[.!?\n]", first_line, maxsplit=1)[0].strip().lower()

    if not first_sentence:
        return "observation"

    scores: dict[str, int] = {}
    for archetype, markers in _ARCHETYPE_MARKERS.items():
        score = sum(1 for m in markers if m in first_sentence)
        if score > 0:
            scores[archetype] = score

    if not scores:
        # Fallback heuristics
        if "?" in first_sentence:
            return "question"
        if any(w in first_sentence for w in ["ко мне", "пришёл", "пришел", "обратил"]):
            return "mini_case"
        return "observation"

    return max(scores, key=scores.get)


# ---------------------------------------------------------------------------
# Claim risk scoring (universal, not niche-specific)
# ---------------------------------------------------------------------------

_CLAIM_RISK_PATTERNS: list[tuple[str, int, str]] = [
    # (pattern, risk_points, reason)
    (r"\b\d{2,3}\s*%\s*(?:людей|клиентов|пациентов|пользователей|компаний|случаев)", 3,
     "конкретный процент без источника"),
    (r"(?:доказано|подтверждено)\s+(?:наукой|клинически|исследовани)", 3,
     "ссылка на доказательства без источника"),
    (r"(?:учёные|исследования|эксперты)\s+(?:доказали|установили|подтвердили|показали)", 3,
     "ссылка на анонимных экспертов/исследования"),
    (r"гарантирован\w*\s+(?:результат|эффект|излечен|успех)", 4,
     "гарантия результата"),
    (r"(?:всегда|100\s*%)\s+(?:работает|помогает|решает|эффективн)", 3,
     "абсолютное утверждение об эффективности"),
    (r"точный\s+диагноз|точная\s+причина|единственная?\s+причина", 3,
     "слишком точный технический/медицинский вывод"),
    (r"(?:вылечит|избавит\s+навсегда|полностью\s+устранит)", 4,
     "медицинское обещание"),
    (r"по\s+закону\s+(?:вы|ты)\s+(?:обязаны?|должны?)", 3,
     "юридическое утверждение без ссылки"),
    (r"(?:центральный\s+банк|минздрав|fda|роспотребнадзор)\s+(?:решил|одобрил|подтвердил)", 3,
     "ссылка на решение регулятора без источника"),
    (r"(?:именно|точно|конкретно)\s+(?:из-за|потому что|причина в)", 2,
     "слишком уверенная причинно-следственная связь"),
    (r"(?:ремень|контроллер|плата|bms|датчик)\s+(?:вышел|сгорел|перегорел|вышла|вышло)\s+из\s+строя", 2,
     "конкретный технический диагноз без основания"),
]


def compute_claim_risk(text: str) -> tuple[int, list[str]]:
    """Compute universal claim-risk score for a text.

    Returns (risk_score, list_of_reasons).
    risk_score: 0 = safe, higher = more risky.
    """
    if not text:
        return 0, []

    lower = text.lower()
    total_risk = 0
    reasons: list[str] = []

    for pattern, points, reason in _CLAIM_RISK_PATTERNS:
        matches = re.findall(pattern, lower)
        if matches:
            total_risk += points * len(matches)
            reasons.append(f"claim_risk: {reason} ({len(matches)}x)")

    # Check for fabricated statistics (standalone numbers with % without context)
    pct_matches = re.findall(r"\b\d{2,3}\s*%", lower)
    if len(pct_matches) >= 2 and "источник" not in lower:
        total_risk += 2
        reasons.append(f"claim_risk: множественные процентные показатели без источника ({len(pct_matches)})")

    return total_risk, reasons
