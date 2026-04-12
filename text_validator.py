"""
text_validator.py — Runtime anti-fabrication and source-fit validator.

Validates generated text AFTER generation and BEFORE acceptance.
Catches:
  1. Fabricated statistics / numeric claims without source support
  2. Fabricated personal experience / service claims without permission
  3. Topic drift from news source
  4. Repeated opener/CTA patterns
  5. Overused text templates

This is NOT prompt-level guidance — it's a runtime gate that rejects or
flags text that violates factual safety rules regardless of what the LLM
produced.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------------------
@dataclass
class TextValidationResult:
    """Result of runtime text validation."""
    is_valid: bool = True
    fake_numeric_claims: list[str] = field(default_factory=list)
    fake_personal_claims: list[str] = field(default_factory=list)
    source_drift_reasons: list[str] = field(default_factory=list)
    template_repeat_hits: list[str] = field(default_factory=list)
    total_risk_score: int = 0
    # Structured log fields
    log_events: list[str] = field(default_factory=list)

    @property
    def should_reject(self) -> bool:
        return not self.is_valid

    def summary(self) -> dict[str, Any]:
        return {
            "valid": self.is_valid,
            "risk_score": self.total_risk_score,
            "fake_numeric": len(self.fake_numeric_claims),
            "fake_personal": len(self.fake_personal_claims),
            "source_drift": len(self.source_drift_reasons),
            "template_repeat": len(self.template_repeat_hits),
        }


# ---------------------------------------------------------------------------
# Fabricated numeric claim patterns
# ---------------------------------------------------------------------------
_FAKE_NUMERIC_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(\d{1,2})\s+из\s+(\d{1,2})\b", re.I),
     "fabricated_ratio"),
    (re.compile(r"\b\d{2,3}\s*%\s*(?:людей|клиентов|случаев|пациентов|водителей|пользователей|компаний|владельцев|автомобилей|мастеров|заведений)", re.I),
     "fabricated_percentage_with_subject"),
    (re.compile(r"(?:по данным|согласно|по статистике|по результатам)\s+(?:аналитиков|исследовани|опрос|экспертов|статистик|СТО|сервис|страхов)", re.I),
     "fabricated_authority_reference"),  # "страхов" intentionally matches both "страховых" and "страховщиков"
    (re.compile(r"(?:исследовани[ея]|опрос[ы]?|аналитики|эксперты)\s+(?:показали|выявили|обнаружили|подтвердили|установили|доказали)", re.I),
     "fabricated_study_claim"),
    (re.compile(r"(?:в\s+\d{4}\s+(?:году?|г\.?))\s+(?:исследовани|опрос|стат|учёные|эксперты|аналитики)", re.I),
     "fabricated_dated_study"),
    (re.compile(r"(?:страховые компании|банки|автосалоны|сервисы|клиники)\s+(?:говорят|утверждают|подтверждают|рекомендуют)", re.I),
     "fabricated_industry_claim"),
    (re.compile(r"(?:мы проверили|мы обзвонили|мы протестировали|мы сравнили|мы изучили)\s+(\d+)\s+", re.I),
     "fabricated_we_tested_N"),
    (re.compile(r"(?:доказано|клинически|научно)\s+(?:подтверждено|доказано|установлено)", re.I),
     "fabricated_scientific_proof"),
    # Named authority without source: "Tom's Hardware выяснили", "Forbes написали"
    (re.compile(r"(?:Tom'?s\s+Hardware|Forbes|Bloomberg|Reuters|TechRadar|Wired|CNET|The Verge)\s+(?:выяснил[иа]?|обнаружил[иа]?|подтвердил[иа]?|написал[иа]?|сообщил[иа]?)", re.I),
     "fabricated_named_authority"),
    # "по данным страховых компаний/банков/аналитиков"
    (re.compile(r"по\s+данным\s+(?:страховых\s+компаний|банков|автосалонов|сервисных\s+центров|производителей)", re.I),
     "fabricated_data_authority"),
    # Municipal/government authority fabrication — city/transport mode critical
    (re.compile(r"(?:по данным|согласно|по решению)\s+(?:департамент[аеу]?|управлени[яе]|администрац\w+|мэри\w*|комитет[аеу]?|министерств[аоу]?)", re.I),
     "fabricated_government_reference"),
    (re.compile(r"(?:департамент транспорта|управление транспорт\w*|комитет по транспорт\w*)\s+(?:решил|объявил|утвердил|сообщил|подтвердил)", re.I),
     "fabricated_transport_department"),
    # Fake local data — prices, tariffs, regulations without source
    (re.compile(r"(?:тариф|стоимость проезда|цена билет\w*|стоимость аренды)\s+(?:\w+\s+){0,3}(?:составляет|равн[аы]|достигает|вырос(?:ла)? до)\s+\d+", re.I),
     "fabricated_local_tariff"),
]


# ---------------------------------------------------------------------------
# Fabricated personal experience patterns
# ---------------------------------------------------------------------------
_FAKE_PERSONAL_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"(?:клиент\s+пришёл|клиент\s+пришел|клиент\s+обратился)", re.I),
     "fabricated_client_came"),
    (re.compile(r"(?:ко\s+мне\s+(?:пришёл|пришел|обратил|приехал|позвонил))", re.I),
     "fabricated_came_to_me"),
    (re.compile(r"(?:в\s+моём?\s+сервисе|в\s+моей\s+мастерской|в\s+нашем\s+сервисе|в\s+нашей\s+мастерской)", re.I),
     "fabricated_my_service"),
    (re.compile(r"(?:мой\s+последний\s+клиент|моя\s+последняя\s+клиентка)", re.I),
     "fabricated_last_client"),
    (re.compile(r"(?:из\s+моей\s+практики|из\s+нашей\s+практики|из\s+моего\s+опыта)", re.I),
     "fabricated_from_practice"),
    (re.compile(r"(?:недавно\s+ко\s+мне|недавно\s+к\s+нам|вчера\s+ко\s+мне)", re.I),
     "fabricated_recently_came"),
    (re.compile(r"(?:звонит\s+клиент|пишет\s+клиент|приезжает\s+клиент)", re.I),
     "fabricated_client_contacts"),
    (re.compile(r"(?:мы\s+часто\s+видим|мы\s+регулярно\s+видим|мы\s+постоянно\s+сталкиваемся)", re.I),
     "fabricated_we_often_see"),
]

# Keywords in input that explicitly allow personal experience
_PERSONAL_PERMISSION_KEYWORDS = [
    "клиент пришёл", "клиент пришел", "ко мне", "в моём сервисе",
    "в моем сервисе", "в мастерской", "мой опыт", "из практики",
    "из моей практики", "личный опыт", "мой случай", "кейс клиента",
    "история клиента", "реальный случай",
]


# ---------------------------------------------------------------------------
# Overused CTA patterns
# ---------------------------------------------------------------------------
_OVERUSED_CTA_PATTERNS: list[tuple[str, str]] = [
    ("проверьте уже сегодня", "cta_check_today"),
    ("проверьте прямо сейчас", "cta_check_now"),
    ("задумайтесь об этом", "cta_think_about"),
    ("спросите себя", "cta_ask_yourself"),
    ("попробуйте и вы", "cta_try_it"),
    ("не откладывайте", "cta_dont_delay"),
    ("начните уже сегодня", "cta_start_today"),
    ("что думаете?", "cta_what_think"),
    ("а как у вас?", "cta_how_about_you"),
    ("делитесь в комментариях", "cta_share_comments"),
]


# ---------------------------------------------------------------------------
# Core validation functions
# ---------------------------------------------------------------------------

def validate_numeric_claims(
    text: str,
    source_facts: list[str] | None = None,
    source_text: str = "",
) -> list[str]:
    """Check for fabricated numeric/authority claims not in source.

    Returns list of violation reasons.
    """
    if not text:
        return []

    violations: list[str] = []
    lower = text.lower()
    source_combined = " ".join(source_facts or []).lower() + " " + source_text.lower()

    for pattern, reason in _FAKE_NUMERIC_PATTERNS:
        matches = pattern.findall(lower)
        if matches:
            # Check if claim is grounded in source
            for match_group in matches:
                match_str = match_group if isinstance(match_group, str) else " ".join(match_group)
                if match_str and source_combined and match_str.lower() in source_combined:
                    continue  # Claim is grounded
                violations.append(reason)
                break  # One violation per pattern is enough

    return violations


def validate_personal_claims(
    text: str,
    allow_personal: bool = False,
    input_text: str = "",
) -> list[str]:
    """Check for fabricated personal experience claims.

    Returns list of violation reasons.
    """
    if not text or allow_personal:
        return []

    # Check if input explicitly allows personal claims
    input_lower = input_text.lower()
    if any(kw in input_lower for kw in _PERSONAL_PERMISSION_KEYWORDS):
        return []

    violations: list[str] = []
    lower = text.lower()

    for pattern, reason in _FAKE_PERSONAL_PATTERNS:
        if pattern.search(lower):
            violations.append(reason)

    return violations


def validate_source_fit(
    text: str,
    source_title: str = "",
    source_summary: str = "",
    source_facts: list[str] | None = None,
) -> tuple[int, list[str]]:
    """Validate that generated text stays on-topic with source.

    Returns (fit_score 0-10, list of drift reasons).
    Higher fit_score = better alignment.
    """
    if not text or not (source_title or source_summary or source_facts):
        return 10, []  # No source to check against

    text_lower = text.lower()
    drift_reasons: list[str] = []
    fit_score = 10

    # Extract key entities from source
    source_entities = set()
    source_combined = f"{source_title} {source_summary}".lower()
    # Split into meaningful words (≥4 chars, not stopwords)
    _stopwords = {
        "что", "как", "это", "для", "при", "или", "они", "быть",
        "который", "этот", "такой", "если", "также", "более", "будет",
        "может", "были", "было", "есть", "все", "всё", "еще", "ещё",
        "уже", "тоже", "свой", "его", "неё", "она", "они",
    }
    for word in re.findall(r"[а-яёa-z]{4,}", source_combined):
        if word not in _stopwords:
            source_entities.add(word)

    if not source_entities:
        return 10, []

    # Count how many source entities appear in generated text
    entity_hits = sum(1 for e in source_entities if e in text_lower)
    coverage = entity_hits / max(len(source_entities), 1)

    if coverage < 0.15:
        fit_score -= 5
        drift_reasons.append(f"low_source_entity_coverage:{coverage:.2f}")

    if coverage < 0.05:
        fit_score -= 3
        drift_reasons.append("very_low_source_coverage")

    # Check title alignment
    if source_title:
        title_words = set(re.findall(r"[а-яёa-z]{4,}", source_title.lower())) - _stopwords
        title_hits = sum(1 for w in title_words if w in text_lower) if title_words else 0
        title_coverage = title_hits / max(len(title_words), 1)
        if title_coverage < 0.2 and len(title_words) >= 3:
            fit_score -= 3
            drift_reasons.append(f"title_drift:{title_coverage:.2f}")

    return max(fit_score, 0), drift_reasons


def validate_template_repetition(
    text: str,
    recent_texts: list[str] | None = None,
) -> list[str]:
    """Check for overused CTA and opener patterns.

    Returns list of pattern hits.
    """
    if not text:
        return []

    hits: list[str] = []
    lower = text.lower()

    for pattern, bucket in _OVERUSED_CTA_PATTERNS:
        if pattern in lower:
            hits.append(f"cta:{bucket}")

    return hits


# ---------------------------------------------------------------------------
# Full text validation (orchestrator)
# ---------------------------------------------------------------------------

def validate_generated_text(
    text: str,
    *,
    generation_mode: str = "manual",
    content_mode: str = "",
    source_title: str = "",
    source_summary: str = "",
    source_facts: list[str] | None = None,
    source_text: str = "",
    input_text: str = "",
    allow_personal: bool = False,
    recent_texts: list[str] | None = None,
    reject_threshold: int = 6,
) -> TextValidationResult:
    """Run all text validation checks and produce a structured result.

    Parameters:
        text: Generated text to validate
        generation_mode: "manual", "autopost", or "news"
        content_mode: Detected content mode from content_modes.py (optional)
        source_title: News source title (news mode)
        source_summary: News source summary (news mode)
        source_facts: Extracted facts from source (news mode)
        source_text: Raw source article text
        input_text: User input / request text
        allow_personal: Whether personal experience is explicitly allowed
        recent_texts: Recent post texts for template repeat detection
        reject_threshold: Risk score threshold for rejection (default: 6)
    """
    # Override reject_threshold if content_mode provides a stricter one
    if content_mode:
        try:
            from content_modes import get_mode_reject_threshold, is_factual_strict
            mode_threshold = get_mode_reject_threshold(content_mode)
            if mode_threshold < reject_threshold:
                reject_threshold = mode_threshold
                logger.info(
                    "TEXT_VALIDATOR_MODE_OVERRIDE content_mode=%s reject_threshold=%d",
                    content_mode, reject_threshold,
                )
            # For strict factual modes, increase penalty weight on authority claims
            _strict_mode = is_factual_strict(content_mode)
        except ImportError:
            _strict_mode = False
    else:
        _strict_mode = False
    result = TextValidationResult()

    if not text:
        return result

    risk = 0

    # 1. Numeric / authority claim validation
    numeric_violations = validate_numeric_claims(
        text,
        source_facts=source_facts,
        source_text=source_text,
    )
    if numeric_violations:
        result.fake_numeric_claims = numeric_violations
        # In strict factual modes, authority violations get higher penalty
        base_penalty = 4 if _strict_mode else 3
        risk += len(numeric_violations) * base_penalty
        # Categorize: authority-like reasons vs pure numeric reasons
        _authority_reasons = {
            "fabricated_authority_reference", "fabricated_study_claim",
            "fabricated_dated_study", "fabricated_industry_claim",
            "fabricated_scientific_proof", "fabricated_named_authority",
            "fabricated_data_authority", "fabricated_government_reference",
            "fabricated_transport_department", "fabricated_local_tariff",
        }
        for v in numeric_violations:
            if v in _authority_reasons:
                result.log_events.append(f"TEXT_FAKE_AUTHORITY_REJECT reason={v}")
            else:
                result.log_events.append(f"TEXT_FAKE_NUMERIC_REJECT reason={v}")
        logger.warning(
            "TEXT_FAKE_NUMERIC_REJECT count=%d reasons=%s text_excerpt=%r",
            len(numeric_violations), numeric_violations, text[:100],
        )

    # 2. Personal experience claim validation
    personal_violations = validate_personal_claims(
        text,
        allow_personal=allow_personal,
        input_text=input_text,
    )
    if personal_violations:
        result.fake_personal_claims = personal_violations
        risk += len(personal_violations) * 3
        for v in personal_violations:
            result.log_events.append(f"TEXT_FAKE_PERSONAL_EXPERIENCE_REJECT reason={v}")
        logger.warning(
            "TEXT_FAKE_PERSONAL_EXPERIENCE_REJECT count=%d reasons=%s text_excerpt=%r",
            len(personal_violations), personal_violations, text[:100],
        )

    # 3. Source-fit validation (news mode only)
    if generation_mode == "news":
        fit_score, drift_reasons = validate_source_fit(
            text,
            source_title=source_title,
            source_summary=source_summary,
            source_facts=source_facts,
        )
        if drift_reasons:
            result.source_drift_reasons = drift_reasons
            risk += (10 - fit_score)
            for reason in drift_reasons:
                result.log_events.append(f"TEXT_DRIFT_REJECT reason={reason}")
            logger.warning(
                "TEXT_SOURCE_FIT_SCORE=%d drift=%s text_excerpt=%r",
                fit_score, drift_reasons, text[:100],
            )
        else:
            logger.info("TEXT_SOURCE_FIT_SCORE=%d", fit_score)

    # 3b. Request-fit validation (manual/autopost modes)
    if generation_mode in ("manual", "autopost") and input_text:
        req_fit_score, req_drift = validate_source_fit(
            text,
            source_title=input_text,
            source_summary="",
            source_facts=None,
        )
        logger.info("TEXT_REQUEST_FIT_SCORE=%d", req_fit_score)
        if req_drift and req_fit_score < 4:
            risk += 2
            for reason in req_drift:
                result.log_events.append(f"TEXT_DRIFT_REJECT reason=request_{reason}")

    # 4. Template repetition
    template_hits = validate_template_repetition(text, recent_texts)
    if template_hits:
        result.template_repeat_hits = template_hits
        risk += len(template_hits)
        for hit in template_hits:
            result.log_events.append(f"TEXT_TEMPLATE_REPEAT_PENALTY={hit}")

    result.total_risk_score = risk
    result.is_valid = risk < reject_threshold

    if not result.is_valid:
        logger.warning(
            "TEXT_VALIDATION_REJECT risk=%d numeric=%d personal=%d drift=%d template=%d",
            risk, len(result.fake_numeric_claims), len(result.fake_personal_claims),
            len(result.source_drift_reasons), len(result.template_repeat_hits),
        )

    return result
