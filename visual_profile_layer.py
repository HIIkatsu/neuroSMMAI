from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

QUERY_CLEAN_RE = re.compile(r"[^a-zA-Z0-9\s-]")
TOKEN_RE = re.compile(r"[^a-zA-Zа-яА-Я0-9\s-]")

_GENERIC_FILLER_TOKENS = {
    "news", "update", "editorial", "media", "photo", "realistic", "community", "business", "open", "office",
    "professional", "quality", "content", "post", "article", "today",
}
_GENERIC_BUSINESS_DRIFT = {
    "handshake", "boardroom", "corporate", "office", "meeting", "coworkers", "smiling team", "generic office",
}
_GLOBAL_FORBIDDEN_TERMS = {"cringe", "meme", "funny", "watermark", "logo", "avatar"}
_QUERY_MAX_TOKENS = 7


@dataclass
class ImageIntent:
    domain: str = "generic"
    primary_subject: str = ""
    secondary_subjects: list[str] = field(default_factory=list)
    scene: str = ""
    forbidden_subjects: list[str] = field(default_factory=list)
    location_context: str = ""
    query_terms_primary: list[str] = field(default_factory=list)
    query_terms_secondary: list[str] = field(default_factory=list)


@dataclass
class VisualProfile:
    domain_family: str = "generic"
    primary_subject: str = "editorial scene"
    secondary_subjects: list[str] = field(default_factory=list)
    scene_type: str = "editorial"
    visual_must_have: list[str] = field(default_factory=list)
    visual_must_not_have: list[str] = field(default_factory=list)
    search_terms_primary: list[str] = field(default_factory=list)
    search_terms_backup: list[str] = field(default_factory=list)
    image_intent: ImageIntent = field(default_factory=ImageIntent)


@dataclass
class ProviderCandidate:
    url: str = ""
    provider: str = ""
    caption: str = ""
    tags: list[str] = field(default_factory=list)
    author: str = ""
    source_query: str = ""
    width: int = 0
    height: int = 0
    query_family: str = "primary"


@dataclass
class ScoreBreakdown:
    score: float
    subject_match: float
    domain_match: float
    scene_match: float
    forbidden_object_penalty: float
    generic_stock_penalty: float
    drift_penalty: float
    repeat_penalty: float
    decision: str
    reason: str


_DOMAIN_ONTOLOGY: dict[str, dict[str, object]] = {
    "finance": {
        "terms": ["finance", "bank", "deposit", "investment", "loan", "mortgage", "stock", "crypto", "финанс", "банк", "депозит", "вклад"],
        "subject": "banking consultation",
        "scene": "bank desk",
        "location": "office",
        "must_have": ["bank", "document", "calculator"],
        "forbidden": ["car showroom", "salon", "food plate"],
    },
    "transport": {
        "terms": ["transport", "bus", "metro", "train", "traffic", "route", "transit", "перевоз", "транспорт", "маршрут"],
        "subject": "public transport",
        "scene": "city street",
        "location": "urban",
        "must_have": ["bus", "street", "station"],
        "forbidden": ["stock chart", "hospital room"],
    },
    "city_local_news": {
        "terms": ["city", "district", "municipal", "local", "neighborhood", "council", "город", "район", "муниципал", "локаль"],
        "subject": "municipal service",
        "scene": "street service",
        "location": "city",
        "must_have": ["street", "public", "service"],
        "forbidden": ["boardroom", "abstract graph"],
    },
    "education": {
        "terms": ["education", "school", "student", "teacher", "course", "exam", "образов", "школ", "студент", "курс"],
        "subject": "learning activity",
        "scene": "classroom",
        "location": "school",
        "must_have": ["student", "classroom", "learning"],
        "forbidden": ["surgery", "engine bay"],
    },
    "healthcare": {
        "terms": ["health", "doctor", "clinic", "medical", "patient", "therapy", "здоров", "врач", "клиник", "мед"],
        "subject": "medical consultation",
        "scene": "clinic room",
        "location": "clinic",
        "must_have": ["doctor", "patient", "clinic"],
        "forbidden": ["race car", "trading desk"],
    },
    "real_estate": {
        "terms": ["real estate", "property", "apartment", "house", "mortgage", "rent", "недвиж", "квартир", "дом", "ипотек"],
        "subject": "property viewing",
        "scene": "home interior",
        "location": "residential",
        "must_have": ["home", "apartment", "property"],
        "forbidden": ["clinic bed", "food closeup"],
    },
    "food": {
        "terms": ["food", "recipe", "kitchen", "meal", "restaurant", "dish", "еда", "рецепт", "кухн"],
        "subject": "food preparation",
        "scene": "kitchen",
        "location": "indoor",
        "must_have": ["dish", "kitchen", "ingredients"],
        "forbidden": ["engine", "stock chart"],
    },
    "auto": {
        "terms": ["auto", "car", "garage", "engine", "mechanic", "tire", "авто", "машин", "двигат", "шины"],
        "subject": "car service",
        "scene": "garage",
        "location": "workshop",
        "must_have": ["car", "mechanic", "garage"],
        "forbidden": ["salad", "classroom"],
    },
    "services": {
        "terms": ["service", "repair", "maintenance", "customer", "workflow", "local business", "сервис", "ремонт", "обслуж"],
        "subject": "service workflow",
        "scene": "service point",
        "location": "commercial",
        "must_have": ["service", "staff", "customer"],
        "forbidden": ["stock chart", "hospital surgery"],
    },
    "lifestyle": {
        "terms": ["lifestyle", "routine", "habit", "wellness", "daily", "лайфстайл", "привыч"],
        "subject": "daily routine",
        "scene": "home lifestyle",
        "location": "home",
        "must_have": ["person", "routine"],
        "forbidden": ["boardroom", "engine workshop"],
    },
    # Backward-compatible aliases used by existing tests/callers.
    "cars": {
        "terms": ["car", "auto", "garage", "engine", "mechanic", "авто", "машин"],
        "subject": "car service",
        "scene": "garage",
        "location": "workshop",
        "must_have": ["car", "mechanic", "garage"],
        "forbidden": ["salad", "classroom"],
    },
    "scooter": {
        "terms": ["scooter", "e-scooter", "samokat", "самокат", "micromobility", "микромобил", "электросамокат"],
        "subject": "urban scooter",
        "scene": "city street",
        "location": "urban",
        "must_have": ["scooter", "street", "wheel"],
        "forbidden": ["sports car", "boardroom"],
    },
    "health": {
        "terms": ["health", "doctor", "clinic", "medical", "patient", "здоров", "врач", "клиник"],
        "subject": "medical consultation",
        "scene": "clinic room",
        "location": "clinic",
        "must_have": ["doctor", "patient", "clinic"],
        "forbidden": ["race car", "trading desk"],
    },
    "local_news": {
        "terms": ["city", "district", "municipal", "local", "community", "local business", "город", "район", "локаль", "новост"],
        "subject": "municipal service",
        "scene": "street service",
        "location": "city",
        "must_have": ["street", "public", "service"],
        "forbidden": ["boardroom", "abstract graph"],
    },
    "gardening": {
        "terms": ["garden", "soil", "seed", "harvest", "plant", "сад", "почв", "семен", "урож"],
        "subject": "gardening practice",
        "scene": "garden",
        "location": "outdoor",
        "must_have": ["soil", "seed", "garden"],
        "forbidden": ["server rack", "trading desk"],
    },
    "electronics": {
        "terms": ["electronics", "device", "gadget", "laptop", "smartphone", "chip", "гаджет", "электрон"],
        "subject": "electronics device",
        "scene": "product desk",
        "location": "indoor",
        "must_have": ["device", "electronics"],
        "forbidden": ["garden field", "operating room"],
    },
    "beauty": {
        "terms": ["beauty", "skincare", "salon", "cosmetic", "makeup", "красот", "салон", "космет"],
        "subject": "beauty care",
        "scene": "beauty studio",
        "location": "indoor",
        "must_have": ["salon", "skincare", "beauty"],
        "forbidden": ["engine bay", "stock chart"],
    },
    "pets": {
        "terms": ["pet", "dog", "cat", "veterinary", "grooming", "питом", "собак", "кошк", "вет"],
        "subject": "pet care",
        "scene": "home or veterinary",
        "location": "indoor",
        "must_have": ["pet", "animal", "care"],
        "forbidden": ["stock chart", "server rack"],
    },
}


def _tokens(text: str) -> list[str]:
    clean = TOKEN_RE.sub(" ", (text or "").lower())
    clean = re.sub(r"\s+", " ", clean).strip()
    return [t for t in clean.split(" ") if len(t) >= 3]


def _compact_unique(tokens: list[str], *, limit: int) -> list[str]:
    out: list[str] = []
    for tok in tokens:
        if tok in _GENERIC_FILLER_TOKENS:
            continue
        if tok not in out:
            out.append(tok)
        if len(out) >= limit:
            break
    return out


def _source_texts(
    *,
    channel_topic: str,
    onboarding_summary: str,
    subniche: str,
    explicit_user_prompt: str,
    title: str,
    body: str,
    text_quality_flagged: bool,
) -> list[tuple[str, int, str]]:
    priority_1 = " ".join(x for x in [channel_topic, onboarding_summary, subniche] if x).strip()
    sources: list[tuple[str, int, str]] = [
        ("channel_onboarding_subniche", 4, priority_1),
        ("explicit_user_prompt", 3, explicit_user_prompt or ""),
        ("title", 2, title or ""),
    ]
    if not text_quality_flagged:
        sources.append(("body", 1, body or ""))
    return sources


def detect_domain_family(
    *,
    title: str = "",
    channel_topic: str = "",
    onboarding_summary: str = "",
    post_intent: str = "",
    body: str = "",
    subniche: str = "",
    explicit_user_prompt: str = "",
    text_quality_flagged: bool = False,
) -> str:
    scores = {k: 0 for k in _DOMAIN_ONTOLOGY}
    for _, weight, text in _source_texts(
        channel_topic=channel_topic,
        onboarding_summary=onboarding_summary,
        subniche=subniche,
        explicit_user_prompt=explicit_user_prompt or post_intent,
        title=title,
        body=body,
        text_quality_flagged=text_quality_flagged,
    ):
        low = text.lower()
        for domain, rule in _DOMAIN_ONTOLOGY.items():
            terms = rule.get("terms", [])
            if any(str(term) in low for term in terms):
                scores[domain] += weight
    best_domain, best_score = max(scores.items(), key=lambda item: item[1])
    return best_domain if best_score > 0 else "generic"


def build_image_intent(
    *,
    title: str = "",
    body: str = "",
    channel_topic: str = "",
    onboarding_summary: str = "",
    subniche: str = "",
    explicit_user_prompt: str = "",
    text_quality_flagged: bool = False,
    content_exclusions: str = "",
    content_constraints: str = "",
) -> ImageIntent:
    short_prompt_tokens = _compact_unique(_tokens(explicit_user_prompt), limit=2)
    short_prompt_is_law = bool(short_prompt_tokens and len(short_prompt_tokens) <= 2)
    domain = detect_domain_family(
        title=title,
        body=body,
        channel_topic=channel_topic,
        onboarding_summary=onboarding_summary,
        subniche=subniche,
        explicit_user_prompt=explicit_user_prompt,
        text_quality_flagged=text_quality_flagged,
    )
    rule = _DOMAIN_ONTOLOGY.get(domain, {})
    primary = str(rule.get("subject") or "editorial subject")
    scene = str(rule.get("scene") or "context scene")
    location = str(rule.get("location") or "")

    source_tokens: list[str] = []
    for source_name, _, text in _source_texts(
        channel_topic=channel_topic,
        onboarding_summary=onboarding_summary,
        subniche=subniche,
        explicit_user_prompt=explicit_user_prompt,
        title=title,
        body=body,
        text_quality_flagged=text_quality_flagged,
    ):
        if short_prompt_is_law and source_name == "body":
            continue
        source_tokens.extend(_tokens(text))
    source_tokens = _compact_unique(source_tokens, limit=20)
    if short_prompt_is_law:
        source_tokens = _compact_unique(short_prompt_tokens + source_tokens, limit=20)

    secondary_subjects = _compact_unique(source_tokens[:12], limit=8)
    must_have = [str(x) for x in rule.get("must_have", [])]

    forbidden = [str(x) for x in rule.get("forbidden", [])]
    forbidden.extend(_tokens(content_exclusions)[:4])
    forbidden.extend(sorted(_GENERIC_BUSINESS_DRIFT))

    primary_terms = _compact_unique(secondary_subjects + _tokens(f"{primary} {scene} {location} {' '.join(must_have)}"), limit=10)
    secondary_terms = _compact_unique(_tokens(f"{primary} {location} {domain.replace('_', ' ')}") + secondary_subjects[2:], limit=10)
    if content_constraints:
        primary_terms = _compact_unique(primary_terms + _tokens(content_constraints), limit=10)

    return ImageIntent(
        domain=domain,
        primary_subject=primary,
        secondary_subjects=secondary_subjects,
        scene=scene,
        forbidden_subjects=_compact_unique(forbidden, limit=12),
        location_context=location,
        query_terms_primary=primary_terms,
        query_terms_secondary=secondary_terms,
    )


def build_visual_profile(
    *,
    title: str = "",
    channel_topic: str = "",
    onboarding_summary: str = "",
    post_intent: str = "",
    body: str = "",
    content_constraints: str = "",
    content_exclusions: str = "",
    subniche: str = "",
    text_quality_flagged: bool = False,
) -> VisualProfile:
    intent = build_image_intent(
        title=title,
        body=body,
        channel_topic=channel_topic,
        onboarding_summary=onboarding_summary,
        subniche=subniche,
        explicit_user_prompt=post_intent,
        text_quality_flagged=text_quality_flagged,
        content_exclusions=content_exclusions,
        content_constraints=content_constraints,
    )
    return VisualProfile(
        domain_family=intent.domain,
        primary_subject=intent.primary_subject,
        secondary_subjects=intent.secondary_subjects,
        scene_type=intent.scene,
        visual_must_have=intent.query_terms_primary[:5],
        visual_must_not_have=intent.forbidden_subjects,
        search_terms_primary=intent.query_terms_primary,
        search_terms_backup=intent.query_terms_secondary,
        image_intent=intent,
    )


def _build_query_from_terms(terms: list[str]) -> str:
    tokens = _compact_unique([tok for term in terms for tok in _tokens(QUERY_CLEAN_RE.sub(" ", term))], limit=_QUERY_MAX_TOKENS)
    return " ".join(tokens)


def profile_search_queries(profile: VisualProfile) -> tuple[str, str]:
    # Build queries only from structured ImageIntent fields.
    intent = profile.image_intent
    primary_query = _build_query_from_terms(intent.query_terms_primary)
    secondary_query = _build_query_from_terms(intent.query_terms_secondary)

    if not primary_query:
        primary_query = _build_query_from_terms([intent.primary_subject, intent.scene, intent.location_context])
    if not secondary_query:
        secondary_query = _build_query_from_terms([intent.primary_subject, *intent.secondary_subjects[:3]])

    logger.info("IMAGE_STOCK_QUERY family=primary token_count=%d query=%r", len(primary_query.split()), primary_query)
    logger.info("IMAGE_STOCK_QUERY family=fallback token_count=%d query=%r", len(secondary_query.split()), secondary_query)
    return primary_query, secondary_query


def score_candidate(
    *,
    candidate: ProviderCandidate,
    profile: VisualProfile,
    used_refs: set[str] | None = None,
    history_duplicate: bool = False,
    min_score: float = 1.5,
) -> ScoreBreakdown:
    metadata_text = " ".join([candidate.caption or "", " ".join(candidate.tags or []), candidate.url or ""]).lower()
    text = " ".join([metadata_text, candidate.source_query or ""]).lower()

    primary_tokens = _tokens(profile.primary_subject)[:4]
    strong_subject_hit = sum(1 for tok in primary_tokens if tok in metadata_text)
    if strong_subject_hit == 0:
        strong_subject_hit = sum(1 for tok in primary_tokens if tok in (candidate.source_query or "").lower())
    subject_hits = sum(1 for tok in primary_tokens if tok in text)
    secondary_hits = sum(1 for tok in profile.secondary_subjects[:8] if tok in text)
    subject_match = float(subject_hits * 1.8 + secondary_hits * 0.5)

    domain_tokens = _tokens(profile.domain_family.replace("_", " "))
    domain_match = 1.5 if any(tok in text for tok in domain_tokens) else 0.0
    scene_match = 1.0 if any(tok in text for tok in _tokens(profile.scene_type)) else 0.0

    forbidden_hits = sum(1 for tok in profile.visual_must_not_have if tok and tok.lower() in text)
    forbidden_hits += sum(1 for tok in _GLOBAL_FORBIDDEN_TERMS if tok in text)
    forbidden_penalty = float(forbidden_hits * 3.0)

    generic_hits = sum(1 for tok in _GENERIC_BUSINESS_DRIFT if tok in text)
    generic_penalty = float(generic_hits * 3.0)

    drift_penalty = 0.0
    if generic_hits >= 1 and strong_subject_hit == 0:
        drift_penalty += 4.0
    if strong_subject_hit == 0:
        drift_penalty += 2.5

    repeat_penalty = 2.5 if (history_duplicate or (used_refs and candidate.url in used_refs)) else 0.0
    score = subject_match + domain_match + scene_match - forbidden_penalty - generic_penalty - drift_penalty - repeat_penalty

    if strong_subject_hit == 0 and candidate.caption:
        decision = "no_image"
        reason = "no_primary_subject_match"
    elif forbidden_hits > 0:
        decision = "no_image"
        reason = "forbidden_subjects_detected"
    elif score < min_score:
        decision = "no_image"
        reason = f"score_below_threshold:{score:.2f}"
    else:
        decision = "accepted"
        reason = "score_ok"

    return ScoreBreakdown(
        score=score,
        subject_match=subject_match,
        domain_match=domain_match,
        scene_match=scene_match,
        forbidden_object_penalty=forbidden_penalty,
        generic_stock_penalty=generic_penalty,
        drift_penalty=drift_penalty,
        repeat_penalty=repeat_penalty,
        decision=decision,
        reason=reason,
    )
