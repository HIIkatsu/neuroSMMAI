from __future__ import annotations

import re
from dataclasses import dataclass, field

LATIN_RE = re.compile(r"[^a-zA-Z0-9\s-]")


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


_DOMAIN_RULES: dict[str, dict[str, list[str] | str]] = {
    "finance": {
        "terms": ["finance", "bank", "deposit", "investment", "money", "loan", "stock", "crypto", "wallet", "финанс", "банк", "вклад", "депозит", "инвест"],
        "subject": "banking and investment",
        "scene": "financial workspace",
        "must_have": ["bank", "deposit", "documents", "calculator", "finance"],
        "must_not": ["sports car", "beauty salon", "pet grooming", "food plate"],
    },
    "cars": {
        "terms": ["car", "automotive", "engine", "vehicle", "garage", "tire", "sedan", "авто", "двигат", "шины", "машин"],
        "subject": "automotive service",
        "scene": "garage or road",
        "must_have": ["car", "mechanic", "road", "engine"],
        "must_not": ["laptop-only", "office meeting", "salad bowl"],
    },
    "scooter": {
        "terms": ["scooter", "micromobility", "handlebar", "wheel", "kick scooter", "e-scooter", "самокат", "микромобил", "подвеск"],
        "subject": "urban scooter",
        "scene": "urban street",
        "must_have": ["scooter", "street", "micromobility"],
        "must_not": ["sports car", "tractor", "hospital bed"],
    },
    "health": {
        "terms": ["health", "doctor", "clinic", "medical", "patient", "wellness", "здоров", "врач", "клиник"],
        "subject": "healthcare consultation",
        "scene": "clinic",
        "must_have": ["doctor", "clinic", "patient"],
        "must_not": ["race track", "luxury car", "stock chart"],
    },
    "local_news": {
        "terms": ["local", "community", "city", "municipal", "district", "service update", "news", "локаль", "город", "район", "новост"],
        "subject": "local community update",
        "scene": "street or municipal service",
        "must_have": ["street", "community", "service"],
        "must_not": ["corporate boardroom", "abstract chart", "studio makeup"],
    },
    "gardening": {
        "terms": ["garden", "gardening", "soil", "seed", "harvest", "plant", "yard", "сад", "огород", "почв", "семен", "урож"],
        "subject": "gardening practice",
        "scene": "garden bed",
        "must_have": ["soil", "seed", "seeds", "plants", "garden"],
        "must_not": ["server rack", "trading desk", "car showroom"],
    },
    "electronics": {
        "terms": ["electronics", "device", "gadget", "chip", "hardware", "smartphone", "laptop", "электрон", "гаджет", "процессор"],
        "subject": "electronics device",
        "scene": "product desk",
        "must_have": ["device", "electronics"],
        "must_not": ["medical surgery", "vegetable garden", "pet kennel"],
    },
    "food": {
        "terms": ["food", "recipe", "meal", "kitchen", "restaurant", "dish", "cook", "еда", "рецепт", "кухн"],
        "subject": "food preparation",
        "scene": "kitchen table",
        "must_have": ["food", "kitchen", "dish"],
        "must_not": ["engine bay", "x-ray room", "bank office"],
    },
    "education": {
        "terms": ["education", "study", "school", "student", "teacher", "learning", "course", "образов", "учеб", "школ", "студент"],
        "subject": "learning environment",
        "scene": "classroom",
        "must_have": ["student", "learning", "classroom"],
        "must_not": ["surgery", "car lift", "stock exchange floor"],
    },
    "beauty": {
        "terms": ["beauty", "skincare", "cosmetic", "makeup", "salon", "hair", "красот", "космет", "салон"],
        "subject": "beauty care",
        "scene": "beauty studio",
        "must_have": ["beauty", "skincare", "salon"],
        "must_not": ["engine", "bank papers", "construction site"],
    },
    "pets": {
        "terms": ["pet", "dog", "cat", "veterinary", "grooming", "animal", "питом", "собак", "кошк", "вет"],
        "subject": "pet care",
        "scene": "home or veterinary",
        "must_have": ["pet", "animal"],
        "must_not": ["stock chart", "tractor field", "circuit board macro"],
    },
    "real_estate": {
        "terms": ["real estate", "apartment", "house", "property", "mortgage", "home", "недвиж", "квартир", "дом", "ипотек"],
        "subject": "property viewing",
        "scene": "interior or exterior property",
        "must_have": ["house", "apartment", "property"],
        "must_not": ["scooter lane", "operating room", "kitchen dish closeup"],
    },
}

_GENERIC_PENALTY_TERMS = {"handshake", "thumbs up", "business team", "generic office", "smiling colleagues", "stock photo"}
_GLOBAL_FORBIDDEN_TERMS = {"cringe", "meme", "funny", "watermark", "logo", "avatar"}

_DOMAIN_PRIMARY_KEYWORD = {
    "finance": "bank deposit",
    "cars": "car mechanic",
    "scooter": "scooter urban",
    "health": "doctor clinic",
    "local_news": "local community",
    "gardening": "soil seed",
    "electronics": "electronics device",
    "food": "food kitchen",
    "education": "student classroom",
    "beauty": "beauty salon",
    "pets": "pet care",
    "real_estate": "property house",
}


def _tokens(text: str) -> list[str]:
    clean = LATIN_RE.sub(" ", (text or "").lower())
    clean = re.sub(r"\s+", " ", clean).strip()
    return [t for t in clean.split(" ") if len(t) >= 3]


def _extract_priority_context(*, title: str, channel_topic: str, onboarding_summary: str, post_intent: str, body: str) -> list[tuple[str, str]]:
    return [
        ("title", title or ""),
        ("channel_topic", channel_topic or ""),
        ("onboarding", onboarding_summary or ""),
        ("post_intent", post_intent or ""),
        ("body", body or ""),
    ]


def detect_domain_family(*, title: str = "", channel_topic: str = "", onboarding_summary: str = "", post_intent: str = "", body: str = "") -> str:
    weighted_hits: dict[str, int] = {k: 0 for k in _DOMAIN_RULES}
    priorities = {"title": 5, "channel_topic": 4, "onboarding": 3, "post_intent": 2, "body": 1}
    for source_name, source_text in _extract_priority_context(
        title=title,
        channel_topic=channel_topic,
        onboarding_summary=onboarding_summary,
        post_intent=post_intent,
        body=body,
    ):
        t = (source_text or "").lower()
        if not t:
            continue
        weight = priorities[source_name]
        for domain, rule in _DOMAIN_RULES.items():
            if any(term in t for term in rule["terms"]):
                weighted_hits[domain] += weight
    best = max(weighted_hits.items(), key=lambda x: x[1])
    return best[0] if best[1] > 0 else "generic"


def build_visual_profile(
    *,
    title: str = "",
    channel_topic: str = "",
    onboarding_summary: str = "",
    post_intent: str = "",
    body: str = "",
    content_constraints: str = "",
    content_exclusions: str = "",
) -> VisualProfile:
    domain = detect_domain_family(
        title=title,
        channel_topic=channel_topic,
        onboarding_summary=onboarding_summary,
        post_intent=post_intent,
        body=body,
    )
    rule = _DOMAIN_RULES.get(domain, {})

    primary_anchor = " ".join(part for part in [title, channel_topic, onboarding_summary, post_intent] if part).strip()
    body_tokens = _tokens(body)[:5]
    primary_tokens = _tokens(primary_anchor)[:8]
    primary_subject = (rule.get("subject") or "editorial scene")
    if primary_tokens:
        primary_subject = f"{primary_subject} {' '.join(primary_tokens[:3])}".strip()

    secondaries: list[str] = []
    for tok in primary_tokens[3:7] + body_tokens[:4]:
        if tok not in secondaries:
            secondaries.append(tok)

    scene = str(rule.get("scene") or "editorial")
    must_have = [str(x) for x in (rule.get("must_have") or [])]
    must_not = [str(x) for x in (rule.get("must_not") or [])]
    if content_exclusions:
        must_not.extend(_tokens(content_exclusions)[:6])
    if content_constraints:
        must_have.extend(_tokens(content_constraints)[:4])

    domain_hint = _DOMAIN_PRIMARY_KEYWORD.get(domain, domain.replace("_", " "))
    primary_terms = [domain_hint, scene, primary_subject] + must_have[:3] + secondaries[:3]
    backup_terms = [domain_hint, primary_subject, domain.replace("_", " ")] + secondaries[:5]

    return VisualProfile(
        domain_family=domain,
        primary_subject=primary_subject,
        secondary_subjects=secondaries[:8],
        scene_type=scene,
        visual_must_have=must_have[:8],
        visual_must_not_have=must_not[:10],
        search_terms_primary=[t for t in primary_terms if t][:8],
        search_terms_backup=[t for t in backup_terms if t][:8],
    )


def profile_search_queries(profile: VisualProfile) -> tuple[str, str]:
    def _build(terms: list[str]) -> str:
        raw = " ".join(terms)
        clean = LATIN_RE.sub(" ", raw)
        clean = re.sub(r"\s+", " ", clean).strip().lower()
        return clean[:180] if clean else "editorial photo"

    return _build(profile.search_terms_primary), _build(profile.search_terms_backup)


def score_candidate(
    *,
    candidate: ProviderCandidate,
    profile: VisualProfile,
    used_refs: set[str] | None = None,
    history_duplicate: bool = False,
    min_score: float = 1.5,
) -> ScoreBreakdown:
    metadata_text = " ".join([
        candidate.caption or "",
        " ".join(candidate.tags or []),
        candidate.url or "",
    ]).lower()
    text = " ".join([
        metadata_text,
        candidate.source_query or "",
    ]).lower()

    subject_hits = sum(1 for tok in _tokens(profile.primary_subject)[:8] if tok in text)
    secondary_hits = sum(1 for tok in profile.secondary_subjects[:8] if tok in text)
    subject_match = float(subject_hits * 1.5 + secondary_hits * 0.7)

    subject_hits_meta = sum(1 for tok in _tokens(profile.primary_subject)[:8] if tok in metadata_text)
    domain_meta_hit = profile.domain_family.replace("_", " ") in metadata_text

    domain_match = 2.0 if profile.domain_family.replace("_", " ") in text else 0.0
    if domain_match == 0.0:
        domain_match = 1.2 if any(tok in text for tok in _tokens(profile.domain_family.replace("_", " "))) else 0.0

    scene_match = 1.3 if any(tok in text for tok in _tokens(profile.scene_type)) else 0.0

    forbidden_hits = sum(1 for tok in profile.visual_must_not_have if tok and tok.lower() in text)
    forbidden_hits += sum(1 for tok in _GLOBAL_FORBIDDEN_TERMS if tok in text)
    forbidden_penalty = float(forbidden_hits * 2.5)

    generic_hits = sum(1 for tok in _GENERIC_PENALTY_TERMS if tok in text)
    if not candidate.caption and not candidate.tags:
        generic_hits += 1
    generic_penalty = float(generic_hits * 3.0)
    if generic_hits >= 2 and (candidate.caption or candidate.tags):
        generic_penalty += 6.0

    drift_penalty = 2.0 if (subject_match == 0.0 and domain_match == 0.0) else 0.0
    if (candidate.caption or candidate.tags) and subject_hits_meta == 0 and not domain_meta_hit:
        drift_penalty += 3.0
    repeat_penalty = 2.5 if (history_duplicate or (used_refs and candidate.url in used_refs)) else 0.0

    score = subject_match + domain_match + scene_match - forbidden_penalty - generic_penalty - drift_penalty - repeat_penalty

    if score < min_score:
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
