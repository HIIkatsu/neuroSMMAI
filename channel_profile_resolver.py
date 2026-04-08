"""
channel_profile_resolver.py — центральный резолвер профиля канала.

Единственное место, где channel profile → unified policy object.
Все модули генерации используют этот объект вместо дублирования логики.

Архитектура:
  channel_profiles DB row
      ↓
  resolve_channel_policy(owner_id) → ChannelPolicy
      ↓
  content.py / image_search.py / news_service.py / scheduler_service.py / actions.py
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from topic_utils import (
    detect_topic_family,
    get_family_image_queries,
    get_family_allowed_visuals,
    get_family_blocked_visuals,
    get_family_news_angle,
    get_family_guardrails,
    get_family_tone,
    get_family_post_angles,
    get_family_cta_style,
    get_family_irrelevant_classes,
    TOPIC_FAMILY_TERMS,
    FAMILY_GENERATION_RULES,
)


# ---------------------------------------------------------------------------
# Policy dataclass — single source of truth for generation context
# ---------------------------------------------------------------------------

@dataclass
class ChannelPolicy:
    """
    Unified policy object derived from a channel profile.
    All generation modules consume this instead of re-inferring from raw text.
    """
    # Identity
    owner_id: int = 0
    channel_target: str = ""

    # Topic classification
    topic_raw: str = ""
    topic_family: str = "generic"
    topic_subfamily: str = ""

    # Audience & voice
    audience_type: str = ""
    style_mode: str = ""

    # Text generation rules
    tone: str = "concrete, human, practical"
    guardrails: str = ""
    banned_patterns: list[str] = field(default_factory=list)
    cta_style: str = ""
    evidence_level: str = ""
    post_angles: list[str] = field(default_factory=list)

    # Content constraints
    preferred_formats: list[str] = field(default_factory=list)
    forbidden_topics: list[str] = field(default_factory=list)
    forbidden_claims: list[str] = field(default_factory=list)
    content_goals: str = ""
    rubric_map: list[str] = field(default_factory=list)

    # Image / visual policy
    visual_policy: str = "auto"
    allowed_visual_classes: list[str] = field(default_factory=list)
    forbidden_visual_classes: list[str] = field(default_factory=list)
    image_queries: list[str] = field(default_factory=list)

    # News transformation
    news_angle: str = ""
    news_policy: str = "standard"

    # Anti-repeat / memory
    sensitivity_flags: list[str] = field(default_factory=list)

    # Posting preferences
    posting_mode: str = "manual"

    # Author role identity (anti-fabrication guardrails)
    author_role_type: str = ""       # expert / master / business_owner / brand / media / blogger / other
    author_role_description: str = ""  # who the author actually is
    author_activities: str = ""        # what the author really does
    author_forbidden_claims: list[str] = field(default_factory=list)  # what must NOT be attributed to the author

    def as_generation_context(self) -> dict[str, Any]:
        """Returns a flat dict for injection into content generation prompts."""
        return {
            "topic_family": self.topic_family,
            "topic_subfamily": self.topic_subfamily,
            "topic_raw": self.topic_raw,
            "audience": self.audience_type,
            "style": self.style_mode,
            "tone": self.tone,
            "guardrails": self.guardrails,
            "cta_style": self.cta_style,
            "post_angles": self.post_angles,
            "forbidden_topics": self.forbidden_topics,
            "forbidden_claims": self.forbidden_claims,
            "preferred_formats": self.preferred_formats,
            "rubric_map": self.rubric_map,
            "visual_policy": self.visual_policy,
            "image_queries": self.image_queries,
            "news_angle": self.news_angle,
            "content_goals": self.content_goals,
            "author_role_type": self.author_role_type,
            "author_role_description": self.author_role_description,
            "author_activities": self.author_activities,
            "author_forbidden_claims": self.author_forbidden_claims,
        }


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def _parse_json_list(raw: str | None) -> list[str]:
    """Parse a JSON array string or newline/comma-separated string into a list."""
    if not raw:
        return []
    raw = raw.strip()
    if raw.startswith("["):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except (json.JSONDecodeError, ValueError):
            pass
    # Fallback: split by newline or comma
    return [x.strip() for x in re.split(r"[\n,;]+", raw) if x.strip()]


def _infer_subfamily(topic_raw: str, family: str) -> str:
    """Infer a topic subfamily from raw text + detected family."""
    q = (topic_raw or "").lower().strip()
    if family == "food":
        if any(x in q for x in ["рецепт", "recipe", "готов", "кулинар", "cooking"]):
            return "cooking"
        if any(x in q for x in ["ресторан", "кафе", "restaurant", "cafe"]):
            return "restaurant"
        if any(x in q for x in ["магазин", "shop", "store", "продукт", "продаж"]):
            return "store"
        if any(x in q for x in ["веган", "vegan", "растительн", "plant-based"]):
            return "vegan"
        if any(x in q for x in ["выпечк", "bakery", "хлеб", "кондитер"]):
            return "bakery"
        return "general"
    if family == "health":
        if any(x in q for x in ["фитнес", "fitness", "спорт", "тренировк", "gym"]):
            return "fitness"
        if any(x in q for x in ["психолог", "mental", "ментальн", "anxiety", "depression"]):
            return "mental_health"
        if any(x in q for x in ["врач", "клиник", "медицин", "doctor", "clinic", "medicine"]):
            return "medicine"
        if any(x in q for x in ["нутрицион", "питание", "nutrition", "diet", "диет"]):
            return "nutrition"
        if any(x in q for x in ["йог", "медит", "yoga", "meditation", "mindfulness"]):
            return "mindfulness"
        return "wellness"
    if family == "beauty":
        if any(x in q for x in ["косметолог", "cosmetolog", "клиник", "clinic", "инъекц", "ботокс"]):
            return "clinic"
        if any(x in q for x in ["маникюр", "nail", "гель", "shellac"]):
            return "nails"
        if any(x in q for x in ["парикмахер", "стриж", "hair", "haircut", "окраш"]):
            return "hair"
        if any(x in q for x in ["косметик", "skincare", "уход", "крем", "сыворотк"]):
            return "skincare"
        return "general"
    if family == "local_business":
        if any(x in q for x in ["ремонт техник", "ремонт ноутбук", "ремонт смартфон", "laptop repair", "phone repair"]):
            return "repair_electronics"
        if any(x in q for x in ["автосервис", "шиномонтаж", "auto service", "tire"]):
            return "auto_service"
        if any(x in q for x in ["ремонт квартир", "строительств", "renovation", "construction"]):
            return "construction"
        if any(x in q for x in ["клининг", "уборк", "cleaning"]):
            return "cleaning"
        return "services"
    if family == "education":
        if any(x in q for x in ["онлайн", "курс", "online", "course", "edtech"]):
            return "online_course"
        if any(x in q for x in ["школ", "school", "учитель", "teacher"]):
            return "school"
        if any(x in q for x in ["репетитор", "tutor", "подготовк", "егэ", "огэ", "exam"]):
            return "tutoring"
        if any(x in q for x in ["коучинг", "тренинг", "coaching", "training"]):
            return "coaching"
        return "general"
    if family == "finance":
        if any(x in q for x in ["крипто", "crypto", "биткоин", "bitcoin", "нфт", "nft"]):
            return "crypto"
        if any(x in q for x in ["инвестиц", "investment", "акции", "stocks", "брокер"]):
            return "investing"
        if any(x in q for x in ["личн финанс", "бюджет", "budget", "personal finance"]):
            return "personal_finance"
        if any(x in q for x in ["налог", "tax", "бухгалтер", "accounting"]):
            return "tax_accounting"
        return "general"
    if family == "marketing":
        if any(x in q for x in ["smm", "телеграм", "telegram", "соцсет", "social media"]):
            return "smm"
        if any(x in q for x in ["seo", "поисковая", "search"]):
            return "seo"
        if any(x in q for x in ["копирайт", "тексты", "copywriting"]):
            return "copywriting"
        if any(x in q for x in ["таргет", "реклам", "advertising", "targeting"]):
            return "paid_ads"
        return "general"
    return ""


# ---------------------------------------------------------------------------
# AI-assisted normalization (synchronous text-based)
# ---------------------------------------------------------------------------

# Predefined family normalization prompts for user-facing onboarding
FAMILY_NORMALIZATION_EXAMPLES: dict[str, list[str]] = {
    "food": ["здоровое питание", "кулинарный блог", "ресторан", "кофейня", "рецепты", "food", "recipe"],
    "health": ["здоровье", "wellness", "фитнес", "медицина", "психология", "зож", "health"],
    "beauty": ["красота", "бьюти", "косметология", "маникюр", "уход за кожей", "beauty", "skincare"],
    "local_business": ["ремонт", "сервис", "мастерская", "малый бизнес", "услуги", "repair", "local"],
    "education": ["образование", "обучение", "курсы", "школа", "репетитор", "education", "learning"],
    "finance": ["финансы", "инвестиции", "деньги", "крипто", "трейдинг", "finance", "investing"],
    "marketing": ["маркетинг", "smm", "реклама", "продвижение", "контент", "marketing", "smm"],
    "lifestyle": ["лайфстайл", "личное развитие", "путешествия", "образ жизни", "lifestyle"],
    "expert_blog": ["экспертный блог", "авторский канал", "эксперт", "специалист", "expert blog"],
    "massage": ["массаж", "массажист", "самомассаж", "реабилитация", "massage"],
    "cars": ["автомобили", "машины", "авто", "электромобили", "cars", "automotive"],
    "gaming": ["игры", "гейминг", "геймер", "консоль", "gaming"],
    "hardware": ["железо", "компьютеры", "ноутбуки", "hardware", "pc"],
    "tech": ["технологии", "программирование", "ИИ", "разработка", "tech", "ai"],
    "business": ["бизнес", "предпринимательство", "стартап", "менеджмент", "business"],
}


def normalize_topic_to_family(topic_raw: str) -> tuple[str, str]:
    """
    Normalizes a raw topic description into (family, subfamily).
    Used during onboarding and wherever freeform text needs to be classified.

    Returns: (family, subfamily) — always returns a valid pair.
    """
    family = detect_topic_family(topic_raw)
    subfamily = _infer_subfamily(topic_raw, family)
    return family, subfamily


def build_onboarding_normalization_message(topic_raw: str) -> dict[str, str]:
    """
    Builds a normalization proposal for the onboarding flow.
    The system detects the family and proposes it to the user for confirmation.

    Returns a dict with:
      - family: detected family
      - subfamily: inferred subfamily
      - display_family: human-readable Russian name
      - display_subfamily: human-readable Russian name
      - proposal_text: text to show the user
    """
    family, subfamily = normalize_topic_to_family(topic_raw)

    family_display: dict[str, str] = {
        "food": "🍽 Еда и кулинария",
        "health": "💪 Здоровье и wellness",
        "beauty": "💄 Красота и бьюти",
        "local_business": "🔧 Местный бизнес и услуги",
        "education": "📚 Образование и обучение",
        "finance": "💰 Финансы и инвестиции",
        "marketing": "📣 Маркетинг и продвижение",
        "lifestyle": "✨ Лайфстайл и образ жизни",
        "expert_blog": "🎯 Экспертный блог",
        "massage": "🖐 Массаж и восстановление",
        "cars": "🚗 Автомобили",
        "gaming": "🎮 Игры и гейминг",
        "hardware": "💻 Железо и компьютеры",
        "tech": "⚙️ Технологии и разработка",
        "business": "📊 Бизнес и предпринимательство",
        "generic": "📌 Общая тема",
    }

    subfamily_display: dict[str, str] = {
        "cooking": "кулинария/рецепты",
        "restaurant": "ресторан/кафе",
        "store": "магазин/продукты",
        "vegan": "веган/растительное",
        "bakery": "выпечка/хлеб",
        "fitness": "фитнес",
        "mental_health": "ментальное здоровье",
        "medicine": "медицина/клиника",
        "nutrition": "питание/нутрициология",
        "mindfulness": "медитация/mindfulness",
        "wellness": "оздоровление",
        "clinic": "косметологическая клиника",
        "nails": "ногти/маникюр",
        "hair": "волосы/парикмахерская",
        "skincare": "уход за кожей",
        "repair_electronics": "ремонт техники",
        "auto_service": "автосервис",
        "construction": "ремонт квартир/строительство",
        "cleaning": "клининг/уборка",
        "services": "услуги",
        "online_course": "онлайн-курсы",
        "school": "школа",
        "tutoring": "репетиторство",
        "coaching": "коучинг/тренинги",
        "crypto": "крипто/блокчейн",
        "investing": "инвестиции/акции",
        "personal_finance": "личные финансы",
        "tax_accounting": "налоги/бухгалтерия",
        "smm": "SMM/соцсети",
        "seo": "SEO",
        "copywriting": "копирайтинг",
        "paid_ads": "платная реклама",
        "general": "общее направление",
    }

    fam_name = family_display.get(family, "📌 Общая тема")
    sub_name = subfamily_display.get(subfamily, subfamily) if subfamily else ""

    if sub_name:
        proposal = f"Определил тему канала как: {fam_name} → {sub_name}\n\nЭто верно?"
    else:
        proposal = f"Определил тему канала как: {fam_name}\n\nЭто верно?"

    return {
        "family": family,
        "subfamily": subfamily,
        "display_family": fam_name,
        "display_subfamily": sub_name,
        "proposal_text": proposal,
    }


# ---------------------------------------------------------------------------
# Central resolver
# ---------------------------------------------------------------------------

async def resolve_channel_policy(
    owner_id: int,
    *,
    profile: dict | None = None,
    profile_id: int | None = None,
) -> ChannelPolicy:
    """
    Resolves a channel profile for owner_id into a ChannelPolicy.

    If `profile` is provided (pre-fetched from DB), uses it directly.
    If `profile_id` is provided, fetches that specific profile (not just active).
    Otherwise fetches the active profile from the database.

    This is the single entry point for all generation modules.
    """
    import db as db_module  # local import: avoids circular deps (db → content → topic_utils ← channel_profile_resolver)

    if profile is None:
        if profile_id:
            # Fetch specific profile by ID
            profiles = await db_module.list_channel_profiles(owner_id=owner_id)
            profile = next((p for p in profiles if int(p.get("id", 0)) == profile_id), None)
        if profile is None:
            profile = await db_module.get_active_channel_profile(owner_id=owner_id)

    if not profile:
        return _default_policy(owner_id)

    return _build_policy_from_profile(owner_id, profile)


def resolve_channel_policy_sync(
    owner_id: int,
    *,
    profile: dict | None = None,
) -> ChannelPolicy:
    """
    Synchronous version for contexts that already have a profile dict.
    Use resolve_channel_policy (async) when you need to fetch from DB.
    """
    if not profile:
        return _default_policy(owner_id)
    return _build_policy_from_profile(owner_id, profile)


def _build_policy_from_profile(owner_id: int, profile: dict) -> ChannelPolicy:
    """Builds a ChannelPolicy from a DB channel_profiles row dict."""
    # Get structured fields if available, else fall back to raw topic
    topic_raw = (profile.get("topic_raw") or profile.get("topic") or "").strip()
    topic_family = (profile.get("topic_family") or "").strip()
    topic_subfamily = (profile.get("topic_subfamily") or "").strip()

    # Auto-detect family if not stored
    if not topic_family:
        topic_family = detect_topic_family(topic_raw)
    # Auto-detect subfamily if not stored
    if not topic_subfamily and topic_raw:
        topic_subfamily = _infer_subfamily(topic_raw, topic_family)

    # Load structured profile fields
    audience_type = (profile.get("audience_type") or "").strip()
    style_mode = (profile.get("style_mode") or "").strip()
    content_goals = (profile.get("content_goals") or "").strip()
    posting_mode = (profile.get("posting_mode") or "manual").strip()
    visual_policy = (profile.get("visual_policy") or "auto").strip()
    news_policy = (profile.get("news_policy") or "standard").strip()

    # Parse list fields
    preferred_formats = _parse_json_list(profile.get("preferred_formats"))
    forbidden_topics = _parse_json_list(profile.get("forbidden_topics"))
    forbidden_claims = _parse_json_list(profile.get("forbidden_claims"))
    forbidden_visual_classes = _parse_json_list(profile.get("forbidden_visual_classes"))
    rubric_map = _parse_json_list(profile.get("rubric_map"))
    sensitivity_flags = _parse_json_list(profile.get("sensitivity_flags"))

    # Author role fields
    author_role_type = (profile.get("author_role_type") or "").strip()
    author_role_description = (profile.get("author_role_description") or "").strip()
    author_activities = (profile.get("author_activities") or "").strip()
    author_forbidden_claims = _parse_json_list(profile.get("author_forbidden_claims"))

    # Get family-level defaults from topic_utils
    family_guardrails = get_family_guardrails(topic_family)
    family_tone = get_family_tone(topic_family)
    family_post_angles = get_family_post_angles(topic_family)
    family_cta_style = get_family_cta_style(topic_family)
    family_image_queries = get_family_image_queries(topic_family)
    family_allowed_visuals = get_family_allowed_visuals(topic_family)
    family_blocked_visuals = get_family_blocked_visuals(topic_family)
    family_news_angle = get_family_news_angle(topic_family)
    family_irrelevant_classes = get_family_irrelevant_classes(topic_family)

    # Merge stored forbidden_visual_classes with family-level blocked visuals
    merged_forbidden_visuals = list(set(forbidden_visual_classes + family_blocked_visuals + family_irrelevant_classes))

    # Get family generation rules
    gen_rules = FAMILY_GENERATION_RULES.get(topic_family) or FAMILY_GENERATION_RULES.get("generic", {})
    banned_patterns = gen_rules.get("banned_patterns", [])
    evidence_level = gen_rules.get("evidence_level", "practical")

    return ChannelPolicy(
        owner_id=owner_id,
        channel_target=(profile.get("channel_target") or "").strip(),
        topic_raw=topic_raw,
        topic_family=topic_family,
        topic_subfamily=topic_subfamily,
        audience_type=audience_type,
        style_mode=style_mode,
        tone=family_tone,
        guardrails=family_guardrails,
        banned_patterns=banned_patterns,
        cta_style=family_cta_style,
        evidence_level=evidence_level,
        post_angles=family_post_angles,
        preferred_formats=preferred_formats,
        forbidden_topics=forbidden_topics,
        forbidden_claims=forbidden_claims,
        content_goals=content_goals,
        rubric_map=rubric_map,
        visual_policy=visual_policy,
        allowed_visual_classes=family_allowed_visuals,
        forbidden_visual_classes=merged_forbidden_visuals,
        image_queries=family_image_queries,
        news_angle=family_news_angle,
        news_policy=news_policy,
        sensitivity_flags=sensitivity_flags,
        posting_mode=posting_mode,
        author_role_type=author_role_type,
        author_role_description=author_role_description,
        author_activities=author_activities,
        author_forbidden_claims=author_forbidden_claims,
    )


def _default_policy(owner_id: int) -> ChannelPolicy:
    """Returns a safe default policy when no profile is configured."""
    return ChannelPolicy(
        owner_id=owner_id,
        topic_family="generic",
        tone="concrete, human, practical",
        guardrails=(
            "Пиши приземлённо, конкретно и по-человечески. "
            "Никаких нелепых сенсаций, искусственного хайпа и выдуманных будущих сценариев."
        ),
        cta_style="поделись мнением, попробуй, расскажи",
        image_queries=get_family_image_queries("generic"),
        news_angle=get_family_news_angle("generic"),
    )


# ---------------------------------------------------------------------------
# Prompt injection helpers (for use in content.py / news_service.py)
# ---------------------------------------------------------------------------

def build_family_rules_block(policy: ChannelPolicy) -> str:
    """
    Builds a text block to inject into generation prompts.
    Covers tone, guardrails, banned patterns, CTA style.
    """
    parts: list[str] = []

    if policy.topic_family and policy.topic_family != "generic":
        family_display = {
            "food": "Еда и кулинария",
            "health": "Здоровье и wellness",
            "beauty": "Красота и бьюти",
            "local_business": "Местный бизнес и услуги",
            "education": "Образование",
            "finance": "Финансы",
            "marketing": "Маркетинг",
            "lifestyle": "Лайфстайл",
            "expert_blog": "Экспертный блог",
            "massage": "Массаж и восстановление",
            "cars": "Автомобили",
            "gaming": "Гейминг",
            "hardware": "Железо/ПК",
            "tech": "Технологии",
            "business": "Бизнес",
        }
        fname = family_display.get(policy.topic_family, policy.topic_family)
        parts.append(f"Ниша канала: {fname}" + (f" / {policy.topic_subfamily}" if policy.topic_subfamily else ""))

    if policy.guardrails:
        parts.append(f"Правила ниши:\n{policy.guardrails}")

    if policy.banned_patterns:
        patterns_str = ", ".join(f'«{p}»' for p in policy.banned_patterns[:6])
        parts.append(f"Запрещённые паттерны: {patterns_str}")

    if policy.cta_style:
        parts.append(f"Стиль призыва к действию: {policy.cta_style}")

    if policy.forbidden_topics:
        ft_str = ", ".join(policy.forbidden_topics[:8])
        parts.append(f"Запрещённые темы (СТРОГО): {ft_str}")

    if policy.forbidden_claims:
        fc_str = ", ".join(policy.forbidden_claims[:6])
        parts.append(f"Запрещённые утверждения: {fc_str}")

    # Author role identity block — hard constraints against fabrication
    author_block = build_author_role_block(policy)
    if author_block:
        parts.append(author_block)

    return "\n\n".join(parts)


def build_author_role_block(policy: ChannelPolicy) -> str:
    """
    Builds a strict author-role identity block for prompt injection.
    These are hard constraints the LLM must never violate.
    """
    if not policy.author_role_type and not policy.author_role_description and not policy.author_activities:
        return ""

    _ROLE_LABELS: dict[str, str] = {
        "expert": "эксперт",
        "master": "мастер / специалист",
        "business_owner": "владелец бизнеса",
        "brand": "бренд",
        "media": "медиа / редакция",
        "blogger": "блогер",
    }

    parts: list[str] = ["РОЛЬ АВТОРА КАНАЛА (СТРОГО соблюдать — нельзя фантазировать):"]

    if policy.author_role_type:
        label = _ROLE_LABELS.get(policy.author_role_type, policy.author_role_type)
        parts.append(f"- Тип автора: {label}")

    if policy.author_role_description:
        parts.append(f"- Кто автор: {policy.author_role_description}")

    if policy.author_activities:
        parts.append(f"- Что автор реально делает: {policy.author_activities}")

    if policy.author_forbidden_claims:
        afc_str = "; ".join(policy.author_forbidden_claims[:8])
        parts.append(f"- НЕЛЬЗЯ приписывать автору: {afc_str}")

    parts.append(
        "- Категорически запрещено выдумывать услуги, компетенции, достижения, "
        "образование или опыт, которые не указаны выше."
    )

    return "\n".join(parts)


def build_news_family_block(policy: ChannelPolicy, news_title: str = "", news_topic: str = "") -> str:
    """
    Builds the family-aware news transformation instruction block.
    Injected into news post generation prompts.
    """
    angle = policy.news_angle or get_family_news_angle("generic")
    family_block = build_family_rules_block(policy)

    parts = [angle]
    if family_block:
        parts.append(family_block)
    if policy.audience_type:
        parts.append(f"Аудитория канала: {policy.audience_type}")

    return "\n\n".join(parts)


def build_image_search_context(policy: ChannelPolicy, post_topic: str = "") -> dict[str, Any]:
    """
    Builds image search context from the channel policy.
    Used by image_search.py to get family-appropriate queries and filters.
    """
    queries = list(policy.image_queries)

    # Add topic-specific query if topic is available
    if post_topic or policy.topic_raw:
        base = post_topic or policy.topic_raw
        queries.insert(0, f"{base} editorial photography professional")

    return {
        "family": policy.topic_family,
        "queries": queries,
        "allowed_classes": policy.allowed_visual_classes,
        "blocked_classes": policy.forbidden_visual_classes,
        "visual_policy": policy.visual_policy,
    }
