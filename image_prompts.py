"""
image_prompts.py — Prompt builder for the generation-first image system.

Converts post/channel/topic context into strong image generation prompts.
Mode-aware: uses content_mode to ensure prompts match the post type.
No dependency on any deleted image modules.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from topic_utils import detect_topic_family, detect_subfamily
from content_modes import (
    detect_content_mode,
    get_mode_image_rules,
    check_image_prompt_relevance,
    fix_image_prompt_for_mode,
    MODE_GENERIC,
)

logger = logging.getLogger(__name__)


@dataclass
class VisualTopicAnchor:
    """Stable semantic anchor for image generation/search."""

    family: str = "generic"
    primary_text: str = ""
    secondary_text: str = ""
    subject_hint: str = ""

# Style guidance per topic family — maps family to photographic style hints
_FAMILY_STYLE: dict[str, str] = {
    "massage": "professional spa photography, warm natural lighting, wellness atmosphere",
    "food": "appetizing food photography, natural lighting, editorial composition",
    "health": "clean health and wellness imagery, natural tones, professional medical context",
    "beauty": "beauty and skincare photography, soft studio lighting, elegant composition",
    "local_business": "authentic small business photography, warm inviting atmosphere",
    "education": "modern education and learning imagery, bright inspiring environment",
    "finance": "professional finance and business imagery, clean modern aesthetics",
    "marketing": "creative marketing and branding visuals, bold modern design",
    "lifestyle": "authentic lifestyle photography, candid warm moments",
    "expert_blog": "thoughtful expert portrait or workspace, professional editorial style",
    "cars": "automotive photography, dramatic lighting, professional car exterior or detail shot",
    "gaming": "gaming culture imagery, dynamic lighting, modern tech aesthetics",
    "hardware": "technology hardware photography, clean studio shots, detailed close-ups",
    "tech": "modern technology and software visuals, clean minimalist design",
    "business": "corporate professional photography, modern office environment",
}

_NEGATIVE_PROMPT = (
    "blurry, low quality, text overlay, watermark, logo, clipart, cartoon, "
    "illustration, 3d render, cgi, anime, painting, sketch, collage, "
    "distorted, deformed, ugly, oversaturated, artificial neon, "
    "stock photo cliché, generic handshake, thumbs up"
)


def build_generation_prompt(
    *,
    title: str = "",
    body: str = "",
    channel_topic: str = "",
    llm_image_prompt: str = "",
    content_mode: str = "",
    channel_style: str = "",
    channel_audience: str = "",
    channel_subniche: str = "",
    onboarding_summary: str = "",
    content_constraints: str = "",
    content_exclusions: str = "",
    visual_style: str = "",
    forbidden_visuals: str = "",
    post_intent: str = "",
    text_quality_flagged: bool = False,
) -> dict[str, str]:
    """Build a structured prompt for image generation.

    Returns dict with keys:
        prompt: str — the main generation prompt
        negative_prompt: str — what to avoid
        style_hint: str — photographic style guidance
        family: str — detected topic family
        content_mode: str — detected or provided content mode
    """
    anchor = _build_visual_topic_anchor(
        title=title,
        body=body,
        channel_topic=channel_topic,
        onboarding_summary=onboarding_summary,
        content_constraints=content_constraints,
        post_intent=post_intent,
        text_quality_flagged=text_quality_flagged,
    )
    family = anchor.family
    style_hint = _FAMILY_STYLE.get(family, "professional editorial photography, natural lighting")

    # Detect content mode if not provided
    if not content_mode:
        content_mode = detect_content_mode(
            title=title, body=body, channel_topic=channel_topic,
        )

    # Get mode-specific image rules
    mode_rules = get_mode_image_rules(content_mode)
    mode_style = mode_rules.get("style", "")
    mode_scene_hint = mode_rules.get("scene_hint", "")
    mode_forbidden = mode_rules.get("forbidden_scenes", "")

    # Build the core prompt
    # Priority: LLM-generated prompt > title + body summary > channel topic
    if llm_image_prompt and llm_image_prompt.strip():
        core = llm_image_prompt.strip()
        # Apply mode-aware relevance fix
        core = fix_image_prompt_for_mode(core, title, content_mode)
    elif title:
        # Extract visual essence from title + first part of body
        body_lead = ""
        if body and not text_quality_flagged:
            body_lead = (body or "").split("\n", 1)[0].strip()[:200]
        core = _extract_visual_essence(title, body_lead, family, content_mode)
    elif channel_topic:
        core = f"Professional photo related to: {channel_topic}"
    else:
        core = "Professional editorial photograph"

    onboarding_context = _compose_onboarding_context(
        channel_style=channel_style,
        channel_audience=channel_audience,
        channel_subniche=channel_subniche,
        onboarding_summary=onboarding_summary,
        visual_style=visual_style,
        post_intent=post_intent,
    )
    hard_limits = _compose_hard_limits(
        content_constraints=content_constraints,
        content_exclusions=content_exclusions,
        forbidden_visuals=forbidden_visuals,
    )

    # Compose the final prompt with mode-aware style
    effective_style = mode_style if mode_style else style_hint
    prompt = f"{core}. {effective_style}. {onboarding_context}. High quality, photorealistic, 4K."

    # Add scene hint for mode specificity
    if mode_scene_hint and mode_scene_hint not in prompt:
        prompt = f"{core}. {mode_scene_hint}. {effective_style}. {onboarding_context}. High quality, photorealistic, 4K."

    # Add forbidden scenes to negative prompt
    negative = _NEGATIVE_PROMPT
    if mode_forbidden:
        negative = f"{_NEGATIVE_PROMPT}, {mode_forbidden}"
    if hard_limits:
        negative = f"{negative}, {hard_limits}"

    logger.info(
        "IMAGE_PROMPT_BUILT family=%s mode=%s prompt_len=%d has_llm_prompt=%s title=%r",
        family, content_mode, len(prompt), bool(llm_image_prompt), (title or "")[:60],
    )

    return {
        "prompt": prompt,
        "negative_prompt": negative,
        "style_hint": effective_style,
        "family": family,
        "content_mode": content_mode,
    }


def _build_visual_topic_anchor(
    *,
    title: str = "",
    body: str = "",
    channel_topic: str = "",
    onboarding_summary: str = "",
    content_constraints: str = "",
    post_intent: str = "",
    text_quality_flagged: bool = False,
) -> VisualTopicAnchor:
    primary_parts = [channel_topic, title, onboarding_summary, content_constraints, post_intent]
    primary = " ".join(x.strip() for x in primary_parts if (x or "").strip())

    secondary = ""
    if body and not text_quality_flagged:
        secondary = body[:220]

    pinned = _detect_pinned_family(primary)
    if not pinned and secondary:
        pinned = _detect_pinned_family(secondary)

    family_source = primary if primary.strip() else secondary
    family = pinned or (detect_topic_family(family_source) if family_source.strip() else "generic")

    subject_hint = _infer_subject_hint(primary=primary, secondary=secondary, family=family)
    return VisualTopicAnchor(family=family, primary_text=primary, secondary_text=secondary, subject_hint=subject_hint)


def _detect_pinned_family(text: str) -> str:
    s = (text or "").lower()
    pins: list[tuple[str, tuple[str, ...]]] = [
        ("finance", ("вклад", "депозит", "банк", "bond", "облигац", "комисси", "invest", "инвест", "crypto risk", "nft")),
        ("cars", ("самокат", "scooter", "micromobility", "микромобил", "suspension", "подвеск", "колес", "repair")),
        ("lifestyle", ("сад", "огород", "почв", "семен", "урож", "agron", "soil", "seed", "harvest", "gardening")),
    ]
    for family, terms in pins:
        if any(t in s for t in terms):
            return family
    return ""


def _infer_subject_hint(*, primary: str, secondary: str, family: str) -> str:
    source = (primary or secondary or "").lower()
    if family == "finance":
        if "nft" in source or "crypto" in source:
            return "crypto investment risk market loss chart"
        if "вклад" in source or "депозит" in source or "deposit" in source:
            return "bank deposit savings documents calculator desk"
        return "finance investment analysis documents calculator"
    if family == "cars" and ("самокат" in source or "scooter" in source):
        return "electric scooter urban repair suspension wheel closeup"
    if family == "lifestyle" and any(x in source for x in ("сад", "почв", "soil", "seed", "harvest", "agron")):
        return "soil seedlings seeds harvest garden agronomy"
    return ""


def _compose_onboarding_context(
    *,
    channel_style: str = "",
    channel_audience: str = "",
    channel_subniche: str = "",
    onboarding_summary: str = "",
    visual_style: str = "",
    post_intent: str = "",
) -> str:
    parts: list[str] = []
    if channel_subniche:
        parts.append(f"Subniche focus: {channel_subniche[:120]}")
    if channel_audience:
        parts.append(f"Audience context: {channel_audience[:140]}")
    if channel_style:
        parts.append(f"Channel tone: {channel_style[:140]}")
    if visual_style:
        parts.append(f"Visual style preference: {visual_style[:120]}")
    if onboarding_summary:
        parts.append(f"Onboarding profile: {onboarding_summary[:180]}")
    if post_intent:
        parts.append(f"Post intent: {post_intent[:100]}")
    return " ".join(parts) if parts else "Brand-safe author-reputation friendly editorial scene"


def _compose_hard_limits(
    *,
    content_constraints: str = "",
    content_exclusions: str = "",
    forbidden_visuals: str = "",
) -> str:
    parts: list[str] = []
    if content_constraints:
        parts.append(content_constraints[:150])
    if content_exclusions:
        parts.append(content_exclusions[:150])
    if forbidden_visuals:
        parts.append(forbidden_visuals[:150])
    if not parts:
        return "no cringe, no absurd scenes, no controversial symbols, no reputational risk visuals"
    return ", ".join(parts)


def _extract_visual_essence(title: str, body_lead: str, family: str, content_mode: str = "") -> str:
    """Distill title and body lead into a concise visual description.

    Mode-aware: uses content_mode to ensure the visual description
    matches the expected scene type.
    """
    # Clean up: remove emojis, special chars, keep meaningful words
    clean_title = re.sub(r"[^\w\s.,!?-]", "", title or "").strip()
    clean_lead = re.sub(r"[^\w\s.,!?-]", "", body_lead or "").strip()

    # Combine and truncate
    combined = f"{clean_title}. {clean_lead}".strip() if clean_lead else clean_title
    # Limit to prevent overly long prompts
    if len(combined) > 300:
        combined = combined[:300].rsplit(" ", 1)[0]

    # Get mode-specific scene hint
    mode_rules = get_mode_image_rules(content_mode) if content_mode else {}
    scene_hint = mode_rules.get("scene_hint", "")

    if scene_hint:
        return f"A professional photograph depicting: {combined}. Context: {scene_hint}"
    return f"A professional photograph depicting: {combined}"


def build_fallback_search_query(
    *,
    title: str = "",
    body: str = "",
    channel_topic: str = "",
    onboarding_summary: str = "",
    content_constraints: str = "",
    post_intent: str = "",
    text_quality_flagged: bool = False,
) -> str:
    """Build a simple English search query for stock photo fallback.

    Returns a clean Latin-only query string suitable for Pexels/Pixabay APIs.
    """
    anchor = _build_visual_topic_anchor(
        title=title,
        body=body,
        channel_topic=channel_topic,
        onboarding_summary=onboarding_summary,
        content_constraints=content_constraints,
        post_intent=post_intent,
        text_quality_flagged=text_quality_flagged,
    )
    family = anchor.family
    combined = " ".join(filter(None, [anchor.primary_text, anchor.secondary_text]))

    # Use subfamily if available
    subfamily = detect_subfamily(family, combined)

    # Map to English search terms
    _FAMILY_SEARCH: dict[str, str] = {
        "massage": "massage therapy spa wellness",
        "food": "food cooking dish meal",
        "health": "health wellness medical",
        "beauty": "beauty skincare cosmetics",
        "cars": "car automotive vehicle",
        "tech": "technology software digital",
        "business": "business office professional",
        "finance": "finance investment money",
        "education": "education learning study",
        "marketing": "marketing branding strategy",
        "lifestyle": "lifestyle modern daily",
        "gaming": "gaming esports controller",
        "hardware": "computer hardware technology",
        "expert_blog": "expert professional workspace",
        "local_business": "small business workshop craft",
    }

    base_query = anchor.subject_hint or _FAMILY_SEARCH.get(family, "professional editorial photo")

    # Add subfamily specifics if available
    if subfamily:
        base_query = f"{subfamily} {base_query}"

    # Strip non-Latin characters for stock photo API compatibility
    clean = re.sub(r"[^a-zA-Z0-9\s]", " ", base_query)
    clean = re.sub(r"\s+", " ", clean).strip()

    logger.debug(
        "IMAGE_FALLBACK_QUERY family=%s subfamily=%s query=%r",
        family, subfamily, clean[:80],
    )

    return clean[:180]
