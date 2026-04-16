"""
image_service.py — Single public entry point for all image generation requests.

Generation flow:
  caller → image_service → prompt build → image generation → validation → dedup check → storage → result

Fallback flow (only if generation fails):
  caller → image_service → generation failed → fallback search → validation → storage → result

This module is the ONLY module external callers should import for image operations.
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass

from image_prompts import build_generation_prompt, build_fallback_search_query
from image_generation import generate_image
from image_validation import (
    validate_image_bytes,
    validate_image_url,
    validate_media_ref,
    validate_image_candidate,
)
from image_storage import save_generated_image
from image_history import get_image_history
from image_fallback import search_stock_photo

logger = logging.getLogger(__name__)

# Mode constants — callers use these
MODE_AUTOPOST = "autopost"
MODE_EDITOR = "editor"

# Latin token regex — used by actions.py for query cleaning
_LATIN_TOKEN_RE = re.compile(r"^[A-Za-z0-9][\w.+-]*$")

# Generation timeout
_GENERATION_TIMEOUT = float(30.0)
_FALLBACK_TIMEOUT = float(15.0)


@dataclass
class ImageResult:
    """Result of an image generation/search request."""
    media_ref: str = ""           # The usable media reference (URL or local path)
    source: str = ""              # "generation", "fallback", "none"
    prompt_used: str = ""         # The prompt that was used
    family: str = ""              # Detected topic family
    failure_reason: str = ""      # Why it failed (if it did)
    is_generated: bool = False    # True if AI-generated, False if stock/fallback


async def get_image(
    *,
    title: str = "",
    body: str = "",
    channel_topic: str = "",
    llm_image_prompt: str = "",
    api_key: str = "",
    model: str = "",
    base_url: str | None = None,
    owner_id: int | None = None,
    mode: str = MODE_EDITOR,
    used_refs: set[str] | None = None,
    text_quality_flagged: bool = False,
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
) -> ImageResult:
    """Generate or find an image for a post.

    This is the SINGLE entry point for all image operations.
    Generation-first: tries AI generation, falls back to stock search only on failure.

    Args:
        title: Post title
        body: Post body text
        channel_topic: Channel topic for context
        llm_image_prompt: Pre-built LLM prompt for image generation
        api_key: API key for the image generation provider
        model: Model name override
        base_url: API base URL override
        owner_id: Owner ID for storage and access control
        mode: "editor" or "autopost"
        used_refs: Set of recently used image refs to avoid
        content_mode: Detected content mode for mode-aware prompt building

    Returns:
        ImageResult with media_ref, source, and metadata
    """
    normalized_mode = _normalize_mode(mode)
    history = get_image_history()

    # 1. Build prompt
    prompt_data = build_generation_prompt(
        title=title,
        body=body,
        channel_topic=channel_topic,
        llm_image_prompt=llm_image_prompt,
        content_mode=content_mode,
        channel_style=channel_style,
        channel_audience=channel_audience,
        channel_subniche=channel_subniche,
        onboarding_summary=onboarding_summary,
        content_constraints=content_constraints,
        content_exclusions=content_exclusions,
        visual_style=visual_style,
        forbidden_visuals=forbidden_visuals,
        post_intent=post_intent,
        text_quality_flagged=text_quality_flagged,
    )
    prompt = prompt_data["prompt"]
    family = prompt_data["family"]
    effective_mode = prompt_data["content_mode"]

    logger.info(
        "IMAGE_SERVICE_START mode=%s owner_id=%s family=%s title=%r",
        normalized_mode, owner_id, family, (title or "")[:60],
    )

    prompt_ok, prompt_reason = validate_image_candidate(
        prompt=prompt,
        title=title,
        body=body,
        channel_topic=channel_topic,
        content_mode=effective_mode,
        canonical_family=family,
    )
    if not prompt_ok:
        logger.warning("IMAGE_SERVICE_PROMPT_REJECT reason=%s title=%r", prompt_reason, (title or "")[:60])
        prompt_data = build_generation_prompt(
            title=title,
            body=body,
            channel_topic=channel_topic,
            content_mode=content_mode,
            channel_style=channel_style,
            channel_audience=channel_audience,
            channel_subniche=channel_subniche,
            onboarding_summary=onboarding_summary,
            content_constraints=content_constraints,
            content_exclusions=content_exclusions,
            visual_style=visual_style,
            forbidden_visuals=forbidden_visuals,
            post_intent=post_intent,
            text_quality_flagged=text_quality_flagged,
        )
        prompt = prompt_data["prompt"]

    # 2. Check prompt dedup
    if history.is_duplicate_prompt(prompt):
        logger.info("IMAGE_SERVICE_PROMPT_DEDUP prompt_sig=duplicate, proceeding with generation anyway")
        # We still generate — dedup is advisory for prompts, not blocking

    generated_ref = ""
    if history.is_duplicate_visual_pattern(prompt):
        logger.info("IMAGE_SERVICE_PATTERN_DEDUP owner_id=%s", owner_id)
    else:
        # 3. Try AI generation (primary path)
        generated_ref = await _try_generation(
            prompt=prompt,
            negative_prompt=prompt_data["negative_prompt"],
            api_key=api_key,
            model=model,
            base_url=base_url,
            owner_id=owner_id,
            history=history,
            used_refs=used_refs,
        )

    if generated_ref:
        logger.info(
            "IMAGE_SERVICE_SUCCESS source=generation ref=%r family=%s owner_id=%s",
            generated_ref[:80], family, owner_id,
        )
        return ImageResult(
            media_ref=generated_ref,
            source="generation",
            prompt_used=prompt[:200],
            family=family,
            is_generated=True,
        )

    # 4. Fallback to stock photo search
    logger.info("IMAGE_SERVICE_GENERATION_FAILED, trying fallback owner_id=%s", owner_id)

    fallback_ref = await _try_fallback(
        title=title,
        body=body,
        channel_topic=channel_topic,
        history=history,
        used_refs=used_refs,
        text_quality_flagged=text_quality_flagged,
        mode=normalized_mode,
        family_context_hint=" ".join(
            x for x in [
                channel_topic,
                onboarding_summary,
                post_intent,
                title,
            ] if (x or "").strip()
        ),
        content_mode=effective_mode,
        normalized_visual_prompt="" if text_quality_flagged else prompt,
        resolved_intent=post_intent,
        canonical_family=family,
        onboarding_summary=onboarding_summary,
        content_constraints=content_constraints,
    )

    if fallback_ref:
        logger.info(
            "IMAGE_SERVICE_SUCCESS source=fallback ref=%r family=%s owner_id=%s",
            fallback_ref[:80], family, owner_id,
        )
        # Record fallback result in history
        history.record(media_ref=fallback_ref)
        return ImageResult(
            media_ref=fallback_ref,
            source="fallback",
            prompt_used=prompt[:200],
            family=family,
            is_generated=False,
        )

    # 5. Complete failure
    logger.warning(
        "IMAGE_SERVICE_NO_IMAGE owner_id=%s family=%s mode=%s",
        owner_id, family, normalized_mode,
    )
    return ImageResult(
        source="none",
        prompt_used=prompt[:200],
        family=family,
        failure_reason="generation_and_fallback_both_failed",
    )


async def validate_image(
    media_ref: str,
    *,
    title: str = "",
    body: str = "",
    channel_topic: str = "",
    mode: str = MODE_AUTOPOST,
) -> bool:
    """Validate an existing image reference.

    Simple validation: checks if the ref is plausible and non-empty.
    Used by scheduler_service.py as a quality gate.
    """
    if not media_ref or not media_ref.strip():
        return True  # Empty ref = no image = valid (caller decides if image is required)

    is_valid, reason = validate_media_ref(media_ref)
    if not is_valid:
        logger.info(
            "IMAGE_VALIDATE_REJECT ref=%r reason=%s mode=%s",
            media_ref[:80], reason, mode,
        )
    return is_valid


async def trigger_unsplash_download(download_location: str) -> bool:
    """No-op stub: Unsplash integration has been removed.

    This stub exists ONLY because miniapp_routes_content.py calls it on
    draft create/update/publish. Removing it would break those callers.
    The stub always returns False and logs explicitly.
    """
    logger.debug(
        "IMAGE_UNSPLASH_NOOP download_location=%r reason=unsplash_removed",
        (download_location or "")[:80],
    )
    return False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

async def _try_generation(
    *,
    prompt: str,
    negative_prompt: str,
    api_key: str,
    model: str,
    base_url: str | None,
    owner_id: int | None,
    history,
    used_refs: set[str] | None,
) -> str:
    """Attempt AI image generation. Returns media ref or empty string."""
    if not _is_valid_image_model(model):
        logger.warning(
            "IMAGE_GENERATION_SKIPPED_INVALID_MODEL model=%r owner_id=%s",
            (model or "")[:120], owner_id,
        )
        return ""
    try:
        image_bytes = await asyncio.wait_for(
            generate_image(
                api_key=api_key,
                prompt=prompt,
                negative_prompt=negative_prompt,
                model=model,
                base_url=base_url,
            ),
            timeout=_GENERATION_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.warning("IMAGE_GENERATION_TIMEOUT owner_id=%s", owner_id)
        return ""
    except Exception as exc:
        logger.error("IMAGE_GENERATION_ERROR owner_id=%s error=%s", owner_id, exc)
        return ""

    if not image_bytes:
        return ""

    # Validate generated bytes
    is_valid, reason = validate_image_bytes(image_bytes)
    if not is_valid:
        logger.warning("IMAGE_GENERATION_INVALID reason=%s owner_id=%s", reason, owner_id)
        return ""

    # Check content dedup
    if history.is_duplicate_content(image_bytes):
        logger.info("IMAGE_GENERATION_CONTENT_DEDUP owner_id=%s", owner_id)
        # Don't reject — duplicates from generation are unlikely but possible
        # Just log and continue

    # Save to storage
    media_ref = save_generated_image(image_bytes, owner_id=owner_id, prompt_hint=prompt[:50])
    if not media_ref:
        logger.error("IMAGE_STORAGE_FAILED owner_id=%s", owner_id)
        return ""

    # Check against used_refs
    if used_refs and media_ref in used_refs:
        logger.info("IMAGE_GENERATION_ALREADY_USED ref=%r owner_id=%s", media_ref[:80], owner_id)
        # Still return it — exact hash collision with used refs is extremely unlikely
        # for generated images

    # Record in history
    history.record(image_bytes=image_bytes, prompt=prompt, media_ref=media_ref)

    return media_ref


async def _try_fallback(
    *,
    title: str,
    body: str,
    channel_topic: str,
    history,
    used_refs: set[str] | None,
    text_quality_flagged: bool = False,
    mode: str = MODE_EDITOR,
    family_context_hint: str = "",
    content_mode: str = "",
    normalized_visual_prompt: str = "",
    resolved_intent: str = "",
    canonical_family: str = "",
    onboarding_summary: str = "",
    content_constraints: str = "",
) -> str:
    """Attempt stock photo fallback. Returns image URL or empty string."""
    query = build_fallback_search_query(
        title=title,
        body=body,
        channel_topic=channel_topic,
        onboarding_summary=onboarding_summary,
        content_constraints=content_constraints,
        post_intent=resolved_intent,
        text_quality_flagged=text_quality_flagged,
    )

    if not query:
        logger.warning("IMAGE_FALLBACK_NO_QUERY")
        return ""

    try:
        url = await asyncio.wait_for(
            search_stock_photo(query),
            timeout=_FALLBACK_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.warning("IMAGE_FALLBACK_TIMEOUT query=%r", query[:40])
        return ""
    except Exception as exc:
        logger.error("IMAGE_FALLBACK_ERROR query=%r error=%s", query[:40], exc)
        return ""

    if not url:
        return ""

    # Validate URL
    is_valid, reason = validate_image_url(url)
    if not is_valid:
        logger.warning("IMAGE_FALLBACK_INVALID_URL url=%r reason=%s", url[:60], reason)
        return ""

    # Check dedup
    if history.is_duplicate_ref(url):
        logger.info("IMAGE_FALLBACK_DEDUP_HIT url=%r", url[:60])
        return ""

    # Check against used_refs
    if used_refs and url in used_refs:
        logger.info("IMAGE_FALLBACK_ALREADY_USED url=%r", url[:60])
        return ""

    stable_visual_context = " ".join(
        x.strip()
        for x in [normalized_visual_prompt, title, channel_topic, resolved_intent, query]
        if (x or "").strip()
    )
    candidate_ok, candidate_reason = validate_image_candidate(
        prompt=stable_visual_context or query,
        title=title,
        body=body,
        channel_topic=channel_topic,
        family_context_hint=family_context_hint,
        content_mode=content_mode,
        media_ref=url,
        allow_family_mismatch_penalty=(text_quality_flagged or mode == MODE_EDITOR),
        enforce_min_prompt_len=False,
        ignore_body_for_family_context=text_quality_flagged,
        canonical_family=canonical_family,
    )
    if not candidate_ok:
        logger.warning("IMAGE_FALLBACK_REJECT reason=%s url=%r", candidate_reason, url[:80])
        return ""
    if "family_mismatch_penalty_" in candidate_reason:
        logger.info("IMAGE_FALLBACK_PENALTY reason=%s url=%r", candidate_reason, url[:80])

    return url


def _normalize_mode(mode: str) -> str:
    m = (mode or "").strip().lower()
    if m in {MODE_AUTOPOST, "news", "auto"}:
        return MODE_AUTOPOST
    return MODE_EDITOR


def _is_valid_image_model(model: str) -> bool:
    m = (model or "").strip().lower()
    if not m:
        # Empty model is allowed — generate_image() will use IMAGE_GENERATION_MODEL/default.
        return True
    # Conservative allow-list: common image generation families/providers.
    image_tokens = (
        "image", "flux", "recraft", "dall", "sd", "stable-diffusion", "midjourney", "imagen",
    )
    text_only_tokens = (
        "mistral", "claude", "llama", "deepseek", "qwen", "gemini", "gpt-4", "gpt-3", "command-r",
    )
    if any(tok in m for tok in image_tokens):
        return True
    if any(tok in m for tok in text_only_tokens):
        return False
    return False
