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
from image_validation import validate_image_bytes, validate_image_url, validate_media_ref
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

    Returns:
        ImageResult with media_ref, source, and metadata
    """
    history = get_image_history()

    # 1. Build prompt
    prompt_data = build_generation_prompt(
        title=title,
        body=body,
        channel_topic=channel_topic,
        llm_image_prompt=llm_image_prompt,
    )
    prompt = prompt_data["prompt"]
    family = prompt_data["family"]

    logger.info(
        "IMAGE_SERVICE_START mode=%s owner_id=%s family=%s title=%r",
        mode, owner_id, family, (title or "")[:60],
    )

    # 2. Check prompt dedup
    if history.is_duplicate_prompt(prompt):
        logger.info("IMAGE_SERVICE_PROMPT_DEDUP prompt_sig=duplicate, proceeding with generation anyway")
        # We still generate — dedup is advisory for prompts, not blocking

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
        owner_id, family, mode,
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
) -> str:
    """Attempt stock photo fallback. Returns image URL or empty string."""
    query = build_fallback_search_query(
        title=title,
        body=body,
        channel_topic=channel_topic,
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
        # Still return — better than nothing
        return url

    # Check against used_refs
    if used_refs and url in used_refs:
        logger.info("IMAGE_FALLBACK_ALREADY_USED url=%r", url[:60])
        # Still return — better than nothing for fallback

    return url
