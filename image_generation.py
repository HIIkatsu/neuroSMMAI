"""
image_generation.py — AI image generation provider integration.

This is generation-first: the primary path is AI image generation,
not stock photo search.

Uses the OpenAI-compatible images.generate API via ai_client.ai_image_generate().
Supports any model accessible through the configured LLM endpoint
(OpenRouter, OpenAI direct, or self-hosted).
"""
from __future__ import annotations

import logging
import os

from ai_client import ai_image_generate

logger = logging.getLogger(__name__)

# Default generation model — can be overridden via IMAGE_GENERATION_MODEL env var
DEFAULT_MODEL = os.getenv("IMAGE_GENERATION_MODEL", "gpt-image-1")
DEFAULT_SIZE = os.getenv("IMAGE_GENERATION_SIZE", "1024x1024")
DEFAULT_QUALITY = os.getenv("IMAGE_GENERATION_QUALITY", "")
DEFAULT_BACKGROUND = os.getenv("IMAGE_GENERATION_BACKGROUND", "")


async def generate_image(
    *,
    api_key: str,
    prompt: str,
    negative_prompt: str = "",
    model: str = "",
    base_url: str | None = None,
    size: str = "",
    quality: str = "",
    background: str = "",
) -> bytes | None:
    """Generate an image using the configured AI provider.

    Returns raw image bytes on success, None on failure.
    Every failure is logged with a clear reason.
    """
    effective_model = model or DEFAULT_MODEL
    effective_size = size or DEFAULT_SIZE
    effective_quality = quality or DEFAULT_QUALITY or None
    effective_background = background or DEFAULT_BACKGROUND or None

    if not api_key:
        logger.warning("IMAGE_GENERATION_SKIP reason=no_api_key")
        return None

    if not prompt or not prompt.strip():
        logger.warning("IMAGE_GENERATION_SKIP reason=empty_prompt")
        return None

    # Compose extra body params if negative prompt is provided
    extra_body: dict | None = None
    if negative_prompt:
        extra_body = {"negative_prompt": negative_prompt}

    logger.info(
        "IMAGE_GENERATION_START model=%s size=%s prompt_len=%d",
        effective_model, effective_size, len(prompt),
    )

    try:
        image_bytes = await ai_image_generate(
            api_key=api_key,
            model=effective_model,
            prompt=prompt,
            base_url=base_url,
            size=effective_size,
            quality=effective_quality,
            background=effective_background,
            extra_body=extra_body,
        )
    except Exception as exc:
        logger.error(
            "IMAGE_GENERATION_FAILED model=%s error_type=%s error=%s",
            effective_model, type(exc).__name__, str(exc)[:300],
        )
        return None

    if image_bytes and len(image_bytes) > 100:
        logger.info(
            "IMAGE_GENERATION_SUCCESS model=%s size=%s bytes=%d",
            effective_model, effective_size, len(image_bytes),
        )
        return image_bytes

    logger.warning(
        "IMAGE_GENERATION_EMPTY model=%s size=%s result_bytes=%d",
        effective_model, effective_size, len(image_bytes) if image_bytes else 0,
    )
    return None
