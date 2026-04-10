"""
image_gateway.py — Unified image gateway for NeuroSMMAI.

Single entry point for ALL image operations across editor, autopost,
news sniper, and manual generation paths.

Architecture:
  get_post_image()  → single async function that every caller uses
  validate_image()  → single validation function

All callers MUST provide:
  - title and/or body (post text)
  - mode ("editor" or "autopost")

The gateway delegates to:
  - image_pipeline_v3.run_pipeline_v3() for candidate search + scoring
  - image_pipeline_v3.validate_image_post_centric_v3() for validation

No legacy fallback. No Unsplash. No dual scoring.
If v3 finds nothing → honest empty string ("no image is better than junk").
"""
from __future__ import annotations

import logging

from image_pipeline_v3 import (
    run_pipeline_v3,
    validate_image_post_centric_v3,
    PipelineResult,
    MODE_AUTOPOST,
    MODE_EDITOR,
)
from visual_intent_v2 import extract_visual_intent_v2

logger = logging.getLogger(__name__)


async def get_post_image(
    *,
    title: str = "",
    body: str = "",
    channel_topic: str = "",
    used_refs: set[str] | None = None,
    mode: str = MODE_AUTOPOST,
) -> PipelineResult:
    """Find the best image for a post.

    This is the ONLY function that should be called for image selection.
    All paths (editor, autopost, news sniper) use this same function.

    Args:
        title: Post title (required for good results).
        body: Post body text.
        channel_topic: Channel topic (weak fallback only).
        used_refs: URLs already used (for deduplication).
        mode: "editor" (lenient, returns multiple candidates) or
              "autopost" (strict, prefers no-image over junk).

    Returns:
        PipelineResult with image_url, score, candidates, etc.
    """
    effective_mode = mode if mode in (MODE_AUTOPOST, MODE_EDITOR) else MODE_AUTOPOST

    if not title and not body:
        # No post content → cannot determine visual intent
        logger.info(
            "IMAGE_GATEWAY_SKIP reason=no_post_content mode=%s topic=%r",
            effective_mode, (channel_topic or "")[:60],
        )
        return PipelineResult(
            mode=effective_mode,
            no_image_reason="no_post_content",
            outcome="NO_IMAGE_NO_CONTENT",
        )

    result = await run_pipeline_v3(
        title=title,
        body=body,
        channel_topic=channel_topic,
        used_refs=used_refs,
        mode=effective_mode,
    )

    if result.has_image:
        logger.info(
            "IMAGE_GATEWAY_ACCEPT mode=%s score=%d provider=%s url=%r query=%r",
            effective_mode, result.score, result.source_provider,
            (result.image_url or "")[:80], (result.matched_query or "")[:60],
        )
    else:
        logger.info(
            "IMAGE_GATEWAY_NO_IMAGE mode=%s reason=%s outcome=%s "
            "evaluated=%d rejected=%d",
            effective_mode, result.no_image_reason, result.outcome,
            result.candidates_evaluated, result.candidates_rejected,
        )

    return result


def validate_image(
    image_ref: str,
    *,
    title: str = "",
    body: str = "",
    channel_topic: str = "",
    image_meta: str = "",
    mode: str = MODE_AUTOPOST,
) -> bool:
    """Validate an image against post content.

    Single validation gate — no legacy checks, no dual scoring.

    Returns True if the image is acceptable, False if it should be rejected.
    """
    if not image_ref:
        return True  # No image to reject

    # Local uploads and telegram files are always acceptable
    if not image_ref.startswith("http"):
        return True

    effective_mode = mode if mode in (MODE_AUTOPOST, MODE_EDITOR) else MODE_AUTOPOST

    # Need post content + image metadata for post-centric validation
    if (title or body) and image_meta:
        intent = extract_visual_intent_v2(
            title=title or "",
            body=body or "",
            channel_topic=channel_topic,
        )
        is_valid, reject_reason = validate_image_post_centric_v3(
            image_ref,
            intent=intent,
            image_meta=image_meta,
            mode=effective_mode,
        )
        if not is_valid:
            logger.warning(
                "IMAGE_GATEWAY_VALIDATE_REJECT reason=%s mode=%s url=%r",
                reject_reason, effective_mode, image_ref[:80],
            )
            return False

    return True
