"""
image_gateway.py — Single canonical entry point for ALL image operations.

Every caller (editor, autopost, news sniper, manual generate) goes through
get_post_image(). There is ONE scoring path, ONE threshold, ONE decision.

Architecture:
  get_post_image()  → find best image for a post (async)
  validate_image()  → validate an existing image against post content (sync)

Principle:
  A wrong image is WORSE than no image.
  If confidence is not high → honest no-image.

Backward-compat exports:
  _LATIN_TOKEN_RE        — regex for Latin token validation (used by actions.py)
  trigger_unsplash_download — no-op stub (Unsplash removed; called by miniapp)
"""
from __future__ import annotations

import logging
import re

from image_pipeline_v3 import (
    run_pipeline_v3,
    validate_image_v3,
    validate_image_post_centric_v3,  # backward compat alias
    PipelineResult,
)
from visual_intent_v2 import extract_visual_intent_v2

logger = logging.getLogger(__name__)

# Canonical mode constants — ALL callers use these
MODE_AUTOPOST = "autopost"
MODE_EDITOR = "editor"

# Backward-compat exports used by actions.py
_LATIN_TOKEN_RE = re.compile(r"^[A-Za-z0-9][\w.+-]*$")


async def trigger_unsplash_download(download_location: str) -> bool:
    """No-op stub. Unsplash removed from pipeline.

    Kept for backward compat — miniapp_routes_content.py calls this.
    """
    return False


async def get_post_image(
    *,
    title: str = "",
    body: str = "",
    channel_topic: str = "",
    used_refs: set[str] | None = None,
    mode: str = MODE_AUTOPOST,
) -> PipelineResult:
    """Find the best image for a post.

    This is the ONLY function for image selection.
    All paths (editor, autopost, news sniper) use this same function
    with the SAME scoring threshold and SAME acceptance logic.

    Mode affects only candidate pool size (editor retrieves more),
    NOT the quality bar.

    Returns PipelineResult. If no confident match → image_url is empty.
    """
    # Mode validation — unknown defaults to autopost (strictest)
    effective_mode = mode if mode in (MODE_AUTOPOST, MODE_EDITOR) else MODE_AUTOPOST

    if not title and not body:
        logger.info(
            "IMAGE_GATEWAY NO_IMAGE reason=no_post_content mode=%s topic=%r",
            effective_mode, (channel_topic or "")[:60],
        )
        return PipelineResult(
            mode=effective_mode,
            no_image_reason="no_post_content",
            outcome="NO_IMAGE",
        )

    result = await run_pipeline_v3(
        title=title,
        body=body,
        channel_topic=channel_topic,
        used_refs=used_refs,
        mode=effective_mode,
    )

    # --- Runtime proof log: every call leaves a clear trail ---
    if result.has_image:
        logger.info(
            "IMAGE_GATEWAY ACCEPT mode=%s score=%d provider=%s "
            "accept_reason=%s url=%r query=%r",
            effective_mode, result.score, result.source_provider,
            result.accept_reason,
            (result.image_url or "")[:80], (result.matched_query or "")[:60],
        )
    else:
        logger.info(
            "IMAGE_GATEWAY NO_IMAGE mode=%s reason=%s "
            "evaluated=%d rejected=%d outcome=%s",
            effective_mode, result.no_image_reason,
            result.candidates_evaluated, result.candidates_rejected,
            result.outcome,
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
    """Validate an existing image against post content.

    Single validation gate. Returns True if acceptable, False otherwise.
    """
    if not image_ref:
        return True
    if not image_ref.startswith("http"):
        return True  # local uploads/telegram files always OK

    if (title or body) and image_meta:
        intent = extract_visual_intent_v2(
            title=title or "",
            body=body or "",
            channel_topic=channel_topic,
        )
        is_valid, reject_reason = validate_image_v3(
            image_ref, intent=intent, image_meta=image_meta,
        )
        if not is_valid:
            logger.warning(
                "IMAGE_GATEWAY VALIDATE_REJECT reason=%s url=%r",
                reject_reason, image_ref[:80],
            )
            return False

    return True
