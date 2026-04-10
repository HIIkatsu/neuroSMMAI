from __future__ import annotations

"""
image_search.py — Backward-compatible shim for the unified image gateway.

All real logic lives in:
  - image_gateway.py (single entry point)
  - image_pipeline_v3.py (orchestrator)
  - image_providers.py (Pexels, Pixabay, Openverse)
  - image_ranker.py (scoring)
  - image_history.py (anti-repeat)
  - visual_intent_v2.py (intent extraction)

This module preserves old function signatures that existing callers expect,
but delegates everything to the unified gateway. No legacy search logic remains.
"""

import logging
import re

from topic_utils import (
    TOPIC_FAMILY_TERMS,
    detect_topic_family,
    detect_subfamily,
    classify_visual_type,
    STRICT_IMAGE_FAMILIES,
    IRRELEVANT_IMAGE_CLASSES,
    VISUAL_TYPE_TEXT_ONLY,
    get_family_image_queries,
    get_subfamily_image_queries,
    get_family_irrelevant_classes,
    get_family_allowed_visuals,
    get_family_blocked_visuals,
    get_family_bad_url_signals,
)

logger = logging.getLogger(__name__)


# Re-exported constants and helpers used by actions.py and others
_LATIN_TOKEN_RE = re.compile(r"^[A-Za-z0-9][\w.+-]*$")


# ---------------------------------------------------------------------------
# Backward-compatible find_image() — delegates to image_gateway
# ---------------------------------------------------------------------------
async def find_image(
    query: str,
    used_refs: set[str] | None = None,
    *,
    topic: str = "",
    post_text: str = "",
    title: str = "",
    mode: str = "",
) -> str:
    """Find the best image for a post (backward-compatible wrapper).

    Delegates to image_gateway.get_post_image().
    """
    from image_gateway import get_post_image, MODE_AUTOPOST, MODE_EDITOR

    effective_mode = mode if mode in (MODE_AUTOPOST, MODE_EDITOR) else MODE_AUTOPOST
    effective_title = title or query or ""
    effective_body = post_text or ""

    result = await get_post_image(
        title=effective_title,
        body=effective_body,
        channel_topic=topic,
        used_refs=used_refs,
        mode=effective_mode,
    )

    if result.has_image:
        logger.info(
            "IMAGE_SEARCH_ACCEPT url=%r score=%d source=%s mode=%s",
            (result.image_url or "")[:80], result.score,
            result.source_provider, effective_mode,
        )
        return result.image_url

    logger.info(
        "IMAGE_SEARCH_NO_MATCH reason=%s outcome=%s mode=%s",
        result.no_image_reason, result.outcome, effective_mode,
    )
    return ""


# ---------------------------------------------------------------------------
# Backward-compatible validate_image_for_autopost()
# ---------------------------------------------------------------------------
def validate_image_for_autopost(
    image_ref: str,
    *,
    topic: str = "",
    prompt: str = "",
    post_text: str = "",
    image_meta: str = "",
    title: str = "",
    mode: str = "",
) -> bool:
    """Quality gate for images (backward-compatible wrapper).

    Delegates to image_gateway.validate_image().
    """
    from image_gateway import validate_image

    return validate_image(
        image_ref,
        title=title or prompt or "",
        body=post_text,
        channel_topic=topic,
        image_meta=image_meta,
        mode=mode or "autopost",
    )


# ---------------------------------------------------------------------------
# Legacy query building — kept for backward compatibility
# ---------------------------------------------------------------------------
def build_best_visual_queries(query: str) -> list[str]:
    """Build search queries from a text query (legacy compatibility).

    Still used by some callers that need raw query construction.
    Returns English search terms suitable for stock photo APIs.
    """
    from visual_intent_v2 import extract_visual_intent_v2

    intent = extract_visual_intent_v2(
        title=query,
        body="",
        channel_topic="",
    )
    if intent.query_terms:
        return intent.query_terms[:10]

    # Fallback: basic family-based queries
    family = detect_topic_family(query)
    return get_family_image_queries(family)[:5]


def build_visual_fallback(query: str) -> str:
    """Single-query fallback (legacy compatibility)."""
    queries = build_best_visual_queries(query)
    return queries[0] if queries else ""


# ---------------------------------------------------------------------------
# Unsplash download trigger — kept as no-op stub
# ---------------------------------------------------------------------------
async def trigger_unsplash_download(download_location: str) -> bool:
    """Trigger Unsplash download (no-op: Unsplash removed from pipeline).

    Kept for backward compatibility — miniapp_routes_content.py calls this
    when saving drafts. Returns False since no download is needed.
    """
    return False

