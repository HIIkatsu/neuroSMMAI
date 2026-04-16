"""
image_fallback.py — Stock photo fallback when AI generation fails.

This module is ONLY used when generation fails completely.
It is NOT the main image path — see image_generation.py for that.

Searches Pexels and Pixabay directly via their public APIs.
"""
from __future__ import annotations

import logging
import os
import httpx

from visual_profile_layer import ProviderCandidate

logger = logging.getLogger(__name__)

PEXELS_API_KEY = (os.getenv("PEXELS_API_KEY") or "").strip()
PIXABAY_API_KEY = (os.getenv("PIXABAY_API_KEY") or "").strip()

_CONNECT_TIMEOUT = 5.0
_READ_TIMEOUT = 10.0

async def search_stock_photo(query: str) -> str:
    """Search stock photo providers for a relevant image.

    Returns image URL on success, empty string on failure.
    Tries Pexels first, then Pixabay.
    """
    if not query or not query.strip():
        logger.warning("IMAGE_FALLBACK_SKIP reason=empty_query")
        return ""

    # Try Pexels
    if PEXELS_API_KEY:
        urls = await _search_pexels(query, limit=1)
        if urls:
            logger.info("IMAGE_FALLBACK_SUCCESS provider=pexels query=%r", query[:60])
            return urls[0].url

    # Try Pixabay
    if PIXABAY_API_KEY:
        urls = await _search_pixabay(query, limit=1)
        if urls:
            logger.info("IMAGE_FALLBACK_SUCCESS provider=pixabay query=%r", query[:60])
            return urls[0].url

    logger.warning(
        "IMAGE_FALLBACK_EXHAUSTED query=%r pexels_key=%s pixabay_key=%s",
        query[:60], bool(PEXELS_API_KEY), bool(PIXABAY_API_KEY),
    )
    return ""


async def search_stock_candidates(query: str, *, query_family: str = "primary") -> list[ProviderCandidate]:
    """Return ordered normalized candidates from stock providers (Pexels → Pixabay)."""
    candidates: list[ProviderCandidate] = []
    if not query or not query.strip():
        return candidates

    if PEXELS_API_KEY:
        for candidate in await _search_pexels(query, limit=3):
            candidate.query_family = query_family
            candidate.source_query = query
            candidates.append(candidate)
    if PIXABAY_API_KEY:
        for candidate in await _search_pixabay(query, limit=3):
            candidate.query_family = query_family
            candidate.source_query = query
            candidates.append(candidate)
    return candidates


async def _search_pexels(query: str, *, limit: int = 1) -> list[ProviderCandidate]:
    """Search Pexels and return normalized candidates."""
    try:
        timeout = httpx.Timeout(connect=_CONNECT_TIMEOUT, read=_READ_TIMEOUT, write=_READ_TIMEOUT, pool=_READ_TIMEOUT)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            r = await client.get(
                "https://api.pexels.com/v1/search",
                params={"query": query, "per_page": 5, "orientation": "landscape"},
                headers={"Authorization": PEXELS_API_KEY},
            )
            r.raise_for_status()
            data = r.json()

        candidates: list[ProviderCandidate] = []
        for photo in data.get("photos") or []:
            src = photo.get("src") or {}
            url = ""
            for key in ("large2x", "large", "original"):
                maybe_url = src.get(key, "")
                if maybe_url and maybe_url.startswith("http"):
                    url = maybe_url
                    break
            if not url:
                continue
            candidates.append(ProviderCandidate(
                url=url,
                provider="pexels",
                caption=str(photo.get("alt") or ""),
                tags=_extract_tags(str(photo.get("alt") or "")),
                author=str((photo.get("photographer") or "")),
                width=int(photo.get("width") or 0),
                height=int(photo.get("height") or 0),
            ))
            if len(candidates) >= max(1, int(limit)):
                break
        return candidates
    except Exception as exc:
        logger.warning("IMAGE_FALLBACK_PEXELS_FAILED query=%r error=%s", query[:40], exc)
    return []


async def _search_pixabay(query: str, *, limit: int = 1) -> list[ProviderCandidate]:
    """Search Pixabay and return normalized candidates."""
    try:
        timeout = httpx.Timeout(connect=_CONNECT_TIMEOUT, read=_READ_TIMEOUT, write=_READ_TIMEOUT, pool=_READ_TIMEOUT)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            r = await client.get(
                "https://pixabay.com/api/",
                params={
                    "key": PIXABAY_API_KEY,
                    "q": query,
                    "image_type": "photo",
                    "orientation": "horizontal",
                    "per_page": 5,
                    "safesearch": "true",
                    "min_width": 800,
                },
            )
            r.raise_for_status()
            data = r.json()

        candidates: list[ProviderCandidate] = []
        for hit in data.get("hits") or []:
            url = hit.get("largeImageURL") or hit.get("webformatURL") or ""
            if not (url and url.startswith("http")):
                continue
            tags_text = str(hit.get("tags") or "")
            candidates.append(ProviderCandidate(
                url=url,
                provider="pixabay",
                caption=tags_text,
                tags=_extract_tags(tags_text),
                author=str(hit.get("user") or ""),
                width=int(hit.get("imageWidth") or 0),
                height=int(hit.get("imageHeight") or 0),
            ))
            if len(candidates) >= max(1, int(limit)):
                break
        return candidates
    except Exception as exc:
        logger.warning("IMAGE_FALLBACK_PIXABAY_FAILED query=%r error=%s", query[:40], exc)
    return []
