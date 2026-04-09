"""
preview_resolver.py — Stable media preview resolution for miniapp.

Centralizes all media reference resolution logic for preview rendering.
Handles:
  - /uploads/ paths (local uploads)
  - /generated-images/ paths (AI-generated)
  - /generated_images/ paths (legacy underscore variant)
  - tgfile: protocol (Telegram file_id proxy)
  - External URLs (Pexels/Pixabay/Unsplash)
  - Auth token injection
  - Stale reference detection
  - Query param preservation for external URLs

Production logs:
  PREVIEW_MEDIA_RESOLVE_OK
  PREVIEW_MEDIA_RESOLVE_FAIL
  PREVIEW_MEDIA_AUTH_FAIL
  PREVIEW_MEDIA_STALE_REF
  PREVIEW_MEDIA_QUERY_STRIPPED
  PREVIEW_MEDIA_RENDER_PATH=...
"""
from __future__ import annotations

import logging
import os
import re
from urllib.parse import urlparse, urlencode, parse_qs, urlunparse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path categories
# ---------------------------------------------------------------------------
RENDER_PATH_UPLOAD = "upload"
RENDER_PATH_GENERATED = "generated"
RENDER_PATH_TELEGRAM = "telegram"
RENDER_PATH_EXTERNAL = "external"
RENDER_PATH_UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Resolution result
# ---------------------------------------------------------------------------
class PreviewResolveResult:
    """Structured result of preview media resolution."""
    __slots__ = (
        "resolved_url", "render_path", "needs_auth", "is_external",
        "is_stale", "error", "original_ref",
    )

    def __init__(
        self,
        resolved_url: str = "",
        render_path: str = RENDER_PATH_UNKNOWN,
        needs_auth: bool = False,
        is_external: bool = False,
        is_stale: bool = False,
        error: str = "",
        original_ref: str = "",
    ):
        self.resolved_url = resolved_url
        self.render_path = render_path
        self.needs_auth = needs_auth
        self.is_external = is_external
        self.is_stale = is_stale
        self.error = error
        self.original_ref = original_ref

    @property
    def ok(self) -> bool:
        return bool(self.resolved_url) and not self.error

    def as_log_dict(self) -> dict:
        return {
            "resolved_url": (self.resolved_url or "")[:120],
            "render_path": self.render_path,
            "needs_auth": self.needs_auth,
            "is_external": self.is_external,
            "is_stale": self.is_stale,
            "error": self.error,
            "original_ref": (self.original_ref or "")[:80],
        }


# ---------------------------------------------------------------------------
# Core resolution
# ---------------------------------------------------------------------------

def resolve_preview_media(
    media_ref: str,
    *,
    auth_token: str = "",
    base_url: str = "",
    check_file_exists: bool = False,
) -> PreviewResolveResult:
    """Resolve a media reference into a renderable preview URL.

    Parameters:
        media_ref: Raw media reference from draft payload
        auth_token: Telegram WebApp init data for auth
        base_url: Base URL prefix for relative paths
        check_file_exists: If True, check if local file exists on disk

    Returns:
        PreviewResolveResult with resolved URL and metadata
    """
    result = PreviewResolveResult(original_ref=media_ref or "")
    raw = (media_ref or "").strip()

    if not raw:
        result.error = "empty_ref"
        logger.warning("PREVIEW_MEDIA_RESOLVE_FAIL reason=empty_ref")
        return result

    # --- tgfile: protocol ---
    if raw.startswith("tgfile:"):
        return _resolve_tgfile(raw, auth_token=auth_token, base_url=base_url)

    # --- /uploads/ paths ---
    if "/uploads/" in raw:
        return _resolve_upload(raw, auth_token=auth_token, base_url=base_url,
                               check_file_exists=check_file_exists)

    # --- /generated_images/ (legacy underscore) → normalize to /generated-images/ ---
    if "/generated_images/" in raw:
        normalized = raw.replace("/generated_images/", "/generated-images/")
        logger.info(
            "PREVIEW_MEDIA_QUERY_STRIPPED legacy_underscore_normalized from=%r to=%r",
            raw[:80], normalized[:80],
        )
        return _resolve_generated(normalized, auth_token=auth_token, base_url=base_url,
                                  check_file_exists=check_file_exists)

    # --- /generated-images/ ---
    if "/generated-images/" in raw:
        return _resolve_generated(raw, auth_token=auth_token, base_url=base_url,
                                  check_file_exists=check_file_exists)

    # --- /api/media/telegram ---
    if raw.startswith("/api/media/telegram"):
        return _resolve_telegram_api(raw, auth_token=auth_token, base_url=base_url)

    # --- External URL (http/https) ---
    if raw.startswith("http://") or raw.startswith("https://"):
        return _resolve_external(raw)

    # --- Unknown path ---
    result.resolved_url = raw
    result.render_path = RENDER_PATH_UNKNOWN
    result.error = "unknown_ref_format"
    logger.warning("PREVIEW_MEDIA_RESOLVE_FAIL reason=unknown_ref_format ref=%r", raw[:80])
    return result


# ---------------------------------------------------------------------------
# Per-type resolvers
# ---------------------------------------------------------------------------

def _resolve_tgfile(
    raw: str, *, auth_token: str, base_url: str,
) -> PreviewResolveResult:
    parts = raw.split(":")
    kind = parts[1] if len(parts) > 1 else "photo"
    file_id = ":".join(parts[2:]) if len(parts) > 2 else ""

    if not file_id:
        logger.warning("PREVIEW_MEDIA_RESOLVE_FAIL reason=tgfile_no_file_id ref=%r", raw[:80])
        return PreviewResolveResult(
            original_ref=raw, error="tgfile_no_file_id", render_path=RENDER_PATH_TELEGRAM,
        )

    from urllib.parse import quote
    path = f"/api/media/telegram?kind={quote(kind)}&file_id={quote(file_id)}"
    url = _inject_auth(base_url + path, auth_token)

    logger.info("PREVIEW_MEDIA_RESOLVE_OK path=telegram ref=%r", raw[:60])
    return PreviewResolveResult(
        resolved_url=url, render_path=RENDER_PATH_TELEGRAM,
        needs_auth=True, original_ref=raw,
    )


def _resolve_upload(
    raw: str, *, auth_token: str, base_url: str, check_file_exists: bool,
) -> PreviewResolveResult:
    # Extract the path after /uploads/
    idx = raw.index("/uploads/")
    rel_path = raw[idx:]
    # Strip any existing query params from the path itself
    clean_path = rel_path.split("?")[0]

    if check_file_exists:
        disk_path = os.path.join(os.getcwd(), clean_path.lstrip("/"))
        if not os.path.isfile(disk_path):
            logger.warning("PREVIEW_MEDIA_STALE_REF path=%r disk=%r", clean_path, disk_path)
            return PreviewResolveResult(
                original_ref=raw, is_stale=True, error="file_not_found",
                render_path=RENDER_PATH_UPLOAD,
            )

    url = _inject_auth(base_url + clean_path, auth_token)
    logger.info("PREVIEW_MEDIA_RESOLVE_OK path=upload ref=%r", raw[:60])
    logger.info("PREVIEW_MEDIA_RENDER_PATH=%s", RENDER_PATH_UPLOAD)
    return PreviewResolveResult(
        resolved_url=url, render_path=RENDER_PATH_UPLOAD,
        needs_auth=True, original_ref=raw,
    )


def _resolve_generated(
    raw: str, *, auth_token: str, base_url: str, check_file_exists: bool,
) -> PreviewResolveResult:
    idx = raw.index("/generated-images/")
    rel_path = raw[idx:]
    clean_path = rel_path.split("?")[0]

    if check_file_exists:
        disk_path = os.path.join(os.getcwd(), clean_path.lstrip("/"))
        if not os.path.isfile(disk_path):
            logger.warning("PREVIEW_MEDIA_STALE_REF path=%r disk=%r", clean_path, disk_path)
            return PreviewResolveResult(
                original_ref=raw, is_stale=True, error="file_not_found",
                render_path=RENDER_PATH_GENERATED,
            )

    url = _inject_auth(base_url + clean_path, auth_token)
    logger.info("PREVIEW_MEDIA_RESOLVE_OK path=generated ref=%r", raw[:60])
    logger.info("PREVIEW_MEDIA_RENDER_PATH=%s", RENDER_PATH_GENERATED)
    return PreviewResolveResult(
        resolved_url=url, render_path=RENDER_PATH_GENERATED,
        needs_auth=True, original_ref=raw,
    )


def _resolve_telegram_api(
    raw: str, *, auth_token: str, base_url: str,
) -> PreviewResolveResult:
    url = _inject_auth(base_url + raw, auth_token)
    logger.info("PREVIEW_MEDIA_RESOLVE_OK path=telegram_api ref=%r", raw[:60])
    return PreviewResolveResult(
        resolved_url=url, render_path=RENDER_PATH_TELEGRAM,
        needs_auth=True, original_ref=raw,
    )


def _resolve_external(raw: str) -> PreviewResolveResult:
    """Resolve external URL — preserve query params (critical for CDN auth)."""
    logger.info("PREVIEW_MEDIA_RESOLVE_OK path=external ref=%r", raw[:80])
    logger.info("PREVIEW_MEDIA_RENDER_PATH=%s", RENDER_PATH_EXTERNAL)
    return PreviewResolveResult(
        resolved_url=raw, render_path=RENDER_PATH_EXTERNAL,
        is_external=True, original_ref=raw,
    )


# ---------------------------------------------------------------------------
# Auth injection helper
# ---------------------------------------------------------------------------

def _inject_auth(url: str, auth_token: str) -> str:
    """Inject auth token into URL if provided."""
    if not auth_token:
        return url
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}tgWebAppData={auth_token}"
