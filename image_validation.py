"""
image_validation.py — Sanity validation for generated/fetched image results.

Validates:
  - Non-empty, non-broken result
  - Minimum file size (not a placeholder/error page)
  - Basic image format detection
  - URL sanity for external references

Every rejection returns a structured reason string.
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path

from topic_utils import detect_topic_family
from content_modes import check_image_prompt_relevance, MODE_GENERIC

logger = logging.getLogger(__name__)

# Minimum valid image size in bytes (reject tiny placeholders/errors)
MIN_IMAGE_BYTES = 1024  # 1 KB
MAX_IMAGE_BYTES = 50 * 1024 * 1024  # 50 MB

# Known image magic bytes
_PNG_MAGIC = b"\x89PNG"
_JPEG_MAGIC = b"\xff\xd8\xff"
_WEBP_MAGIC = b"RIFF"
_GIF_MAGIC = b"GIF8"

# Bad URL patterns (avatars, icons, placeholders)
_BAD_URL_PARTS = ("avatar", "icon", "logo", "sprite", "thumb", "placeholder", "1x1", "blank")
_REPUTATIONAL_RISK_TERMS = (
    "meme", "prank", "clickbait", "nsfw", "gore", "fetish", "fake",
    "conspiracy", "political rally", "hate", "violence", "weapon closeup",
    "absurd", "clown", "cringe",
)
_CLOSE_FAMILY_GROUPS = (
    frozenset({"cars", "local_business", "city_transport"}),
    frozenset({"marketing", "tech", "finance", "business"}),
    frozenset({"food", "lifestyle", "health"}),
)


def validate_image_candidate(
    *,
    prompt: str,
    title: str = "",
    body: str = "",
    channel_topic: str = "",
    family_context_hint: str = "",
    content_mode: str = MODE_GENERIC,
    media_ref: str = "",
    allow_family_mismatch_penalty: bool = False,
) -> tuple[bool, str]:
    """Validate semantic and reputational quality of an image candidate."""
    p = (prompt or "").strip().lower()
    if len(p) < 24:
        return False, "prompt_too_short"

    for bad in _REPUTATIONAL_RISK_TERMS:
        if bad in p:
            return False, f"reputational_risk_prompt_{bad.replace(' ', '_')}"

    if media_ref:
        ref_lower = media_ref.lower()
        for bad in _REPUTATIONAL_RISK_TERMS:
            if bad.replace(" ", "") in ref_lower or bad.replace(" ", "_") in ref_lower:
                return False, f"reputational_risk_ref_{bad.replace(' ', '_')}"

    is_relevant, reason = check_image_prompt_relevance(
        title=title,
        image_prompt=prompt,
        content_mode=content_mode,
    )
    if not is_relevant:
        return False, f"prompt_not_relevant_{reason}"

    post_family = detect_topic_family(family_context_hint) if family_context_hint else "generic"
    if post_family == "generic":
        post_family = detect_topic_family(" ".join(filter(None, [title, body, channel_topic])))
    prompt_family = detect_topic_family(prompt)
    if post_family != "generic" and prompt_family not in (post_family, "generic"):
        if allow_family_mismatch_penalty and _is_close_family(post_family, prompt_family):
            return True, f"family_mismatch_penalty_post_{post_family}_prompt_{prompt_family}"
        return False, f"family_mismatch_post_{post_family}_prompt_{prompt_family}"

    return True, "ok"


def _is_close_family(post_family: str, prompt_family: str) -> bool:
    if post_family == prompt_family:
        return True
    if "generic" in {post_family, prompt_family}:
        return True
    for group in _CLOSE_FAMILY_GROUPS:
        if post_family in group and prompt_family in group:
            return True
    return False


def validate_image_bytes(data: bytes | None) -> tuple[bool, str]:
    """Validate raw image bytes.

    Returns (is_valid, reason).
    """
    if data is None:
        return False, "null_data"

    if len(data) < MIN_IMAGE_BYTES:
        return False, f"too_small_{len(data)}_bytes"

    if len(data) > MAX_IMAGE_BYTES:
        return False, f"too_large_{len(data)}_bytes"

    # Check for valid image format
    if not _is_image_format(data):
        return False, "invalid_image_format"

    return True, "ok"


def validate_image_file(path: str) -> tuple[bool, str]:
    """Validate a saved image file on disk.

    Returns (is_valid, reason).
    """
    if not path:
        return False, "empty_path"

    p = Path(path)
    if not p.exists():
        return False, "file_not_found"

    if not p.is_file():
        return False, "not_a_file"

    size = p.stat().st_size
    if size < MIN_IMAGE_BYTES:
        return False, f"file_too_small_{size}_bytes"

    if size > MAX_IMAGE_BYTES:
        return False, f"file_too_large_{size}_bytes"

    # Check extension
    ext = p.suffix.lower()
    if ext not in (".png", ".jpg", ".jpeg", ".webp", ".gif"):
        return False, f"bad_extension_{ext}"

    return True, "ok"


def validate_image_url(url: str) -> tuple[bool, str]:
    """Validate an external image URL for basic sanity.

    Returns (is_valid, reason).
    """
    if not url or not url.strip():
        return False, "empty_url"

    url_lower = url.lower()

    if not url_lower.startswith(("http://", "https://")):
        return False, "not_http_url"

    for bad in _BAD_URL_PARTS:
        if bad in url_lower:
            return False, f"bad_url_pattern_{bad}"

    return True, "ok"


def validate_media_ref(media_ref: str) -> tuple[bool, str]:
    """Validate any media reference (URL, local path, or telegram file ref).

    Returns (is_valid, reason).
    This is a lightweight check for use by callers who just need to know
    if a ref is plausible before using it.
    """
    if not media_ref or not media_ref.strip():
        return True, "empty_is_ok"  # Empty means no image, which is valid

    ref = media_ref.strip()

    # Telegram file reference
    if ref.startswith("tgfile:"):
        return True, "telegram_ref"

    # Local path
    if ref.startswith("/") or ref.startswith("./") or ref.startswith("generated_images/") or ref.startswith("uploads/"):
        return True, "local_path"

    # URL path
    if ref.startswith("/generated-images/") or ref.startswith("/uploads/"):
        return True, "url_path"

    # HTTP URL
    if ref.startswith("http://") or ref.startswith("https://"):
        return validate_image_url(ref)

    # Unknown format — accept but warn
    logger.debug("IMAGE_VALIDATE_UNKNOWN_REF_FORMAT ref=%r", ref[:80])
    return True, "unknown_format_accepted"


def _is_image_format(data: bytes) -> bool:
    """Check if bytes look like a valid image format."""
    if len(data) < 4:
        return False
    return (
        data[:4] == _PNG_MAGIC
        or data[:3] == _JPEG_MAGIC
        or data[:4] == _WEBP_MAGIC
        or data[:4] == _GIF_MAGIC
    )


def detect_image_extension(data: bytes) -> str:
    """Detect image file extension from magic bytes."""
    if len(data) < 4:
        return ".png"  # default
    if data[:4] == _PNG_MAGIC:
        return ".png"
    if data[:3] == _JPEG_MAGIC:
        return ".jpg"
    if data[:4] == _WEBP_MAGIC:
        return ".webp"
    if data[:4] == _GIF_MAGIC:
        return ".gif"
    return ".png"  # default fallback
