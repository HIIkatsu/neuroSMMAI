"""
image_storage.py — Save generated images, return media/storage references.

Integrates with the existing app flow:
  - Saves to generated_images/ directory (same as existing serving path)
  - Returns references compatible with /generated-images/ serving endpoint
  - Handles owner-scoped naming for access control
"""
from __future__ import annotations

import hashlib
import logging
import os
import time
from pathlib import Path

from image_validation import detect_image_extension, validate_image_bytes

logger = logging.getLogger(__name__)

# Use the same directory the app already serves from
BASE_DIR = Path(__file__).resolve().parent
GENERATED_DIR = BASE_DIR / "generated_images"
GENERATED_DIR.mkdir(parents=True, exist_ok=True)


def save_generated_image(
    image_bytes: bytes,
    *,
    owner_id: int | None = None,
    prompt_hint: str = "",
) -> str:
    """Save generated image bytes to disk.

    Returns the media reference string compatible with the app's
    /generated-images/ serving endpoint (e.g., "/generated-images/123_abc.png").

    Returns empty string on failure.
    """
    # Validate first
    is_valid, reason = validate_image_bytes(image_bytes)
    if not is_valid:
        logger.warning(
            "IMAGE_STORAGE_REJECT reason=%s owner_id=%s",
            reason, owner_id,
        )
        return ""

    # Detect format
    ext = detect_image_extension(image_bytes)

    # Build filename: {owner_id}_{content_hash}.{ext}
    # All components are safe: owner_id is int, content_hash is hex, ext is from fixed set
    content_hash = hashlib.sha256(image_bytes).hexdigest()[:16]
    ts = int(time.time())

    if owner_id:
        filename = f"{int(owner_id)}_{content_hash}_{ts}{ext}"
    else:
        filename = f"gen_{content_hash}_{ts}{ext}"

    # Sanitize: ensure no path traversal by stripping directory separators
    filename = filename.replace("/", "").replace("\\", "").replace("..", "")
    filepath = GENERATED_DIR / filename

    # Verify the resolved path stays within GENERATED_DIR
    if not filepath.resolve().is_relative_to(GENERATED_DIR.resolve()):
        logger.error("IMAGE_STORAGE_PATH_ESCAPE file=%s", filename)
        return ""

    try:
        filepath.write_bytes(image_bytes)
        logger.info(
            "IMAGE_STORAGE_SAVED file=%s bytes=%d owner_id=%s",
            filename, len(image_bytes), owner_id,
        )
        # Return the URL path that miniapp_server.py serves
        return f"/generated-images/{filename}"
    except Exception as exc:
        logger.error(
            "IMAGE_STORAGE_FAILED file=%s error=%s owner_id=%s",
            filename, exc, owner_id,
        )
        return ""


def image_exists(media_ref: str) -> bool:
    """Check if a generated image reference still exists on disk."""
    if not media_ref:
        return False

    ref = media_ref.strip()

    # Handle /generated-images/ URL paths
    if ref.startswith("/generated-images/"):
        filename = ref.split("/")[-1]
        return (GENERATED_DIR / filename).is_file()

    # Handle generated_images/ relative paths
    if ref.startswith("generated_images/"):
        return (BASE_DIR / ref).is_file()

    # Handle absolute paths
    if os.path.isabs(ref):
        return os.path.isfile(ref)

    return False
