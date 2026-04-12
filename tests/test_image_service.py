"""
test_image_service.py — Smoke tests for the new generation-first image system.

Tests cover:
  - Module structure and exports
  - Prompt building
  - Image validation (bytes, files, URLs)
  - Dedup/history
  - Fallback query building
  - ImageResult structure
  - End-to-end flow with mocked generation
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure repo root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# 1. Module structure and exports
# ---------------------------------------------------------------------------
class TestImageServiceExports(unittest.TestCase):
    """image_service must export all required public symbols."""

    def test_get_image_exists(self):
        from image_service import get_image
        self.assertTrue(asyncio.iscoroutinefunction(get_image))

    def test_validate_image_exists(self):
        from image_service import validate_image
        self.assertTrue(asyncio.iscoroutinefunction(validate_image))

    def test_trigger_unsplash_download_exists(self):
        from image_service import trigger_unsplash_download
        self.assertTrue(asyncio.iscoroutinefunction(trigger_unsplash_download))

    def test_mode_constants(self):
        from image_service import MODE_AUTOPOST, MODE_EDITOR
        self.assertEqual(MODE_AUTOPOST, "autopost")
        self.assertEqual(MODE_EDITOR, "editor")

    def test_latin_token_re(self):
        from image_service import _LATIN_TOKEN_RE
        self.assertTrue(_LATIN_TOKEN_RE.match("hello"))
        self.assertFalse(_LATIN_TOKEN_RE.match("привет"))

    def test_image_result_dataclass(self):
        from image_service import ImageResult
        r = ImageResult(media_ref="/test.png", source="generation")
        self.assertEqual(r.media_ref, "/test.png")
        self.assertEqual(r.source, "generation")
        self.assertTrue(r.failure_reason == "")


# ---------------------------------------------------------------------------
# 2. Prompt building
# ---------------------------------------------------------------------------
class TestImagePrompts(unittest.TestCase):
    """image_prompts must build structured prompts from context."""

    def test_build_with_llm_prompt(self):
        from image_prompts import build_generation_prompt
        result = build_generation_prompt(
            title="Test", body="Body", llm_image_prompt="A beautiful sunset over ocean",
        )
        self.assertIn("sunset", result["prompt"].lower())
        self.assertIn("negative_prompt", result)

    def test_build_without_llm_prompt(self):
        from image_prompts import build_generation_prompt
        result = build_generation_prompt(title="Рецепт борща", body="Как варить борщ")
        self.assertIn("prompt", result)
        self.assertEqual(result["family"], "food")
        self.assertTrue(len(result["prompt"]) > 20)

    def test_build_empty_context(self):
        from image_prompts import build_generation_prompt
        result = build_generation_prompt()
        self.assertIn("prompt", result)
        self.assertTrue(len(result["prompt"]) > 10)

    def test_fallback_search_query(self):
        from image_prompts import build_fallback_search_query
        q = build_fallback_search_query(title="BMW X5", channel_topic="Автомобили")
        # Should be Latin-only
        import re
        self.assertTrue(re.match(r'^[a-zA-Z0-9\s]*$', q), f"Expected latin-only: {q!r}")
        self.assertTrue(len(q) > 5)

    def test_family_detection_in_prompts(self):
        from image_prompts import build_generation_prompt
        families = {
            "massage": "Массаж спины: расслабление мышц",
            "cars": "Замена масла в двигателе",
            "hardware": "Обзор нового процессора Intel",
        }
        for expected_family, title in families.items():
            result = build_generation_prompt(title=title)
            self.assertEqual(
                result["family"], expected_family,
                f"Title {title!r} should detect {expected_family}, got {result['family']}",
            )


# ---------------------------------------------------------------------------
# 3. Image validation
# ---------------------------------------------------------------------------
class TestImageValidation(unittest.TestCase):
    """image_validation must correctly validate images."""

    def test_validate_null_data(self):
        from image_validation import validate_image_bytes
        ok, reason = validate_image_bytes(None)
        self.assertFalse(ok)
        self.assertEqual(reason, "null_data")

    def test_validate_too_small(self):
        from image_validation import validate_image_bytes
        ok, reason = validate_image_bytes(b"\x89PNG\x00\x00")
        self.assertFalse(ok)
        self.assertIn("too_small", reason)

    def test_validate_invalid_format(self):
        from image_validation import validate_image_bytes
        ok, reason = validate_image_bytes(b"NOT_AN_IMAGE" * 200)
        self.assertFalse(ok)
        self.assertEqual(reason, "invalid_image_format")

    def test_validate_valid_png(self):
        from image_validation import validate_image_bytes
        data = b"\x89PNG" + b"\x00" * 2000
        ok, reason = validate_image_bytes(data)
        self.assertTrue(ok)
        self.assertEqual(reason, "ok")

    def test_validate_valid_jpeg(self):
        from image_validation import validate_image_bytes
        data = b"\xff\xd8\xff" + b"\x00" * 2000
        ok, reason = validate_image_bytes(data)
        self.assertTrue(ok)

    def test_validate_url_empty(self):
        from image_validation import validate_image_url
        ok, reason = validate_image_url("")
        self.assertFalse(ok)
        self.assertEqual(reason, "empty_url")

    def test_validate_url_not_http(self):
        from image_validation import validate_image_url
        ok, reason = validate_image_url("ftp://example.com/img.jpg")
        self.assertFalse(ok)

    def test_validate_url_bad_pattern(self):
        from image_validation import validate_image_url
        ok, reason = validate_image_url("https://example.com/avatar.jpg")
        self.assertFalse(ok)
        self.assertIn("avatar", reason)

    def test_validate_url_good(self):
        from image_validation import validate_image_url
        ok, reason = validate_image_url("https://images.pexels.com/photos/123/photo.jpg")
        self.assertTrue(ok)

    def test_validate_media_ref_empty(self):
        from image_validation import validate_media_ref
        ok, reason = validate_media_ref("")
        self.assertTrue(ok)  # Empty is valid (no image)

    def test_validate_media_ref_telegram(self):
        from image_validation import validate_media_ref
        ok, reason = validate_media_ref("tgfile:photo:abc123")
        self.assertTrue(ok)

    def test_detect_extension(self):
        from image_validation import detect_image_extension
        self.assertEqual(detect_image_extension(b"\x89PNG\x00\x00"), ".png")
        self.assertEqual(detect_image_extension(b"\xff\xd8\xff\x00"), ".jpg")
        self.assertEqual(detect_image_extension(b"RIFF\x00\x00"), ".webp")
        self.assertEqual(detect_image_extension(b"GIF8\x00\x00"), ".gif")


# ---------------------------------------------------------------------------
# 4. Image storage
# ---------------------------------------------------------------------------
class TestImageStorage(unittest.TestCase):
    """image_storage must save and reference images correctly."""

    def test_save_valid_png(self):
        from image_storage import save_generated_image, GENERATED_DIR
        data = b"\x89PNG" + b"\x00" * 2000
        ref = save_generated_image(data, owner_id=12345)
        self.assertTrue(ref.startswith("/generated-images/"))
        self.assertIn("12345", ref)
        # Cleanup
        filename = ref.split("/")[-1]
        filepath = GENERATED_DIR / filename
        if filepath.exists():
            filepath.unlink()

    def test_save_rejects_tiny_data(self):
        from image_storage import save_generated_image
        ref = save_generated_image(b"\x89PNG\x00")
        self.assertEqual(ref, "")

    def test_save_rejects_invalid_format(self):
        from image_storage import save_generated_image
        ref = save_generated_image(b"not_an_image" * 200)
        self.assertEqual(ref, "")


# ---------------------------------------------------------------------------
# 5. Image history / dedup
# ---------------------------------------------------------------------------
class TestImageHistory(unittest.TestCase):
    """image_history must track and detect duplicates."""

    def test_content_dedup(self):
        from image_history import ImageHistory
        h = ImageHistory(maxlen=10, ttl=3600)
        data = b"\x89PNG" + b"\x00" * 2000
        self.assertFalse(h.is_duplicate_content(data))
        h.record(image_bytes=data)
        self.assertTrue(h.is_duplicate_content(data))

    def test_prompt_dedup(self):
        from image_history import ImageHistory
        h = ImageHistory(maxlen=10, ttl=3600)
        prompt = "professional photo of pasta"
        self.assertFalse(h.is_duplicate_prompt(prompt))
        h.record(prompt=prompt)
        self.assertTrue(h.is_duplicate_prompt(prompt))

    def test_ref_dedup(self):
        from image_history import ImageHistory
        h = ImageHistory(maxlen=10, ttl=3600)
        ref = "https://images.pexels.com/photo.jpg"
        self.assertFalse(h.is_duplicate_ref(ref))
        h.record(media_ref=ref)
        self.assertTrue(h.is_duplicate_ref(ref))

    def test_different_content_not_duplicate(self):
        from image_history import ImageHistory
        h = ImageHistory(maxlen=10, ttl=3600)
        data1 = b"\x89PNG" + b"\x01" * 2000
        data2 = b"\x89PNG" + b"\x02" * 2000
        h.record(image_bytes=data1)
        self.assertFalse(h.is_duplicate_content(data2))

    def test_global_history_singleton(self):
        from image_history import get_image_history
        h1 = get_image_history()
        h2 = get_image_history()
        self.assertIs(h1, h2)

    def test_size_property(self):
        from image_history import ImageHistory
        h = ImageHistory(maxlen=10, ttl=3600)
        self.assertEqual(h.size, 0)
        h.record(prompt="test")
        self.assertEqual(h.size, 1)
        h.record(image_bytes=b"\x89PNG" + b"\x00" * 2000, prompt="test2", media_ref="ref")
        self.assertEqual(h.size, 4)  # 1 old prompt + 1 hash + 1 prompt + 1 ref


# ---------------------------------------------------------------------------
# 6. End-to-end flow with mocked generation
# ---------------------------------------------------------------------------
class TestImageServiceFlow(unittest.TestCase):
    """Test the full generation-first flow with mocked providers."""

    def test_get_image_generation_success(self):
        """When generation succeeds, should return generated image."""
        from image_service import get_image

        fake_png = b"\x89PNG" + b"\x00" * 2000

        with patch("image_service.generate_image", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = fake_png
            result = asyncio.run(
                get_image(
                    title="Test post",
                    body="Test body",
                    api_key="test-key",
                    owner_id=999,
                )
            )
        self.assertTrue(result.media_ref)
        self.assertEqual(result.source, "generation")
        self.assertTrue(result.is_generated)

        # Cleanup saved file
        from image_storage import GENERATED_DIR
        filename = result.media_ref.split("/")[-1]
        filepath = GENERATED_DIR / filename
        if filepath.exists():
            filepath.unlink()

    def test_get_image_generation_fails_fallback_succeeds(self):
        """When generation fails but fallback succeeds, should return fallback."""
        from image_service import get_image

        with patch("image_service.generate_image", new_callable=AsyncMock) as mock_gen, \
             patch("image_service.search_stock_photo", new_callable=AsyncMock) as mock_search:
            mock_gen.return_value = None
            mock_search.return_value = "https://images.pexels.com/photos/123/pexels-photo-123.jpeg"
            result = asyncio.run(
                get_image(
                    title="Test post",
                    body="Test body",
                    api_key="test-key",
                )
            )
        self.assertTrue(result.media_ref)
        self.assertEqual(result.source, "fallback")
        self.assertFalse(result.is_generated)

    def test_get_image_both_fail(self):
        """When both generation and fallback fail, should return empty with reason."""
        from image_service import get_image

        with patch("image_service.generate_image", new_callable=AsyncMock) as mock_gen, \
             patch("image_service.search_stock_photo", new_callable=AsyncMock) as mock_search:
            mock_gen.return_value = None
            mock_search.return_value = ""
            result = asyncio.run(
                get_image(
                    title="Test post",
                    api_key="test-key",
                )
            )
        self.assertEqual(result.media_ref, "")
        self.assertEqual(result.source, "none")
        self.assertTrue(result.failure_reason)

    def test_get_image_no_api_key(self):
        """When no API key, generation skips but fallback should still try."""
        from image_service import get_image

        with patch("image_service.search_stock_photo", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = "https://images.pexels.com/test.jpg"
            result = asyncio.run(
                get_image(title="Test", body="Body")
            )
        # Should fall through to fallback
        self.assertIn(result.source, ("fallback", "none"))


# ---------------------------------------------------------------------------
# 7. Trigger unsplash download (no-op)
# ---------------------------------------------------------------------------
class TestTriggerUnsplashDownload(unittest.TestCase):
    """trigger_unsplash_download must be a no-op stub."""

    def test_returns_false(self):
        from image_service import trigger_unsplash_download
        result = asyncio.run(
            trigger_unsplash_download("https://example.com/download")
        )
        self.assertFalse(result)

    def test_accepts_any_string(self):
        from image_service import trigger_unsplash_download
        for val in ("", "https://test.com", "random", None):
            result = asyncio.run(
                trigger_unsplash_download(val or "")
            )
            self.assertFalse(result)


# ---------------------------------------------------------------------------
# 8. Validate image (quality gate)
# ---------------------------------------------------------------------------
class TestValidateImage(unittest.TestCase):
    """image_service.validate_image quality gate tests."""

    def test_empty_ref_is_valid(self):
        from image_service import validate_image
        result = asyncio.run(
            validate_image("", title="Test")
        )
        self.assertTrue(result)

    def test_http_url_is_valid(self):
        from image_service import validate_image
        result = asyncio.run(
            validate_image("https://images.pexels.com/photo.jpg", title="Test")
        )
        self.assertTrue(result)

    def test_telegram_ref_is_valid(self):
        from image_service import validate_image
        result = asyncio.run(
            validate_image("tgfile:photo:abc123", title="Test")
        )
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
