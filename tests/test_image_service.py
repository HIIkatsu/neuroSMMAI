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
import httpx
from unittest.mock import AsyncMock, MagicMock, patch
from openai import NotFoundError

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

    def test_build_includes_onboarding_context(self):
        from image_prompts import build_generation_prompt
        result = build_generation_prompt(
            title="Как выбрать зимние шины",
            body="Практический список критериев выбора.",
            channel_topic="Автомобили",
            channel_style="экспертно и без хайпа",
            channel_audience="владельцы семейных автомобилей",
            channel_subniche="обслуживание авто",
            onboarding_summary="автомеханик 10 лет опыта",
            post_intent="practical checklist",
        )
        prompt_lower = result["prompt"].lower()
        self.assertIn("audience context", prompt_lower)
        self.assertIn("subniche focus", prompt_lower)
        self.assertIn("onboarding profile", prompt_lower)


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
             patch("image_service.build_fallback_search_query", return_value="car service workshop mechanic tools editorial"), \
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
             patch("image_service.build_fallback_search_query", return_value="car maintenance service station mechanic editorial"), \
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

    def test_get_image_rejects_reputationally_risky_fallback(self):
        from image_service import get_image

        with patch("image_service.generate_image", new_callable=AsyncMock) as mock_gen, \
             patch("image_service.build_fallback_search_query", return_value="car service workshop mechanic tools editorial"), \
             patch("image_service.search_stock_photo", new_callable=AsyncMock) as mock_search:
            mock_gen.return_value = None
            mock_search.return_value = "https://example.com/cringe_meme_funny.jpg"
            result = asyncio.run(
                get_image(
                    title="Технический обзор серверов",
                    body="Разбираем инфраструктуру дата-центра.",
                    channel_topic="B2B инфраструктура",
                    api_key="test-key",
                )
            )
        self.assertEqual(result.source, "none")

    def test_get_image_allows_family_mismatch_penalty_when_text_flagged(self):
        from image_service import get_image
        from image_history import ImageHistory

        with patch("image_service.get_image_history", return_value=ImageHistory(maxlen=10, ttl=3600)), \
             patch("image_service.generate_image", new_callable=AsyncMock) as mock_gen, \
             patch("image_service.build_fallback_search_query", return_value="car maintenance service station mechanic editorial"), \
             patch("image_service.search_stock_photo", new_callable=AsyncMock) as mock_search:
            mock_gen.return_value = None
            mock_search.return_value = "https://images.pexels.com/photos/12345/editorial-photo.jpeg?case=text_flagged"
            result = asyncio.run(
                get_image(
                    title="Как выбрать тормозные колодки для городской езды",
                    body="Тема ушла в еду и десерты, но канал про обслуживание автомобиля.",
                    channel_topic="Автомобили",
                    api_key="test-key",
                    model="gpt-image-1",
                    text_quality_flagged=True,
                )
            )
        self.assertEqual(result.source, "fallback")

    def test_fallback_not_rejected_only_for_short_query(self):
        from image_service import get_image
        from image_history import ImageHistory

        with patch("image_service.get_image_history", return_value=ImageHistory(maxlen=10, ttl=3600)), \
             patch("image_service.generate_image", new_callable=AsyncMock) as mock_gen, \
             patch("image_service.build_fallback_search_query", return_value="car"), \
             patch("image_service.search_stock_photo", new_callable=AsyncMock) as mock_search:
            mock_gen.return_value = None
            mock_search.return_value = "https://images.pexels.com/photos/12345/editorial-photo.jpeg?case=short_prompt"
            result = asyncio.run(
                get_image(
                    title="Сервис и обслуживание авто",
                    body="Практика по выбору СТО и инструментов",
                    channel_topic="Автомобили",
                    api_key="test-key",
                )
            )
        self.assertEqual(result.source, "fallback")


class TestImageCandidateValidation(unittest.TestCase):
    def test_candidate_family_mismatch_rejected(self):
        from image_validation import validate_image_candidate
        ok, reason = validate_image_candidate(
            prompt="Professional photo of gourmet pasta dish in kitchen",
            title="Как выбрать тормозные колодки",
            body="Практический гайд для водителей.",
            channel_topic="Автомобили",
        )
        self.assertFalse(ok)
        self.assertTrue(
            "family_mismatch" in reason or "prompt_not_relevant" in reason,
            reason,
        )

    def test_candidate_reputation_risk_rejected(self):
        from image_validation import validate_image_candidate
        ok, reason = validate_image_candidate(
            prompt="Professional documentary scene",
            title="Новости городской инфраструктуры",
            channel_topic="городские новости",
            media_ref="https://cdn.example.com/nsfw-scene.jpg",
        )
        self.assertFalse(ok)
        self.assertIn("reputational_risk_ref", reason)

    def test_candidate_family_mismatch_penalty_when_allowed(self):
        from image_validation import validate_image_candidate
        ok, reason = validate_image_candidate(
            prompt="Professional photo of a car service workshop, mechanic tools and repair station interior",
            title="Как выбрать тормозные колодки для автомобиля",
            body="Практический гайд для водителей.",
            channel_topic="Автомобили и обслуживание",
            allow_family_mismatch_penalty=True,
        )
        self.assertTrue(ok, reason)
        self.assertIn("family_mismatch_penalty", reason)

    def test_family_context_hint_overrides_offtopic_body(self):
        from image_validation import validate_image_candidate
        ok, reason = validate_image_candidate(
            prompt="Professional photo of car maintenance in local workshop",
            title="Советы по автосервису",
            body="Лучший десерт с клубникой и кремом.",
            channel_topic="Автомобили",
            family_context_hint="Автомобили локальный автосервис",
        )
        self.assertTrue(ok, reason)

    def test_gross_family_mismatch_still_rejected_even_with_penalty_flag(self):
        from image_validation import validate_image_candidate
        ok, reason = validate_image_candidate(
            prompt="Professional macro photo of gourmet pasta and dessert plate",
            title="Как заменить тормозную жидкость в автомобиле",
            body="Пошаговый сервисный чеклист.",
            channel_topic="Автомобили",
            allow_family_mismatch_penalty=True,
        )
        self.assertFalse(ok)
        self.assertTrue("family_mismatch" in reason or "prompt_not_relevant" in reason, reason)

    def test_low_confidence_body_does_not_poison_family_context(self):
        from image_validation import validate_image_candidate
        ok, reason = validate_image_candidate(
            prompt="Professional photo of a mechanic tools and service station interior",
            title="Обслуживание автомобиля: чеклист на сезон",
            body="Рецепт пасты с десертом и клубникой",
            channel_topic="Автомобили",
            family_context_hint="Автомобили обслуживание сервис",
            ignore_body_for_family_context=True,
        )
        self.assertTrue(ok, reason)


class TestImageHistoryPatternDedup(unittest.TestCase):
    def test_visual_pattern_dedup(self):
        from image_history import ImageHistory
        h = ImageHistory(maxlen=10, ttl=3600)
        prompt = "Professional editorial photo of a mechanic working on a car engine in service garage"
        self.assertFalse(h.is_duplicate_visual_pattern(prompt))
        h.record(prompt=prompt)
        self.assertTrue(h.is_duplicate_visual_pattern(prompt))


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


# ---------------------------------------------------------------------------
# 9. Strengthened critical-path tests
# ---------------------------------------------------------------------------
class TestGenerationValidationRejection(unittest.TestCase):
    """Generation that produces invalid bytes must be rejected, not silently accepted."""

    def test_generation_returns_invalid_format_rejected(self):
        """If AI returns non-image bytes, flow should reject and try fallback."""
        from image_service import get_image

        non_image_bytes = b"NOT_AN_IMAGE_JUST_TEXT" * 200  # >1KB but not image

        with patch("image_service.generate_image", new_callable=AsyncMock) as mock_gen, \
             patch("image_service.build_fallback_search_query", return_value="car maintenance service station mechanic editorial"), \
             patch("image_service.search_stock_photo", new_callable=AsyncMock) as mock_search:
            mock_gen.return_value = non_image_bytes
            mock_search.return_value = ""
            result = asyncio.run(
                get_image(title="Test", api_key="test-key")
            )
        # Should fail because validation rejects non-image bytes
        self.assertEqual(result.source, "none")
        self.assertTrue(result.failure_reason)

    def test_generation_returns_tiny_data_rejected(self):
        """If AI returns tiny data (<1KB), should be rejected."""
        from image_service import get_image

        tiny_png = b"\x89PNG" + b"\x00" * 10  # Valid magic but too small

        with patch("image_service.generate_image", new_callable=AsyncMock) as mock_gen, \
             patch("image_service.build_fallback_search_query", return_value="car maintenance service station mechanic editorial"), \
             patch("image_service.search_stock_photo", new_callable=AsyncMock) as mock_search:
            mock_gen.return_value = tiny_png
            mock_search.return_value = ""
            result = asyncio.run(
                get_image(title="Test", api_key="test-key")
            )
        self.assertEqual(result.source, "none")

    def test_generation_exception_triggers_fallback(self):
        """If generation raises, should fall through to fallback."""
        from image_service import get_image

        with patch("image_service.generate_image", new_callable=AsyncMock) as mock_gen, \
             patch("image_service.search_stock_photo", new_callable=AsyncMock) as mock_search:
            mock_gen.side_effect = RuntimeError("Provider down")
            mock_search.return_value = "https://images.pexels.com/photos/999/fallback.jpeg"
            result = asyncio.run(
                get_image(title="Test", api_key="test-key")
            )
        self.assertEqual(result.source, "fallback")
        self.assertTrue(result.media_ref)

    def test_invalid_image_model_skips_generation_and_uses_fallback(self):
        """Text-only model must skip AI generation and proceed to fallback."""
        from image_service import get_image
        from image_history import ImageHistory

        with patch("image_service.get_image_history", return_value=ImageHistory(maxlen=10, ttl=3600)), \
             patch("image_service.generate_image", new_callable=AsyncMock) as mock_gen, \
             patch("image_service.build_fallback_search_query", return_value="car maintenance service station mechanic editorial"), \
             patch("image_service.search_stock_photo", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = "https://images.pexels.com/photos/999/fallback.jpeg"
            result = asyncio.run(
                get_image(
                    title="Как подготовить автомобиль к сезону и сервису",
                    body="Подбор полезных проверок и ухода за машиной.",
                    channel_topic="Автомобили",
                    api_key="test-key",
                    model="mistral-small-2603",
                )
            )
        mock_gen.assert_not_called()
        self.assertEqual(result.source, "fallback")


class TestStorageResultPath(unittest.TestCase):
    """Storage must return correct media refs for the generated-images serving path."""

    def test_storage_ref_format(self):
        """Saved image ref must start with /generated-images/ and include owner_id."""
        from image_storage import save_generated_image, GENERATED_DIR
        data = b"\xff\xd8\xff" + b"\x00" * 2000  # JPEG
        ref = save_generated_image(data, owner_id=42)
        self.assertTrue(ref.startswith("/generated-images/"), f"Bad ref format: {ref}")
        self.assertIn("42", ref)
        self.assertTrue(ref.endswith(".jpg"))
        # Cleanup
        filename = ref.split("/")[-1]
        (GENERATED_DIR / filename).unlink(missing_ok=True)

    def test_storage_no_owner_uses_gen_prefix(self):
        """Without owner_id, filename uses gen_ prefix."""
        from image_storage import save_generated_image, GENERATED_DIR
        data = b"\x89PNG" + b"\x00" * 2000
        ref = save_generated_image(data, owner_id=None)
        self.assertTrue(ref.startswith("/generated-images/gen_"))
        filename = ref.split("/")[-1]
        (GENERATED_DIR / filename).unlink(missing_ok=True)

    def test_image_exists_check(self):
        """image_exists should correctly detect saved files."""
        from image_storage import save_generated_image, image_exists, GENERATED_DIR
        data = b"\x89PNG" + b"\x00" * 2000
        ref = save_generated_image(data, owner_id=77)
        self.assertTrue(image_exists(ref))
        # Cleanup
        filename = ref.split("/")[-1]
        (GENERATED_DIR / filename).unlink(missing_ok=True)
        self.assertFalse(image_exists(ref))


class TestCallerIntegration(unittest.TestCase):
    """Verify that actions.py caller functions are correctly wired."""

    def test_resolve_post_image_exists_and_is_async(self):
        """actions.resolve_post_image must exist and be async."""
        from actions import resolve_post_image
        import inspect
        self.assertTrue(inspect.iscoroutinefunction(resolve_post_image))

    def test_resolve_post_image_signature_has_mode(self):
        """resolve_post_image must accept mode parameter."""
        from actions import resolve_post_image
        import inspect
        sig = inspect.signature(resolve_post_image)
        self.assertIn("mode", sig.parameters)

    def test_resolve_post_image_calls_get_image(self):
        """resolve_post_image should delegate to image_service.get_image."""
        import inspect
        import actions
        source = inspect.getsource(actions.resolve_post_image)
        self.assertIn("get_image", source)

    def test_generate_post_payload_imports_image_service(self):
        """actions.py must import from image_service, not deleted modules."""
        import actions
        import inspect
        source = inspect.getsource(actions)
        self.assertIn("from image_service import", source)
        self.assertNotIn("from image_gateway", source)
        self.assertNotIn("from image_search", source)
        self.assertNotIn("from image_pipeline", source)

    def test_scheduler_imports_validate_image(self):
        """scheduler_service.py must import validate_image from image_service."""
        with open(os.path.join(os.path.dirname(__file__), "..", "scheduler_service.py")) as f:
            source = f.read()
        self.assertIn("from image_service import validate_image", source)

    def test_miniapp_imports_trigger_unsplash(self):
        """miniapp_routes_content.py must import trigger_unsplash_download from image_service."""
        with open(os.path.join(os.path.dirname(__file__), "..", "miniapp_routes_content.py")) as f:
            source = f.read()
        self.assertIn("from image_service import trigger_unsplash_download", source)


class TestFallbackBadUrlRejected(unittest.TestCase):
    """Fallback URLs that fail validation should not be returned."""

    def test_fallback_avatar_url_rejected(self):
        """Fallback returning avatar URL should fail validation."""
        from image_service import get_image

        with patch("image_service.generate_image", new_callable=AsyncMock) as mock_gen, \
             patch("image_service.search_stock_photo", new_callable=AsyncMock) as mock_search:
            mock_gen.return_value = None
            mock_search.return_value = "https://example.com/avatar/photo.jpg"
            result = asyncio.run(
                get_image(title="Test", api_key="test-key")
            )
        # Avatar URL fails validation — should be rejected
        self.assertEqual(result.source, "none")

    def test_fallback_empty_url_gives_no_result(self):
        """Fallback returning empty URL should give no result."""
        from image_service import get_image

        with patch("image_service.generate_image", new_callable=AsyncMock) as mock_gen, \
             patch("image_service.search_stock_photo", new_callable=AsyncMock) as mock_search:
            mock_gen.return_value = None
            mock_search.return_value = ""
            result = asyncio.run(
                get_image(title="Test", api_key="test-key")
            )
        self.assertEqual(result.source, "none")


class TestAiImageGenerateContract(unittest.TestCase):
    """ai_client.ai_image_generate contract tests."""

    def test_no_api_key_returns_none(self):
        from ai_client import ai_image_generate
        result = asyncio.run(ai_image_generate("", "model", "prompt"))
        self.assertIsNone(result)

    def test_no_model_returns_none(self):
        from ai_client import ai_image_generate
        result = asyncio.run(ai_image_generate("key", "", "prompt"))
        self.assertIsNone(result)

    def test_no_prompt_returns_none(self):
        from ai_client import ai_image_generate
        result = asyncio.run(ai_image_generate("key", "model", ""))
        self.assertIsNone(result)

    def test_requests_b64_json_format(self):
        """Should request b64_json response_format to avoid URL round-trip."""
        from ai_client import ai_image_generate
        import inspect
        source = inspect.getsource(ai_image_generate)
        self.assertIn("b64_json", source)

    def test_404_html_error_is_logged_and_returns_none(self):
        from ai_client import ai_image_generate

        request = httpx.Request("POST", "https://openrouter.ai/api/v1/images/generations")
        response = httpx.Response(status_code=404, request=request)
        not_found = NotFoundError(
            "404 page",
            response=response,
            body="<!DOCTYPE html><html><body>Not Found</body></html>",
        )
        fake_client = MagicMock()
        fake_client.images.generate = AsyncMock(side_effect=not_found)

        with patch("ai_client._get_shared_client", new_callable=AsyncMock) as mock_get_client:
            mock_get_client.return_value = fake_client
            with self.assertLogs("ai_client", level="WARNING") as log_ctx:
                result = asyncio.run(
                    ai_image_generate(
                        "key",
                        "openai/dall-e-3",
                        "Car service editorial photo",
                        base_url="https://openrouter.ai/api/v1",
                    )
                )
        self.assertIsNone(result)
        joined = "\n".join(log_ctx.output)
        self.assertIn("API_ERROR", joined)
        self.assertIn("response_prefix", joined)
        self.assertIn("<!DOCTYPE html>", joined)

    def test_known_openrouter_gpt_image_combo_fast_skips(self):
        from ai_client import ai_image_generate
        with patch("ai_client._get_shared_client", new_callable=AsyncMock) as mock_get_client:
            result = asyncio.run(
                ai_image_generate(
                    "key",
                    "openai/gpt-image-1",
                    "studio portrait",
                    base_url="https://openrouter.ai/api/v1",
                )
            )
        self.assertIsNone(result)
        mock_get_client.assert_not_called()


class TestEndToEndGenerationSuccess(unittest.TestCase):
    """Full end-to-end: generation produces valid image → stored → returned."""

    def test_full_flow_png(self):
        from image_service import get_image

        fake_png = b"\x89PNG" + os.urandom(3000)  # Random valid-looking PNG bytes

        with patch("image_service.generate_image", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = fake_png
            result = asyncio.run(
                get_image(
                    title="Beautiful landscape",
                    body="Mountain lake at sunset",
                    channel_topic="Nature photography",
                    api_key="test-key",
                    owner_id=123,
                    mode="editor",
                )
            )

        self.assertEqual(result.source, "generation")
        self.assertTrue(result.is_generated)
        self.assertTrue(result.media_ref.startswith("/generated-images/"))
        self.assertTrue(result.family)  # Family detection works
        self.assertTrue(len(result.prompt_used) > 10)
        self.assertEqual(result.failure_reason, "")

        # Verify file exists on disk
        from image_storage import image_exists, GENERATED_DIR
        self.assertTrue(image_exists(result.media_ref))

        # Cleanup
        filename = result.media_ref.split("/")[-1]
        (GENERATED_DIR / filename).unlink(missing_ok=True)

    def test_full_flow_jpeg(self):
        from image_service import get_image

        fake_jpeg = b"\xff\xd8\xff" + os.urandom(3000)

        with patch("image_service.generate_image", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = fake_jpeg
            result = asyncio.run(
                get_image(
                    title="Рецепт блинов",
                    body="Тонкие блины на молоке",
                    api_key="test-key",
                    owner_id=456,
                )
            )

        self.assertEqual(result.source, "generation")
        self.assertTrue(result.media_ref.endswith(".jpg"))
        self.assertEqual(result.family, "food")

        # Cleanup
        from image_storage import GENERATED_DIR
        filename = result.media_ref.split("/")[-1]
        (GENERATED_DIR / filename).unlink(missing_ok=True)


class TestNoDeletedModuleReferences(unittest.TestCase):
    """Verify the new image system has zero references to deleted modules."""

    _DELETED_MODULES = [
        "image_gateway", "image_search", "image_pipeline",
        "image_pipeline_v3", "image_ranker", "ai_image_generator",
    ]

    def _check_file(self, filepath):
        with open(filepath) as f:
            content = f.read()
        for mod in self._DELETED_MODULES:
            # Only check imports, not comments
            for line in content.splitlines():
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if f"from {mod}" in stripped or f"import {mod}" in stripped:
                    self.fail(f"{filepath} has import of deleted module: {mod!r} in: {stripped!r}")

    def test_image_service_clean(self):
        self._check_file(os.path.join(os.path.dirname(__file__), "..", "image_service.py"))

    def test_image_generation_clean(self):
        self._check_file(os.path.join(os.path.dirname(__file__), "..", "image_generation.py"))

    def test_image_prompts_clean(self):
        self._check_file(os.path.join(os.path.dirname(__file__), "..", "image_prompts.py"))

    def test_image_validation_clean(self):
        self._check_file(os.path.join(os.path.dirname(__file__), "..", "image_validation.py"))

    def test_image_storage_clean(self):
        self._check_file(os.path.join(os.path.dirname(__file__), "..", "image_storage.py"))

    def test_image_history_clean(self):
        self._check_file(os.path.join(os.path.dirname(__file__), "..", "image_history.py"))

    def test_image_fallback_clean(self):
        self._check_file(os.path.join(os.path.dirname(__file__), "..", "image_fallback.py"))

    def test_actions_clean(self):
        self._check_file(os.path.join(os.path.dirname(__file__), "..", "actions.py"))

    def test_scheduler_service_clean(self):
        self._check_file(os.path.join(os.path.dirname(__file__), "..", "scheduler_service.py"))

    def test_miniapp_routes_content_clean(self):
        self._check_file(os.path.join(os.path.dirname(__file__), "..", "miniapp_routes_content.py"))
