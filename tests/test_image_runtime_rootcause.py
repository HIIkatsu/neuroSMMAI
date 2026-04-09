"""Regression tests for root-cause image runtime fixes.

Tests cover:
  1. Scooter topic must not resolve to local_business
  2. Pixabay /get/ URL without metadata must not be rejected by URL keyword mismatch
  3. Pexels asset URL without topic words must not be auto-rejected
  4. Deprecated HF provider path is skipped cleanly
  5. Post subject overrides weak channel family
  6. Legacy fallback, if preserved, is explicit in logs and not final by default
  7. CDN asset URL detection helper works correctly
  8. Decision trace logging helper exists and runs without error
  9. Family detection uses post text over channel topic

Run with:
    BOT_TOKEN=test:token python -m unittest tests.test_image_runtime_rootcause -v
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, AsyncMock

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("BOT_TOKEN", "test:token")
os.environ.setdefault("OPENROUTER_API_KEY", "test-key")

from topic_utils import detect_topic_family
from image_search import (
    validate_image_for_autopost,
    _is_cdn_asset_url,
    _provider_from_url,
    _log_image_decision_trace,
)
from ai_image_generator import generate_ai_image, DEFAULT_MODEL


# ---------------------------------------------------------------------------
# 1. Scooter topic must NOT resolve to local_business
# ---------------------------------------------------------------------------
class TestScooterTopicFamily(unittest.TestCase):
    """Scooter topics should not be hijacked into local_business family."""

    def test_scooter_cost_not_local_business(self):
        """'Сколько стоит езда на проколотом самокате' must not → local_business."""
        family = detect_topic_family("Сколько стоит езда на проколотом самокате")
        self.assertNotEqual(family, "local_business",
                            "Scooter cost topic must not resolve to local_business")

    def test_scooter_tire_marks_not_local_business(self):
        """'Ваш самокат оставляет отметины на асфальте' must not → local_business."""
        family = detect_topic_family("Ваш самокат оставляет отметины на асфальте")
        self.assertNotEqual(family, "local_business",
                            "Scooter tire marks topic must not resolve to local_business")

    def test_scooter_wheel_wear_not_local_business(self):
        """'протектор на задних колесах самоката стерт' must not → local_business."""
        family = detect_topic_family("протектор на задних колесах самоката стерт до индикатора")
        self.assertNotEqual(family, "local_business",
                            "Scooter wheel wear topic must not resolve to local_business")


# ---------------------------------------------------------------------------
# 2. Pixabay /get/ URL without metadata must NOT be rejected by URL keyword
# ---------------------------------------------------------------------------
class TestPixabayGetUrlNotRejected(unittest.TestCase):
    """Pixabay CDN /get/ URLs must not be hard-rejected by URL keyword check."""

    def test_pixabay_get_url_passes_validation(self):
        """pixabay.com/get/<hash> URL should pass without metadata."""
        url = "https://pixabay.com/get/gf6172cb3dddedad1d4c7e1e9df0b2441916667f02a5a688531c8732"
        result = validate_image_for_autopost(
            url,
            topic="Сколько стоит езда на проколотом самокате",
            prompt="scooter tire flat repair cost",
            post_text="По данным сервисных центров, 4 из 10 самокатов имеют проблемы с шинами.",
        )
        self.assertTrue(result,
                        "Pixabay /get/ URL must not be rejected solely by URL keyword mismatch")

    def test_pixabay_cdn_url_passes_validation(self):
        """cdn.pixabay.com URL should pass without metadata."""
        url = "https://cdn.pixabay.com/photo/2020/01/01/000000_960_720.jpg"
        result = validate_image_for_autopost(
            url,
            topic="Урожай и лампочка",
            prompt="harvest light bulb",
            post_text="Ваш урожай сожжет лампочка? Вот почему важно выбирать правильное освещение.",
        )
        self.assertTrue(result,
                        "cdn.pixabay.com URL must not be rejected by URL keyword mismatch")


# ---------------------------------------------------------------------------
# 3. Pexels asset URL without topic words must NOT be auto-rejected
# ---------------------------------------------------------------------------
class TestPexelsAssetUrlNotRejected(unittest.TestCase):
    """Pexels asset URLs must not be hard-rejected by URL keyword check."""

    def test_pexels_photo_url_passes_validation(self):
        """images.pexels.com/photos/<id>/... URL should pass without metadata."""
        url = "https://images.pexels.com/photos/11350076/pexels-photo-11350076.jpeg?auto=compress"
        result = validate_image_for_autopost(
            url,
            topic="Ваш урожай сожжет лампочка? Вот почему",
            prompt="harvest lighting indoor plants grow",
            post_text="Неправильное освещение может уничтожить рассаду за неделю.",
        )
        self.assertTrue(result,
                        "Pexels asset URL must not be rejected solely by URL having no topic keywords")

    def test_pexels_url_with_different_topic(self):
        """Pexels photo URL with generic topic should also pass."""
        url = "https://images.pexels.com/photos/99999/pexels-photo-99999.jpeg?auto=compress&w=600"
        result = validate_image_for_autopost(
            url,
            topic="Как выбрать правильный инструмент для дачи",
            prompt="garden tool selection",
            post_text="Правильный инструмент экономит 3 часа работы в неделю.",
        )
        self.assertTrue(result,
                        "Pexels URL must pass validation even without topic keywords in URL")


# ---------------------------------------------------------------------------
# 4. Deprecated HF provider path is skipped cleanly
# ---------------------------------------------------------------------------
class TestHfDeprecatedProviderSkipped(unittest.TestCase):
    """Deprecated HuggingFace provider path must be skipped without retries."""

    def test_default_model_skipped_immediately(self):
        """generate_ai_image with default model should return None immediately."""
        loop = asyncio.new_event_loop()
        try:
            with patch.dict(os.environ, {"HF_API_KEY": "test-key-123", "HF_IMAGE_MODEL": ""}):
                result = loop.run_until_complete(
                    generate_ai_image("test topic", "test prompt", "test text")
                )
        finally:
            loop.close()
        self.assertIsNone(result,
                          "Default deprecated HF model must be skipped immediately")

    def test_default_model_no_network_call(self):
        """No HTTP request should be made when default model is deprecated."""
        loop = asyncio.new_event_loop()
        try:
            with patch.dict(os.environ, {"HF_API_KEY": "test-key-123", "HF_IMAGE_MODEL": ""}):
                with patch("ai_image_generator.httpx.AsyncClient") as mock_client:
                    result = loop.run_until_complete(
                        generate_ai_image("topic", "prompt", "text")
                    )
                    mock_client.assert_not_called()
        finally:
            loop.close()
        self.assertIsNone(result)

    def test_explicit_model_not_skipped(self):
        """A non-default model should still attempt to generate (not blocked)."""
        loop = asyncio.new_event_loop()
        try:
            with patch.dict(os.environ, {
                "HF_API_KEY": "test-key-123",
                "HF_IMAGE_MODEL": "custom/my-model-v2"
            }):
                with patch("ai_image_generator.httpx.AsyncClient") as mock_client:
                    mock_response = AsyncMock()
                    mock_response.status_code = 500
                    mock_response.text = "error"
                    mock_response.headers = {}

                    mock_instance = AsyncMock()
                    mock_instance.post.return_value = mock_response
                    mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
                    mock_instance.__aexit__ = AsyncMock(return_value=False)
                    mock_client.return_value = mock_instance

                    result = loop.run_until_complete(
                        generate_ai_image("topic", "prompt", "text")
                    )
                    self.assertTrue(mock_instance.post.called or mock_client.called,
                                    "Custom model should attempt API call, not skip")
        finally:
            loop.close()

    def test_410_response_no_retry(self):
        """A 410 response from HF should return None immediately, not retry."""
        loop = asyncio.new_event_loop()
        try:
            with patch.dict(os.environ, {
                "HF_API_KEY": "test-key",
                "HF_IMAGE_MODEL": "custom/model-v2"
            }):
                with patch("ai_image_generator.httpx.AsyncClient") as mock_client:
                    mock_response = AsyncMock()
                    mock_response.status_code = 410
                    mock_response.text = '{"error":"deprecated"}'
                    mock_response.headers = {"content-type": "application/json"}

                    mock_instance = AsyncMock()
                    mock_instance.post.return_value = mock_response
                    mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
                    mock_instance.__aexit__ = AsyncMock(return_value=False)
                    mock_client.return_value = mock_instance

                    result = loop.run_until_complete(
                        generate_ai_image("topic", "prompt", "text")
                    )
                    self.assertIsNone(result, "410 should return None immediately")
                    self.assertEqual(mock_instance.post.call_count, 1,
                                     "410 response must not trigger retry")
        finally:
            loop.close()


# ---------------------------------------------------------------------------
# 5. Post subject overrides weak channel family
# ---------------------------------------------------------------------------
class TestPostSubjectOverridesChannelFamily(unittest.TestCase):
    """Post subject (from post_text) should dominate over channel topic for family detection."""

    def test_scooter_post_in_generic_channel_not_local_business(self):
        """A scooter post should not be forced into local_business by channel topic."""
        url = "https://pixabay.com/get/g730860eefb2fa82d6d24dddc5f5eb97fc2e05066"
        result = validate_image_for_autopost(
            url,
            topic="общие новости",
            prompt="scooter",
            post_text="Сколько стоит езда на проколотом самокате? По данным сервисных центров...",
        )
        # The important thing is: the family used for validation should be
        # derived from post_text, not forced into local_business from channel topic.
        self.assertTrue(result,
                        "Post about scooters in generic channel must not be rejected by "
                        "local_business family matching on CDN URL")

    def test_harvest_lighting_post_not_hijacked_by_channel(self):
        """A gardening/lighting post should use the post's subject family."""
        url = "https://images.pexels.com/photos/11350076/pexels-photo-11350076.jpeg?auto=compress"
        result = validate_image_for_autopost(
            url,
            topic="generic channel",
            prompt="harvest light indoor",
            post_text="Ваш урожай сожжет лампочка? Вот почему важно выбирать правильное освещение.",
        )
        self.assertTrue(result,
                        "Post about harvest/lighting must not be rejected by generic "
                        "channel family matching on asset URL")


# ---------------------------------------------------------------------------
# 6. Legacy fallback is explicit and not final by default
# ---------------------------------------------------------------------------
class TestLegacyFallbackExplicit(unittest.TestCase):
    """Legacy fallback in find_image must be explicitly logged."""

    def test_find_image_has_legacy_fallback_log(self):
        """find_image source code must contain IMAGE_LEGACY_FALLBACK_USED log."""
        import image_search
        import inspect
        source = inspect.getsource(image_search.find_image)
        self.assertIn("IMAGE_LEGACY_FALLBACK_USED", source,
                       "find_image must explicitly log when legacy fallback is used")

    def test_find_image_has_v3_accept_log(self):
        """find_image source code must contain IMAGE_V3_ACCEPT log."""
        import image_search
        import inspect
        source = inspect.getsource(image_search.find_image)
        self.assertIn("IMAGE_V3_ACCEPT", source,
                       "find_image must log IMAGE_V3_ACCEPT when v3 pipeline accepts")

    def test_find_image_has_v3_reject_log(self):
        """find_image source code must contain IMAGE_V3_REJECT log."""
        import image_search
        import inspect
        source = inspect.getsource(image_search.find_image)
        self.assertIn("IMAGE_V3_REJECT", source,
                       "find_image must log IMAGE_V3_REJECT when v3 pipeline rejects")

    def test_find_image_has_v3_no_match_log(self):
        """find_image source code must contain IMAGE_V3_NO_MATCH log."""
        import image_search
        import inspect
        source = inspect.getsource(image_search.find_image)
        self.assertIn("IMAGE_V3_NO_MATCH", source,
                       "find_image must log IMAGE_V3_NO_MATCH when v3 finds nothing")


# ---------------------------------------------------------------------------
# 7. CDN asset URL detection helper
# ---------------------------------------------------------------------------
class TestCdnAssetUrlDetection(unittest.TestCase):
    """_is_cdn_asset_url must correctly identify known CDN/asset URL patterns."""

    def test_pixabay_get_url(self):
        self.assertTrue(_is_cdn_asset_url(
            "https://pixabay.com/get/gf6172cb3dddedad1d4c7e1e9df0b2441916667f02a5a688531c8732"
        ))

    def test_pexels_photos_url(self):
        self.assertTrue(_is_cdn_asset_url(
            "https://images.pexels.com/photos/11350076/pexels-photo-11350076.jpeg?auto=compress"
        ))

    def test_unsplash_photo_url(self):
        self.assertTrue(_is_cdn_asset_url(
            "https://images.unsplash.com/photo-1234567890?w=400"
        ))

    def test_cdn_pixabay_url(self):
        self.assertTrue(_is_cdn_asset_url(
            "https://cdn.pixabay.com/photo/2020/01/01/000000_960_720.jpg"
        ))

    def test_random_url_not_cdn(self):
        self.assertFalse(_is_cdn_asset_url(
            "https://example.com/images/scooter-repair-workshop.jpg"
        ))

    def test_semantic_url_not_cdn(self):
        self.assertFalse(_is_cdn_asset_url(
            "https://blog.example.com/wp-content/uploads/massage-therapy-studio.jpg"
        ))


# ---------------------------------------------------------------------------
# 8. Decision trace logging helper
# ---------------------------------------------------------------------------
class TestDecisionTraceLogging(unittest.TestCase):
    """_log_image_decision_trace must run without errors."""

    def test_trace_log_runs_without_error(self):
        """Decision trace helper should not raise."""
        _log_image_decision_trace(
            url="https://pixabay.com/get/abc123",
            mode="autopost",
            subject="самокат",
            family="generic",
            provider="pixabay",
            has_meta=False,
            has_page_url=False,
            reject_reason="",
            accept_reason="cdn_skip",
            final_score=0,
            legacy_fallback_used=False,
        )

    def test_trace_log_with_reject(self):
        """Decision trace with reject reason should not raise."""
        _log_image_decision_trace(
            url="https://example.com/bad-image.jpg",
            mode="editor",
            subject="ремонт",
            family="local_business",
            provider="unknown",
            has_meta=True,
            has_page_url=True,
            reject_reason="cross_family",
            accept_reason="",
            final_score=-10,
            legacy_fallback_used=True,
        )


# ---------------------------------------------------------------------------
# 9. Provider detection from URL
# ---------------------------------------------------------------------------
class TestProviderFromUrl(unittest.TestCase):
    """_provider_from_url must correctly identify providers."""

    def test_pixabay(self):
        self.assertEqual(_provider_from_url("https://pixabay.com/get/abc"), "pixabay")

    def test_pexels(self):
        self.assertEqual(_provider_from_url("https://images.pexels.com/photos/123"), "pexels")

    def test_unsplash(self):
        self.assertEqual(_provider_from_url("https://images.unsplash.com/photo-123"), "unsplash")

    def test_unknown(self):
        self.assertEqual(_provider_from_url("https://example.com/image.jpg"), "unknown")


# ---------------------------------------------------------------------------
# 10. Family detection with post text priority
# ---------------------------------------------------------------------------
class TestFamilyDetectionPostTextPriority(unittest.TestCase):
    """validate_image_for_autopost should use post_text for family detection."""

    def test_non_semantic_url_with_post_text(self):
        """Post text about generic topic with CDN URL should not be rejected."""
        # This tests that family detection from post_text does not force
        # a strict family match against a CDN URL that has no keywords.
        url = "https://pixabay.com/get/ge51b4a9f91370657cc85dbeef88c980025c0a14b"
        result = validate_image_for_autopost(
            url,
            topic="Растения и освещение",
            prompt="indoor plants lighting",
            post_text="Ваш урожай сожжет лампочка? Правильное освещение — ключ к здоровой рассаде.",
        )
        self.assertTrue(result,
                        "CDN URL with generic post text should not be rejected")


if __name__ == "__main__":
    unittest.main()
