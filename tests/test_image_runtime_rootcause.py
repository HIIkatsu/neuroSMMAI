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
from image_gateway import validate_image
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
        result = validate_image(
            url,
            title="scooter tire flat repair cost",
            body="По данным сервисных центров, 4 из 10 самокатов имеют проблемы с шинами.",
            channel_topic="Сколько стоит езда на проколотом самокате",
        )
        self.assertTrue(result,
                        "Pixabay /get/ URL must not be rejected solely by URL keyword mismatch")

    def test_pixabay_cdn_url_passes_validation(self):
        """cdn.pixabay.com URL should pass without metadata."""
        url = "https://cdn.pixabay.com/photo/2020/01/01/000000_960_720.jpg"
        result = validate_image(
            url,
            title="harvest light bulb",
            body="Ваш урожай сожжет лампочка? Вот почему важно выбирать правильное освещение.",
            channel_topic="Урожай и лампочка",
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
        result = validate_image(
            url,
            title="harvest lighting indoor plants grow",
            body="Неправильное освещение может уничтожить рассаду за неделю.",
            channel_topic="Ваш урожай сожжет лампочка? Вот почему",
        )
        self.assertTrue(result,
                        "Pexels asset URL must not be rejected solely by URL having no topic keywords")

    def test_pexels_url_with_different_topic(self):
        """Pexels photo URL with generic topic should also pass."""
        url = "https://images.pexels.com/photos/99999/pexels-photo-99999.jpeg?auto=compress&w=600"
        result = validate_image(
            url,
            title="garden tool selection",
            body="Правильный инструмент экономит 3 часа работы в неделю.",
            channel_topic="Как выбрать правильный инструмент для дачи",
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
                    mock_instance.post.assert_called()
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
        result = validate_image(
            url,
            title="scooter",
            body="Сколько стоит езда на проколотом самокате? По данным сервисных центров...",
            channel_topic="общие новости",
        )
        # The important thing is: the family used for validation should be
        # derived from post_text, not forced into local_business from channel topic.
        self.assertTrue(result,
                        "Post about scooters in generic channel must not be rejected by "
                        "local_business family matching on CDN URL")

    def test_harvest_lighting_post_not_hijacked_by_channel(self):
        """A gardening/lighting post should use the post's subject family."""
        url = "https://images.pexels.com/photos/11350076/pexels-photo-11350076.jpeg?auto=compress"
        result = validate_image(
            url,
            title="harvest light indoor",
            body="Ваш урожай сожжет лампочка? Вот почему важно выбирать правильное освещение.",
            channel_topic="generic channel",
        )
        self.assertTrue(result,
                        "Post about harvest/lighting must not be rejected by generic "
                        "channel family matching on asset URL")


# ---------------------------------------------------------------------------
# 6. Image gateway logs are present in the unified pipeline
# ---------------------------------------------------------------------------
class TestUnifiedGatewayLogs(unittest.TestCase):
    """image_gateway.get_post_image must have structured logging."""

    def test_gateway_has_accept_log(self):
        """image_gateway source must contain IMAGE_GATEWAY ACCEPT log."""
        import image_gateway
        import inspect
        source = inspect.getsource(image_gateway.get_post_image)
        self.assertIn("IMAGE_GATEWAY", source)
        self.assertIn("ACCEPT", source,
                       "get_post_image must log ACCEPT when image is found")

    def test_gateway_has_no_image_log(self):
        """image_gateway source must contain IMAGE_GATEWAY NO_IMAGE log."""
        import image_gateway
        import inspect
        source = inspect.getsource(image_gateway.get_post_image)
        self.assertIn("NO_IMAGE", source,
                       "get_post_image must log NO_IMAGE when no image found")

    def test_gateway_has_skip_log(self):
        """image_gateway source must log when skipping due to no content."""
        import image_gateway
        import inspect
        source = inspect.getsource(image_gateway.get_post_image)
        self.assertIn("no_post_content", source,
                       "get_post_image must log when skipping due to no content")

    def test_get_post_image_delegates_to_pipeline(self):
        """get_post_image in image_gateway must delegate to run_pipeline_v3."""
        import image_gateway
        import inspect
        source = inspect.getsource(image_gateway.get_post_image)
        self.assertIn("run_pipeline_v3", source,
                       "get_post_image must delegate to run_pipeline_v3")


# ---------------------------------------------------------------------------
# 7. Image validation accepts CDN URLs correctly
# ---------------------------------------------------------------------------
class TestCdnUrlValidation(unittest.TestCase):
    """validate_image must handle CDN URLs without false rejection."""

    def test_pixabay_cdn_url_not_rejected(self):
        """Pixabay CDN URL should not be rejected for generic posts."""
        result = validate_image(
            "https://pixabay.com/get/gf6172cb3dddedad1d4c7e1e9df0b2441916667f02a5a688531c8732",
            title="общий",
            body="Интересные факты о мире.",
        )
        self.assertTrue(result)

    def test_pexels_cdn_url_not_rejected(self):
        """Pexels CDN URL should not be rejected for generic posts."""
        result = validate_image(
            "https://images.pexels.com/photos/11350076/pexels-photo-11350076.jpeg?auto=compress",
            title="общий",
            body="Новые тренды этого года.",
        )
        self.assertTrue(result)


# ---------------------------------------------------------------------------
# 8. Image gateway validate_image function
# ---------------------------------------------------------------------------
class TestGatewayValidation(unittest.TestCase):
    """image_gateway.validate_image must work for basic cases."""

    def test_validate_empty_ref(self):
        """Empty image ref should be accepted (nothing to reject)."""
        from image_gateway import validate_image
        self.assertTrue(validate_image(""))

    def test_validate_local_ref(self):
        """Local file path should be accepted."""
        from image_gateway import validate_image
        self.assertTrue(validate_image("/uploads/image.jpg"))

    def test_validate_http_without_meta(self):
        """HTTP URL without image_meta should be accepted (no data to reject on)."""
        from image_gateway import validate_image
        self.assertTrue(validate_image(
            "https://example.com/image.jpg",
            title="test",
            body="test body",
        ))


# ---------------------------------------------------------------------------
# 9. Gateway mode propagation
# ---------------------------------------------------------------------------
class TestGatewayModePropagation(unittest.TestCase):
    """image_gateway must correctly handle mode parameter."""

    def test_editor_mode_accepted(self):
        """Editor mode should be recognized."""
        from image_gateway import MODE_EDITOR
        self.assertEqual(MODE_EDITOR, "editor")

    def test_autopost_mode_accepted(self):
        """Autopost mode should be recognized."""
        from image_gateway import MODE_AUTOPOST
        self.assertEqual(MODE_AUTOPOST, "autopost")

    def test_resolve_post_image_has_mode_param(self):
        """actions.resolve_post_image must accept mode parameter."""
        import inspect
        from actions import resolve_post_image
        sig = inspect.signature(resolve_post_image)
        self.assertIn("mode", sig.parameters,
                       "resolve_post_image must accept mode parameter")


# ---------------------------------------------------------------------------
# 10. Family detection with post text priority
# ---------------------------------------------------------------------------
class TestFamilyDetectionPostTextPriority(unittest.TestCase):
    """validate_image should use post_text for family detection."""

    def test_non_semantic_url_with_post_text(self):
        """Post text about generic topic with CDN URL should not be rejected."""
        # This tests that family detection from post_text does not force
        # a strict family match against a CDN URL that has no keywords.
        url = "https://pixabay.com/get/ge51b4a9f91370657cc85dbeef88c980025c0a14b"
        result = validate_image(
            url,
            title="indoor plants lighting",
            body="Ваш урожай сожжет лампочка? Правильное освещение — ключ к здоровой рассаде.",
            channel_topic="Растения и освещение",
        )
        self.assertTrue(result,
                        "CDN URL with generic post text should not be rejected")


if __name__ == "__main__":
    unittest.main()
