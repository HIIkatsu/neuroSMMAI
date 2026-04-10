"""
test_image_gateway.py — Tests for the unified image gateway.

Tests the new single entry point for all image operations.
"""
import os
import sys
import unittest
import inspect
from unittest.mock import patch, AsyncMock, MagicMock

os.environ.setdefault("BOT_TOKEN", "test:token")
os.environ.setdefault("OPENROUTER_API_KEY", "test-key")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from image_gateway import get_post_image, validate_image, MODE_AUTOPOST, MODE_EDITOR
from image_pipeline_v3 import PipelineResult, MODE_AUTOPOST as V3_AUTO, MODE_EDITOR as V3_ED
from visual_intent_v2 import VisualIntentV2


# ---------------------------------------------------------------------------
# 1. Gateway API contract
# ---------------------------------------------------------------------------
class TestGatewayContract(unittest.TestCase):
    """Verify the gateway has the correct function signatures."""

    def test_get_post_image_is_async(self):
        self.assertTrue(inspect.iscoroutinefunction(get_post_image))

    def test_get_post_image_returns_pipeline_result(self):
        """Return type annotation should reference PipelineResult."""
        hints = get_post_image.__annotations__
        self.assertIn("return", hints)
        # With __future__ annotations, hints are strings
        ret = str(hints["return"])
        self.assertIn("PipelineResult", ret)

    def test_validate_image_is_sync(self):
        self.assertFalse(inspect.iscoroutinefunction(validate_image))

    def test_validate_image_returns_bool(self):
        hints = validate_image.__annotations__
        self.assertIn("return", hints)
        ret = str(hints["return"])
        self.assertIn("bool", ret)

    def test_mode_constants(self):
        self.assertEqual(MODE_AUTOPOST, "autopost")
        self.assertEqual(MODE_EDITOR, "editor")

    def test_get_post_image_params(self):
        sig = inspect.signature(get_post_image)
        params = set(sig.parameters.keys())
        self.assertIn("title", params)
        self.assertIn("body", params)
        self.assertIn("channel_topic", params)
        self.assertIn("used_refs", params)
        self.assertIn("mode", params)

    def test_validate_image_params(self):
        sig = inspect.signature(validate_image)
        params = set(sig.parameters.keys())
        self.assertIn("image_ref", params)
        self.assertIn("title", params)
        self.assertIn("body", params)
        self.assertIn("channel_topic", params)
        self.assertIn("image_meta", params)
        self.assertIn("mode", params)


# ---------------------------------------------------------------------------
# 2. Gateway behavior — no content
# ---------------------------------------------------------------------------
class TestGatewayNoContent(unittest.TestCase):
    """Gateway should skip image search when no post content is provided."""

    def _run(self, coro):
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    def test_no_title_no_body_returns_no_image(self):
        result = self._run(get_post_image(title="", body=""))
        self.assertFalse(result.has_image)
        self.assertEqual(result.no_image_reason, "no_post_content")

    def test_no_title_no_body_autopost(self):
        result = self._run(get_post_image(title="", body="", mode=MODE_AUTOPOST))
        self.assertFalse(result.has_image)
        self.assertEqual(result.mode, MODE_AUTOPOST)

    def test_no_title_no_body_editor(self):
        result = self._run(get_post_image(title="", body="", mode=MODE_EDITOR))
        self.assertFalse(result.has_image)
        self.assertEqual(result.mode, MODE_EDITOR)


# ---------------------------------------------------------------------------
# 3. Gateway delegates to v3 pipeline
# ---------------------------------------------------------------------------
class TestGatewayDelegation(unittest.TestCase):
    """Gateway must delegate to run_pipeline_v3."""

    def test_source_delegates_to_pipeline(self):
        source = inspect.getsource(get_post_image)
        self.assertIn("run_pipeline_v3", source)

    def test_source_uses_mode(self):
        source = inspect.getsource(get_post_image)
        self.assertIn("effective_mode", source)


# ---------------------------------------------------------------------------
# 4. Gateway mode handling
# ---------------------------------------------------------------------------
class TestGatewayModeHandling(unittest.TestCase):
    """Gateway should correctly handle mode parameter."""

    def _run(self, coro):
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    @patch("image_gateway.run_pipeline_v3")
    def test_default_mode_is_autopost(self, mock_pipeline):
        mock_result = PipelineResult(mode=MODE_AUTOPOST)
        mock_pipeline.return_value = mock_result

        result = self._run(get_post_image(title="test"))
        mock_pipeline.assert_called_once()
        call_kwargs = mock_pipeline.call_args[1]
        self.assertEqual(call_kwargs["mode"], MODE_AUTOPOST)

    @patch("image_gateway.run_pipeline_v3")
    def test_editor_mode_passed_through(self, mock_pipeline):
        mock_result = PipelineResult(mode=MODE_EDITOR)
        mock_pipeline.return_value = mock_result

        result = self._run(get_post_image(title="test", mode=MODE_EDITOR))
        mock_pipeline.assert_called_once()
        call_kwargs = mock_pipeline.call_args[1]
        self.assertEqual(call_kwargs["mode"], MODE_EDITOR)

    @patch("image_gateway.run_pipeline_v3")
    def test_invalid_mode_defaults_to_autopost(self, mock_pipeline):
        mock_result = PipelineResult(mode=MODE_AUTOPOST)
        mock_pipeline.return_value = mock_result

        result = self._run(get_post_image(title="test", mode="invalid"))
        call_kwargs = mock_pipeline.call_args[1]
        self.assertEqual(call_kwargs["mode"], MODE_AUTOPOST)


# ---------------------------------------------------------------------------
# 5. Validation basic cases
# ---------------------------------------------------------------------------
class TestValidateBasic(unittest.TestCase):
    """Basic validation cases."""

    def test_empty_ref_always_accepted(self):
        self.assertTrue(validate_image(""))

    def test_none_like_empty(self):
        self.assertTrue(validate_image(""))

    def test_local_upload_accepted(self):
        self.assertTrue(validate_image("/uploads/image.jpg"))

    def test_telegram_file_accepted(self):
        self.assertTrue(validate_image("tgfile:AgACAgIAAxk"))

    def test_http_without_meta_accepted(self):
        """Without image_meta, no post-centric check is possible → accept."""
        self.assertTrue(validate_image(
            "https://images.pexels.com/photos/123/abc.jpg",
            title="test",
            body="some body text",
        ))


# ---------------------------------------------------------------------------
# 6. Validation with v3 pipeline
# ---------------------------------------------------------------------------
class TestValidateWithPipeline(unittest.TestCase):
    """Validation with post-centric v3 checks."""

    @patch("image_gateway.validate_image_post_centric_v3")
    @patch("image_gateway.extract_visual_intent_v2")
    def test_rejects_when_v3_rejects(self, mock_intent, mock_validate):
        mock_intent.return_value = VisualIntentV2(
            subject="car", sense="automobile", scene="road",
            imageability="HIGH", query_terms=["car"],
        )
        mock_validate.return_value = (False, "wrong_sense")

        result = validate_image(
            "https://example.com/image.jpg",
            title="Автомобили",
            body="Обзор машин",
            image_meta="food dish restaurant",
        )
        self.assertFalse(result)

    @patch("image_gateway.validate_image_post_centric_v3")
    @patch("image_gateway.extract_visual_intent_v2")
    def test_accepts_when_v3_accepts(self, mock_intent, mock_validate):
        mock_intent.return_value = VisualIntentV2(
            subject="car", sense="automobile", scene="road",
            imageability="HIGH", query_terms=["car"],
        )
        mock_validate.return_value = (True, "")

        result = validate_image(
            "https://example.com/image.jpg",
            title="Автомобили",
            body="Обзор машин",
            image_meta="car road driving",
        )
        self.assertTrue(result)


# ---------------------------------------------------------------------------
# 7. Backward compatibility — image_search shim
# ---------------------------------------------------------------------------
class TestBackwardCompatShim(unittest.TestCase):
    """image_search.py functions must still work as wrappers."""

    def test_find_image_exists(self):
        from image_search import find_image
        self.assertTrue(inspect.iscoroutinefunction(find_image))

    def test_validate_image_for_autopost_exists(self):
        from image_search import validate_image_for_autopost
        self.assertFalse(inspect.iscoroutinefunction(validate_image_for_autopost))

    def test_build_best_visual_queries_exists(self):
        from image_search import build_best_visual_queries
        self.assertTrue(callable(build_best_visual_queries))

    def test_trigger_unsplash_download_exists(self):
        from image_search import trigger_unsplash_download
        self.assertTrue(inspect.iscoroutinefunction(trigger_unsplash_download))

    def test_latin_token_re_exported(self):
        from image_search import _LATIN_TOKEN_RE
        self.assertTrue(_LATIN_TOKEN_RE.match("hello"))
        self.assertFalse(_LATIN_TOKEN_RE.match("привет"))


# ---------------------------------------------------------------------------
# 8. Integration — resolve_post_image propagates mode
# ---------------------------------------------------------------------------
class TestResolvePostImageMode(unittest.TestCase):
    """actions.resolve_post_image must accept and propagate mode."""

    def test_resolve_post_image_accepts_mode(self):
        from actions import resolve_post_image
        sig = inspect.signature(resolve_post_image)
        self.assertIn("mode", sig.parameters)

    def test_resolve_post_image_source_uses_mode(self):
        from actions import resolve_post_image
        source = inspect.getsource(resolve_post_image)
        self.assertIn("effective_mode", source)
        self.assertIn("MODE_AUTOPOST", source)
        self.assertIn("MODE_EDITOR", source)


# ---------------------------------------------------------------------------
# 9. No legacy imports in production code
# ---------------------------------------------------------------------------
class TestNoLegacyImports(unittest.TestCase):
    """Production code must not import deleted legacy modules."""

    def test_actions_no_legacy_imports(self):
        import actions
        source = inspect.getsource(actions)
        self.assertNotIn("from image_pipeline import", source)
        self.assertNotIn("from visual_intent import", source)

    def test_scheduler_no_legacy_imports(self):
        import scheduler_service
        source = inspect.getsource(scheduler_service)
        self.assertNotIn("validate_image_for_autopost", source)
        self.assertNotIn("from image_search import validate", source)

    def test_gateway_no_legacy_imports(self):
        import image_gateway
        source = inspect.getsource(image_gateway)
        self.assertNotIn("image_pipeline", source.replace("image_pipeline_v3", ""))
        self.assertNotIn("visual_intent ", source.replace("visual_intent_v2", ""))


# ---------------------------------------------------------------------------
# 10. Gateway with used_refs deduplication
# ---------------------------------------------------------------------------
class TestGatewayDedup(unittest.TestCase):
    """Gateway must pass used_refs through to pipeline."""

    def _run(self, coro):
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    @patch("image_gateway.run_pipeline_v3")
    def test_used_refs_passed_to_pipeline(self, mock_pipeline):
        mock_pipeline.return_value = PipelineResult()
        refs = {"https://old1.jpg", "https://old2.jpg"}
        self._run(get_post_image(title="test", used_refs=refs))
        call_kwargs = mock_pipeline.call_args[1]
        self.assertEqual(call_kwargs["used_refs"], refs)

    @patch("image_gateway.run_pipeline_v3")
    def test_none_used_refs_handled(self, mock_pipeline):
        mock_pipeline.return_value = PipelineResult()
        self._run(get_post_image(title="test", used_refs=None))
        call_kwargs = mock_pipeline.call_args[1]
        self.assertIsNone(call_kwargs["used_refs"])


# ---------------------------------------------------------------------------
# 11. Pipeline result structure
# ---------------------------------------------------------------------------
class TestPipelineResultStructure(unittest.TestCase):
    """PipelineResult must have all expected fields."""

    def test_has_image_property(self):
        r = PipelineResult(image_url="https://example.com/img.jpg")
        self.assertTrue(r.has_image)

    def test_no_image_property(self):
        r = PipelineResult()
        self.assertFalse(r.has_image)

    def test_trace_summary(self):
        r = PipelineResult(
            mode=MODE_AUTOPOST,
            outcome="ACCEPT_BEST",
            image_url="https://example.com/img.jpg",
            score=30,
        )
        summary = r.trace_summary()
        self.assertIn("mode", summary)
        self.assertIn("outcome", summary)
        self.assertIn("score", summary)

    def test_editor_candidates_default_empty(self):
        r = PipelineResult()
        self.assertEqual(r.editor_candidates, [])


if __name__ == "__main__":
    unittest.main()
