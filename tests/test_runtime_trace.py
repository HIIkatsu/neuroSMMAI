"""Runtime tracing and canonical path tests.

Validates that:
1. All text generation flows go through the canonical ``generate_post_bundle``
2. All image flows go through the canonical ``resolve_post_image`` → ``find_image``
3. Channel label responses prefer ``display_label`` over raw target/id
4. Runtime tracing emits structured trace events
5. No stale fallback to legacy prompt builders

Run with:
    BOT_TOKEN=test:token python -m pytest tests/test_runtime_trace.py -v
"""
from __future__ import annotations

import asyncio
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure the project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("BOT_TOKEN", "test:token")
os.environ.setdefault("OPENROUTER_API_KEY", "test-key")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _DBBase(unittest.TestCase):
    """Base class that provisions a temporary SQLite DB for each test."""

    def setUp(self):
        import db
        self.db = db
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self._orig_path = db.DB_PATH
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        db.DB_PATH = Path(self._tmp.name)
        db._db_pool = None
        self.loop.run_until_complete(db.init_db())

    def tearDown(self):
        self.loop.run_until_complete(self.db.close_pool())
        self.db.DB_PATH = self._orig_path
        self.loop.close()
        try:
            os.unlink(self._tmp.name)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# 1. Runtime trace module unit tests
# ---------------------------------------------------------------------------

class TestRuntimeTraceModule(unittest.TestCase):
    """Test runtime_trace module fundamentals."""

    def test_new_trace_id_is_unique(self):
        from runtime_trace import new_trace_id
        ids = {new_trace_id() for _ in range(100)}
        self.assertEqual(len(ids), 100, "trace IDs should be unique")

    def test_trace_id_format(self):
        from runtime_trace import new_trace_id
        tid = new_trace_id()
        self.assertEqual(len(tid), 12)
        self.assertTrue(tid.isalnum())

    def test_trace_text_generation_returns_payload(self):
        from runtime_trace import trace_text_generation, new_trace_id
        payload = trace_text_generation(
            trace_id=new_trace_id(),
            route="test_route",
            source_mode="manual",
            requested_topic="test topic",
            channel_topic="channel topic",
        )
        self.assertEqual(payload["event"], "text_generation")
        self.assertEqual(payload["route"], "test_route")
        self.assertEqual(payload["source_mode"], "manual")

    def test_trace_image_selection_returns_payload(self):
        from runtime_trace import trace_image_selection, new_trace_id
        payload = trace_image_selection(
            trace_id=new_trace_id(),
            route="test_route",
            built_query="cat photo",
            provider_result_count=5,
        )
        self.assertEqual(payload["event"], "image_selection")
        self.assertEqual(payload["provider_result_count"], 5)

    def test_trace_channel_label_returns_payload(self):
        from runtime_trace import trace_channel_label, new_trace_id
        payload = trace_channel_label(
            trace_id=new_trace_id(),
            channel_profile_id=42,
            channel_target="@mychannel",
            channel_title="My Channel",
            display_label="My Channel",
            route="/api/channels",
        )
        self.assertEqual(payload["event"], "channel_label")
        self.assertEqual(payload["display_label"], "My Channel")

    def test_debug_fields_off_by_default(self):
        from runtime_trace import debug_fields
        result = debug_fields({"trace_id": "abc", "event": "test"})
        self.assertIsNone(result)

    def test_debug_fields_on_when_env_set(self):
        from runtime_trace import debug_fields
        with patch.dict(os.environ, {"DEBUG_RUNTIME_TRACE": "1"}):
            result = debug_fields({"trace_id": "abc", "event": "test", "secret_key": "xxx"})
            self.assertIsNotNone(result)
            self.assertEqual(result["trace_id"], "abc")
            self.assertNotIn("secret_key", result)

    def test_trace_timer(self):
        from runtime_trace import TraceTimer
        import time
        timer = TraceTimer()
        timer.__enter__()
        time.sleep(0.01)
        timer.__exit__(None, None, None)
        self.assertGreater(timer.elapsed_ms, 0)

    def test_is_debug_trace_disabled_by_default(self):
        from runtime_trace import is_debug_trace_enabled
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("DEBUG_RUNTIME_TRACE", None)
            self.assertFalse(is_debug_trace_enabled())

    def test_is_debug_trace_enabled_with_flag(self):
        from runtime_trace import is_debug_trace_enabled
        with patch.dict(os.environ, {"DEBUG_RUNTIME_TRACE": "1"}):
            self.assertTrue(is_debug_trace_enabled())


# ---------------------------------------------------------------------------
# 2. Canonical text generation path tests
# ---------------------------------------------------------------------------

class TestCanonicalTextGenerationPath(unittest.TestCase):
    """Verify all text generation routes go through generate_post_bundle."""

    def test_generate_post_payload_calls_generate_post_bundle(self):
        """actions.generate_post_payload must call content.generate_post_bundle."""
        import actions
        # The function is imported at module level — verify it references the right one
        from content import generate_post_bundle as canonical
        self.assertIs(actions.generate_post_bundle, canonical)

    def test_generate_post_text_calls_generate_post_bundle(self):
        """content.generate_post_text must delegate to generate_post_bundle."""
        import content
        import inspect
        source = inspect.getsource(content.generate_post_text)
        self.assertIn("generate_post_bundle", source,
                       "generate_post_text must call generate_post_bundle")

    def test_news_service_uses_generate_post_text(self):
        """news_service.build_news_post must call generate_post_text (which calls generate_post_bundle)."""
        import news_service
        import inspect
        source = inspect.getsource(news_service.build_news_post)
        self.assertIn("generate_post_text", source,
                       "build_news_post must call generate_post_text")

    def test_scheduler_uses_generate_post_payload(self):
        """scheduler_service must call generate_post_payload for autopost."""
        import scheduler_service
        # Check the import exists
        self.assertTrue(hasattr(scheduler_service, 'generate_post_payload'))

    def test_no_stale_prompt_builder_in_routes(self):
        """miniapp_routes_content should not contain inline prompt assembly for generation."""
        import miniapp_routes_content
        import inspect
        source = inspect.getsource(miniapp_routes_content.ai_generate_text)
        # The route should delegate to generate_post_bundle, not build prompts inline
        self.assertIn("generate_post_bundle", source)
        # Should not contain raw prompt templates for generation
        self.assertNotIn("Ты копирайтер", source)
        self.assertNotIn("Напиши пост", source)

    def test_generate_post_bundle_is_single_entry_point(self):
        """All generation paths must go through generate_post_bundle, not alternatives."""
        import inspect
        import actions
        import content

        # actions.generate_post_payload must call generate_post_bundle
        payload_source = inspect.getsource(actions.generate_post_payload)
        self.assertIn("generate_post_bundle", payload_source)

        # content.generate_post_text must call generate_post_bundle
        text_source = inspect.getsource(content.generate_post_text)
        self.assertIn("generate_post_bundle", text_source)

    def test_editorial_engine_not_used_for_generation(self):
        """editorial_engine should not be imported in generation routes."""
        import miniapp_routes_content
        import inspect
        source = inspect.getsource(miniapp_routes_content)
        self.assertNotIn("from editorial_engine import", source)
        self.assertNotIn("import editorial_engine", source)


# ---------------------------------------------------------------------------
# 3. Canonical image path tests
# ---------------------------------------------------------------------------

class TestCanonicalImagePath(unittest.TestCase):
    """Verify editor and autopost image selection go through canonical path."""

    def test_resolve_post_image_calls_image_service(self):
        """actions.resolve_post_image must call get_image from image_service."""
        import actions
        import inspect
        source = inspect.getsource(actions.resolve_post_image)
        self.assertIn("get_image", source)

    def test_generate_post_payload_uses_image_service(self):
        """actions.generate_post_payload must use the image service for image generation."""
        import actions
        import inspect
        source = inspect.getsource(actions.generate_post_payload)
        self.assertIn("get_image", source)

    def test_resolve_post_image_has_trace_id_param(self):
        """resolve_post_image should accept trace_id for end-to-end tracing."""
        import actions
        import inspect
        sig = inspect.signature(actions.resolve_post_image)
        self.assertIn("trace_id", sig.parameters)

    def test_image_service_single_entry_point(self):
        """image_service.get_image should be the single entry point."""
        import image_service
        self.assertTrue(hasattr(image_service, 'get_image'))
        import inspect
        source = inspect.getsource(image_service.get_image)
        self.assertIn("generate_image", source)
        self.assertIn("search_stock_photo", source)


# ---------------------------------------------------------------------------
# 4. Channel label / display_label tests
# ---------------------------------------------------------------------------

class TestChannelDisplayLabel(_DBBase):
    """Verify channel label resolution prefers title/display_label over raw target/id."""

    def test_enrich_display_label_prefers_title(self):
        from miniapp_bootstrap_service import enrich_display_label
        ch = {"title": "My Channel", "channel_target": "@mychannel"}
        enrich_display_label(ch)
        self.assertEqual(ch["display_label"], "My Channel")

    def test_enrich_display_label_falls_back_to_target(self):
        from miniapp_bootstrap_service import enrich_display_label
        ch = {"title": "", "channel_target": "@mychannel"}
        enrich_display_label(ch)
        self.assertEqual(ch["display_label"], "@mychannel")

    def test_enrich_display_label_rejects_numeric_title(self):
        from miniapp_bootstrap_service import enrich_display_label
        ch = {"title": "-1001234567890", "channel_target": "-1001234567890"}
        enrich_display_label(ch)
        self.assertEqual(ch["display_label"], "Канал без названия")

    def test_enrich_display_label_rejects_numeric_target(self):
        from miniapp_bootstrap_service import enrich_display_label
        ch = {"title": "", "channel_target": "-1001234567890"}
        enrich_display_label(ch)
        self.assertEqual(ch["display_label"], "Канал без названия")

    def test_enrich_display_label_none_channel(self):
        from miniapp_bootstrap_service import enrich_display_label
        result = enrich_display_label(None)
        self.assertIsNone(result)

    def test_enrich_display_label_at_prefixed_target(self):
        from miniapp_bootstrap_service import enrich_display_label
        ch = {"title": "", "channel_target": "@best_channel"}
        enrich_display_label(ch)
        self.assertEqual(ch["display_label"], "@best_channel")

    def test_bootstrap_enriches_channels(self):
        """bootstrap_core_payload should add display_label to channels."""
        owner_id = 99001
        self.loop.run_until_complete(
            self.db.upsert_channel_profile(
                owner_id=owner_id,
                channel_target="@testchan",
                title="Test Title",
                topic="test topic",
                make_active=True,
            )
        )
        payload = self.loop.run_until_complete(
            __import__('miniapp_bootstrap_service').bootstrap_core_payload(owner_id)
        )
        active = payload.get("active_channel")
        self.assertIsNotNone(active)
        self.assertEqual(active.get("display_label"), "Test Title")

    def test_bootstrap_numeric_id_channel_gets_fallback_label(self):
        """When channel_target is numeric ID and title is empty, display_label should be fallback."""
        owner_id = 99002
        self.loop.run_until_complete(
            self.db.upsert_channel_profile(
                owner_id=owner_id,
                channel_target="-1001234567890",
                title="",
                topic="test",
                make_active=True,
            )
        )
        payload = self.loop.run_until_complete(
            __import__('miniapp_bootstrap_service').bootstrap_core_payload(owner_id)
        )
        active = payload.get("active_channel")
        self.assertIsNotNone(active)
        # display_label should NOT be the raw numeric ID
        self.assertNotEqual(active.get("display_label"), "-1001234567890")
        self.assertFalse(re.match(r'^-?\d+$', active.get("display_label", "")))


# ---------------------------------------------------------------------------
# 5. Trace integration in generate_post_bundle
# ---------------------------------------------------------------------------

class TestTraceInGeneratePostBundle(unittest.TestCase):
    """Verify that generate_post_bundle emits trace events."""

    def test_bundle_source_contains_trace_calls(self):
        """generate_post_bundle source code should contain trace_text_generation calls."""
        import content
        import inspect
        source = inspect.getsource(content.generate_post_bundle)
        self.assertIn("trace_text_generation", source)
        self.assertIn("_trace_id", source)
        self.assertIn("_trace_timer", source)

    def test_resolve_post_image_contains_trace_calls(self):
        """resolve_post_image source code should contain trace_image_selection calls."""
        import actions
        import inspect
        source = inspect.getsource(actions.resolve_post_image)
        self.assertIn("trace_image_selection", source)
        self.assertIn("_tid", source)

    def test_generate_post_payload_contains_trace_calls(self):
        """generate_post_payload should contain trace calls."""
        import actions
        import inspect
        source = inspect.getsource(actions.generate_post_payload)
        self.assertIn("trace_text_generation", source)
        self.assertIn("trace_image_selection", source)
        self.assertIn("_trace_id", source)


# ---------------------------------------------------------------------------
# 6. Verify no stale/legacy prompt builders
# ---------------------------------------------------------------------------

class TestNoStaleFallback(unittest.TestCase):
    """Ensure no old/legacy prompt builders are used in the current runtime paths."""

    def test_no_route_local_prompt_assembly(self):
        """Routes should not assemble LLM prompts inline (except rewrite which is intentionally different)."""
        import miniapp_routes_content
        import inspect
        # Check ai_generate_text doesn't have inline prompt
        gen_text_src = inspect.getsource(miniapp_routes_content.ai_generate_text)
        self.assertNotIn("Ты копирайтер", gen_text_src)
        self.assertNotIn("Напиши пост для", gen_text_src)

    def test_generate_post_payload_delegates_to_bundle(self):
        """generate_post_payload should not have inline LLM prompt assembly."""
        import actions
        import inspect
        source = inspect.getsource(actions.generate_post_payload)
        # Should call generate_post_bundle, not ai_chat directly
        self.assertIn("generate_post_bundle", source)
        self.assertNotIn("ai_chat(", source)

    def test_scheduler_does_not_use_direct_ai_chat(self):
        """Scheduler autopost path should use generate_post_payload, not direct ai_chat."""
        import scheduler_service
        import inspect
        # _job_post_regular should call generate_post_payload
        source = inspect.getsource(scheduler_service.SchedulerService._job_post_regular)
        self.assertIn("generate_post_payload", source)


# ---------------------------------------------------------------------------
# 7. Debug mode flag tests
# ---------------------------------------------------------------------------

class TestDebugMode(unittest.TestCase):
    """Verify DEBUG_RUNTIME_TRACE flag controls debug output."""

    def test_debug_fields_returns_none_when_off(self):
        from runtime_trace import debug_fields
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("DEBUG_RUNTIME_TRACE", None)
            self.assertIsNone(debug_fields({"trace_id": "x"}))

    def test_debug_fields_returns_dict_when_on(self):
        from runtime_trace import debug_fields
        with patch.dict(os.environ, {"DEBUG_RUNTIME_TRACE": "1"}):
            result = debug_fields({"trace_id": "x", "event": "test"})
            self.assertIsNotNone(result)
            self.assertIn("trace_id", result)

    def test_debug_fields_excludes_unsafe_keys(self):
        from runtime_trace import debug_fields
        with patch.dict(os.environ, {"DEBUG_RUNTIME_TRACE": "1"}):
            result = debug_fields({
                "trace_id": "x",
                "api_key": "secret",
                "full_prompt": "long text",
                "route": "test",
            })
            self.assertNotIn("api_key", result)
            self.assertNotIn("full_prompt", result)
            self.assertIn("route", result)


if __name__ == "__main__":
    unittest.main()
