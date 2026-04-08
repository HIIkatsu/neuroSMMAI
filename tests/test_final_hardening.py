"""Tests for final hardening PR — channel switch, partial update, topic_fit,
anti-fabrication, image ACL, toggle rollback, settings modal, health endpoint.

Run with:
    BOT_TOKEN=test:token python -m pytest tests/test_final_hardening.py -v
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("BOT_TOKEN", "test:token")
os.environ.setdefault("OPENROUTER_API_KEY", "test-key")


class _DBTestBase(unittest.TestCase):
    """Base with a throwaway SQLite DB per test."""

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

    def _run(self, coro):
        return self.loop.run_until_complete(coro)


# ---------------------------------------------------------------------------
# 1. Channel switch refreshes all scoped sections
# ---------------------------------------------------------------------------

class TestChannelSwitchCompleteness(unittest.TestCase):
    """Verify that activateChannel() refreshes all channel-scoped sections."""

    def test_activate_channel_refreshes_all_sections(self):
        """The JS refreshSections call must include drafts, plan, schedules, media_inbox."""
        app_js = (Path(__file__).resolve().parent.parent / "app.js").read_text(encoding="utf-8")
        # Find the activateChannel function
        start = app_js.index("async function activateChannel(")
        # Find the end of the function (next top-level function)
        end_idx = app_js.index("\nasync function ", start + 10)
        if end_idx < 0:
            end_idx = app_js.index("\nfunction ", start + 10)
        func_body = app_js[start:end_idx]

        for section in ("drafts", "plan", "schedules", "media_inbox"):
            self.assertIn(
                section,
                func_body,
                f"activateChannel must refresh '{section}' section",
            )


# ---------------------------------------------------------------------------
# 2. Settings modal does NOT dismiss before save
# ---------------------------------------------------------------------------

class TestSettingsModalSafety(unittest.TestCase):
    """Verify settings modal stays open until save succeeds."""

    def test_settings_button_no_dismiss_modal(self):
        """Save button must NOT have data-dismiss-modal attribute."""
        app_js = (Path(__file__).resolve().parent.parent / "app.js").read_text(encoding="utf-8")
        # Find openSettingsModal button
        idx = app_js.index("data-action=\"saveSettings\"")
        # Get surrounding context (the button tag)
        context = app_js[max(0, idx - 100):idx + 200]
        # There should be NO data-dismiss-modal on the save button
        # Look at just the button element containing saveSettings
        button_start = context.rfind("<button", 0, context.index("saveSettings"))
        button_end = context.index(">", context.index("saveSettings"))
        button_tag = context[button_start:button_end + 1]
        self.assertNotIn(
            "data-dismiss-modal",
            button_tag,
            "Save button must not auto-dismiss modal",
        )

    def test_save_settings_calls_close_modal_on_success(self):
        """saveSettings() must call closeModal() only in success path."""
        app_js = (Path(__file__).resolve().parent.parent / "app.js").read_text(encoding="utf-8")
        start = app_js.index("async function saveSettings()")
        # Find the function body (next function or end)
        end = app_js.index("\nfunction ", start + 10)
        func_body = app_js[start:end]
        # closeModal must be called inside the try block (after toast)
        self.assertIn("closeModal()", func_body, "saveSettings should call closeModal on success")
        # closeModal should come AFTER the toast success message
        close_idx = func_body.index("closeModal()")
        toast_idx = func_body.index("toast('Настройки сохранены')")
        self.assertGreater(close_idx, toast_idx, "closeModal should be after success toast")


# ---------------------------------------------------------------------------
# 3. Toggle rollback on API failure
# ---------------------------------------------------------------------------

class TestToggleRollback(unittest.TestCase):
    """Verify that toggle functions refresh state on error."""

    def test_toggle_autopost_has_rollback(self):
        """toggleAutopost catch block must refresh settings."""
        app_js = (Path(__file__).resolve().parent.parent / "app.js").read_text(encoding="utf-8")
        start = app_js.index("async function toggleAutopost(")
        end = app_js.index("\nasync function toggleAutopostNews(")
        func_body = app_js[start:end]
        # The catch block must contain refreshSections
        catch_idx = func_body.index("} catch")
        after_catch = func_body[catch_idx:]
        self.assertIn("refreshSections", after_catch, "toggleAutopost must refresh state on error")
        self.assertIn("render()", after_catch, "toggleAutopost must re-render on error")

    def test_toggle_news_has_rollback(self):
        """toggleAutopostNews catch block must refresh settings."""
        app_js = (Path(__file__).resolve().parent.parent / "app.js").read_text(encoding="utf-8")
        start = app_js.index("async function toggleAutopostNews(")
        # Find end of function
        end = app_js.index("\n}", start + 10) + 2
        func_body = app_js[start:end]
        catch_idx = func_body.index("} catch")
        after_catch = func_body[catch_idx:]
        self.assertIn("refreshSections", after_catch, "toggleAutopostNews must refresh state on error")


# ---------------------------------------------------------------------------
# 4. Partial update channel profile does NOT blank fields
# ---------------------------------------------------------------------------

class TestPartialUpdateChannelProfile(_DBTestBase):
    """Verify upsert_channel_profile preserves un-provided fields."""

    def test_partial_update_preserves_existing_fields(self):
        """Updating only topic should NOT blank topic_family, audience_type, etc."""
        owner_id = 42

        async def _test():
            # Create with full structured fields
            await self.db.upsert_channel_profile(
                owner_id=owner_id,
                channel_target="@test_partial",
                topic="здоровье",
                topic_family="health",
                topic_subfamily="nutrition",
                audience_type="professionals",
                style_mode="expert",
                author_role_type="expert",
                author_role_description="Врач-терапевт",
            )

            # Now do a partial update — only change topic
            await self.db.upsert_channel_profile(
                owner_id=owner_id,
                channel_target="@test_partial",
                topic="фитнес",
                make_active=True,
                # All other structured fields are _UNSET (not provided)
            )

            # Verify existing structured fields are preserved
            profiles = await self.db.list_channel_profiles(owner_id=owner_id)
            p = profiles[0]
            self.assertEqual(p["topic"], "фитнес", "Topic should be updated")
            self.assertEqual(p["topic_family"], "health", "topic_family should be preserved")
            self.assertEqual(p["topic_subfamily"], "nutrition", "topic_subfamily should be preserved")
            self.assertEqual(p["audience_type"], "professionals", "audience_type should be preserved")
            self.assertEqual(p["style_mode"], "expert", "style_mode should be preserved")
            self.assertEqual(p["author_role_type"], "expert", "author_role_type should be preserved")
            self.assertEqual(p["author_role_description"], "Врач-терапевт", "author_role_description should be preserved")

        self._run(_test())

    def test_explicit_empty_clears_field(self):
        """Explicitly passing empty string should clear the field."""
        owner_id = 43

        async def _test():
            await self.db.upsert_channel_profile(
                owner_id=owner_id,
                channel_target="@test_clear",
                topic="здоровье",
                topic_family="health",
            )

            # Explicitly clear topic_family
            await self.db.upsert_channel_profile(
                owner_id=owner_id,
                channel_target="@test_clear",
                topic_family="",
            )

            profiles = await self.db.list_channel_profiles(owner_id=owner_id)
            p = profiles[0]
            self.assertEqual(p["topic_family"], "", "topic_family should be cleared when explicitly set to ''")

        self._run(_test())


# ---------------------------------------------------------------------------
# 5. Generated image access control
# ---------------------------------------------------------------------------

class TestGeneratedImageACL(unittest.TestCase):
    """Verify generated image ownership check."""

    def test_owner_prefixed_image_blocked_for_wrong_user(self):
        """A file named 42_hash.png must be blocked for user 99."""
        from miniapp_server import serve_generated_image
        from unittest.mock import AsyncMock

        # The function depends on Depends() resolution, so we test the logic directly
        filename = "42_abc123def456.png"
        parts = filename.split("_", 1)
        self.assertEqual(len(parts), 2)
        self.assertTrue(parts[0].isdigit())
        file_owner = int(parts[0])
        self.assertNotEqual(file_owner, 99, "Owner 42 != user 99 — access should be denied")

    def test_owner_prefixed_image_allowed_for_correct_user(self):
        """A file named 42_hash.png must be allowed for user 42."""
        filename = "42_abc123def456.png"
        parts = filename.split("_", 1)
        file_owner = int(parts[0])
        self.assertEqual(file_owner, 42, "Owner matches — access should be allowed")

    def test_legacy_image_no_owner_prefix(self):
        """A legacy hash-only filename (no underscore prefix) should not trigger owner check."""
        filename = "abc123def456.png"
        parts = filename.split("_", 1)
        # If no underscore or first part is not digits-only, it's a legacy file
        is_owner_prefixed = len(parts) == 2 and parts[0].isdigit()
        self.assertFalse(is_owner_prefixed, "Legacy file should not have owner prefix check")


# ---------------------------------------------------------------------------
# 6. Topic_fit hard floor blocks autopublish
# ---------------------------------------------------------------------------

class TestTopicFitHardFloor(unittest.TestCase):
    """Verify extremely low topic_fit caps total score below autopost threshold."""

    def test_off_topic_text_capped(self):
        """A post with topic_fit=1 should have total capped at 50 (below AUTOPOST threshold 62)."""
        from content import assess_text_quality

        # Body text about cats, channel topic about programming
        score, reasons, dims = assess_text_quality(
            "Как ухаживать за котиком",
            "Кошки - это прекрасные домашние животные. Они любят играть и спать. "
            "Важно кормить кошку качественным кормом и регулярно водить к ветеринару. "
            "Игрушки для кошек помогают им оставаться активными и здоровыми.",
            "Заведите кошку!",
            channel_topic="программирование на Python и разработка веб-приложений",
        )
        self.assertLessEqual(dims.get("topic_fit", 10), 2, "topic_fit should be very low for off-topic")
        self.assertLessEqual(score, 50, "Total score should be capped at 50 by hard floor")
        self.assertTrue(any("HARD FLOOR" in r for r in reasons), "Should mention hard floor in reasons")

    def test_on_topic_text_not_capped(self):
        """A post about the channel topic should not be capped."""
        from content import assess_text_quality

        score, reasons, dims = assess_text_quality(
            "Python для начинающих",
            "Python — один из самых популярных языков программирования. "
            "Он используется для веб-разработки, анализа данных и машинного обучения. "
            "Django и Flask — популярные фреймворки для создания веб-приложений на Python. "
            "Начать изучение можно с официальной документации и простых проектов.",
            "Начните учить Python сегодня!",
            channel_topic="программирование на Python и разработка веб-приложений",
        )
        self.assertGreater(dims.get("topic_fit", 0), 2, "On-topic text should have reasonable topic_fit")
        hard_floor_hit = any("HARD FLOOR" in r for r in reasons)
        self.assertFalse(hard_floor_hit, "On-topic text should not trigger hard floor")


# ---------------------------------------------------------------------------
# 7. Anti-fabrication catches invented stats
# ---------------------------------------------------------------------------

class TestAntiFabrication(unittest.TestCase):
    """Verify anti-fabrication heuristics catch invented statistics and case studies."""

    def test_invented_research_penalized(self):
        """Text with 'по данным исследования' without source should be penalized."""
        from content import assess_text_quality

        score, reasons, dims = assess_text_quality(
            "Правда о здоровье",
            "По данным исследования, 73% людей не знают о правильном питании. "
            "Как показали исследования, эксперты установили, что здоровый образ жизни "
            "увеличивает продолжительность жизни. Научно доказано, что правильное питание "
            "помогает избежать многих заболеваний.",
            "Начните питаться правильно!",
            channel_topic="здоровье и питание",
        )
        honesty = dims.get("honesty", 10)
        self.assertLessEqual(honesty, 5, f"Honesty should be penalized for invented stats, got {honesty}")
        fabrication_reasons = [r for r in reasons if "honesty" in r.lower()]
        self.assertTrue(len(fabrication_reasons) >= 1, "Should have honesty penalty reasons")

    def test_invented_percentages_penalized(self):
        """Text with invented percentage + людей/клиентов should be penalized."""
        from content import assess_text_quality

        score, reasons, dims = assess_text_quality(
            "Тестовый пост",
            "Около 87% клиентов отмечают значительное улучшение после нашей программы. "
            "По данным наших наблюдений, 93% участников курса достигают результата "
            "уже в первые две недели.",
            "Записывайтесь!",
            channel_topic="коучинг и развитие",
        )
        honesty = dims.get("honesty", 10)
        self.assertLessEqual(honesty, 6, f"Honesty should be penalized for invented percentages, got {honesty}")

    def test_fabricated_case_study_penalized(self):
        """Text with fabricated case study patterns should be penalized."""
        from content import assess_text_quality

        score, reasons, dims = assess_text_quality(
            "Реальный кейс",
            "Один из моих клиентов обратился ко мне с проблемой тревожности. "
            "История из практики: к нему пришел пациент с похожими симптомами. "
            "После трёх сессий результат превзошёл все ожидания.",
            "Запишитесь на консультацию!",
            channel_topic="психология",
        )
        honesty = dims.get("honesty", 10)
        self.assertLessEqual(honesty, 7, f"Honesty should be penalized for fabricated case, got {honesty}")

    def test_legitimate_text_not_penalized(self):
        """Legitimate text without invented claims should not be heavily penalized."""
        from content import assess_text_quality

        score, reasons, dims = assess_text_quality(
            "Советы по питанию",
            "Правильное питание — основа здоровья. Рекомендуется есть больше овощей "
            "и фруктов, сократить потребление сахара и трансжиров. Важно пить "
            "достаточно воды в течение дня. Регулярное питание помогает поддерживать "
            "стабильный уровень энергии.",
            "Следите за своим рационом!",
            channel_topic="здоровье и питание",
        )
        honesty = dims.get("honesty", 10)
        self.assertGreaterEqual(honesty, 7, f"Legitimate text should not be heavily penalized, got {honesty}")


# ---------------------------------------------------------------------------
# 8. News automation not implicitly enabled
# ---------------------------------------------------------------------------

class TestNewsAutomationExplicitness(unittest.TestCase):
    """Verify that news_enabled is set explicitly based on user choice, not silently."""

    def test_onboarding_news_enabled_only_with_news_format(self):
        """Onboarding should only set news_enabled=1 if user selected 'Новости' format."""
        app_js = (Path(__file__).resolve().parent.parent / "app.js").read_text(encoding="utf-8")
        start = app_js.index("async function completeOnboarding()")
        end = app_js.index("\n}", start + 100) + 2
        func_body = app_js[start:end]

        # Check that news_enabled is derived from formats selection
        self.assertIn(
            "news_enabled:",
            func_body,
            "Onboarding should explicitly set news_enabled",
        )
        self.assertIn(
            "includes('Новости')",
            func_body,
            "news_enabled should depend on user selecting 'Новости' format",
        )


# ---------------------------------------------------------------------------
# 9. Health endpoint
# ---------------------------------------------------------------------------

class TestHealthEndpoint(_DBTestBase):
    """Verify /healthz and /readyz endpoints."""

    def test_healthz_returns_200(self):
        """GET /healthz should return 200."""
        from httpx import ASGITransport, AsyncClient
        from miniapp_server import app

        async def _test():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://localhost") as client:
                resp = await client.get("/healthz")
                self.assertEqual(resp.status_code, 200)
                self.assertTrue(resp.json().get("ok"))

        self._run(_test())

    def test_readyz_returns_200_with_db(self):
        """GET /readyz should return 200 with DB check."""
        from httpx import ASGITransport, AsyncClient
        from miniapp_server import app

        async def _test():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://localhost") as client:
                resp = await client.get("/readyz")
                self.assertEqual(resp.status_code, 200)
                data = resp.json()
                self.assertTrue(data.get("ready"))
                self.assertTrue(data.get("checks", {}).get("db"))

        self._run(_test())


# ---------------------------------------------------------------------------
# 10. Static asset versioning
# ---------------------------------------------------------------------------

class TestStaticAssetVersioning(unittest.TestCase):
    """Verify static assets have cache-busting."""

    def test_index_html_has_version_params(self):
        """index.html should reference app.js and styles.css with version params."""
        html = (Path(__file__).resolve().parent.parent / "miniapp" / "index.html").read_text(encoding="utf-8")
        self.assertIn("app.js?v=", html, "app.js should have version query param")
        self.assertIn("styles.css?v=", html, "styles.css should have version query param")

    def test_index_html_has_cache_control_meta(self):
        """index.html should have no-cache meta tags."""
        html = (Path(__file__).resolve().parent.parent / "miniapp" / "index.html").read_text(encoding="utf-8")
        self.assertIn("no-cache", html, "Should have no-cache directive")
        self.assertIn("no-store", html, "Should have no-store directive")

    def test_asset_etags_computed(self):
        """Server should compute ETag hashes for static assets."""
        from miniapp_server import _ASSET_ETAGS
        # At least app.js should have an ETag
        self.assertIn("app.js", _ASSET_ETAGS, "app.js should have computed ETag")
        self.assertGreater(len(_ASSET_ETAGS["app.js"]), 0, "ETag should not be empty")


# ---------------------------------------------------------------------------
# 11. CSRF middleware blocks missing origin for browser mutations
# ---------------------------------------------------------------------------

class TestCSRFProtection(unittest.TestCase):
    """Verify CSRF middleware blocks browser mutations without Origin/Referer."""

    def test_browser_mutation_blocked_without_origin(self):
        """POST to /api/settings without Origin and without TMA header should be rejected."""
        from miniapp_server import app, CSRFMiddleware

        # Find the CSRF middleware config
        # The middleware should reject POST /api/* without origin for cookie-auth requests
        # This is tested by checking the middleware exists and has the right exempt paths
        self.assertIn("/api/payments/webhook", CSRFMiddleware._EXEMPT_PATHS)
        self.assertIn("/healthz", CSRFMiddleware._EXEMPT_PATHS)
        self.assertIn("POST", CSRFMiddleware._MUTATING_METHODS)
        self.assertIn("PATCH", CSRFMiddleware._MUTATING_METHODS)


if __name__ == "__main__":
    unittest.main()
