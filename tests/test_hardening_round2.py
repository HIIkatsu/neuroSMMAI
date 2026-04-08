"""Tests for hardening round 2 — channel profile pinning, plan diversity,
channel title fallback, multi-channel generation isolation, density scoring.

Run with:
    BOT_TOKEN=test:token python -m pytest tests/test_hardening_round2.py -v
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

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
# 1. Multi-channel settings isolation for content generation
# ---------------------------------------------------------------------------

class TestMultiChannelGenerationIsolation(_DBTestBase):
    """Two channels with different settings must produce independent contexts."""

    async def _setup_two_channels(self):
        """Create two channels for owner=100 with very different settings."""
        owner = 100
        await self.db.upsert_channel_profile(
            owner_id=owner,
            channel_target="@finance_channel",
            topic="Личные финансы и инвестиции",
            make_active=True,
            title="Финансовый канал",
        )
        profiles = await self.db.list_channel_profiles(owner_id=owner)
        pid_a = int(next(p for p in profiles if p["channel_target"] == "@finance_channel")["id"])
        await self.db.save_channel_setting(owner, "channel_style", "деловой экспертный", channel_profile_id=pid_a)
        await self.db.save_channel_setting(owner, "channel_audience", "инвесторы 30+", channel_profile_id=pid_a)
        await self.db.save_channel_setting(owner, "author_role_type", "expert", channel_profile_id=pid_a)

        await self.db.upsert_channel_profile(
            owner_id=owner,
            channel_target="@cooking_channel",
            topic="Домашняя кулинария и рецепты",
            make_active=False,
            title="Кулинарный канал",
        )
        profiles = await self.db.list_channel_profiles(owner_id=owner)
        pid_b = int(next(p for p in profiles if p["channel_target"] == "@cooking_channel")["id"])
        await self.db.save_channel_setting(owner, "channel_style", "тёплый дружелюбный", channel_profile_id=pid_b)
        await self.db.save_channel_setting(owner, "channel_audience", "домохозяйки 25-45", channel_profile_id=pid_b)
        await self.db.save_channel_setting(owner, "author_role_type", "blogger", channel_profile_id=pid_b)

        return owner, pid_a, pid_b

    def test_get_channel_settings_with_explicit_profile_id(self):
        """Explicit channel_profile_id always returns that channel's settings, not active."""

        async def _test():
            owner, pid_a, pid_b = await self._setup_two_channels()

            # Active is finance (pid_a), but explicitly request cooking (pid_b)
            cooking = await self.db.get_channel_settings(owner, channel_profile_id=pid_b)
            self.assertEqual(cooking["topic"], "Домашняя кулинария и рецепты")
            self.assertEqual(cooking["channel_style"], "тёплый дружелюбный")
            self.assertEqual(cooking["author_role_type"], "blogger")

            # Without explicit id — should return active (finance)
            active = await self.db.get_channel_settings(owner)
            self.assertEqual(active["topic"], "Личные финансы и инвестиции")
            self.assertEqual(active["channel_style"], "деловой экспертный")

        self._run(_test())

    def test_switching_active_does_not_affect_explicit_reads(self):
        """After switching active channel, explicit profile reads still work."""

        async def _test():
            owner, pid_a, pid_b = await self._setup_two_channels()

            # Switch active to cooking
            await self.db.set_active_channel_profile(pid_b, owner_id=owner)

            # Explicit read of finance should still return finance settings
            finance = await self.db.get_channel_settings(owner, channel_profile_id=pid_a)
            self.assertEqual(finance["topic"], "Личные финансы и инвестиции")
            self.assertEqual(finance["author_role_type"], "expert")

            # Active read should now return cooking
            active = await self.db.get_channel_settings(owner)
            self.assertEqual(active["topic"], "Домашняя кулинария и рецепты")

        self._run(_test())


# ---------------------------------------------------------------------------
# 2. Plan service loads settings via channel_profile_id
# ---------------------------------------------------------------------------

class TestPlanServiceChannelIsolation(_DBTestBase):
    """_load_strategy_settings must pass channel_profile_id to get_channel_settings."""

    def test_load_strategy_settings_passes_channel_profile_id(self):
        """_load_strategy_settings should use the given channel_profile_id."""

        async def _test():
            from miniapp_plan_service import _load_strategy_settings

            owner = 200
            await self.db.upsert_channel_profile(
                owner_id=owner,
                channel_target="@chan_x",
                topic="IT безопасность",
                make_active=True,
                title="IT канал",
            )
            profiles = await self.db.list_channel_profiles(owner_id=owner)
            pid = int(profiles[0]["id"])
            await self.db.save_channel_setting(owner, "channel_audience", "айтишники 25-40", channel_profile_id=pid)

            # Call with explicit channel_profile_id
            settings = await _load_strategy_settings(owner, "IT безопасность", channel_profile_id=pid)
            self.assertEqual(settings["channel_audience"], "айтишники 25-40")
            self.assertEqual(settings["topic"], "IT безопасность")

        self._run(_test())

    def test_load_strategy_settings_uses_topic_fallback(self):
        """When profile has no topic, _load_strategy_settings uses the passed topic."""

        async def _test():
            from miniapp_plan_service import _load_strategy_settings

            owner = 201
            await self.db.upsert_channel_profile(
                owner_id=owner,
                channel_target="@empty_chan",
                topic="",
                make_active=True,
                title="Пустой канал",
            )

            settings = await _load_strategy_settings(owner, "Новая тема")
            self.assertEqual(settings["topic"], "Новая тема")

        self._run(_test())


# ---------------------------------------------------------------------------
# 3. Channel title display fallback
# ---------------------------------------------------------------------------

class TestChannelTitleFallback(unittest.TestCase):
    """Verify app.js has proper channel title fallback logic."""

    def test_numeric_id_blocked_in_active_channel_title(self):
        """activeChannelTitle() must block numeric IDs and return fallback."""
        app_js = (Path(__file__).resolve().parent.parent / "app.js").read_text(encoding="utf-8")
        # Must have regex that blocks numeric-only channel targets
        self.assertIn("/^-?\\d+$/", app_js, "Should have numeric ID blocking regex")
        self.assertIn("Канал без названия", app_js, "Should have fallback label for numeric IDs")

    def test_resolve_channel_label_exists(self):
        """resolveChannelLabel function should exist and handle fallbacks."""
        app_js = (Path(__file__).resolve().parent.parent / "app.js").read_text(encoding="utf-8")
        self.assertIn("function resolveChannelLabel", app_js)
        # Should fall back to activeChannelTitle() when no match found
        self.assertIn("activeChannelTitle()", app_js)


# ---------------------------------------------------------------------------
# 4. Density scoring tighter targets
# ---------------------------------------------------------------------------

class TestDensityScoring(unittest.TestCase):
    """Verify updated density scoring with 60-120 word target."""

    def _assess(self, body_words: int):
        """Generate text with approximate word count and assess."""
        from content import assess_text_quality
        # Build a text that's roughly body_words long
        word = "тестовое слово для проверки длины"
        body = " ".join([word] * (body_words // 4 + 1))
        # Trim to approximate target
        words = body.split()[:body_words]
        body = " ".join(words)

        score, reasons, dims = assess_text_quality(
            title="Тестовый заголовок для проверки",
            body=body,
            cta="Напишите что думаете?",
            channel_topic="тестовая тема",
        )
        return score, reasons, dims

    def test_short_text_penalized(self):
        """Text under 40 words should be heavily penalized in density."""
        _, _, dims = self._assess(30)
        self.assertLessEqual(dims["density"], 3, "Under 40 words should score density ≤ 3")

    def test_long_text_penalized(self):
        """Text over 180 words should get density penalty."""
        _, reasons, dims = self._assess(200)
        density_reasons = [r for r in reasons if "density" in r.lower() and "цель" in r.lower()]
        self.assertTrue(
            len(density_reasons) > 0 or dims["density"] < 10,
            "Over 180 words should get density penalty or reason"
        )

    def test_optimal_length_not_penalized(self):
        """Text between 60-120 words should not get length penalty."""
        _, reasons, dims = self._assess(90)
        density_length_reasons = [r for r in reasons if "density" in r.lower() and ("цель" in r.lower() or "длинн" in r.lower())]
        # Should have no length-related density penalty
        self.assertEqual(
            len(density_length_reasons), 0,
            f"90-word text should not have length penalty, got: {density_length_reasons}"
        )


# ---------------------------------------------------------------------------
# 5. Format rotation: no news boost bias
# ---------------------------------------------------------------------------

class TestFormatRotationDiversity(unittest.TestCase):
    """Verify that format rotation doesn't over-boost news."""

    def test_no_news_extra_weight(self):
        """build_plan_format_rotation should NOT give news extra weight."""
        from channel_strategy import build_plan_format_rotation
        settings = {
            "topic": "Фитнес",
            "channel_audience": "спортсмены",
            "channel_formats": json.dumps(["Новости", "Разборы", "Полезные советы", "FAQ"]),
            "news_enabled": "1",
        }
        rotation = build_plan_format_rotation(settings, 20)
        # Count news items in rotation
        news_count = sum(1 for item in rotation if item["label"] == "Новости")
        practical_count = sum(1 for item in rotation if item["label"] in {"Разборы", "Полезные советы", "FAQ", "Кейсы"})
        # Practical formats should outnumber news (since they get boosted)
        self.assertGreater(
            practical_count, news_count,
            f"Practical formats ({practical_count}) should outnumber news ({news_count})"
        )

    def test_practical_formats_boosted(self):
        """Разборы, Полезные советы, FAQ, Кейсы should be boosted in rotation."""
        from channel_strategy import build_plan_format_rotation
        settings = {
            "topic": "Маркетинг",
            "channel_audience": "маркетологи",
            "channel_formats": json.dumps(["Разборы", "Новости", "FAQ", "Кейсы"]),
            "news_enabled": "0",
        }
        rotation = build_plan_format_rotation(settings, 12)
        labels = [item["label"] for item in rotation]
        # At least some practical formats should appear more than once in first 6
        practical_in_first_6 = sum(1 for l in labels[:6] if l in {"Разборы", "FAQ", "Кейсы"})
        self.assertGreaterEqual(practical_in_first_6, 3, "Practical formats should dominate first 6 slots")


# ---------------------------------------------------------------------------
# 6. Text budget reduction
# ---------------------------------------------------------------------------

class TestTextBudget(unittest.TestCase):
    """Verify text budget was reduced for shorter posts."""

    def test_text_only_budget_reduced(self):
        """AUTOPOST_TEXT_BUDGET should be ≤ 1800 (was 2400)."""
        from content import AUTOPOST_TEXT_BUDGET
        self.assertLessEqual(AUTOPOST_TEXT_BUDGET, 1800, "Text budget should be tightened")

    def test_caption_budget_unchanged(self):
        """AUTOPOST_CAPTION_BUDGET should remain at 900."""
        from content import AUTOPOST_CAPTION_BUDGET
        self.assertEqual(AUTOPOST_CAPTION_BUDGET, 900)


# ---------------------------------------------------------------------------
# 7. Active channel profile ID helper
# ---------------------------------------------------------------------------

class TestActiveChannelProfileIdHelper(_DBTestBase):
    """_active_channel_profile_id should return the correct ID."""

    def test_returns_active_profile_id(self):
        """Helper returns the correct active profile ID."""

        async def _test():
            from miniapp_routes_content import _active_channel_profile_id

            owner = 300
            await self.db.upsert_channel_profile(
                owner_id=owner,
                channel_target="@chan_active",
                topic="Активный канал",
                make_active=True,
                title="Активный",
            )
            profiles = await self.db.list_channel_profiles(owner_id=owner)
            expected_pid = int(profiles[0]["id"])

            pid = await _active_channel_profile_id(owner)
            self.assertEqual(pid, expected_pid)

        self._run(_test())

    def test_returns_none_for_no_profile(self):
        """Helper returns None when no profile exists."""

        async def _test():
            from miniapp_routes_content import _active_channel_profile_id
            pid = await _active_channel_profile_id(999999)
            self.assertIsNone(pid)

        self._run(_test())


# ---------------------------------------------------------------------------
# 8. Onboarding is per-channel (not per-owner)
# ---------------------------------------------------------------------------

class TestOnboardingPerChannel(_DBTestBase):
    """Onboarding completion is channel-scoped, not owner-scoped."""

    def test_second_channel_not_inherited_onboarding(self):
        """A new channel should NOT inherit onboarding_completed from the first channel."""

        async def _test():
            owner = 400
            # Create channel A with onboarding_completed
            await self.db.upsert_channel_profile(
                owner_id=owner,
                channel_target="@chan_a",
                topic="Тема A",
                make_active=True,
                title="Канал A",
            )
            profiles = await self.db.list_channel_profiles(owner_id=owner)
            pid_a = int(profiles[0]["id"])
            await self.db.save_channel_setting(owner, "onboarding_completed", "1", channel_profile_id=pid_a)

            # Create channel B — should NOT have onboarding_completed
            await self.db.upsert_channel_profile(
                owner_id=owner,
                channel_target="@chan_b",
                topic="",
                make_active=False,
                title="Канал B",
            )
            profiles = await self.db.list_channel_profiles(owner_id=owner)
            pid_b = int(next(p for p in profiles if p["channel_target"] == "@chan_b")["id"])

            settings_b = await self.db.get_channel_settings(owner, channel_profile_id=pid_b)
            self.assertNotEqual(
                settings_b.get("onboarding_completed"), "1",
                "New channel should not inherit onboarding_completed from first channel"
            )

        self._run(_test())


# ---------------------------------------------------------------------------
# 9. Content routes pin channel_profile_id (code structure test)
# ---------------------------------------------------------------------------

class TestContentRoutesPinChannelId(unittest.TestCase):
    """Content generation routes must call _active_channel_profile_id."""

    def test_generate_text_pins_cpid(self):
        """ai_generate_text should pin channel_profile_id at request start."""
        import inspect
        from miniapp_routes_content import ai_generate_text
        source = inspect.getsource(ai_generate_text)
        self.assertIn("_active_channel_profile_id", source)
        self.assertIn("channel_profile_id=cpid", source)

    def test_rewrite_pins_cpid(self):
        """ai_rewrite should pin channel_profile_id."""
        import inspect
        from miniapp_routes_content import ai_rewrite
        source = inspect.getsource(ai_rewrite)
        self.assertIn("_active_channel_profile_id", source)

    def test_generate_post_pins_cpid(self):
        """ai_generate_post should pin channel_profile_id."""
        import inspect
        from miniapp_routes_content import ai_generate_post
        source = inspect.getsource(ai_generate_post)
        self.assertIn("_active_channel_profile_id", source)

    def test_plan_generate_pins_cpid(self):
        """generate_plan should pin channel_profile_id."""
        import inspect
        from miniapp_routes_content import generate_plan
        source = inspect.getsource(generate_plan)
        self.assertIn("_active_channel_profile_id", source)
        self.assertIn("channel_profile_id=cpid", source)

    def test_competitor_spy_pins_cpid(self):
        """competitor_spy should pin channel_profile_id."""
        import inspect
        from miniapp_routes_content import competitor_spy
        source = inspect.getsource(competitor_spy)
        self.assertIn("_active_channel_profile_id", source)


# ---------------------------------------------------------------------------
# 10. News automation not enabled by default
# ---------------------------------------------------------------------------

class TestNewsNotEnabledByDefault(_DBTestBase):
    """News should not be silently enabled without explicit user choice."""

    def test_new_channel_news_disabled_by_default(self):
        """A freshly created channel should have news_enabled='' (not '1')."""

        async def _test():
            owner = 500
            await self.db.upsert_channel_profile(
                owner_id=owner,
                channel_target="@new_chan",
                topic="Новая тема",
                make_active=True,
                title="Новый канал",
            )
            profiles = await self.db.list_channel_profiles(owner_id=owner)
            pid = int(profiles[0]["id"])
            settings = await self.db.get_channel_settings(owner, channel_profile_id=pid)
            news_val = settings.get("news_enabled", "")
            self.assertIn(news_val, ("", "0"), "news_enabled should default to disabled")

        self._run(_test())


# ---------------------------------------------------------------------------
# 11. Autopost disabled mode guard
# ---------------------------------------------------------------------------

class TestAutopostDisabledGuard(unittest.TestCase):
    """When posts_enabled is not '1', scheduler must not publish."""

    def test_scheduler_checks_posts_enabled(self):
        """Scheduler _job_post_regular checks posts_enabled before publishing."""
        import inspect
        import scheduler_service
        source = inspect.getsource(scheduler_service)
        # The _job_post_regular method must check posts_enabled
        self.assertIn("posts_enabled", source, "Scheduler must check posts_enabled")


if __name__ == "__main__":
    unittest.main()
