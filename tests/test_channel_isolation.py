"""Multi-channel settings isolation tests.

Verifies that channel-scoped settings are read exclusively from the
``channel_profiles`` table, and that saving settings for Channel A
never contaminates Channel B.

Run with:
    BOT_TOKEN=test:token python -m pytest tests/test_channel_isolation.py -v
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure the project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("BOT_TOKEN", "test:token")
os.environ.setdefault("OPENROUTER_API_KEY", "test-key")


class _ChannelIsolationBase(unittest.TestCase):
    """Base class that provisions a temporary SQLite DB for each test."""

    def setUp(self):
        import db
        self.db = db
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self._orig_path = db.DB_PATH
        # Use a real temp file so every _connect() sees the same DB
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


class TestChannelSettingsIsolation(_ChannelIsolationBase):
    """Core isolation: get_channel_settings returns ONLY profile data."""

    # ---------------------------------------------------------------
    # Helper: create a channel profile directly
    # ---------------------------------------------------------------
    async def _create_profile(self, owner_id: int, channel_target: str, **kwargs):
        await self.db.upsert_channel_profile(
            owner_id=owner_id,
            channel_target=channel_target,
            topic=kwargs.get("topic", ""),
            make_active=kwargs.get("make_active", False),
            **{k: v for k, v in kwargs.items() if k not in ("topic", "make_active")},
        )
        profiles = await self.db.list_channel_profiles(owner_id=owner_id)
        return next(
            (p for p in profiles if p.get("channel_target") == channel_target),
            None,
        )

    # ---------------------------------------------------------------
    # Test: two channels for same owner stay isolated
    # ---------------------------------------------------------------
    def test_two_channels_isolated(self):
        """Settings saved for channel A must NOT leak into channel B."""

        async def _test():
            owner = 42

            # Create channel A (active) with topic=Finance
            pa = await self._create_profile(owner, "@chan_a", topic="Финансы", make_active=True)
            self.assertIsNotNone(pa)
            pid_a = int(pa["id"])

            # Save channel_style for channel A
            await self.db.save_channel_setting(owner, "channel_style", "formal_business", channel_profile_id=pid_a)

            # Create channel B (not active) with topic=Gaming
            pb = await self._create_profile(owner, "@chan_b", topic="Гейминг", make_active=False)
            self.assertIsNotNone(pb)
            pid_b = int(pb["id"])

            # Verify channel A returns its own settings
            settings_a = await self.db.get_channel_settings(owner, channel_profile_id=pid_a)
            self.assertEqual(settings_a["topic"], "Финансы")
            self.assertEqual(settings_a["channel_style"], "formal_business")

            # Verify channel B does NOT inherit channel A's style
            settings_b = await self.db.get_channel_settings(owner, channel_profile_id=pid_b)
            self.assertEqual(settings_b["topic"], "Гейминг")
            self.assertEqual(settings_b["channel_style"], "", "Channel B must NOT inherit A's style")

        self._run(_test())

    # ---------------------------------------------------------------
    # Test: save_channel_setting does NOT write to owner-level settings
    # ---------------------------------------------------------------
    def test_save_does_not_write_owner_level(self):
        """Saving a channel-scoped key must not create an owner-level settings row."""

        async def _test():
            owner = 43
            pa = await self._create_profile(owner, "@chan_c", topic="Массаж", make_active=True)
            pid = int(pa["id"])

            # Save via the new save_channel_setting
            await self.db.save_channel_setting(owner, "channel_audience", "бизнесмены 35+", channel_profile_id=pid)

            # Verify it's in channel profile
            settings = await self.db.get_channel_settings(owner, channel_profile_id=pid)
            self.assertEqual(settings["channel_audience"], "бизнесмены 35+")

            # Verify owner-level settings table does NOT have it
            owner_val = await self.db.get_setting("channel_audience", owner_id=owner)
            self.assertIsNone(owner_val, "Channel-scoped key must NOT be in owner-level settings")

        self._run(_test())

    # ---------------------------------------------------------------
    # Test: get_channel_settings does NOT fall back to owner-level
    # ---------------------------------------------------------------
    def test_no_owner_fallback_for_channel_keys(self):
        """Even if owner-level settings exist, get_channel_settings must ignore them."""

        async def _test():
            owner = 44
            # Seed owner-level settings directly (simulating legacy data)
            await self.db.set_setting("channel_style", "legacy_owner_style", owner_id=owner)
            await self.db.set_setting("topic", "Legacy Topic", owner_id=owner)

            # Create a channel profile with NO style set
            pa = await self._create_profile(owner, "@chan_d", topic="Тестовая тема", make_active=True)
            pid = int(pa["id"])

            settings = await self.db.get_channel_settings(owner, channel_profile_id=pid)

            # Must NOT get "legacy_owner_style" — profile has empty string
            self.assertEqual(settings["channel_style"], "")

            # topic comes from the profile itself
            self.assertEqual(settings["topic"], "Тестовая тема")

        self._run(_test())

    # ---------------------------------------------------------------
    # Test: active channel switching returns correct data
    # ---------------------------------------------------------------
    def test_active_channel_switching(self):
        """get_channel_settings(owner) must return active channel's data only."""

        async def _test():
            owner = 45

            # Create and activate channel A
            pa = await self._create_profile(owner, "@chan_e", topic="AI новости", make_active=True)
            pid_a = int(pa["id"])
            await self.db.save_channel_setting(owner, "channel_style", "tech_expert", channel_profile_id=pid_a)

            # Create and activate channel B (deactivates A)
            pb = await self._create_profile(owner, "@chan_f", topic="Кулинария", make_active=True)
            pid_b = int(pb["id"])
            await self.db.save_channel_setting(owner, "channel_style", "friendly_cook", channel_profile_id=pid_b)

            # get_channel_settings without explicit profile_id should return active (B)
            settings = await self.db.get_channel_settings(owner)
            self.assertEqual(settings["topic"], "Кулинария")
            self.assertEqual(settings["channel_style"], "friendly_cook")

            # Explicitly requesting A should still return A's data
            settings_a = await self.db.get_channel_settings(owner, channel_profile_id=pid_a)
            self.assertEqual(settings_a["topic"], "AI новости")
            self.assertEqual(settings_a["channel_style"], "tech_expert")

        self._run(_test())

    # ---------------------------------------------------------------
    # Test: no profile -> empty strings, NOT owner-level data
    # ---------------------------------------------------------------
    def test_no_profile_returns_empty(self):
        """If no channel profile exists, return empty strings for all keys."""

        async def _test():
            owner = 46
            # Seed some owner-level data
            await self.db.set_setting("topic", "owner_topic", owner_id=owner)
            await self.db.set_setting("channel_audience", "owner_audience", owner_id=owner)

            # No profiles created for this owner
            settings = await self.db.get_channel_settings(owner)
            self.assertEqual(settings["topic"], "")
            self.assertEqual(settings["channel_audience"], "")
            self.assertEqual(settings["channel_target"], "")

        self._run(_test())

    # ---------------------------------------------------------------
    # Test: save without profile falls back to owner-level (safe)
    # ---------------------------------------------------------------
    def test_save_without_profile_falls_back(self):
        """When no profile exists, save_channel_setting should write to owner-level."""

        async def _test():
            owner = 47
            # No profile for this owner
            await self.db.save_channel_setting(owner, "topic", "some_topic")

            # Should be in owner-level settings as fallback
            val = await self.db.get_setting("topic", owner_id=owner)
            self.assertEqual(val, "some_topic")

        self._run(_test())


class TestEditorialEngineChannelScoped(_ChannelIsolationBase):
    """Verify editorial_engine reads from channel profile, not flat settings."""

    def test_load_editorial_settings_uses_channel_profile(self):
        """load_editorial_settings must read from channel profile, not owner-level."""

        async def _test():
            from editorial_engine import load_editorial_settings

            owner = 50
            # Seed DIFFERENT data in owner-level settings vs channel profile
            await self.db.set_setting("topic", "OWNER_TOPIC", owner_id=owner)
            await self.db.set_setting("channel_audience", "OWNER_AUDIENCE", owner_id=owner)

            await self.db.upsert_channel_profile(
                owner_id=owner,
                channel_target="@edit_chan",
                topic="PROFILE_TOPIC",
                make_active=True,
            )
            profiles = await self.db.list_channel_profiles(owner_id=owner)
            pid = int(profiles[0]["id"])
            await self.db.save_channel_setting(owner, "channel_audience", "PROFILE_AUDIENCE", channel_profile_id=pid)

            settings = await load_editorial_settings(owner)
            self.assertEqual(settings.get("topic"), "PROFILE_TOPIC")
            self.assertEqual(settings.get("channel_audience"), "PROFILE_AUDIENCE")

        self._run(_test())


class TestNewsCandidatesChannelScoped(_ChannelIsolationBase):
    """Verify get_channel_settings resolves correct profile for news."""

    def test_non_active_channel_gets_own_topic(self):
        """Requesting settings for a non-active channel_profile_id must return that channel's topic."""

        async def _test():
            owner = 55

            # Active channel: tech topic
            await self.db.upsert_channel_profile(
                owner_id=owner, channel_target="@tech_chan",
                topic="AI и технологии", make_active=True,
            )

            # Non-active channel: cooking topic
            await self.db.upsert_channel_profile(
                owner_id=owner, channel_target="@cook_chan",
                topic="Кулинария и рецепты", make_active=False,
            )

            profiles = await self.db.list_channel_profiles(owner_id=owner)
            cook_profile = next(p for p in profiles if p["channel_target"] == "@cook_chan")
            cook_pid = int(cook_profile["id"])

            # Verify get_channel_settings for cook profile returns cooking topic
            cook_settings = await self.db.get_channel_settings(owner, channel_profile_id=cook_pid)
            self.assertEqual(cook_settings["topic"], "Кулинария и рецепты")

            # Verify active profile returns tech topic
            active_settings = await self.db.get_channel_settings(owner)
            self.assertEqual(active_settings["topic"], "AI и технологии")

        self._run(_test())


if __name__ == "__main__":
    unittest.main()
