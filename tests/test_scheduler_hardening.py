"""Scheduler hardening tests — durable dedup, cooldown, tier gate, bounded memory.

Run with:
    BOT_TOKEN=test:token python -m pytest tests/test_scheduler_hardening.py -v
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
from zoneinfo import ZoneInfo

# Ensure the project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("BOT_TOKEN", "test:token")
os.environ.setdefault("OPENROUTER_API_KEY", "test-key")


class _SchedulerTestBase(unittest.TestCase):
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

    def _run(self, coro):
        return self.loop.run_until_complete(coro)


# ---------------------------------------------------------------------------
# 1. Durable dedup tests
# ---------------------------------------------------------------------------


class TestDurableDedup(_SchedulerTestBase):
    """Verify that scheduler_dedup table prevents re-processing after restart."""

    def test_dedup_key_not_found_initially(self):
        """A fresh dedup key should not be found."""
        found = self._run(self.db.check_scheduler_dedup("schedule:42:1:2025-01-01 10:00"))
        self.assertFalse(found)

    def test_set_and_check_dedup(self):
        """After setting a dedup key, check_scheduler_dedup should return True."""
        key = "schedule:42:1:2025-01-01 10:00"
        self._run(self.db.set_scheduler_dedup(key, trigger_type="schedule", owner_id=42))
        found = self._run(self.db.check_scheduler_dedup(key))
        self.assertTrue(found, "Dedup key should be found after set")

    def test_dedup_survives_pool_reset(self):
        """Simulate restart by closing pool and reinitializing — dedup must persist."""
        key = "plan:99:555:2025-06-15 14:00"
        self._run(self.db.set_scheduler_dedup(key, trigger_type="plan", owner_id=99))

        # Simulate restart: close pool, re-init
        self._run(self.db.close_pool())
        self.db._db_pool = None
        self._run(self.db._init_pool())

        # Key should still be there
        found = self._run(self.db.check_scheduler_dedup(key))
        self.assertTrue(found, "Dedup key must survive pool reset (simulated restart)")

    def test_dedup_idempotent_insert(self):
        """Inserting the same key twice should not raise or duplicate."""
        key = "schedule:1:2:2025-01-01 10:00"
        self._run(self.db.set_scheduler_dedup(key, trigger_type="schedule", owner_id=1))
        # Second insert should succeed (INSERT OR IGNORE)
        self._run(self.db.set_scheduler_dedup(key, trigger_type="schedule", owner_id=1))
        found = self._run(self.db.check_scheduler_dedup(key))
        self.assertTrue(found)

    def test_cleanup_removes_old_entries(self):
        """cleanup_scheduler_dedup should remove entries older than threshold."""
        key_old = "schedule:1:1:old"
        key_recent = "schedule:1:2:recent"

        # Insert old entry with a timestamp 72 hours ago
        old_ts = (datetime.utcnow() - timedelta(hours=72)).isoformat(timespec="seconds")
        recent_ts = datetime.utcnow().isoformat(timespec="seconds")

        async def _seed():
            async with self.db._db_ctx() as conn:
                await conn.execute(
                    "INSERT INTO scheduler_dedup(dedup_key, trigger_type, owner_id, created_at) VALUES(?,?,?,?)",
                    (key_old, "schedule", 1, old_ts),
                )
                await conn.execute(
                    "INSERT INTO scheduler_dedup(dedup_key, trigger_type, owner_id, created_at) VALUES(?,?,?,?)",
                    (key_recent, "schedule", 1, recent_ts),
                )
                await conn.commit()

        self._run(_seed())

        # Cleanup with 48h threshold
        deleted = self._run(self.db.cleanup_scheduler_dedup(max_age_hours=48))
        self.assertGreaterEqual(deleted, 1)

        # Old key should be gone, recent should remain
        self.assertFalse(self._run(self.db.check_scheduler_dedup(key_old)))
        self.assertTrue(self._run(self.db.check_scheduler_dedup(key_recent)))


# ---------------------------------------------------------------------------
# 2. Cooldown fail-closed tests
# ---------------------------------------------------------------------------


class TestCooldownFailClosed(_SchedulerTestBase):
    """Verify that corrupted timestamps block publication (fail-closed)."""

    def test_corrupted_timestamp_blocks_publish(self):
        """A corrupted last-post timestamp must cause cooldown to return False."""
        from scheduler_service import _check_post_cooldown

        # Patch get_last_post_time to return a corrupted string
        with patch("scheduler_service.get_last_post_time", new_callable=AsyncMock, return_value="NOT-A-DATE"):
            result = self._run(_check_post_cooldown(42, "Europe/Moscow", channel_target="@testchan"))
            self.assertFalse(result, "Corrupted timestamp must block publication (fail-closed)")

    def test_valid_old_timestamp_allows_publish(self):
        """A valid timestamp older than the cooldown gap should allow publication."""
        from scheduler_service import _check_post_cooldown, _MIN_POST_GAP_MINUTES

        old_ts = (datetime.now(ZoneInfo("Europe/Moscow")) - timedelta(minutes=_MIN_POST_GAP_MINUTES + 10)).isoformat()
        with patch("scheduler_service.get_last_post_time", new_callable=AsyncMock, return_value=old_ts):
            result = self._run(_check_post_cooldown(42, "Europe/Moscow", channel_target="@testchan"))
            self.assertTrue(result, "Old enough timestamp should allow publication")

    def test_recent_timestamp_blocks_publish(self):
        """A valid timestamp within the cooldown gap should block publication."""
        from scheduler_service import _check_post_cooldown, _MIN_POST_GAP_MINUTES

        recent_ts = (datetime.now(ZoneInfo("Europe/Moscow")) - timedelta(minutes=_MIN_POST_GAP_MINUTES - 5)).isoformat()
        with patch("scheduler_service.get_last_post_time", new_callable=AsyncMock, return_value=recent_ts):
            result = self._run(_check_post_cooldown(42, "Europe/Moscow", channel_target="@testchan"))
            self.assertFalse(result, "Recent timestamp should block publication")

    def test_no_last_post_allows_publish(self):
        """No previous post at all should allow publication."""
        from scheduler_service import _check_post_cooldown

        with patch("scheduler_service.get_last_post_time", new_callable=AsyncMock, return_value=None):
            result = self._run(_check_post_cooldown(42, "Europe/Moscow", channel_target="@testchan"))
            self.assertTrue(result, "No previous post should allow publication")

    def test_empty_string_timestamp_allows_publish(self):
        """Empty string (no post) should allow publication (same as None)."""
        from scheduler_service import _check_post_cooldown

        with patch("scheduler_service.get_last_post_time", new_callable=AsyncMock, return_value=""):
            result = self._run(_check_post_cooldown(42, "Europe/Moscow", channel_target="@testchan"))
            # get_last_post_time returns None for empty; but if it returned "",
            # the code should handle it gracefully
            self.assertTrue(result)


# ---------------------------------------------------------------------------
# 3. Plan tick tier gate tests
# ---------------------------------------------------------------------------


class TestPlanTickTierGate(_SchedulerTestBase):
    """Verify that free-tier users cannot get auto-publish through plan tick."""

    def test_free_tier_plan_item_not_published(self):
        """A free-tier user's plan item must be skipped by _job_plan_tick."""
        from scheduler_service import SchedulerService

        # Create a mock bot
        mock_bot = MagicMock()
        mock_bot._config = MagicMock()
        svc = SchedulerService(mock_bot, "Europe/Moscow")

        # Create a plan item for a free-tier user
        owner_id = 777

        async def _setup_and_run():
            # Create a channel profile
            await self.db.upsert_channel_profile(
                owner_id=owner_id,
                channel_target="@free_chan",
                topic="Free Topic",
                make_active=True,
            )
            profiles = await self.db.list_channel_profiles(owner_id=owner_id)
            cpid = int(profiles[0]["id"])

            # Register user as owner
            await self.db.set_setting("channel_target", "@free_chan", owner_id=owner_id)

            # Create a plan item due right now
            now_str = datetime.now(ZoneInfo("Europe/Moscow")).strftime("%Y-%m-%d %H:%M")
            created_str = datetime.utcnow().isoformat(timespec="seconds")
            async with self.db._db_ctx() as conn:
                await conn.execute(
                    "INSERT INTO plan_items(dt, kind, payload, created_at, enabled, posted, owner_id, channel_profile_id) VALUES(?,?,?,?,?,?,?,?)",
                    (now_str, "topic", "Test Topic", created_str, 1, 0, owner_id, cpid),
                )
                await conn.commit()

            # Mock _is_paid_tier to return False (free tier)
            with patch("scheduler_service._is_paid_tier", new_callable=AsyncMock, return_value=False):
                # Mock _job_post_plan_item to track if it's called
                with patch.object(svc, "_job_post_plan_item", new_callable=AsyncMock) as mock_publish:
                    await svc._job_plan_tick()
                    mock_publish.assert_not_called()

        self._run(_setup_and_run())

    def test_paid_tier_plan_item_proceeds(self):
        """A paid-tier user's plan item should proceed (not blocked by tier gate)."""
        from scheduler_service import SchedulerService

        mock_bot = MagicMock()
        mock_bot._config = MagicMock()
        svc = SchedulerService(mock_bot, "Europe/Moscow")

        owner_id = 888

        async def _setup_and_run():
            await self.db.upsert_channel_profile(
                owner_id=owner_id,
                channel_target="@paid_chan",
                topic="Paid Topic",
                make_active=True,
            )
            profiles = await self.db.list_channel_profiles(owner_id=owner_id)
            cpid = int(profiles[0]["id"])

            await self.db.set_setting("channel_target", "@paid_chan", owner_id=owner_id)

            now_str = datetime.now(ZoneInfo("Europe/Moscow")).strftime("%Y-%m-%d %H:%M")
            created_str = datetime.utcnow().isoformat(timespec="seconds")
            async with self.db._db_ctx() as conn:
                await conn.execute(
                    "INSERT INTO plan_items(dt, kind, payload, created_at, enabled, posted, owner_id, channel_profile_id) VALUES(?,?,?,?,?,?,?,?)",
                    (now_str, "topic", "Paid Topic", created_str, 1, 0, owner_id, cpid),
                )
                await conn.commit()

            with patch("scheduler_service._is_paid_tier", new_callable=AsyncMock, return_value=True):
                with patch("scheduler_service._check_post_cooldown", new_callable=AsyncMock, return_value=True):
                    with patch.object(svc, "_job_post_plan_item", new_callable=AsyncMock) as mock_publish:
                        await svc._job_plan_tick()
                        mock_publish.assert_called_once()

        self._run(_setup_and_run())


# ---------------------------------------------------------------------------
# 4. Bounded runtime memory tests
# ---------------------------------------------------------------------------


class TestBoundedRuntimeMemory(unittest.TestCase):
    """Verify that in-memory caches don't grow unbounded."""

    def test_channel_recent_angles_bounded(self):
        """_CHANNEL_RECENT_ANGLES must be trimmed when it exceeds _MAX_ANGLES_CHANNELS."""
        from scheduler_service import (
            _CHANNEL_RECENT_ANGLES,
            _MAX_ANGLES_CHANNELS,
            SchedulerService,
        )

        # Fill the dict beyond the limit
        _CHANNEL_RECENT_ANGLES.clear()
        for i in range(_MAX_ANGLES_CHANNELS + 100):
            _CHANNEL_RECENT_ANGLES[(i, f"@chan_{i}")] = [0, 1, 2]

        self.assertGreater(len(_CHANNEL_RECENT_ANGLES), _MAX_ANGLES_CHANNELS)

        # Create service and trigger trim
        mock_bot = MagicMock()
        svc = SchedulerService(mock_bot, "Europe/Moscow")
        svc._trim_dedup_caches()

        self.assertLessEqual(
            len(_CHANNEL_RECENT_ANGLES),
            _MAX_ANGLES_CHANNELS,
            "_CHANNEL_RECENT_ANGLES should be trimmed to max size",
        )
        _CHANNEL_RECENT_ANGLES.clear()  # cleanup

    def test_last_schedule_run_bounded(self):
        """_last_schedule_run must be trimmed when it exceeds _DEDUP_MAX_SIZE."""
        from scheduler_service import _DEDUP_MAX_SIZE, SchedulerService

        mock_bot = MagicMock()
        svc = SchedulerService(mock_bot, "Europe/Moscow")

        # Fill beyond limit
        for i in range(_DEDUP_MAX_SIZE + 100):
            svc._last_schedule_run[(i, i)] = f"2025-01-01 10:{i:02d}"

        self.assertGreater(len(svc._last_schedule_run), _DEDUP_MAX_SIZE)

        svc._trim_dedup_caches()

        self.assertLessEqual(
            len(svc._last_schedule_run),
            _DEDUP_MAX_SIZE,
            "_last_schedule_run should be trimmed",
        )


# ---------------------------------------------------------------------------
# 5. News tick dedup / safety tests
# ---------------------------------------------------------------------------


class TestNewsDedupSafety(_SchedulerTestBase):
    """Verify that news dedup persists in DB and corrupted timestamps block news."""

    def test_news_log_prevents_reuse(self):
        """A logged news URL should be detected as used."""
        self._run(self.db.log_news("https://example.com/news1", "Title 1", owner_id=42, channel_target="@chan"))
        used = self._run(self.db.is_news_used("https://example.com/news1", owner_id=42, channel_target="@chan"))
        self.assertTrue(used, "Logged news URL must be detected as used")

    def test_news_log_survives_restart(self):
        """News log must survive pool reset (simulated restart)."""
        self._run(self.db.log_news("https://example.com/news2", "Title 2", owner_id=42, channel_target="@chan"))

        # Simulate restart
        self._run(self.db.close_pool())
        self.db._db_pool = None
        self._run(self.db._init_pool())

        used = self._run(self.db.is_news_used("https://example.com/news2", owner_id=42, channel_target="@chan"))
        self.assertTrue(used, "News dedup must survive restart")

    def test_news_not_reused_across_channels(self):
        """A news URL used for one channel should also be detected for another channel
        of the same owner (owner-level dedup fallback)."""
        self._run(self.db.log_news("https://example.com/news3", "Title 3", owner_id=42, channel_target="@chan_a"))

        # Check for a different channel of the same owner
        used = self._run(self.db.is_news_used("https://example.com/news3", owner_id=42, channel_target="@chan_b"))
        self.assertTrue(used, "News URL should be detected as used across channels for same owner")


# ---------------------------------------------------------------------------
# 6. Health endpoint tests
# ---------------------------------------------------------------------------


class TestHealthEndpoints(unittest.TestCase):
    """Verify health and readiness endpoints exist and respond correctly."""

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

    def test_healthz_returns_200(self):
        """GET /healthz should return 200 with ok=True."""
        from httpx import ASGITransport, AsyncClient
        from miniapp_server import app

        async def _test():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://localhost") as client:
                resp = await client.get("/healthz")
                self.assertEqual(resp.status_code, 200)
                self.assertTrue(resp.json().get("ok"))

        self.loop.run_until_complete(_test())

    def test_readyz_returns_200_with_db(self):
        """GET /readyz should return 200 with ready=True when DB is accessible."""
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

        self.loop.run_until_complete(_test())


if __name__ == "__main__":
    unittest.main()
