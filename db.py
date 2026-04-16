
from __future__ import annotations

import asyncio
import re
import aiosqlite
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

DB_PATH = Path(__file__).resolve().parent / "bot.db"

# ---------------------------------------------------------------------------
# Connection pool
# ---------------------------------------------------------------------------
# A small pool of pre-configured aiosqlite connections avoids re-opening and
# re-configuring a new SQLite connection (WAL, pragmas) on every DB call.
# Pool size is intentionally small: SQLite is single-writer, so large pools
# only help for concurrent reads. 5 connections is a safe, tested default.
# ---------------------------------------------------------------------------
_DB_POOL_SIZE = 5
_db_pool: asyncio.Queue | None = None


async def _init_pool() -> None:
    """Pre-fill the connection pool. Must be called once from init_db()."""
    global _db_pool
    new_pool: asyncio.Queue = asyncio.Queue(maxsize=_DB_POOL_SIZE)
    for _ in range(_DB_POOL_SIZE):
        conn = await _connect()
        new_pool.put_nowait(conn)
    _db_pool = new_pool


async def close_pool() -> None:
    """Drain and close all pooled connections. Call on application shutdown."""
    global _db_pool
    pool = _db_pool
    _db_pool = None
    if pool is None:
        return
    while not pool.empty():
        try:
            conn = pool.get_nowait()
            await conn.close()
        except Exception:
            pass


class _db_ctx:
    """Async context manager that acquires a connection from the pool.

    If the pool is empty or not yet initialized, falls back to creating a
    fresh connection so callers never block indefinitely.

    Usage::

        async with _db_ctx() as db:
            cur = await db.execute(...)
    """

    _conn: aiosqlite.Connection | None = None
    _from_pool: bool = False

    async def __aenter__(self) -> aiosqlite.Connection:
        if _db_pool is not None:
            try:
                self._conn = _db_pool.get_nowait()
                self._from_pool = True
                return self._conn
            except asyncio.QueueEmpty:
                pass
        self._conn = await _connect()
        self._from_pool = False
        return self._conn

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        conn = self._conn
        self._conn = None
        if conn is None:
            return
        if exc_type is not None:
            # Roll back any open transaction so the connection stays clean
            try:
                await conn.rollback()
            except Exception:
                pass
        if self._from_pool and _db_pool is not None:
            try:
                _db_pool.put_nowait(conn)
                return
            except asyncio.QueueFull:
                pass
        try:
            await conn.close()
        except Exception:
            pass


async def _connect() -> aiosqlite.Connection:
    db = await aiosqlite.connect(DB_PATH)
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA synchronous=NORMAL")
    await db.execute("PRAGMA temp_store=MEMORY")
    await db.execute("PRAGMA cache_size=-20000")
    return db


def _scope_key(key: str, owner_id: int | None = None) -> str:
    if owner_id in (None, 0):
        return key
    return f"u:{int(owner_id)}:{key}"


def _topic_signature(text: str) -> str:
    raw = str(text or '').lower()
    raw = ''.join(ch if ch.isalnum() or ch.isspace() else ' ' for ch in raw)
    words = [w for w in raw.split() if len(w) >= 4]
    return ' '.join(words[:6]).strip()


# ---------------------------------------------------------------------------
# SQL identifier whitelist — prevents SQL injection via dynamic table/column names.
# Only tables and columns that appear in our schema are allowed.
# ---------------------------------------------------------------------------
_ALLOWED_TABLES = frozenset({
    "settings", "schedules", "plan_items", "post_logs", "draft_posts",
    "generation_history", "user_media_inbox", "news_logs", "channel_profiles",
    "dm_memory", "subscriptions", "feature_quotas", "payment_events",
    "user_subscriptions", "user_feature_quotas", "schema_versions",
    "scheduler_dedup",
})

_ALLOWED_COLUMNS = frozenset({
    "owner_id", "telegram_user_id", "channel_target", "channel_profile_id",
    "status", "posted", "draft_source", "enabled", "is_active",
    # channel_profiles settings columns:
    "channel_style", "channel_audience", "channel_style_preset", "channel_mode",
    "channel_formats", "channel_frequency", "content_rubrics", "rubrics_schedule",
    "post_scenarios", "content_exclusions", "content_constraints", "channel_signature",
    "news_sources", "news_enabled", "news_interval_hours", "news_strict_mode",
    "posts_enabled", "source_auto_draft", "onboarding_completed",
    "author_role_type", "author_role_description", "author_activities",
    "author_forbidden_claims", "auto_image", "topic", "posting_mode",
    # draft fields:
    "text", "prompt", "media_type", "media_ref", "media_meta_json",
    "buttons_json", "pin_post", "comments_enabled", "ad_mark",
    "first_reaction", "reply_to_message_id", "send_silent", "news_source_json",
    "mirror_targets",
})


def _validate_sql_identifier(name: str, allowed: frozenset[str], kind: str = "identifier") -> str:
    """Validate that *name* is in the explicit whitelist.

    Raises ValueError for any value not in *allowed*, preventing SQL injection
    through dynamic table/column names.
    """
    if name not in allowed:
        raise ValueError(f"Disallowed SQL {kind}: {name!r}")
    return name


async def _column_exists(db: aiosqlite.Connection, table: str, column: str) -> bool:
    _validate_sql_identifier(table, _ALLOWED_TABLES, "table")
    cur = await db.execute(f"PRAGMA table_info({table})")
    rows = await cur.fetchall()
    return any(r[1] == column for r in rows)




def _safe_int(value, default: int = 0) -> int:
    try:
        if value is None:
            return default
        if isinstance(value, str):
            value = value.strip()
            if value == "":
                return default
        return int(value)
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Channel target validation — centralised format check
# ---------------------------------------------------------------------------
# @username: starts with letter, 4-32 alphanumeric/underscore chars after @
_CHANNEL_TARGET_USERNAME_RE = re.compile(r"^@[A-Za-z][A-Za-z0-9_]{3,31}$")
# Numeric Telegram chat/channel ID: optional minus sign, 6-15 digits
_CHANNEL_TARGET_NUMERIC_RE = re.compile(r"^-?1\d{12}$|^-?\d{6,15}$")


def validate_channel_target(raw: str) -> str:
    """Normalise and validate a channel_target value.

    Accepted formats:
    - ``@username`` (Telegram channel username, 4-32 chars after @)
    - Numeric Telegram chat ID (e.g. ``-1001234567890``, always ≥6 digits)

    Returns the normalised string, or raises ``ValueError`` for junk input.
    """
    value = str(raw or "").strip()
    if not value:
        raise ValueError("channel_target is empty")
    if value.startswith("@"):
        if _CHANNEL_TARGET_USERNAME_RE.match(value):
            return value
        raise ValueError(f"Invalid channel username format: {value!r}")
    # Numeric ID
    if _CHANNEL_TARGET_NUMERIC_RE.match(value):
        return value
    raise ValueError(f"Invalid channel_target format: {value!r}")


def _draft_from_row(row) -> dict:
    if not row:
        return None
    return {
        "id": row[0],
        "owner_id": _safe_int(row[1]),
        "telegram_user_id": _safe_int(row[1]),
        "channel_target": row[2],
        "text": row[3],
        "prompt": row[4],
        "topic": row[5],
        "media_type": row[6],
        "media_ref": row[7],
        "media_meta_json": row[8],
        "buttons_json": row[9],
        "pin_post": _safe_int(row[10]),
        "comments_enabled": _safe_int(row[11]),
        "ad_mark": _safe_int(row[12]),
        "first_reaction": row[13],
        "reply_to_message_id": _safe_int(row[14]),
        "status": row[15],
        "created_at": row[16],
        "updated_at": row[17],
        "send_silent": _safe_int(row[18]) if len(row) > 18 else 0,
        "draft_source": str(row[19]) if len(row) > 19 else "",
        "news_source_json": str(row[20]) if len(row) > 20 else "",
    }

async def _ensure_settings_schema(db: aiosqlite.Connection):
    await db.execute(
        """
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )


async def _ensure_schedules_schema(db: aiosqlite.Connection):
    await db.execute(
        """
        CREATE TABLE IF NOT EXISTS schedules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            time_hhmm TEXT NOT NULL,
            days TEXT NOT NULL DEFAULT '*'
        )
        """
    )
    if not await _column_exists(db, "schedules", "enabled"):
        await db.execute("ALTER TABLE schedules ADD COLUMN enabled INTEGER NOT NULL DEFAULT 1")
    if not await _column_exists(db, "schedules", "owner_id"):
        await db.execute("ALTER TABLE schedules ADD COLUMN owner_id INTEGER NOT NULL DEFAULT 0")
    # Per-channel schedules: link schedule to a specific channel profile
    if not await _column_exists(db, "schedules", "channel_profile_id"):
        await db.execute("ALTER TABLE schedules ADD COLUMN channel_profile_id INTEGER NOT NULL DEFAULT 0")
    await db.execute("CREATE INDEX IF NOT EXISTS idx_schedules_owner ON schedules(owner_id, id)")


async def _ensure_plan_schema(db: aiosqlite.Connection):
    await db.execute(
        """
        CREATE TABLE IF NOT EXISTS plan_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dt TEXT NOT NULL,
            kind TEXT NOT NULL,
            payload TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    if not await _column_exists(db, "plan_items", "enabled"):
        await db.execute("ALTER TABLE plan_items ADD COLUMN enabled INTEGER NOT NULL DEFAULT 1")
    if not await _column_exists(db, "plan_items", "posted"):
        await db.execute("ALTER TABLE plan_items ADD COLUMN posted INTEGER NOT NULL DEFAULT 0")
    if not await _column_exists(db, "plan_items", "owner_id"):
        await db.execute("ALTER TABLE plan_items ADD COLUMN owner_id INTEGER NOT NULL DEFAULT 0")
    # Migration: per-channel content plan
    if not await _column_exists(db, "plan_items", "channel_profile_id"):
        await db.execute("ALTER TABLE plan_items ADD COLUMN channel_profile_id INTEGER NOT NULL DEFAULT 0")
    await db.execute("CREATE INDEX IF NOT EXISTS idx_plan_items_owner ON plan_items(owner_id, id)")


async def _ensure_dm_memory_schema(db: aiosqlite.Connection):
    await db.execute(
        """
        CREATE TABLE IF NOT EXISTS dm_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            text TEXT NOT NULL,
            ts TEXT NOT NULL
        )
        """
    )
    await db.execute("CREATE INDEX IF NOT EXISTS idx_dm_memory_user ON dm_memory(user_id, id DESC)")


async def _ensure_post_log_schema(db: aiosqlite.Connection):
    await db.execute(
        """
        CREATE TABLE IF NOT EXISTS post_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            owner_id INTEGER NOT NULL DEFAULT 0,
            channel_target TEXT NOT NULL DEFAULT '',
            content_type TEXT NOT NULL DEFAULT 'text',
            text TEXT NOT NULL DEFAULT '',
            prompt TEXT NOT NULL DEFAULT '',
            topic TEXT NOT NULL DEFAULT '',
            file_id TEXT NOT NULL DEFAULT '',
            telegram_message_id INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        )
        """
    )
    if not await _column_exists(db, "post_logs", "post_views"):
        await db.execute("ALTER TABLE post_logs ADD COLUMN post_views INTEGER NOT NULL DEFAULT 0")
    if not await _column_exists(db, "post_logs", "reactions_count"):
        await db.execute("ALTER TABLE post_logs ADD COLUMN reactions_count INTEGER NOT NULL DEFAULT 0")
    if not await _column_exists(db, "post_logs", "forwards_count"):
        await db.execute("ALTER TABLE post_logs ADD COLUMN forwards_count INTEGER NOT NULL DEFAULT 0")
    if not await _column_exists(db, "post_logs", "comments_count"):
        await db.execute("ALTER TABLE post_logs ADD COLUMN comments_count INTEGER NOT NULL DEFAULT 0")
    if not await _column_exists(db, "post_logs", "engagement_score"):
        await db.execute("ALTER TABLE post_logs ADD COLUMN engagement_score REAL NOT NULL DEFAULT 0")
    if not await _column_exists(db, "post_logs", "topic_signature"):
        await db.execute("ALTER TABLE post_logs ADD COLUMN topic_signature TEXT NOT NULL DEFAULT ''")
    await db.execute("CREATE INDEX IF NOT EXISTS idx_post_logs_owner ON post_logs(owner_id, id DESC)")


async def _ensure_draft_schema(db: aiosqlite.Connection):
    await db.execute(
        """
        CREATE TABLE IF NOT EXISTS draft_posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            owner_id INTEGER NOT NULL DEFAULT 0,
            channel_target TEXT NOT NULL DEFAULT '',
            text TEXT NOT NULL DEFAULT '',
            prompt TEXT NOT NULL DEFAULT '',
            topic TEXT NOT NULL DEFAULT '',
            media_type TEXT NOT NULL DEFAULT 'none',
            media_ref TEXT NOT NULL DEFAULT '',
            media_meta_json TEXT NOT NULL DEFAULT '',
            buttons_json TEXT NOT NULL DEFAULT '[]',
            pin_post INTEGER NOT NULL DEFAULT 0,
            comments_enabled INTEGER NOT NULL DEFAULT 1,
            ad_mark INTEGER NOT NULL DEFAULT 0,
            first_reaction TEXT NOT NULL DEFAULT '',
            reply_to_message_id INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'draft',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    if not await _column_exists(db, "draft_posts", "media_meta_json"):
        await db.execute("ALTER TABLE draft_posts ADD COLUMN media_meta_json TEXT NOT NULL DEFAULT ''")
    if not await _column_exists(db, "draft_posts", "send_silent"):
        await db.execute("ALTER TABLE draft_posts ADD COLUMN send_silent INTEGER NOT NULL DEFAULT 0")
    if not await _column_exists(db, "draft_posts", "draft_source"):
        await db.execute("ALTER TABLE draft_posts ADD COLUMN draft_source TEXT NOT NULL DEFAULT ''")
    if not await _column_exists(db, "draft_posts", "news_source_json"):
        await db.execute("ALTER TABLE draft_posts ADD COLUMN news_source_json TEXT NOT NULL DEFAULT ''")
    # Mirror publishing: JSON array of extra channel_targets for linked publishing
    if not await _column_exists(db, "draft_posts", "mirror_targets"):
        await db.execute("ALTER TABLE draft_posts ADD COLUMN mirror_targets TEXT NOT NULL DEFAULT '[]'")
    await db.execute("CREATE INDEX IF NOT EXISTS idx_draft_posts_owner ON draft_posts(owner_id, id DESC)")







async def _ensure_generation_history_schema(db: aiosqlite.Connection):
    await db.execute(
        '''
        CREATE TABLE IF NOT EXISTS generation_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            owner_id INTEGER NOT NULL DEFAULT 0,
            telegram_user_id INTEGER NOT NULL DEFAULT 0,
            source TEXT NOT NULL DEFAULT 'generate-post',
            prompt TEXT NOT NULL DEFAULT '',
            topic TEXT NOT NULL DEFAULT '',
            title TEXT NOT NULL DEFAULT '',
            body TEXT NOT NULL DEFAULT '',
            cta TEXT NOT NULL DEFAULT '',
            short TEXT NOT NULL DEFAULT '',
            safety_status TEXT NOT NULL DEFAULT 'ok',
            draft_id INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        )
        '''
    )
    await db.execute("CREATE INDEX IF NOT EXISTS idx_generation_history_owner ON generation_history(owner_id, id DESC)")


async def _ensure_media_inbox_schema(db: aiosqlite.Connection):
    await db.execute(
        """
        CREATE TABLE IF NOT EXISTS user_media_inbox (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            owner_id INTEGER NOT NULL DEFAULT 0,
            telegram_user_id INTEGER NOT NULL DEFAULT 0,
            message_id INTEGER NOT NULL DEFAULT 0,
            media_group_id TEXT NOT NULL DEFAULT '',
            kind TEXT NOT NULL DEFAULT 'video',
            file_id TEXT NOT NULL DEFAULT '',
            file_unique_id TEXT NOT NULL DEFAULT '',
            file_name TEXT NOT NULL DEFAULT '',
            file_size INTEGER NOT NULL DEFAULT 0,
            mime_type TEXT NOT NULL DEFAULT '',
            duration INTEGER NOT NULL DEFAULT 0,
            width INTEGER NOT NULL DEFAULT 0,
            height INTEGER NOT NULL DEFAULT 0,
            caption TEXT NOT NULL DEFAULT '',
            source_chat_id INTEGER NOT NULL DEFAULT 0,
            source_message_date TEXT NOT NULL DEFAULT '',
            used_count INTEGER NOT NULL DEFAULT 0,
            last_used_at TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL
        )
        """
    )
    await db.execute("CREATE INDEX IF NOT EXISTS idx_media_inbox_owner ON user_media_inbox(owner_id, id DESC)")
    await db.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_media_inbox_owner_unique ON user_media_inbox(owner_id, file_unique_id)")


async def _ensure_user_binding_column(db: aiosqlite.Connection, table: str, source_column: str = "owner_id"):
    _validate_sql_identifier(table, _ALLOWED_TABLES, "table")
    _validate_sql_identifier(source_column, _ALLOWED_COLUMNS, "column")
    if not await _column_exists(db, table, "telegram_user_id"):
        await db.execute(f"ALTER TABLE {table} ADD COLUMN telegram_user_id INTEGER NOT NULL DEFAULT 0")
    try:
        await db.execute(
            f"UPDATE {table} SET telegram_user_id=COALESCE(NULLIF(telegram_user_id, 0), {source_column}, 0) "
            f"WHERE COALESCE(telegram_user_id, 0)=0"
        )
    except Exception:
        pass
    await db.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_telegram_user_id ON {table}(telegram_user_id)")

async def _ensure_channel_profiles_schema(db: aiosqlite.Connection):
    await db.execute(
        """
        CREATE TABLE IF NOT EXISTS channel_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            owner_id INTEGER NOT NULL DEFAULT 0,
            title TEXT NOT NULL DEFAULT '',
            channel_target TEXT NOT NULL DEFAULT '',
            topic TEXT NOT NULL DEFAULT '',
            is_active INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            topic_raw TEXT NOT NULL DEFAULT '',
            topic_family TEXT NOT NULL DEFAULT '',
            topic_subfamily TEXT NOT NULL DEFAULT '',
            audience_type TEXT NOT NULL DEFAULT '',
            style_mode TEXT NOT NULL DEFAULT '',
            content_goals TEXT NOT NULL DEFAULT '',
            preferred_formats TEXT NOT NULL DEFAULT '',
            forbidden_topics TEXT NOT NULL DEFAULT '',
            forbidden_claims TEXT NOT NULL DEFAULT '',
            visual_policy TEXT NOT NULL DEFAULT 'auto',
            forbidden_visual_classes TEXT NOT NULL DEFAULT '',
            rubric_map TEXT NOT NULL DEFAULT '',
            news_policy TEXT NOT NULL DEFAULT 'standard',
            posting_mode TEXT NOT NULL DEFAULT 'manual',
            sensitivity_flags TEXT NOT NULL DEFAULT ''
        )
        """
    )
    await db.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_channel_profiles_unique ON channel_profiles(owner_id, channel_target)")
    await db.execute("CREATE INDEX IF NOT EXISTS idx_channel_profiles_owner ON channel_profiles(owner_id, is_active, id)")
    # Add new structured profile columns to existing DBs (migrations)
    new_columns = [
        ("topic_raw", "TEXT NOT NULL DEFAULT ''"),
        ("topic_family", "TEXT NOT NULL DEFAULT ''"),
        ("topic_subfamily", "TEXT NOT NULL DEFAULT ''"),
        ("audience_type", "TEXT NOT NULL DEFAULT ''"),
        ("style_mode", "TEXT NOT NULL DEFAULT ''"),
        ("content_goals", "TEXT NOT NULL DEFAULT ''"),
        ("preferred_formats", "TEXT NOT NULL DEFAULT ''"),
        ("forbidden_topics", "TEXT NOT NULL DEFAULT ''"),
        ("forbidden_claims", "TEXT NOT NULL DEFAULT ''"),
        ("visual_policy", "TEXT NOT NULL DEFAULT 'auto'"),
        ("forbidden_visual_classes", "TEXT NOT NULL DEFAULT ''"),
        ("rubric_map", "TEXT NOT NULL DEFAULT ''"),
        ("news_policy", "TEXT NOT NULL DEFAULT 'standard'"),
        ("posting_mode", "TEXT NOT NULL DEFAULT 'manual'"),
        ("sensitivity_flags", "TEXT NOT NULL DEFAULT ''"),
        # Author role fields for anti-fabrication guardrails
        ("author_role_type", "TEXT NOT NULL DEFAULT ''"),
        ("author_role_description", "TEXT NOT NULL DEFAULT ''"),
        ("author_activities", "TEXT NOT NULL DEFAULT ''"),
        ("author_forbidden_claims", "TEXT NOT NULL DEFAULT ''"),
        # Per-channel content and autopost settings (migrated from owner-level settings table)
        ("channel_style", "TEXT NOT NULL DEFAULT ''"),
        ("channel_audience", "TEXT NOT NULL DEFAULT ''"),
        ("channel_style_preset", "TEXT NOT NULL DEFAULT ''"),
        ("channel_mode", "TEXT NOT NULL DEFAULT ''"),
        ("channel_formats", "TEXT NOT NULL DEFAULT '[]'"),
        ("channel_frequency", "TEXT NOT NULL DEFAULT ''"),
        ("content_rubrics", "TEXT NOT NULL DEFAULT ''"),
        ("rubrics_schedule", "TEXT NOT NULL DEFAULT ''"),
        ("post_scenarios", "TEXT NOT NULL DEFAULT ''"),
        ("content_exclusions", "TEXT NOT NULL DEFAULT ''"),
        ("content_constraints", "TEXT NOT NULL DEFAULT '[]'"),
        ("channel_signature", "TEXT NOT NULL DEFAULT ''"),
        ("news_sources", "TEXT NOT NULL DEFAULT ''"),
        ("news_enabled", "TEXT NOT NULL DEFAULT ''"),
        ("news_interval_hours", "TEXT NOT NULL DEFAULT ''"),
        ("news_strict_mode", "TEXT NOT NULL DEFAULT ''"),
        ("posts_enabled", "TEXT NOT NULL DEFAULT ''"),
        ("source_auto_draft", "TEXT NOT NULL DEFAULT ''"),
        ("onboarding_completed", "TEXT NOT NULL DEFAULT ''"),
        # Auto-image toggle: "1" = auto-attach images to posts, "0" = text-only
        ("auto_image", "TEXT NOT NULL DEFAULT '1'"),
    ]
    for col_name, col_def in new_columns:
        # Safe: col_name and col_def are hardcoded string literals, not user input.
        # SQLite does not support parameterized ALTER TABLE ADD COLUMN, so f-string
        # is necessary here. Never pass user data to this function.
        try:
            await db.execute(f"ALTER TABLE channel_profiles ADD COLUMN {col_name} {col_def}")
        except Exception:
            pass  # Column already exists (expected on re-migration)


async def _ensure_news_schema(db: aiosqlite.Connection):
    await db.execute(
        """
        CREATE TABLE IF NOT EXISTS news_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            owner_id INTEGER NOT NULL DEFAULT 0,
            source_url TEXT NOT NULL DEFAULT '',
            source_title TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL
        )
        """
    )
    await db.execute("CREATE INDEX IF NOT EXISTS idx_news_logs_owner ON news_logs(owner_id, id DESC)")
    # Migration: add channel_target for per-channel news dedup
    if not await _column_exists(db, "news_logs", "channel_target"):
        await db.execute("ALTER TABLE news_logs ADD COLUMN channel_target TEXT NOT NULL DEFAULT ''")
    await db.execute("CREATE INDEX IF NOT EXISTS idx_news_logs_channel ON news_logs(owner_id, channel_target, source_url)")


async def _ensure_subscriptions_schema(db: aiosqlite.Connection):
    await db.execute(
        """
        CREATE TABLE IF NOT EXISTS user_subscriptions (
            owner_id INTEGER PRIMARY KEY,
            subscription_tier TEXT NOT NULL DEFAULT 'free',
            generations_used INTEGER NOT NULL DEFAULT 0,
            generations_reset_at TEXT NOT NULL DEFAULT '',
            trial_ends_at TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL DEFAULT '',
            subscription_expires_at TEXT NOT NULL DEFAULT '',
            auto_renew INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    if not await _column_exists(db, "user_subscriptions", "subscription_expires_at"):
        await db.execute(
            "ALTER TABLE user_subscriptions ADD COLUMN subscription_expires_at TEXT NOT NULL DEFAULT ''"
        )
    if not await _column_exists(db, "user_subscriptions", "auto_renew"):
        await db.execute(
            "ALTER TABLE user_subscriptions ADD COLUMN auto_renew INTEGER NOT NULL DEFAULT 0"
        )


async def _ensure_feature_quotas_schema(db: aiosqlite.Connection):
    await db.execute(
        """
        CREATE TABLE IF NOT EXISTS user_feature_quotas (
            owner_id INTEGER NOT NULL,
            feature TEXT NOT NULL,
            used INTEGER NOT NULL DEFAULT 0,
            reset_at TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (owner_id, feature)
        )
        """
    )


async def _ensure_payment_events_schema(db: aiosqlite.Connection):
    """Payment events table — stores every successful payment for audit + idempotency."""
    await db.execute(
        """
        CREATE TABLE IF NOT EXISTS payment_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            payment_id TEXT NOT NULL DEFAULT '',
            owner_id INTEGER NOT NULL,
            tier TEXT NOT NULL DEFAULT '',
            method TEXT NOT NULL DEFAULT '',
            amount TEXT NOT NULL DEFAULT '',
            currency TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'success',
            payload TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL DEFAULT ''
        )
        """
    )
    await db.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_payment_events_payment_id "
        "ON payment_events(payment_id) WHERE payment_id != ''"
    )
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_payment_events_owner "
        "ON payment_events(owner_id, id DESC)"
    )


async def _ensure_scheduler_dedup_schema(db: aiosqlite.Connection):
    """Durable dedup table for scheduler — survives process restarts.

    Each row represents a scheduler event that was already processed.
    The ``dedup_key`` is unique and prevents the same schedule slot,
    plan item, or news item from being processed twice in the same
    time window.

    Rows older than 48 hours are garbage-collected on each startup so
    the table stays small.
    """
    await db.execute(
        """
        CREATE TABLE IF NOT EXISTS scheduler_dedup (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dedup_key TEXT NOT NULL UNIQUE,
            trigger_type TEXT NOT NULL DEFAULT '',
            owner_id INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT ''
        )
        """
    )
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_scheduler_dedup_key "
        "ON scheduler_dedup(dedup_key)"
    )
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_scheduler_dedup_created "
        "ON scheduler_dedup(created_at)"
    )
    # Garbage-collect entries older than 48 hours on schema init
    cutoff = (datetime.utcnow() - timedelta(hours=48)).isoformat(timespec="seconds")
    await db.execute("DELETE FROM scheduler_dedup WHERE created_at < ? AND created_at != ''", (cutoff,))


# Keys that should live per-channel in channel_profiles rather than in the
# global settings table.  Used by backfill migration and by get_channel_settings().
_CHANNEL_SETTINGS_KEYS = (
    "channel_style", "channel_audience", "channel_style_preset", "channel_mode",
    "channel_formats", "channel_frequency", "content_rubrics", "rubrics_schedule",
    "post_scenarios", "content_exclusions", "content_constraints", "channel_signature",
    "news_sources", "news_enabled", "news_interval_hours", "news_strict_mode",
    "posts_enabled", "source_auto_draft", "onboarding_completed",
    "author_role_type", "author_role_description", "author_activities", "author_forbidden_claims",
    "auto_image",
)


async def _backfill_channel_settings(db: aiosqlite.Connection):
    """One-time migration: copy owner-level settings into active channel profiles.

    Only fills fields that are currently empty ('') in the channel profile,
    so manually set per-channel values are never overwritten.
    """
    cur = await db.execute(
        "SELECT id, owner_id FROM channel_profiles WHERE is_active=1"
    )
    active_profiles = await cur.fetchall()
    for profile_id, owner_id in active_profiles:
        if not owner_id:
            continue
        # Check if this profile has already been backfilled (channel_style is a good proxy)
        chk = await db.execute(
            "SELECT channel_style, onboarding_completed FROM channel_profiles WHERE id=?",
            (profile_id,)
        )
        chk_row = await chk.fetchone()
        if chk_row and (chk_row[0] or chk_row[1]):
            continue  # Already has data, skip

        # Read owner-scoped settings
        for key in _CHANNEL_SETTINGS_KEYS:
            assert key.isidentifier(), f"Invalid column name: {key}"  # safety: prevent SQL injection
            scoped = _scope_key(key, owner_id)
            val_cur = await db.execute("SELECT value FROM settings WHERE key=?", (scoped,))
            val_row = await val_cur.fetchone()
            if not val_row:
                val_cur = await db.execute("SELECT value FROM settings WHERE key=?", (key,))
                val_row = await val_cur.fetchone()
            if val_row and val_row[0]:
                try:
                    await db.execute(
                        f"UPDATE channel_profiles SET {key}=? WHERE id=? AND ({key}='' OR {key} IS NULL)",
                        (val_row[0], profile_id),
                    )
                except Exception:
                    pass  # Column might not exist yet in edge cases


async def _backfill_schedule_channel_profile(db: aiosqlite.Connection):
    """Assign existing schedules (channel_profile_id=0) to the owner's active channel."""
    cur = await db.execute(
        "SELECT DISTINCT owner_id FROM schedules WHERE channel_profile_id=0 AND owner_id>0"
    )
    owners = await cur.fetchall()
    for (owner_id,) in owners:
        prof_cur = await db.execute(
            "SELECT id FROM channel_profiles WHERE owner_id=? AND is_active=1 ORDER BY id DESC LIMIT 1",
            (owner_id,),
        )
        prof = await prof_cur.fetchone()
        if prof:
            await db.execute(
                "UPDATE schedules SET channel_profile_id=? WHERE owner_id=? AND channel_profile_id=0",
                (prof[0], owner_id),
            )


async def _backfill_plan_items_channel_profile(db: aiosqlite.Connection):
    """Assign existing plan_items (channel_profile_id=0) to the owner's active channel."""
    cur = await db.execute(
        "SELECT DISTINCT owner_id FROM plan_items WHERE channel_profile_id=0 AND owner_id>0"
    )
    owners = await cur.fetchall()
    for (owner_id,) in owners:
        prof_cur = await db.execute(
            "SELECT id FROM channel_profiles WHERE owner_id=? AND is_active=1 ORDER BY id DESC LIMIT 1",
            (owner_id,),
        )
        prof = await prof_cur.fetchone()
        if prof:
            await db.execute(
                "UPDATE plan_items SET channel_profile_id=? WHERE owner_id=? AND channel_profile_id=0",
                (prof[0], owner_id),
            )


async def init_db():
    # Schema migrations run before the pool is created so they always use
    # a dedicated temporary connection (safe: no pooled connection is dirty).
    db = await _connect()
    try:
        await _ensure_settings_schema(db)
        await _ensure_schedules_schema(db)
        await _ensure_plan_schema(db)
        await _ensure_dm_memory_schema(db)
        await _ensure_post_log_schema(db)
        await _ensure_draft_schema(db)
        await _ensure_generation_history_schema(db)
        await _ensure_media_inbox_schema(db)
        await _ensure_news_schema(db)
        await _ensure_channel_profiles_schema(db)
        await _ensure_subscriptions_schema(db)
        await _ensure_feature_quotas_schema(db)
        await _ensure_payment_events_schema(db)
        await _ensure_scheduler_dedup_schema(db)

        for table in ("schedules", "plan_items", "post_logs", "draft_posts", "generation_history", "user_media_inbox", "news_logs", "channel_profiles"):
            await _ensure_user_binding_column(db, table)

        defaults = {
            "posts_enabled": "0",
            "posting_mode": "manual",
            "news_enabled": "0",
            "news_interval_hours": "6",
            "news_sources": "who.int,mayoclinic.org,nih.gov",
        }
        for k, v in defaults.items():
            cur = await db.execute("SELECT value FROM settings WHERE key=?", (k,))
            row = await cur.fetchone()
            if not row:
                await db.execute("INSERT OR REPLACE INTO settings(key,value) VALUES(?,?)", (k, v))

        # --- Backfill: migrate owner-level settings → active channel profile ---
        # This runs once per profile that hasn't been backfilled yet.
        # It copies channel-specific settings from the owner-scoped settings table
        # into the channel_profiles row so they become truly per-channel.
        await _backfill_channel_settings(db)

        # --- Backfill: assign existing schedules to active channel profile ---
        await _backfill_schedule_channel_profile(db)

        # --- Backfill: assign existing plan_items to active channel profile ---
        await _backfill_plan_items_channel_profile(db)

        # --- Performance indexes (idempotent) ---
        _perf_indexes = [
            "CREATE INDEX IF NOT EXISTS idx_draft_posts_owner_status ON draft_posts(owner_id, status)",
            "CREATE INDEX IF NOT EXISTS idx_draft_posts_channel ON draft_posts(channel_target)",
            "CREATE INDEX IF NOT EXISTS idx_draft_posts_source ON draft_posts(draft_source)",
            "CREATE INDEX IF NOT EXISTS idx_plan_items_owner_posted ON plan_items(owner_id, posted)",
            "CREATE INDEX IF NOT EXISTS idx_plan_items_channel ON plan_items(channel_profile_id)",
            "CREATE INDEX IF NOT EXISTS idx_post_logs_owner ON post_logs(owner_id, id DESC)",
            "CREATE INDEX IF NOT EXISTS idx_post_logs_channel ON post_logs(channel_target)",
            "CREATE INDEX IF NOT EXISTS idx_schedules_owner ON schedules(owner_id, enabled)",
            "CREATE INDEX IF NOT EXISTS idx_schedules_channel ON schedules(channel_profile_id)",
            "CREATE INDEX IF NOT EXISTS idx_news_logs_owner ON news_logs(owner_id, channel_target)",
            "CREATE INDEX IF NOT EXISTS idx_news_logs_dedup ON news_logs(source_url)",
            "CREATE INDEX IF NOT EXISTS idx_channel_profiles_owner ON channel_profiles(owner_id, is_active)",
        ]
        for stmt in _perf_indexes:
            try:
                await db.execute(stmt)
            except Exception:
                pass  # Index might already exist differently

        # --- Schema version tracking (idempotent) ---
        await db.execute("""
            CREATE TABLE IF NOT EXISTS schema_versions (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL DEFAULT (datetime('now')),
                description TEXT NOT NULL DEFAULT ''
            )
        """)
        await db.execute(
            "INSERT OR IGNORE INTO schema_versions(version, description) VALUES(?, ?)",
            (1, "initial version tracking + performance indexes"),
        )

        await db.commit()
    finally:
        await db.close()

    # Pool is initialised after schema setup so every pooled connection is
    # guaranteed to find the schema already in place.
    await _init_pool()


# ---------- settings ----------
async def get_setting(key: str, owner_id: int | None = None) -> str | None:
    scoped = _scope_key(key, owner_id)
    async with _db_ctx() as db:
        cur = await db.execute("SELECT value FROM settings WHERE key=?", (scoped,))
        row = await cur.fetchone()
        if row:
            return row[0]
        if owner_id not in (None, 0):
            cur = await db.execute("SELECT value FROM settings WHERE key=?", (key,))
            row = await cur.fetchone()
            return row[0] if row else None
        return None


async def get_settings_bulk(keys: list[str], owner_id: int | None = None) -> dict[str, str | None]:
    unique_keys = [str(k) for k in dict.fromkeys(keys) if str(k).strip()]
    if not unique_keys:
        return {}
    scoped_keys = [_scope_key(k, owner_id) for k in unique_keys]
    placeholders = ",".join("?" for _ in scoped_keys)
    out: dict[str, str | None] = {k: None for k in unique_keys}
    db = await _connect()
    try:
        cur = await db.execute(f"SELECT key, value FROM settings WHERE key IN ({placeholders})", tuple(scoped_keys))
        rows = await cur.fetchall()
        scoped_map = {str(r[0]): r[1] for r in rows}
        for original, scoped in zip(unique_keys, scoped_keys):
            if scoped in scoped_map:
                out[original] = scoped_map[scoped]
        if owner_id not in (None, 0):
            missing = [k for k, v in out.items() if v is None]
            if missing:
                placeholders2 = ",".join("?" for _ in missing)
                cur = await db.execute(f"SELECT key, value FROM settings WHERE key IN ({placeholders2})", tuple(missing))
                rows = await cur.fetchall()
                fallback = {str(r[0]): r[1] for r in rows}
                for k in missing:
                    if k in fallback:
                        out[k] = fallback[k]
    finally:
        await db.close()
    return out


async def set_setting(key: str, value: str, owner_id: int | None = None):
    scoped = _scope_key(key, owner_id)
    db = await _connect()
    try:
        await db.execute("INSERT OR REPLACE INTO settings(key,value) VALUES(?,?)", (scoped, value))
        await db.commit()
    finally:
        await db.close()


async def get_posts_enabled(owner_id: int | None = None) -> bool:
    v = await get_setting("posts_enabled", owner_id=owner_id)
    return (v or "0").strip() not in ("0", "false", "False", "no", "No")


async def set_posts_enabled(enabled: bool, owner_id: int | None = None):
    await set_setting("posts_enabled", "1" if enabled else "0", owner_id=owner_id)


async def get_channel_settings(owner_id: int, channel_profile_id: int | None = None) -> dict[str, str]:
    """Read channel-specific settings from the channel_profiles table.

    Returns a dict of setting keys → values sourced **exclusively** from the
    channel profile.  No owner-level ``settings`` fallback is performed for
    channel-scoped keys so that one channel's configuration can never leak
    into another.

    Resolution order:
    1. If *channel_profile_id* is given, use that profile.
    2. Otherwise fall back to the active profile for *owner_id*.
    3. If no profile is found at all, return empty strings for every key.
    """
    profile: dict | None = None
    if channel_profile_id:
        async with _db_ctx() as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                "SELECT * FROM channel_profiles WHERE id=? AND owner_id=?",
                (int(channel_profile_id), int(owner_id)),
            )
            row = await cur.fetchone()
            if row:
                profile = dict(row)
    if not profile:
        profile = await get_active_channel_profile(owner_id=owner_id)

    # Build result exclusively from the channel profile — no owner-level
    # fallback for channel-scoped keys.
    result: dict[str, str] = {}

    for key in _CHANNEL_SETTINGS_KEYS:
        profile_val = str(profile.get(key, "")).strip() if profile else ""
        result[key] = profile_val

    # topic, channel_target and posting_mode live directly on the profile row
    if profile:
        result["topic"] = str(profile.get("topic") or "")
        result["channel_target"] = str(profile.get("channel_target") or "")
        result["posting_mode"] = str(profile.get("posting_mode") or "manual")
    else:
        result["topic"] = ""
        result["channel_target"] = ""
        result["posting_mode"] = "manual"

    return result


async def save_channel_setting(owner_id: int, key: str, value: str, channel_profile_id: int | None = None):
    """Save a setting.

    * Channel-scoped keys (topic/style/audience/…) are written **only** to the
      ``channel_profiles`` row — never to the flat ``settings`` table.  This
      prevents cross-channel contamination that previously occurred because
      ``get_channel_settings`` fell back to the owner-level ``settings`` row
      when a profile column was empty.
    * Non-channel keys (e.g. ``auto_hashtags``) are stored in the ``settings``
      table as before.
    """
    is_channel_key = key in _CHANNEL_SETTINGS_KEYS or key in ("topic", "posting_mode")

    if not is_channel_key:
        # Pure owner-level setting — write to flat settings table only
        await set_setting(key, value, owner_id=owner_id)
        return

    # --- Channel-scoped key: write to channel_profiles only ---
    pid = int(channel_profile_id) if channel_profile_id else 0
    if not pid:
        profile = await get_active_channel_profile(owner_id=owner_id)
        pid = int(profile.get("id", 0)) if profile else 0
    if not pid:
        # No channel profile exists yet — fall back to owner-level settings
        # so the value is not lost entirely.  Once a profile is created the
        # backfill migration will copy it over.
        await set_setting(key, value, owner_id=owner_id)
        return

    allowed_cols = set(_CHANNEL_SETTINGS_KEYS) | {"topic", "posting_mode"}
    if key not in allowed_cols:
        return
    assert key.isidentifier(), f"Invalid column name: {key}"
    now = datetime.utcnow().isoformat(timespec="seconds")
    async with _db_ctx() as db:
        await db.execute(
            f"UPDATE channel_profiles SET {key}=?, updated_at=? WHERE id=? AND owner_id=?",
            (value, now, pid, int(owner_id)),
        )
        await db.commit()


# ---------- schedules ----------
async def list_schedules(owner_id: int | None = 0, channel_profile_id: int | None = None) -> list[dict]:
    async with _db_ctx() as db:
        if owner_id is None:
            cur = await db.execute(
                "SELECT id, time_hhmm, days, enabled, owner_id, channel_profile_id FROM schedules ORDER BY owner_id ASC, id ASC"
            )
            rows = await cur.fetchall()
        elif channel_profile_id:
            cur = await db.execute(
                "SELECT id, time_hhmm, days, enabled, owner_id, channel_profile_id FROM schedules WHERE owner_id=? AND channel_profile_id=? ORDER BY id ASC",
                (int(owner_id), int(channel_profile_id)),
            )
            rows = await cur.fetchall()
        else:
            cur = await db.execute(
                "SELECT id, time_hhmm, days, enabled, owner_id, channel_profile_id FROM schedules WHERE owner_id=? ORDER BY id ASC",
                (int(owner_id),),
            )
            rows = await cur.fetchall()
        return [
            {
                "id": r[0],
                "time_hhmm": r[1],
                "time": r[1],
                "days": r[2],
                "enabled": int(r[3]),
                "owner_id": int(r[4]),
                "channel_profile_id": int(r[5]) if len(r) > 5 else 0,
            }
            for r in rows
        ]


async def list_schedule(owner_id: int | None = 0):
    return await list_schedules(owner_id=owner_id)


async def add_schedule(time_hhmm: str, days: str = "*", owner_id: int | None = 0, channel_profile_id: int | None = 0):
    async with _db_ctx() as db:
        owner = int(owner_id or 0)
        cpid = int(channel_profile_id or 0)
        await db.execute(
            "INSERT INTO schedules(time_hhmm, days, enabled, owner_id, telegram_user_id, channel_profile_id) VALUES(?,?,1,?,?,?)",
            (time_hhmm.strip(), (days or "*").strip(), owner, owner, cpid),
        )
        await db.commit()


async def clear_schedules(owner_id: int | None = 0):
    async with _db_ctx() as db:
        if owner_id is None:
            await db.execute("DELETE FROM schedules")
        else:
            await db.execute("DELETE FROM schedules WHERE owner_id=?", (int(owner_id),))
        await db.commit()


# ---------- plan ----------
async def add_plan_item(
    dt: str,
    kind: str | None = None,
    payload: str | None = None,
    enabled: bool = True,
    *,
    topic: str = "",
    prompt: str = "",
    posted: int = 0,
    owner_id: int | None = 0,
    channel_profile_id: int | None = 0,
):
    dt = (dt or "").strip()
    if kind and payload is not None and not topic and not prompt:
        _kind = kind.strip()
        _payload = payload.strip()
    else:
        topic = (topic or "").strip()
        prompt = (prompt or "").strip()
        if prompt:
            _kind = "prompt"
            _payload = prompt
        else:
            _kind = "topic"
            _payload = topic
    async with _db_ctx() as db:
        await db.execute(
            """
            INSERT INTO plan_items(dt, kind, payload, created_at, enabled, posted, owner_id, channel_profile_id)
            VALUES(?,?,?,?,?,?,?,?)
            """,
            (
                dt, _kind, _payload, datetime.utcnow().isoformat(timespec="seconds"),
                1 if enabled else 0, int(posted), int(owner_id or 0), int(channel_profile_id or 0),
            ),
        )
        await db.commit()


async def list_plan_items(limit: int = 50, owner_id: int | None = 0, *, channel_profile_id: int | None = None) -> list[dict]:
    async with _db_ctx() as db:
        if channel_profile_id is not None and channel_profile_id > 0:
            cur = await db.execute(
                "SELECT id, dt, kind, payload, created_at, enabled, posted, owner_id, channel_profile_id FROM plan_items WHERE owner_id=? AND channel_profile_id=? ORDER BY dt ASC, id ASC LIMIT ?",
                (int(owner_id or 0), int(channel_profile_id), int(limit)),
            )
        elif owner_id is None:
            cur = await db.execute(
                "SELECT id, dt, kind, payload, created_at, enabled, posted, owner_id, channel_profile_id FROM plan_items ORDER BY dt ASC, id ASC LIMIT ?",
                (int(limit),),
            )
        else:
            cur = await db.execute(
                "SELECT id, dt, kind, payload, created_at, enabled, posted, owner_id, channel_profile_id FROM plan_items WHERE owner_id=? ORDER BY dt ASC, id ASC LIMIT ?",
                (int(owner_id), int(limit)),
            )
        rows = await cur.fetchall()
        out = []
        for r in rows:
            topic = r[3] if r[2] == "topic" else ""
            prompt = r[3] if r[2] == "prompt" else ""
            out.append(
                {
                    "id": r[0],
                    "dt": r[1],
                    "kind": r[2],
                    "payload": r[3],
                    "created_at": r[4],
                    "enabled": int(r[5]),
                    "posted": int(r[6]),
                    "owner_id": int(r[7]),
                    "channel_profile_id": int(r[8]) if len(r) > 8 and r[8] else 0,
                    "topic": topic,
                    "prompt": prompt,
                }
            )
        return out


async def list_plan_items_active_not_posted(owner_id: int | None = 0, *, channel_profile_id: int | None = None) -> list[dict]:
    async with _db_ctx() as db:
        if channel_profile_id is not None and channel_profile_id > 0:
            cur = await db.execute(
                "SELECT id, dt, kind, payload, created_at, enabled, posted, owner_id, channel_profile_id FROM plan_items WHERE enabled=1 AND posted=0 AND owner_id=? AND channel_profile_id=? ORDER BY dt ASC, id ASC",
                (int(owner_id or 0), int(channel_profile_id)),
            )
        elif owner_id is None:
            cur = await db.execute(
                "SELECT id, dt, kind, payload, created_at, enabled, posted, owner_id, channel_profile_id FROM plan_items WHERE enabled=1 AND posted=0 ORDER BY dt ASC, id ASC"
            )
        else:
            cur = await db.execute(
                "SELECT id, dt, kind, payload, created_at, enabled, posted, owner_id, channel_profile_id FROM plan_items WHERE enabled=1 AND posted=0 AND owner_id=? ORDER BY dt ASC, id ASC",
                (int(owner_id),),
            )
        rows = await cur.fetchall()
        out = []
        for r in rows:
            topic = r[3] if r[2] == "topic" else ""
            prompt = r[3] if r[2] == "prompt" else ""
            out.append(
                {
                    "id": r[0],
                    "dt": r[1],
                    "kind": r[2],
                    "payload": r[3],
                    "created_at": r[4],
                    "enabled": int(r[5]),
                    "posted": int(r[6]),
                    "owner_id": int(r[7]),
                    "channel_profile_id": int(r[8]) if len(r) > 8 and r[8] else 0,
                    "topic": topic,
                    "prompt": prompt,
                }
            )
        return out


async def mark_plan_posted(item_id: int, owner_id: int | None = 0):
    async with _db_ctx() as db:
        await db.execute("UPDATE plan_items SET posted=1 WHERE id=? AND owner_id=?", (int(item_id), int(owner_id or 0)))
        await db.commit()


async def delete_plan_item(item_id: int, owner_id: int | None = 0):
    async with _db_ctx() as db:
        await db.execute("DELETE FROM plan_items WHERE id=? AND owner_id=?", (int(item_id), int(owner_id or 0)))
        await db.commit()


async def get_plan_item(item_id: int, owner_id: int | None = 0) -> dict | None:
    async with _db_ctx() as db:
        cur = await db.execute(
            "SELECT id, dt, kind, payload, created_at, enabled, posted, owner_id FROM plan_items WHERE id=? AND owner_id=? LIMIT 1",
            (int(item_id), int(owner_id or 0)),
        )
        r = await cur.fetchone()
        if not r:
            return None
        topic = r[3] if r[2] == "topic" else ""
        prompt = r[3] if r[2] == "prompt" else ""
        return {
            "id": r[0],
            "dt": r[1],
            "kind": r[2],
            "payload": r[3],
            "created_at": r[4],
            "enabled": int(r[5]),
            "posted": int(r[6]),
            "owner_id": int(r[7]),
            "topic": topic,
            "prompt": prompt,
        }


async def update_plan_item(item_id: int, owner_id: int | None = 0, *, dt: str | None = None, topic: str | None = None, prompt: str | None = None):
    fields = []
    values = []
    if dt is not None:
        fields.append("dt=?")
        values.append((dt or "").strip())
    if topic is not None:
        fields.append("kind=?")
        fields.append("payload=?")
        values.extend(["topic", (topic or "").strip()])
    elif prompt is not None:
        fields.append("kind=?")
        fields.append("payload=?")
        values.extend(["prompt", (prompt or "").strip()])
    if not fields:
        return
    values.extend([int(item_id), int(owner_id or 0)])
    async with _db_ctx() as db:
        await db.execute(f"UPDATE plan_items SET {', '.join(fields)} WHERE id=? AND owner_id=?", tuple(values))
        await db.commit()


async def clear_unposted_plan_items(owner_id: int | None = 0):
    async with _db_ctx() as db:
        if owner_id is None:
            await db.execute("DELETE FROM plan_items WHERE posted=0")
        else:
            await db.execute("DELETE FROM plan_items WHERE owner_id=? AND posted=0", (int(owner_id),))
        await db.commit()


# ---------- dm memory ----------
async def dm_add_message(user_id: int, role: str, text: str):
    async with _db_ctx() as db:
        await db.execute(
            "INSERT INTO dm_memory(user_id, role, text, ts) VALUES(?,?,?,?)",
            (int(user_id), role, text, datetime.utcnow().isoformat(timespec="seconds")),
        )
        await db.commit()


async def dm_get_recent(user_id: int, limit: int = 14) -> list[dict]:
    async with _db_ctx() as db:
        cur = await db.execute(
            "SELECT role, text, ts FROM dm_memory WHERE user_id=? ORDER BY id DESC LIMIT ?",
            (int(user_id), int(limit)),
        )
        rows = await cur.fetchall()
        return [{"role": r[0], "text": r[1], "ts": r[2]} for r in reversed(rows)]


# ---------- post logs ----------
async def log_post(
    *,
    owner_id: int | None = 0,
    channel_target: str,
    content_type: str = "text",
    text: str = "",
    prompt: str = "",
    topic: str = "",
    file_id: str = "",
    telegram_message_id: int = 0,
    created_at: Optional[str] = None,
    post_views: int = 0,
    reactions_count: int = 0,
    forwards_count: int = 0,
    comments_count: int = 0,
    engagement_score: float = 0.0,
):
    created_at = created_at or datetime.utcnow().isoformat(timespec="seconds")
    async with _db_ctx() as db:
        await db.execute(
            """
            INSERT INTO post_logs(
                owner_id, telegram_user_id, channel_target, content_type, text, prompt, topic, file_id, telegram_message_id, created_at,
                post_views, reactions_count, forwards_count, comments_count, engagement_score, topic_signature
            )
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                int(owner_id or 0),
                int(owner_id or 0),
                channel_target,
                content_type,
                text,
                prompt,
                topic,
                file_id or "",
                int(telegram_message_id or 0),
                created_at,
                int(post_views or 0),
                int(reactions_count or 0),
                int(forwards_count or 0),
                int(comments_count or 0),
                float(engagement_score or 0.0),
                _topic_signature(topic or prompt or text),
            ),
        )
        await db.commit()


async def get_last_post_time(owner_id: int, *, channel_target: str = "") -> str | None:
    """Return ISO timestamp of the most recent published post.

    If *channel_target* is provided, scopes to that channel (per-channel cooldown).
    Otherwise falls back to owner-level (any channel).
    """
    async with _db_ctx() as db:
        if channel_target:
            cur = await db.execute(
                "SELECT created_at FROM post_logs WHERE owner_id=? AND channel_target=? ORDER BY id DESC LIMIT 1",
                (int(owner_id), channel_target),
            )
        else:
            cur = await db.execute(
                "SELECT created_at FROM post_logs WHERE owner_id=? ORDER BY id DESC LIMIT 1",
                (int(owner_id),),
            )
        row = await cur.fetchone()
        return str(row[0]).strip() if row and row[0] else None


async def list_recent_posts(owner_id: int | None = 0, limit: int = 10) -> list[dict]:
    async with _db_ctx() as db:
        if owner_id is None:
            cur = await db.execute(
                "SELECT id, owner_id, channel_target, content_type, text, prompt, topic, file_id, telegram_message_id, created_at FROM post_logs ORDER BY id DESC LIMIT ?",
                (int(limit),),
            )
        else:
            cur = await db.execute(
                "SELECT id, owner_id, channel_target, content_type, text, prompt, topic, file_id, telegram_message_id, created_at FROM post_logs WHERE owner_id=? ORDER BY id DESC LIMIT ?",
                (int(owner_id), int(limit)),
            )
        rows = await cur.fetchall()
        return [
            {
                "id": r[0],
                "owner_id": int(r[1]),
                "channel_target": r[2],
                "content_type": r[3],
                "text": r[4],
                "prompt": r[5],
                "topic": r[6],
                "file_id": r[7],
                "telegram_message_id": int(r[8]),
                "created_at": r[9],
            }
            for r in rows
        ]


async def get_post_analytics_snapshot(owner_id: int | None = 0, limit: int = 120) -> dict:
    owner = int(owner_id or 0)
    async with _db_ctx() as db:
        cur = await db.execute(
            """
            SELECT id, channel_target, text, prompt, topic, post_views, reactions_count, forwards_count, comments_count, engagement_score, topic_signature, created_at
            FROM post_logs
            WHERE owner_id=?
            ORDER BY id DESC
            LIMIT ?
            """,
            (owner, int(limit)),
        )
        rows = await cur.fetchall()

    posts = []
    topic_buckets: dict[str, dict] = {}
    views_known = False
    total_views = total_reactions = total_forwards = total_comments = 0
    for r in rows:
        views = int(r[5] or 0)
        reactions = int(r[6] or 0)
        forwards = int(r[7] or 0)
        comments = int(r[8] or 0)
        engagement = float(r[9] or 0.0)
        signature = str(r[10] or '').strip() or _topic_signature(r[4] or r[3] or r[2]) or 'прочее'
        label = str(r[4] or r[3] or r[2] or '').strip()[:80] or signature
        created_at = str(r[11] or '')
        if views > 0:
            views_known = True
        total_views += views
        total_reactions += reactions
        total_forwards += forwards
        total_comments += comments
        bucket = topic_buckets.setdefault(signature, {
            'topic_signature': signature, 'label': label, 'posts': 0, 'views_total': 0, 'reactions_total': 0, 'forwards_total': 0, 'comments_total': 0, 'engagement_total': 0.0, 'last_post_at': created_at
        })
        bucket['posts'] += 1
        bucket['views_total'] += views
        bucket['reactions_total'] += reactions
        bucket['forwards_total'] += forwards
        bucket['comments_total'] += comments
        bucket['engagement_total'] += engagement
        if created_at > str(bucket.get('last_post_at') or ''):
            bucket['last_post_at'] = created_at
        posts.append({
            'id': int(r[0]), 'channel_target': r[1], 'text': r[2], 'prompt': r[3], 'topic': r[4], 'views': views, 'reactions': reactions, 'forwards': forwards, 'comments': comments, 'engagement_score': engagement, 'topic_signature': signature, 'created_at': created_at
        })

    top_topics = list(topic_buckets.values())
    for item in top_topics:
        posts_count = max(int(item['posts']), 1)
        item['avg_views'] = round(item['views_total'] / posts_count, 1)
        item['avg_engagement'] = round(item['engagement_total'] / posts_count, 2)
    top_topics.sort(key=lambda x: (x['avg_views'], x['avg_engagement'], x['posts']), reverse=True)
    top_topics = top_topics[:6]

    return {
        'posts': posts,
        'top_topics': top_topics,
        'views_known': views_known,
        'avg_views': round(total_views / max(len(posts), 1), 1) if views_known and posts else 0,
        'avg_reactions': round(total_reactions / max(len(posts), 1), 1) if posts else 0,
        'avg_forwards': round(total_forwards / max(len(posts), 1), 1) if posts else 0,
        'avg_comments': round(total_comments / max(len(posts), 1), 1) if posts else 0,
        'total_posts_considered': len(posts),
    }


async def get_recent_channel_history(owner_id: int | None = 0, limit: int = 12) -> dict:
    owner = int(owner_id or 0)
    async with _db_ctx() as db:
        cur = await db.execute(
            """
            SELECT text, prompt, topic, created_at
            FROM draft_posts
            WHERE owner_id=?
            ORDER BY updated_at DESC, id DESC
            LIMIT ?
            """,
            (owner, int(limit)),
        )
        draft_rows = await cur.fetchall()

        cur = await db.execute(
            """
            SELECT text, content_type, created_at
            FROM post_logs
            WHERE owner_id=?
            ORDER BY id DESC
            LIMIT ?
            """,
            (owner, int(limit)),
        )
        post_rows = await cur.fetchall()

        cur = await db.execute(
            """
            SELECT prompt, dt
            FROM plan_items
            WHERE owner_id=?
            ORDER BY dt DESC, id DESC
            LIMIT ?
            """,
            (owner, int(limit)),
        )
        plan_rows = await cur.fetchall()

    def _clean(v: str, max_len: int = 220) -> str:
        raw = str(v or '').strip()
        return raw[:max_len]

    async with _db_ctx() as db:
        cur = await db.execute(
            "SELECT title, body, prompt, topic FROM generation_history WHERE owner_id=? ORDER BY id DESC LIMIT ?",
            (owner, int(limit)),
        )
        generation_rows = await cur.fetchall()

    recent_posts = [_clean(r[0]) for r in post_rows if _clean(r[0])]
    recent_drafts = [_clean(r[0] or r[1] or r[2]) for r in draft_rows if _clean(r[0] or r[1] or r[2])]
    recent_plan = [_clean(r[0], 160) for r in plan_rows if _clean(r[0], 160)]
    recent_generations = [_clean(' — '.join(part for part in [r[0], r[1]] if _clean(part)), 260) for r in generation_rows if _clean(' — '.join(part for part in [r[0], r[1]] if _clean(part)), 260)]
    return {
        'recent_posts': recent_posts,
        'recent_drafts': recent_drafts,
        'recent_plan': recent_plan,
        'recent_generations': recent_generations,
    }


async def add_generation_history(
    *,
    owner_id: int | None = 0,
    source: str = "generate-post",
    prompt: str = "",
    topic: str = "",
    title: str = "",
    body: str = "",
    cta: str = "",
    short: str = "",
    safety_status: str = "ok",
    draft_id: int = 0,
):
    now = datetime.utcnow().isoformat(timespec="seconds")
    async with _db_ctx() as db:
        await db.execute(
            "INSERT INTO generation_history(owner_id, telegram_user_id, source, prompt, topic, title, body, cta, short, safety_status, draft_id, created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                int(owner_id or 0), int(owner_id or 0), str(source or "generate-post"), str(prompt or ""), str(topic or ""),
                str(title or ""), str(body or ""), str(cta or ""), str(short or ""), str(safety_status or "ok"), int(draft_id or 0), now,
            ),
        )
        await db.commit()


async def list_recent_generation_history(owner_id: int | None = 0, limit: int = 12) -> list[dict]:
    async with _db_ctx() as db:
        cur = await db.execute(
            "SELECT id, source, prompt, topic, title, body, cta, short, safety_status, draft_id, created_at FROM generation_history WHERE owner_id=? ORDER BY id DESC LIMIT ?",
            (int(owner_id or 0), int(limit)),
        )
        rows = await cur.fetchall()
    return [
        {
            "id": int(r[0] or 0), "source": r[1], "prompt": r[2], "topic": r[3], "title": r[4],
            "body": r[5], "cta": r[6], "short": r[7], "safety_status": r[8], "draft_id": int(r[9] or 0), "created_at": r[10],
        }
        for r in rows
    ]


async def mark_generation_history_draft_saved(owner_id: int | None = 0, draft_id: int = 0, text: str = ""):
    owner = int(owner_id or 0)
    draft = int(draft_id or 0)
    sample = str(text or "").strip()[:300]
    if not owner or not draft or not sample:
        return
    async with _db_ctx() as db:
        await db.execute(
            "UPDATE generation_history SET draft_id=? WHERE owner_id=? AND draft_id=0 AND substr(body, 1, 300)=?",
            (draft, owner, sample),
        )
        await db.commit()


async def get_post_stats(owner_id: int | None = 0) -> dict:
    async with _db_ctx() as db:
        where = ""
        params: tuple = ()
        if owner_id is not None:
            where = "WHERE owner_id=?"
            params = (int(owner_id),)

        cur = await db.execute(
            f"SELECT COUNT(*), COALESCE(SUM(CASE WHEN content_type='photo' THEN 1 ELSE 0 END), 0), COALESCE(AVG(LENGTH(text)), 0), COALESCE(SUM(CASE WHEN created_at >= datetime('now', '-7 day') THEN 1 ELSE 0 END), 0) FROM post_logs {where}",
            params,
        )
        total, photo_count, avg_len, posted_last_7d = await cur.fetchone()

        cur = await db.execute(
            f"SELECT COUNT(*) FROM schedules {'WHERE owner_id=?' if owner_id is not None else ''}",
            params,
        )
        schedules_total = (await cur.fetchone())[0]

        cur = await db.execute(
            f"SELECT COUNT(*), COALESCE(SUM(CASE WHEN posted=1 THEN 1 ELSE 0 END), 0), COALESCE(SUM(CASE WHEN posted=0 THEN 1 ELSE 0 END), 0) FROM plan_items {'WHERE owner_id=?' if owner_id is not None else ''}",
            params,
        )
        plan_total, plan_posted, plan_pending = await cur.fetchone()

        drafts_where = "WHERE owner_id=? AND status='draft'" if owner_id is not None else "WHERE status='draft'"
        cur = await db.execute(
            f"SELECT COUNT(*) FROM draft_posts {drafts_where}",
            params,
        )
        drafts_total = (await cur.fetchone())[0]

        cur = await db.execute(
            f"SELECT COUNT(*) FROM user_media_inbox {'WHERE owner_id=?' if owner_id is not None else ''}",
            params,
        )
        media_inbox_total = (await cur.fetchone())[0]

        total_posts = int(total or 0)
        photo_posts = int(photo_count or 0)
        text_posts = int(total_posts - photo_posts)
        avg_posts_per_week = round(float(posted_last_7d or 0) / 7.0, 2)

        return {
            "total_posts": total_posts,
            "photo_posts": photo_posts,
            "text_posts": text_posts,
            "avg_length": int(avg_len or 0),
            "schedules_total": int(schedules_total or 0),
            "plan_total": int(plan_total or 0),
            "plan_posted": int(plan_posted or 0),
            "plan_pending": int(plan_pending or 0),
            "drafts_total": int(drafts_total or 0),
            "media_inbox_total": int(media_inbox_total or 0),
            "posted_last_7d": int(posted_last_7d or 0),
            "avg_posts_per_week": avg_posts_per_week,
        }


# ---------- draft editor ----------
async def create_draft(
    *,
    owner_id: int | None = 0,
    channel_target: str = "",
    text: str = "",
    prompt: str = "",
    topic: str = "",
    media_type: str = "none",
    media_ref: str = "",
    media_meta_json: str = "",
    buttons_json: str = "[]",
    pin_post: int = 0,
    comments_enabled: int = 1,
    ad_mark: int = 0,
    first_reaction: str = "",
    reply_to_message_id: int = 0,
    status: str = "draft",
    send_silent: int = 0,
    draft_source: str = "",
    news_source_json: str = "",
) -> int:
    now = datetime.utcnow().isoformat(timespec="seconds")
    async with _db_ctx() as db:
        cur = await db.execute(
            """
            INSERT INTO draft_posts(
                owner_id, telegram_user_id, channel_target, text, prompt, topic, media_type, media_ref, media_meta_json, buttons_json,
                pin_post, comments_enabled, ad_mark, first_reaction, reply_to_message_id, status, created_at, updated_at, send_silent,
                draft_source, news_source_json
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                int(owner_id or 0),
                int(owner_id or 0),
                channel_target,
                text,
                prompt,
                topic,
                media_type,
                media_ref,
                media_meta_json,
                buttons_json,
                int(pin_post),
                int(comments_enabled),
                int(ad_mark),
                first_reaction,
                int(reply_to_message_id or 0),
                status,
                now,
                now,
                int(send_silent),
                str(draft_source or ""),
                str(news_source_json or ""),
            ),
        )
        await db.commit()
        return int(cur.lastrowid)


async def get_draft(draft_id: int, owner_id: int | None = 0) -> dict | None:
    async with _db_ctx() as db:
        cur = await db.execute(
            """
            SELECT id, owner_id, channel_target, text, prompt, topic, media_type, media_ref, media_meta_json, buttons_json,
                   pin_post, comments_enabled, ad_mark, first_reaction, reply_to_message_id, status, created_at, updated_at,
                   send_silent, draft_source, news_source_json
            FROM draft_posts WHERE id=? AND owner_id=?
            """,
            (int(draft_id), int(owner_id or 0)),
        )
        row = await cur.fetchone()
        return _draft_from_row(row)


async def get_latest_draft(owner_id: int | None = 0) -> dict | None:
    async with _db_ctx() as db:
        cur = await db.execute(
            """
            SELECT id, owner_id, channel_target, text, prompt, topic, media_type, media_ref, media_meta_json, buttons_json,
                   pin_post, comments_enabled, ad_mark, first_reaction, reply_to_message_id, status, created_at, updated_at,
                   send_silent, draft_source, news_source_json
            FROM draft_posts WHERE owner_id=? AND status='draft' ORDER BY id DESC LIMIT 1
            """,
            (int(owner_id or 0),),
        )
        row = await cur.fetchone()
        return _draft_from_row(row)


async def update_draft_field(draft_id: int, owner_id: int | None, field: str, value):
    allowed = {
        "channel_target",
        "text",
        "prompt",
        "topic",
        "media_type",
        "media_ref",
        "media_meta_json",
        "buttons_json",
        "pin_post",
        "comments_enabled",
        "ad_mark",
        "first_reaction",
        "reply_to_message_id",
        "status",
        "send_silent",
        "news_source_json",
    }
    if field not in allowed:
        raise ValueError("Unsupported draft field")
    now = datetime.utcnow().isoformat(timespec="seconds")
    async with _db_ctx() as db:
        await db.execute(
            f"UPDATE draft_posts SET {field}=?, updated_at=? WHERE id=? AND owner_id=?",
            (value, now, int(draft_id), int(owner_id or 0)),
        )
        await db.commit()


async def delete_draft(draft_id: int, owner_id: int | None = 0):
    async with _db_ctx() as db:
        await db.execute("DELETE FROM draft_posts WHERE id=? AND owner_id=?", (int(draft_id), int(owner_id or 0)))
        await db.commit()


async def list_drafts(owner_id: int | None = 0, limit: int = 20) -> list[dict]:
    async with _db_ctx() as db:
        cur = await db.execute(
            """
            SELECT id, owner_id, channel_target, text, prompt, topic, media_type, media_ref, media_meta_json, buttons_json,
                   pin_post, comments_enabled, ad_mark, first_reaction, reply_to_message_id, status, created_at, updated_at,
                   send_silent, draft_source, news_source_json
            FROM draft_posts WHERE owner_id=? ORDER BY id DESC LIMIT ?
            """,
            (int(owner_id or 0), int(limit)),
        )
        rows = await cur.fetchall()
        return [_draft_from_row(r) for r in rows]


async def count_drafts(owner_id: int | None = 0, status: str | None = None) -> int:
    async with _db_ctx() as db:
        if status is None:
            cur = await db.execute(
                "SELECT COUNT(*) FROM draft_posts WHERE owner_id=?",
                (int(owner_id or 0),),
            )
        else:
            cur = await db.execute(
                "SELECT COUNT(*) FROM draft_posts WHERE owner_id=? AND status=?",
                (int(owner_id or 0), str(status)),
            )
        row = await cur.fetchone()
        return int(row[0] or 0) if row else 0


async def list_all_active_draft_media_refs() -> list[str]:
    """Return media_ref values for all active (draft/publishing) drafts across all owners.

    This is a batch query that avoids the N+1 pattern of iterating owners.
    """
    async with _db_ctx() as conn:
        cur = await conn.execute(
            "SELECT media_ref FROM draft_posts WHERE status IN ('draft', 'publishing') AND media_ref IS NOT NULL AND media_ref != ''",
        )
        rows = await cur.fetchall()
        return [str(r[0]) for r in rows if r[0]]


# ---------- media inbox ----------
async def upsert_user_media(
    *,
    owner_id: int,
    message_id: int = 0,
    media_group_id: str = "",
    kind: str = "video",
    file_id: str = "",
    file_unique_id: str = "",
    file_name: str = "",
    file_size: int = 0,
    mime_type: str = "",
    duration: int = 0,
    width: int = 0,
    height: int = 0,
    caption: str = "",
    source_chat_id: int = 0,
    source_message_date: str = "",
) -> int:
    now = datetime.utcnow().isoformat(timespec="seconds")
    async with _db_ctx() as db:
        if file_unique_id:
            cur = await db.execute(
                "SELECT id, used_count FROM user_media_inbox WHERE owner_id=? AND file_unique_id=?",
                (int(owner_id), file_unique_id),
            )
            row = await cur.fetchone()
            if row:
                await db.execute(
                    """
                    UPDATE user_media_inbox
                    SET telegram_user_id=?, message_id=?, media_group_id=?, kind=?, file_id=?, file_name=?,
                        file_size=?, mime_type=?, duration=?, width=?, height=?, caption=?, source_chat_id=?,
                        source_message_date=?, created_at=?
                    WHERE id=?
                    """,
                    (
                        int(owner_id), int(message_id or 0), media_group_id or "", kind or "video", file_id or "",
                        file_name or "", int(file_size or 0), mime_type or "", int(duration or 0), int(width or 0),
                        int(height or 0), caption or "", int(source_chat_id or 0), source_message_date or "", now, int(row[0]),
                    ),
                )
                await db.commit()
                return int(row[0])
        cur = await db.execute(
            """
            INSERT INTO user_media_inbox(
                owner_id, telegram_user_id, message_id, media_group_id, kind, file_id, file_unique_id,
                file_name, file_size, mime_type, duration, width, height, caption, source_chat_id, source_message_date, created_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                int(owner_id), int(owner_id), int(message_id or 0), media_group_id or "", kind or "video", file_id or "",
                file_unique_id or "", file_name or "", int(file_size or 0), mime_type or "", int(duration or 0),
                int(width or 0), int(height or 0), caption or "", int(source_chat_id or 0), source_message_date or "", now,
            ),
        )
        await db.commit()
        return int(cur.lastrowid)


async def list_user_media(owner_id: int, limit: int = 30, kind: str | None = None) -> list[dict]:
    async with _db_ctx() as db:
        if kind:
            cur = await db.execute(
                """
                SELECT id, owner_id, message_id, media_group_id, kind, file_id, file_unique_id, file_name, file_size, mime_type,
                       duration, width, height, caption, source_chat_id, source_message_date, used_count, last_used_at, created_at
                FROM user_media_inbox
                WHERE owner_id=? AND kind=?
                ORDER BY id DESC LIMIT ?
                """,
                (int(owner_id), str(kind), int(limit)),
            )
        else:
            cur = await db.execute(
                """
                SELECT id, owner_id, message_id, media_group_id, kind, file_id, file_unique_id, file_name, file_size, mime_type,
                       duration, width, height, caption, source_chat_id, source_message_date, used_count, last_used_at, created_at
                FROM user_media_inbox
                WHERE owner_id=?
                ORDER BY id DESC LIMIT ?
                """,
                (int(owner_id), int(limit)),
            )
        rows = await cur.fetchall()
    out=[]
    for r in rows:
        out.append({
            "id": int(r[0]), "owner_id": int(r[1]), "telegram_user_id": int(r[1]), "message_id": int(r[2] or 0),
            "media_group_id": r[3] or "", "kind": r[4] or "video", "file_id": r[5] or "", "file_unique_id": r[6] or "",
            "file_name": r[7] or "", "file_size": int(r[8] or 0), "mime_type": r[9] or "", "duration": int(r[10] or 0),
            "width": int(r[11] or 0), "height": int(r[12] or 0), "caption": r[13] or "", "source_chat_id": int(r[14] or 0),
            "source_message_date": r[15] or "", "used_count": int(r[16] or 0), "last_used_at": r[17] or "", "created_at": r[18] or "",
        })
    return out


async def get_user_media(item_id: int, owner_id: int) -> dict | None:
    """Fetch a single media item by id and owner. Uses a direct WHERE clause instead
    of fetching up to 200 rows and scanning in Python (previous N+1 pattern)."""
    async with _db_ctx() as db:
        cur = await db.execute(
            """
            SELECT id, owner_id, message_id, media_group_id, kind, file_id, file_unique_id, file_name, file_size, mime_type,
                   duration, width, height, caption, source_chat_id, source_message_date, used_count, last_used_at, created_at
            FROM user_media_inbox
            WHERE id=? AND owner_id=?
            LIMIT 1
            """,
            (int(item_id), int(owner_id)),
        )
        row = await cur.fetchone()
    if not row:
        return None
    return {
        "id": int(row[0]), "owner_id": int(row[1]), "telegram_user_id": int(row[1]), "message_id": int(row[2] or 0),
        "media_group_id": row[3] or "", "kind": row[4] or "video", "file_id": row[5] or "", "file_unique_id": row[6] or "",
        "file_name": row[7] or "", "file_size": int(row[8] or 0), "mime_type": row[9] or "", "duration": int(row[10] or 0),
        "width": int(row[11] or 0), "height": int(row[12] or 0), "caption": row[13] or "", "source_chat_id": int(row[14] or 0),
        "source_message_date": row[15] or "", "used_count": int(row[16] or 0), "last_used_at": row[17] or "", "created_at": row[18] or "",
    }


async def mark_user_media_used(item_id: int, owner_id: int) -> None:
    now = datetime.utcnow().isoformat(timespec="seconds")
    async with _db_ctx() as db:
        await db.execute(
            """
            UPDATE user_media_inbox
            SET used_count = COALESCE(used_count, 0) + 1,
                last_used_at = ?
            WHERE id=? AND owner_id=?
            """,
            (now, int(item_id), int(owner_id)),
        )
        await db.commit()


async def unmark_user_media_used(item_id: int, owner_id: int) -> None:
    now = datetime.utcnow().isoformat(timespec="seconds")
    async with _db_ctx() as db:
        await db.execute(
            """
            UPDATE user_media_inbox
            SET used_count = CASE WHEN COALESCE(used_count, 0) > 0 THEN used_count - 1 ELSE 0 END,
                last_used_at = ?
            WHERE id=? AND owner_id=?
            """,
            (now, int(item_id), int(owner_id)),
        )
        await db.commit()


async def delete_user_media(item_id: int, owner_id: int) -> None:
    async with _db_ctx() as db:
        await db.execute(
            "DELETE FROM user_media_inbox WHERE id=? AND owner_id=?",
            (int(item_id), int(owner_id)),
        )
        await db.commit()


async def assign_empty_drafts_channel(owner_id: int, channel_target: str) -> None:
    owner = int(owner_id or 0)
    target = (channel_target or '').strip()
    if not target:
        return
    now = datetime.utcnow().isoformat(timespec='seconds')
    async with _db_ctx() as db:
        await db.execute(
            "UPDATE draft_posts SET channel_target=?, updated_at=? WHERE owner_id=? AND COALESCE(channel_target,'')=''",
            (target, now, owner),
        )
        await db.commit()


# ---------- owner discovery ----------
async def list_owner_ids() -> list[int]:
    async with _db_ctx() as db:
        ids = set()
        for table in ("schedules", "plan_items", "post_logs", "draft_posts", "generation_history", "user_media_inbox", "news_logs", "channel_profiles"):
            try:
                cur = await db.execute(f"SELECT DISTINCT COALESCE(NULLIF(telegram_user_id, 0), owner_id) FROM {table}")
                rows = await cur.fetchall()
                ids.update(int(r[0] or 0) for r in rows)
            except Exception:
                pass
        cur = await db.execute("SELECT key FROM settings WHERE key LIKE 'u:%:%'")
        for (key,) in await cur.fetchall():
            try:
                ids.add(int(str(key).split(":", 2)[1]))
            except Exception:
                pass
        ids.discard(0)
        return sorted(ids)


# ---------- news helpers ----------
async def is_news_used(source_url: str, owner_id: int | None = 0, *, channel_target: str = "") -> bool:
    """Check if a news URL was already used.

    If *channel_target* is provided, checks per-channel dedup first,
    then falls back to owner-level (same news won't be reused across channels either).
    """
    async with _db_ctx() as db:
        if channel_target:
            cur = await db.execute(
                "SELECT 1 FROM news_logs WHERE owner_id=? AND channel_target=? AND source_url=? LIMIT 1",
                (int(owner_id or 0), channel_target, source_url),
            )
            row = await cur.fetchone()
            if row is not None:
                return True
        # Fallback: also check owner-level dedup (avoid same news in different channels)
        cur = await db.execute(
            "SELECT 1 FROM news_logs WHERE owner_id=? AND source_url=? LIMIT 1",
            (int(owner_id or 0), source_url),
        )
        row = await cur.fetchone()
        return row is not None


async def log_news(source_url: str, source_title: str = "", owner_id: int | None = 0, *, channel_target: str = ""):
    """Log a news item as used.  Stores channel_target for per-channel dedup."""
    async with _db_ctx() as db:
        await db.execute(
            "INSERT INTO news_logs(owner_id, telegram_user_id, source_url, source_title, created_at, channel_target) VALUES(?,?,?,?,?,?)",
            (int(owner_id or 0), int(owner_id or 0), source_url, source_title,
             datetime.utcnow().isoformat(timespec="seconds"), channel_target),
        )
        await db.commit()


async def get_recent_post_topics(owner_id: int | None = 0, limit: int = 15) -> list[str]:
    """Return recent post titles/topics for anti-repetition injection into LLM prompts.

    Pulls from generation_history (titles) and post_logs (topics), deduplicates, and
    returns up to *limit* short strings. Designed to be lightweight for SQLite on low-RAM servers.
    """
    owner = int(owner_id or 0)
    topics: list[str] = []
    seen: set[str] = set()

    async with _db_ctx() as db:
        cur = await db.execute(
            "SELECT title, topic FROM generation_history WHERE owner_id=? ORDER BY id DESC LIMIT ?",
            (owner, limit),
        )
        for row in await cur.fetchall():
            text = (row[0] or row[1] or "").strip()[:120]
            if text and text not in seen:
                seen.add(text)
                topics.append(text)

        remaining = limit - len(topics)
        if remaining > 0:
            cur = await db.execute(
                "SELECT topic, prompt FROM post_logs WHERE owner_id=? ORDER BY id DESC LIMIT ?",
                (owner, limit),
            )
            for row in await cur.fetchall():
                text = (row[0] or row[1] or "").strip()[:120]
                if text and text not in seen and len(topics) < limit:
                    seen.add(text)
                    topics.append(text)

    return topics


async def get_owner_bootstrap_snapshot(owner_id: int, *, drafts_limit: int = 50, plan_limit: int = 300, media_limit: int = 24) -> dict:
    owner = int(owner_id or 0)
    settings_keys = [
        "posts_enabled", "posting_mode", "news_enabled", "news_interval_hours",
        "news_sources", "auto_hashtags", "topic", "channel_target"
    ]
    db = await _connect()
    try:
        cur = await db.execute(
            "SELECT * FROM channel_profiles WHERE owner_id=? ORDER BY is_active DESC, id ASC",
            (owner,),
        )
        channel_rows = await cur.fetchall()
        cols = [d[0] for d in cur.description] if cur.description else []
        def _cp_row(r: tuple) -> dict:
            d = dict(zip(cols, r)) if cols else {
                "id": r[0], "owner_id": r[1], "title": r[2], "channel_target": r[3],
                "topic": r[4], "is_active": r[5], "created_at": r[6], "updated_at": r[7],
            }
            d["is_active"] = int(d.get("is_active") or 0)
            d["id"] = int(d.get("id") or 0)
            return d
        channels = [_cp_row(r) for r in channel_rows]
        active = next((ch for ch in channels if int(ch.get("is_active",0)) == 1), None)

        cur = await db.execute(
            "SELECT id, owner_id, channel_target, text, prompt, topic, media_type, media_ref, media_meta_json, buttons_json, pin_post, comments_enabled, ad_mark, first_reaction, reply_to_message_id, status, created_at, updated_at, send_silent, draft_source, news_source_json FROM draft_posts WHERE owner_id=? ORDER BY id DESC LIMIT ?",
            (owner, int(drafts_limit)),
        )
        rows = await cur.fetchall()
        drafts = [_draft_from_row(r) for r in rows]

        cur = await db.execute(
            "SELECT id, dt, kind, payload, created_at, enabled, posted, owner_id FROM plan_items WHERE owner_id=? ORDER BY dt ASC, id ASC LIMIT ?",
            (owner, int(plan_limit)),
        )
        rows = await cur.fetchall()
        plan = []
        for r in rows:
            payload = r[3]
            plan.append({
                "id": r[0], "dt": r[1], "kind": r[2], "payload": payload, "created_at": r[4],
                "enabled": int(r[5]), "posted": int(r[6]), "owner_id": int(r[7]),
                "topic": payload if r[2] == "topic" else "", "prompt": payload if r[2] == "prompt" else "",
            })

        cur = await db.execute(
            "SELECT id, time_hhmm, days, enabled, owner_id FROM schedules WHERE owner_id=? ORDER BY id ASC",
            (owner,),
        )
        rows = await cur.fetchall()
        schedules = [{
            "id": r[0], "time_hhmm": r[1], "time": r[1], "days": r[2], "enabled": int(r[3]), "owner_id": int(r[4])
        } for r in rows]

        cur = await db.execute(
            "SELECT COUNT(*), COALESCE(SUM(CASE WHEN content_type='photo' THEN 1 ELSE 0 END), 0), COALESCE(AVG(LENGTH(text)), 0) FROM post_logs WHERE owner_id=?",
            (owner,),
        )
        total, photo_count, avg_len = await cur.fetchone()
        cur = await db.execute("SELECT COUNT(*) FROM schedules WHERE owner_id=?", (owner,))
        schedules_total = (await cur.fetchone())[0]
        cur = await db.execute("SELECT COUNT(*), COALESCE(SUM(CASE WHEN posted=1 THEN 1 ELSE 0 END), 0) FROM plan_items WHERE owner_id=?", (owner,))
        plan_total, plan_posted = await cur.fetchone()
        stats = {
            "total_posts": int(total or 0), "photo_posts": int(photo_count or 0), "text_posts": int((total or 0) - (photo_count or 0)),
            "avg_length": int(avg_len or 0), "schedules_total": int(schedules_total or 0), "plan_total": int(plan_total or 0),
            "plan_posted": int(plan_posted or 0),
        }

        cur = await db.execute(
            "SELECT id, owner_id, message_id, media_group_id, kind, file_id, file_unique_id, file_name, file_size, mime_type, duration, width, height, caption, source_chat_id, source_message_date, used_count, last_used_at, created_at FROM user_media_inbox WHERE owner_id=? ORDER BY id DESC LIMIT ?",
            (owner, int(media_limit)),
        )
        rows = await cur.fetchall()
        media = [{
            "id": int(r[0]), "owner_id": int(r[1]), "telegram_user_id": int(r[1]), "message_id": int(r[2] or 0), "media_group_id": r[3] or "",
            "kind": r[4] or "video", "file_id": r[5] or "", "file_unique_id": r[6] or "", "file_name": r[7] or "", "file_size": int(r[8] or 0),
            "mime_type": r[9] or "", "duration": int(r[10] or 0), "width": int(r[11] or 0), "height": int(r[12] or 0),
            "caption": r[13] or "", "source_chat_id": int(r[14] or 0), "source_message_date": r[15] or "",
            "used_count": int(r[16] or 0), "last_used_at": r[17] or "", "created_at": r[18] or "",
        } for r in rows]

        cur = await db.execute("SELECT COUNT(*) FROM draft_posts WHERE owner_id=? AND status='draft'", (owner,))
        draft_row = await cur.fetchone()
        drafts_current = int(draft_row[0] or 0) if draft_row else 0
    finally:
        await db.close()

    settings = await get_settings_bulk(settings_keys, owner_id=owner)
    return {
        "channels": channels,
        "active_channel": active,
        "drafts": drafts,
        "plan": plan,
        "schedules": schedules,
        "stats": stats,
        "media_inbox": media,
        "drafts_current": drafts_current,
        "settings": {
            "posts_enabled": str(settings.get("posts_enabled") or "0"),
            "posting_mode": str(settings.get("posting_mode") or "manual"),
            "news_enabled": str(settings.get("news_enabled") or "0"),
            "news_interval_hours": str(settings.get("news_interval_hours") or "6"),
            "news_sources": str(settings.get("news_sources") or ""),
            "auto_hashtags": str(settings.get("auto_hashtags") or "0"),
            "topic": str(settings.get("topic") or (active.get("topic", "") if active else "")),
            "channel_target": str(settings.get("channel_target") or (active.get("channel_target", "") if active else "")),
        },
    }


# ---------- channels ----------
_CHANNEL_PROFILE_STRUCTURED_FIELDS = [
    "topic_raw", "topic_family", "topic_subfamily", "audience_type", "style_mode",
    "content_goals", "preferred_formats", "forbidden_topics", "forbidden_claims",
    "visual_policy", "forbidden_visual_classes", "rubric_map", "news_policy",
    "posting_mode", "sensitivity_flags",
]


def _channel_profile_row_to_dict(r: tuple, columns: list[str]) -> dict:
    """Maps a channel_profiles DB row to a dict using column name list."""
    result = {}
    for i, col in enumerate(columns):
        if i < len(r):
            val = r[i]
            if col in ("id", "owner_id", "is_active"):
                result[col] = int(val) if val is not None else 0
            else:
                result[col] = val or ""
    return result


_UNSET = object()  # sentinel: field was not provided by caller


async def upsert_channel_profile(
    owner_id: int | None,
    channel_target: str,
    *,
    title: str = "",
    topic: str = "",
    make_active: bool = True,
    # Structured profile fields — _UNSET means "keep existing value on update"
    topic_raw: str | object = _UNSET,
    topic_family: str | object = _UNSET,
    topic_subfamily: str | object = _UNSET,
    audience_type: str | object = _UNSET,
    style_mode: str | object = _UNSET,
    content_goals: str | object = _UNSET,
    preferred_formats: str | object = _UNSET,
    forbidden_topics: str | object = _UNSET,
    forbidden_claims: str | object = _UNSET,
    visual_policy: str | object = _UNSET,
    forbidden_visual_classes: str | object = _UNSET,
    rubric_map: str | object = _UNSET,
    news_policy: str | object = _UNSET,
    posting_mode: str | object = _UNSET,
    sensitivity_flags: str | object = _UNSET,
    # Author role fields
    author_role_type: str | object = _UNSET,
    author_role_description: str | object = _UNSET,
    author_activities: str | object = _UNSET,
    author_forbidden_claims: str | object = _UNSET,
):
    owner = int(owner_id or 0)
    channel_target = (channel_target or "").strip()
    if not channel_target:
        return
    # Never default title to a numeric channel_target (raw Telegram IDs like -100...).
    # Only use channel_target as fallback when it looks like a human-readable @username.
    _raw_title = (title or "").strip()
    if _raw_title and not re.match(r'^-?\d+$', _raw_title):
        title = _raw_title
    else:
        _ct = (channel_target or "").strip()
        if _ct and not re.match(r'^-?\d+$', _ct):
            title = _ct
        else:
            title = _raw_title  # keep whatever was provided, even empty
    topic = (topic or "").strip()
    now = datetime.utcnow().isoformat(timespec="seconds")
    async with _db_ctx() as db:
        if make_active:
            await db.execute("UPDATE channel_profiles SET is_active=0 WHERE owner_id=?", (owner,))
        cur = await db.execute(
            "SELECT id, topic, title FROM channel_profiles WHERE owner_id=? AND channel_target=? LIMIT 1",
            (owner, channel_target),
        )
        row = await cur.fetchone()
        if row:
            profile_id = int(row[0])
            # Prefer provided title → existing DB title → non-numeric channel_target; never raw IDs
            _existing_title = (row[2] or "").strip() if row[2] else ""
            _candidate_titles = [title, _existing_title]
            _non_numeric_ct = channel_target if (channel_target and not re.match(r'^-?\d+$', channel_target)) else ""
            if _non_numeric_ct:
                _candidate_titles.append(_non_numeric_ct)
            new_title = next((t for t in _candidate_titles if t and not re.match(r'^-?\d+$', t)), title or _existing_title or "")
            new_topic = topic or row[1] or ""

            # Build partial UPDATE: only set fields that were explicitly passed
            updates: list[str] = ["title=?", "topic=?", "is_active=?", "updated_at=?"]
            params: list = [new_title, new_topic, 1 if make_active else 0, now]

            _structured_fields = {
                "topic_raw": topic_raw, "topic_family": topic_family,
                "topic_subfamily": topic_subfamily, "audience_type": audience_type,
                "style_mode": style_mode, "content_goals": content_goals,
                "preferred_formats": preferred_formats, "forbidden_topics": forbidden_topics,
                "forbidden_claims": forbidden_claims, "visual_policy": visual_policy,
                "forbidden_visual_classes": forbidden_visual_classes, "rubric_map": rubric_map,
                "news_policy": news_policy, "posting_mode": posting_mode,
                "sensitivity_flags": sensitivity_flags,
                "author_role_type": author_role_type, "author_role_description": author_role_description,
                "author_activities": author_activities, "author_forbidden_claims": author_forbidden_claims,
            }
            for col, val in _structured_fields.items():
                if val is not _UNSET:
                    # Special case: topic_raw defaults to topic if explicitly provided but empty
                    if col == "topic_raw" and not val:
                        val = new_topic
                    updates.append(f"{col}=?")
                    params.append(str(val))

            params.append(profile_id)
            await db.execute(
                f"UPDATE channel_profiles SET {', '.join(updates)} WHERE id=?",
                tuple(params),
            )
        else:
            # INSERT: use explicit values or sensible defaults for new profiles
            def _val(v, default=""):
                return str(v) if v is not _UNSET else default

            _topic_raw = _val(topic_raw) or topic
            await db.execute(
                """INSERT INTO channel_profiles(
                   owner_id, telegram_user_id, title, channel_target, topic, is_active, created_at, updated_at,
                   topic_raw, topic_family, topic_subfamily, audience_type, style_mode,
                   content_goals, preferred_formats, forbidden_topics, forbidden_claims,
                   visual_policy, forbidden_visual_classes, rubric_map, news_policy,
                   posting_mode, sensitivity_flags,
                   author_role_type, author_role_description, author_activities, author_forbidden_claims
                   ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (owner, owner, title, channel_target, topic, 1 if make_active else 0, now, now,
                 _topic_raw, _val(topic_family), _val(topic_subfamily), _val(audience_type), _val(style_mode),
                 _val(content_goals), _val(preferred_formats), _val(forbidden_topics), _val(forbidden_claims),
                 _val(visual_policy, "auto"), _val(forbidden_visual_classes), _val(rubric_map),
                 _val(news_policy, "standard"), _val(posting_mode, "manual"), _val(sensitivity_flags),
                 _val(author_role_type), _val(author_role_description), _val(author_activities),
                 _val(author_forbidden_claims)),
            )
        await db.commit()


async def list_channel_profiles(owner_id: int | None = 0) -> list[dict]:
    async with _db_ctx() as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT * FROM channel_profiles WHERE owner_id=? ORDER BY is_active DESC, id ASC",
            (int(owner_id or 0),),
        )
        rows = await cur.fetchall()
        return [dict(r) for r in rows]


async def get_active_channel_profile(owner_id: int | None = 0) -> dict | None:
    async with _db_ctx() as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT * FROM channel_profiles WHERE owner_id=? AND is_active=1 ORDER BY id DESC LIMIT 1",
            (int(owner_id or 0),),
        )
        r = await cur.fetchone()
        if not r:
            return None
        return dict(r)


async def set_active_channel_profile(profile_id: int, owner_id: int | None = 0) -> dict | None:
    owner = int(owner_id or 0)
    async with _db_ctx() as db:
        await db.execute("UPDATE channel_profiles SET is_active=0 WHERE owner_id=?", (owner,))
        await db.execute("UPDATE channel_profiles SET is_active=1, updated_at=? WHERE owner_id=? AND id=?", (datetime.utcnow().isoformat(timespec='seconds'), owner, int(profile_id)))
        await db.commit()
    profile = await get_active_channel_profile(owner_id=owner)
    if profile:
        await set_setting("channel_target", profile.get("channel_target", ""), owner_id=owner)
        await set_setting("topic", profile.get("topic", ""), owner_id=owner)
        await assign_empty_drafts_channel(owner, profile.get("channel_target", ""))
    return profile


async def sync_channel_profile_topic(owner_id: int | None, channel_target: str, topic: str):
    owner = int(owner_id or 0)
    async with _db_ctx() as db:
        await db.execute(
            "UPDATE channel_profiles SET topic=?, updated_at=? WHERE owner_id=? AND channel_target=?",
            ((topic or "").strip(), datetime.utcnow().isoformat(timespec='seconds'), owner, (channel_target or '').strip()),
        )
        await db.commit()


async def delete_schedule(schedule_id: int, owner_id: int | None = 0):
    async with _db_ctx() as db:
        await db.execute(
            "DELETE FROM schedules WHERE id=? AND owner_id=?",
            (int(schedule_id), int(owner_id or 0)),
        )
        await db.commit()


async def delete_channel_profile(profile_id: int, owner_id: int | None = 0) -> dict | None:
    owner = int(owner_id or 0)
    async with _db_ctx() as db:
        cur = await db.execute(
            "SELECT id, channel_target, is_active FROM channel_profiles WHERE id=? AND owner_id=? LIMIT 1",
            (int(profile_id), owner),
        )
        row = await cur.fetchone()
        if not row:
            return None
        was_active = int(row[2]) == 1
        removed_target = str(row[1] or "")
        await db.execute("DELETE FROM channel_profiles WHERE id=? AND owner_id=?", (int(profile_id), owner))
        if was_active:
            cur = await db.execute(
                "SELECT id, channel_target, topic FROM channel_profiles WHERE owner_id=? ORDER BY id ASC LIMIT 1",
                (owner,),
            )
            replacement = await cur.fetchone()
            await db.execute("UPDATE channel_profiles SET is_active=0 WHERE owner_id=?", (owner,))
            if replacement:
                await db.execute(
                    "UPDATE channel_profiles SET is_active=1, updated_at=? WHERE id=?",
                    (datetime.utcnow().isoformat(timespec='seconds'), int(replacement[0])),
                )
        await db.commit()

    if was_active:
        new_active = await get_active_channel_profile(owner_id=owner)
        await set_setting("channel_target", new_active.get("channel_target", "") if new_active else "", owner_id=owner)
        await set_setting("topic", new_active.get("topic", "") if new_active else "", owner_id=owner)
    elif removed_target and (await get_setting("channel_target", owner_id=owner) or "") == removed_target:
        await set_setting("channel_target", "", owner_id=owner)
    return await get_active_channel_profile(owner_id=owner)


async def list_recent_image_refs(owner_id: int | None = 0, limit: int = 30) -> list[str]:
    async with _db_ctx() as db:
        cur = await db.execute(
            """
            SELECT file_id
            FROM post_logs
            WHERE owner_id=? AND file_id!=''
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(owner_id or 0), int(limit)),
        )
        rows = await cur.fetchall()
        return [str(r[0]).strip() for r in rows if r and str(r[0]).strip()]


async def list_recent_draft_image_refs(owner_id: int | None = 0, limit: int = 30) -> list[str]:
    async with _db_ctx() as db:
        cur = await db.execute(
            """
            SELECT media_ref
            FROM draft_posts
            WHERE owner_id=? AND media_ref!=''
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(owner_id or 0), int(limit)),
        )
        rows = await cur.fetchall()
        return [str(r[0]).strip() for r in rows if r and str(r[0]).strip()]

async def claim_draft_for_publish(draft_id: int, owner_id: int) -> bool:
    """Atomically set status to 'publishing' if currently 'draft'. Returns True if claimed."""
    now = datetime.utcnow().isoformat(timespec="seconds")
    async with _db_ctx() as db:
        cur = await db.execute(
            "UPDATE draft_posts SET status='publishing', updated_at=? WHERE id=? AND owner_id=? AND status='draft'",
            (now, int(draft_id), int(owner_id)),
        )
        await db.commit()
        return (cur.rowcount or 0) > 0


async def release_draft_claim(draft_id: int, owner_id: int, status: str = "draft") -> None:
    """Reset a stuck 'publishing' draft back to 'draft' or 'failed'. Safe to call after a publish failure."""
    allowed = {"draft", "failed"}
    if status not in allowed:
        status = "draft"
    now = datetime.utcnow().isoformat(timespec="seconds")
    async with _db_ctx() as db:
        await db.execute(
            "UPDATE draft_posts SET status=?, updated_at=? WHERE id=? AND owner_id=? AND status='publishing'",
            (status, now, int(draft_id), int(owner_id)),
        )
        await db.commit()


# ---------- subscriptions ----------

TIER_FREE = "free"
TIER_PRO = "pro"
TIER_MAX = "max"
FREE_TIER_GENERATIONS_LIMIT = 5  # per calendar month (main generation: generate-text, generate-post, drafts/generate)
TRIAL_DAYS = 3

# ---------------------------------------------------------------------------
# Per-feature free-tier quotas (monthly).
# Features NOT listed here fall back to the main generation quota above.
# Pro / Max users have unlimited access to all features (no quota check).
# ---------------------------------------------------------------------------
FREE_TIER_FEATURE_LIMITS: dict[str, int] = {
    "generate":      5,   # generate-text, generate-post, drafts/generate
    "rewrite":       5,   # ai/rewrite
    "hashtags":      5,   # ai/add-hashtags (lightweight, generous)
    "assets":        3,   # ai/assets
    "plan_generate": 2,   # plan/generate
    "assistant":     3,   # assistant/chat
    "ai_insights":   1,   # analytics/ai-insights
    "voice":         1,   # voice-to-post (Whisper + generation, expensive)
    "news_generate": 1,   # on-demand news post generation (manual trigger)
}

# Pro tier gets very generous limits for features that cost real money (AI calls).
# None means unlimited.
PRO_TIER_FEATURE_LIMITS: dict[str, int | None] = {
    "generate":      None,
    "rewrite":       None,
    "hashtags":      None,
    "assets":        None,
    "plan_generate": None,
    "assistant":     None,
    "ai_insights":   None,
    "voice":         None,
    "news_generate": None,
}

# Max tier — same as Pro but documented separately for future expansion.
MAX_TIER_FEATURE_LIMITS: dict[str, int | None] = {
    "generate":      None,
    "rewrite":       None,
    "hashtags":      None,
    "assets":        None,
    "plan_generate": None,
    "assistant":     None,
    "ai_insights":   None,
    "voice":         None,
    "news_generate": None,
}


async def ensure_web_user(owner_id: int) -> None:
    """Ensure a user_subscriptions record exists for the given owner_id.

    Used during web login to guarantee the user has at least a free-tier record
    before they hit any authenticated endpoint.  This is a targeted INSERT OR
    IGNORE — it does NOT read back the record or trigger side-effects.
    """
    now = datetime.utcnow().isoformat(timespec="seconds")
    async with _db_ctx() as conn:
        await conn.execute(
            "INSERT OR IGNORE INTO user_subscriptions"
            "(owner_id, subscription_tier, generations_used, "
            "generations_reset_at, trial_ends_at, updated_at, subscription_expires_at, auto_renew) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (int(owner_id), TIER_FREE, 0, now[:7], "", now, "", 0),
        )
        await conn.commit()


async def get_user_subscription(owner_id: int) -> dict:
    """Return subscription record for user. Creates a default free record if absent."""
    now = datetime.utcnow().isoformat(timespec="seconds")
    async with _db_ctx() as db:
        cur = await db.execute(
            "SELECT subscription_tier, generations_used, generations_reset_at, trial_ends_at, updated_at, "
            "subscription_expires_at, auto_renew "
            "FROM user_subscriptions WHERE owner_id=?",
            (int(owner_id),),
        )
        row = await cur.fetchone()
        if not row:
            await db.execute(
                "INSERT OR IGNORE INTO user_subscriptions(owner_id, subscription_tier, generations_used, "
                "generations_reset_at, trial_ends_at, updated_at, subscription_expires_at, auto_renew) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (int(owner_id), TIER_FREE, 0, now[:7], "", now, "", 0),
            )
            await db.commit()
            return {
                "subscription_tier": TIER_FREE,
                "generations_used": 0,
                "generations_reset_at": now[:7],
                "trial_ends_at": "",
                "updated_at": now,
                "subscription_expires_at": "",
                "auto_renew": False,
            }
        return {
            "subscription_tier": str(row[0] or TIER_FREE),
            "generations_used": int(row[1] or 0),
            "generations_reset_at": str(row[2] or now[:7]),
            "trial_ends_at": str(row[3] or ""),
            "updated_at": str(row[4] or now),
            "subscription_expires_at": str(row[5] or ""),
            "auto_renew": bool(int(row[6] or 0)),
        }


async def set_user_subscription(
    owner_id: int,
    tier: str,
    *,
    trial_days: int | None = None,
    expires_at: str | None = None,
    auto_renew: bool | None = None,
) -> None:
    """Set subscription tier for user. Optionally start a trial period or set expiry."""
    allowed = {TIER_FREE, TIER_PRO, TIER_MAX}
    if tier not in allowed:
        tier = TIER_FREE
    now = datetime.utcnow()
    now_iso = now.isoformat(timespec="seconds")
    trial_ends = ""
    if trial_days:
        trial_ends = (now + timedelta(days=trial_days)).isoformat(timespec="seconds")
    sub_expires = expires_at if expires_at is not None else (
        (now + timedelta(days=30)).isoformat(timespec="seconds") if tier != TIER_FREE else ""
    )
    renew_val = 1 if auto_renew else 0
    async with _db_ctx() as db:
        await db.execute(
            "INSERT INTO user_subscriptions(owner_id, subscription_tier, generations_used, "
            "generations_reset_at, trial_ends_at, updated_at, subscription_expires_at, auto_renew) "
            "VALUES(?,?,0,?,?,?,?,?) "
            "ON CONFLICT(owner_id) DO UPDATE SET subscription_tier=excluded.subscription_tier, "
            "trial_ends_at=excluded.trial_ends_at, updated_at=excluded.updated_at, "
            "subscription_expires_at=excluded.subscription_expires_at, auto_renew=excluded.auto_renew",
            (int(owner_id), tier, now_iso[:7], trial_ends, now_iso, sub_expires, renew_val),
        )
        await db.commit()


async def increment_generations_used(owner_id: int) -> int:
    """Increment monthly generation counter. Resets automatically on new calendar month. Returns new count."""
    now = datetime.utcnow()
    month_key = now.isoformat(timespec="seconds")[:7]  # "YYYY-MM"
    now_iso = now.isoformat(timespec="seconds")
    async with _db_ctx() as db:
        cur = await db.execute(
            "SELECT generations_used, generations_reset_at FROM user_subscriptions WHERE owner_id=?",
            (int(owner_id),),
        )
        row = await cur.fetchone()
        if not row:
            await db.execute(
                "INSERT OR IGNORE INTO user_subscriptions(owner_id, subscription_tier, generations_used, "
                "generations_reset_at, trial_ends_at, updated_at) VALUES(?,?,1,?,?,?)",
                (int(owner_id), TIER_FREE, month_key, "", now_iso),
            )
            await db.commit()
            return 1
        used = int(row[0] or 0)
        reset_at = str(row[1] or "")
        if reset_at != month_key:
            used = 0
        new_used = used + 1
        await db.execute(
            "UPDATE user_subscriptions SET generations_used=?, generations_reset_at=?, updated_at=? WHERE owner_id=?",
            (new_used, month_key, now_iso, int(owner_id)),
        )
        await db.commit()
        return new_used


async def is_generation_allowed(owner_id: int) -> tuple[bool, dict]:
    """Check whether the user can perform one more generation.

    Returns (allowed, subscription_dict).
    Free tier users are limited to FREE_TIER_GENERATIONS_LIMIT per month.
    Trial and paid users have unlimited generations.
    """
    sub = await get_user_subscription(owner_id)
    tier = sub["subscription_tier"]
    now = datetime.utcnow()

    # Active trial counts as pro access
    trial_ends = sub.get("trial_ends_at") or ""
    if trial_ends:
        try:
            trial_dt = datetime.fromisoformat(trial_ends)
            if trial_dt > now:
                return True, sub
        except Exception:
            pass

    if tier in (TIER_PRO, TIER_MAX):
        return True, sub

    # Free tier: check monthly limit, auto-reset on new month
    month_key = now.isoformat(timespec="seconds")[:7]
    used = int(sub.get("generations_used") or 0)
    reset_at = str(sub.get("generations_reset_at") or "")
    if reset_at != month_key:
        used = 0
    return used < FREE_TIER_GENERATIONS_LIMIT, sub


# ---------- per-feature quota helpers ----------


def _feature_limit_for_tier(feature: str, tier: str) -> int | None:
    """Return monthly limit for *feature* given *tier*.

    ``None`` means unlimited.  Missing feature key falls back to the main
    generation limit for free tier, or unlimited for paid tiers.
    """
    if tier == TIER_MAX:
        return MAX_TIER_FEATURE_LIMITS.get(feature)
    if tier == TIER_PRO:
        return PRO_TIER_FEATURE_LIMITS.get(feature)
    # free tier
    return FREE_TIER_FEATURE_LIMITS.get(feature, FREE_TIER_GENERATIONS_LIMIT)


async def get_feature_usage(owner_id: int, feature: str) -> int:
    """Return how many times *feature* has been used this calendar month."""
    now = datetime.utcnow()
    month_key = now.isoformat(timespec="seconds")[:7]
    async with _db_ctx() as conn:
        cur = await conn.execute(
            "SELECT used, reset_at FROM user_feature_quotas WHERE owner_id=? AND feature=?",
            (int(owner_id), feature),
        )
        row = await cur.fetchone()
        if not row:
            return 0
        used = int(row[0] or 0)
        reset_at = str(row[1] or "")
        if reset_at != month_key:
            return 0
        return used


async def is_feature_allowed(owner_id: int, feature: str) -> tuple[bool, dict, int, int | None]:
    """Check whether *owner_id* can use *feature* once more.

    Returns ``(allowed, subscription_dict, used_this_month, limit_or_none)``.
    ``limit`` is ``None`` when the user has unlimited access.

    Reads both the subscription record and the feature-usage counter in a
    single DB connection to avoid opening two separate connections for what is
    logically one atomic check (previously called get_user_subscription then
    get_feature_usage separately).
    """
    now = datetime.utcnow()
    now_iso = now.isoformat(timespec="seconds")
    month_key = now_iso[:7]

    async with _db_ctx() as conn:
        # --- subscription record ---
        cur = await conn.execute(
            "SELECT subscription_tier, generations_used, generations_reset_at, trial_ends_at, updated_at, "
            "subscription_expires_at, auto_renew "
            "FROM user_subscriptions WHERE owner_id=?",
            (int(owner_id),),
        )
        row = await cur.fetchone()
        if not row:
            await conn.execute(
                "INSERT OR IGNORE INTO user_subscriptions(owner_id, subscription_tier, generations_used, "
                "generations_reset_at, trial_ends_at, updated_at, subscription_expires_at, auto_renew) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (int(owner_id), TIER_FREE, 0, month_key, "", now_iso, "", 0),
            )
            await conn.commit()
            sub = {
                "subscription_tier": TIER_FREE,
                "generations_used": 0,
                "generations_reset_at": month_key,
                "trial_ends_at": "",
                "updated_at": now_iso,
                "subscription_expires_at": "",
                "auto_renew": False,
            }
        else:
            sub = {
                "subscription_tier": str(row[0] or TIER_FREE),
                "generations_used": int(row[1] or 0),
                "generations_reset_at": str(row[2] or month_key),
                "trial_ends_at": str(row[3] or ""),
                "updated_at": str(row[4] or now_iso),
                "subscription_expires_at": str(row[5] or ""),
                "auto_renew": bool(int(row[6] or 0)),
            }

        tier = sub["subscription_tier"]

        # Active trial counts as pro access
        trial_ends = sub.get("trial_ends_at") or ""
        if trial_ends:
            try:
                if datetime.fromisoformat(trial_ends) > now:
                    return True, sub, 0, None
            except Exception:
                pass

        limit = _feature_limit_for_tier(feature, tier)
        if limit is None:
            return True, sub, 0, None

        # --- feature usage (same connection) ---
        cur = await conn.execute(
            "SELECT used, reset_at FROM user_feature_quotas WHERE owner_id=? AND feature=?",
            (int(owner_id), feature),
        )
        row = await cur.fetchone()
        if not row:
            return True, sub, 0, limit
        used = int(row[0] or 0)
        reset_at = str(row[1] or "")
        if reset_at != month_key:
            used = 0
        return used < limit, sub, used, limit


async def increment_feature_used(owner_id: int, feature: str) -> int:
    """Increment the monthly counter for *feature*. Returns new count."""
    now = datetime.utcnow()
    month_key = now.isoformat(timespec="seconds")[:7]
    async with _db_ctx() as conn:
        cur = await conn.execute(
            "SELECT used, reset_at FROM user_feature_quotas WHERE owner_id=? AND feature=?",
            (int(owner_id), feature),
        )
        row = await cur.fetchone()
        if not row:
            await conn.execute(
                "INSERT OR IGNORE INTO user_feature_quotas(owner_id, feature, used, reset_at) VALUES(?,?,1,?)",
                (int(owner_id), feature, month_key),
            )
            await conn.commit()
            return 1
        used = int(row[0] or 0)
        reset_at = str(row[1] or "")
        if reset_at != month_key:
            used = 0
        new_used = used + 1
        await conn.execute(
            "UPDATE user_feature_quotas SET used=?, reset_at=? WHERE owner_id=? AND feature=?",
            (new_used, month_key, int(owner_id), feature),
        )
        await conn.commit()
        return new_used


async def get_all_feature_usage(owner_id: int) -> dict[str, int]:
    """Return ``{feature: used_count}`` for the current calendar month."""
    now = datetime.utcnow()
    month_key = now.isoformat(timespec="seconds")[:7]
    result: dict[str, int] = {}
    async with _db_ctx() as conn:
        cur = await conn.execute(
            "SELECT feature, used, reset_at FROM user_feature_quotas WHERE owner_id=?",
            (int(owner_id),),
        )
        for row in await cur.fetchall():
            feat = str(row[0])
            used = int(row[1] or 0)
            reset_at = str(row[2] or "")
            result[feat] = used if reset_at == month_key else 0
    return result


# ---------- draft limits by tier ----------

DRAFT_LIMITS_BY_TIER: dict[str, int] = {
    TIER_FREE: 5,
    TIER_PRO: 15,
    TIER_MAX: 50,
}


async def get_draft_limit(owner_id: int) -> int:
    """Return the maximum number of saved drafts allowed for this user's tier."""
    sub = await get_user_subscription(owner_id)
    tier = sub.get("subscription_tier", TIER_FREE)
    return DRAFT_LIMITS_BY_TIER.get(tier, DRAFT_LIMITS_BY_TIER[TIER_FREE])


# ---------- channel limits by tier ----------

CHANNEL_LIMITS_BY_TIER: dict[str, int] = {
    TIER_FREE: 1,
    TIER_PRO: 3,
    TIER_MAX: 10,
}


async def get_channel_limit(owner_id: int) -> int:
    """Return the maximum number of channels allowed for this user's tier."""
    sub = await get_user_subscription(owner_id)
    tier = sub.get("subscription_tier", TIER_FREE)
    return CHANNEL_LIMITS_BY_TIER.get(tier, CHANNEL_LIMITS_BY_TIER[TIER_FREE])


async def check_and_downgrade_expired_subscription(owner_id: int) -> dict:
    """Lazy per-user check: if subscription_expires_at is in the past, downgrade to free.

    Returns the current (possibly downgraded) subscription dict.
    """
    sub = await get_user_subscription(owner_id)
    tier = sub.get("subscription_tier", TIER_FREE)
    if tier == TIER_FREE:
        return sub
    expires_at = sub.get("subscription_expires_at") or ""
    if not expires_at:
        return sub
    try:
        expires_dt = datetime.fromisoformat(expires_at)
    except Exception:
        return sub
    if expires_dt < datetime.utcnow():
        await set_user_subscription(owner_id, TIER_FREE, expires_at="")
        return await get_user_subscription(owner_id)
    return sub


# ---------- admin helpers ----------

async def get_admin_stats() -> dict:
    """Return global stats for the admin panel."""
    async with _db_ctx() as db:
        cur = await db.execute("SELECT COUNT(*) FROM user_subscriptions")
        row = await cur.fetchone()
        total = int(row[0] or 0) if row else 0

        cur = await db.execute(
            "SELECT subscription_tier, COUNT(*) as cnt FROM user_subscriptions GROUP BY subscription_tier"
        )
        rows = await cur.fetchall()
        by_tier: dict[str, int] = {}
        for r in rows:
            by_tier[str(r[0] or TIER_FREE)] = int(r[1] or 0)

    return {
        "total_users": total,
        "free": by_tier.get(TIER_FREE, 0),
        "pro": by_tier.get(TIER_PRO, 0),
        "max": by_tier.get(TIER_MAX, 0),
    }


async def get_all_user_ids() -> list[int]:
    """Return the list of all owner_ids that have interacted with the bot."""
    async with _db_ctx() as db:
        cur = await db.execute("SELECT owner_id FROM user_subscriptions")
        rows = await cur.fetchall()
    return [int(r[0]) for r in rows if r[0]]


async def expire_overdue_subscriptions() -> int:
    """Downgrade paid users to free tier when subscription_expires_at is in the past.

    Returns the number of users downgraded.
    """
    now_iso = datetime.utcnow().isoformat(timespec="seconds")
    async with _db_ctx() as db:
        cur = await db.execute(
            "UPDATE user_subscriptions SET subscription_tier=?, auto_renew=0, updated_at=? "
            "WHERE subscription_tier != ? "
            "AND subscription_expires_at != '' "
            "AND subscription_expires_at < ?",
            (TIER_FREE, now_iso, TIER_FREE, now_iso),
        )
        await db.commit()
        return cur.rowcount or 0


# ---------- payment events ----------

async def record_payment_event(
    payment_id: str,
    owner_id: int,
    tier: str,
    method: str,
    amount: str,
    currency: str,
    payload: str = "",
) -> bool:
    """Record a payment event. Returns True if inserted, False if duplicate (idempotent).

    Uses payment_id as a unique key — calling twice with the same payment_id is a no-op.
    """
    import sqlite3
    now_iso = datetime.utcnow().isoformat(timespec="seconds")
    try:
        async with _db_ctx() as db:
            await db.execute(
                "INSERT INTO payment_events"
                "(payment_id, owner_id, tier, method, amount, currency, status, payload, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, 'success', ?, ?)",
                (str(payment_id), int(owner_id), str(tier), str(method),
                 str(amount), str(currency), str(payload), now_iso),
            )
            await db.commit()
        return True
    except sqlite3.IntegrityError:
        # UNIQUE constraint violation → duplicate payment_id → already processed
        return False


async def is_payment_processed(payment_id: str) -> bool:
    """Check if a payment with this ID has already been recorded."""
    if not payment_id:
        return False
    async with _db_ctx() as db:
        cur = await db.execute(
            "SELECT 1 FROM payment_events WHERE payment_id=? LIMIT 1",
            (str(payment_id),),
        )
        row = await cur.fetchone()
    return row is not None


# ---------------------------------------------------------------------------
# Durable scheduler dedup helpers
# ---------------------------------------------------------------------------

async def check_scheduler_dedup(dedup_key: str) -> bool:
    """Return True if *dedup_key* has already been processed (should skip)."""
    async with _db_ctx() as conn:
        cur = await conn.execute(
            "SELECT 1 FROM scheduler_dedup WHERE dedup_key=? LIMIT 1",
            (dedup_key,),
        )
        return (await cur.fetchone()) is not None


async def set_scheduler_dedup(dedup_key: str, trigger_type: str = "", owner_id: int = 0) -> bool:
    """Mark *dedup_key* as processed.  Returns True on success, False if already exists."""
    now = datetime.utcnow().isoformat(timespec="seconds")
    try:
        async with _db_ctx() as conn:
            await conn.execute(
                "INSERT OR IGNORE INTO scheduler_dedup(dedup_key, trigger_type, owner_id, created_at) VALUES(?,?,?,?)",
                (dedup_key, trigger_type, int(owner_id), now),
            )
            await conn.commit()
            return True
    except Exception:
        return False


async def cleanup_scheduler_dedup(max_age_hours: int = 48) -> int:
    """Remove dedup entries older than *max_age_hours*. Returns count deleted."""
    cutoff = (datetime.utcnow() - timedelta(hours=max_age_hours)).isoformat(timespec="seconds")
    async with _db_ctx() as conn:
        cur = await conn.execute("DELETE FROM scheduler_dedup WHERE created_at < ? AND created_at != ''", (cutoff,))
        await conn.commit()
        return cur.rowcount or 0
