from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from aiogram import Bot
from aiogram.client.default import DefaultBotProperties
from fastapi import Depends, HTTPException

import db
from auth import get_current_telegram_user
from config import load_config
from miniapp_settings_service import normalize_settings_snapshot

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
GENERATED_DIR = BASE_DIR / "generated_images"
GENERATED_DIR.mkdir(exist_ok=True)

_AI_POST_COOLDOWN_SECONDS = 12
_AI_POST_LAST_CALLS: dict[int, float] = {}

_BOOTSTRAP_TTL_SECONDS = 2.0
_BOOTSTRAP_CACHE: dict[tuple[int, str], tuple[float, dict[str, Any]]] = {}


def clean_text(v: str | None) -> str:
    return (v or "").strip()


def cache_get(owner_id: int, key: str) -> dict[str, Any] | None:
    item = _BOOTSTRAP_CACHE.get((int(owner_id), key))
    if not item:
        return None
    ts, payload = item
    if time.monotonic() - ts > _BOOTSTRAP_TTL_SECONDS:
        _BOOTSTRAP_CACHE.pop((int(owner_id), key), None)
        return None
    return dict(payload)


def cache_set(owner_id: int, key: str, payload: dict[str, Any]) -> dict[str, Any]:
    _BOOTSTRAP_CACHE[(int(owner_id), key)] = (time.monotonic(), dict(payload))
    return payload


def cache_invalidate(owner_id: int, *keys: str) -> None:
    owner = int(owner_id)
    if not keys:
        keys = ('bootstrap', 'core', 'channels', 'drafts', 'plan', 'schedules', 'stats', 'settings', 'media_inbox')
    for key in keys:
        _BOOTSTRAP_CACHE.pop((owner, key), None)


def drafts_limit_value() -> int:
    """Return the global fallback draft limit (from config). Use tier-based limit when possible."""
    cfg = load_config()
    return max(1, int(getattr(cfg, "max_active_drafts_per_user", 15) or 15))


async def drafts_limit_state(owner_id: int) -> dict[str, int | bool]:
    current = int(await db.count_drafts(owner_id=owner_id, status="draft") or 0)
    limit_max = await db.get_draft_limit(owner_id)
    return {"current": current, "max": limit_max, "reached": current >= limit_max}


async def ensure_drafts_capacity(owner_id: int) -> None:
    limit = await drafts_limit_state(owner_id)
    if limit["reached"]:
        sub = await db.get_user_subscription(owner_id)
        raise HTTPException(
            status_code=403,
            detail={
                "code": "draft_limit_reached",
                "message": (
                    f"Достигнут лимит черновиков ({limit['current']} / {limit['max']}) для вашего тарифа. "
                    "Удали часть черновиков или перейди на более высокий тариф."
                ),
                "subscription": sub,
                "drafts_current": limit["current"],
                "drafts_max": limit["max"],
            },
        )


async def create_bot() -> Bot:
    config = load_config()
    return Bot(token=config.bot_token, default=DefaultBotProperties(parse_mode=None))


async def current_user_id(
    telegram_user: dict[str, Any] = Depends(get_current_telegram_user),
) -> int:
    user_id = telegram_user.get("id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Не удалось определить пользователя Telegram")
    owner_id = int(user_id)
    # Lazy subscription expiry check: downgrade to free if subscription has expired
    await db.check_and_downgrade_expired_subscription(owner_id)
    return owner_id


async def owned_channel_targets(owner_id: int) -> set[str]:
    """Return the set of channel_target values owned by *owner_id*."""
    profiles = await db.list_channel_profiles(owner_id=owner_id)
    return {str(p.get("channel_target") or "") for p in profiles if p.get("channel_target")}


async def verify_channel_ownership(owner_id: int, channel_target: str) -> None:
    """Raise 403 if *channel_target* does not belong to *owner_id*."""
    if not channel_target:
        return  # empty target is ok (will use active channel)
    owned = await owned_channel_targets(owner_id)
    if str(channel_target) not in owned:
        raise HTTPException(status_code=403, detail="Нельзя работать с чужим каналом")


async def owner_settings(telegram_user_id: int, active: dict[str, Any] | None = None) -> dict[str, str]:
    # Get channel-specific settings from channel profile with owner fallback
    channel_settings = await db.get_channel_settings(telegram_user_id)

    # Also get owner-level-only settings (not stored in channel profile)
    owner_only = await db.get_settings_bulk([
        "auto_hashtags",
    ], owner_id=telegram_user_id)

    # Merge: channel settings take priority, then add owner-only settings
    merged = dict(channel_settings)
    for k, v in owner_only.items():
        if v is not None and k not in merged:
            merged[k] = v

    return normalize_settings_snapshot(merged, active)
