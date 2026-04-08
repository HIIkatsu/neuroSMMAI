from __future__ import annotations

import os
import re
from typing import Any

import db
from config import load_config
from miniapp_analytics_service import build_channel_analytics, recent_channel_history
from miniapp_media_service import build_media_shortcuts
from miniapp_settings_service import build_operator_profile, normalize_settings_snapshot
from miniapp_shared import cache_get, cache_set, owner_settings


def enrich_display_label(channel: dict[str, Any] | None) -> dict[str, Any] | None:
    """Add ``display_label`` to a channel dict.

    Prefers the human-readable ``title``; falls back to ``channel_target``
    (e.g. ``@mychannel``).  Raw numeric Telegram chat IDs are never used as
    the label — ``Канал без названия`` is shown instead.
    """
    if not channel:
        return channel
    title = str(channel.get("title") or "").strip()
    target = str(channel.get("channel_target") or "").strip()
    if title and not re.match(r'^-?\d+$', title):
        label = title
    elif target and not re.match(r'^-?\d+$', target):
        label = target
    else:
        label = "Канал без названия"
    channel["display_label"] = label
    return channel


def _active_drafts(rows: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    return [d for d in (rows or []) if str((d or {}).get('status') or 'draft').strip().lower() == 'draft']


def _build_analytics_compat(
    stats: dict[str, Any] | None,
    history: dict[str, list[str]] | None,
    analytics_snapshot: dict[str, Any] | None,
    *,
    settings: dict[str, Any] | None,
    active_channel: dict[str, Any] | None,
    channels: list[dict[str, Any]] | None,
    drafts: list[dict[str, Any]] | None,
    plan_items: list[dict[str, Any]] | None,
    media_items: list[dict[str, Any]] | None,
    schedules: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    try:
        return build_channel_analytics(
            stats,
            history,
            analytics_snapshot,
            settings=settings,
            active_channel=active_channel,
            channels=channels,
            drafts=drafts,
            plan_items=plan_items,
            media_items=media_items,
            schedules=schedules,
        )
    except TypeError:
        # Backward compatibility with older miniapp_analytics_service signatures.
        return build_channel_analytics(stats, history, analytics_snapshot)


async def bootstrap_core_payload(telegram_user_id: int) -> dict[str, Any]:
    active = await db.get_active_channel_profile(owner_id=telegram_user_id)
    channels = await db.list_channel_profiles(owner_id=telegram_user_id)
    # Enrich channels with display_label at the source
    enrich_display_label(active)
    for ch in channels:
        enrich_display_label(ch)
    stats = await db.get_post_stats(owner_id=telegram_user_id)
    drafts = _active_drafts(await db.list_drafts(owner_id=telegram_user_id, limit=80))
    drafts_current = len(drafts)
    plan_items = await db.list_plan_items(owner_id=telegram_user_id, limit=300)
    schedules = await db.list_schedules(owner_id=telegram_user_id)
    media_inbox = await db.list_user_media(owner_id=telegram_user_id, limit=24)
    history = await recent_channel_history(telegram_user_id)
    shortcuts = await build_media_shortcuts(telegram_user_id)
    cfg = load_config()
    analytics_snapshot = await db.get_post_analytics_snapshot(owner_id=telegram_user_id)
    settings = await owner_settings(telegram_user_id, active)
    analytics = _build_analytics_compat(
        stats,
        history,
        analytics_snapshot,
        settings=settings,
        active_channel=active,
        channels=channels,
        drafts=drafts,
        plan_items=plan_items,
        media_items=media_inbox,
        schedules=schedules,
    )
    return {
        "telegram_user_id": telegram_user_id,
        "active_channel": active,
        "channels": channels,
        "drafts_current": drafts_current,
        "stats": stats,
        "analytics": analytics,
        "settings": settings,
        "operator_profile": build_operator_profile(
            settings=settings,
            active_channel=active,
            channels=channels,
            drafts_current=drafts_current,
            plan_items=plan_items,
            schedules=schedules,
            stats=stats,
            analytics=analytics,
        ),
        **shortcuts,
        "bot_username": getattr(cfg, "bot_username", None) or os.getenv("BOT_USERNAME", "").strip(),
        "limits": {
            "upload_image_limit_mb": cfg.upload_image_limit_mb,
            "upload_video_limit_mb": cfg.upload_video_limit_mb,
            "upload_document_limit_mb": cfg.upload_document_limit_mb,
            "temp_media_quota_mb_per_user": cfg.temp_media_quota_mb_per_user,
            "drafts_max": await db.get_draft_limit(telegram_user_id),
            "channels_max": await db.get_channel_limit(telegram_user_id),
        },
        "subscription": await db.get_user_subscription(telegram_user_id),
    }


async def owner_summary(telegram_user_id: int) -> dict[str, Any]:
    cached = cache_get(telegram_user_id, 'bootstrap')
    if cached:
        return cached
    payload = await db.get_owner_bootstrap_snapshot(telegram_user_id, drafts_limit=80, plan_limit=300, media_limit=24)
    payload["telegram_user_id"] = telegram_user_id
    # Enrich channels with display_label
    enrich_display_label(payload.get("active_channel"))
    for ch in (payload.get("channels") or []):
        enrich_display_label(ch)
    payload.update(await build_media_shortcuts(telegram_user_id))
    payload["settings"] = normalize_settings_snapshot(payload.get("settings") or {}, payload.get("active_channel"))
    history = await recent_channel_history(telegram_user_id)
    drafts = _active_drafts(payload.get("drafts") or [])
    analytics = _build_analytics_compat(
        payload.get("stats") or {},
        history,
        await db.get_post_analytics_snapshot(owner_id=telegram_user_id),
        settings=payload.get("settings") or {},
        active_channel=payload.get("active_channel"),
        channels=payload.get("channels") or [],
        drafts=drafts,
        plan_items=payload.get("plan") or [],
        media_items=payload.get("media_inbox") or [],
        schedules=payload.get("schedules") or [],
    )
    payload["analytics"] = analytics
    payload["operator_profile"] = build_operator_profile(
        settings=payload.get("settings") or {},
        active_channel=payload.get("active_channel"),
        channels=payload.get("channels") or [],
        drafts_current=len(drafts),
        plan_items=payload.get("plan") or [],
        schedules=payload.get("schedules") or [],
        stats=payload.get("stats") or {},
        analytics=analytics,
    )
    cache_set(telegram_user_id, 'bootstrap', payload)
    return payload
