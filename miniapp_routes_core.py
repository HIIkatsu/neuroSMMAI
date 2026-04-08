from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any
from pydantic import BaseModel

from fastapi import APIRouter, Depends, HTTPException

import db
from ai_client import ai_chat
from auth import get_current_telegram_user
from channel_security import ChannelAccessError, verify_channel_access
from config import load_config
from miniapp_analytics_service import build_channel_analytics, recent_channel_history
from miniapp_bootstrap_service import bootstrap_core_payload, owner_summary
from miniapp_media_service import build_media_shortcuts, serialize_media_item
from miniapp_plan_service import days_label, strip_generated_labels
from miniapp_schemas import AIAnalyticsRequest, ChannelActivate, ChannelCreate, SettingsUpdate
from miniapp_settings_service import normalize_settings_update
from miniapp_shared import cache_get, cache_invalidate, cache_set, clean_text, create_bot, current_user_id, drafts_limit_value, owner_settings, verify_channel_ownership
from channel_profile_resolver import normalize_topic_to_family, resolve_channel_policy, build_family_rules_block

router = APIRouter(tags=["core"])
logger = logging.getLogger(__name__)


class AssistantChatRequest(BaseModel):
    question: str = ""
    session_history: list[dict] = []


def _build_channel_analytics_safe(*args, **kwargs):
    try:
        return build_channel_analytics(*args, **kwargs)
    except TypeError:
        stats = args[0] if len(args) > 0 else kwargs.get('stats')
        history = args[1] if len(args) > 1 else kwargs.get('history')
        snapshot = args[2] if len(args) > 2 else kwargs.get('analytics_snapshot')
        return build_channel_analytics(stats, history, snapshot)


async def _require_pro_tier(owner_id: int, feature: str = "функция") -> None:
    """Raise HTTP 402 if the user is not on a paid tier (pro/max)."""
    from datetime import datetime
    sub = await db.get_user_subscription(owner_id)
    tier = sub.get("subscription_tier", db.TIER_FREE)
    trial_ends = sub.get("trial_ends_at") or ""
    if trial_ends:
        try:
            if datetime.fromisoformat(trial_ends) > datetime.utcnow():
                return
        except Exception:
            pass
    if tier in (db.TIER_PRO, db.TIER_MAX):
        return
    raise HTTPException(
        status_code=402,
        detail={
            "code": "upgrade_required",
            "message": f"Функция «{feature}» доступна в тарифе PRO. Узнай подробнее о тарифах.",
            "subscription": sub,
        },
    )


async def _require_generation_quota(owner_id: int) -> None:
    """Raise HTTP 402 if the free-tier user has exhausted their monthly generation quota."""
    allowed, sub = await db.is_generation_allowed(owner_id)
    if not allowed:
        raise HTTPException(
            status_code=402,
            detail={
                "code": "limit_reached",
                "message": (
                    f"Бесплатный лимит исчерпан ({db.FREE_TIER_GENERATIONS_LIMIT} генерации/мес). "
                    "Перейди на тариф PRO для безлимитных генераций."
                ),
                "subscription": sub,
            },
        )


async def _require_feature_quota(owner_id: int, feature: str, label: str = "") -> None:
    """Raise HTTP 402 if the user has exhausted their per-feature monthly quota."""
    allowed, sub, used, limit = await db.is_feature_allowed(owner_id, feature)
    if not allowed:
        feat_label = label or feature
        raise HTTPException(
            status_code=402,
            detail={
                "code": "limit_reached",
                "feature": feature,
                "message": (
                    f"Лимит «{feat_label}» исчерпан ({used}/{limit} в этом месяце). "
                    "Перейди на тариф PRO для безлимитного доступа."
                ),
                "subscription": sub,
                "used": used,
                "limit": limit,
            },
        )



@router.get("/api/bootstrap/core")
async def bootstrap_core(telegram_user_id: int = Depends(current_user_id)):
    cached = cache_get(telegram_user_id, 'core')
    if cached:
        cached["user"] = {"id": telegram_user_id}
        return cached
    payload = await bootstrap_core_payload(telegram_user_id)
    payload["user"] = {"id": telegram_user_id}
    return cache_set(telegram_user_id, 'core', payload)


@router.get("/api/channels")
async def get_channels(telegram_user_id: int = Depends(current_user_id)):
    cached = cache_get(telegram_user_id, 'channels')
    if cached:
        return cached
    channels = await db.list_channel_profiles(owner_id=telegram_user_id)
    active = await db.get_active_channel_profile(owner_id=telegram_user_id)
    max_channels = await db.get_channel_limit(telegram_user_id)
    return cache_set(telegram_user_id, 'channels', {
        "channels": channels,
        "active_channel": active,
        "max_channels": max_channels,
    })


@router.get("/api/drafts")
async def get_drafts(telegram_user_id: int = Depends(current_user_id)):
    cached = cache_get(telegram_user_id, 'drafts')
    if cached:
        return cached
    drafts = await db.list_drafts(owner_id=telegram_user_id, limit=50)
    drafts_current = len([d for d in drafts if str((d or {}).get('status') or 'draft') == 'draft'])
    return cache_set(telegram_user_id, 'drafts', {"drafts": drafts, "drafts_current": drafts_current})


@router.get("/api/plan")
async def get_plan(telegram_user_id: int = Depends(current_user_id)):
    cached = cache_get(telegram_user_id, 'plan')
    if cached:
        return cached
    plan = await db.list_plan_items(owner_id=telegram_user_id, limit=300)
    return cache_set(telegram_user_id, 'plan', {"plan": plan})


@router.get("/api/schedules")
async def get_schedules(telegram_user_id: int = Depends(current_user_id)):
    cached = cache_get(telegram_user_id, 'schedules')
    if cached:
        return cached
    schedules = await db.list_schedules(owner_id=telegram_user_id)
    for row in schedules:
        row["days_label"] = days_label(row.get("days", "*"))
    return cache_set(telegram_user_id, 'schedules', {"schedules": schedules})


@router.get("/api/media/inbox")
async def get_media_inbox(telegram_user_id: int = Depends(current_user_id)):
    cached = cache_get(telegram_user_id, 'media_inbox')
    if cached:
        return cached
    media_inbox = [serialize_media_item(item) for item in await db.list_user_media(owner_id=telegram_user_id, limit=24)]
    return cache_set(telegram_user_id, 'media_inbox', {"media_inbox": media_inbox})


@router.get("/api/media/latest")
async def get_media_latest(telegram_user_id: int = Depends(current_user_id)):
    return await build_media_shortcuts(telegram_user_id)


@router.get("/api/settings")
async def get_settings_view(telegram_user_id: int = Depends(current_user_id)):
    cached = cache_get(telegram_user_id, 'settings')
    if cached:
        return cached
    active = await db.get_active_channel_profile(owner_id=telegram_user_id)
    settings = await owner_settings(telegram_user_id, active)
    return cache_set(telegram_user_id, 'settings', {"settings": settings})


@router.post("/api/analytics/ai-insights")
async def ai_analytics_insights(
    data: AIAnalyticsRequest,
    telegram_user_id: int = Depends(current_user_id),
):
    await _require_feature_quota(telegram_user_id, "ai_insights", "AI-аналитика")
    stats = await db.get_post_stats(owner_id=telegram_user_id)
    history = await recent_channel_history(telegram_user_id)
    analytics_snapshot = await db.get_post_analytics_snapshot(owner_id=telegram_user_id)
    active = await db.get_active_channel_profile(owner_id=telegram_user_id)
    settings = await owner_settings(telegram_user_id, active)
    analytics = _build_channel_analytics_safe(
        stats,
        history,
        analytics_snapshot,
        settings=settings,
        active_channel=active,
        channels=await db.list_channel_profiles(owner_id=telegram_user_id),
        drafts=[d for d in await db.list_drafts(owner_id=telegram_user_id, limit=100) if str((d or {}).get('status') or 'draft') == 'draft'],
        plan_items=await db.list_plan_items(owner_id=telegram_user_id, limit=300),
        media_items=await db.list_user_media(owner_id=telegram_user_id, limit=80),
        schedules=await db.list_schedules(owner_id=telegram_user_id),
    )
    top_topics = list(analytics_snapshot.get('top_topics') or [])[:5]
    focus = clean_text(data.focus) or 'общий разбор'
    config = load_config()
    if not getattr(config, 'openrouter_api_key', None):
        return {
            'ok': True,
            'text': (
                f"Текущая готовность канала: {analytics.get('score')}%. "
                f"Слабое место сейчас — {analytics.get('next_step')}. "
                + (f"Лидирующие темы по фактам: {', '.join(str(x.get('label') or x.get('topic_signature') or '') for x in top_topics[:3])}. " if top_topics else '')
                + ("В базе пока нет накопленных просмотров и реакций, поэтому выводы делаются по фактам из плана, черновиков и ритма публикаций." if not analytics_snapshot.get('views_known') else "Разбор уже учитывает просмотры, реакции, комментарии и пересылки там, где они есть в базе.")
            ),
            'analytics': analytics,
        }

    prompt = (
        "Ты сильный ИИ-аналитик Telegram-канала. Не пересказывай обычный блок аналитики и не дублируй формулировки прогресс-баров. "
        "Твоя задача — сделать выводы второго уровня: заметить закономерности, узкие места, сильные и слабые связки между темами, ритмом и результатом. "
        "Не выдумывай цифры, просмотры и вовлечение, если их нет в данных. Опирайся только на факты ниже. "
        f"Фокус пользователя: {focus}. "
        f"Готовность канала: {analytics.get('score')}%. "
        f"Сигналы: {json.dumps(analytics.get('signals') or [], ensure_ascii=False)}. "
        f"Сводка: {json.dumps(analytics.get('summary') or {}, ensure_ascii=False)}. "
        f"Топ тем: {json.dumps(top_topics, ensure_ascii=False)}. "
        f"Недоступные блоки: {json.dumps(analytics.get('unavailable') or [], ensure_ascii=False)}. "
        "Верни компактный, но умный разбор в 4 блоках: 1) главный вывод по каналу; 2) что реально тянет результаты вниз и почему; 3) что стоит масштабировать из того, что уже работает; 4) план на 7 дней: три конкретных шага и один A/B тест. "
        "Не повторяй дословно фразы из блока аналитики."
    )
    raw = await ai_chat(
        api_key=config.openrouter_api_key,
        model=config.openrouter_model,
        base_url=getattr(config, 'openrouter_base_url', None),
        temperature=0.35,
        messages=[{'role': 'user', 'content': prompt}],
    )
    await db.increment_feature_used(telegram_user_id, "ai_insights")
    return {'ok': True, 'text': strip_generated_labels(raw), 'analytics': analytics}


@router.post("/api/assistant/chat")
async def assistant_chat(
    data: AssistantChatRequest,
    telegram_user_id: int = Depends(current_user_id),
):
    await _require_feature_quota(telegram_user_id, "assistant", "Ассистент")
    question = clean_text(data.question or '')
    if not question:
        raise HTTPException(status_code=400, detail='Вопрос пустой')

    # Fire all owner-scoped reads concurrently instead of sequentially.
    # All queries are scoped to the same telegram_user_id so there is no
    # cross-owner data mixing.  `active` (channel profile) is needed by
    # owner_settings, so that one runs after the gather.
    (
        stats,
        history,
        analytics_snapshot,
        active,
        channels,
        drafts_all,
        plan_items,
        media_items,
        schedules,
    ) = await asyncio.gather(
        db.get_post_stats(owner_id=telegram_user_id),
        recent_channel_history(telegram_user_id),
        db.get_post_analytics_snapshot(owner_id=telegram_user_id),
        db.get_active_channel_profile(owner_id=telegram_user_id),
        db.list_channel_profiles(owner_id=telegram_user_id),
        db.list_drafts(owner_id=telegram_user_id, limit=100),
        db.list_plan_items(owner_id=telegram_user_id, limit=300),
        db.list_user_media(owner_id=telegram_user_id, limit=80),
        db.list_schedules(owner_id=telegram_user_id),
    )
    drafts = [d for d in drafts_all if str((d or {}).get('status') or 'draft') == 'draft']
    settings = await owner_settings(telegram_user_id, active)
    analytics = _build_channel_analytics_safe(
        stats,
        history,
        analytics_snapshot,
        settings=settings,
        active_channel=active,
        channels=channels,
        drafts=drafts,
        plan_items=plan_items,
        media_items=media_items,
        schedules=schedules,
    )

    # Resolve family-aware channel policy for assistant context
    family_ctx = ""
    try:
        policy = await resolve_channel_policy(telegram_user_id, profile=active)
        if policy and policy.topic_family and policy.topic_family != "generic":
            family_rules = build_family_rules_block(policy)
            family_ctx = (
                f"Ниша канала (автоопределение): {policy.topic_family}"
                + (f" / {policy.topic_subfamily}" if policy.topic_subfamily else "")
                + ". "
            )
            if family_rules:
                family_ctx += f"Правила ниши: {family_rules}. "
    except Exception:
        pass

    summary = analytics.get('summary') or {}
    signals = analytics.get('signals') or []
    weakest = sorted(signals, key=lambda x: int(x.get('value') or 0))[0] if signals else None
    config = load_config()

    if not getattr(config, 'openrouter_api_key', None):
        parts = [f"Готовность канала сейчас {analytics.get('score', 0)}%."]
        if weakest:
            parts.append(f"Слабее всего сейчас «{weakest.get('label', 'сигнал')}» — {weakest.get('value', 0)}%.")
            action = str(weakest.get('action') or '').strip()
            if action:
                parts.append(f"Что сделать: {action}.")
        parts.append(f"В запасе {int(summary.get('drafts_count') or 0)} черновиков, {int(summary.get('plan_count') or 0)} идей и {int(summary.get('media_count') or 0)} медиа.")
        return {'ok': True, 'text': ' '.join(parts), 'analytics': analytics}

    prompt = (
        "Ты — встроенный ИИ-помощник в Mini App NeuroSMM. "
        "Твоя задача — помогать пользователю управлять Telegram-каналом: отвечать на вопросы, давать конкретные рекомендации и направлять к нужным действиям.\n\n"
        "ПРАВИЛА ОТВЕТА:\n"
        "1. Отвечай ТОЛЬКО на основе реальных данных пользователя ниже. Не выдумывай.\n"
        "2. Если данных не хватает — скажи, чего именно не хватает, и что нужно сделать.\n"
        "3. Сначала — прямой ответ на 1-2 предложения. Потом пустая строка.\n"
        "4. Если уместно — блок «Что сделать:» с 2-3 пунктами (не больше 4).\n"
        "5. Если вопрос простой (что такое X, как работает Y) — отвечай коротко, без блока действий.\n"
        "6. Если пользователь ошибся — объясни ошибку, предложи точечное исправление.\n"
        "7. Не повторяй одни и те же советы. Давай разнообразные рекомендации.\n"
        "8. Не используй markdown-заголовки, таблицы, дисклеймеры.\n"
        "9. Каждую мысль — на новой строке. Текст должен легко читаться в чате.\n"
        "10. Будь конкретен: вместо «улучшите контент» → «добавь CTA в конце поста».\n\n"
        "НАВИГАЦИЯ ПО РАЗДЕЛАМ:\n"
        "- Черновики и посты → вкладка «Посты»\n"
        "- Идеи и план → вкладка «План»\n"
        "- Расписание и автопостинг → вкладка «Автопост»\n"
        "- Каналы и подключение → вкладка «Каналы»\n"
        "- Профиль канала и стиль → кнопка «Профиль» на главной\n"
        "- Аналитика → кнопка на панели готовности\n\n"
        f"{family_ctx}"
        f"Вопрос: {question}\n\n"
        f"Активный канал: {json.dumps(active or {{}}, ensure_ascii=False)}\n"
        f"Настройки: {json.dumps(settings or {{}}, ensure_ascii=False)}\n"
        f"Статистика: {json.dumps(stats or {{}}, ensure_ascii=False)}\n"
        f"Аналитика: {json.dumps(analytics or {{}}, ensure_ascii=False)}\n"
        f"Черновики: {len(drafts)}. План: {len([x for x in plan_items if int(x.get('posted') or 0) != 1])}. Медиа: {len(media_items)}. Слоты: {len([x for x in schedules if int(x.get('enabled', 1) or 0) != 0])}.\n"
        f"История канала: {json.dumps(history or {{}}, ensure_ascii=False)}"
    )

    ctx_items = [
        m for m in (data.session_history or [])[-5:]
        if isinstance(m, dict) and m.get('role') in ('user', 'ai') and str(m.get('text') or '').strip()
    ]
    messages: list[dict] = []
    for m in ctx_items:
        role = 'user' if m['role'] == 'user' else 'assistant'
        messages.append({'role': role, 'content': str(m['text']).strip()[:600]})
    messages.append({'role': 'user', 'content': prompt})

    raw = await ai_chat(
        api_key=config.openrouter_api_key,
        model=config.openrouter_model,
        base_url=getattr(config, 'openrouter_base_url', None),
        temperature=0.35,
        max_tokens=500,
        messages=messages,
    )
    text = strip_generated_labels(raw or '').strip()
    if not text:
        fallback = [f"Готовность канала {analytics.get('score', 0)}%."]
        if weakest:
            fallback.append(f"Слабое место: {weakest.get('label', 'сигнал')} ({weakest.get('value', 0)}%).")
            if weakest.get('action'):
                fallback.append(f"Что сделать: {weakest.get('action')}.")
        return {'ok': True, 'text': ' '.join(fallback), 'analytics': analytics}
    await db.increment_feature_used(telegram_user_id, "assistant")
    return {'ok': True, 'text': text, 'analytics': analytics}


@router.get("/api/bootstrap")
async def bootstrap(
    telegram_user_id: int = Depends(current_user_id),
    telegram_user: dict[str, Any] = Depends(get_current_telegram_user),
):
    payload = await owner_summary(telegram_user_id)
    raw_user = telegram_user.get("raw_user") or {}

    payload["user"] = {
        "id": telegram_user_id,
        "username": raw_user.get("username"),
        "first_name": raw_user.get("first_name"),
        "last_name": raw_user.get("last_name"),
    }
    payload["meta"] = {
        "days_options": [7, 14, 30, 60],
        "posting_modes": ["both", "posts", "news"],
        "posting_mode_labels": {"both": "Посты и новости", "posts": "Только посты", "news": "Только новости"},
    }
    payload["bot_username"] = getattr(load_config(), "bot_username", None) or os.getenv("BOT_USERNAME", "").strip()
    payload.update(await build_media_shortcuts(telegram_user_id))
    cfg = load_config()
    from billing_service import RUB_PRICES
    payload["limits"] = {
        "drafts_max": await db.get_draft_limit(telegram_user_id),
        "drafts_current": payload.get("drafts_current", 0),
        "channels_max": await db.get_channel_limit(telegram_user_id),
        "generations_limit_free": db.FREE_TIER_GENERATIONS_LIMIT,
        "feature_limits_free": db.FREE_TIER_FEATURE_LIMITS,
        "feature_usage": await db.get_all_feature_usage(telegram_user_id),
        "upload_image_limit_mb": cfg.upload_image_limit_mb,
        "upload_video_limit_mb": cfg.upload_video_limit_mb,
        "upload_document_limit_mb": cfg.upload_document_limit_mb,
        "temp_media_quota_mb_per_user": cfg.temp_media_quota_mb_per_user,
    }
    payload["subscription"] = await db.get_user_subscription(telegram_user_id)
    payload["tariffs"] = {
        "pro_rub": RUB_PRICES.get(db.TIER_PRO, 490),
        "max_rub": RUB_PRICES.get(db.TIER_MAX, 990),
        "pro_stars": cfg.stars_pro_price,
        "max_stars": cfg.stars_max_price,
        "draft_limits": db.DRAFT_LIMITS_BY_TIER,
        "channel_limits": db.CHANNEL_LIMITS_BY_TIER,
        "feature_limits_free": db.FREE_TIER_FEATURE_LIMITS,
    }
    return payload


@router.post("/api/news/sniper/run")
async def news_sniper_run(telegram_user_id: int = Depends(current_user_id)):
    """Manually trigger one News Sniper cycle for the current user (Max tier only)."""
    sub = await db.get_user_subscription(telegram_user_id)
    tier = sub.get("subscription_tier", db.TIER_FREE)
    if tier != db.TIER_MAX:
        raise HTTPException(
            status_code=402,
            detail={
                "code": "upgrade_required",
                "message": "Ручной запуск News Sniper доступен только на тарифе Max.",
                "subscription": sub,
            },
        )

    from news_service import fetch_news_candidates, build_news_post, build_news_source_meta, is_source_confident
    from content import _remove_fabricated_refs
    from miniapp_shared import ensure_drafts_capacity

    cfg = load_config()
    if not getattr(cfg, "openrouter_api_key", ""):
        raise HTTPException(status_code=503, detail="AI ключ не настроен")

    candidates = await fetch_news_candidates(owner_id=telegram_user_id, limit=3)
    if not candidates:
        raise HTTPException(status_code=404, detail="Актуальных новостей по теме канала не найдено. Попробуй позже.")

    # Pick the first source-confident candidate
    item = None
    for candidate in candidates:
        if is_source_confident(candidate):
            item = candidate
            break
    if not item:
        raise HTTPException(status_code=404, detail="Не найдено новостей с подтверждённым источником. Попробуй позже.")

    title = item.get("title", "")
    topic = item.get("topic", "")

    text = await build_news_post(cfg, item, owner_id=telegram_user_id)
    if not text:
        raise HTTPException(status_code=502, detail="Не удалось сгенерировать текст. Попробуй ещё раз.")

    text, _, _ = _remove_fabricated_refs(text)

    # Build source metadata JSON
    news_source_json = build_news_source_meta(item)

    ch_settings = await db.get_channel_settings(telegram_user_id)
    channel_target = (ch_settings.get("channel_target") or "").strip()

    # Verify that channel_target actually belongs to this user
    if channel_target:
        await verify_channel_ownership(telegram_user_id, channel_target)

    await ensure_drafts_capacity(telegram_user_id)
    draft_id = await db.create_draft(
        owner_id=telegram_user_id,
        channel_target=channel_target,
        text=text,
        prompt=title,
        topic=topic,
        draft_source="news_sniper",
        news_source_json=news_source_json,
    )

    await db.log_news(item["link"], title, owner_id=telegram_user_id)

    cache_invalidate(telegram_user_id, 'drafts', 'core', 'bootstrap')
    draft = await db.get_draft(draft_id, owner_id=telegram_user_id)
    return {"ok": True, "draft_id": draft_id, "draft": draft, "title": title}


@router.post("/api/channels")
async def create_channel(
    data: ChannelCreate,
    telegram_user_id: int = Depends(current_user_id),
):
    requested_target = clean_text(data.channel_target)
    existing = await db.list_channel_profiles(owner_id=telegram_user_id)
    # Skip limit check if the channel already exists (re-upsert allowed)
    is_existing = any(str(x.get("channel_target") or "") == requested_target for x in existing)
    if not is_existing:
        sub = await db.get_user_subscription(telegram_user_id)
        tier = sub.get("subscription_tier", db.TIER_FREE)
        max_channels = db.CHANNEL_LIMITS_BY_TIER.get(tier, db.CHANNEL_LIMITS_BY_TIER[db.TIER_FREE])
        if len(existing) >= max_channels:
            raise HTTPException(
                status_code=403,
                detail={
                    "code": "channel_limit_reached",
                    "message": (
                        f"Достигнут лимит каналов ({len(existing)} / {max_channels}) для вашего тарифа. "
                        "Перейдите на более высокий тариф, чтобы добавить больше каналов."
                    ),
                    "subscription": sub,
                    "channels_current": len(existing),
                    "channels_max": max_channels,
                },
            )

    bot = await create_bot()
    try:
        try:
            access = await verify_channel_access(bot, telegram_user_id, requested_target)
        except ChannelAccessError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
    finally:
        await bot.session.close()

    title = clean_text(data.title) or access.title
    topic = clean_text(data.topic)

    # Auto-detect topic family for structured profile
    topic_family, topic_subfamily = normalize_topic_to_family(topic) if topic else ("", "")

    await db.upsert_channel_profile(
        owner_id=telegram_user_id,
        channel_target=access.channel_target,
        title=title,
        topic=topic,
        make_active=bool(data.make_active),
        topic_raw=topic,
        topic_family=topic_family,
        topic_subfamily=topic_subfamily,
    )

    if data.make_active:
        await db.set_setting("channel_target", access.channel_target, owner_id=telegram_user_id)
        if topic:
            await db.set_setting("topic", topic, owner_id=telegram_user_id)
        if hasattr(db, 'assign_empty_drafts_channel'):
            await db.assign_empty_drafts_channel(telegram_user_id, access.channel_target)

    cache_invalidate(telegram_user_id)
    return await owner_summary(telegram_user_id)


@router.post("/api/channels/activate")
async def activate_channel(
    data: ChannelActivate,
    telegram_user_id: int = Depends(current_user_id),
):
    profile = await db.set_active_channel_profile(data.profile_id, owner_id=telegram_user_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Канал не найден")
    if hasattr(db, 'assign_empty_drafts_channel'):
        await db.assign_empty_drafts_channel(telegram_user_id, profile.get('channel_target', ''))
    cache_invalidate(telegram_user_id)
    return await owner_summary(telegram_user_id)


@router.post("/api/channels/normalize-topic")
async def normalize_channel_topic(
    data: dict,
    telegram_user_id: int = Depends(current_user_id),
):
    """
    Normalizes a freeform topic description into a structured family/subfamily.
    Used during onboarding to propose a canonical topic family to the user.

    Request body: {"topic_raw": "..."}
    Response: {"family": "...", "subfamily": "...", "display_family": "...", "proposal_text": "..."}
    """
    from channel_profile_resolver import build_onboarding_normalization_message
    topic_raw = (data.get("topic_raw") or "").strip()
    if not topic_raw:
        return {"family": "generic", "subfamily": "", "display_family": "📌 Общая тема", "proposal_text": "Не удалось определить тему — опишите её подробнее."}
    result = build_onboarding_normalization_message(topic_raw)
    return result


@router.delete("/api/channels/{profile_id}")
async def delete_channel(
    profile_id: int,
    telegram_user_id: int = Depends(current_user_id),
):
    profiles_before = await db.list_channel_profiles(owner_id=telegram_user_id)
    existed_before = any(int(x.get("id") or 0) == int(profile_id) for x in profiles_before)
    if not existed_before:
        raise HTTPException(status_code=404, detail="Канал не найден")

    await db.delete_channel_profile(profile_id, owner_id=telegram_user_id)
    cache_invalidate(telegram_user_id)
    return await owner_summary(telegram_user_id)


@router.get("/api/_legacy/settings")
async def settings_legacy(telegram_user_id: int = Depends(current_user_id)):
    return {"settings": (await owner_summary(telegram_user_id))["settings"]}


@router.patch("/api/settings")
async def patch_settings(
    data: SettingsUpdate,
    telegram_user_id: int = Depends(current_user_id),
):
    fields = normalize_settings_update(data.model_dump(exclude_unset=True))

    if "posting_mode" in fields and fields["posting_mode"] not in {"both", "posts", "news"}:
        raise HTTPException(status_code=400, detail="Недопустимый режим публикации")

    if "channel_target" in fields and fields["channel_target"]:
        await verify_channel_ownership(telegram_user_id, fields["channel_target"])

    if "onboarding_completed" in fields and str(fields["onboarding_completed"]) == "1":
        ch_settings = await db.get_channel_settings(telegram_user_id)
        topic = (fields.get("topic") or (ch_settings.get("topic") or "")).strip()
        audience = (fields.get("channel_audience") or (ch_settings.get("channel_audience") or "")).strip()
        if not topic or not audience:
            logger.warning("patch_settings: ignoring onboarding_completed=1 for owner_id=%s — profile incomplete (topic_len=%d audience_len=%d)", telegram_user_id, len(topic), len(audience))
            del fields["onboarding_completed"]
        else:
            logger.info("patch_settings: setting onboarding_completed=1 for owner_id=%s", telegram_user_id)

    # Save all fields: channel-specific fields go to both channel_profiles AND
    # owner settings (for backwards compat); owner-only fields go to settings only.
    for key, value in fields.items():
        await db.save_channel_setting(telegram_user_id, key, str(value))

    if "topic" in fields:
        active = await db.get_active_channel_profile(owner_id=telegram_user_id)
        if active:
            new_topic = str(fields["topic"])
            await db.sync_channel_profile_topic(
                telegram_user_id,
                active.get("channel_target", ""),
                new_topic,
            )
            # Also update the structured topic_family/topic_subfamily in profile
            if new_topic:
                topic_family, topic_subfamily = normalize_topic_to_family(new_topic)
                await db.upsert_channel_profile(
                    owner_id=telegram_user_id,
                    channel_target=active.get("channel_target", ""),
                    topic=new_topic,
                    make_active=True,
                    topic_raw=new_topic,
                    topic_family=topic_family,
                    topic_subfamily=topic_subfamily,
                )

    # Sync author role fields to channel profile when any of them change
    _AUTHOR_ROLE_KEYS = {"author_role_type", "author_role_description", "author_activities", "author_forbidden_claims"}
    if _AUTHOR_ROLE_KEYS & fields.keys():
        active = await db.get_active_channel_profile(owner_id=telegram_user_id)
        if active:
            await db.upsert_channel_profile(
                owner_id=telegram_user_id,
                channel_target=active.get("channel_target", ""),
                make_active=True,
                author_role_type=str(fields.get("author_role_type", active.get("author_role_type", ""))),
                author_role_description=str(fields.get("author_role_description", active.get("author_role_description", ""))),
                author_activities=str(fields.get("author_activities", active.get("author_activities", ""))),
                author_forbidden_claims=str(fields.get("author_forbidden_claims", active.get("author_forbidden_claims", ""))),
            )

    cache_invalidate(telegram_user_id)
    return {"ok": True, "settings": (await owner_summary(telegram_user_id))["settings"]}


@router.get("/api/stats")
async def stats(telegram_user_id: int = Depends(current_user_id)):
    return await db.get_post_stats(owner_id=telegram_user_id)
