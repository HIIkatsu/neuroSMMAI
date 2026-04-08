from __future__ import annotations

import asyncio
from datetime import datetime
import logging
import re
import time
from typing import Any

logger = logging.getLogger(__name__)

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

import actions
import db
from ai_client import ai_chat
from competitor_service import analyse_competitor_and_generate
from config import load_config
from content import generate_post_bundle, generate_post_text, _remove_fabricated_refs, _safety_consistency_pass, _author_role_kwargs
from image_search import trigger_unsplash_download
from miniapp_analytics_service import recent_channel_history
from miniapp_media_service import extract_telegram_file_id_from_media_ref, find_inbox_item_for_media, find_inbox_item_for_video_media
from miniapp_plan_service import days_label, generate_plan_items_ai, hashtags_text, normalized_post_text
from miniapp_schemas import AIAddHashtags, AIAssets, AIGeneratePost, AIGenerateText, AIRewrite, DraftCreate, DraftGenerate, DraftPublish, DraftUpdate, PlanCreate, PlanGenerate, PlanUpdate, ScheduleCreate
from miniapp_shared import _AI_POST_COOLDOWN_SECONDS, _AI_POST_LAST_CALLS, cache_invalidate, clean_text, create_bot, current_user_id, ensure_drafts_capacity, owned_channel_targets, verify_channel_ownership
from topic_utils import detect_topic_family, get_family_guardrails
from runtime_trace import new_trace_id, trace_text_generation, debug_fields, is_debug_trace_enabled, TraceTimer

router = APIRouter(tags=["content"])


async def _active_channel_profile_id(owner_id: int) -> int | None:
    """Return the active channel profile ID for the given owner.

    Pinning the profile ID at request start prevents races where the user
    switches channels mid-request and the ``get_channel_settings(owner_id)``
    call would silently resolve to a different profile.
    """
    active = await db.get_active_channel_profile(owner_id=owner_id)
    return int(active["id"]) if active else None


async def _owned_targets(owner_id: int) -> set[str]:
    """Kept as thin wrapper for backward compat — delegates to shared helper."""
    return await owned_channel_targets(owner_id)


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
    """Raise HTTP 402 if the user has exhausted their per-feature monthly quota.

    Unlike ``_require_generation_quota`` this checks the granular
    ``user_feature_quotas`` table so that each capability has its own
    free-tier budget.
    """
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


async def _require_pro_tier(owner_id: int, feature: str = "функция") -> None:
    """Raise HTTP 402 if the user is not on a paid tier (pro/max)."""
    sub = await db.get_user_subscription(owner_id)
    tier = sub.get("subscription_tier", db.TIER_FREE)
    # Active trial counts as pro
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


async def _require_max_tier(owner_id: int, feature: str = "функция") -> None:
    """Raise HTTP 402 if the user is not on the max tier."""
    sub = await db.get_user_subscription(owner_id)
    tier = sub.get("subscription_tier", db.TIER_FREE)
    if tier == db.TIER_MAX:
        return
    raise HTTPException(
        status_code=402,
        detail={
            "code": "upgrade_required",
            "message": f"Функция «{feature}» доступна только в тарифе MAX. Узнай подробнее о тарифах.",
            "subscription": sub,
        },
    )


@router.get("/api/_legacy/drafts")
async def list_drafts_legacy(telegram_user_id: int = Depends(current_user_id)):
    drafts = await db.list_drafts(owner_id=telegram_user_id, limit=100)
    return {"drafts": drafts, "drafts_current": int(await db.count_drafts(owner_id=telegram_user_id, status='draft') or 0)}


@router.post("/api/drafts")
async def create_draft(
    data: DraftCreate,
    telegram_user_id: int = Depends(current_user_id),
):
    ch_settings = await db.get_channel_settings(telegram_user_id)
    channel_target = clean_text(data.channel_target) or (ch_settings.get("channel_target") or "")
    topic = clean_text(data.topic) or (ch_settings.get("topic") or "")

    if channel_target:
        owned = await _owned_targets(telegram_user_id)
        if channel_target not in owned:
            logger.warning("create_draft: ownership check failed owner_id=%s channel=%r", telegram_user_id, channel_target)
            raise HTTPException(status_code=403, detail="Нельзя использовать чужой канал")

    await ensure_drafts_capacity(telegram_user_id)

    latest_draft = await db.get_latest_draft(owner_id=telegram_user_id)
    if latest_draft:
        same_payload = (
            clean_text(str(latest_draft.get('text') or '')) == clean_text(str(data.text or ''))
            and clean_text(str(latest_draft.get('prompt') or '')) == clean_text(str(data.prompt or ''))
            and clean_text(str(latest_draft.get('topic') or '')) == topic
            and clean_text(str(latest_draft.get('channel_target') or '')) == channel_target
            and clean_text(str(latest_draft.get('media_ref') or '')) == clean_text(str(data.media_ref or ''))
            and clean_text(str(latest_draft.get('buttons_json') or '[]')) == clean_text(str(data.buttons_json or '[]'))
        )
        if same_payload:
            try:
                latest_created = datetime.fromisoformat(str(latest_draft.get('created_at') or '').replace('Z', '+00:00'))
                delta = abs((datetime.utcnow() - latest_created.replace(tzinfo=None)).total_seconds())
            except Exception:
                delta = 999
            if delta <= 8:
                return {"ok": True, "draft": latest_draft, "deduped": True}

    draft_id = await db.create_draft(
        owner_id=telegram_user_id,
        channel_target=channel_target,
        text=data.text,
        prompt=data.prompt,
        topic=topic,
        media_type=data.media_type,
        media_ref=data.media_ref,
        media_meta_json=data.media_meta_json,
        buttons_json=data.buttons_json,
        pin_post=int(data.pin_post),
        comments_enabled=int(data.comments_enabled),
        ad_mark=int(data.ad_mark),
        first_reaction=data.first_reaction,
        reply_to_message_id=int(data.reply_to_message_id or 0),
        status="draft",
    )
    draft = await db.get_draft(draft_id, owner_id=telegram_user_id)
    try:
        await db.mark_generation_history_draft_saved(owner_id=telegram_user_id, draft_id=draft_id, text=str(data.text or ""))
    except Exception:
        pass
    if str(data.media_meta_json or "").strip():
        await trigger_unsplash_download(data.media_meta_json)
    cache_invalidate(telegram_user_id, 'bootstrap', 'drafts', 'core', 'media_inbox')
    return {"ok": True, "draft": draft}


@router.patch("/api/drafts/{draft_id}")
async def patch_draft(
    draft_id: int,
    data: DraftUpdate,
    telegram_user_id: int = Depends(current_user_id),
):
    draft = await db.get_draft(draft_id, owner_id=telegram_user_id)
    if not draft:
        raise HTTPException(status_code=404, detail="Черновик не найден")

    fields = data.model_dump(exclude_unset=True)
    if fields.get("channel_target"):
        owned = await _owned_targets(telegram_user_id)
        if str(fields["channel_target"]) not in owned:
            raise HTTPException(status_code=403, detail="Нельзя привязать черновик к чужому каналу")

    # Prevent clients from directly setting internal lifecycle statuses
    if "status" in fields:
        allowed_statuses = {"draft"}
        if str(fields["status"] or "").strip().lower() not in allowed_statuses:
            logger.warning("patch_draft: blocked attempt to set restricted status=%r owner_id=%s draft_id=%s", fields["status"], telegram_user_id, draft_id)
            del fields["status"]

    old_media_ref = str(draft.get("media_ref") or "")
    old_media_meta = str(draft.get("media_meta_json") or "")
    for key, value in fields.items():
        await db.update_draft_field(draft_id, telegram_user_id, key, value)
    new_draft = await db.get_draft(draft_id, owner_id=telegram_user_id)
    if str(new_draft.get("media_ref") or "") != old_media_ref or str(new_draft.get("media_meta_json") or "") != old_media_meta:
        await trigger_unsplash_download(str(new_draft.get("media_meta_json") or ""))
    cache_invalidate(telegram_user_id, 'bootstrap', 'drafts', 'core', 'media_inbox')
    return {"ok": True, "draft": new_draft}


@router.delete("/api/drafts/{draft_id}")
async def remove_draft(
    draft_id: int,
    telegram_user_id: int = Depends(current_user_id),
):
    draft = await db.get_draft(draft_id, owner_id=telegram_user_id)
    if not draft:
        raise HTTPException(status_code=404, detail="Черновик не найден")

    media_ref = str(draft.get('media_ref') or '')
    linked_item = await find_inbox_item_for_media(telegram_user_id, media_ref)
    await db.delete_draft(draft_id, owner_id=telegram_user_id)
    if linked_item and hasattr(db, 'unmark_user_media_used'):
        await db.unmark_user_media_used(int(linked_item.get('id') or 0), owner_id=telegram_user_id)
    if media_ref and actions._is_local_media(media_ref):
        actions._cleanup_local_media(media_ref)
    cache_invalidate(telegram_user_id, 'bootstrap', 'drafts', 'core', 'media_inbox')
    cache_invalidate(telegram_user_id, 'bootstrap', 'drafts', 'core')
    cache_invalidate(telegram_user_id, 'bootstrap', 'plan', 'core')
    cache_invalidate(telegram_user_id, 'bootstrap', 'schedules', 'core', 'stats')
    cache_invalidate(telegram_user_id, 'bootstrap', 'settings', 'core')
    return {"ok": True}


@router.post("/api/drafts/generate")
async def generate_draft(
    data: DraftGenerate,
    telegram_user_id: int = Depends(current_user_id),
):
    await _require_feature_quota(telegram_user_id, "generate", "Генерация черновика")
    await ensure_drafts_capacity(telegram_user_id)
    config = load_config()
    history = await recent_channel_history(telegram_user_id)
    draft_id = await actions.create_generated_draft(
        config,
        data.prompt,
        owner_id=telegram_user_id,
        force_image=True,
    )
    await db.increment_generations_used(telegram_user_id)
    await db.increment_feature_used(telegram_user_id, "generate")
    cache_invalidate(telegram_user_id, 'bootstrap', 'drafts', 'core', 'media_inbox')
    return {"ok": True, "draft": await db.get_draft(draft_id, owner_id=telegram_user_id)}




@router.post("/api/ai/generate-text")
async def ai_generate_text(
    data: AIGenerateText,
    telegram_user_id: int = Depends(current_user_id),
):
    await _require_feature_quota(telegram_user_id, "generate", "Генерация текста")
    config = load_config()
    # Pin channel_profile_id at request start to avoid mid-request channel-switch races
    cpid = await _active_channel_profile_id(telegram_user_id)
    # Use channel-scoped settings for consistent per-channel generation
    history, ch_settings = await asyncio.gather(
        recent_channel_history(telegram_user_id),
        db.get_channel_settings(telegram_user_id, channel_profile_id=cpid),
    )
    topic = clean_text(data.topic) or (ch_settings.get("topic") or "")
    bundle = await generate_post_bundle(
        config.openrouter_api_key,
        config.openrouter_model,
        topic=topic,
        prompt=clean_text(data.prompt),
        channel_style=(ch_settings.get("channel_style") or ""),
        content_rubrics=(ch_settings.get("content_rubrics") or ""),
        post_scenarios=(ch_settings.get("post_scenarios") or ""),
        channel_audience=(ch_settings.get("channel_audience") or ""),
        content_constraints=(ch_settings.get("content_constraints") or ""),
        recent_posts=(history.get("recent_posts", []) + history.get("recent_drafts", []))[:10],
        recent_plan=history.get("recent_plan", [])[:10],
        base_url=config.openrouter_base_url,
        owner_id=telegram_user_id,
    )
    text = "\n\n".join(part for part in [bundle.get("title",""), bundle.get("body",""), bundle.get("cta","")] if part)
    await db.increment_generations_used(telegram_user_id)
    await db.increment_feature_used(telegram_user_id, "generate")
    _trace_id = bundle.get("_trace_id", "")
    _trace = trace_text_generation(
        trace_id=_trace_id or new_trace_id(),
        route="/api/ai/generate-text",
        source_mode="manual",
        requested_topic=clean_text(data.topic),
        channel_topic=topic,
        author_role=str(ch_settings.get("author_role_type") or ""),
        prompt_builder="generate_post_bundle",
        planner_used=False, writer_used=bool(text),
        rewrite_used=False, final_archetype="",
    )
    result = {"ok": True, "text": normalized_post_text(text), **bundle}
    _dbg = debug_fields(_trace)
    if _dbg:
        result["_debug"] = _dbg
    return result


@router.post("/api/ai/add-hashtags")
async def ai_add_hashtags(
    data: AIAddHashtags,
    telegram_user_id: int = Depends(current_user_id),
):
    await _require_feature_quota(telegram_user_id, "hashtags", "Хэштеги")
    ch_settings = await db.get_channel_settings(telegram_user_id)
    topic = clean_text(data.topic) or clean_text(data.prompt) or (ch_settings.get("topic") or "")
    text = clean_text(data.text)
    if not text:
        raise HTTPException(status_code=400, detail="Сначала добавь текст поста")
    result_text = hashtags_text(text, topic)
    if result_text.strip() == text.strip():
        tags = actions.generate_hashtags(topic or text[:120], text)
        if tags:
            result_text = f"{text.rstrip()}\n\n{tags}".strip()
    await db.increment_feature_used(telegram_user_id, "hashtags")
    return {"ok": True, "text": result_text}



@router.post("/api/ai/rewrite")
async def ai_rewrite(
    data: AIRewrite,
    telegram_user_id: int = Depends(current_user_id),
):
    await _require_feature_quota(telegram_user_id, "rewrite", "Рерайт текста")
    config = load_config()
    cpid = await _active_channel_profile_id(telegram_user_id)
    ch_settings = await db.get_channel_settings(telegram_user_id, channel_profile_id=cpid)
    topic = clean_text(data.topic) or (ch_settings.get("topic") or "")
    base_text = clean_text(data.text)
    if not base_text:
        raise HTTPException(status_code=400, detail="Сначала добавь текст поста")
    mode_map = {
        "improve": "Сделай текст сильнее, живее и чище, но не меняй смысл.",
        "shorter": "Сделай текст короче и плотнее, убери лишнее.",
        "selling": "Сделай текст чуть более продающим, но без давления и дешёвых штампов.",
        "softer": "Сделай текст мягче, теплее и спокойнее.",
    }
    mode_instruction = mode_map.get(clean_text(data.mode) or 'improve', mode_map['improve'])
    style = (ch_settings.get("channel_style") or "").strip()
    family = detect_topic_family(topic) if topic else "generic"
    family_guardrails = get_family_guardrails(family) if family != "generic" else ""
    guardrails_line = f"\nПравила ниши: {family_guardrails}" if family_guardrails else ""
    raw = await ai_chat(
        api_key=config.openrouter_api_key,
        model=config.openrouter_model,
        base_url=config.openrouter_base_url,
        temperature=0.72,
        messages=[{
            "role": "user",
            "content": f"""
Ты редактор Telegram-канала.

Тема канала: {topic or 'без общей темы'}
Стиль канала: {style or 'живой, простой, уверенный'}{guardrails_line}

Исходный текст:
{base_text}

Задача: {mode_instruction}

Правила:
- Только готовый текст поста.
- Без markdown и служебных пояснений.
- Без фраз вроде «в этой статье», «это не просто», «давайте разберёмся».
- Без лишней философии, нужен обычный живой пост для Telegram.
- НЕ выдумывай @упоминания пользователей, названия каналов, ссылки, URL или источники. Если в исходном тексте нет ссылок и @упоминаний, не добавляй их.
""".strip()
        }]
    )
    result_text = normalized_post_text(raw or base_text)
    # Apply fabrication cleanup to rewritten text — same rules as main pipeline
    result_text, _, _ = _remove_fabricated_refs(result_text)
    # Final safety / consistency pass — use channel-scoped author role settings
    ar_settings: dict[str, str] = {}
    for _ar_key in ("author_role_type", "author_role_description", "author_activities", "author_forbidden_claims"):
        ar_settings[_ar_key] = (ch_settings.get(_ar_key) or "").strip()
    result_text = _safety_consistency_pass(result_text, **_author_role_kwargs(ar_settings))
    await db.increment_feature_used(telegram_user_id, "rewrite")
    trace_text_generation(
        trace_id=new_trace_id(),
        route="/api/ai/rewrite",
        source_mode="rewrite",
        requested_topic=topic,
        channel_topic=topic,
        author_role=str(ar_settings.get("author_role_type") or ""),
        prompt_builder="ai_chat_direct",
        planner_used=False, writer_used=False,
        rewrite_used=True, final_archetype="",
    )
    return {"ok": True, "text": result_text}


@router.post("/api/ai/assets")
async def ai_assets(
    data: AIAssets,
    telegram_user_id: int = Depends(current_user_id),
):
    await _require_feature_quota(telegram_user_id, "assets", "Ассеты")
    config = load_config()
    ch_settings = await db.get_channel_settings(telegram_user_id)
    topic = clean_text(data.topic) or (ch_settings.get("topic") or "")
    style = (ch_settings.get("channel_style") or "").strip()
    rubrics = (ch_settings.get("content_rubrics") or "").strip()
    scenarios = (ch_settings.get("post_scenarios") or "").strip()
    source = clean_text(data.text) or clean_text(data.prompt) or topic
    history = await recent_channel_history(telegram_user_id)
    bundle = await generate_post_bundle(
        config.openrouter_api_key,
        config.openrouter_model,
        topic=topic,
        prompt=source,
        channel_style=style,
        content_rubrics=rubrics,
        post_scenarios=scenarios,
        recent_posts=(history.get("recent_posts", []) + history.get("recent_drafts", []))[:10],
        recent_plan=history.get("recent_plan", [])[:10],
        base_url=config.openrouter_base_url,
        owner_id=telegram_user_id,
    )
    await db.increment_feature_used(telegram_user_id, "assets")
    return {"ok": True, **bundle}


@router.post("/api/ai/generate-post")
async def ai_generate_post(
    data: AIGeneratePost,
    telegram_user_id: int = Depends(current_user_id),
):
    await _require_feature_quota(telegram_user_id, "generate", "Генерация поста")
    config = load_config()
    cpid = await _active_channel_profile_id(telegram_user_id)
    ch_settings = await db.get_channel_settings(telegram_user_id, channel_profile_id=cpid)
    channel_topic = clean_text(ch_settings.get("topic") or "")
    request_topic = clean_text(data.topic)
    topic = request_topic or channel_topic
    prompt = clean_text(data.prompt)
    current_text = clean_text(data.current_text)
    existing_media_ref = clean_text(data.media_ref)
    existing_media_type = clean_text(data.media_type) or ("photo" if existing_media_ref else "none")
    existing_media_meta_json = str(data.media_meta_json or "").strip()
    requested_draft_id = int(data.draft_id or 0)

    source_prompt = prompt or current_text or request_topic or channel_topic
    if not source_prompt:
        raise HTTPException(status_code=400, detail="Нужна тема или задача для ИИ")

    now_ts = time.time()
    last_ts = float(_AI_POST_LAST_CALLS.get(int(telegram_user_id), 0.0) or 0.0)
    remaining = int(_AI_POST_COOLDOWN_SECONDS - (now_ts - last_ts))
    if remaining > 0:
        raise HTTPException(status_code=429, detail=f"Подожди {remaining} сек. перед следующей генерацией")

    try:
        logger.info("ai_generate_post start owner_id=%s topic=%r prompt=%r", telegram_user_id, topic[:80] if topic else "", source_prompt[:80] if source_prompt else "")
        should_force_image = bool(data.force_image and (bool(prompt) or not existing_media_ref))

        # --- Manual generation with 1 automatic retry on quality failure ---
        _MANUAL_MAX_ATTEMPTS = 2
        payload = None
        last_error = None
        for attempt in range(1, _MANUAL_MAX_ATTEMPTS + 1):
            try:
                payload = await actions.generate_post_payload(
                    config,
                    source_prompt,
                    owner_id=telegram_user_id,
                    force_image=should_force_image,
                    current_media_ref=existing_media_ref,
                    generation_path="editor",
                )
                if payload and (payload.get("text") or payload.get("body")):
                    break  # Success
                payload = None
            except RuntimeError as gen_err:
                last_error = gen_err
                logger.warning(
                    "ai_generate_post: attempt %d/%d failed owner_id=%s: %s",
                    attempt, _MANUAL_MAX_ATTEMPTS, telegram_user_id, str(gen_err)[:200],
                )
                payload = None
                continue

        if not payload:
            # Build human-readable failure reason from the last error
            user_reason = ""
            if last_error:
                err_text = str(last_error).lower()
                if "качеств" in err_text or "quality" in err_text:
                    user_reason = "Текст не прошёл проверку качества"
                elif "тем" in err_text or "topic" in err_text:
                    user_reason = "Текст вышел не по теме канала"
                elif "коротк" in err_text or "short" in err_text or "density" in err_text:
                    user_reason = "Текст оказался слишком коротким или пустым"
                elif "факт" in err_text or "fabricat" in err_text:
                    user_reason = "Обнаружены сомнительные утверждения"
                else:
                    user_reason = "Не удалось получить достаточно качественный текст"
            reason_suffix = f" ({user_reason})" if user_reason else ""
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "generation_failed",
                    "message": f"Не удалось сгенерировать качественный пост{reason_suffix}. Попробуйте ещё раз или измените запрос.",
                    "retryable": True,
                    "reason": user_reason,
                },
            )

        # Cooldown starts only after successful generation so that a
        # transient LLM failure does not force the user to wait.
        _AI_POST_LAST_CALLS[int(telegram_user_id)] = time.time()
        try:
            await db.add_generation_history(
                owner_id=telegram_user_id,
                source="generate-post",
                prompt=source_prompt,
                topic=topic,
                title=str(payload.get("title") or ""),
                body=str(payload.get("body") or ""),
                cta=str(payload.get("cta") or ""),
                short=str(payload.get("short") or ""),
                safety_status=str(payload.get("safety_status") or "ok"),
            )
        except Exception:
            pass
        await db.increment_generations_used(telegram_user_id)
        await db.increment_feature_used(telegram_user_id, "generate")
    except HTTPException:
        raise
    except Exception as exc:
        import logging
        logging.getLogger(__name__).exception("ai_generate_post failed: %s", exc)
        message = str(exc)[:200]
        if any(marker in message.lower() for marker in ["не нашел достаточно сильного свежего инфоповода", "не нашёл достаточно сильного свежего инфоповода", "live-режима нет подтвержденных свежих фактов", "live-режима нет подтверждённых свежих фактов"]):
            raise HTTPException(status_code=422, detail=message) from exc
        # Sanitize: never expose internal Python tracebacks or class names to the user
        raise HTTPException(
            status_code=502,
            detail="Произошла ошибка при генерации поста. Попробуйте ещё раз через несколько секунд.",
        ) from exc

    generated_media_ref = clean_text(payload.get("media_ref") or "")
    generated_media_type = clean_text(payload.get("media_type") or ("photo" if generated_media_ref else "none"))
    generated_media_meta_json = str(payload.get("media_meta_json") or "").strip()

    if prompt:
        existing_media_ref = ""
        existing_media_type = "none"
        existing_media_meta_json = ""

    image_warning = ""
    if not existing_media_ref and data.force_image and not generated_media_ref:
        logger.debug(
            "AI_GENERATE_POST_IMAGE_MISSING topic=%r prompt=%r force_image=%s draft_id=%s",
            topic, prompt, bool(data.force_image), requested_draft_id,
        )
        image_warning = "Подходящее изображение не найдено, пост создан без медиа"

    text = normalized_post_text(payload.get("text") or "")
    title = clean_text(payload.get("title") or "")
    cta = clean_text(payload.get("cta") or "")
    short = clean_text(payload.get("short") or "")
    button_text = clean_text(payload.get("button_text") or "")
    warning = image_warning
    auto_hashtags = (await db.get_setting("auto_hashtags", owner_id=telegram_user_id) or "0") == "1"
    if auto_hashtags:
        try:
            text = hashtags_text(text, topic or prompt or current_text)
        except Exception as ht_exc:
            logger.warning("ai_generate_post hashtags failed owner_id=%s: %s", telegram_user_id, ht_exc)
            if not warning:
                warning = "Хэштеги не удалось добавить, но пост сохранён"

    channel_target = clean_text(data.channel_target) or (ch_settings.get("channel_target") or "")
    if channel_target:
        owned = await _owned_targets(telegram_user_id)
        if str(channel_target) not in owned:
            raise HTTPException(status_code=403, detail="Нельзя привязать пост к чужому каналу")

    final_media_ref = existing_media_ref or generated_media_ref
    final_media_type = existing_media_type if existing_media_ref else (generated_media_type or "none")
    final_media_meta_json = existing_media_meta_json if existing_media_ref else generated_media_meta_json

    if requested_draft_id:
        draft = await db.get_draft(requested_draft_id, owner_id=telegram_user_id)
        if not draft:
            raise HTTPException(status_code=404, detail="Черновик не найден")
        await db.update_draft_field(requested_draft_id, telegram_user_id, "channel_target", channel_target)
        await db.update_draft_field(requested_draft_id, telegram_user_id, "text", text)
        await db.update_draft_field(requested_draft_id, telegram_user_id, "prompt", prompt or source_prompt)
        await db.update_draft_field(requested_draft_id, telegram_user_id, "topic", topic)
        await db.update_draft_field(requested_draft_id, telegram_user_id, "media_type", final_media_type)
        await db.update_draft_field(requested_draft_id, telegram_user_id, "media_ref", final_media_ref)
        await db.update_draft_field(requested_draft_id, telegram_user_id, "media_meta_json", final_media_meta_json)
        await db.update_draft_field(requested_draft_id, telegram_user_id, "buttons_json", clean_text(data.buttons_json) or "[]")
        await db.update_draft_field(requested_draft_id, telegram_user_id, "pin_post", int(data.pin_post or 0))
        await db.update_draft_field(requested_draft_id, telegram_user_id, "comments_enabled", 0 if int(data.comments_enabled or 0) == 0 else 1)
        await db.update_draft_field(requested_draft_id, telegram_user_id, "ad_mark", int(data.ad_mark or 0))
        await trigger_unsplash_download(final_media_meta_json)
        cache_invalidate(telegram_user_id, 'drafts', 'core', 'bootstrap')
        return {"ok": True, "updated": True, "draft": await db.get_draft(requested_draft_id, owner_id=telegram_user_id), "image_warning": image_warning, "warning": warning, "title": title, "cta": cta, "short": short, "button_text": button_text, "media_ref": final_media_ref, "media_type": final_media_type, "media_meta_json": final_media_meta_json}

    await ensure_drafts_capacity(telegram_user_id)
    draft_id = await db.create_draft(
        owner_id=telegram_user_id,
        channel_target=channel_target,
        text=text,
        prompt=prompt or source_prompt,
        topic=topic,
        media_type=final_media_type,
        media_ref=final_media_ref,
        media_meta_json=final_media_meta_json,
        buttons_json=clean_text(data.buttons_json) or "[]",
        pin_post=int(data.pin_post or 0),
        comments_enabled=0 if int(data.comments_enabled or 0) == 0 else 1,
        ad_mark=int(data.ad_mark or 0),
        first_reaction="",
        reply_to_message_id=0,
        status="draft",
    )
    await trigger_unsplash_download(final_media_meta_json)
    cache_invalidate(telegram_user_id, 'drafts', 'core', 'bootstrap')
    return {"ok": True, "updated": False, "draft": await db.get_draft(draft_id, owner_id=telegram_user_id), "image_warning": image_warning, "warning": warning, "title": title, "cta": cta, "short": short, "button_text": button_text, "media_ref": final_media_ref, "media_type": final_media_type, "media_meta_json": final_media_meta_json}

@router.post("/api/drafts/publish")
async def publish_draft(
    data: DraftPublish,
    telegram_user_id: int = Depends(current_user_id),
):
    draft = await db.get_draft(data.draft_id, owner_id=telegram_user_id)
    if not draft:
        raise HTTPException(status_code=404, detail="Черновик не найден")

    if draft.get("channel_target"):
        owned = await _owned_targets(telegram_user_id)
        if str(draft["channel_target"]) not in owned:
            raise HTTPException(status_code=403, detail="Нельзя публиковать в чужой канал")

    # Validate mirror targets (linked publishing)
    mirror_targets: list[str] = []
    if data.mirror_targets:
        owned = await _owned_targets(telegram_user_id)
        primary = str(draft.get("channel_target") or "")
        for mt in data.mirror_targets:
            mt = str(mt).strip()
            if mt and mt != primary and mt in owned:
                mirror_targets.append(mt)

    # Allow retry of previously-failed drafts: reset status back to "draft" so the
    # atomic claim below can succeed.
    current_status = str(draft.get("status") or "draft").strip().lower()
    if current_status == "failed":
        await db.update_draft_field(data.draft_id, telegram_user_id, "status", "draft")
        draft = await db.get_draft(data.draft_id, owner_id=telegram_user_id) or draft

    # Atomically claim the draft to prevent double-publish
    if not await db.claim_draft_for_publish(data.draft_id, telegram_user_id):
        # Re-read to get the fresh status for a meaningful error message
        fresh = await db.get_draft(data.draft_id, owner_id=telegram_user_id)
        fresh_status = str((fresh or draft).get("status") or "draft").strip().lower()
        if fresh_status == "publishing":
            raise HTTPException(status_code=409, detail="Черновик уже публикуется")
        if fresh_status == "published":
            raise HTTPException(status_code=409, detail="Черновик уже опубликован")
        raise HTTPException(status_code=409, detail="Черновик недоступен для публикации")

    # Pass the draft with status="publishing" so actions.publish_draft() skips its
    # own redundant claim (which would fail and abort the publish).
    draft = {**draft, "status": "publishing"}

    cleanup_item = await find_inbox_item_for_media(telegram_user_id, str(draft.get('media_ref') or ''))

    logger.info("publish_draft attempt owner_id=%s draft_id=%s channel=%s mirror=%s", telegram_user_id, data.draft_id, str(draft.get("channel_target") or ""), mirror_targets)
    bot = await create_bot()
    try:
        result = await actions.publish_draft(bot, draft, owner_id=telegram_user_id)

        # Mirror publishing: send same content to additional channels
        mirror_results: list[dict] = []
        if result.ok and mirror_targets:
            text = str(draft.get("text") or "")
            media_ref = str(draft.get("media_ref") or "")
            media_type = str(draft.get("media_type") or "none")
            for mt_channel in mirror_targets:
                try:
                    if media_type != "none" and media_ref:
                        if media_type == "video":
                            msg = await bot.send_video(mt_channel, video=media_ref, caption=text[:1024], parse_mode="HTML")
                        else:
                            msg = await bot.send_photo(mt_channel, photo=media_ref, caption=text[:1024], parse_mode="HTML")
                    else:
                        msg = await bot.send_message(mt_channel, text=text, parse_mode="HTML")
                    mirror_results.append({"channel": mt_channel, "ok": True, "message_id": msg.message_id})
                    await db.log_post(
                        owner_id=telegram_user_id,
                        channel_target=mt_channel,
                        content_type=media_type if media_type != "none" else "text",
                        text=text,
                        topic=str(draft.get("topic") or ""),
                        file_id=media_ref,
                        telegram_message_id=msg.message_id,
                    )
                except Exception as e:
                    logger.warning("mirror publish failed channel=%s: %s", mt_channel, e)
                    mirror_results.append({"channel": mt_channel, "ok": False, "error": str(e)[:200]})

            # Audit trail: log mirror results summary
            ok_count = sum(1 for mr in mirror_results if mr.get("ok"))
            fail_count = len(mirror_results) - ok_count
            logger.info("mirror publish complete owner=%s draft=%s ok=%d fail=%d",
                        telegram_user_id, data.draft_id, ok_count, fail_count)
    finally:
        await bot.session.close()

    if not result.ok:
        await db.update_draft_field(data.draft_id, telegram_user_id, "status", "draft")
        logger.warning("publish_draft failed owner_id=%s draft_id=%s error=%s", telegram_user_id, data.draft_id, result.error)
        raise HTTPException(status_code=400, detail=result.error or "Ошибка публикации")

    if cleanup_item and hasattr(db, 'delete_user_media'):
        await db.delete_user_media(int(cleanup_item.get('id') or 0), owner_id=telegram_user_id)
        pending_id = int(await db.get_setting('pending_media_item_id', owner_id=telegram_user_id) or 0)
        if pending_id == int(cleanup_item.get('id') or 0):
            await db.set_setting('pending_media_item_id', '0', owner_id=telegram_user_id)
            await db.set_setting('pending_media_kind', '', owner_id=telegram_user_id)
    cache_invalidate(telegram_user_id, 'bootstrap', 'drafts', 'core', 'media_inbox', 'stats')

    response = {
        "ok": True,
        "status": "published_now",
        "message": "Пост опубликован",
        "telegram_message_id": int(result.message_id or 0),
        "channel_target": str(result.channel or draft.get('channel_target') or ''),
        "content_type": str(result.content_type or draft.get('media_type') or 'text'),
    }
    if mirror_results:
        response["mirror_results"] = mirror_results
        # Signal partial mirror failure so frontend can display appropriately
        mirror_failures = [mr for mr in mirror_results if not mr.get("ok")]
        if mirror_failures:
            response["mirror_partial_failure"] = True
            response["mirror_failed_channels"] = [mf["channel"] for mf in mirror_failures]
    return response


@router.get("/api/_legacy/plan")
async def list_plan_legacy(telegram_user_id: int = Depends(current_user_id)):
    return {"plan": await db.list_plan_items(owner_id=telegram_user_id, limit=500)}


@router.post("/api/plan")
async def create_plan_item(
    data: PlanCreate,
    telegram_user_id: int = Depends(current_user_id),
):
    await db.add_plan_item(
        dt=clean_text(data.dt),
        owner_id=telegram_user_id,
        topic=clean_text(data.topic),
        prompt=clean_text(data.prompt),
    )
    cache_invalidate(telegram_user_id, 'plan', 'core', 'bootstrap')
    return {"ok": True, "status": "plan_created"}


@router.patch("/api/plan/{item_id}")
async def patch_plan_item(
    item_id: int,
    data: PlanUpdate,
    telegram_user_id: int = Depends(current_user_id),
):
    item = await db.get_plan_item(item_id, owner_id=telegram_user_id)
    if not item:
        raise HTTPException(status_code=404, detail="Элемент плана не найден")

    await db.update_plan_item(
        item_id,
        owner_id=telegram_user_id,
        dt=data.dt,
        topic=data.topic,
        prompt=data.prompt,
    )
    cache_invalidate(telegram_user_id, 'plan', 'core', 'bootstrap')
    return {"ok": True, "item": await db.get_plan_item(item_id, owner_id=telegram_user_id)}


@router.delete("/api/plan/{item_id}")
async def delete_plan_item(
    item_id: int,
    telegram_user_id: int = Depends(current_user_id),
):
    item = await db.get_plan_item(item_id, owner_id=telegram_user_id)
    if not item:
        raise HTTPException(status_code=404, detail="Элемент плана не найден")

    await db.delete_plan_item(item_id, owner_id=telegram_user_id)
    cache_invalidate(telegram_user_id, 'plan', 'core', 'bootstrap')
    return {"ok": True, "status": "plan_deleted"}


@router.post("/api/plan/generate")
async def generate_plan(
    data: PlanGenerate,
    telegram_user_id: int = Depends(current_user_id),
):
    await _require_feature_quota(telegram_user_id, "plan_generate", "Генерация плана")
    cpid = await _active_channel_profile_id(telegram_user_id)
    ch_settings = await db.get_channel_settings(telegram_user_id, channel_profile_id=cpid)
    topic = (
        clean_text(data.topic)
        or (ch_settings.get("topic") or "")
    )

    config = load_config()
    history = await recent_channel_history(telegram_user_id)
    items = await generate_plan_items_ai(
        data.start_date,
        data.days,
        data.posts_per_day,
        topic,
        data.post_time,
        config=config,
        owner_id=telegram_user_id,
        history=history,
        channel_profile_id=cpid,
    )

    if data.clear_existing:
        await db.clear_unposted_plan_items(owner_id=telegram_user_id)

    for item in items:
        await db.add_plan_item(
            dt=item["dt"],
            owner_id=telegram_user_id,
            prompt=item["prompt"],
        )

    cache_invalidate(telegram_user_id, 'plan', 'core', 'bootstrap')
    await db.increment_feature_used(telegram_user_id, "plan_generate")
    return {"ok": True, "created": len(items)}


@router.get("/api/_legacy/schedules")
async def list_schedules_legacy(telegram_user_id: int = Depends(current_user_id)):
    rows = await db.list_schedules(owner_id=telegram_user_id)
    for row in rows:
        row["days_label"] = days_label(row.get("days", "*"))
    return {"schedules": rows}


@router.post("/api/schedules")
async def create_schedule(
    data: ScheduleCreate,
    telegram_user_id: int = Depends(current_user_id),
):
    await _require_pro_tier(telegram_user_id, "Автопостинг по расписанию")
    time_hhmm = clean_text(data.time_hhmm)
    days = clean_text(data.days) or "*"
    if not re.match(r"^([01]\d|2[0-3]):[0-5]\d$", time_hhmm):
        raise HTTPException(status_code=400, detail="Время должно быть в формате HH:MM")

    # Link schedule to the currently active channel profile
    active = await db.get_active_channel_profile(owner_id=telegram_user_id)
    cpid = int(active.get("id", 0)) if active else 0

    await db.add_schedule(
        time_hhmm=time_hhmm,
        days=days,
        owner_id=telegram_user_id,
        channel_profile_id=cpid,
    )
    cache_invalidate(telegram_user_id, 'schedules', 'core', 'bootstrap')
    return {"ok": True, "status": "schedule_created", "time_hhmm": time_hhmm, "days": days}


@router.delete("/api/schedules/{schedule_id}")
async def delete_schedule(
    schedule_id: int,
    telegram_user_id: int = Depends(current_user_id),
):
    rows = await db.list_schedules(owner_id=telegram_user_id)
    target = next((r for r in rows if int(r["id"]) == int(schedule_id)), None)
    if not target:
        raise HTTPException(status_code=404, detail="Расписание не найдено")
    await db.delete_schedule(schedule_id, owner_id=telegram_user_id)
    cache_invalidate(telegram_user_id, 'schedules', 'core', 'bootstrap')
    return {"ok": True, "status": "schedule_deleted"}


# ---------- Competitor Spy ----------

class CompetitorSpyRequest(BaseModel):
    channel_link: str


@router.post("/api/competitor/spy")
async def competitor_spy(
    data: CompetitorSpyRequest,
    telegram_user_id: int = Depends(current_user_id),
):
    """Analyse a public competitor channel and save 3 new unique drafts (Max tier only)."""
    await _require_max_tier(telegram_user_id, "Шпион конкурентов")

    channel_link = (data.channel_link or "").strip()
    if not channel_link:
        raise HTTPException(status_code=400, detail="Укажи ссылку на канал конкурента")

    # Load user context for personalised generation — channel-scoped
    cpid = await _active_channel_profile_id(telegram_user_id)
    ch_settings = await db.get_channel_settings(telegram_user_id, channel_profile_id=cpid)

    try:
        generated = await analyse_competitor_and_generate(
            channel_link,
            user_topic=ch_settings.get("topic") or "",
            user_style=ch_settings.get("channel_style") or "",
            user_audience=ch_settings.get("channel_audience") or "",
            ai_chat_fn=ai_chat,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    if not generated:
        raise HTTPException(status_code=502, detail="ИИ не вернул черновики. Попробуй ещё раз.")

    channel_target = ch_settings.get("channel_target") or ""
    saved_drafts = []
    for item in generated:
        text = item.get("text") or ""
        topic = item.get("topic") or ""

        # Check draft capacity before each save (it throws 403 if limit reached)
        await ensure_drafts_capacity(telegram_user_id)

        draft_id = await db.create_draft(
            owner_id=telegram_user_id,
            channel_target=channel_target,
            text=text,
            prompt=f"🕵️‍♂️ Шпион: {channel_link}",
            topic=topic,
            media_type="",
            media_ref="",
            media_meta_json="{}",
            buttons_json="[]",
            pin_post=0,
            comments_enabled=1,
            ad_mark=0,
            first_reaction="",
            reply_to_message_id=0,
            send_silent=0,
            status="draft",
            draft_source="spy",
        )
        draft = await db.get_draft(draft_id, owner_id=telegram_user_id)
        if draft:
            saved_drafts.append(draft)

    cache_invalidate(telegram_user_id, 'drafts', 'core', 'bootstrap')
    return {"ok": True, "drafts": saved_drafts, "count": len(saved_drafts)}


