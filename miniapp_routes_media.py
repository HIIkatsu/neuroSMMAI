from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

import db
from config import load_config
from media_lifecycle import save_upload_file
from miniapp_media_service import build_media_shortcuts, delete_media_inbox_item as delete_media_inbox_item_service, guess_media_type_from_kind, media_storage_ref, serialize_media_item, stream_telegram_media
from miniapp_shared import UPLOAD_DIR, cache_get, cache_invalidate, cache_set, current_user_id, ensure_drafts_capacity

router = APIRouter(tags=["media"])


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



@router.get("/api/_legacy/media/inbox")
async def list_media_inbox_legacy(telegram_user_id: int = Depends(current_user_id)):
    rows = [
        serialize_media_item(item)
        for item in await db.list_user_media(owner_id=telegram_user_id, limit=50)
        if int(item.get('used_count') or 0) == 0
    ]
    return {"media_inbox": rows}


@router.post("/api/media/inbox/{item_id}/draft")
async def create_draft_from_inbox(
    item_id: int,
    telegram_user_id: int = Depends(current_user_id),
):
    item = await db.get_user_media(item_id, owner_id=telegram_user_id)
    if not item:
        raise HTTPException(status_code=404, detail="Файл из чата не найден")

    await ensure_drafts_capacity(telegram_user_id)

    media_ref = media_storage_ref(item)
    if not media_ref:
        raise HTTPException(status_code=400, detail="У файла нет Telegram file_id")

    channel_target = await db.get_setting("channel_target", owner_id=telegram_user_id) or ""
    topic = await db.get_setting("topic", owner_id=telegram_user_id) or ""
    draft_id = await db.create_draft(
        owner_id=telegram_user_id,
        channel_target=channel_target,
        text=str(item.get("caption") or "").strip(),
        prompt="",
        topic=str(topic),
        media_type=guess_media_type_from_kind(str(item.get("kind") or "video")),
        media_ref=media_ref,
        buttons_json="[]",
        pin_post=0,
        comments_enabled=1,
        ad_mark=0,
        first_reaction="",
        reply_to_message_id=0,
        status="draft",
    )
    await db.mark_user_media_used(item_id, owner_id=telegram_user_id)
    pending_id = int(await db.get_setting("pending_media_item_id", owner_id=telegram_user_id) or 0)
    if pending_id == int(item_id):
        await db.set_setting("pending_media_item_id", "0", owner_id=telegram_user_id)
        await db.set_setting("pending_media_kind", "", owner_id=telegram_user_id)
    cache_invalidate(telegram_user_id, 'drafts', 'media_inbox', 'core', 'bootstrap')
    return {"ok": True, "draft": await db.get_draft(draft_id, owner_id=telegram_user_id), "draft_id": draft_id}


@router.post("/api/media/inbox/{item_id}/attach")
async def attach_media_inbox_to_draft(
    item_id: int,
    payload: dict[str, Any],
    telegram_user_id: int = Depends(current_user_id),
):
    draft_id = int(payload.get("draft_id") or 0)
    if not draft_id:
        raise HTTPException(status_code=400, detail="Не указан черновик")
    item = await db.get_user_media(item_id, owner_id=telegram_user_id)
    if not item:
        raise HTTPException(status_code=404, detail="Файл из чата не найден")
    draft = await db.get_draft(draft_id, owner_id=telegram_user_id)
    if not draft:
        raise HTTPException(status_code=404, detail="Черновик не найден")
    media_ref = media_storage_ref(item)
    media_type = guess_media_type_from_kind(str(item.get("kind") or "video"))
    await db.update_draft_field(draft_id, telegram_user_id, "media_ref", media_ref)
    await db.update_draft_field(draft_id, telegram_user_id, "media_type", media_type)
    await db.mark_user_media_used(item_id, owner_id=telegram_user_id)
    pending_id = int(await db.get_setting("pending_media_item_id", owner_id=telegram_user_id) or 0)
    if pending_id == int(item_id):
        await db.set_setting("pending_media_item_id", "0", owner_id=telegram_user_id)
        await db.set_setting("pending_media_kind", "", owner_id=telegram_user_id)
    cache_invalidate(telegram_user_id, 'drafts', 'media_inbox', 'core', 'bootstrap')
    return {"ok": True, "draft": await db.get_draft(draft_id, owner_id=telegram_user_id)}


@router.delete("/api/media/inbox/{item_id}")
async def delete_media_inbox_item(
    item_id: int,
    telegram_user_id: int = Depends(current_user_id),
):
    item = await db.get_user_media(item_id, owner_id=telegram_user_id)
    if not item:
        raise HTTPException(status_code=404, detail="Файл из чата не найден")

    await delete_media_inbox_item_service(telegram_user_id, item_id)

    pending_id = int(await db.get_setting("pending_media_item_id", owner_id=telegram_user_id) or 0)
    if pending_id == int(item_id):
        await db.set_setting("pending_media_item_id", "0", owner_id=telegram_user_id)
        await db.set_setting("pending_media_kind", "", owner_id=telegram_user_id)
    cache_invalidate(telegram_user_id, 'media_inbox', 'core', 'bootstrap')
    return {"ok": True, "status": "media_deleted"}


@router.get("/api/media/telegram")
async def telegram_media_proxy(
    file_id: str = Query(...),
    kind: str = Query("photo"),
    telegram_user_id: int = Depends(current_user_id),
):
    return await stream_telegram_media(telegram_user_id, file_id, kind)



@router.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    telegram_user_id: int = Depends(current_user_id),
):
    cfg = load_config()
    try:
        saved = await save_upload_file(
            file,
            owner_id=telegram_user_id,
            uploads_dir=UPLOAD_DIR,
            config=cfg,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Не удалось сохранить файл") from exc

    cache_invalidate(telegram_user_id, 'bootstrap', 'media_inbox')
    return {
        "ok": True,
        "url": saved["media_ref"],
        "media_ref": saved["media_ref"],
        "media_type": saved["media_type"],
        "filename": (file.filename or "upload.bin").strip(),
        "stored_name": saved["stored_name"],
        "size": int(saved["size"]),
    }
