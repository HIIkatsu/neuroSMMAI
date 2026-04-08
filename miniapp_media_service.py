from __future__ import annotations

import io
import mimetypes
from typing import Any
from urllib.parse import parse_qs, urlparse

import aiosqlite
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

import db
from media_lifecycle import save_upload_file
from miniapp_shared import UPLOAD_DIR, cache_invalidate, create_bot, ensure_drafts_capacity


def media_storage_ref(item: dict[str, Any]) -> str:
    kind = str(item.get("kind") or "video").strip()
    file_id = str(item.get("file_id") or "").strip()
    if not file_id:
        return ""
    return f"tgfile:{kind}:{file_id}"


def serialize_media_item(item: dict[str, Any]) -> dict[str, Any]:
    kind = str(item.get("kind") or "video").strip()
    file_id = str(item.get("file_id") or "").strip()
    storage_ref = media_storage_ref(item)
    preview_url = f"/api/media/telegram?kind={kind}&file_id={file_id}" if file_id and kind == "photo" else ""
    title = str(item.get("caption") or item.get("file_name") or ("Фото из чата" if kind == "photo" else "Видео из чата")).strip()
    return {**item, "title": title, "media_ref": storage_ref, "preview_url": preview_url, "thumb_url": preview_url}


async def resolve_pending_media(owner_id: int) -> dict[str, Any] | None:
    pending_id = int(await db.get_setting("pending_media_item_id", owner_id=owner_id) or 0)
    if not pending_id:
        return None
    item = await db.get_user_media(pending_id, owner_id=owner_id)
    if item:
        return serialize_media_item(item)
    await db.set_setting("pending_media_item_id", "0", owner_id=owner_id)
    await db.set_setting("pending_media_kind", "", owner_id=owner_id)
    return None


async def resolve_latest_chat_media(owner_id: int) -> dict[str, Any] | None:
    items = await db.list_user_media(owner_id=owner_id, limit=50)
    for item in items:
        if int(item.get('used_count') or 0) == 0:
            return serialize_media_item(item)
    return None


async def build_media_shortcuts(owner_id: int) -> dict[str, Any]:
    pending_media = await resolve_pending_media(owner_id)
    latest_media = await resolve_latest_chat_media(owner_id)
    return {"pending_media": pending_media, "latest_media": latest_media, "has_pending_media": bool(pending_media)}


def guess_media_type_from_kind(kind: str) -> str:
    return "photo" if str(kind or "").strip() == "photo" else "video"


def extract_telegram_file_id_from_media_ref(media_ref: str, kind: str | None = None) -> str:
    raw = str(media_ref or '').strip()
    if not raw:
        return ''
    if raw.startswith('tgfile:'):
        parts = raw.split(':')
        if len(parts) >= 3 and (kind is None or parts[1] == kind):
            return ':'.join(parts[2:]).strip()
        return ''
    if raw.startswith('/api/media/telegram'):
        parsed = urlparse(raw)
        qs = parse_qs(parsed.query)
        file_id = str((qs.get('file_id') or [''])[0]).strip()
        ref_kind = str((qs.get('kind') or [''])[0]).strip()
        if file_id and (kind is None or not ref_kind or ref_kind == kind):
            return file_id
    return ''


async def find_inbox_item_for_video_media(owner_id: int, media_ref: str) -> dict[str, Any] | None:
    file_id = extract_telegram_file_id_from_media_ref(media_ref, 'video')
    if not file_id:
        return None
    for item in await db.list_user_media(owner_id=owner_id, limit=100, kind='video'):
        if str(item.get('file_id') or '').strip() == file_id:
            return item
    return None


async def find_inbox_item_for_media(owner_id: int, media_ref: str) -> dict[str, Any] | None:
    file_id = extract_telegram_file_id_from_media_ref(media_ref, None)
    if not file_id:
        return None
    for item in await db.list_user_media(owner_id=owner_id, limit=200):
        if str(item.get('file_id') or '').strip() == file_id:
            return item
    return None


async def delete_media_inbox_item(owner_id: int, item_id: int) -> None:
    if hasattr(db, 'delete_user_media'):
        await db.delete_user_media(int(item_id), owner_id=owner_id)
    else:
        async with aiosqlite.connect(db.DB_PATH) as conn:
            await conn.execute(
                "DELETE FROM user_media_inbox WHERE id=? AND owner_id=?",
                (int(item_id), int(owner_id)),
            )
            await conn.commit()


async def stream_telegram_media(owner_id: int, file_id: str, kind: str) -> StreamingResponse:
    items = await db.list_user_media(owner_id=owner_id, limit=200)
    allowed_ids = {str(item.get("file_id") or "").strip() for item in items if item.get("file_id")}
    if str(file_id).strip() not in allowed_ids:
        raise HTTPException(status_code=404, detail="Файл не найден")

    bot = await create_bot()
    try:
        tg_file = await bot.get_file(file_id)
        if not tg_file.file_path:
            raise HTTPException(status_code=404, detail="Не удалось получить файл Telegram")
        buffer = io.BytesIO()
        await bot.download_file(tg_file.file_path, destination=buffer)
        buffer.seek(0)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Не удалось загрузить файл из Telegram") from exc
    finally:
        await bot.session.close()

    media_kind = str(kind or "").strip()
    content_type = (
        "image/jpeg" if media_kind == "photo" else
        "video/mp4" if media_kind in {"video", "document_video"} else
        mimetypes.guess_type(str(tg_file.file_path or ""))[0] or "application/octet-stream"
    )
    return StreamingResponse(buffer, media_type=content_type)
