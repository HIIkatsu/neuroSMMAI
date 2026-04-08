from __future__ import annotations

import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import db


IMAGE_TYPES = {
    "image/jpeg", "image/png", "image/webp", "image/gif",
}
VIDEO_TYPES = {
    "video/mp4", "video/quicktime", "video/x-matroska", "video/webm",
}
DOCUMENT_TYPES = {
    "application/pdf", "text/plain",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm"}
DOCUMENT_EXTS = {".pdf", ".txt"}


@dataclass
class UploadDecision:
    media_type: str
    suffix: str
    max_bytes: int


def _safe_suffix(filename: str) -> str:
    suffix = Path(filename or "").suffix.lower().strip()
    if not suffix or len(suffix) > 10:
        return ".bin"
    return suffix


def decide_upload(content_type: str, filename: str, *, config) -> UploadDecision:
    ct = (content_type or "").split(";", 1)[0].strip().lower()
    suffix = _safe_suffix(filename)

    if ct in IMAGE_TYPES or suffix in IMAGE_EXTS:
        return UploadDecision("photo", suffix if suffix in IMAGE_EXTS else ".jpg", int(config.upload_image_limit_mb) * 1024 * 1024)

    if ct in VIDEO_TYPES or suffix in VIDEO_EXTS:
        return UploadDecision("video", suffix if suffix in VIDEO_EXTS else ".mp4", int(config.upload_video_limit_mb) * 1024 * 1024)

    if ct in DOCUMENT_TYPES or suffix in DOCUMENT_EXTS:
        return UploadDecision("document", suffix if suffix in DOCUMENT_EXTS else ".bin", int(config.upload_document_limit_mb) * 1024 * 1024)

    raise ValueError("Поддерживаются только изображения, видео и безопасные документы")


async def current_temp_usage_bytes(owner_id: int, *, uploads_dir: Path, generated_dir: Path) -> int:
    refs: set[str] = set()
    for draft in await db.list_drafts(owner_id=owner_id, limit=1000):
        ref = str(draft.get("media_ref") or "").strip()
        if ref:
            refs.add(ref)

    total = 0
    for ref in refs:
        candidate = resolve_local_media_path(ref, uploads_dir=uploads_dir, generated_dir=generated_dir)
        if candidate and candidate.is_file():
            try:
                total += candidate.stat().st_size
            except OSError:
                pass
    return total


def resolve_local_media_path(media_ref: str, *, uploads_dir: Path, generated_dir: Path) -> Path | None:
    value = (media_ref or "").strip()
    if not value or value.startswith("tgfile:"):
        return None
    if value.startswith("/uploads/"):
        return uploads_dir / value.removeprefix("/uploads/")
    if value.startswith("uploads/"):
        return uploads_dir / value.removeprefix("uploads/")
    if value.startswith("/generated-images/"):
        return generated_dir / value.removeprefix("/generated-images/")
    if value.startswith("generated_images/"):
        return generated_dir / value.removeprefix("generated_images/")
    path = Path(value)
    if path.is_absolute() or str(path).startswith("."):
        return path
    return None


async def save_upload_file(upload_file, *, owner_id: int, uploads_dir: Path, config) -> dict:
    decision = decide_upload(upload_file.content_type or "", upload_file.filename or "", config=config)
    quota_bytes = int(config.temp_media_quota_mb_per_user) * 1024 * 1024
    usage = await current_temp_usage_bytes(owner_id, uploads_dir=uploads_dir, generated_dir=uploads_dir.parent / "generated_images")
    if usage >= quota_bytes:
        raise ValueError("Превышена квота временных медиа. Удалите старые черновики или опубликуйте их.")

    stored_name = f"{owner_id}_{uuid.uuid4().hex}{decision.suffix}"
    stored_path = uploads_dir / stored_name
    total = 0
    try:
        with stored_path.open("wb") as out:
            while True:
                chunk = await upload_file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > decision.max_bytes:
                    raise ValueError("Файл слишком большой для выбранного типа медиа")
                if usage + total > quota_bytes:
                    raise ValueError("Превышена квота временных медиа пользователя")
                out.write(chunk)
    except Exception:
        try:
            stored_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise
    finally:
        await upload_file.close()

    return {
        "stored_name": stored_name,
        "stored_path": stored_path,
        "size": total,
        "media_type": decision.media_type,
        "media_ref": f"/uploads/{stored_name}",
    }


def _iter_files(*dirs: Path) -> Iterable[Path]:
    for directory in dirs:
        if not directory.exists():
            continue
        for item in directory.iterdir():
            if item.is_file():
                yield item


async def cleanup_temp_media(*, uploads_dir: Path, generated_dir: Path, config) -> dict:
    import time as _time
    import logging as _logging
    _log = _logging.getLogger(__name__)

    # Batch fetch all active draft media refs in a single query instead of N+1
    active_refs: set[Path] = set()
    if hasattr(db, 'list_all_active_draft_media_refs'):
        all_drafts = await db.list_all_active_draft_media_refs()
    else:
        # Fallback: db.list_all_active_draft_media_refs not available yet
        all_drafts = []
        for owner_id in await db.list_owner_ids():
            for draft in await db.list_drafts(owner_id=owner_id, limit=1000):
                status = str(draft.get("status") or "draft").strip().lower()
                if status not in {"draft", "publishing"}:
                    continue
                all_drafts.append(str(draft.get("media_ref") or ""))

    for ref_val in all_drafts:
        ref_str = str(ref_val or "").strip()
        if not ref_str:
            continue
        path = resolve_local_media_path(ref_str, uploads_dir=uploads_dir, generated_dir=generated_dir)
        if path:
            try:
                active_refs.add(path.resolve())
            except OSError:
                pass

    removed = 0
    now = _time.time()

    for file in _iter_files(uploads_dir, generated_dir):
        try:
            resolved = file.resolve()
            if resolved in active_refs:
                continue
            age_hours = (now - file.stat().st_mtime) / 3600.0
            suffix = file.suffix.lower()
            if file.parent == generated_dir:
                retention = int(config.generated_media_retention_hours)
            elif suffix in IMAGE_EXTS:
                retention = int(config.uploaded_photo_retention_hours)
            elif suffix in VIDEO_EXTS:
                retention = int(config.uploaded_video_retention_hours)
            else:
                retention = int(config.uploaded_document_retention_hours)
            if age_hours < retention:
                continue
            file.unlink(missing_ok=True)
            removed += 1
        except OSError as e:
            _log.warning("cleanup_temp_media: failed to process %s: %s", file, e)
        except Exception as e:
            _log.warning("cleanup_temp_media: unexpected error for %s: %s", file, e)
    return {"removed": removed}
