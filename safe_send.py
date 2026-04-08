from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from aiogram.exceptions import TelegramBadRequest, TelegramRetryAfter
from aiogram.types import FSInputFile

logger = logging.getLogger(__name__)


RETRYABLE_PHRASES = (
    "ClientOSError",
    "Can not write request body",
    "Connection reset",
    "Timeout",
)

_MAX_RETRY_AFTER_SECONDS = 300


def _extract_retry_after(exc: Exception) -> float | None:
    """Parse retry_after seconds from a TelegramRetryAfter or a 429 response."""
    if isinstance(exc, TelegramRetryAfter):
        try:
            return max(1.0, float(exc.retry_after))
        except Exception:
            return 30.0
    raw = repr(exc)
    if "429" in raw or "Too Many Requests" in raw or "retry_after" in raw.lower():
        import re
        m = re.search(r"retry_after['\"]?\s*[:=]\s*(\d+)", raw, re.IGNORECASE)
        if m:
            return max(1.0, float(m.group(1)))
        return 30.0
    return None


def _clone_input(value: Any):
    if isinstance(value, FSInputFile):
        path = getattr(value, "path", "") or ""
        filename = getattr(value, "filename", None)
        if path and os.path.isfile(path):
            return FSInputFile(path, filename=filename or os.path.basename(path) or "upload.bin")
    return value


def _refresh_kwargs(kwargs: dict):
    refreshed = dict(kwargs or {})
    for key in ("thumbnail", "thumb", "cover", "photo", "video", "document"):
        if key in refreshed:
            refreshed[key] = _clone_input(refreshed[key])
    return refreshed


def _is_retryable_error(exc: Exception) -> bool:
    raw = repr(exc)
    return any(part in raw for part in RETRYABLE_PHRASES)


async def _handle_rate_limit(exc: Exception, attempt: int, label: str) -> bool:
    """Return True if caller should retry after sleeping, False to give up."""
    retry_after = _extract_retry_after(exc)
    if retry_after is not None:
        wait = min(retry_after, _MAX_RETRY_AFTER_SECONDS)
        logger.warning("%s: 429 rate limit, retry_after=%.0fs, waiting %.0fs (attempt=%d)", label, retry_after, wait, attempt + 1)
        await asyncio.sleep(wait)
        return True
    if _is_retryable_error(exc) and attempt < 2:
        await asyncio.sleep(1.5 * (attempt + 1))
        return True
    return False


async def safe_send(bot, chat_id: int | str, text: str, reply_markup=None, **kwargs):
    for attempt in range(5):
        try:
            return await bot.send_message(
                chat_id,
                text,
                reply_markup=reply_markup,
                request_timeout=300,
                **kwargs,
            )
        except TelegramBadRequest as e:
            logger.warning("safe_send Telegram error: %s", e)
            return False
        except Exception as e:
            logger.warning("safe_send error: %r attempt=%d", e, attempt + 1)
            if await _handle_rate_limit(e, attempt, "safe_send"):
                continue
            return False
    return False


async def safe_send_photo(bot, chat_id: int | str, photo, caption: str, reply_markup=None, **kwargs):
    for attempt in range(5):
        try:
            current_photo = _clone_input(photo)
            current_kwargs = _refresh_kwargs(kwargs)
            return await bot.send_photo(
                chat_id,
                photo=current_photo,
                caption=caption,
                reply_markup=reply_markup,
                request_timeout=300,
                **current_kwargs,
            )
        except TelegramBadRequest as e:
            logger.warning("safe_send_photo Telegram error: %s", e)
            return False
        except Exception as e:
            logger.warning("safe_send_photo error: %r attempt=%d", e, attempt + 1)
            if await _handle_rate_limit(e, attempt, "safe_send_photo"):
                continue
            return False
    return False


async def answer_plain(message, text: str, reply_markup=None):
    try:
        await message.answer(text, reply_markup=reply_markup)
    except TelegramBadRequest as e:
        logger.warning("answer_plain Telegram error: %s", e)
    except Exception as e:
        logger.warning("answer_plain error: %r", e)


async def safe_send_video(bot, chat_id: int | str, video, caption: str, reply_markup=None, **kwargs):
    for attempt in range(5):
        try:
            return await bot.send_video(
                chat_id,
                video=_clone_input(video),
                caption=caption,
                reply_markup=reply_markup,
                request_timeout=900,
                **_refresh_kwargs(kwargs),
            )
        except TelegramBadRequest as e:
            logger.warning("safe_send_video Telegram error: %s", e)
            return False
        except Exception as e:
            logger.warning("safe_send_video error: %r attempt=%d", e, attempt + 1)
            if await _handle_rate_limit(e, attempt, "safe_send_video"):
                continue
            return False
    return False


async def safe_send_document(bot, chat_id: int | str, document, caption: str, reply_markup=None, **kwargs):
    for attempt in range(5):
        try:
            return await bot.send_document(
                chat_id,
                document=_clone_input(document),
                caption=caption,
                reply_markup=reply_markup,
                request_timeout=900,
                **_refresh_kwargs(kwargs),
            )
        except TelegramBadRequest as e:
            logger.warning("safe_send_document Telegram error: %s", e)
            return False
        except Exception as e:
            logger.warning("safe_send_document error: %r attempt=%d", e, attempt + 1)
            if await _handle_rate_limit(e, attempt, "safe_send_document"):
                continue
            return False
    return False
