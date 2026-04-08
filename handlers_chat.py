from __future__ import annotations

from aiogram import Router, F
from aiogram.types import Message

from db import set_setting, get_setting
from safe_send import answer_plain

router = Router()


def _admin_ids(message: Message) -> set[int]:
    cfg = message.bot._dp["config"]  # type: ignore[attr-defined]
    return set(getattr(cfg, "admin_ids", set()) or set())


def is_admin(message: Message) -> bool:
    uid = message.from_user.id if message.from_user else 0
    return uid in _admin_ids(message)


@router.message(F.text == "/bind_here")
async def bind_here(message: Message):
    # только админ может привязать чат
    if not is_admin(message):
        await answer_plain(message, "⛔ Только для админа.")
        return

    chat_id = str(message.chat.id)
    await set_setting("bound_chat", chat_id)
    await answer_plain(message, f"✅ Чат привязан: {chat_id}\nТеперь я могу отвечать/реагировать в этом чате (если включишь режим).")


@router.message(F.chat.type.in_({"group", "supergroup"}))
async def any_group_message(message: Message):
    # сейчас не делаем умные ответы в группе (чтобы не спамил)
    # позже добавим режим “отвечать в группе только при упоминании”
    return