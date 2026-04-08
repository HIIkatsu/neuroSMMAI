from __future__ import annotations

from dataclasses import dataclass

from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest


class ChannelAccessError(Exception):
    def __init__(self, message: str, status_code: int = 403):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@dataclass
class ChannelAccess:
    channel_id: int
    title: str
    channel_target: str


async def verify_channel_access(bot: Bot, telegram_user_id: int, raw_channel_target: str) -> ChannelAccess:
    channel_target = (raw_channel_target or "").strip()
    if not channel_target:
        raise ChannelAccessError("Укажи канал: @username или -100...")

    try:
        chat = await bot.get_chat(channel_target)
    except TelegramBadRequest:
        raise ChannelAccessError("Не удалось открыть канал. Проверь @username или numeric id канала.", status_code=400)

    if getattr(chat, "type", None) != "channel":
        raise ChannelAccessError("Разрешены только Telegram-каналы", status_code=400)

    channel_id = int(getattr(chat, "id", 0) or 0)
    if not channel_id:
        raise ChannelAccessError("Не удалось определить channel_id", status_code=400)

    try:
        user_member = await bot.get_chat_member(channel_id, telegram_user_id)
    except TelegramBadRequest:
        raise ChannelAccessError("Вы не являетесь администратором этого канала или канал недоступен для проверки.")
    if getattr(user_member, "status", None) not in {"creator", "administrator"}:
        raise ChannelAccessError("Вы не являетесь администратором этого канала.")

    me = await bot.get_me()
    try:
        bot_member = await bot.get_chat_member(channel_id, me.id)
    except TelegramBadRequest:
        raise ChannelAccessError("Бот не является администратором этого канала или не может проверить свои права.")
    if getattr(bot_member, "status", None) != "administrator":
        raise ChannelAccessError("Бот не является администратором этого канала.")

    can_post = getattr(bot_member, "can_post_messages", None)
    can_edit = getattr(bot_member, "can_edit_messages", None)
    if can_post is False and can_edit is False:
        raise ChannelAccessError("У бота нет прав на публикацию в этом канале.")

    raw_title = (getattr(chat, "title", None) or "").strip()
    # Only use channel_target as fallback if it's a human-readable @username, not raw numeric ID
    if raw_title:
        title = raw_title
    elif channel_target and not channel_target.lstrip("@").lstrip("-").isdigit():
        title = channel_target
    else:
        title = ""  # let downstream enrich_display_label handle the "Канал без названия" fallback
    return ChannelAccess(channel_id=channel_id, title=title, channel_target=str(channel_id))
