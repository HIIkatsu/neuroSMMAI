
from __future__ import annotations

import asyncio
import logging
from typing import Any

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

import db
from safe_send import answer_plain

logger = logging.getLogger(__name__)

router = Router()


def _admin_ids(message: Message | CallbackQuery) -> set[int]:
    obj = message if isinstance(message, Message) else message.message
    if obj is None:
        return set()
    cfg = obj.bot._dp["config"]  # type: ignore[attr-defined]
    return set(getattr(cfg, "admin_ids", set()) or set())


def _is_admin(event: Message | CallbackQuery) -> bool:
    user = event.from_user
    uid = user.id if user else 0
    return uid in _admin_ids(event)


def _admin_dashboard_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="📊 Статистика", callback_data="admin:stats")],
        [InlineKeyboardButton(text="🎁 Выдать тариф", callback_data="admin:grant_tier")],
        [InlineKeyboardButton(text="📢 Рассылка", callback_data="admin:broadcast")],
    ])


def _tier_keyboard(prefix: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🆓 Free", callback_data=f"{prefix}:free")],
        [InlineKeyboardButton(text="⭐ Pro", callback_data=f"{prefix}:pro")],
        [InlineKeyboardButton(text="💎 Max", callback_data=f"{prefix}:max")],
        [InlineKeyboardButton(text="◀️ Назад", callback_data="admin:back")],
    ])


# ----- FSM states -----

class AdminStates(StatesGroup):
    waiting_grant_user_id = State()
    waiting_grant_tier = State()
    waiting_broadcast_text = State()


# ----- /admin command -----

@router.message(Command("admin"))
async def cmd_admin(message: Message):
    if not _is_admin(message):
        return  # silently ignore for non-admins
    await message.answer(
        "🛡 <b>Панель администратора</b>\n\nВыберите действие:",
        parse_mode="HTML",
        reply_markup=_admin_dashboard_keyboard(),
    )


# ----- /bind_here (existing feature) -----

@router.message(F.text == "/bind_here")
async def bind_here(message: Message):
    if not _is_admin(message):
        await answer_plain(message, "⛔ Только для админа.")
        return
    chat_id = str(message.chat.id)
    await db.set_setting("bound_chat", chat_id)
    await answer_plain(message, f"✅ Чат привязан: {chat_id}")


@router.message(F.chat.type.in_({"group", "supergroup"}))
async def any_group_message(message: Message):
    return


# ----- Callbacks: stats -----

@router.callback_query(F.data == "admin:stats")
async def cb_admin_stats(callback: CallbackQuery):
    if not _is_admin(callback):
        await callback.answer("⛔ Нет доступа", show_alert=True)
        return
    await callback.answer()
    try:
        stats = await db.get_admin_stats()
    except Exception as exc:
        logger.exception("admin stats error")
        await callback.message.answer(f"Ошибка при получении статистики: {exc}")
        return

    text = (
        "📊 <b>Статистика бота</b>\n\n"
        f"👥 Всего пользователей: <b>{stats['total_users']}</b>\n"
        f"🆓 Free: <b>{stats['free']}</b>\n"
        f"⭐ Pro: <b>{stats['pro']}</b>\n"
        f"💎 Max: <b>{stats['max']}</b>"
    )
    await callback.message.answer(text, parse_mode="HTML", reply_markup=_admin_dashboard_keyboard())


# ----- Callbacks: grant tier -----

@router.callback_query(F.data == "admin:grant_tier")
async def cb_admin_grant_tier(callback: CallbackQuery, state: FSMContext):
    if not _is_admin(callback):
        await callback.answer("⛔ Нет доступа", show_alert=True)
        return
    await callback.answer()
    await state.set_state(AdminStates.waiting_grant_user_id)
    await callback.message.answer(
        "🎁 <b>Выдать тариф</b>\n\nВведите Telegram ID пользователя (число):",
        parse_mode="HTML",
    )


@router.message(AdminStates.waiting_grant_user_id)
async def admin_got_user_id(message: Message, state: FSMContext):
    if not _is_admin(message):
        return
    raw = (message.text or "").strip()
    if not raw.lstrip("-").isdigit():
        await message.answer("❌ Введите числовой Telegram ID:")
        return
    await state.update_data(grant_user_id=int(raw))
    await state.set_state(AdminStates.waiting_grant_tier)
    await message.answer(
        f"Выберите тариф для пользователя <code>{raw}</code>:",
        parse_mode="HTML",
        reply_markup=_tier_keyboard("admin:set_tier"),
    )


@router.callback_query(F.data.startswith("admin:set_tier:"))
async def cb_admin_set_tier(callback: CallbackQuery, state: FSMContext):
    if not _is_admin(callback):
        await callback.answer("⛔ Нет доступа", show_alert=True)
        return
    tier = callback.data.split(":")[-1]
    data = await state.get_data()
    user_id = data.get("grant_user_id")
    await state.clear()
    if not user_id:
        await callback.answer("Нет данных о пользователе. Начни заново.", show_alert=True)
        return
    try:
        await db.set_user_subscription(int(user_id), tier)
        # Invalidate cached bootstrap so the Mini App picks up the new tier immediately
        try:
            from miniapp_shared import cache_invalidate
            cache_invalidate(int(user_id))
        except Exception:
            pass  # cache module may not be available in all contexts
    except Exception as exc:
        logger.exception("admin set tier error")
        await callback.answer(f"Ошибка: {exc}", show_alert=True)
        return
    tier_label = {"free": "Free 🆓", "pro": "Pro ⭐", "max": "Max 💎"}.get(tier, tier)
    await callback.answer(f"✅ Тариф {tier_label} выдан", show_alert=True)
    await callback.message.answer(
        f"✅ Пользователю <code>{user_id}</code> назначен тариф <b>{tier_label}</b>.",
        parse_mode="HTML",
        reply_markup=_admin_dashboard_keyboard(),
    )


# ----- Callbacks: broadcast -----

@router.callback_query(F.data == "admin:broadcast")
async def cb_admin_broadcast(callback: CallbackQuery, state: FSMContext):
    if not _is_admin(callback):
        await callback.answer("⛔ Нет доступа", show_alert=True)
        return
    await callback.answer()
    await state.set_state(AdminStates.waiting_broadcast_text)
    await callback.message.answer(
        "📢 <b>Рассылка</b>\n\nОтправьте текст сообщения для рассылки всем пользователям бота.\n\n"
        "Поддерживается HTML-разметка. Для отмены напишите /cancel.",
        parse_mode="HTML",
    )


@router.message(F.text == "/cancel")
async def cmd_cancel(message: Message, state: FSMContext):
    current = await state.get_state()
    if current:
        await state.clear()
        await message.answer("❌ Действие отменено.", reply_markup=None)


@router.message(AdminStates.waiting_broadcast_text)
async def admin_got_broadcast_text(message: Message, state: FSMContext):
    if not _is_admin(message):
        return
    text = message.text or message.caption or ""
    if not text.strip():
        await message.answer("❌ Пустое сообщение. Введите текст рассылки:")
        return
    await state.clear()

    user_ids = await db.get_all_user_ids()
    total = len(user_ids)
    sent = 0
    failed = 0

    status_msg = await message.answer(
        f"📤 Начинаю рассылку на <b>{total}</b> пользователей…",
        parse_mode="HTML",
    )

    for uid in user_ids:
        try:
            await message.bot.send_message(uid, text, parse_mode="HTML")
            sent += 1
        except Exception as exc:
            failed += 1
            logger.debug("broadcast: failed to send to uid=%s: %s", uid, exc)
        await asyncio.sleep(0.1)  # ~10 msg/sec, safely below Telegram's rate limit

    try:
        await status_msg.edit_text(
            f"✅ Рассылка завершена.\n"
            f"📤 Отправлено: <b>{sent}</b> / {total}\n"
            f"❌ Не доставлено: <b>{failed}</b>",
            parse_mode="HTML",
        )
    except Exception:
        pass

    await message.answer("Вернуться в меню:", reply_markup=_admin_dashboard_keyboard())


# ----- Back button -----

@router.callback_query(F.data == "admin:back")
async def cb_admin_back(callback: CallbackQuery, state: FSMContext):
    await state.clear()
    await callback.answer()
    await callback.message.answer(
        "🛡 <b>Панель администратора</b>\n\nВыберите действие:",
        parse_mode="HTML",
        reply_markup=_admin_dashboard_keyboard(),
    )

