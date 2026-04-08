from aiogram import F, Router
from aiogram.types import CallbackQuery

from handlers_private_messages import TARIFFS_TEXT, _subscription_keyboard
from safe_send import answer_plain

router = Router()


@router.callback_query(F.data == "show_tariffs")
async def cb_show_tariffs(callback: CallbackQuery):
    await callback.answer()
    cfg = callback.message.bot._dp["config"]  # type: ignore[attr-defined]
    kb = _subscription_keyboard(cfg)
    await callback.message.answer(TARIFFS_TEXT, parse_mode="HTML", reply_markup=kb)
