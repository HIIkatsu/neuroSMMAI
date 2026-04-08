from __future__ import annotations

import io
import logging
import re
from datetime import datetime as _dt
from typing import Callable, Awaitable

from aiogram import F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    Message,
    PreCheckoutQuery,
    ReplyKeyboardMarkup,
    WebAppInfo,
)

from ai_client import ai_chat, whisper_transcribe
from billing_service import process_successful_payment, send_stars_invoice, create_yookassa_payment, RUB_PRICES, TIER_LABELS
from content import generate_post_bundle
from db import get_settings_bulk, list_drafts, list_plan_items, list_schedules, get_setting, create_draft, get_user_subscription, TIER_PRO, TIER_MAX, TIER_FREE, is_generation_allowed, FREE_TIER_GENERATIONS_LIMIT, increment_generations_used, is_feature_allowed, increment_feature_used
from safe_send import answer_plain

router = Router()
logger = logging.getLogger(__name__)

QUICK_STATUS_LABEL = "📊 Статус"
QUICK_HELP_LABEL = "❓ Что умеет бот"
QUICK_APP_LABEL = "📱 Mini App"
QUICK_CHANNEL_LABEL = "🔗 Как подключить канал"
QUICK_PUBLISH_LABEL = "🚀 Почему не публикуется"

_FALSE_VALUES = {"0", "false", "False", "no", "No", "off", "Off"}


def _owner_id(message: Message) -> int:
    return int(message.from_user.id) if message.from_user else 0


def _config(message: Message):
    return message.bot._dp["config"]  # type: ignore[attr-defined]


def _miniapp_url(message: Message) -> str:
    cfg = _config(message)
    return (getattr(cfg, "miniapp_url", "") or "").strip()


def _is_enabled(value: str | None, default: bool = False) -> bool:
    raw = str(value if value is not None else ("1" if default else "0")).strip()
    return raw not in _FALSE_VALUES


def _main_keyboard() -> ReplyKeyboardMarkup:
    rows = [
        [KeyboardButton(text=QUICK_APP_LABEL)],
        [
            KeyboardButton(text=QUICK_STATUS_LABEL),
            KeyboardButton(text=QUICK_HELP_LABEL),
        ],
        [
            KeyboardButton(text=QUICK_CHANNEL_LABEL),
            KeyboardButton(text=QUICK_PUBLISH_LABEL),
        ],
    ]
    return ReplyKeyboardMarkup(
        keyboard=rows,
        resize_keyboard=True,
        is_persistent=True,
        input_field_placeholder="Спроси про функции бота или нажми кнопку",
    )


def _miniapp_inline_keyboard(message: Message, *, label: str = "Открыть Mini App") -> InlineKeyboardMarkup | None:
    app_url = _miniapp_url(message)
    if not app_url:
        return None
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=label, web_app=WebAppInfo(url=app_url))],
        ]
    )


def _start_inline_keyboard(message: Message) -> InlineKeyboardMarkup | None:
    """Inline keyboard for /start: Mini App button + Tariffs button."""
    app_url = _miniapp_url(message)
    if not app_url:
        return None
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="🚀 Открыть приложение", web_app=WebAppInfo(url=app_url))],
            [InlineKeyboardButton(text="💎 Тарифы", callback_data="show_tariffs")],
        ]
    )


async def _send_with_app_button(message: Message, text: str, *, app_label: str = "Открыть Mini App") -> None:
    inline_markup = _miniapp_inline_keyboard(message, label=app_label)
    await answer_plain(
        message,
        text,
        reply_markup=inline_markup or _main_keyboard(),
    )


async def _load_context(owner_id: int) -> dict:
    settings = await get_settings_bulk(
        [
            "channel_target",
            "topic",
            "news_enabled",
            "posts_enabled",
            "posting_mode",
            "channel_style",
            "channel_audience",
            "content_formats",
            "content_rubrics",
            "content_constraints",
            "channel_frequency",
            "auto_mode",
        ],
        owner_id=owner_id,
    )
    schedules = await list_schedules(owner_id=owner_id)
    drafts = await list_drafts(owner_id=owner_id, limit=50)
    plan_items = await list_plan_items(limit=100, owner_id=owner_id)
    return {
        "settings": settings,
        "schedules": schedules,
        "drafts": drafts,
        "plan_items": plan_items,
    }


async def _build_status_text(owner_id: int) -> str:
    ctx = await _load_context(owner_id)
    settings = ctx["settings"]
    schedules = ctx["schedules"]
    drafts = ctx["drafts"]
    plan_items = ctx["plan_items"]

    channel = settings.get("channel_target") or "не привязан"
    topic = settings.get("topic") or "не задана"
    news_enabled = _is_enabled(settings.get("news_enabled"), default=False)
    posts_enabled = _is_enabled(settings.get("posts_enabled"), default=True)
    posting_mode = (settings.get("posting_mode") or settings.get("auto_mode") or "manual").strip() or "manual"
    enabled_schedules = sum(1 for x in schedules if int(x.get("enabled") or 1) != 0)
    ready_parts: list[str] = []
    missing_parts: list[str] = []

    if channel != "не привязан":
        ready_parts.append("канал")
    else:
        missing_parts.append("канал")

    if topic != "не задана":
        ready_parts.append("тема")
    else:
        missing_parts.append("тема")

    if enabled_schedules > 0:
        ready_parts.append("ритм")
    else:
        missing_parts.append("ритм")

    if drafts:
        ready_parts.append("черновики")
    else:
        missing_parts.append("черновики")

    readiness = "Готов к работе" if len(missing_parts) <= 1 else "Есть узкие места"

    return (
        "Краткий статус\n\n"
        f"Состояние: {readiness}\n"
        f"Канал: {channel}\n"
        f"Тема: {topic}\n"
        f"Постинг: {'включён' if posts_enabled else 'выключен'}\n"
        f"Авто-новости: {'включены' if news_enabled else 'выключены'}\n"
        f"Режим: {posting_mode}\n"
        f"Черновики: {len(drafts)}\n"
        f"Расписание: {enabled_schedules}\n"
        f"План: {len(plan_items)}\n"
        f"Сильные стороны: {', '.join(ready_parts) if ready_parts else 'пока нет'}\n"
        f"Нужно добить: {', '.join(missing_parts) if missing_parts else 'критичных дыр нет'}\n\n"
        "Если нужно менять конфиг, план или публикации — открой Mini App кнопкой ниже."
    )


HELP_TEXT = (
    "Я ИИ-консультант по NeuroSMM.\n\n"
    "Что делаю в чате:\n"
    "• объясняю функции бота и Mini App\n"
    "• помогаю понять, почему не работает публикация\n"
    "• подсказываю, как подключить канал и включить авто-режим\n"
    "• даю идеи постов и объясняю, как работает контент-план\n"
    "• показываю краткий статус без лишней навигации\n\n"
    "Быстрые команды:\n"
    "• /app — открыть Mini App\n"
    "• /status — краткий статус\n"
    "• /help — что умеет бот\n\n"
    "Можешь писать обычным текстом, например:\n"
    "• как подключить канал\n"
    "• почему не публикуется пост\n"
    "• как включить авто-режим\n"
    "• что мне выложить сегодня\n"
    "• как работает контент-план"
)


CONSULTANT_SYSTEM_PROMPT = (
    "Ты — встроенный ИИ-консультант Telegram-бота NeuroSMM. "
    "Главный интерфейс продукта — Mini App, а чат нужен для консультаций и быстрых действий. "
    "Отвечай коротко, по делу, на русском. "
    "Не используй markdown-оформление: никаких **, __, #, таблиц, code fences и ссылок вида [text](url). "
    "Пиши чистым текстом. Допустимы короткие пункты с символом •. "
    "Нельзя делать вид, что ты выполнил действие, создал пост, поменял настройки или опубликовал что-то, если этого не было. "
    "Если пользователь просит реальное действие, направь его в Mini App и скажи, в каком разделе это находится. "
    "Если вопрос про публикацию, канал, авто-режим или контент-план, давай конкретные шаги проверки, а не общие слова. "
    "Если вопрос про идеи постов, предлагай 3-5 тем, привязанных к теме канала, аудитории и формату. "
    "Обычно отвечай в 5-9 строк."
)


TOPIC_HINTS = {
    QUICK_CHANNEL_LABEL: "Как подключить канал и выдать права боту?",
    QUICK_PUBLISH_LABEL: "Почему может не публиковаться пост и что проверить?",
}


def _strip_markdown(text: str) -> str:
    cleaned = text or ""
    cleaned = cleaned.replace("**", "")
    cleaned = cleaned.replace("__", "")
    cleaned = cleaned.replace("`", "")
    cleaned = re.sub(r"^\s*#+\s?", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1: \2", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def _normalized_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower().replace("ё", "е"))


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    src = _normalized_text(text)
    return any(needle in src for needle in needles)


async def _reply_connect_channel(message: Message, ctx: dict) -> str:
    channel = ctx["settings"].get("channel_target") or ""
    return (
        "Как подключить канал\n\n"
        "1. Добавь бота в админы нужного канала.\n"
        "2. Дай право публиковать сообщения. Для авто-режима этого критично достаточно.\n"
        "3. В Mini App открой раздел Канал и выбери нужный канал.\n"
        "4. После привязки проверь тестовую публикацию через черновик.\n\n"
        f"Сейчас привязка: {channel or 'канал ещё не выбран'}.\n"
        "Если канал не виден, обычно проблема в правах бота или в том, что он не добавлен в канал как админ."
    )


async def _reply_publish_diagnostics(message: Message, ctx: dict) -> str:
    settings = ctx["settings"]
    drafts = ctx["drafts"]
    schedules = ctx["schedules"]

    issues: list[str] = []
    fixes: list[str] = []

    if not (settings.get("channel_target") or "").strip():
        issues.append("канал не привязан")
        fixes.append("в Mini App открой Канал и выбери канал")
    if not _is_enabled(settings.get("posts_enabled"), default=True):
        issues.append("постинг выключен")
        fixes.append("включи постинг в настройках канала")
    if not drafts:
        issues.append("нет черновиков")
        fixes.append("создай или сгенерируй черновик")
    if not any(int(x.get("enabled") or 1) != 0 for x in schedules):
        issues.append("нет активного расписания")
        fixes.append("добавь хотя бы один слот публикации")

    if not issues:
        return (
            "Базовые блокеры не вижу.\n\n"
            "Что проверить дальше:\n"
            "• есть ли у бота права админа в канале\n"
            "• не удалён ли канал после привязки\n"
            "• не упирается ли публикация в конкретный битый черновик или медиа\n"
            "• не выключен ли авто-режим логикой сценария\n\n"
            "Открой Mini App: сначала проверь Канал и Расписание, потом зайди в Черновики и попробуй ручную публикацию конкретного поста."
        )

    issue_block = "\n".join(f"• {x}" for x in issues)
    fix_block = "\n".join(f"• {x}" for x in fixes)
    return (
        "Почему сейчас может не публиковаться\n\n"
        f"Проблемы:\n{issue_block}\n\n"
        f"Что сделать:\n{fix_block}\n\n"
        "Порядок нормальный такой: сначала привязка канала, потом включённый постинг, потом хотя бы один черновик и хотя бы один слот расписания."
    )


async def _reply_autopilot(message: Message, ctx: dict) -> str:
    settings = ctx["settings"]
    enabled_schedules = sum(1 for x in ctx["schedules"] if int(x.get("enabled") or 1) != 0)
    drafts_count = len(ctx["drafts"])
    channel = (settings.get("channel_target") or "").strip()
    topic = (settings.get("topic") or "").strip()
    blockers: list[str] = []
    if not channel:
        blockers.append("не привязан канал")
    if not topic:
        blockers.append("не задана тема")
    if enabled_schedules < 1:
        blockers.append("нет расписания")
    if drafts_count < 1:
        blockers.append("нет запаса черновиков")

    if blockers:
        return (
            "Чтобы включать авто-режим без боли, сначала закрой базу.\n\n"
            f"Сейчас мешает: {', '.join(blockers)}.\n\n"
            "Что нужно:\n"
            "• привязанный канал\n"
            "• тема и базовая стратегия\n"
            "• минимум один активный слот публикации\n"
            "• хотя бы 3–5 нормальных черновиков или план\n\n"
            "Делается это в Mini App по цепочке: Онбординг → Стратегия → План → Черновики → Расписание."
        )

    return (
        "Авто-режим можно включать. База уже собрана.\n\n"
        f"Сейчас есть: канал, тема, {enabled_schedules} слотов и {drafts_count} черновиков.\n\n"
        "Дальше открой Mini App и проверь:\n"
        "• режим публикации\n"
        "• ритм расписания\n"
        "• чтобы в черновиках не лежал мусор\n"
        "• чтобы контент-план не состоял из общих тем без пользы"
    )


async def _reply_content_plan(message: Message, ctx: dict) -> str:
    plan_items = ctx["plan_items"]
    settings = ctx["settings"]
    topic = (settings.get("topic") or "без темы").strip()
    audience = (settings.get("channel_audience") or "аудитория не описана").strip()
    if not plan_items:
        return (
            "Контент-план у тебя пустой или почти пустой. Это плохо для автопилота.\n\n"
            f"Сейчас тема: {topic}. Аудитория: {audience}.\n\n"
            "Нормальная структура плана для NeuroSMM:\n"
            "• практические разборы\n"
            "• частые ошибки\n"
            "• кейсы или ситуации из практики\n"
            "• вовлекающие вопросы без мусорной мотивации\n"
            "• 1–2 более лёгких поста между плотными полезными материалами\n\n"
            "Открой Mini App и сначала собери 10–14 тем, а не 2–3 случайных идеи."
        )

    sample = []
    for item in plan_items[:5]:
        value = (item.get("prompt") or item.get("title") or item.get("idea") or "").strip()
        if value:
            sample.append(value)
    sample_block = "\n".join(f"• {x[:110]}" for x in sample) if sample else "• идеи пока слишком пустые"
    return (
        "По плану уже есть база, но смотри на качество тем, а не только на количество.\n\n"
        f"Сейчас в плане: {len(plan_items)} позиций.\n"
        f"Первые темы:\n{sample_block}\n\n"
        "Сильный план должен чередовать форматы и не повторять один и тот же заход.\n"
        "Если видишь однообразные общие формулировки, их надо переписать до генерации черновиков."
    )


async def _reply_post_ideas(message: Message, ctx: dict) -> str:
    settings = ctx["settings"]
    topic = (settings.get("topic") or "").strip()
    audience = (settings.get("channel_audience") or "").strip()
    formats_raw = (settings.get("content_formats") or settings.get("content_rubrics") or "").strip()
    formats = [x.strip(" •,;|/") for x in re.split(r"[,;\n|/]+", formats_raw) if x.strip(" •,;|/")]
    format_hint = ", ".join(formats[:4]) if formats else "разборы, кейсы, FAQ, короткие наблюдения"

    if not topic:
        return (
            "Сначала задай тему канала в Mini App. Без темы я дам слишком общие идеи, это мусор.\n\n"
            "Открой онбординг или стратегию, заполни тему и аудиторию, потом вернись с вопросом."
        )

    prompt = (
        "Сгенерируй 5 идей постов для Telegram-канала.\n"
        f"Тема канала: {topic}.\n"
        f"Аудитория: {audience or 'не описана'}.\n"
        f"Предпочтительные форматы: {format_hint}.\n"
        "Нужны идеи без воды, без шаблонов и без клишированных заголовков.\n"
        "Для каждой идеи дай: короткий заголовок, зачем это читателю, какой формат лучше взять.\n"
        "Не используй markdown. Пиши чистым текстом, компактно."
    )
    answer = await _consult_llm(message, ctx, prompt, max_tokens=420)
    if answer:
        return answer

    return (
        "Идеи на сейчас\n\n"
        f"1. Частая ошибка в теме «{topic}» — что люди делают не так и как исправить.\n"
        f"2. Кейс или ситуация из практики для аудитории «{audience or 'канала'}».\n"
        f"3. Короткий разбор инструмента, подхода или решения внутри темы «{topic}».\n"
        "4. FAQ-пост с одним острым вопросом, который реально мешает принять решение.\n"
        "5. Пост-сравнение двух подходов без хайпа и без пустой мотивации."
    )


async def _consult_llm(message: Message, ctx: dict, user_text: str, *, max_tokens: int = 450) -> str:
    cfg = _config(message)
    settings = ctx["settings"]
    schedules = ctx["schedules"]
    drafts = ctx["drafts"]
    plan_items = ctx["plan_items"]

    schedule_count = sum(1 for x in schedules if int(x.get("enabled") or 1) != 0)
    sample_plan = []
    for item in plan_items[:4]:
        value = (item.get("prompt") or item.get("title") or item.get("idea") or "").strip()
        if value:
            sample_plan.append(value[:100])

    context_lines = [
        f"channel_target={settings.get('channel_target') or 'не привязан'}",
        f"topic={settings.get('topic') or 'не задана'}",
        f"audience={settings.get('channel_audience') or 'не задана'}",
        f"style={settings.get('channel_style') or 'не задан'}",
        f"formats={settings.get('content_formats') or settings.get('content_rubrics') or 'не заданы'}",
        f"constraints={settings.get('content_constraints') or 'нет'}",
        f"posting_mode={settings.get('posting_mode') or settings.get('auto_mode') or 'manual'}",
        f"news_enabled={settings.get('news_enabled') or '0'}",
        f"posts_enabled={settings.get('posts_enabled') or '1'}",
        f"drafts_count={len(drafts)}",
        f"plan_count={len(plan_items)}",
        f"schedule_count={schedule_count}",
        f"plan_samples={' | '.join(sample_plan) if sample_plan else 'нет'}",
        f"miniapp_url={getattr(cfg, 'miniapp_url', '') or ''}",
    ]

    messages = [
        {"role": "system", "content": CONSULTANT_SYSTEM_PROMPT},
        {
            "role": "system",
            "content": "Текущий контекст пользователя:\n" + "\n".join(context_lines),
        },
        {"role": "user", "content": user_text.strip()},
    ]

    answer = await ai_chat(
        cfg.llm_api_key,
        cfg.llm_model,
        messages,
        temperature=0.25,
        base_url=cfg.llm_base_url,
        max_tokens=min(max_tokens, int(getattr(cfg, "llm_max_tokens", 700) or 700)),
    )
    return _strip_markdown(answer) if answer else ""


async def _consult(message: Message, user_text: str) -> str:
    owner_id = _owner_id(message)
    ctx = await _load_context(owner_id)
    route = _select_route(user_text)
    if route:
        return await route(message, ctx)

    answer = await _consult_llm(message, ctx, user_text)
    if answer:
        return answer

    return (
        "Сейчас ИИ-ответ не сгенерировался.\n\n"
        "Попробуй ещё раз или открой Mini App кнопкой ниже."
    )


RouteHandler = Callable[[Message, dict], Awaitable[str]]


def _select_route(user_text: str) -> RouteHandler | None:
    src = _normalized_text(user_text)

    if _contains_any(src, ("подключ", "привяз", "добавить канал", "как подключить канал", "права боту", "админ")):
        return _reply_connect_channel
    if _contains_any(src, ("не публику", "не отправ", "не выходит пост", "почему не публикуется", "почему не публикуется пост", "публикац")):
        return _reply_publish_diagnostics
    if _contains_any(src, ("авто режим", "авторежим", "автопилот", "auto mode", "как включить авто", "автопост")):
        return _reply_autopilot
    if _contains_any(src, ("контент план", "контент-план", "план контента", "план публикаций", "как работает план")):
        return _reply_content_plan
    if _contains_any(src, ("что мне выложить", "что выложить", "идею поста", "идеи постов", "что опубликовать", "что написать сегодня")):
        return _reply_post_ideas
    return None


@router.message(F.chat.type == "private", CommandStart())
async def cmd_start(message: Message):
    text = (
        "Привет! Я твой личный ИИ-SMMщик. 🤖\n\n"
        "Я сам пишу крутые тексты, ищу новости, подбираю картинки "
        "и публикую их по расписанию. Экономлю до 15 часов рутины в неделю!\n\n"
        "Нажми кнопку ниже, чтобы запустить свой канал на автопилоте бесплатно."
    )
    inline_markup = _start_inline_keyboard(message)
    await answer_plain(
        message,
        text,
        reply_markup=inline_markup or _main_keyboard(),
    )


TARIFFS_TEXT = (
    "💎 <b>Тарифы NeuroSMM</b>\n\n"
    "🆓 <b>Free — бесплатно</b>\n"
    "• 5 генераций постов в месяц\n"
    "• Ручная публикация\n"
    "• Без автопостинга\n\n"
    "🚀 <b>Pro — 490 ₽/мес</b>\n"
    "• Безлимитные генерации\n"
    "• Автопостинг по расписанию\n"
    "• Голос → пост (Voice-to-Post)\n"
    "• ⚡ News Sniper: мониторинг новостей\n\n"
    "💎 <b>Max / Agency — 990 ₽/мес</b>\n"
    "• Всё из Pro, включая News Sniper\n"
    "• 🕵️‍♂️ Шпион конкурентов\n"
    "• Мультиканальность (до 10 каналов)\n"
    "• Приоритетная поддержка\n\n"
    "Выберите тариф и способ оплаты 👇"
)


def _subscription_keyboard(cfg) -> InlineKeyboardMarkup:
    """Inline keyboard with purchase options for Pro and Max."""
    stars_pro = getattr(cfg, "stars_pro_price", 99)
    stars_max = getattr(cfg, "stars_max_price", 249)
    rows = []
    # Pro row
    rows.append([
        InlineKeyboardButton(
            text=f"⭐ Pro — {stars_pro} ⭐",
            callback_data="buy:stars:pro",
        ),
        InlineKeyboardButton(text="💳 Pro — 490 ₽", callback_data="buy:yoo:pro"),
    ])
    # Max row
    rows.append([
        InlineKeyboardButton(
            text=f"💎 Max — {stars_max} ⭐",
            callback_data="buy:stars:max",
        ),
        InlineKeyboardButton(text="💳 Max — 990 ₽", callback_data="buy:yoo:max"),
    ])
    return InlineKeyboardMarkup(inline_keyboard=rows)


@router.message(F.chat.type == "private", Command("tariffs"))
async def cmd_tariffs(message: Message):
    cfg = _config(message)
    kb = _subscription_keyboard(cfg)
    try:
        await message.answer(TARIFFS_TEXT, parse_mode="HTML", reply_markup=kb)
    except Exception:
        await answer_plain(message, TARIFFS_TEXT, reply_markup=_main_keyboard())


@router.message(F.chat.type == "private", Command("plan", "subscription"))
async def cmd_plan(message: Message):
    """Show current subscription status and available upgrade options."""
    owner_id = _owner_id(message)
    cfg = _config(message)
    try:
        sub = await get_user_subscription(owner_id)
        tier = sub.get("subscription_tier", "free")
        expires = sub.get("subscription_expires_at", "")
        tier_labels = {"free": "🆓 Free", "pro": "⭐ Pro", "max": "💎 Max"}
        tier_label = tier_labels.get(tier, tier.capitalize())
        if expires:
            try:
                exp_dt = _dt.fromisoformat(expires)
                expires_str = exp_dt.strftime("%d.%m.%Y")
            except Exception:
                expires_str = expires
            status_line = f"Ваш тариф: <b>{tier_label}</b> (до {expires_str})"
        else:
            status_line = f"Ваш тариф: <b>{tier_label}</b>"
        header = f"📋 <b>Подписка NeuroSMM</b>\n\n{status_line}\n\n"
        kb = _subscription_keyboard(cfg)
        await message.answer(header + TARIFFS_TEXT, parse_mode="HTML", reply_markup=kb)
    except Exception:
        await answer_plain(message, TARIFFS_TEXT, reply_markup=_main_keyboard())


# ---- Callback: buy via Stars or YooKassa ----

@router.callback_query(F.data.startswith("buy:"))
async def cb_buy(callback: CallbackQuery):
    """Handle buy:stars:pro / buy:stars:max / buy:yoo:pro / buy:yoo:max callbacks."""
    await callback.answer()
    parts = (callback.data or "").split(":")
    if len(parts) < 3:
        return
    method = parts[1]   # 'stars' or 'yoo'
    tier = parts[2]     # 'pro' or 'max'
    if tier not in (TIER_PRO, TIER_MAX):
        return
    owner_id = callback.from_user.id if callback.from_user else 0
    cfg = callback.message.bot._dp["config"]  # type: ignore[attr-defined]

    if method == "stars":
        try:
            await send_stars_invoice(callback.message.bot, owner_id, tier)
        except Exception as exc:
            logger.exception("[PAYMENT:INVOICE_ERROR] method=stars owner_id=%s tier=%s", owner_id, tier)
            await callback.message.answer(f"⚠️ Не удалось создать счёт: {exc}")
    elif method == "yoo":
        try:
            result = await create_yookassa_payment(tier, owner_id)
            url = result.get("confirmation_url", "")
            if url:
                amount_rub = RUB_PRICES.get(tier, 0)
                tier_label = TIER_LABELS.get(tier, tier.capitalize())
                pay_keyboard = InlineKeyboardMarkup(inline_keyboard=[[
                    InlineKeyboardButton(
                        text=f"💳 Оплатить {amount_rub} ₽",
                        url=url,
                    )
                ]])
                await callback.message.answer(
                    f"💎 <b>Оплата тарифа {tier_label}</b>\n\n"
                    "Нажмите кнопку ниже, чтобы перейти к оплате.\n"
                    "После успешной оплаты тариф активируется автоматически.",
                    parse_mode="HTML",
                    reply_markup=pay_keyboard,
                )
            else:
                logger.warning("[PAYMENT:NO_URL] method=yookassa owner_id=%s tier=%s", owner_id, tier)
                await callback.message.answer("⚠️ Не удалось получить ссылку на оплату.")
        except RuntimeError as exc:
            await callback.message.answer(f"⚠️ {exc}")
        except Exception as exc:
            logger.exception("[PAYMENT:INVOICE_ERROR] method=yookassa owner_id=%s tier=%s", owner_id, tier)
            await callback.message.answer(
                "⚠️ Ошибка при создании платежа. Попробуйте позже или "
                "обратитесь в поддержку."
            )


# ---- Telegram Stars: pre_checkout and successful_payment ----

@router.pre_checkout_query()
async def handle_pre_checkout(query: PreCheckoutQuery):
    """Always approve pre-checkout for Stars payments."""
    logger.info(
        "[PAYMENT:PRE_CHECKOUT] user_id=%s payload=%s currency=%s total=%s",
        query.from_user.id if query.from_user else 0,
        query.invoice_payload,
        query.currency,
        query.total_amount,
    )
    await query.answer(ok=True)


@router.message(F.successful_payment)
async def handle_successful_payment(message: Message):
    """Process successful Telegram Stars payment and activate subscription."""
    payment = message.successful_payment
    if not payment:
        return
    owner_id = _owner_id(message)
    payload = payment.invoice_payload or ""
    # payload format: "tier:pro" or "tier:max"
    if payload.startswith("tier:"):
        tier = payload.split(":", 1)[1]
    else:
        tier = payload

    # Extract payment details for audit trail
    telegram_payment_id = payment.telegram_payment_charge_id or ""
    stars_amount = str(payment.total_amount or "")
    currency = payment.currency or "XTR"

    try:
        activated = await process_successful_payment(
            owner_id, tier,
            payment_id=telegram_payment_id,
            method="stars",
            amount=stars_amount,
            currency=currency,
        )
        tier_labels = {TIER_PRO: "Pro ⭐", TIER_MAX: "Max 💎"}
        tier_label = tier_labels.get(tier, tier.capitalize())
        if activated:
            await answer_plain(
                message,
                f"🎉 Оплата прошла успешно! Тариф <b>{tier_label}</b> активирован на 30 дней.",
                reply_markup=_main_keyboard(),
            )
            logger.info("[PAYMENT:USER_NOTIFIED] method=stars owner_id=%s tier=%s", owner_id, tier)
        else:
            # Duplicate payment — still confirm to user gracefully
            await answer_plain(
                message,
                f"✅ Тариф <b>{tier_label}</b> уже активирован. Приятного использования!",
                reply_markup=_main_keyboard(),
            )
    except Exception:
        logger.exception(
            "[PAYMENT:HANDLER_ERROR] method=stars owner_id=%s payload=%s",
            owner_id, payload,
        )
        await answer_plain(
            message,
            "⚠️ Оплата получена, но при активации тарифа возникла ошибка. "
            "Обратитесь в поддержку — мы обязательно поможем.",
            reply_markup=_main_keyboard(),
        )




@router.message(F.chat.type == "private", Command("help"))
async def cmd_help(message: Message):
    await _send_with_app_button(message, HELP_TEXT)


@router.message(F.chat.type == "private", Command("app"))
async def cmd_app(message: Message):
    app_url = _miniapp_url(message)
    if app_url:
        await _send_with_app_button(
            message,
            "Открой Mini App кнопкой ниже. Там вся основная работа: онбординг, стратегия, план, черновики, расписание, публикация и аналитика.",
            app_label="Открыть панель",
        )
        return
    await answer_plain(message, "Mini App URL не настроен в конфиге. Проверь MINIAPP_URL.", reply_markup=_main_keyboard())


@router.message(F.chat.type == "private", Command("status"))
async def cmd_status(message: Message):
    await _send_with_app_button(message, await _build_status_text(_owner_id(message)))


@router.message(F.chat.type == "private", F.voice)
async def handle_voice_message(message: Message):
    """Voice-to-Post: transcribe voice note and generate a draft."""
    owner_id = _owner_id(message)
    cfg = _config(message)

    # --- Tier check: Voice-to-Post uses per-feature quota ---
    # Free users get a small trial allowance (e.g. 1/month);
    # Pro/Max get unlimited.
    allowed, sub, used, limit = await is_feature_allowed(owner_id, "voice")
    if not allowed:
        limit_str = f" ({used}/{limit} в этом месяце)" if limit else ""
        await answer_plain(
            message,
            f"🔒 Лимит Voice-to-Post исчерпан{limit_str}.\n"
            "Перейди на тариф Pro для безлимитного доступа — /tariffs",
            reply_markup=_main_keyboard(),
        )
        return

    api_key = getattr(cfg, "llm_api_key", "") or ""
    miniapp_url = _miniapp_url(message)

    await answer_plain(message, "🎙️ Обрабатываю голосовое сообщение…")

    try:
        voice = message.voice
        bot = message.bot
        file = await bot.get_file(voice.file_id)
        buf = io.BytesIO()
        await bot.download_file(file.file_path, destination=buf)
        audio_bytes = buf.getvalue()
    except Exception:
        logger.exception("handle_voice_message: failed to download voice file owner_id=%s", owner_id)
        await answer_plain(message, "⚠️ Не удалось скачать голосовое. Попробуй ещё раз.", reply_markup=_main_keyboard())
        return

    transcript = await whisper_transcribe(api_key, audio_bytes, filename="voice.ogg")
    if not transcript:
        await answer_plain(
            message,
            "⚠️ Не удалось распознать голос. Убедись, что микрофон записал чётко, или напиши текст вручную.",
            reply_markup=_main_keyboard(),
        )
        return

    # Generate post draft from transcribed text
    topic = (await get_setting("topic", owner_id=owner_id) or "").strip() or transcript[:80]
    channel_target = (await get_setting("channel_target", owner_id=owner_id) or "").strip()

    try:
        settings = await get_settings_bulk(
            ["channel_style", "channel_audience", "content_rubrics", "post_scenarios", "content_constraints"],
            owner_id=owner_id,
        )
        bundle = await generate_post_bundle(
            api_key=api_key,
            model=getattr(cfg, "llm_model", "") or "",
            topic=topic,
            prompt=transcript,
            owner_id=owner_id,
            channel_style=settings.get("channel_style") or "",
            channel_audience=settings.get("channel_audience") or "",
            content_rubrics=settings.get("content_rubrics") or "",
            post_scenarios=settings.get("post_scenarios") or "",
            content_constraints=settings.get("content_constraints") or "",
            base_url=getattr(cfg, "llm_base_url", None),
            generation_path="voice",
        )
        post_text = "\n\n".join(filter(None, [bundle.get("title", ""), bundle.get("body", ""), bundle.get("cta", "")])).strip()
        if not post_text:
            post_text = bundle.get("body") or transcript
    except Exception:
        logger.exception("handle_voice_message: generation failed owner_id=%s", owner_id)
        post_text = transcript

    try:
        draft_id = await create_draft(
            owner_id=owner_id,
            channel_target=channel_target,
            text=post_text,
            prompt=transcript,
            topic=topic,
            draft_source="voice",
        )
    except Exception:
        logger.exception("handle_voice_message: create_draft failed owner_id=%s", owner_id)
        await answer_plain(message, "⚠️ Пост распознан, но сохранить черновик не удалось. Попробуй позже.", reply_markup=_main_keyboard())
        return

    await increment_feature_used(owner_id, "voice")

    app_link = miniapp_url or "Mini App"
    reply_text = (
        f"✅ Голосовое обработано! Пост сгенерирован и ждёт в черновиках.\n\n"
        f"🎙️ Распознанный текст: «{transcript[:120]}{'…' if len(transcript) > 120 else ''}»"
    )
    inline_markup = _miniapp_inline_keyboard(message, label="Открыть черновики")
    await answer_plain(message, reply_text, reply_markup=inline_markup or _main_keyboard())


@router.message(F.chat.type == "private", F.text.in_({QUICK_STATUS_LABEL, "статус", "status"}))
async def quick_status(message: Message):
    await _send_with_app_button(message, await _build_status_text(_owner_id(message)))


@router.message(F.chat.type == "private", F.text.in_({QUICK_HELP_LABEL, "помощь", "help"}))
async def quick_help(message: Message):
    await _send_with_app_button(message, HELP_TEXT)


@router.message(F.chat.type == "private", F.text == QUICK_APP_LABEL)
async def quick_app(message: Message):
    await cmd_app(message)


@router.message(F.chat.type == "private", F.text.in_(set(TOPIC_HINTS.keys())))
async def quick_topic(message: Message):
    user_text = TOPIC_HINTS.get((message.text or "").strip(), "")
    answer = await _consult(message, user_text)
    await _send_with_app_button(message, answer)


@router.message(F.chat.type == "private", F.text)
async def consultant_chat(message: Message):
    user_text = (message.text or "").strip()
    if not user_text:
        await answer_plain(message, "Напиши вопрос по функциям NeuroSMM.", reply_markup=_main_keyboard())
        return
    answer = await _consult(message, user_text)
    await _send_with_app_button(message, answer)
