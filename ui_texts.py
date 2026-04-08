
from __future__ import annotations

from datetime import datetime
import json

_MODE_LABELS = {"both": "посты + новости", "posts": "только посты", "news": "только новости"}
_DAY_LABELS = {"mon": "пн", "tue": "вт", "wed": "ср", "thu": "чт", "fri": "пт", "sat": "сб", "sun": "вс"}


def _pretty_dt(value: str) -> str:
    if not value:
        return value
    raw = value.strip()
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"):
        try:
            return datetime.strptime(raw, fmt).strftime("%d.%m.%Y %H:%M")
        except Exception:
            pass
    try:
        return datetime.fromisoformat(raw).strftime("%d.%m.%Y %H:%M")
    except Exception:
        return raw


def _human_days(days: str) -> str:
    d = (days or "*").strip().lower()
    if not d or d == "*":
        return "каждый день"
    parts = [p.strip() for p in d.split(",") if p.strip()]
    out = [_DAY_LABELS.get(p, p) for p in parts]
    return " ".join(out) if out else "каждый день"


def menu_text() -> str:
    return (
        "🤖 УПРАВЛЕНИЕ КАНАЛОМ\n\n"
        "• 📝 Создать пост — создаёт черновик и открывает редактор\n"
        "• 📅 Расписание\n"
        "• 📆 Контент-план\n"
        "• 📰 Авто-новости\n"
        "• 📊 Статистика\n"
    )


def help_text(is_admin: bool) -> str:
    txt = (
        "❓ КОМАНДЫ\n"
        "• /help — помощь\n"
        "• /status — статус\n\n"
        "Понимаю обычный текст:\n"
        "• пост про массаж\n"
        "• пост без картинки про массаж\n"
        "• через 5 минут пост про шею\n"
        "• каждые 3 дня пост про осанку\n"
        "• авто-новости включить\n"
        "• свежая новость\n"
    )
    if is_admin:
        txt += (
            "\n👑 АДМИН\n"
            "• канал @username — привязать канал\n"
            "• тема: массаж — основная тема канала\n"
            "• тема новостей: массаж и восстановление\n"
            "• интервал новостей: 6 часов\n"
            "• источники новостей: who.int, mayoclinic.org, nih.gov\n"
            "• редактор поста открывается ПЕРЕД публикацией\n"
        )
    return txt


def status_text(channel: str | None, topic: str | None, posts_enabled: bool = True, posting_mode: str = "both", bound_chat: str | None = None, news_enabled: bool = False) -> str:
    return (
        "⚙️ СТАТУС\n\n"
        f"Канал: {channel or 'не привязан'}\n"
        f"Чат: {bound_chat or 'не привязан'}\n"
        f"Тема: {topic or 'не задана'}\n"
        f"Постинг: {'включён' if posts_enabled else 'выключен'}\n"
        f"Авто-новости: {'включены' if news_enabled else 'выключены'}\n"
        f"Режим: {_MODE_LABELS.get((posting_mode or 'both').strip().lower(), posting_mode)}\n"
        "Часовой пояс: Europe/Moscow"
    )


def schedules_text(items: list[dict]) -> str:
    if not items:
        return "📅 РАСПИСАНИЕ ПУСТО\n\nПримеры:\n• добавь расписание 10:30 пн ср пт\n• каждые 3 дня пост про массаж"
    lines = ["📅 РАСПИСАНИЕ", ""]
    for item in items:
        enabled = "✅" if int(item.get("enabled", 1)) else "⛔"
        lines.append(f"{enabled} #{item.get('id')}  ⏰ {item.get('time', '??:??')}")
        lines.append(f"   📆 {_human_days(item.get('days', '*'))}")
        lines.append("")
    return "\n".join(lines).strip()


def plan_text(items: list[dict]) -> str:
    if not items:
        return "🗂 КОНТЕНТ-ПЛАН ПУСТ\n\nПример:\n• добавь в план 2026-03-10 18:00 тема: упражнения"
    lines = ["🗂 КОНТЕНТ-ПЛАН", ""]
    for item in items[:50]:
        marker = "✅" if int(item.get("posted", 0)) else "🕓"
        if item.get("kind") == "draft":
            label = f"черновик #{item.get('payload')}"
        else:
            label = (item.get("topic") or item.get("prompt") or item.get("payload") or "").strip()
        if len(label) > 110:
            label = label[:107] + "..."
        lines.append(f"{marker} #{item.get('id')}  {_pretty_dt(item.get('dt', ''))}")
        lines.append(f"   {label or 'без текста'}")
        lines.append("")
    return "\n".join(lines).strip()


def draft_text(draft: dict) -> str:
    try:
        buttons_count = len(json.loads(draft.get("buttons_json") or "[]"))
    except Exception:
        buttons_count = 0
    media = "есть" if (draft.get("media_type") == "photo" and draft.get("media_ref")) else "нет"
    text = (draft.get("text") or "").strip()
    if len(text) > 800:
        text = text[:797] + "..."
    return (
        "📝 ЧЕРНОВИК ПОСТА\n\n"
        f"ID: {draft.get('id')}\n"
        f"Канал: {draft.get('channel_target') or 'не привязан'}\n"
        f"Тема: {draft.get('topic') or 'не задана'}\n"
        f"Картинка: {media}\n"
        f"Кнопки: {buttons_count}\n"
        f"Закреп: {'да' if int(draft.get('pin_post', 0)) else 'нет'}\n\n"
        f"{text or 'Текст пуст'}"
    )


def news_settings_text(enabled: bool, topic: str, interval_hours: int, sources: str) -> str:
    return (
        "📰 АВТО-НОВОСТИ\n\n"
        f"Статус: {'включены' if enabled else 'выключены'}\n"
        f"Тема: {topic or 'не задана'}\n"
        f"Интервал: {interval_hours} ч\n"
        f"Источники: {sources or 'не заданы'}\n\n"
        "Команды:\n"
        "• авто-новости включить\n"
        "• авто-новости выключить\n"
        "• тема новостей: массаж\n"
        "• интервал новостей: 6 часов\n"
        "• источники новостей: who.int, mayoclinic.org, nih.gov\n"
        "• свежая новость"
    )
