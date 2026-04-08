
from db import get_post_stats, get_channel_settings

async def build_channel_stats(bot, owner_id: int = 0) -> str:
    stats = await get_post_stats(owner_id=owner_id)
    ch_settings = await get_channel_settings(owner_id)
    channel = ch_settings.get("channel_target") or ""
    topic = ch_settings.get("topic") or ""
    news_enabled = (ch_settings.get("news_enabled") or "0").strip() not in ("0", "false", "False")

    return (
        "📊 СТАТИСТИКА БОТА\n\n"
        f"Канал: {channel or 'не привязан'}\n"
        f"Тема: {topic or 'не задана'}\n"
        f"Авто-новости: {'включены' if news_enabled else 'выключены'}\n\n"
        f"Всего постов: {stats.get('total_posts', 0)}\n"
        f"Текстовых: {stats.get('text_posts', 0)}\n"
        f"С картинками: {stats.get('photo_posts', 0)}\n"
        f"Средняя длина поста: {stats.get('avg_length', 0)} символов\n\n"
        f"Записей в расписании: {stats.get('schedules_total', 0)}\n"
        f"Записей в контент-плане: {stats.get('plan_total', 0)}\n"
        f"Уже опубликовано из плана: {stats.get('plan_posted', 0)}"
    )
