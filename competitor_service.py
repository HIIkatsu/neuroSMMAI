"""Competitor Spy service — fetches and analyses public Telegram channel posts."""
from __future__ import annotations

import logging
import re
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
}

_MAX_POSTS = 10
_MIN_TEXT_LEN = 40


def _parse_channel_name(link: str) -> str | None:
    """Extract channel username from a t.me link or a bare @username."""
    link = (link or "").strip().lstrip("@")
    # https://t.me/channame or t.me/channame or channame
    m = re.search(r"t\.me/(?:s/)?([A-Za-z0-9_]{3,})", link)
    if m:
        return m.group(1)
    # bare word
    if re.match(r"^[A-Za-z0-9_]{3,}$", link):
        return link
    return None


def _extract_posts_html(html: str) -> list[str]:
    """Extract plain text from tg-channel post bubbles without BeautifulSoup."""
    texts: list[str] = []
    # Each post is wrapped in <div class="tgme_widget_message_text ...">
    for block in re.findall(
        r'<div[^>]+class="[^"]*tgme_widget_message_text[^"]*"[^>]*>(.*?)</div>',
        html,
        re.DOTALL | re.IGNORECASE,
    ):
        # Strip HTML tags
        plain = re.sub(r"<[^>]+>", " ", block)
        # Decode basic HTML entities
        plain = plain.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        plain = plain.replace("&nbsp;", " ").replace("&#39;", "'").replace("&quot;", '"')
        plain = re.sub(r"\s+", " ", plain).strip()
        if len(plain) >= _MIN_TEXT_LEN:
            texts.append(plain)
        if len(texts) >= _MAX_POSTS:
            break
    return texts


async def fetch_competitor_posts(channel_link: str) -> list[str]:
    """Download the web preview of a public TG channel and return the latest post texts."""
    channel_name = _parse_channel_name(channel_link)
    if not channel_name:
        raise ValueError(f"Не удалось распознать название канала: {channel_link!r}")

    url = f"https://t.me/s/{channel_name}"
    try:
        async with httpx.AsyncClient(timeout=15, headers=_HEADERS, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            raise ValueError(f"Канал @{channel_name} не найден или является приватным.")
        raise RuntimeError(f"Ошибка при получении канала @{channel_name}: HTTP {exc.response.status_code}")
    except httpx.RequestError as exc:
        raise RuntimeError(f"Не удалось подключиться к t.me: {exc}")

    posts = _extract_posts_html(resp.text)
    if not posts:
        raise ValueError(
            f"Не удалось найти публичные текстовые посты в канале @{channel_name}. "
            "Убедись, что канал публичный и содержит текстовые посты."
        )
    return posts


async def analyse_competitor_and_generate(
    channel_link: str,
    *,
    user_topic: str = "",
    user_style: str = "",
    user_audience: str = "",
    ai_chat_fn: Any,
) -> list[dict[str, str]]:
    """Fetch competitor posts, analyse them, and return 3 draft dicts (topic + text)."""
    posts = await fetch_competitor_posts(channel_link)
    channel_name = _parse_channel_name(channel_link) or channel_link

    numbered = "\n\n".join(f"{i+1}. {p}" for i, p in enumerate(posts))
    prompt = (
        f"Ты опытный SMM-стратег. Я дам тебе последние посты конкурирующего Telegram-канала @{channel_name}.\n"
        f"Твоя задача:\n"
        f"1. Определи 3 наиболее сильные темы/форматы из этих постов.\n"
        f"2. Для каждой темы напиши НОВЫЙ, УНИКАЛЬНЫЙ пост для МОЕГО канала.\n"
        f"   Мой канал: тема — «{user_topic or 'не указана'}», "
        f"стиль — «{user_style or 'живой, без канцелярита'}», "
        f"аудитория — «{user_audience or 'не указана'}».\n"
        f"3. Не копируй посты конкурента — переосмысли смыслы для моей аудитории.\n\n"
        f"Посты конкурента:\n{numbered}\n\n"
        f"Ответ строго в формате JSON-массива из 3 объектов:\n"
        f'[{{"topic": "заголовок темы", "text": "полный текст поста"}}, ...]\n'
        f"Только JSON, без объяснений."
    )

    raw = await ai_chat_fn([{"role": "user", "content": prompt}], max_tokens=1800)
    raw = raw.strip()

    # Extract JSON array from response
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if not m:
        raise RuntimeError("ИИ вернул некорректный ответ. Попробуй ещё раз.")

    import json
    try:
        drafts = json.loads(m.group(0))
    except json.JSONDecodeError:
        raise RuntimeError("Не удалось разобрать ответ ИИ. Попробуй ещё раз.")

    result = []
    for item in drafts[:3]:
        if isinstance(item, dict) and item.get("text"):
            result.append({
                "topic": str(item.get("topic") or ""),
                "text": str(item.get("text") or ""),
            })
    return result
