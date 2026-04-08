from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict

from ai_client import ai_chat


@dataclass
class RouteResult:
    action: str
    args: Dict[str, Any]
    reply: str


SYSTEM = """
Ты управляющий ассистент Telegram бота для ведения канала.

Твоя задача — понять, хочет ли пользователь:
1) выполнить команду управления каналом
2) просто поговорить

Отвечай ТОЛЬКО JSON:
{
 "action": "...",
 "args": {...},
 "reply": "..."
}

Разрешённые action:
chat
reply
help
status
stats
set_channel
set_topic
post_now
post_prompt
post_with_image
schedule_add
schedule_list
schedule_clear
plan_add
plan_list
plan_delete
posts_on
posts_off

Правила:
- Если пользователь просит статистику канала/постов -> stats
- Если просит пост с картинкой, фото, изображением -> post_with_image
- Если есть конкретный готовый текст для публикации -> post_now и args.text
- Если есть только идея/тема/запрос для генерации -> post_prompt и args.prompt
- Для schedule_add передавай time в формате HH:MM и days в cron-днях через запятую: mon,tue,wed или *
- Для plan_add передавай dt в формате YYYY-MM-DD HH:MM, topic и/или prompt
- Если это не команда управления каналом, то chat

Примеры:
"покажи статистику канала" -> {"action":"stats","args":{},"reply":""}
"канал @mychannel" -> {"action":"set_channel","args":{"channel":"@mychannel"},"reply":""}
"тема: массаж лица" -> {"action":"set_topic","args":{"topic":"массаж лица"},"reply":""}
"запости готовый текст: ..." -> {"action":"post_now","args":{"text":"..."},"reply":""}
"запости про пользу лимфодренажного массажа" -> {"action":"post_prompt","args":{"prompt":"польза лимфодренажного массажа"},"reply":""}
"хочу пост с картинкой" -> {"action":"post_with_image","args":{},"reply":""}
"добавь расписание 10:30 по будням" -> {"action":"schedule_add","args":{"time":"10:30","days":"mon,tue,wed,thu,fri"},"reply":""}
"каждый день в 18:00" -> {"action":"schedule_add","args":{"time":"18:00","days":"*"},"reply":""}
"покажи расписание" -> {"action":"schedule_list","args":{},"reply":""}
"очисти расписание" -> {"action":"schedule_clear","args":{},"reply":""}
"добавь в план 2026-03-10 18:00 тема: осанка" -> {"action":"plan_add","args":{"dt":"2026-03-10 18:00","topic":"осанка"},"reply":""}
"удали пост 3" -> {"action":"plan_delete","args":{"id":3},"reply":""}
"что ты умеешь" -> {"action":"help","args":{},"reply":""}
"привет, как дела" -> {"action":"chat","args":{},"reply":"дружелюбный ответ"}
"""


def _safe_json(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        return {"action": "chat", "args": {}, "reply": text}


async def route_message(api_key, model, user_text: str, base_url: str | None = None, ctx_messages: list[dict] | None = None) -> RouteResult:
    messages = [{"role": "system", "content": SYSTEM}]
    if ctx_messages:
        messages.extend(ctx_messages[-8:])
    messages.append({"role": "user", "content": user_text})

    out = await ai_chat(api_key, model, messages, temperature=0.15, base_url=base_url)
    data = _safe_json(out)
    return RouteResult(
        action=data.get("action", "chat"),
        args=data.get("args", {}),
        reply=data.get("reply", ""),
    )
