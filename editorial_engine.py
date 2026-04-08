from __future__ import annotations

import json
import re
from typing import Any

import db
from channel_strategy import build_generation_strategy, parse_json_list


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def trusted_default_sources(topic: str) -> list[str]:
    topic_l = clean_text(topic).lower()
    mapping: list[tuple[tuple[str, ...], list[str]]] = [
        (("chip", "чип", "полупровод", "nvidia", "amd", "intel", "tsmc", "semiconductor"), [
            "reuters.com", "bloomberg.com", "anandtech.com", "tomshardware.com", "techpowerup.com", "semiengineering.com", "eetimes.com"
        ]),
        (("ai", "ии", "llm", "openai", "нейросет", "искусственный интеллект", "gpt"), [
            "reuters.com", "apnews.com", "theverge.com", "techcrunch.com", "arstechnica.com", "wired.com"
        ]),
        (("game", "gaming", "игр", "steam", "playstation", "xbox", "nintendo"), [
            "theverge.com", "ign.com", "gamespot.com", "pcgamer.com", "eurogamer.net"
        ]),
        (("массаж", "здоров", "физиотерап", "осанк", "шея", "спина", "wellness", "rehab", "physio"), [
            "massagemag.com", "amtamassage.org", "massagetoday.com", "abmp.com", "healthline.com"
        ]),
        (("финанс", "акци", "рынок", "инвест", "эконом", "finance", "market", "stock"), [
            "reuters.com", "bloomberg.com", "ft.com", "wsj.com", "cnbc.com"
        ]),
    ]
    for keys, domains in mapping:
        if any(k in topic_l for k in keys):
            return domains
    return ["reuters.com", "apnews.com", "bbc.com", "bloomberg.com"]


def normalize_sources(value: Any, topic: str = "") -> list[str]:
    raw_list = parse_json_list(value)
    if not raw_list and isinstance(value, str) and "," in value:
        raw_list = [x.strip() for x in value.split(",") if x.strip()]
    out: list[str] = []
    seen: set[str] = set()
    for item in raw_list:
        d = clean_text(item).lower().strip('/')
        d = re.sub(r"^https?://", "", d)
        d = re.sub(r"^www\.", "", d)
        if not d or d in seen:
            continue
        seen.add(d)
        out.append(d)
    if not out:
        out = trusted_default_sources(topic)
    return out[:10]


def prompt_looks_news(prompt: str) -> bool:
    p = clean_text(prompt).lower()
    if not p:
        return False
    markers = [
        "новост", "свеж", "актуал", "что произошло", "что нового", "latest", "breaking", "today", "сегодня", "на этой неделе", "за день"
    ]
    return any(m in p for m in markers)


async def load_editorial_settings(owner_id: int | None, overrides: dict[str, Any] | None = None, *, channel_profile_id: int | None = None) -> dict[str, Any]:
    settings: dict[str, Any] = {}
    if owner_id:
        try:
            settings = await db.get_channel_settings(int(owner_id or 0), channel_profile_id=channel_profile_id)
        except Exception:
            settings = {}
    for k, v in (overrides or {}).items():
        if v not in (None, ""):
            settings[k] = v
    return settings


async def build_editorial_context(
    owner_id: int | None,
    *,
    topic: str = "",
    prompt: str = "",
    channel_style: str = "",
    content_rubrics: str = "",
    post_scenarios: str = "",
    channel_audience: str = "",
    content_constraints: str = "",
    recent_posts: list[str] | None = None,
    recent_plan: list[str] | None = None,
) -> dict[str, Any]:
    settings = await load_editorial_settings(owner_id, {
        'topic': topic,
        'channel_style': channel_style,
        'content_rubrics': content_rubrics,
        'post_scenarios': post_scenarios,
        'channel_audience': channel_audience,
        'content_constraints': content_constraints,
    })
    strategy = build_generation_strategy(settings)
    sources = normalize_sources(settings.get('news_sources'), strategy.get('topic') or topic)
    history = {'recent_posts': recent_posts or [], 'recent_plan': recent_plan or []}
    if owner_id:
        try:
            db_history = await db.get_recent_channel_history(owner_id=int(owner_id or 0), limit=12)
        except Exception:
            db_history = {}
        if not history['recent_posts']:
            history['recent_posts'] = list((db_history or {}).get('recent_posts') or [])[:10]
        if not history['recent_plan']:
            history['recent_plan'] = list((db_history or {}).get('recent_plan') or [])[:10]
        history['recent_drafts'] = list((db_history or {}).get('recent_drafts') or [])[:10]
    else:
        history['recent_drafts'] = []
    requested = clean_text(prompt) or clean_text(topic) or clean_text(strategy.get('topic')) or 'теме канала'
    return {
        'owner_id': int(owner_id or 0),
        'settings': settings,
        'strategy': strategy,
        'sources': sources,
        'requested': requested,
        'topic': clean_text(strategy.get('topic') or topic) or 'теме канала',
        'mode': clean_text(strategy.get('mode')),
        'news_enabled': bool(strategy.get('news_enabled')),
        'history': history,
        'profile_ready': bool(clean_text(strategy.get('topic')) and clean_text(strategy.get('audience')) and (clean_text(strategy.get('style_preset')) or clean_text(strategy.get('style_text')))),
    }


def history_block(context: dict[str, Any], *, limit: int = 8) -> str:
    items: list[str] = []
    for key in ('recent_posts', 'recent_drafts', 'recent_plan'):
        for value in list((context.get('history') or {}).get(key) or [])[:limit]:
            value = clean_text(value)
            if value:
                items.append(value)
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        low = item.lower()
        if low in seen:
            continue
        seen.add(low)
        out.append(f"- {item}")
    return "\n".join(out[:limit]) or "- пока нет данных"


def profile_block(context: dict[str, Any]) -> str:
    s = context.get('strategy') or {}
    return "\n".join([
        f"- тема: {clean_text(context.get('topic')) or 'не указана'}",
        f"- аудитория: {clean_text(s.get('audience')) or 'не указана'}",
        f"- стиль: {clean_text(s.get('style_preset') or s.get('style_text')) or 'не указан'}",
        f"- режим: {clean_text(s.get('mode')) or 'не указан'}",
        f"- ритм: {clean_text(s.get('frequency_hint')) or 'не указан'}",
        f"- ограничения: {clean_text(s.get('constraint_line')) or 'без ограничений'}",
        "- форматы:",
        clean_text(s.get('rubric_text')) or '- не указаны',
        f"- trusted sources: {', '.join(context.get('sources') or []) or 'не заданы'}",
    ])


def serialise_sources_for_setting(domains: list[str]) -> str:
    return json.dumps(domains, ensure_ascii=False)
