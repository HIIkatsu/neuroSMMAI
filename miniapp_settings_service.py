from __future__ import annotations

import json
from typing import Any

_ALLOWED_POSTING_MODES = {"both", "posts", "news", "manual"}
_TRUE_VALUES = {"1", "true", "yes", "on", "y"}
_FALSE_VALUES = {"0", "false", "no", "off", "n"}
_JSON_ARRAY_FIELDS = {"channel_formats", "content_constraints"}
_STRING_LIMITS = {
    "topic": 160,
    "channel_target": 64,
    "channel_style": 160,
    "content_rubrics": 800,
    "rubrics_schedule": 800,
    "post_scenarios": 500,
    "news_sources": 500,
    "channel_audience": 160,
    "channel_style_preset": 160,
    "channel_mode": 160,
    "channel_frequency": 100,
    "content_exclusions": 1000,
    "channel_signature": 200,
    "author_role_type": 40,
    "author_role_description": 300,
    "author_activities": 500,
    "author_forbidden_claims": 1000,
}


def _clean_text(value: Any, limit: int | None = None) -> str:
    text = str(value or "").strip()
    text = " ".join(text.split())
    if limit is not None:
        text = text[:limit]
    return text


def _normalize_bool_str(value: Any, *, default: str = "0") -> str:
    raw = _clean_text(value).lower()
    if not raw:
        return default
    if raw in _TRUE_VALUES:
        return "1"
    if raw in _FALSE_VALUES:
        return "0"
    return default


def _normalize_int_str(value: Any, *, default: int, min_value: int, max_value: int) -> str:
    try:
        parsed = int(str(value).strip())
    except Exception:
        parsed = default
    parsed = max(min_value, min(max_value, parsed))
    return str(parsed)


def _normalize_json_string_list(value: Any, *, max_items: int = 8, item_limit: int = 60) -> str:
    items: list[str] = []
    if isinstance(value, list):
        source = value
    else:
        raw = str(value or "").strip()
        if not raw:
            return "[]"
        try:
            parsed = json.loads(raw)
            source = parsed if isinstance(parsed, list) else [part.strip() for part in raw.split(",")]
        except Exception:
            source = [part.strip() for part in raw.split(",")]
    seen: set[str] = set()
    for item in source:
        text = _clean_text(item, item_limit)
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        items.append(text)
        if len(items) >= max_items:
            break
    return json.dumps(items, ensure_ascii=False)


def normalize_settings_snapshot(raw: dict[str, Any] | None, active: dict[str, Any] | None = None) -> dict[str, str]:
    src = dict(raw or {})
    active = active or {}
    topic = _clean_text(src.get("topic") or active.get("topic"), _STRING_LIMITS["topic"])
    channel_target = _clean_text(src.get("channel_target") or active.get("channel_target"), _STRING_LIMITS["channel_target"])
    posting_mode = _clean_text(src.get("posting_mode") or "manual").lower()
    if posting_mode not in _ALLOWED_POSTING_MODES:
        posting_mode = "manual"
    return {
        "posts_enabled": _normalize_bool_str(src.get("posts_enabled"), default="0"),
        "posting_mode": posting_mode,
        "news_enabled": _normalize_bool_str(src.get("news_enabled"), default="0"),
        "news_interval_hours": _normalize_int_str(src.get("news_interval_hours"), default=6, min_value=1, max_value=168),
        "news_sources": _clean_text(src.get("news_sources"), _STRING_LIMITS["news_sources"]),
        "auto_hashtags": _normalize_bool_str(src.get("auto_hashtags"), default="0"),
        "topic": topic,
        "channel_target": channel_target,
        "channel_style": _clean_text(src.get("channel_style"), _STRING_LIMITS["channel_style"]),
        "content_rubrics": _clean_text(src.get("content_rubrics"), _STRING_LIMITS["content_rubrics"]),
        "rubrics_schedule": _clean_text(src.get("rubrics_schedule"), _STRING_LIMITS["rubrics_schedule"]),
        "post_scenarios": _clean_text(src.get("post_scenarios"), _STRING_LIMITS["post_scenarios"]),
        "content_exclusions": _clean_text(src.get("content_exclusions"), _STRING_LIMITS["content_exclusions"]),
        "news_strict_mode": _normalize_bool_str(src.get("news_strict_mode"), default="1"),
        "onboarding_completed": _normalize_bool_str(src.get("onboarding_completed"), default="0"),
        "channel_audience": _clean_text(src.get("channel_audience"), _STRING_LIMITS["channel_audience"]),
        "channel_style_preset": _clean_text(src.get("channel_style_preset"), _STRING_LIMITS["channel_style_preset"]),
        "channel_mode": _clean_text(src.get("channel_mode"), _STRING_LIMITS["channel_mode"]),
        "channel_formats": _normalize_json_string_list(src.get("channel_formats"), max_items=8, item_limit=60),
        "channel_frequency": _clean_text(src.get("channel_frequency"), _STRING_LIMITS["channel_frequency"]),
        "content_constraints": _normalize_json_string_list(src.get("content_constraints"), max_items=8, item_limit=80),
        "channel_signature": _clean_text(src.get("channel_signature"), _STRING_LIMITS["channel_signature"]),
        "source_auto_draft": _normalize_bool_str(src.get("source_auto_draft"), default="1"),
        # Author role fields
        "author_role_type": _clean_text(src.get("author_role_type"), _STRING_LIMITS["author_role_type"]),
        "author_role_description": _clean_text(src.get("author_role_description"), _STRING_LIMITS["author_role_description"]),
        "author_activities": _clean_text(src.get("author_activities"), _STRING_LIMITS["author_activities"]),
        "author_forbidden_claims": _normalize_json_string_list(src.get("author_forbidden_claims"), max_items=10, item_limit=120),
        # Auto-image toggle: "1" = auto-attach images, "0" = text-only posts
        "auto_image": _normalize_bool_str(src.get("auto_image"), default="1"),
    }


def normalize_settings_update(fields: dict[str, Any]) -> dict[str, str]:
    normalized = normalize_settings_snapshot(fields)
    out: dict[str, str] = {}
    for key in fields.keys():
        if key in normalized:
            out[key] = normalized[key]
    return out


def _parse_json_array(raw: str) -> list[str]:
    try:
        data = json.loads(raw or "[]")
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    return [str(x).strip() for x in data if str(x).strip()]


def build_operator_profile(
    *,
    settings: dict[str, str] | None,
    active_channel: dict[str, Any] | None,
    channels: list[dict[str, Any]] | None = None,
    drafts_current: int = 0,
    plan_items: list[dict[str, Any]] | None = None,
    schedules: list[dict[str, Any]] | None = None,
    stats: dict[str, Any] | None = None,
    analytics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    settings = normalize_settings_snapshot(settings, active_channel)
    channels = channels or []
    plan_items = plan_items or []
    schedules = schedules or []
    stats = stats or {}
    analytics = analytics or {}

    formats = _parse_json_array(settings.get("channel_formats", "[]"))
    constraints = _parse_json_array(settings.get("content_constraints", "[]"))
    topic_ready = bool(settings.get("topic"))
    channel_ready = bool(active_channel and active_channel.get("channel_target"))
    audience_ready = bool(settings.get("channel_audience"))
    style_ready = bool(settings.get("channel_style_preset") or settings.get("channel_style"))
    mode_ready = bool(settings.get("channel_mode") or settings.get("posting_mode"))
    formats_ready = len(formats) >= 2
    rhythm_ready = bool(settings.get("channel_frequency")) or any(int(x.get("enabled") or 0) == 1 for x in schedules)
    onboarding_ready = settings.get("onboarding_completed") == "1"
    drafts_ready = int(drafts_current or 0) > 0
    plan_ready = any(int(item.get("enabled") or 0) == 1 for item in plan_items)
    history_ready = int(stats.get("total_posts") or 0) >= 3

    score = 0
    score += 22 if topic_ready else 0
    score += 18 if channel_ready else 0
    score += 10 if audience_ready else 0
    score += 10 if style_ready else 0
    score += 10 if mode_ready else 0
    score += 12 if formats_ready else 0
    score += 10 if rhythm_ready else 0
    score += 8 if onboarding_ready else 0

    blockers: list[str] = []
    recommendations: list[dict[str, str]] = []
    if not channel_ready:
        blockers.append("Не выбран активный канал")
        recommendations.append({
            "id": "channel",
            "label": "Подключить канал",
            "hint": "Без активного канала бот не может публиковать и строить нормальный сценарий работы.",
            "cta": "Открыть каналы",
        })
    if not topic_ready:
        blockers.append("Не задана тема канала")
        recommendations.append({
            "id": "topic",
            "label": "Зафиксировать тему",
            "hint": "Пока тема размыта, AI будет расползаться в случайный контент.",
            "cta": "Заполнить тему",
        })
    if not formats_ready:
        blockers.append("Не выбраны рабочие форматы контента")
        recommendations.append({
            "id": "formats",
            "label": "Выбрать 3–5 форматов",
            "hint": "Иначе лента быстро станет однотипной и будет похожа на AI-мусор.",
            "cta": "Настроить форматы",
        })
    if not rhythm_ready:
        blockers.append("Не задан ритм публикаций")
        recommendations.append({
            "id": "rhythm",
            "label": "Настроить частоту",
            "hint": "Без частоты бот не превращается в автопилот, а остается просто генератором вручную.",
            "cta": "Указать частоту",
        })
    if not onboarding_ready:
        recommendations.append({
            "id": "onboarding",
            "label": "Дожать onboarding",
            "hint": "Сейчас профиль канала неполный — из-за этого главная логика продукта недонастроена.",
            "cta": "Закончить onboarding",
        })
    if not drafts_ready and not plan_ready:
        recommendations.append({
            "id": "pipeline",
            "label": "Собрать очередь публикаций",
            "hint": "Нет ни черновиков, ни плана. Даже хороший профиль канала без очереди не дает автопилот.",
            "cta": "Создать контент",
        })

    autopilot_ready = bool(
        channel_ready and topic_ready and formats_ready and rhythm_ready and (drafts_ready or plan_ready)
    )
    if autopilot_ready:
        score = max(score, 78)
    if history_ready:
        score = min(100, score + 6)

    if score >= 85:
        stage = "autopilot"
        summary = "Профиль канала собран. Можно уходить в режим оператора, а не ручного автопостинга."
    elif score >= 65:
        stage = "ready"
        summary = "Основа уже есть, но осталось убрать несколько дыр, чтобы бот не скатывался в хаос."
    elif score >= 40:
        stage = "setup"
        summary = "Каркас собран частично. Продукт еще слишком зависит от ручных действий пользователя."
    else:
        stage = "broken"
        summary = "Сейчас это не автопилот канала, а набор разрозненных функций без плотной настройки."

    next_action = recommendations[0] if recommendations else {
        "id": "scale",
        "label": "Масштабировать рабочую связку",
        "hint": "Профиль уже собран. Дальше главное — накапливать историю и улучшать темы по фактам.",
        "cta": "Открыть аналитику",
    }

    return {
        "score": int(max(0, min(100, score))),
        "stage": stage,
        "summary": summary,
        "autopilot_ready": autopilot_ready,
        "has_history": history_ready,
        "blockers": blockers,
        "next_action": next_action,
        "recommendations": recommendations[:4],
        "content_profile": {
            "topic": settings.get("topic", ""),
            "audience": settings.get("channel_audience", ""),
            "style": settings.get("channel_style_preset") or settings.get("channel_style", ""),
            "mode": settings.get("channel_mode") or settings.get("posting_mode", "manual"),
            "formats": formats,
            "constraints": constraints,
            "frequency": settings.get("channel_frequency", ""),
        },
        "pipeline": {
            "drafts_current": int(drafts_current or 0),
            "plan_total": len(plan_items),
            "enabled_schedules": sum(1 for x in schedules if int(x.get("enabled") or 0) == 1),
            "channels_total": len(channels),
            "posts_total": int(stats.get("total_posts") or 0),
        },
    }
