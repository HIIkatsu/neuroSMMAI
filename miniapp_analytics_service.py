from __future__ import annotations

from typing import Any
import re

import db
from miniapp_shared import clean_text


async def recent_channel_history(owner_id: int) -> dict[str, list[str]]:
    try:
        return await db.get_recent_channel_history(owner_id=owner_id, limit=12)
    except Exception:
        return {"recent_posts": [], "recent_drafts": [], "recent_plan": [], "recent_generations": []}


def recent_history_lines(history: dict[str, list[str]] | None, limit: int = 8) -> str:
    history = history or {}
    items: list[str] = []
    for key in ("recent_posts", "recent_drafts", "recent_plan", "recent_generations"):
        for value in history.get(key, [])[:limit]:
            value = clean_text(str(value or ""))
            if value:
                items.append(value)
    seen: list[str] = []
    for item in items:
        low = item.lower()
        if low not in seen:
            seen.append(low)
    return "\n".join(f"- {x}" for x in seen[:limit]) or "- пока нет данных"


def history_texts(history: dict[str, list[str]] | None) -> list[str]:
    out: list[str] = []
    for key in ("recent_posts", "recent_drafts", "recent_plan", "recent_generations"):
        for value in (history or {}).get(key, []):
            value = clean_text(str(value or ""))
            if value:
                out.append(value)
    return out


def _safe_json_list(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [clean_text(str(x)) for x in raw if clean_text(str(x))]
    text = clean_text(str(raw or ''))
    if not text:
        return []
    if text.startswith('[') and text.endswith(']'):
        try:
            import json
            data = json.loads(text)
            if isinstance(data, list):
                return [clean_text(str(x)) for x in data if clean_text(str(x))]
        except Exception:
            pass
    return [clean_text(x) for x in text.split(',') if clean_text(x)]


def text_topic_signature(text: str) -> str:
    raw = clean_text(text).lower()
    raw = re.sub(r"#[\wа-яА-ЯёЁ_]+", " ", raw)
    raw = re.sub(r"[^a-zа-яё0-9\s]", " ", raw)
    words = [w for w in raw.split() if len(w) >= 4]
    if not words:
        return raw[:42]
    return " ".join(words[:6])


def history_repeat_rate(history: dict[str, list[str]] | None) -> int:
    texts = history_texts(history)
    if not texts:
        return 0
    signatures = [text_topic_signature(x) for x in texts if text_topic_signature(x)]
    if not signatures:
        return 0
    unique = len(set(signatures))
    total = len(signatures)
    return max(0, round((1 - unique / max(total, 1)) * 100))


def _dedupe_rate(values: list[str]) -> int:
    cleaned = [clean_text(x).lower() for x in values if clean_text(x)]
    if not cleaned:
        return 0
    return max(0, round((1 - len(set(cleaned)) / max(len(cleaned), 1)) * 100))


def factual_signal(key: str, label: str, value: int, hint: str, action: str, sample: int = 0) -> dict[str, Any]:
    score = max(0, min(100, int(round(value))))
    confidence = "high" if sample >= 4 else ("low" if sample > 0 else "none")
    return {
        "key": key,
        "label": label,
        "value": score,
        "hint": hint,
        "action": action,
        "sample": int(sample),
        "confidence": confidence,
        "available": True,
    }


def analytics_recommendations(signals: list[dict[str, Any]], *, plan_count: int = 0, drafts_count: int = 0, schedules_total: int = 0, onboarding_done: bool = True, active_channel: bool = True) -> list[str]:
    """Return up to 6 highly actionable, feature-specific recommendations sorted by priority."""
    recs: list[str] = []

    # Critical blockers first
    if not active_channel:
        recs.append("Подключи Telegram-канал в разделе «Настройки → Канал» — без этого автопостинг невозможен.")
    if not onboarding_done:
        recs.append("Заверши онбординг: укажи тему, аудиторию, стиль и форматы, чтобы разблокировать автогенерацию постов.")

    # Content reserve blockers
    need_plan = max(0, 5 - plan_count)
    if plan_count == 0:
        recs.append("Добавь минимум 5 идей в контент-план (раздел «План») — без них автопилот остановится.")
    elif plan_count < 3:
        recs.append(f"Добавь ещё {need_plan} записей в контент-план, чтобы автопилот работал без остановок.")

    need_drafts = max(0, 3 - drafts_count)
    if drafts_count == 0:
        recs.append("Создай хотя бы 3 черновика (раздел «Черновики») — это резерв на случай сбоя автогенерации.")
    elif drafts_count < 2:
        recs.append(f"Подготовь ещё {need_drafts} черновика как запас: автопостинг сможет взять их при задержке генерации.")

    # Schedule blocker
    if schedules_total < 1:
        recs.append("Настрой расписание публикаций (раздел «Автопостинг») — выбери дни и время для автовыхода постов.")

    # Signal-based recommendations for remaining weak areas
    for signal in sorted(signals, key=lambda x: int(x.get("value") or 0)):
        action = clean_text(str(signal.get("action") or ""))
        if action and action not in recs:
            recs.append(action)
        if len(recs) >= 6:
            break

    if not recs:
        recs.append("Продолжай публиковать — накопленные данные сделают аналитику точнее.")
    return recs[:6]


def build_channel_analytics(
    stats: dict[str, Any] | None = None,
    history: dict[str, list[str]] | None = None,
    analytics_snapshot: dict[str, Any] | None = None,
    *,
    settings: dict[str, Any] | None = None,
    active_channel: dict[str, Any] | None = None,
    channels: list[dict[str, Any]] | None = None,
    drafts: list[dict[str, Any]] | None = None,
    plan_items: list[dict[str, Any]] | None = None,
    media_items: list[dict[str, Any]] | None = None,
    schedules: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    stats = stats or {}
    history = history or {}
    analytics_snapshot = analytics_snapshot or {}
    settings = settings or {}
    drafts = drafts or []
    plan_items = plan_items or []
    media_items = media_items or []
    schedules = schedules or []

    texts = history_texts(history)
    sample_size = len(texts)
    topic = clean_text(settings.get('topic'))
    audience = clean_text(settings.get('channel_audience'))
    style = clean_text(settings.get('channel_style') or settings.get('channel_style_preset'))
    mode = clean_text(settings.get('channel_mode') or settings.get('posting_mode'))
    frequency = clean_text(settings.get('channel_frequency'))
    formats = _safe_json_list(settings.get('channel_formats'))
    constraints = _safe_json_list(settings.get('content_constraints'))
    news_sources = _safe_json_list(settings.get('news_sources_json') or settings.get('news_sources'))
    onboarding_flag = str(settings.get('onboarding_completed') or '0') == '1'
    onboarding_done = onboarding_flag or bool(topic and audience and style and len(formats) >= 2 and (frequency or mode))

    plan_count = len([x for x in plan_items if int(x.get('posted') or 0) == 0]) if plan_items else int(stats.get('plan_pending') or stats.get('plan_total') or 0)
    drafts_count = len([x for x in drafts if str((x or {}).get('status') or 'draft').strip().lower() == 'draft']) if drafts else int(stats.get('drafts_total') or 0)
    media_count = len(media_items) if media_items else int(stats.get('media_inbox_total') or 0)
    schedules_total = len([x for x in schedules if int(x.get('enabled', 1) or 0) == 1]) if schedules else int(stats.get('schedules_total') or 0)
    total_posts = int(stats.get('total_posts') or 0)
    posted_last_7d = int(stats.get('posted_last_7d') or 0)

    media_refs = [str(x.get('media_ref') or x.get('file_id') or '') for x in drafts + media_items if str(x.get('media_ref') or x.get('file_id') or '').strip()]
    media_repeat_rate = _dedupe_rate(media_refs)
    text_repeat_rate = history_repeat_rate(history)
    repeat_rate = max(text_repeat_rate, round((text_repeat_rate * 0.65) + (media_repeat_rate * 0.35)))


    # ---------------------------------------------------------------------------
    # Readiness score — point-based formula (each factor has a clear contribution)
    # Factors only include user-controlled actions; 100% is genuinely hard to reach.
    #
    # Factor                          Max pts   Notes
    # ─────────────────────────────── ──────    ─────────────────────────────────
    # Channel connected               20        bot has access to the channel
    # Topic set (non-empty)           10        channel_topic configured
    # Audience description            5         channel_audience filled
    # ≥1 active schedule slot         15        autoposting enabled
    # Draft buffer (3+ ready)         15        partial: 1→5, 2→10, 3+→15
    # Content plan items (5+ unused)  10        partial: 1→2, 2→4, 3→6, 4→8, 5+→10
    # Media available (≥1 item)       5         any media in inbox/library
    # Posting mode configured         5         channel_mode set
    # Style/format configured         5         style + ≥1 format set
    # Recent activity (post in 7d)    10        at least 1 post in last 7 days
    # ─────────────────────────────── ──────    ─────────────────────────────────
    # Total possible                  100
    # ---------------------------------------------------------------------------
    pts_channel   = 20 if active_channel else 0
    pts_topic     = 10 if topic else 0
    pts_audience  = 5 if audience else 0
    pts_schedule  = 15 if schedules_total >= 1 else 0
    pts_drafts    = (15 if drafts_count >= 3 else (10 if drafts_count == 2 else (5 if drafts_count == 1 else 0)))
    pts_plan      = min(10, plan_count * 2)          # 2 pts per item, cap at 10
    pts_media     = 5 if media_count >= 1 else 0
    pts_mode      = 5 if mode else 0
    pts_style     = 5 if (style and len(formats) >= 1) else 0
    pts_activity  = 10 if posted_last_7d >= 1 else 0

    score = (
        pts_channel + pts_topic + pts_audience + pts_schedule +
        pts_drafts + pts_plan + pts_media + pts_mode + pts_style + pts_activity
    )
    score = max(0, min(100, score))

    # Build per-signal values for the analytics UI (0-100 scale).
    # These are derived from the factor points to produce signal values that
    # feel proportional on the UI — they are display metrics only, not inputs
    # to the main score.  The main score is already computed above.
    profile_score = max(0, min(100, pts_topic * 6 + pts_audience * 8 + pts_mode * 8 + pts_style * 8))
    content_reserve = max(0, min(100, pts_drafts * 5 + pts_plan * 6))
    media_ready = max(0, min(100, pts_media * 20))
    autopost_readiness = max(0, min(100, pts_channel + pts_schedule * 3 + min(20, posted_last_7d * 10)))
    variety_score = max(0, min(100, 100 - min(80, round(repeat_rate * 1.5))))
    news_ready = 100 if mode not in {'news', 'both'} else max(0, min(100, min(60, len(news_sources) * 15) + (20 if topic else 0) + (10 if audience else 0) + (10 if frequency else 0)))

    signals = [
        factual_signal('profile', 'Профиль канала', max(0, min(100, profile_score)), 'Насколько полно задана стратегия канала', 'Дозаполни тему, аудиторию, стиль, режим и форматы в разделе «Настройки»', len(formats) + len(constraints)),
        factual_signal('content_reserve', 'Запас контента', content_reserve, f'{plan_count} в плане · {drafts_count} черновиков', f'Добавь {max(0, 5 - plan_count)} идей в «План» и {max(0, 3 - drafts_count)} черновиков для стабильной работы', plan_count + drafts_count),
        factual_signal('media', 'Медиарезерв', media_ready, f'{media_count} файлов · медиа-повторы {media_repeat_rate}%', 'Загрузи больше уникальных фото/видео в «Медиатеку» — разнообразие обложек повышает охваты', media_count),
        factual_signal('autopost', 'Автопостинг', autopost_readiness, f'{schedules_total} слотов · {posted_last_7d} публикаций за 7 дней', 'Настрой расписание в разделе «Автопостинг»: выбери дни и время выхода постов', schedules_total + posted_last_7d),
        factual_signal('variety', 'Разнообразие', variety_score, f'Текстовые повторы {text_repeat_rate}% · медиа-повторы {media_repeat_rate}%', 'Сгенерируй новый контент-план с разными рубриками и углами подачи', sample_size),
        factual_signal('news', 'Актуальность режима', news_ready, f'{len(news_sources)} источников · режим {mode or "не задан"}', 'Добавь источники новостей в «Настройки → Новости» и укажи режим «Новости»', len(news_sources)),
    ]

    unavailable = []
    top_topics = list(analytics_snapshot.get('top_topics') or [])[:6]
    views_known = bool(analytics_snapshot.get('views_known'))
    avg_views = float(analytics_snapshot.get('avg_views') or 0)
    avg_reactions = float(analytics_snapshot.get('avg_reactions') or 0)
    avg_comments = float(analytics_snapshot.get('avg_comments') or 0)
    avg_forwards = float(analytics_snapshot.get('avg_forwards') or 0)
    stats_posts_considered = int(analytics_snapshot.get('total_posts_considered') or 0)
    if views_known:
        views_score = max(0, min(100, round((min(avg_views, 1200) / 1200.0) * 100)))
        engagement_rate = 0.0 if avg_views <= 0 else ((avg_reactions + avg_comments + avg_forwards) / max(avg_views, 1.0)) * 100.0
        engagement_score = max(0, min(100, round((min(engagement_rate, 12.0) / 12.0) * 100)))
        signals.extend([
            factual_signal('views', 'Средние просмотры', views_score, f'≈ {round(avg_views)} просмотров на пост', 'Усиль первые строки, обложки и заголовки', stats_posts_considered),
            factual_signal('engagement', 'Вовлечение', engagement_score, f'≈ {avg_reactions:.1f} реакций · {avg_comments:.1f} комментариев · {avg_forwards:.1f} пересылок', 'Добавь вопросы, опросы и явный call-to-action', stats_posts_considered),
        ])
    else:
        unavailable.append({'key':'views','label':'Просмотры и вовлечение','hint':'Пока недостаточно накопленной статистики.','action':'Дай системе накопить больше публикаций.','available':False})

    weakest = min(signals, key=lambda x: int(x.get('value') or 0)) if signals else None
    strongest = max(signals, key=lambda x: int(x.get('value') or 0)) if signals else None

    # next_step: pick the single most urgent actionable step
    if not active_channel:
        next_step = "Подключи Telegram-канал в разделе «Настройки → Канал» — без этого автопостинг невозможен."
    elif not onboarding_done:
        next_step = "Заверши онбординг: укажи тему, аудиторию, стиль и форматы, чтобы разблокировать автогенерацию постов."
    elif plan_count == 0:
        next_step = "Добавь минимум 5 идей в контент-план (раздел «План») — без них автопилот остановится."
    elif drafts_count == 0:
        next_step = "Создай хотя бы 3 черновика (раздел «Черновики») — резерв на случай сбоя автогенерации."
    elif schedules_total < 1:
        next_step = "Настрой расписание публикаций (раздел «Автопостинг») — выбери дни и время для автовыхода постов."
    else:
        next_step = clean_text(str(weakest.get('action') if weakest else 'Усиль профиль и запас контента'))
    summary = {
        'score': score,
        'readiness': score,
        'repeat_rate': repeat_rate,
        'text_repeat_rate': text_repeat_rate,
        'media_repeat_rate': media_repeat_rate,
        'queue_items': plan_count + drafts_count,
        'posted_last_7d': posted_last_7d,
        'sample_size': sample_size,
        'views_known': views_known,
        'avg_views': round(avg_views, 2),
        'avg_reactions': round(avg_reactions, 2),
        'avg_comments': round(avg_comments, 2),
        'avg_forwards': round(avg_forwards, 2),
        'plan_count': plan_count,
        'drafts_count': drafts_count,
        'media_count': media_count,
        'schedule_count': schedules_total,
        'onboarding_completed': 1 if onboarding_done else 0,
        'active_channel': 1 if active_channel else 0,
    }
    return {
        'score': score,
        'readiness': score,
        'repeat_rate': repeat_rate,
        'next_step': next_step,
        'strongest_area': strongest,
        'weakest_area': weakest,
        'weakest_key': str(weakest.get('key') if weakest else ''),
        'signals': signals,
        'recommendations': analytics_recommendations(signals, plan_count=plan_count, drafts_count=drafts_count, schedules_total=schedules_total, onboarding_done=onboarding_done, active_channel=bool(active_channel)),
        'summary': summary,
        'top_topics': top_topics,
        'unavailable': unavailable,
        'rubrics': formats,
        'rhythm': 'сильный' if posted_last_7d >= 5 else ('стабильный' if posted_last_7d >= 2 else 'слабый'),
    }

