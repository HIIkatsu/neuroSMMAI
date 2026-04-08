from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import re
from typing import Any

import db
from ai_client import ai_chat
from channel_strategy import build_generation_strategy, build_plan_format_rotation
from miniapp_analytics_service import recent_history_lines
from miniapp_shared import clean_text
from news_service import fetch_news_candidates, plan_news_brief


def strip_generated_labels(text: str) -> str:
    raw = str(text or "").replace("**", "").replace("__", "").replace("###", "").strip()
    if not raw:
        return ""
    drop_exact = {
        "заголовок", "title", "headline", "текст поста", "пост",
        "хештеги", "хэштеги", "hashtags", "подпись", "caption",
    }
    cleaned_lines: list[str] = []
    for line in raw.splitlines():
        line = line.strip().strip("—-•*")
        if not line:
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            continue
        lower = line.lower().strip(":").strip()
        if lower in drop_exact:
            continue
        line = re.sub(r"^(заголовок|title|headline|текст поста|пост|хештеги|хэштеги|hashtags|подпись|caption)\s*:\s*", "", line, flags=re.I)
        line = line.strip().strip('"“”«»')
        if line:
            cleaned_lines.append(line)
    while cleaned_lines and cleaned_lines[-1] == "":
        cleaned_lines.pop()
    text = "\n".join(cleaned_lines).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def hashtags_text(text: str, topic: str) -> str:
    import actions
    base = strip_generated_labels(text)
    tags = actions.generate_hashtags(topic or "", base or "")
    if not tags:
        return base
    existing = {part.lower() for part in re.findall(r"#[\wа-яА-ЯёЁ]+", base)}
    filtered = [tag for tag in tags.split() if tag.lower() not in existing]
    if not filtered:
        return base
    return f"{base}\n\n{' '.join(filtered)}".strip()


def normalized_post_text(text: str) -> str:
    return strip_generated_labels(text)


def days_label(days: str) -> str:
    value = (days or "*").strip()
    if value == "*":
        return "Каждый день"
    return ", ".join(p.strip() for p in value.split(",") if p.strip())


def idea_buckets(topic: str) -> list[str]:
    core = topic.strip() or "теме канала"
    return [
        f"Практический разбор частой ошибки в теме {core}",
        f"Что реально важно понимать в теме {core} новичку",
        f"Разница между двумя подходами в теме {core}",
        f"Как оценивать качество решений в теме {core}",
        f"Что обычно недооценивают в теме {core}",
        f"Кейс из практики по теме {core}",
        f"Разбор спорного тезиса по теме {core}",
        f"Что изменилось в подходах к теме {core}",
        f"Подборка рабочих инструментов по теме {core}",
        f"Чек-лист для подписчиков по теме {core}",
        f"Разбор типичного вопроса аудитории по теме {core}",
        f"Как не слить время и деньги в теме {core}",
        f"Сравнение стратегий и подходов в теме {core}",
        f"Практический ориентир: по каким признакам понимать, что в теме {core} всё идет нормально",
    ]


def _headline_to_prompt(title: str, topic: str, domain: str = "") -> str:
    title = clean_text(title)
    base = title or f"свежий тренд в теме {topic}"
    domain_tail = f" (ориентир: {domain})" if domain else ""
    variants = [
        f"Новостной разбор: что на самом деле стоит за темой «{base}» и почему это важно подписчикам{domain_tail}",
        f"Практический пост по свежему инфоповоду: как использовать тему «{base}» в интересах аудитории{domain_tail}",
        f"Разбор тренда без хайпа: что в новости «{base}» реально влияет на людей и рынок{domain_tail}",
        f"Контрастный пост: что в теме «{base}» выглядит громко, а что действительно меняет ситуацию{domain_tail}",
    ]
    return variants[hash(base) % len(variants)]


def _fallback_interest_mix(topic: str) -> list[str]:
    core = topic.strip() or "канала"
    return [
        f"Разбор свежего тренда: что в теме {core} прямо сейчас начинает менять поведение людей",
        f"Анти-хайп пост: какой популярный тезис в теме {core} звучит громко, но на практике слабый",
        f"Полу-продающий разбор: за какой конкретный результат аудитория реально готова платить в теме {core}",
        f"Практический чек-лист: как понять, что в теме {core} вы движетесь в правильную сторону",
        f"Сравнение двух подходов: где в теме {core} чаще теряют время, а где получают результат",
        f"Разбор частой ошибки: что в теме {core} обещают слишком просто, хотя в жизни всё сложнее",
        f"Пост для вовлечения: какой вопрос подписчики стесняются задать по теме {core}, хотя он действительно важный",
        f"Кейс-формат: как один маленький сдвиг в теме {core} даёт заметный результат",
    ]


def extract_json_array(raw: str) -> list[Any]:
    body = (raw or "").strip()
    if not body:
        return []
    body = re.sub(r"^```(?:json)?\s*", "", body, flags=re.I)
    body = re.sub(r"\s*```$", "", body)
    try:
        data = json.loads(body)
        return data if isinstance(data, list) else []
    except Exception:
        pass
    match = re.search(r"(\[.*\])", body, flags=re.S)
    if not match:
        return []
    try:
        data = json.loads(match.group(1))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def sanitize_plan_topic_line(value: str, subject: str) -> str:
    raw = clean_text(strip_generated_labels(value))
    if not raw:
        return f"Полезный пост по теме {subject}"
    raw = re.sub(r"\s+[—-]\s+", ": ", raw)
    raw = re.sub(r"^(заголовок|title|headline)\s*:\s*", "", raw, flags=re.I)
    banned = [
        r"как\s+выбрать\s+мастера",
        r"как\s+найти\s+мастера",
        r"как\s+не\s+ошибиться\s+при\s+выборе\s+мастера",
        r"совет\s+автора",
        r"рекомендац(?:ия|ии)\s+автора",
        r"личн(?:ое|ый)\s+мнение\s+автора",
        r"опрос\b",
        r"давай\s+поговорим\s+о\s+канале",
        r"знакомство\s+с\s+аудиторией",
    ]
    lowered = raw.lower()
    if any(re.search(pattern, lowered, flags=re.I) for pattern in banned):
        return f"Практический пост для подписчиков по теме {subject}"
    raw = re.sub(r"\b(совет автора|рекомендация автора|личное мнение автора)\b", "", raw, flags=re.I).strip(" —:-")
    raw = re.sub(r"\s{2,}", " ", raw).strip()
    if len(raw) < 18:
        raw = f"{raw} — прикладной разбор для подписчиков" if raw else f"Полезный пост по теме {subject}"
    return raw or f"Полезный пост по теме {subject}"


async def _load_strategy_settings(owner_id: int, topic: str) -> dict[str, Any]:
    keys = [
        'topic', 'channel_audience', 'channel_style', 'channel_style_preset', 'channel_mode',
        'channel_formats', 'channel_frequency', 'content_constraints', 'news_enabled',
        'news_sources', 'content_rubrics', 'rubrics_schedule', 'post_scenarios',
        'content_exclusions',
    ]
    settings = await db.get_settings_bulk(keys, owner_id=owner_id)
    if topic and not str(settings.get('topic') or '').strip():
        settings['topic'] = topic
    return settings


async def generate_plan_items_ai(
    start_date: str | None,
    days: int,
    posts_per_day: int,
    topic: str,
    post_time: str,
    *,
    config: Any,
    owner_id: int,
    history: dict[str, list[str]] | None = None,
) -> list[dict[str, str]]:
    subject = clean_text(topic) or 'канала'
    try:
        dt = datetime.strptime(start_date, '%Y-%m-%d').date() if start_date else datetime.now().date()
    except Exception:
        dt = datetime.now().date()

    days = min(max(days, 1), 90)
    posts_per_day = min(max(posts_per_day, 1), 4)
    base_hour, base_minute = 12, 0
    try:
        hh, mm = (post_time or '12:00').split(':')
        base_hour, base_minute = int(hh), int(mm)
    except Exception:
        pass

    total_items = days * posts_per_day
    settings = await _load_strategy_settings(owner_id, subject)
    strategy = build_generation_strategy(settings)
    rotation = build_plan_format_rotation(settings, total_items)
    history_block = recent_history_lines(history, limit=12)
    news_candidates = await fetch_news_candidates(owner_id=owner_id, limit=min(6, max(3, total_items))) if bool(strategy.get('news_enabled', True)) else []
    news_block = plan_news_brief(news_candidates)
    today = datetime.now(timezone.utc).astimezone().strftime('%Y-%m-%d')
    format_targets = '\n'.join(
        f"{idx + 1}. {item['label']}: {item['pattern']}" for idx, item in enumerate(rotation[:min(len(rotation), 18)])
    ) or '- без явных форматов'
    rubric_block = strategy.get('rubric_text') or '- форматы не заданы'
    audience_line = strategy.get("audience") or "не указана"
    style_line = strategy.get("style_preset") or strategy.get("style_text") or "не указан"
    constraint_line = strategy.get("constraint_line") or "без ограничений"
    exclusions_raw = str(settings.get("content_exclusions") or "").strip()
    exclusions_block = f'\nСТРОГО ЗАПРЕЩЕНО в любой идее (не упоминать, не намекать):\n{exclusions_raw}\n' if exclusions_raw else ''
    rubrics_schedule_raw = str(settings.get("rubrics_schedule") or settings.get("content_rubrics") or "").strip()
    rubrics_schedule_block = (
        f'\nРубрики с заданной периодичностью (СТРОГО соблюдай расписание при распределении идей по датам):\n'
        f'{rubrics_schedule_raw}\n'
        f'Например: если рубрика "Мемы - по пятницам" — все идеи этой рубрики должны планироваться на пятницы. '
        f'Если "раз в неделю" — одна идея этой рубрики на весь план. '
        f'Дата начала плана: {dt.isoformat()}, планируем на {days} дн.\n'
    ) if rubrics_schedule_raw else ''
    # Determine news share based on channel mode
    _mode = str(strategy.get("mode") or "").lower()
    _is_news_channel = "новост" in str(subject).lower() or _mode == "news"
    _news_cap = max(2, total_items // 3) if _is_news_channel else max(1, total_items // 5)

    prompt = (
        'Ты сильный редактор и контент-стратег Telegram-канала.\n'
        f'Сегодняшняя дата: {today}.\n'
        f'Тема канала: {subject}.\n'
        f'Нужно придумать {total_items} РАЗНЫХ конкретных идей для будущих постов.\n\n'
        'СТРОГИЕ ТРЕБОВАНИЯ К РАЗНООБРАЗИЮ РУБРИК:\n'
        f'Максимум {_news_cap} из {total_items} идей могут быть новостными. Остальные — другие форматы.\n'
        'Обязательно используй МИНИМУМ 4 разных формата из этого списка:\n'
        '- Разбор частой ошибки / мифа\n'
        '- Практический совет / чек-лист / инструкция\n'
        '- Сравнение двух подходов / продуктов / решений\n'
        '- Мини-кейс / история из практики\n'
        '- FAQ — ответ на частый вопрос аудитории\n'
        '- Неочевидный инсайт / нестандартный взгляд\n'
        '- Подборка / топ / список полезного\n'
        '- Новость с анализом (не просто «произошло X»)\n'
        'НИКОГДА не ставь подряд 2+ идеи одного формата. Чередуй!\n\n'
        'СТРОГИЕ ТРЕБОВАНИЯ К ИДЕЯМ:\n'
        '1. Каждая идея — конкретный угол подачи или формат, НЕ общая тема. '
        'Не «разбор новостей по теме», а «почему большинство [целевой аудитории] делают X неправильно и как это исправить».\n'
        '2. Идея должна быть сразу готова к генерации поста: формат + угол + практическая ценность для аудитории.\n'
        '3. ЗАПРЕЩЕНО: «Напиши пост о новостях», «Расскажи о теме», «Обзор последних событий», '
        '«Советы от автора», «Знакомство с каналом», «Опрос подписчиков» — любые заглушки и мусор.\n'
        '4. Не выдумывай конкретные факты, цифры, релизы или события. Угол подачи — да, ложные факты — нет.\n'
        '5. Каждая идея ОБЯЗАНА учитывать аудиторию, стиль и ограничения канала ниже.\n'
        f'6. КАЖДАЯ идея должна быть НЕПОСРЕДСТВЕННО о теме «{subject}», а не о смежных абстрактных вещах.\n\n'
        f'{exclusions_block}'
        f'{rubrics_schedule_block}'
        f'Профиль канала (ОБЯЗАТЕЛЬНО учитывать в каждой идее):\n'
        f'- Целевая аудитория: {audience_line}\n'
        f'- Стиль подачи: {style_line}\n'
        f'- Режим: {strategy.get("mode") or "не указан"}\n'
        f'- Ритм публикаций: {strategy.get("frequency_hint") or "не указан"}\n'
        f'- Контентные ограничения (строго соблюдать): {constraint_line}\n'
        f'- Форматы и рубрики канала:\n{rubric_block}\n\n'
        f'Желаемое чередование форматов для ближайших идей:\n{format_targets}\n\n'
        f'Недавние материалы — НЕ повторять темы, углы и структуру:\n{history_block}\n\n'
        'Верни только JSON-массив объектов формата:\n'
        '[{"prompt": "..."}]\n\n'
        'Требования к каждому prompt:\n'
        '- одна строка, без заголовка и без markdown;\n'
        '- без конструкции «Заголовок: ...»;\n'
        '- без слов «пост про», «идея для поста», «контент», «автор», «канал»;\n'
        '- должен быть явно виден формат или угол подачи (разбор ошибки / кейс / сравнение / чек-лист / контраст / инсайт);\n'
        '- написан под конкретную аудиторию канала, не для всех подряд;\n'
        '- должен выглядеть интересно живому тематическому каналу, а не как AI-мусор.'
    )
    raw = await ai_chat(
        api_key=config.openrouter_api_key,
        model=config.openrouter_model,
        base_url=getattr(config, 'openrouter_base_url', None),
        temperature=0.82,
        messages=[{'role': 'user', 'content': prompt}],
        max_tokens=min(1200, 90 + total_items * 46),
    )
    data = extract_json_array(raw)
    if not data:
        return generate_plan_items(start_date, days, posts_per_day, subject, post_time, news_candidates=news_candidates)

    out: list[dict[str, str]] = []
    idx = 0
    for day_i in range(max(1, min(days, 90))):
        cur = dt + timedelta(days=day_i)
        for slot in range(max(1, min(posts_per_day, 4))):
            total_minutes = base_hour * 60 + base_minute + slot * 180
            hh = (total_minutes // 60) % 24
            mm = total_minutes % 60
            item = data[idx] if idx < len(data) and isinstance(data[idx], dict) else {}
            idx += 1
            idea = clean_text(str(item.get('prompt') or item.get('idea') or item.get('topic') or item.get('title') or ''))
            if not idea and idx - 1 < len(news_candidates):
                news_item = news_candidates[idx - 1]
                idea = _headline_to_prompt(news_item.get('title', ''), subject, news_item.get('domain', ''))
            if not idea and rotation:
                idea = rotation[(idx - 1) % len(rotation)].get('pattern') or ''
            if not idea:
                fallback_mix = _fallback_interest_mix(subject)
                idea = fallback_mix[(idx - 1) % len(fallback_mix)]
            prompt_text = sanitize_plan_topic_line(idea or f'Интересный пост по теме {subject}', subject)
            out.append({'dt': f'{cur.isoformat()} {hh:02d}:{mm:02d}', 'prompt': prompt_text, 'kind': 'prompt'})
    return out


def generate_plan_items(start_date: str | None, days: int, posts_per_day: int, topic: str, post_time: str, news_candidates: list[dict] | None = None) -> list[dict[str, str]]:
    try:
        dt = datetime.strptime(start_date, '%Y-%m-%d').date() if start_date else datetime.now().date()
    except Exception:
        dt = datetime.now().date()

    days = min(max(days, 1), 90)
    posts_per_day = min(max(posts_per_day, 1), 4)
    base_hour, base_minute = 12, 0
    try:
        hh, mm = (post_time or '12:00').split(':')
        base_hour, base_minute = int(hh), int(mm)
    except Exception:
        pass

    bucket = idea_buckets(topic)
    trendy = _fallback_interest_mix(topic or "канала")
    # Cap news candidates to prevent news overload in fallback plans
    news_ideas = [_headline_to_prompt((item or {}).get('title', ''), topic or 'канала', (item or {}).get('domain', '')) for item in (news_candidates or [])[:4]]
    # Interleave: non-news ideas dominate, news sprinkled in (max 1 in 5)
    non_news = trendy + bucket
    all_ideas: list[str] = []
    ni = 0
    for i, idea in enumerate(non_news):
        all_ideas.append(idea)
        if (i + 1) % 4 == 0 and ni < len(news_ideas):
            all_ideas.append(news_ideas[ni])
            ni += 1
    # Append remaining news at end
    all_ideas.extend(news_ideas[ni:])
    out: list[dict[str, str]] = []
    idx = 0
    for day_i in range(days):
        cur = dt + timedelta(days=day_i)
        for slot in range(posts_per_day):
            total_minutes = base_hour * 60 + base_minute + slot * 180
            hh = (total_minutes // 60) % 24
            mm = total_minutes % 60
            prompt = sanitize_plan_topic_line(all_ideas[idx % len(all_ideas)], topic or 'канала')
            idx += 1
            out.append({'dt': f'{cur.isoformat()} {hh:02d}:{mm:02d}', 'prompt': prompt, 'kind': 'prompt'})
    return out
