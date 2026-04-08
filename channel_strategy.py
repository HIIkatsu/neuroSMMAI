from __future__ import annotations

import json
from collections import Counter
from typing import Any


def parse_json_list(value: str | list[str] | None) -> list[str]:
    if isinstance(value, list):
        src = value
    else:
        raw = str(value or '').strip()
        if not raw:
            return []
        try:
            src = json.loads(raw)
        except Exception:
            src = [x.strip() for x in raw.split(',') if x.strip()]
    out: list[str] = []
    seen: set[str] = set()
    for item in src:
        text = str(item or '').strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def normalize_channel_settings(settings: dict[str, Any] | None) -> dict[str, Any]:
    s = dict(settings or {})
    formats = parse_json_list(s.get('channel_formats'))
    constraints = parse_json_list(s.get('content_constraints'))
    return {
        'topic': str(s.get('topic') or '').strip(),
        'audience': str(s.get('channel_audience') or '').strip(),
        'style_preset': str(s.get('channel_style_preset') or '').strip(),
        'style_text': str(s.get('channel_style') or '').strip(),
        'mode': str(s.get('channel_mode') or '').strip(),
        'frequency': str(s.get('channel_frequency') or '').strip(),
        'formats': formats,
        'constraints': constraints,
        'news_enabled': str(s.get('news_enabled') or '0').strip() == '1',
        'news_sources': str(s.get('news_sources') or '').strip(),
        'content_rubrics': str(s.get('content_rubrics') or '').strip(),
        'post_scenarios': str(s.get('post_scenarios') or '').strip(),
    }


def build_format_catalog(topic: str, audience: str, formats: list[str], *, news_enabled: bool = False) -> list[dict[str, str]]:
    core = topic or 'теме канала'
    audience_tail = f' для аудитории «{audience}»' if audience else ''
    normalized = list(formats or [])
    if news_enabled and 'Новости' not in normalized:
        normalized = ['Новости', *normalized]
    fallback = ['Разборы', 'Полезные советы', 'FAQ', 'Кейсы', 'Вовлекающие посты']
    if not normalized:
        normalized = fallback

    catalog: dict[str, dict[str, str]] = {
        'Новости': {
            'label': 'Новости',
            'brief': f'только реальные свежие поводы по теме {core}{audience_tail}; короткая выжимка, что произошло и почему это важно',
            'pattern': f'Короткая выжимка по актуальному событию в теме {core}: что произошло, почему это важно, что это меняет{audience_tail}',
        },
        'Разборы': {
            'label': 'Разборы',
            'brief': f'разбор спорного тезиса, инструмента или подхода в теме {core}{audience_tail}',
            'pattern': f'Разбор по теме {core}: один конкретный тезис, где ошибаются, как смотреть трезво{audience_tail}',
        },
        'Полезные советы': {
            'label': 'Полезные советы',
            'brief': f'практическая польза, инструкции, чек-листы без воды по теме {core}{audience_tail}',
            'pattern': f'Практический пост по теме {core}: конкретный совет или чек-лист{audience_tail}',
        },
        'FAQ': {
            'label': 'FAQ',
            'brief': f'типичный вопрос аудитории и чёткий ответ по теме {core}{audience_tail}',
            'pattern': f'Ответ на частый вопрос аудитории по теме {core}{audience_tail}',
        },
        'Подборки': {
            'label': 'Подборки',
            'brief': f'полезные подборки инструментов, подходов или источников по теме {core}{audience_tail}',
            'pattern': f'Подборка по теме {core}: список реально полезного{audience_tail}',
        },
        'Мифы и ошибки': {
            'label': 'Мифы и ошибки',
            'brief': f'миф, заблуждение или типичная ошибка в теме {core}{audience_tail}',
            'pattern': f'Типичная ошибка в теме {core}: в чём ловушка и как делать лучше{audience_tail}',
        },
        'Кейсы': {
            'label': 'Кейсы',
            'brief': f'кейсы, жизненные ситуации, сценарии из практики по теме {core}{audience_tail}',
            'pattern': f'Кейс или жизненная ситуация по теме {core}: что было, что сработало, вывод{audience_tail}',
        },
        'Вовлекающие посты': {
            'label': 'Вовлекающие посты',
            'brief': f'живой вопрос аудитории по теме {core}{audience_tail}, который запускает обсуждение без мусора',
            'pattern': f'Вопрос аудитории по теме {core}: попросить мнение или опыт{audience_tail}',
        },
    }
    out=[]
    seen=set()
    for name in normalized:
        item = catalog.get(name)
        if not item:
            item = {
                'label': name,
                'brief': f'сильный формат для темы {core}{audience_tail}',
                'pattern': f'Сильный пост в формате «{name}» по теме {core}{audience_tail}',
            }
        if item['label'].lower() in seen:
            continue
        seen.add(item['label'].lower())
        out.append(item)
    return out


def build_frequency_hint(code: str) -> str:
    mapping = {
        'daily_1': 'обычно один сильный пост в день',
        'daily_2': 'до двух разных по формату публикаций в день',
        'weekly_3_5': 'примерно 3–5 публикаций в неделю, ставка на качество',
        'flexible': 'гибкий ритм, публикуем только реально стоящие темы',
    }
    return mapping.get(str(code or '').strip(), 'ритм не задан')


def build_generation_strategy(settings: dict[str, Any] | None) -> dict[str, Any]:
    s = normalize_channel_settings(settings)
    catalog = build_format_catalog(s['topic'], s['audience'], s['formats'], news_enabled=s['news_enabled'])
    constraints = s['constraints']
    constraint_line = '; '.join(constraints) if constraints else 'без специальных ограничений'
    freshness = (
        'Допускается актуальный угол, но без выдумывания дат, цифр, релизов и событий. '
        'Если нет подтверждённого инфоповода, делай evergreen-пост с текущей практической ценностью.'
        if ('Новости' in s['formats'] or s['news_enabled'] or s['style_preset'] == 'Новостной') else
        'Ставка не на псевдоновости, а на сильные практические темы, кейсы, разборы и вопросы аудитории.'
    )
    lines = [f"- {item['label']}: {item['brief']}" for item in catalog]
    return {
        **s,
        'format_catalog': catalog,
        'rubric_lines': lines,
        'rubric_text': '\n'.join(lines),
        'constraint_line': constraint_line,
        'freshness_line': freshness,
        'frequency_hint': build_frequency_hint(s['frequency']),
    }


def build_plan_format_rotation(settings: dict[str, Any] | None, total_items: int) -> list[dict[str, str]]:
    strategy = build_generation_strategy(settings)
    catalog = strategy['format_catalog'] or build_format_catalog(strategy['topic'], strategy['audience'], [])
    if not catalog:
        catalog = build_format_catalog(strategy['topic'], strategy['audience'], ['Разборы'])
    weights = Counter()
    for item in catalog:
        label = item['label']
        weights[label] = 1
        if label in {'Разборы', 'Полезные советы'}:
            weights[label] += 1
        if label == 'Новости' and total_items >= 5:
            weights[label] += 1
    pool: list[dict[str, str]] = []
    for item in catalog:
        for _ in range(weights[item['label']]):
            pool.append(item)
    out = []
    for idx in range(max(1, total_items)):
        out.append(pool[idx % len(pool)])
    return out
