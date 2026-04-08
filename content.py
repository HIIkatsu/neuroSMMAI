from __future__ import annotations


def _raise_generation_error(raw: str) -> None:
    msg = (raw or "").strip()
    lowered = msg.lower()

    if not msg:
        raise RuntimeError("Не удалось получить ответ модели")
    if "requires more credits" in lowered:
        raise RuntimeError("Недостаточно кредитов для генерации")
    if "prompt tokens limit exceeded" in lowered:
        raise RuntimeError("Слишком длинный запрос к модели")
    if "can only afford" in lowered:
        raise RuntimeError("Недостаточно доступных токенов для генерации")
    if "error code: 402" in lowered or "openrouter" in lowered:
        raise RuntimeError("Модель не смогла обработать запрос")
    if "```json" in lowered or '"title"' in lowered or '"body"' in lowered:
        raise RuntimeError("Модель вернула ответ в неверном формате")

    if len(msg) > 300:
        msg = msg[:300].rstrip() + "..."

    raise RuntimeError(msg)

import hashlib
import json
import logging
import os
import random
import re
import time
from datetime import datetime, timezone
from typing import Any

from ai_client import ai_chat
from db import get_setting, get_recent_post_topics, get_channel_settings
from channel_strategy import build_generation_strategy
from miniapp_analytics_service import recent_history_lines
from topic_utils import (
    detect_topic_family,
    get_family_guardrails as _get_family_guardrails,
    get_family_post_angles,
    get_family_cta_style,
)

logger = logging.getLogger(__name__)
_ERROR_PREFIXES = ("⚠️", "⏳", "🌐")

# ---------------------------------------------------------------------------
# Autopost single-message budget
# ---------------------------------------------------------------------------
# Autopost text MUST fit in a single Telegram message to avoid split posts.
# For photo posts: caption limit is 1024 chars.
# For text-only posts: message limit is 4096 chars.
# We use conservative values to account for signature/formatting overhead.
AUTOPOST_CAPTION_BUDGET = 900   # for posts with media (under 1024 caption limit)
AUTOPOST_TEXT_BUDGET = 1800     # for text-only posts — compact Telegram format (was 2400)


def enforce_autopost_budget(
    title: str,
    body: str,
    cta: str,
    *,
    has_media: bool = True,
) -> tuple[str, str, str]:
    """Trim autopost text components to fit within a single Telegram message.

    Returns (title, body, cta) trimmed to budget.  Trimming is done
    intelligently at sentence/paragraph boundaries.
    """
    budget = AUTOPOST_CAPTION_BUDGET if has_media else AUTOPOST_TEXT_BUDGET
    full = "\n\n".join(p for p in [title, body, cta] if p)
    if len(full) <= budget:
        return title, body, cta

    # Priority: keep title and cta, trim body
    reserved = len(title) + len(cta) + 4  # separators
    body_budget = max(200, budget - reserved)

    if len(body) > body_budget:
        # Trim body at sentence boundary
        parts = re.split(r"(?<=[.!?…])\s+", body)
        acc: list[str] = []
        cur = 0
        for p in parts:
            cost = len(p) + (2 if acc else 0)
            if cur + cost > body_budget:
                break
            acc.append(p)
            cur += cost
        body = " ".join(acc).strip() if acc else body[:body_budget].rstrip() + "…"

    # Final check
    full = "\n\n".join(p for p in [title, body, cta] if p)
    if len(full) > budget:
        # Drop cta if still over
        cta = ""
        full = "\n\n".join(p for p in [title, body] if p)
        if len(full) > budget:
            body = body[: budget - len(title) - 4].rstrip() + "…"

    return title, body, cta


# ---------- low-level helpers ----------

def clean_text(text: str) -> str:
    text = str(text or "").strip().replace("ё", "е")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _looks_like_ai_error(text: str) -> bool:
    t = (text or "").strip()
    return (not t) or t.startswith(_ERROR_PREFIXES)


def _strip_ai_cliches(text: str) -> str:
    out = str(text or "")
    # NOTE: "контент" deliberately NOT banned — it's a legitimate word for marketing/SMM channels
    banned = [
        "это не просто",
        "это не про",
        "в этой статье",
        "как ии",
        "нейросеть",
        "нейросети",
        "давайте разберемся",
        "давайте разберёмся",
        "глубокое погружение",
        "ключ к успеху",
        "магия",
        "революционный",
        "идеальное решение",
        "сегодня поговорим",
        "сегодня мы поговорим",
        "в современном мире",
        "в современном быстро меняющемся мире",
        "привет, друзья",
        "привет друзья",
        "дорогие друзья",
        "в наше время",
        "как всегда",
        "не секрет, что",
        "не секрет что",
        "хочу поделиться",
        "думаю, многие согласятся",
        "важно понимать",
        "стоит отметить",
        "в заключение",
        "подведём итог",
        "подведем итог",
        "технологии будущего",
        # Weak generic endings that degrade text quality
        "именно поэтому это важно",
        "вот почему это важно",
        "вот почему это так важно",
        "это важно, потому что",
        "в конечном счёте",
        "в конечном счете",
        "в итоге можно сказать",
        "в целом можно сказать",
        "таким образом можно сказать",
        "подводя итог",
        "в общем и целом",
        "резюмируя вышесказанное",
        "как мы видим",
        "как видим из вышесказанного",
        "что в итоге",
        "в конце концов",
        "в заключение хочется отметить",
        "в заключение стоит отметить",
        "на самом деле всё просто",
        "всё начинается с малого",
        "помни об этом",
        "задумайся об этом",
        "именно из таких мелочей",
        "именно такие детали",
        "именно в этом и заключается",
        "вот в чём секрет",
        "вот и весь секрет",
        "пишите в комментариях, что думаете",
        "пишите в комментарии",
        "делитесь мнением в комментариях",
        "ставьте лайк если",
        "подписывайтесь если",
    ]
    for b in banned:
        out = re.sub(re.escape(b), "", out, flags=re.I)
    # Remove "не X, не Y, не Z" repetitive negation chains (3+ repetitions)
    out = re.sub(r"(не\s+\S+,\s*){2,}не\s+\S+", "", out, flags=re.I)
    out = re.sub(r"\n{3,}", "\n\n", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    return out.strip(" \n-—")


# Dismissive/toxic imperatives that damage channel reputation.
# These patterns catch the *class* of problem (dismissive tone toward the niche),
# not a single hardcoded phrase.  Combined into a single alternation for efficiency.
# Only clearly toxic forms are matched; ambiguous phrases are left to the system prompt.
_DISMISSIVE_RE: re.Pattern[str] = re.compile(
    r"\b(?:"
    r"забей(?:те)?\s+(?:на|про)"
    r"|плюнь(?:те)?\s+(?:на|про)"
    r"|наплюй(?:те)?\s+(?:на|про)"
    r"|бросай(?:те)?\s+(?:эту|всё|всё\s+это)"
    r"|бросьте\s+(?:эту|всё|всё\s+это)"
    r"|хватит\s+(?:ныть|жаловаться|морочить)"
    r")\b",
    re.I,
)


def _strip_dismissive_phrases(text: str) -> str:
    """Remove dismissive/toxic imperatives that undermine the promoted niche.

    Matches the *class* of problem (aggressively casual, reputation-damaging
    phrases) rather than a single hardcoded string.  Splits on sentence
    boundaries once, removes sentences matching the pattern, then rejoins.
    """
    out = str(text or "")
    sentences = re.split(r"(?<=[.!?])(?:\s+|$)", out)
    cleaned = [s for s in sentences if s and not _DISMISSIVE_RE.search(s)]
    return " ".join(cleaned).strip()


# Regex to detect @mention handles in generated text
_AT_MENTION_RE = re.compile(r"@[A-Za-zА-ЯЁа-яё0-9_]{2,32}")
# Regex to detect URLs in generated text
_URL_RE = re.compile(r"https?://[^\s\"'<>]{4,}|www\.[^\s\"'<>]{4,}")


def _remove_fabricated_refs(text: str) -> tuple[str, int, int]:
    """Detect and remove fabricated @mentions and URLs from generated text.

    Generated posts must not contain invented @usernames, channel names, or
    links that were not explicitly provided in the input.  This function
    removes them and returns (cleaned_text, at_mentions_removed, urls_removed).
    """
    at_mentions = _AT_MENTION_RE.findall(text)
    urls = _URL_RE.findall(text)

    cleaned = text
    # Remove @mentions entirely (including any trailing punctuation context)
    cleaned = _AT_MENTION_RE.sub("", cleaned)
    cleaned = _URL_RE.sub("", cleaned)

    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    if at_mentions or urls:
        logger.info(
            "FABRICATION_CHECK at_mentions_found=%d urls_found=%d removed=%d text_preview=%r",
            len(at_mentions), len(urls), len(at_mentions) + len(urls),
            cleaned[:120],
        )
    return cleaned, len(at_mentions), len(urls)


def _apply_fabrication_cleanup(bundle: dict) -> None:
    """Apply _remove_fabricated_refs to all text fields in a generated bundle in-place."""
    for field in ("title", "body", "cta", "short"):
        val = bundle.get(field)
        if val:
            cleaned, n_at, n_url = _remove_fabricated_refs(val)
            if n_at or n_url:
                bundle[field] = cleaned


# ---------------------------------------------------------------------------
# Final safety / consistency pass — anti-fabrication for author role
# ---------------------------------------------------------------------------

# Phrases that indicate the LLM is fabricating promises or guarantees
_PROMISE_PATTERNS = re.compile(
    r"гарантирую|гарантируем|100\s*%\s*результат|обещаю|обещаем|"
    r"вылечи[тм]|навсегда избав|полностью устран|мгновенн\w+ результат|"
    r"за\s+\d+\s+дн[яей]+\s+(?:вы|ты)\s+(?:получи|достигн|измени)|"
    r"без\s+риск\w+\s+и\s+последствий",
    re.IGNORECASE,
)

# Minimum length of a line after promise-pattern removal to keep it (otherwise discard)
_MIN_SOFTENED_LINE_LEN = 15

# Phrases that fabricate specific professional credentials
_CREDENTIAL_FABRICATION = re.compile(
    r"(?:диплом|сертификат|лицензи[яию]|аккредитаци[яию]|учёная?\s+степен[ьи]|"
    r"кандидат\s+наук|доктор\s+наук|PhD|MBA|профессор)\s",
    re.IGNORECASE,
)

# Risky / controversial phrases that should be softened
_RISKY_CLAIM_PATTERNS = re.compile(
    r"(?:доказано\s+наукой|учёные\s+доказали|научно\s+подтверждено|"
    r"официально\s+признан\w*|по\s+закону\s+(?:вы|ты)\s+(?:обязан|должен)|"
    r"врачи\s+(?:скрывают|молчат|не\s+расскажут)|правда\s+которую\s+скрывают|"
    r"фарм\w*\s+(?:мафия|лобби|заговор))",
    re.IGNORECASE,
)


def _safety_consistency_pass(text: str, *, author_role_type: str = "", author_role_description: str = "", author_activities: str = "", author_forbidden_claims: str = "") -> str:
    """
    Final safety / consistency pass applied after all generation.

    Rules:
    1. Don't fabricate about the author — remove invented services/credentials
    2. Don't promise guaranteed results
    3. Don't contradict the channel profile
    4. Soften risky/controversial formulations
    """
    if not text or not text.strip():
        return text

    lines = text.split("\n")
    cleaned_lines: list[str] = []

    # Parse forbidden claims into a list of lowered terms for matching
    forbidden_lower: list[str] = []
    if author_forbidden_claims:
        raw_claims = author_forbidden_claims.strip()
        if raw_claims.startswith("["):
            try:
                parsed = json.loads(raw_claims)
                if isinstance(parsed, list):
                    forbidden_lower = [str(c).lower().strip() for c in parsed if str(c).strip()]
            except (ValueError, TypeError):
                pass
        if not forbidden_lower:
            forbidden_lower = [c.strip().lower() for c in re.split(r"[;,\n]+", raw_claims) if c.strip()]

    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append(line)
            continue

        lowered = stripped.lower()

        # 1. Remove sentences containing forbidden author claims
        if forbidden_lower:
            skip_line = False
            for claim in forbidden_lower:
                if claim and claim in lowered:
                    logger.info("SAFETY_PASS: removed line matching forbidden author claim %r: %r", claim, stripped[:100])
                    skip_line = True
                    break
            if skip_line:
                continue

        # 2. Soften promise/guarantee patterns — replace sentence with softened version
        if _PROMISE_PATTERNS.search(stripped):
            softened = _PROMISE_PATTERNS.sub("", stripped).strip()
            if len(softened) < _MIN_SOFTENED_LINE_LEN:
                logger.info("SAFETY_PASS: removed promise line: %r", stripped[:100])
                continue
            cleaned_lines.append(softened)
            logger.info("SAFETY_PASS: softened promise in line: %r", stripped[:100])
            continue

        # 3. Remove fabricated credentials when author profile doesn't mention them
        if author_role_type or author_role_description:
            cred_match = _CREDENTIAL_FABRICATION.search(stripped)
            if cred_match:
                combined_profile = f"{author_role_type} {author_role_description} {author_activities}".lower()
                if cred_match.group(0).lower().strip() not in combined_profile:
                    logger.info("SAFETY_PASS: removed fabricated credential: %r", stripped[:100])
                    continue

        # 4. Remove risky/controversial claims
        if _RISKY_CLAIM_PATTERNS.search(stripped):
            logger.info("SAFETY_PASS: removed risky claim: %r", stripped[:100])
            continue

        cleaned_lines.append(line)

    result = "\n".join(cleaned_lines)
    # Clean up any resulting double blank lines
    result = re.sub(r"\n{3,}", "\n\n", result).strip()
    return result


def _apply_safety_pass(bundle: dict, *, author_role_type: str = "", author_role_description: str = "", author_activities: str = "", author_forbidden_claims: str = "") -> None:
    """Apply _safety_consistency_pass to all text fields in a generated bundle in-place."""
    for field_name in ("title", "body", "cta", "short"):
        val = bundle.get(field_name)
        if val:
            bundle[field_name] = _safety_consistency_pass(
                val,
                author_role_type=author_role_type,
                author_role_description=author_role_description,
                author_activities=author_activities,
                author_forbidden_claims=author_forbidden_claims,
            )


def _author_role_kwargs(settings: dict) -> dict[str, str]:
    """Extract author role fields from settings dict for safety pass kwargs."""
    return {
        "author_role_type": str(settings.get("author_role_type") or ""),
        "author_role_description": str(settings.get("author_role_description") or ""),
        "author_activities": str(settings.get("author_activities") or ""),
        "author_forbidden_claims": str(settings.get("author_forbidden_claims") or ""),
    }


# Patterns that indicate a weak/abstract ending paragraph
_WEAK_ENDING_PATTERNS: list[str] = [
    r"именно поэтому",
    r"вот почему",
    r"это важно потому что",
    r"в мире.{0,15}где",
    r"в эпоху.{0,25}когда",
    r"в современном мире",
    r"задумайтесь",
    r"подумайте об этом",
    r"это не просто.{0,40}это",
    r"это больше чем просто",
    r"помните.{0,20}главное",
    r"главное.{0,20}помнить",
    r"в конечном счёте",
    r"в конечном счете",
    r"суть в том что",
    r"суть проста",
    r"всё просто",
    r"на самом деле всё",
    r"на самом деле все",
    r"мы живём в то время",
    r"мы живем в то время",
    r"это симптом",
    r"это признак",
    r"это тренд",
    r"это не случайно",
    r"это закономерно",
    r"это говорит о",
    r"общий вывод",
    r"главный вывод",
]

_WEAK_ENDING_RE = re.compile(
    "|".join(_WEAK_ENDING_PATTERNS), flags=re.I | re.UNICODE
)


# Max length of a paragraph eligible for weak-ending trimming.
# Paragraphs longer than this are likely real content, not filler.
_MAX_TRIMMABLE_PARAGRAPH_LEN = 250

# Minimum body length (chars) that must remain after trimming a weak ending.
# Prevents trimming from leaving too little content.
_MIN_CONTENT_AFTER_TRIM = 200


def _trim_weak_ending(body: str) -> str:
    """Remove a weak/abstract final paragraph if it adds no value.

    Strategy:
    - Split into paragraphs
    - Check the last paragraph for generic philosophical/moral patterns
    - If weak AND body is still long enough without it, remove it
    - Better to end on a strong concrete paragraph than a padded moral
    """
    if not body:
        return body
    paragraphs = [p.strip() for p in re.split(r"\n\n+", body) if p.strip()]
    if len(paragraphs) <= 1:
        return body  # Nothing to trim — single paragraph

    last = paragraphs[-1]
    # Only trim short-to-medium final paragraphs (long ones are likely content)
    if len(last) > _MAX_TRIMMABLE_PARAGRAPH_LEN:
        return body

    if _WEAK_ENDING_RE.search(last):
        remaining = "\n\n".join(paragraphs[:-1])
        # Only trim if enough content remains
        if len(remaining) >= _MIN_CONTENT_AFTER_TRIM:
            return remaining

    return body



def _extract_json_object(raw: str) -> dict[str, Any]:
    body = str(raw or "").strip()
    if not body:
        return {}

    body = re.sub(r"^\s*```(?:json)?\s*", "", body, flags=re.I)
    body = re.sub(r"\s*```\s*$", "", body, flags=re.I).strip()

    try:
        data = json.loads(body)
        return data if isinstance(data, dict) else {}
    except Exception:
        pass

    start = body.find("{")
    if start >= 0:
        depth = 0
        in_string = False
        escape = False
        end = -1

        for idx, ch in enumerate(body[start:], start=start):
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = idx + 1
                    break

        if end > start:
            candidate = body[start:end].strip()
            try:
                data = json.loads(candidate)
                if isinstance(data, dict):
                    return data
            except Exception:
                pass

    fields = {}
    patterns = {
        "title": r'"title"\s*:\s*"((?:\\.|[^"\\])*)"',
        "body": r'"body"\s*:\s*"((?:\\.|[^"\\])*)"',
        "cta": r'"cta"\s*:\s*"((?:\\.|[^"\\])*)"',
        "short": r'"short"\s*:\s*"((?:\\.|[^"\\])*)"',
        "button_text": r'"button_text"\s*:\s*"((?:\\.|[^"\\])*)"',
        "image_prompt": r'"image_prompt"\s*:\s*"((?:\\.|[^"\\])*)"',
    }
    for key, pat in patterns.items():
        m = re.search(pat, body, flags=re.S | re.I)
        if m:
            val = m.group(1).replace('\\n', '\n').replace('\\"', '"').strip()
            fields[key] = val

    return fields if ("title" in fields and "body" in fields) else {}





def _extract_partial_json_object(raw: str) -> dict[str, Any]:
    body = str(raw or "").strip()
    if not body:
        return {}
    result: dict[str, Any] = {}
    patterns = {
        "title": r'"title"\s*:\s*"((?:[^"\\]|\\.)*)"',
        "body": r'"body"\s*:\s*"((?:[^"\\]|\\.)*)"',
        "cta": r'"cta"\s*:\s*"((?:[^"\\]|\\.)*)"',
        "short": r'"short"\s*:\s*"((?:[^"\\]|\\.)*)"',
        "button_text": r'"button_text"\s*:\s*"((?:[^"\\]|\\.)*)"',
        "image_prompt": r'"image_prompt"\s*:\s*"((?:[^"\\]|\\.)*)"',
    }
    for key, pattern in patterns.items():
        m = re.search(pattern, body, flags=re.S)
        if m:
            try:
                result[key] = bytes(m.group(1), "utf-8").decode("unicode_escape")
            except Exception:
                result[key] = m.group(1).replace('\"', '"').replace("\n", "\n")
    return result

def _cleanup_model_field(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value.strip()
    else:
        text = str(value).strip()
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"^\s*```[a-zA-Z0-9_-]*\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)
    return text.strip("`").strip()


def _extract_plain_text_post(raw: str) -> dict[str, str]:
    text = _strip_ai_cliches(str(raw or ""))
    if not text or _looks_like_ai_error(text):
        return {}
    lowered = text.lower()
    if "```json" in lowered or '"title"' in lowered or '"body"' in lowered:
        return {}
    parts = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if not parts:
        return {}
    title = parts[0][:90].strip(" \"'«»") if len(parts[0].split()) <= 10 else ""
    body = "\n\n".join(parts[1:] if title else parts)
    if not body:
        return {}
    short = " ".join(body.split()[:18]).strip()
    if short and not short.endswith((".", "!", "?", "…")):
        short += "…"
    return {
        "title": title or "Новый пост",
        "body": body,
        "cta": "",
        "short": short,
        "button_text": "Подробнее",
    }

def _hash_seed(*parts: str) -> int:
    raw = "|".join(str(x or "") for x in parts)
    return int(hashlib.sha1(raw.encode("utf-8", "ignore")).hexdigest(), 16)


def _pick(subject: str, items: list[str], salt: str = "") -> str:
    if not items:
        return ""
    idx = _hash_seed(subject, salt, str(int(time.time() // 900))) % len(items)
    return items[idx]


# ---------- topic families / angles ----------

def _topic_family(topic: str, prompt: str) -> str:
    """Wrapper around the unified detect_topic_family from topic_utils."""
    return detect_topic_family(f"{topic} {prompt}")




def _family_angles(family: str, requested: str) -> list[dict[str, str]]:
    if family == "massage":
        return [
            {"label": "client_situation", "opening": "Обычно о массаже вспоминают не в теории, а в тот момент, когда обычный день начинает ощущаться тяжелее, чем должен.", "focus": "узнаваемый бытовой сценарий, понятный повод обратиться к специалисту"},
            {"label": "practical_benefit", "opening": "Лучше всего массаж ценят тогда, когда эффект ощущается не в словах, а в самочувствии, движении и восстановлении в течение дня.", "focus": "ощутимая польза, спокойный профессиональный тон, понятный результат"},
            {"label": "routine_explainer", "opening": "Хороший пост про массаж полезен тогда, когда показывает, как процедура вписывается в обычную неделю человека без пафоса и лишних обещаний.", "focus": "рутина, регулярность, восстановление, повседневная польза"},
            {"label": "trust_building", "opening": "Доверие к специалисту появляется не из громких слов, а из понятного процесса, аккуратной работы и ожидаемой пользы.", "focus": "доверие, экспертность, взрослая подача без оправданий"},
            {"label": "soft_conversion", "opening": "Мягко-продающий пост про массаж работает лучше, когда помогает человеку яснее понять свой запрос, а не когда пытается продавить решение.", "focus": "бережная продажа, польза, понятный следующий шаг"},
            {"label": "useful_observation", "opening": "В теме массажа сильнее всего цепляют не громкие тезисы, а узнаваемые наблюдения из обычной жизни.", "focus": "наблюдение, конкретика, жизненная узнаваемость"},
        ]
    if family == "food":
        return [
            {"label": "sensory_appeal", "opening": "В еде сильнее всего цепляет не описание блюда, а момент, когда человек ощущает вкус ещё до первого укуса.", "focus": "чувственный язык, ароматы, текстуры, вкусовые образы"},
            {"label": "practical_recipe", "opening": "Лучшие кулинарные посты — те, которые человек может применить уже сегодня, не переделывая весь распорядок дня.", "focus": "конкретный рецепт, доступные ингредиенты, понятная техника"},
            {"label": "product_story", "opening": "За каждым хорошим ингредиентом стоит история: откуда, как обрабатывается, почему этот, а не другой.", "focus": "история продукта, фермер/производитель, причина выбора"},
            {"label": "trend_taste", "opening": "Гастрономические тренды интересны не сами по себе, а когда понятно, как и зачем их попробовать прямо сейчас.", "focus": "тренд через личный опыт, практичная точка входа"},
            {"label": "chef_insight", "opening": "Секреты хорошей кухни — это не магия, а техника, которую можно освоить, если знать, на что обращать внимание.", "focus": "лайфхак шефа, профессиональный приём для домашней кухни"},
            {"label": "experience_angle", "opening": "Хороший ресторанный опыт часто определяется не тем, что в меню, а тем, как всё ощущается в совокупности.", "focus": "атмосфера, сервис, момент посещения, эмоция"},
        ]
    if family == "health":
        return [
            {"label": "practical_habit", "opening": "Здоровье в повседневной жизни строится не из грандиозных решений, а из небольших привычек, которые легко встроить в обычный день.", "focus": "конкретная привычка, простой шаг, реальный сценарий"},
            {"label": "myth_debunk", "opening": "Вокруг здоровья много убедительно звучащих советов, которые на деле работают совсем не так, как принято думать.", "focus": "разбор мифа, доказательная база, спокойный тон"},
            {"label": "body_signal", "opening": "Тело чаще всего заранее подаёт сигналы, что что-то идёт не так — важно знать, на что обращать внимание.", "focus": "симптом или сигнал, когда стоит отреагировать, без запугивания"},
            {"label": "evidence_insight", "opening": "Исследования в области здоровья часто приходят к выводам, которые несколько расходятся с привычными установками.", "focus": "данные, исследования, практический вывод с оговорками"},
            {"label": "prevention_tip", "opening": "Профилактика работает лучше лечения — но только когда меры конкретные, применимые и не требуют героических усилий.", "focus": "профилактика, простое действие, понятная польза"},
            {"label": "professional_view", "opening": "Взгляд специалиста на бытовой вопрос о здоровье часто оказывается заметно спокойнее, чем то, что пишут в интернете.", "focus": "экспертный взгляд, доказательность, уважение к читателю"},
        ]
    if family == "beauty":
        return [
            {"label": "product_review", "opening": "Хорошее средство или процедура в бьюти — это когда разница заметна в жизни, а не только на красивом фото.", "focus": "честный отзыв, реальный результат, конкретный продукт"},
            {"label": "technique_tip", "opening": "В уходе за собой техника часто важнее продукта: правильные движения, последовательность и время нанесения меняют результат.", "focus": "техника нанесения/процедуры, профессиональный приём"},
            {"label": "ingredient_breakdown", "opening": "Состав косметики — это не просто список, а понятие о том, что реально работает на твою кожу или волосы.", "focus": "разбор состава, ключевой ингредиент, практический вывод"},
            {"label": "routine_builder", "opening": "Бьюти-рутина работает не потому что она дорогая, а потому что она регулярная и подходит конкретному типу кожи.", "focus": "построение рутины, последовательность, персональный подход"},
            {"label": "trend_filter", "opening": "Не каждый бьюти-тренд стоит следовать — иногда разумнее понять, кому он подходит, а кому нет.", "focus": "тренд через призму практики, для кого работает"},
            {"label": "expert_advice", "opening": "Хороший специалист в бьюти-индустрии часто даёт советы, которые идут вразрез с маркетинговыми обещаниями, но реально помогают.", "focus": "профессиональный совет, честность, доверие к эксперту"},
        ]
    if family == "local_business":
        return [
            {"label": "process_transparency", "opening": "Лучший способ завоевать доверие клиента — не обещать больше, а показать, как ты работаешь изнутри.", "focus": "прозрачность процесса, показать компетентность через детали"},
            {"label": "client_case", "opening": "Реальный кейс клиента — это самое честное доказательство того, как работает сервис на практике.", "focus": "кейс с результатом, конкретная задача, решение, итог"},
            {"label": "client_mistake", "opening": "Чаще всего клиент сталкивается с проблемой не из-за сложности задачи, а из-за одной типичной ошибки в самом начале.", "focus": "частая ошибка клиента, как её избежать, польза"},
            {"label": "why_specialist", "opening": "Попытки сэкономить на специалисте часто стоят дороже, чем профессиональное решение с первого раза.", "focus": "аргумент в пользу специалиста, понятная логика, доверие"},
            {"label": "seasonal_tip", "opening": "В каждом сезоне есть задачи, которые лучше решать заранее, чтобы не попасть в очередь или не столкнуться с форс-мажором.", "focus": "сезонный совет, своевременность, практическая польза"},
            {"label": "master_observation", "opening": "Из практики мастера чаще всего выходят самые полезные советы — те, которые нельзя найти в общих статьях.", "focus": "наблюдение из опыта, живой профессиональный голос"},
        ]
    if family == "education":
        return [
            {"label": "skill_breakdown", "opening": "Хороший образовательный пост — тот, после которого человек понимает что-то новое или умеет делать что-то лучше.", "focus": "конкретный навык, понятные шаги, применимость прямо сейчас"},
            {"label": "myth_in_learning", "opening": "В учёбе и обучении живёт много стойких заблуждений, которые мешают людям прогрессировать быстрее.", "focus": "разбор мифа об обучении, спокойный аргументированный тон"},
            {"label": "student_story", "opening": "История ученика, который добился результата, работает лучше любой рекламы — особенно когда в ней есть конкретный путь.", "focus": "история с конкретным путём и результатом, без пафоса"},
            {"label": "method_tip", "opening": "Метод обучения часто важнее количества времени, потраченного на учёбу.", "focus": "техника обучения, инструмент, подход, применимость"},
            {"label": "resource_pick", "opening": "Хорошая подборка ресурсов по нужной теме экономит часы поиска и помогает двигаться в нужном направлении.", "focus": "ресурсы, инструменты, подборка с объяснением"},
            {"label": "faq_answer", "opening": "Самые частые вопросы учеников чаще всего указывают на то, что объясняется плохо, — и хороший пост закрывает именно это.", "focus": "ответ на частый вопрос, ясность, применимость"},
        ]
    if family == "finance":
        return [
            {"label": "instrument_breakdown", "opening": "Финансовые инструменты чаще всего выглядят сложнее, чем они есть, если смотреть на маркетинговые описания, а не на механику.", "focus": "как работает инструмент, плюсы и минусы, для кого подходит"},
            {"label": "common_mistake", "opening": "Самые дорогие ошибки в личных финансах — не экзотические, а те, которые делает большинство, думая, что поступает разумно.", "focus": "типичная ошибка, как её избежать, практический вывод"},
            {"label": "market_signal", "opening": "Рынок часто даёт сигналы заранее, и их интерпретация — навык, который отличает уверенного инвестора от растерянного.", "focus": "рыночный сигнал, интерпретация, что это значит для читателя"},
            {"label": "planning_tip", "opening": "Финансовое планирование работает не из-за сложных схем, а из-за нескольких простых, но регулярных действий.", "focus": "конкретный шаг в планировании, применимость, без обещаний"},
            {"label": "myth_debunk", "opening": "Многие финансовые мифы звучат убедительно — и именно это делает их опасными для тех, кто хочет копить или инвестировать.", "focus": "миф в финансах, доказательная база, осторожный тон"},
            {"label": "trend_analysis", "opening": "Финансовый тренд интересен не сам по себе, а тем, что он означает для конкретного человека с конкретными деньгами.", "focus": "тренд через призму читателя, практический вывод"},
        ]
    if family == "marketing":
        return [
            {"label": "tactic_breakdown", "opening": "Маркетинговые тактики работают не сами по себе, а в системе — и понять, когда применять каждую, важнее, чем знать все сразу.", "focus": "конкретная тактика, когда работает, примеры"},
            {"label": "case_study", "opening": "Реальный кейс с цифрами и выводами — самый честный ответ на вопрос «а это вообще работает?».", "focus": "кейс с метриками, что сработало и почему"},
            {"label": "tool_review", "opening": "Хороший маркетинговый инструмент — тот, который экономит время и даёт измеримый результат, а не просто красиво выглядит.", "focus": "инструмент, как использовать, что даёт"},
            {"label": "mistake_analysis", "opening": "Ошибки в маркетинге часто выглядят как нормальная практика — именно поэтому их так сложно замечать изнутри.", "focus": "ошибка в маркетинге, как распознать, как исправить"},
            {"label": "platform_insight", "opening": "Понимание алгоритмов и поведения аудитории на платформе — это то, что отличает контент, который работает, от контента ради контента.", "focus": "инсайт по платформе, алгоритм, поведение аудитории"},
            {"label": "trend_filter", "opening": "Не каждый маркетинговый тренд стоит внедрять — иногда важнее понять, подходит ли он под конкретную нишу и аудиторию.", "focus": "тренд через фильтр ниши и аудитории"},
        ]
    if family == "lifestyle":
        return [
            {"label": "personal_story", "opening": "Лучшие лайфстайл-посты — те, в которых человек узнаёт что-то из своей жизни, даже если история чужая.", "focus": "личная история, узнаваемость, человеческий тон"},
            {"label": "habit_tip", "opening": "Небольшие привычки, встроенные в обычный день, часто меняют качество жизни заметнее, чем громкие решения.", "focus": "конкретная привычка, как начать, реальный эффект"},
            {"label": "observation", "opening": "Точное наблюдение о повседневной жизни цепляет сильнее, чем советы, которые начинаются со слова «надо».", "focus": "наблюдение, узнаваемость, живой язык"},
            {"label": "recommendation", "opening": "Хорошая рекомендация — книги, фильма, места или привычки — работает только тогда, когда за ней стоит личный опыт.", "focus": "рекомендация с объяснением почему, без навязывания"},
            {"label": "trend_reflection", "opening": "Жизненные тренды стоит рассматривать не как инструкцию, а как повод подумать, что из этого подходит именно тебе.", "focus": "тренд через личную призму, без давления"},
            {"label": "reflection", "opening": "Иногда полезно остановиться и сформулировать то, что давно ощущалось, но ещё не было сказано вслух.", "focus": "рефлексия, глубина, тихий человеческий тон"},
        ]
    if family == "expert_blog":
        return [
            {"label": "expert_opinion", "opening": "Авторский взгляд на привычную тему часто оказывается ценнее, чем подборка общеизвестных фактов.", "focus": "уникальная позиция автора, экспертный угол"},
            {"label": "myth_debunk", "opening": "В любой профессиональной нише есть устойчивые заблуждения, которые принято не оспаривать — до тех пор, пока не разобраться детально.", "focus": "разбор заблуждения с опорой на опыт и знание"},
            {"label": "practice_insight", "opening": "Лучшие профессиональные инсайты — те, которые появляются только из реальной практики, а не из книг.", "focus": "опыт из практики, что на самом деле работает"},
            {"label": "nonobvious_angle", "opening": "Иногда самый полезный взгляд на ситуацию — тот, который идёт вразрез с популярным мнением.", "focus": "нестандартная точка зрения, обоснованная позиция"},
            {"label": "controversial_take", "opening": "Честная позиция по спорному вопросу всегда полезнее, чем нейтральный ответ «с одной стороны — с другой стороны».", "focus": "чёткая позиция, аргументация, взрослый тон"},
            {"label": "mini_case", "opening": "Небольшой кейс из профессиональной жизни даёт больше, чем абстрактный совет — особенно когда виден реальный результат.", "focus": "мини-кейс, конкретный путь, практический вывод"},
        ]
    if family == "cars":
        return [
            {"label": "practical_benefit", "opening": "Автомобильная тема становится сильной тогда, когда из неё можно вынести спокойное прикладное решение для повседневной жизни.", "focus": "реальная польза, обслуживание, выбор, расходы без драматизации"},
            {"label": "routine_explainer", "opening": "Большинство автомобильных вопросов важны не сами по себе, а в тот момент, когда от них зависит удобство и предсказуемость машины каждый день.", "focus": "повседневное использование, обслуживание, практический сценарий"},
            {"label": "decision_help", "opening": "Хороший авто-пост помогает принимать решения спокойнее: без споров ради споров, без крикливых выводов и без дешёвого давления.", "focus": "выбор, диагностика, обслуживание, понятная логика"},
            {"label": "trust_building", "opening": "В сервисной автомобильной теме доверие строится на ясности: что проверять, когда чинить и за что действительно стоит платить.", "focus": "доверие, компетентность, практический контроль"},
            {"label": "soft_conversion", "opening": "Мягко-продающий пост в авто-нише должен не пугать, а показывать человеку более спокойный и понятный путь решения задачи.", "focus": "бережная продажа, польза, экономия времени и нервов"},
            {"label": "useful_observation", "opening": "Лучше всего в авто-теме работают наблюдения, которые человек узнаёт на своей машине уже в ближайшую неделю.", "focus": "узнаваемость, прикладной вывод, реальная жизнь владельца"},
        ]
    if family == "gaming":
        return [
            {"label": "player_situation", "opening": "Игровая тема цепляет сильнее, когда отталкивается от привычек обычного игрока, а не от громкости инфоповода.", "focus": "игровой опыт, реальный сценарий, привычки игроков"},
            {"label": "value_pick", "opening": "Хороший игровой пост помогает понять, где у игрока реально появляется ценность: во времени, впечатлениях или выборе платформы.", "focus": "ценность, выбор, деньги и время игрока"},
            {"label": "trend_explainer", "opening": "Даже когда тема завязана на тренде, сильный игровой пост переводит его в понятный вывод для человека за клавиатурой.", "focus": "тренды без пустого хайпа, что это меняет для игрока"},
            {"label": "useful_observation", "opening": "В игровых темах лучше всего работают не крики про индустрию, а точные наблюдения, в которых игрок узнаёт себя.", "focus": "наблюдение, узнаваемость, взрослый тон"},
        ]
    if family == "hardware":
        return [
            {"label": "decision_help", "opening": "Сильный пост про железо начинается не с восторга от новинки, а с ясного вопроса: что это меняет для реального пользователя.", "focus": "выбор, баланс, практический сценарий использования"},
            {"label": "practical_benefit", "opening": "В теме железа ценность ощущается там, где человек понимает не характеристики, а итоговый выигрыш в комфорте, скорости или сроке службы.", "focus": "прикладная польза, рациональный выбор"},
            {"label": "trend_explainer", "opening": "Тренды в железе интересны только тогда, когда из них можно сделать нормальный вывод без маркетингового шума.", "focus": "новинки, рынок, взрослая интерпретация без хайпа"},
            {"label": "useful_observation", "opening": "Лучшие посты про ПК и железо часто строятся на одном точном наблюдении, которое сразу собирает картину выбора.", "focus": "узнаваемая ошибка выбора, конкретика, спокойная подача"},
            {"label": "soft_conversion", "opening": "Мягкая продажа в теме железа работает только тогда, когда читатель чувствует, что ему помогают выбрать разумнее, а не просто продают громче.", "focus": "помощь в выборе, доверие, аккуратная продажа"},
        ]
    return [
        {"label": "practical_benefit", "opening": "Сильный пост лучше всего начинается с понятной жизненной пользы, а не с попытки выглядеть громче темы.", "focus": "польза, конкретика, узнаваемый сценарий"},
        {"label": "useful_observation", "opening": "Чаще всего человека цепляет одно точное наблюдение, после которого тема становится ближе и понятнее.", "focus": "наблюдение, узнаваемость, взрослый тон"},
        {"label": "trend_explainer", "opening": "Даже актуальную тему лучше раскрывать через понятный вывод для человека, а не через сам инфоповод.", "focus": "актуальность без лишнего шума"},
        {"label": "decision_help", "opening": "Хороший экспертный пост помогает принять решение спокойнее и яснее, чем до прочтения.", "focus": "решение, польза, практический вывод"},
    ]


def _opening_key(text: str) -> str:
    raw = clean_text(text or "")
    if not raw:
        return ""
    first = re.split(r"[.!?\n]", raw, maxsplit=1)[0]
    words = [w for w in re.sub(r"[^a-zа-я0-9\s]", " ", first.lower()).split() if len(w) >= 3]
    return " ".join(words[:9])


def _recent_opening_keys(recent_posts: list[str] | None) -> set[str]:
    return {_opening_key(x) for x in (recent_posts or []) if _opening_key(x)}


def _choose_angle(topic: str, prompt: str, recent_posts: list[str] | None = None, recent_plan: list[str] | None = None) -> dict[str, str]:
    family = _topic_family(topic, prompt)
    recent_text = " ".join((recent_posts or [])[:8] + (recent_plan or [])[:8]).lower()
    used_openings = _recent_opening_keys(recent_posts)
    candidates = _family_angles(family, prompt)
    scored: list[tuple[int, dict[str, str]]] = []
    for item in candidates:
        score = 100
        if item["label"] in recent_text:
            score -= 18
        if _opening_key(item["opening"]) in used_openings:
            score -= 28
        if item["label"] in {"practical_benefit", "decision_help", "useful_observation", "trust_building"}:
            score += 8
        if item["label"] in {"soft_conversion", "trend_explainer"}:
            score += 3
        scored.append((score, item))
    scored.sort(key=lambda x: (-x[0], x[1]["label"]))
    top = [x[1] for x in scored[:max(2, min(4, len(scored)))]]
    salt = str(int(time.time() // 120))
    return top[_hash_seed(topic, prompt, salt, str(len(recent_posts or []))) % len(top)]



# ---------- fallback posts ----------

def _massage_variants(requested: str) -> list[dict[str, str]]:
    return [
        {
            "title": "Почему после массажа нет эффекта",
            "body": "Часто проблема не в массаже, а в ожиданиях. Кто-то хочет снять напряжение за один сеанс, кто-то надеется, что боль уйдёт сама без изменений в нагрузке, сне и привычках. В итоге человек вроде бы сходил на массаж, а через день говорит, что «всё вернулось».\n\nОбычно заметный эффект появляется там, где запрос понятный: зажатая шея, тяжесть в плечах, перегруз после тренировок, усталость от сидячей работы. И лучше всего он держится, когда человек понимает, зачем идёт на сеанс и какого результата ждёт в реальной жизни.",
            "cta": "Если хочешь, следующим постом могу разобрать, с какими запросами на массаж чаще всего действительно есть смысл идти.",
            "short": "Чаще всего люди разочаровываются в массаже не из-за процедуры, а из-за неправильных ожиданий.",
        },
        {
            "title": "По каким сигналам пора на массаж",
            "body": "Тело обычно заранее показывает, что уже накопился перегруз: к вечеру тяжелеют плечи, хуже двигается шея, после сна нет ощущения лёгкости, а поясница ноет даже без явной причины. Многие тянут до последнего и начинают считать такое состояние нормой.\n\nНа деле это как раз те ситуации, где массаж может дать понятную пользу: снять лишнее напряжение, вернуть свободу движения и помочь телу быстрее восстановиться. Хороший ориентир простой: после сеанса жить и двигаться должно стать легче, а не просто приятно полежать на столе.",
            "cta": "Если нужно, могу отдельно сделать пост про сигналы перегруза у шеи, спины и поясницы.",
            "short": "Тело почти всегда заранее показывает, что напряжение уже накопилось.",
        },
        {
            "title": "За что клиент возвращается снова",
            "body": "На массаж возвращаются не из-за красивых обещаний, а из-за понятного результата. Когда после сеанса становится легче держать осанку, уменьшается тяжесть в плечах и тело двигается свободнее, доверие появляется само. Именно поэтому сильнее работает не абстрактная реклама, а честный разговор о том, какой результат человек реально может заметить в жизни.\n\nДля специалиста это важный ориентир: люди ценят не общие слова про заботу о себе, а понятный эффект. Чем яснее клиент понимает, зачем он идёт и что должен почувствовать после, тем выше шанс, что массаж станет для него полезной привычкой, а не разовой попыткой «что-то попробовать».",
            "cta": "Если интересно, могу следующим постом разобрать, какие ошибки чаще всего съедают эффект уже в первый день.",
            "short": "Главная ценность массажа — не процесс, а понятный результат для человека.",
        },
        {
            "title": "Когда массаж особенно нужен",
            "body": "Есть периоды, когда тело особенно быстро накапливает перегруз: много сидячей работы, стресс, тренировки без нормального восстановления, работа руками или привычка терпеть дискомфорт до последнего. В такие моменты массаж полезен не как редкая роскошь, а как понятный способ вовремя снять лишнее напряжение и не доводить себя до состояния, когда уже всё мешает.\n\nСмысл не в громких словах про wellness, а в обычной жизни. Когда после процедуры человеку легче повернуть шею, свободнее дышится и проще проживать день без постоянного ощущения зажатости, вот тогда массаж начинает восприниматься как реальная польза, а не просто приятный ритуал.",
            "cta": "Если хочешь, могу продолжить эту тему постом о том, как встроить массаж в обычный график без перегруза.",
            "short": "Массаж сильнее всего раскрывается тогда, когда тело уже подаёт явные сигналы перегруза.",
        },
        {
            "title": "Массаж и хроническое напряжение: что важно понимать",
            "body": "Хроническое мышечное напряжение отличается от обычной усталости: оно накапливается незаметно и не проходит само после сна или выходного. Плечи не расслабляются к утру, шея снова зажата к полудню, а спина нагружается даже в позах, которые должны быть удобными.\n\nМассаж в таком случае работает лучше всего не как разовая процедура, а как регулярная практика с понятной целью. Задача не «размять тело», а системно снизить уровень фонового напряжения так, чтобы организм успевал восстанавливаться между нагрузками.",
            "cta": "",
            "short": "Хроническое напряжение накапливается незаметно — массаж помогает, когда применяется систематически.",
        },
        {
            "title": "Частая ошибка: приходить только когда совсем плохо",
            "body": "Многие приходят на массаж тогда, когда уже совсем не могут игнорировать дискомфорт. Шея болит неделю, плечи скованы с утра, или спина даёт о себе знать при каждом движении. В этот момент специалисту труднее работать, а телу нужно больше сеансов, чтобы вернуться в нормальное состояние.\n\nПрофилактический подход проще: приходить раньше, когда напряжение ещё умеренное. Именно тогда тело откликается быстрее, эффект держится дольше, и сам человек меньше страдает между сеансами.",
            "cta": "",
            "short": "Чем раньше приходишь на массаж — тем легче и дольше держится результат.",
        },
        {
            "title": "Три вещи, которые усиливают эффект массажа",
            "body": "Сеанс массажа даёт хороший старт, но итоговый результат во многом зависит от того, что происходит после. Три вещи реально помогают удержать эффект дольше.\n\nПервое — вода. Тело легче восстанавливается, когда после процедуры достаточно пить. Второе — движение. Умеренная активность в следующий день помогает «закрепить» расслабление мышц и не дать напряжению вернуться сразу. Третье — тепло. Контрастный душ или грелка на проработанную зону усиливают эффект расслабления и помогают мышцам дольше оставаться в спокойном состоянии.",
            "cta": "",
            "short": "Результат массажа держится дольше, если правильно провести ближайшие 24 часа после сеанса.",
        },
        {
            "title": "Самомассаж: когда помогает, а когда нет",
            "body": "Самомассаж — полезный инструмент, если понимать его место. Он хорошо снимает лёгкую усталость, помогает после тренировок, расслабляет шею после долгой работы за компьютером. Но он не заменяет работу специалиста там, где нужна глубокая проработка или точная работа с конкретными мышцами.\n\nПростое правило: если дискомфорт уходит через 5–10 минут самомассажа и не возвращается в течение дня, это хороший знак. Если нет, или если напряжение появляется снова к вечеру, стоит обратиться к профессионалу.",
            "cta": "",
            "short": "Самомассаж — хороший помощник, но не замена профессиональной работе с мышцами.",
        },
    ]

def _generic_fallback_post(topic: str, prompt: str, recent_posts: list[str] | None = None) -> dict[str, str]:
    requested = (prompt or topic or "теме канала").strip()
    family = _topic_family(topic, prompt)
    if family == "massage":
        variants = _massage_variants(requested)
        chosen = variants[_hash_seed(requested, str(len(recent_posts or []))) % len(variants)]
        return {**chosen, "button_text": "Подробнее"}
    title = requested[:1].upper() + requested[1:] if requested else "Новый пост"
    body = (
        f"Тема «{requested}» лучше всего раскрывается через конкретную пользу для читателя: где люди чаще ошибаются, на что реально смотреть в первую очередь и какой вывод можно применить уже сейчас.\n\n"
        f"Сильный пост по теме «{requested}» должен не просто объяснять, а помогать принять решение, избежать лишних трат времени и увидеть практический смысл в теме без воды и шаблонов."
    )
    return {
        "title": title,
        "body": body,
        "cta": "Если нужно, могу продолжить это серией из нескольких коротких постов под разными углами.",
        "short": f"{title} — коротко и по делу.",
        "button_text": "Подробнее",
    }


# ---------- owner settings ----------

async def _load_owner_strategy_settings(
    owner_id: int | None,
    topic: str,
    channel_style: str,
    content_rubrics: str,
    post_scenarios: str,
    channel_audience: str,
    content_constraints: str,
) -> dict[str, Any]:
    """Load strategy settings for content generation.

    Uses get_channel_settings() which reads from the active channel profile first,
    then falls back to owner-level settings — so this is channel-aware.
    """
    settings = {
        "topic": topic,
        "channel_style": channel_style,
        "content_rubrics": content_rubrics,
        "post_scenarios": post_scenarios,
        "channel_audience": channel_audience,
        "content_constraints": content_constraints,
        "content_exclusions": "",
    }
    if not owner_id:
        return settings
    try:
        import db as _db_mod  # local import to avoid circular deps
        ch_settings = await _db_mod.get_channel_settings(owner_id)

        mapping = {
            "topic": "topic",
            "channel_style": "channel_style",
            "channel_style_preset": "channel_style_preset",
            "content_rubrics": "content_rubrics",
            "post_scenarios": "post_scenarios",
            "channel_audience": "channel_audience",
            "content_constraints": "content_constraints",
            "content_exclusions": "content_exclusions",
            "channel_mode": "channel_mode",
            "channel_frequency": "channel_frequency",
            "channel_formats": "channel_formats",
            "news_enabled": "news_enabled",
            "news_sources": "news_sources",
            "author_role_type": "author_role_type",
            "author_role_description": "author_role_description",
            "author_activities": "author_activities",
            "author_forbidden_claims": "author_forbidden_claims",
        }
        for key, target in mapping.items():
            value = ch_settings.get(key)
            if value not in (None, ""):
                settings[target] = value
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning("owner strategy settings load failed: %s", exc)
    return settings


def _blend_instruction(channel_topic: str, requested: str) -> str:
    ct = (channel_topic or "").strip()
    rq = (requested or "").strip()
    if not ct or not rq or ct.lower() == rq.lower():
        return "Раскрывай тему прямо, без лишних уводов в сторону."
    return (
        f"Главная тема поста — «{rq}». Тему канала «{ct}» используй только как мягкий угол подачи: "
        "пример, полезный мостик для аудитории или контекст, но не подмену основной темы. "
        "Ориентир по пропорции: около 85% текста про запрошенную тему и до 15% — аккуратная привязка к теме канала."
    )


def _normalize_lengths(title: str, body: str, cta: str, short: str, button_text: str) -> dict[str, str]:
    title = clean_text(title)[:86].strip(' "\'«»')
    if len(title.split()) > 10:
        title = " ".join(title.split()[:10]).strip()
    body = str(body or "").strip()
    body = re.sub(r"\n{3,}", "\n\n", body)
    body = re.sub(r"(^|\n)\s*\d+\.\s+", r"\1— ", body)
    body = re.sub(r"\n\s*[-•]\s+", "\n— ", body)
    body = _strip_ai_cliches(body)
    # Trim weak abstract endings — better to end strong than pad with generic moral
    body = _trim_weak_ending(body)
    if len(body) > 450:
        parts = re.split(r"(?<=[.!?…])\s+", body)
        acc = []
        cur = 0
        for p in parts:
            if cur + len(p) + (1 if acc else 0) > 420:
                break
            acc.append(p)
            cur += len(p) + (1 if acc else 0)
        body = " ".join(acc).strip() or body[:420].rstrip() + "…"
    cta = clean_text(cta)[:120]
    short = clean_text(short)[:120]
    button_text = clean_text(button_text)[:32] or "Подробнее"
    return {
        "title": title or "Новый пост",
        "body": body,
        "cta": cta,
        "short": short or "Короткий анонс поста.",
        "button_text": button_text,
    }







def _derive_post_intent(topic: str, prompt: str, body: str) -> str:
    src = clean_text(" ".join([topic or "", prompt or "", body or ""])).lower()
    family = _topic_family(topic, prompt)
    checks = [
        (("новост", "тренд", "сейчас", "свеж", "новинк"), 'news'),
        (("ошибк", "миф", "заблужд", "разбор"), 'myth_busting'),
        (("сравнен", " vs ", "или", "лучше", "хуже"), 'comparison'),
        (("как ", "почему", "что делать", "совет", "пошаг"), 'howto'),
        (("кейс", "истор", "опыт", "пример"), 'case'),
        (("купить", "запис", "заказать", "скидк", "цен", "услуг"), 'selling'),
    ]
    intent = 'educational'
    for keys, label in checks:
        if any(k in src for k in keys):
            intent = label
            break
    if family in {"massage", "cars"} and intent in {"news", "myth_busting", "comparison"}:
        return 'howto' if 'как ' in src or 'что делать' in src else 'educational'
    if family == 'gaming' and intent == 'comparison' and 'новые игры' in src:
        return 'educational'
    return intent




def _derive_visual_brief(topic: str, prompt: str, title: str, body: str, intent: str) -> str:
    family = _topic_family(topic, prompt)
    src = clean_text(" ".join([prompt or "", title or "", body or ""])).lower()
    chunks: list[str] = []
    if intent == 'news':
        chunks.append('editorial photo')
    if family == 'massage':
        chunks.extend(['therapeutic massage', 'realistic photo'])
        if any(k in src for k in ('шея', 'neck', 'плеч')):
            chunks.extend(['neck shoulders', 'massage session'])
        elif any(k in src for k in ('спина', 'back', 'поясниц')):
            chunks.extend(['back massage', 'lower back'])
        else:
            chunks.extend(['massage therapist hands', 'calm studio'])
    elif family == 'cars':
        chunks.extend(['automotive', 'realistic photo'])
        if any(k in src for k in ('аккум', 'акб', 'battery')):
            chunks.extend(['car battery', 'engine bay'])
        elif any(k in src for k in ('электросамокат', 'самокат', 'scooter')):
            chunks.extend(['electric scooter', 'battery repair'])
        elif any(k in src for k in ('салон', 'interior', 'cockpit')):
            chunks.extend(['car interior', 'dashboard'])
        else:
            chunks.extend(['car exterior', 'service bay'])
    elif family == 'gaming':
        chunks.extend(['gaming setup', 'pc gaming', 'realistic photo'])
        if any(k in src for k in ('пк', 'pc', 'компьют', 'желез')):
            chunks.extend(['computer desk', 'monitor'])
        else:
            chunks.extend(['gaming desk', 'screen'])
    elif family == 'hardware':
        chunks.extend(['computer hardware', 'pc setup', 'realistic photo'])
        if any(k in src for k in ('ноут', 'laptop')):
            chunks.extend(['laptop workspace'])
        else:
            chunks.extend(['desktop workstation'])
    elif family == 'tech':
        chunks.extend(['technology', 'software', 'professional'])
        if any(k in src for k in ('ai', 'ии', 'нейросет', 'machine learning', 'gpt', 'llm')):
            chunks.extend(['artificial intelligence', 'data visualization'])
        elif any(k in src for k in ('devops', 'cloud', 'облак', 'сервер', 'docker', 'kubernetes')):
            chunks.extend(['server infrastructure', 'data center'])
        elif any(k in src for k in ('код', 'программ', 'разработ', 'developer', 'coding')):
            chunks.extend(['code screen', 'developer workspace'])
        else:
            chunks.extend(['technology workspace', 'professional office'])
    elif family == 'business':
        chunks.extend(['business', 'professional', 'office'])
        if any(k in src for k in ('маркетинг', 'marketing', 'smm', 'контент')):
            chunks.extend(['marketing strategy', 'analytics dashboard'])
        elif any(k in src for k in ('финанс', 'finance', 'инвестиц', 'investment')):
            chunks.extend(['financial data', 'investment chart'])
        else:
            chunks.extend(['business meeting', 'corporate workspace'])
    else:
        words = [w for w in re.findall(r"[a-zа-я0-9]+", f"{prompt} {title}") if len(w) > 3]
        chunks.extend(words[:6])
    title_words = [w for w in re.findall(r"[a-zA-Zа-яА-Я0-9]+", f"{prompt} {title}") if len(w) > 4][:3]
    chunks.extend(title_words)
    out = " ".join(dict.fromkeys(chunks))
    return clean_text(out)[:180]


def _title_root(title: str) -> str:
    words = [w for w in re.findall(r"[a-zа-я0-9]+", clean_text(title).lower()) if len(w) >= 4]
    return " ".join(words[:4])


def _recent_title_roots(items: list[str] | None) -> set[str]:
    return {_title_root(x) for x in (items or []) if _title_root(x)}




def _family_high_interest_hooks(family: str) -> list[str]:
    mapping = {
        "food": [
            "какой вкус или аромат человек сможет воссоздать дома уже сегодня",
            "какой ингредиент или техника меняет блюдо сильнее всего",
            "что делает ресторан или рецепт по-настоящему особенным прямо сейчас",
            "какой кулинарный тренд стоит попробовать и почему именно этот",
        ],
        "health": [
            "какая привычка даёт максимальную пользу при минимальных усилиях",
            "что исследования говорят о популярном совете по здоровью",
            "какой сигнал тела стоит заметить раньше, чем это станет проблемой",
            "как сделать заботу о здоровье реальной частью обычного дня",
        ],
        "beauty": [
            "какое средство или техника реально меняет результат, а не просто красиво описано",
            "что опытный специалист делает иначе, чем большинство людей дома",
            "какой тренд в уходе стоит попробовать и для кого он подходит",
            "как выбрать косметику под свой тип кожи без лишних трат",
        ],
        "local_business": [
            "что делает мастера заслуживающим доверия — конкретно и без лишних слов",
            "как понять, что работа сделана по-настоящему хорошо",
            "какую ошибку клиента легко избежать, если знать заранее",
            "что реально важно при выборе местного специалиста или сервиса",
        ],
        "education": [
            "какой навык стоит освоить прямо сейчас и как начать за разумное время",
            "что реально мешает учиться быстрее и эффективнее",
            "какой метод или инструмент меняет качество обучения",
            "что лучший учитель делает иначе, чем средний",
        ],
        "finance": [
            "какой финансовый шаг даёт наибольший эффект при минимальных усилиях",
            "что большинство людей делают неправильно со своими деньгами",
            "как понять, стоит ли конкретный инструмент внимания прямо сейчас",
            "что рынок или экономика означают для обычного человека",
        ],
        "marketing": [
            "какая тактика реально работает в текущих условиях, а не только в теории",
            "что отличает контент, который продаёт, от контента, который просто существует",
            "как измерить, работает ли маркетинг — без лишней сложности",
            "какой инструмент экономит больше всего времени при хорошем результате",
        ],
        "lifestyle": [
            "какая небольшая привычка реально меняет качество дня",
            "что человек узнаёт из чужой истории, как из своей",
            "какое наблюдение о жизни стоит озвучить, пока его не сказал никто другой",
            "что делает обычный день чуть лучше — конкретно и без пафоса",
        ],
        "expert_blog": [
            "какой нестандартный взгляд эксперта расходится с популярным мнением",
            "что опыт специалиста показывает о реальном положении дел в нише",
            "какой честный вывод из практики помогает избежать типичной ошибки",
            "что знают опытные специалисты и чего не знает большинство",
        ],
        "massage": [
            "какой понятный результат человек замечает в обычном дне после хорошего сеанса",
            "как массаж вписывается в нормальный ритм жизни без пафоса и волшебства",
            "какие бытовые сигналы чаще всего подсказывают, что телу нужно восстановление",
            "что люди особенно ценят в спокойной и профессиональной работе специалиста",
        ],
        "cars": [
            "какое решение по машине делает жизнь владельца спокойнее уже на этой неделе",
            "что в обслуживании автомобиля реально влияет на удобство и предсказуемость",
            "как смотреть на выбор и ремонт машины без лишнего шума и споров",
            "какой практический вывод водитель может применить сразу",
        ],
        "gaming": [
            "что в игровой теме реально меняет опыт обычного игрока прямо сейчас",
            "какие игровые новинки стоят внимания не по шуму, а по реальной ценности",
            "что помогает игроку выбирать игры спокойнее и разумнее",
            "какой вывод из тренда можно применить без лишнего хайпа",
        ],
        "hardware": [
            "какое решение по железу выглядит разумнее в реальном использовании, а не в рекламе",
            "что в выборе ПК или ноутбука действительно влияет на ежедневный комфорт",
            "какой практический вывод помогает не переплачивать за характеристики ради характеристик",
            "как увидеть реальную ценность новинки без маркетингового шума",
        ],
        "tech": [
            "какое технологическое решение реально упрощает работу разработчика уже сейчас",
            "что из новинок в IT стоит внимания не по хайпу, а по практической пользе",
            "как принять обоснованное техническое решение без перегрузки информацией",
            "какой инструмент или подход экономит время в повседневной разработке",
        ],
        "business": [
            "какое бизнес-решение действительно влияет на результат, а не только на отчётность",
            "что из маркетинговых практик работает в реальности, а не только в теории",
            "как предприниматель может увидеть ситуацию яснее без лишнего шума",
            "какой практический шаг может улучшить показатели уже на этой неделе",
        ],
        "generic": [
            "какой практический вывод человек может применить сразу после прочтения",
            "что в теме реально меняется для обычного человека без громких слов",
            "какое точное наблюдение помогает увидеть тему яснее",
            "что здесь полезно не в теории, а в реальной жизни",
        ],
    }
    return mapping.get(family, mapping["generic"])


def _structure_variants() -> list[str]:
    return [
        "reader situation -> useful explanation -> practical takeaway",
        "specific observation -> why it matters -> clear conclusion",
        "common real-life scenario -> better approach -> reader benefit",
        "practical question -> calm explanation -> next step",
        "recognizable problem -> grounded insight -> actionable result",
    ]


def _family_guardrails(family: str) -> str:
    """Returns generation guardrails for a given topic family. Uses unified rules from topic_utils."""
    return _get_family_guardrails(family)


def _freshness_line(family: str, requested: str) -> str:
    hook = _pick(requested, _family_high_interest_hooks(family), salt="trend-hook")
    structure = _pick(requested, _structure_variants(), salt="structure")
    return (
        f"Сделай подачу живой, уверенной и редакторски взрослой, но без искусственного конфликта, дешёвого хайпа и сенсационности. "
        f"Рабочий угол: «{hook}». Структура: {structure}."
    )




def _recent_phrases(recent_posts: list[str] | None, limit: int = 10) -> list[str]:
    phrases: list[str] = []
    for item in (recent_posts or [])[:limit]:
        raw = clean_text(item)
        if not raw:
            continue
        first = re.split(r"[.!?\n]", raw, maxsplit=1)[0].strip()
        if len(first) >= 16:
            phrases.append(first[:110])
    return phrases[:6]


def _word_set(text: str) -> set[str]:
    raw = re.sub(r"[^a-zа-яё0-9\s]", " ", clean_text(text).lower())
    return {w for w in raw.split() if len(w) >= 4}


def _similarity_ratio(a: str, b: str) -> float:
    sa = _word_set(a)
    sb = _word_set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(min(len(sa), len(sb)), 1)


def _too_similar_to_recent(title: str, body: str, recent_posts: list[str] | None) -> bool:
    candidate = clean_text(f"{title} {body}")
    for item in recent_posts or []:
        if _similarity_ratio(candidate, item) >= 0.48:
            return True
    return False





def _is_service_expert_niche(channel_topic: str, requested: str) -> bool:
    """Detect if the channel is a service/expert niche where trust-building tone matters."""
    family = _topic_family(channel_topic, requested)
    # Families where the content tone should emphasize trust, expertise and credibility
    return family in {
        "massage", "health", "beauty", "local_business",
        "education", "finance", "expert_blog",
    }

def _safety_violations(channel_topic: str, requested: str, title: str, body: str, cta: str) -> list[str]:
    text = clean_text(" ".join([channel_topic, requested, title, body, cta])).lower()
    body_text = clean_text(body).lower()
    violations: list[str] = []
    for phrase in [
        "опасно обращаться", "опасно ремонтировать", "не стоит обращаться", "не доверяйте частным",
        "только у дилера", "только у официального", "вне гарантийных сервисов опасно",
        "не ходите к", "хуже чем", "обманывают все", "без иллюзий", "вся правда",
        "вам делали неправильно", "клиенты ждут слишком многого", "не вина мастера",
    ]:
        if phrase in text:
            violations.append(f"репутационно токсичная формулировка: {phrase}")
    if any(word in text for word in ["мошенник", "развод", "обман", "афер", "ужас", "катастроф"]):
        violations.append("избыточно токсичная или обвинительная подача")
    if _is_service_expert_niche(channel_topic, requested):
        risky = [
            "не лечит", "не работает", "опасно", "не обращайтесь", "не ходите", "только у нас", "только к нам", "массаж не лечит", "не лечит то, чего нет",
            "вы сами виноваты", "это не вина мастера", "клиент виноват", "большинство клиентов ждёт",
            "вам просто кажется", "перераспределяет боль", "лечения", "диагноз", "гарантированно",
            "реабилитация", "за 10 минут", "мгновенно", "сразу уберет", "сразу уберёт",
        ]
        if any(p in text for p in risky):
            violations.append("слишком медицинская, обещающая или спорная подача для экспертной услуги")
        if any(p in body_text for p in ["не лечит", "не работает", "не убирает", "не спасет", "не спасёт"]):
            violations.append("слишком жёсткая формулировка про услугу")
        if any(p in text for p in ["вы виноваты", "вы не поняли", "клиенты ждут", "без иллюзий", "вина мастера", "не вина мастера"]):
            violations.append("текст ставит аудиторию или автора в токсичную позицию")
    if len(re.findall(r"!", body or "")) >= 3:
        violations.append("слишком крикливая подача")
    return violations


def _voice_variant() -> str:
    variants = [
        "спокойный экспертный тон без пафоса",
        "живой редакторский тон с конкретикой",
        "плотная практическая подача без воды",
        "человеческий разговорный тон без фамильярности",
    ]
    return variants[int(time.time() * 1000) % len(variants)]


# ---------------------------------------------------------------------------
# Generation strategy modes — explicit angle archetypes that guide prompt
# toward specific content patterns instead of generic retries
# ---------------------------------------------------------------------------

GENERATION_STRATEGY_MODES: list[dict[str, str]] = [
    {
        "mode": "pain_problem",
        "label": "Проблема / боль аудитории",
        "prompt_hint": (
            "Начни с конкретной проблемы, боли или дискомфорта, с которой читатель реально сталкивается. "
            "Первый абзац — узнаваемая ситуация. Потом — разбор и практическое решение. "
            "НЕ начинай с абстрактного рассуждения. Начни с ситуации из жизни."
        ),
    },
    {
        "mode": "myth_busting",
        "label": "Развенчание мифа",
        "prompt_hint": (
            "Возьми распространённое заблуждение по теме и разбери, почему оно неверно. "
            "Первый абзац — сформулируй миф как цитату или расхожее мнение. "
            "Потом — спокойно разбери с конкретикой, почему это не работает или работает иначе."
        ),
    },
    {
        "mode": "trend_explanation",
        "label": "Тренд / что меняется",
        "prompt_hint": (
            "Расскажи о заметном изменении, тренде или сдвиге в нише. "
            "Первый абзац — что именно меняется (конкретный факт или наблюдение). "
            "Потом — что это значит для читателя практически. Без общих слов о «будущем»."
        ),
    },
    {
        "mode": "practical_takeaway",
        "label": "Практический вывод / инструкция",
        "prompt_hint": (
            "Дай одну конкретную практическую рекомендацию, которую читатель может применить сразу. "
            "Первый абзац — ситуация, в которой этот совет нужен. "
            "Потом — чёткий алгоритм или объяснение, почему именно так. Не философия, а действие."
        ),
    },
    {
        "mode": "contrarian_take",
        "label": "Нестандартный взгляд / противоположное мнение",
        "prompt_hint": (
            "Представь точку зрения, которая противоречит общепринятой или мейнстримной позиции по теме. "
            "Первый абзац — чётко обозначь, с чем ты не согласен и почему. "
            "Потом — аргументированная позиция с конкретными наблюдениями. Не провокация ради провокации."
        ),
    },
    {
        "mode": "checklist_minicase",
        "label": "Чеклист / мини-кейс",
        "prompt_hint": (
            "Разбери конкретную ситуацию или дай краткий пошаговый чеклист из 3-5 пунктов. "
            "Первый абзац — контекст: кому, когда и зачем это нужно. "
            "Потом — конкретные шаги или разбор случая. Каждый пункт — 1-2 предложения, не абзацы."
        ),
    },
]


def _pick_strategy_mode(
    family: str,
    recent_posts: list[str] | None = None,
    recent_plan: list[str] | None = None,
) -> dict[str, str]:
    """Select a strategy mode that hasn't been used recently for this channel.

    HEURISTIC: scans recent posts and plan for mode labels/keywords to avoid
    repeating the same archetype. Falls back to weighted random.
    """
    recent_text = " ".join((recent_posts or [])[:8] + (recent_plan or [])[:8]).lower()

    # Score each mode: prefer modes whose keywords are NOT in recent text
    scored: list[tuple[int, dict[str, str]]] = []
    for mode in GENERATION_STRATEGY_MODES:
        score = 100
        mode_kw = mode["mode"].replace("_", " ")
        # Penalise if mode label keywords appear in recent posts
        if mode_kw in recent_text or mode["label"].lower() in recent_text:
            score -= 40
        # Light bonus for high-value modes
        if mode["mode"] in ("practical_takeaway", "pain_problem"):
            score += 5
        scored.append((score, mode))

    scored.sort(key=lambda x: (-x[0], x[1]["mode"]))
    top = [x[1] for x in scored[:max(3, len(scored) // 2)]]
    # Deterministic-ish selection based on current time window
    idx = int(time.time() // 600) % len(top)
    return top[idx]


# ---------------------------------------------------------------------------
# Role-specific prompt blocks — concrete writing constraints per author type
# ---------------------------------------------------------------------------

_ROLE_PROMPT_BLOCKS: dict[str, str] = {
    "media": (
        "ГОЛОС КАНАЛА — МЕДИА / РЕДАКЦИЯ:\n"
        "- Пиши от третьего лица как редакция новостного / информационного канала.\n"
        "- ЗАПРЕЩЕНО: «я», «мой опыт», «мои клиенты», «наши подписчики». Никакого личного тона.\n"
        "- Допустимый hook: факт, цифра, цитата источника, заявление эксперта. НЕ личное наблюдение.\n"
        "- Конкретика: ссылайся на события, данные, тренды. Не на «свой опыт».\n"
        "- Формат конкретики: «По данным X...», «Как сообщает Y...», «В 2024 году Z...»"
    ),
    "expert": (
        "ГОЛОС КАНАЛА — ЭКСПЕРТ / СПЕЦИАЛИСТ:\n"
        "- Пиши от первого лица как практикующий специалист.\n"
        "- Допустимо: «я», «мой опыт», «мои клиенты», «в моей практике».\n"
        "- ЗАПРЕЩЕНО: выдумывать конкретные кейсы, цифры, истории пациентов/клиентов.\n"
        "- Допустимый hook: наблюдение из практики, типичная ситуация клиента, профессиональный инсайт.\n"
        "- Если не уверен в факте — пиши «по опыту» или «часто бывает», а не «исследования показали»."
    ),
    "master": (
        "ГОЛОС КАНАЛА — МАСТЕР / ПРАКТИК:\n"
        "- Пиши от первого лица как мастер-практик.\n"
        "- Допустимо: «я», «ко мне приходят», «из моего опыта».\n"
        "- ЗАПРЕЩЕНО: выдумывать кейсы, придумывать цифры, врать про результаты.\n"
        "- Допустимый hook: рабочий момент, типичная ошибка клиента, профессиональное наблюдение.\n"
        "- Тон: спокойный, уверенный, без пафоса. Мастер делится, а не поучает."
    ),
    "business_owner": (
        "ГОЛОС КАНАЛА — ВЛАДЕЛЕЦ БИЗНЕСА:\n"
        "- Пиши от лица бизнеса/команды. Допускается «мы», «наша команда», «наш продукт».\n"
        "- ЗАПРЕЩЕНО: «я как блогер», «мой личный опыт» — это не личный блог.\n"
        "- Допустимый hook: ситуация клиента, бизнес-наблюдение, решение задачи команды.\n"
        "- Конкретика: реальные процессы, решения, результаты для клиентов."
    ),
    "brand": (
        "ГОЛОС КАНАЛА — БРЕНД:\n"
        "- Пиши от лица бренда. «Мы», «наш продукт», «для наших клиентов».\n"
        "- ЗАПРЕЩЕНО: личный блогерский тон, «я заметил», «мой опыт».\n"
        "- Допустимый hook: польза продукта, ситуация клиента, решение через продукт/сервис.\n"
        "- Тон: уверенный, заботливый, профессиональный. Не фамильярный и не рекламный."
    ),
    "blogger": (
        "ГОЛОС КАНАЛА — БЛОГЕР / АВТОР:\n"
        "- Пиши живо, от первого лица, с личными наблюдениями и эмоциями.\n"
        "- Допустимо: «я», «заметил», «попробовал», «мне кажется».\n"
        "- ЗАПРЕЩЕНО: писать сухо как корпоративная рассылка или новостной дайджест.\n"
        "- Допустимый hook: личное наблюдение, история из жизни, неожиданный поворот.\n"
        "- Формат: человеческий, разговорный, без казённых оборотов."
    ),
    "educator": (
        "ГОЛОС КАНАЛА — ПРЕПОДАВАТЕЛЬ / ОБРАЗОВАТЕЛЬНЫЙ КАНАЛ:\n"
        "- Пиши как опытный преподаватель или наставник.\n"
        "- Допустимо: «я», «в моей практике», но без маркетинговых штампов.\n"
        "- ЗАПРЕЩЕНО: «мои клиенты», «наш продукт», фамильярный блогерский тон.\n"
        "- Допустимый hook: учебный пример, типичное заблуждение, практическое упражнение.\n"
        "- Тон: ясный, структурированный, дружелюбный, без снисходительности."
    ),
}


def _get_role_prompt_block(author_role_type: str) -> str:
    """Return role-specific prompt block. Falls back to empty string."""
    role = (author_role_type or "").strip().lower()
    return _ROLE_PROMPT_BLOCKS.get(role, "")





def _title_quality_issues(title: str, family: str) -> list[str]:
    t = clean_text(title or "")
    low = t.lower()
    issues: list[str] = []
    if not t:
        return ["пустой заголовок"]
    if len(t) < 12:
        issues.append("слишком короткий заголовок")
    if len(t) > 110:
        issues.append("слишком длинный заголовок")

    cheap_patterns = [
        "шок", "сенсац", "вся правда", "без иллюз", "вас обманы", "никто не говорит",
        "срочно", "немедленно", "за 5 минут", "за 10 минут", "мгновенно",
        "гарантированно", "100%", "без ошибок", "без усилий", "без вложений",
        "деньги на ветер", "не выбрасывать деньги", "терять деньги", "развод", "кошмар",
    ]
    if any(pat in low for pat in cheap_patterns):
        issues.append("дешёвый или манипулятивный кликбейт в заголовке")

    hostile_patterns = [
        "не работает", "разочарован", "разочарование", "вина мастера", "не вина мастера",
        "опасно", "бесполезно", "обман", "ошибка клиента",
    ]
    if any(pat in low for pat in hostile_patterns):
        issues.append("репутационно токсичная или конфликтная подача в заголовке")

    if low.count("как ") > 1:
        issues.append("перегруженный заголовок")

    family = (family or "").strip().lower()
    if family in {"massage", "health", "service", "cars", "auto", "repair", "battery"}:
        service_overpromises = [
            "вылеч", "исправим все", "исправим всё", "починим все", "починим всё",
            "оживить", "навсегда", "моментально", "сразу решит", "гарантия результата",
        ]
        if any(pat in low for pat in service_overpromises):
            issues.append("слишком обещающий заголовок для экспертной или сервисной ниши")

    return issues

def _quality_issues(channel_topic: str, requested: str, normalized: dict[str, str], recent_posts: list[str] | None) -> list[str]:
    issues: list[str] = []
    title = normalized.get("title", "")
    body = normalized.get("body", "")
    cta = normalized.get("cta", "")
    family = _topic_family(channel_topic, requested)
    text = clean_text(f"{title} {body} {cta}").lower()
    if _too_similar_to_recent(title, body, recent_posts):
        issues.append("слишком похоже на недавние тексты")
    issues.extend(_title_quality_issues(title, family))
    issues.extend(_safety_violations(channel_topic, requested, title, body, cta))
    for phrase in [
        "в 2026 году", "революцион", "магия", "идеальное решение", "только у нас",
        "за 10 минут", "мгновенно", "без усилий", "без лишних движений", "как не выбрасывать деньги",
        "как оживить", "где реальная ценность", "что реально тянет", "как выбрать без сожалений",
    ]:
        if phrase in text:
            issues.append(f"дешёвая или слишком обещающая формулировка: {phrase}")
    if family in {"massage", "cars"} and any(x in text for x in ["реабилитация", "пациенты", "диагноз", "лечение"]):
        issues.append("неуместный медицинский или слишком формальный тон для сервисной ниши")
    if family == "gaming" and any(x in text for x in ["покупка нового пк", "автомобил", "машина"]):
        issues.append("текст ушёл в соседнюю тему вместо исходного игрового запроса")
    roots = [_title_root(x) for x in (recent_posts or [])[:8] if _title_root(x)]
    current_root = _title_root(title)
    if current_root and current_root in roots:
        issues.append("слишком похожий смысловой шаблон заголовка")
    return list(dict.fromkeys(issues))


# ---------------------------------------------------------------------------
# Text quality assessment — engineering-level pre-publish quality gate
# ---------------------------------------------------------------------------
# Criteria are intentionally transparent and rule-based (not black-box).
# Each check returns (score_delta, reason) so that the final decision is
# explainable and adjustable.

_WATER_PHRASES: list[str] = [
    "в наше время", "в современном мире", "как известно", "не секрет",
    "все знают", "каждый знает", "всем известно", "давно доказано",
    "стоит отметить", "нельзя не отметить", "следует отметить",
    "в рамках данной темы", "в контексте этого", "из этого следует",
    "как правило", "по большому счету", "по большому счёту",
    "на самом деле", "по факту", "если честно",
    "сегодня мы поговорим", "давайте разберёмся", "давайте разберемся",
    "поговорим о том", "сегодня расскажем", "в этой статье",
    "в этом посте", "в нашем канале",
]

_TAUTOLOGY_ROOTS: list[tuple[str, str]] = [
    ("важно", "важность"),
    ("нужно", "нужность"),
    ("полезно", "полезность"),
    ("интересно", "интересность"),
    ("уникальн", "уникальн"),
    ("качествен", "качеств"),
]

# Common Russian function words excluded from repetition detection — these naturally
# appear many times in any text and should not count as tautology.
_COMMON_WORDS_EXCLUDE: set[str] = {
    "этого", "после", "когда", "перед", "более", "можно", "нужно", "этот", "чтобы",
    "также", "именно", "очень", "всего", "может", "будет", "свой", "было", "были",
}

# ---------------------------------------------------------------------------
# Semantic topic clusters — related concepts that indicate topic relevance
# even without exact token overlap.  HEURISTIC: hand-curated per family.
# ---------------------------------------------------------------------------

_TOPIC_SEMANTIC_CLUSTERS: dict[str, list[str]] = {
    "massage": ["мышц", "спин", "шея", "плечи", "тело", "напряжен", "расслаблен", "сеанс", "процедур", "восстановлен", "осанк"],
    "health": ["организм", "тело", "профилактик", "привычк", "самочувств", "врач", "иммунитет", "сон", "стресс", "давлен", "питани"],
    "food": ["вкус", "блюд", "ингредиент", "рецепт", "кухн", "продукт", "готов", "аромат", "подач"],
    "beauty": ["кож", "уход", "средств", "процедур", "крем", "маск", "волос", "косметик", "результат"],
    "finance": ["деньг", "бюджет", "инвести", "доход", "расход", "экономи", "капитал", "вклад", "кредит", "накоплен"],
    "marketing": ["аудитори", "контент", "трафик", "конверси", "бренд", "продвижен", "клиент", "лид", "реклам", "воронк"],
    "tech": ["код", "разработк", "сервер", "api", "систем", "архитектур", "деплой", "инфраструктур", "баз данн"],
    "education": ["учеб", "навык", "курс", "обучен", "знани", "практик", "студент", "метод", "урок"],
    "local_business": ["клиент", "сервис", "качеств", "заказ", "обслуживан", "мастер", "работ", "специалист"],
    "cars": ["двигател", "автомобил", "ремонт", "обслуживан", "модел", "пробег", "запчаст", "колес"],
    "gaming": ["игр", "геймпле", "сюжет", "онлайн", "мультиплеер", "релиз", "платформ"],
    "hardware": ["процессор", "видеокарт", "оперативн", "компьютер", "ноутбук", "монитор", "сборк"],
    "business": ["компани", "предпринимател", "прибыл", "управлен", "команд", "стратеги", "рынок", "клиент"],
    "lifestyle": ["жизн", "привычк", "комфорт", "баланс", "отношен", "путешеств", "дом"],
    "expert_blog": ["эксперт", "практик", "опыт", "вывод", "наблюден", "анализ", "мнени"],
}


def assess_text_quality(
    title: str,
    body: str,
    cta: str,
    *,
    channel_topic: str = "",
    requested: str = "",
    author_role_type: str = "",
) -> tuple[int, list[str], dict[str, int]]:
    """Multi-criteria quality assessment for generated posts.

    Returns (score, reasons, dimension_scores) where:
    - score is 0-100 (higher = better)
    - reasons is a list of human-readable explanations for penalties
    - dimension_scores maps each quality dimension to its sub-score (0-10)

    10 quality dimensions (each scored 0-10, sum = 0-100):
    1. hook — strength of the opening / first sentence
    2. specificity — concrete facts vs vague water
    3. value — useful information density
    4. naturalness — language quality, no AI slop
    5. topic_fit — relevance to channel topic
    6. role_fit — consistency with author role/persona
    7. honesty — absence of fabrication / clickbait
    8. density — text length and information-per-word ratio
    9. readability — sentence variety, flow, scannability
    10. publish_ready — can be published without manual editing

    Designed to be transparent, adjustable, and NOT a black box.
    """
    full_text = clean_text(f"{title} {body} {cta}")
    lower_text = full_text.lower()
    body_lower = clean_text(body).lower()
    body_words = len(body.split()) if body else 0
    dims: dict[str, int] = {}
    reasons: list[str] = []

    # --- 1. HOOK: strength of the opening ---
    hook_score = 10
    first_line = body.split("\n", 1)[0].strip() if body else ""
    first_sentence = re.split(r"[.!?\n]", first_line, maxsplit=1)[0].strip()
    if not first_sentence:
        hook_score = 0
        reasons.append("hook: пустое начало текста")
    else:
        first_lower = first_sentence.lower()
        boring_openers = [
            "в наше время", "в современном мире", "все знают", "давайте",
            "поговорим", "сегодня", "многие люди", "не секрет",
            "последнее время", "все мы знаем", "ни для кого не секрет",
            "многие задаются вопросом", "как известно",
            "каждый из нас", "не будем лукавить", "важно понимать",
            "стоит отметить", "нужно понимать", "в этом посте",
            "в этой статье", "давайте разберёмся", "давайте разберемся",
        ]
        if any(first_lower.startswith(opener) for opener in boring_openers):
            hook_score -= 6
            reasons.append("hook: скучный/шаблонный заход")
        if len(first_sentence.split()) < 4:
            hook_score -= 2
            reasons.append("hook: слишком короткое первое предложение")
        # Bonus: question or number in opening = stronger hook
        if re.search(r"\d", first_sentence) and hook_score >= 6:
            hook_score = min(10, hook_score + 1)
    dims["hook"] = max(0, hook_score)

    # --- 2. SPECIFICITY: concrete vs vague ---
    spec_score = 10
    water_hits = sum(1 for phrase in _WATER_PHRASES if phrase in lower_text)
    if water_hits >= 4:
        spec_score = 0
        reasons.append(f"specificity: текст состоит из воды ({water_hits} водных фраз)")
    elif water_hits >= 3:
        spec_score -= min(9, water_hits * 3)
        reasons.append(f"specificity: слишком много воды ({water_hits} водных фраз)")
    elif water_hits >= 1:
        spec_score -= min(5, water_hits * 2)
        reasons.append(f"specificity: водные фразы ({water_hits})")
    # Bonus for numbers/facts
    numbers_count = len(re.findall(r"\d+[%₽$€]?", full_text))
    if numbers_count >= 2 and spec_score < 10:
        spec_score = min(10, spec_score + 1)
    dims["specificity"] = max(0, spec_score)

    # --- 3. VALUE: useful information density ---
    value_score = 10
    # Check for empty-calorie phrases that add words without meaning
    filler_phrases = [
        "стоит отметить", "нельзя не отметить", "следует отметить",
        "важно понимать", "нужно понимать", "стоит задуматься",
        "нельзя недооценивать", "играет важную роль", "является ключевым",
        "по сути", "по факту", "если честно", "на самом деле",
    ]
    filler_hits = sum(1 for f in filler_phrases if f in lower_text)
    if filler_hits >= 3:
        value_score -= min(6, filler_hits * 2)
        reasons.append(f"value: много фраз-заполнителей ({filler_hits})")
    elif filler_hits >= 1:
        value_score -= filler_hits
    # Very short body has low informational value
    if body_words < 15:
        value_score = 0
        reasons.append("value: текст слишком короткий для контента")
    elif body_words < 40:
        value_score -= 5
        reasons.append("value: слишком короткий для полезного контента")
    dims["value"] = max(0, value_score)

    # --- 4. NATURALNESS: language quality, no AI slop ---
    nat_score = 10
    ai_markers = [
        "в заключение", "подводя итог", "таким образом, мы видим",
        "данный подход", "данная тема", "данный вопрос",
        "комплексный подход", "системный подход", "парадигма",
        "синергия", "оптимизация процессов", "в рамках данного",
        "резюмируя вышесказанное", "необходимо подчеркнуть",
        "ключевым аспектом является", "репрезентативн",
        "на сегодняшний день", "так сказать", "как бы",
        "подведём итог", "подведем итог",
    ]
    ai_hits = sum(1 for marker in ai_markers if marker in lower_text)
    if ai_hits >= 4:
        nat_score = 0
        reasons.append(f"naturalness: типичный AI-текст ({ai_hits} маркеров)")
    elif ai_hits >= 3:
        nat_score -= min(9, ai_hits * 3)
        reasons.append(f"naturalness: сильные признаки AI-текста ({ai_hits} маркеров)")
    elif ai_hits >= 1:
        nat_score -= min(6, ai_hits * 3)
        reasons.append(f"naturalness: AI-маркеры ({ai_hits})")
    # Check for overly formal tone
    formal_markers = [
        "осуществлять", "реализовывать", "имплементировать",
        "функционирование", "вышеуказанн", "нижеследующ",
    ]
    formal_hits = sum(1 for m in formal_markers if m in lower_text)
    if formal_hits >= 2:
        nat_score -= min(4, formal_hits * 2)
        reasons.append("naturalness: канцелярский тон")
    dims["naturalness"] = max(0, nat_score)

    # --- 5. TOPIC_FIT: relevance to channel topic ---
    topic_score = 10
    if channel_topic:
        topic_words = [w for w in re.findall(r"[а-яёa-z]{3,}", channel_topic.lower()) if len(w) >= 4]
        if topic_words:
            topic_hits = sum(1 for tw in topic_words if tw in lower_text)
            hit_ratio = topic_hits / len(topic_words)
            # Semantic cluster expansion (HEURISTIC): if direct token overlap is
            # low, check if text contains related concepts from the topic family
            family_for_topic = detect_topic_family(channel_topic)
            cluster_terms = _TOPIC_SEMANTIC_CLUSTERS.get(family_for_topic, [])
            cluster_hits = sum(1 for ct in cluster_terms if ct in lower_text) if cluster_terms else 0

            # --- Subject-drift detection ---
            # Check if the text talks about a DIFFERENT topic family than the channel
            text_family = detect_topic_family(lower_text[:400])
            is_drifted = (text_family != "generic" and family_for_topic != "generic"
                          and text_family != family_for_topic)

            if topic_hits == 0 and cluster_hits == 0:
                topic_score = 1
                reasons.append("topic_fit: текст не связан с темой канала")
            elif topic_hits == 0 and cluster_hits >= 2 and not is_drifted:
                topic_score = 5
                reasons.append("topic_fit: тема угадывается косвенно, но нет прямой связи")
            elif topic_hits == 0 and cluster_hits >= 2 and is_drifted:
                topic_score = 3
                reasons.append(f"topic_fit: текст ушёл в смежную тему ({text_family})")
            elif hit_ratio < 0.3 and cluster_hits < 2:
                topic_score = 4
                reasons.append("topic_fit: слабая связь с темой канала")
            elif hit_ratio < 0.3 and cluster_hits >= 2:
                topic_score = 6
            elif is_drifted and hit_ratio < 0.5:
                topic_score = 5
                reasons.append(f"topic_fit: частичный дрифт в {text_family}")
    else:
        # No channel topic configured — can't penalize
        topic_score = 7
    dims["topic_fit"] = max(0, topic_score)

    # --- 6. ROLE_FIT: consistency with author role ---
    role_score = 10
    role_type = (author_role_type or "").strip().lower()
    if role_type:
        _personal_markers = ["мой опыт", "я заметил", "у меня", "моя практика", "мои клиенты"]
        _media_markers = ["по данным", "сообщает", "стало известно", "источник"]
        _brand_markers = ["наша команда", "мы разработали", "наш продукт"]
        has_personal = any(m in lower_text for m in _personal_markers)
        has_media = any(m in lower_text for m in _media_markers)
        has_brand = any(m in lower_text for m in _brand_markers)
        if role_type in ("media",) and has_personal:
            role_score -= 4
            reasons.append("role_fit: медиа-канал пишет от первого лица как блогер")
        if role_type in ("blogger", "expert", "master") and has_media and not has_personal:
            role_score -= 3
            reasons.append("role_fit: экспертный/авторский канал пишет обезличенно как новостник")
        if role_type in ("brand", "business_owner") and has_personal and not has_brand:
            role_score -= 3
            reasons.append("role_fit: бренд пишет как личный блогер")
    dims["role_fit"] = max(0, role_score)

    # --- 7. HONESTY: absence of fabrication / clickbait ---
    honesty_score = 10
    clickbait_phrases = [
        "шок", "сенсация", "вы не поверите", "это изменит всё",
        "секрет, который", "только у нас", "эксклюзивно",
        "гарантированный результат", "100% работает", "мгновенный результат",
        "без усилий", "за 5 минут", "идеальное решение",
    ]
    cb_hits = sum(1 for p in clickbait_phrases if p in lower_text)
    if cb_hits >= 2:
        honesty_score -= min(6, cb_hits * 3)
        reasons.append(f"honesty: кликбейтные обещания ({cb_hits})")
    elif cb_hits == 1:
        honesty_score -= 2
        reasons.append("honesty: кликбейтный элемент")
    # Vague unverifiable claims
    vague_claims = [
        "учёные доказали", "исследования показывают", "эксперты утверждают",
        "по статистике", "согласно исследованиям",
    ]
    vague_hits = sum(1 for v in vague_claims if v in lower_text)
    if vague_hits >= 1 and "источник:" not in lower_text:
        honesty_score -= min(4, vague_hits * 2)
        reasons.append("honesty: непроверяемые утверждения без источника")
    # Invented statistics / fabricated research claims (common LLM hallucination)
    _invented_stats = [
        "по данным исследования", "по данным исследований",
        "клинически доказано", "клинически подтверждено",
        "эксперты установили", "как показали исследования",
        "научно доказано", "научно подтверждено",
        "доказано наукой", "учёные установили",
    ]
    invented_stat_hits = sum(1 for p in _invented_stats if p in lower_text)
    if invented_stat_hits >= 1 and "источник:" not in lower_text:
        honesty_score -= min(5, invented_stat_hits * 3)
        reasons.append(f"honesty: вероятно выдуманные исследования/статистика ({invented_stat_hits})")
    # Invented percentages without source (e.g. "73% людей", "по данным 85%")
    _invented_pct = re.findall(r"\b\d{2,3}\s*%\s*(?:людей|клиентов|пациентов|респондентов|участников|пользователей|компаний|предпринимателей)", lower_text)
    if _invented_pct and "источник:" not in lower_text:
        honesty_score -= min(4, len(_invented_pct) * 2)
        reasons.append(f"honesty: выдуманные процентные показатели без источника ({len(_invented_pct)})")
    # Fabricated case studies / client stories
    _fabricated_case = [
        "один из моих клиентов", "одна из моих клиенток",
        "история из практики", "реальный случай из",
        "мой клиент рассказал", "ко мне обратился клиент",
        "был случай когда", "недавно ко мне пришел",
    ]
    case_hits = sum(1 for p in _fabricated_case if p in lower_text)
    if case_hits >= 1:
        honesty_score -= min(4, case_hits * 2)
        reasons.append(f"honesty: вероятно выдуманные кейсы/истории клиентов ({case_hits})")
    # Medical / legal / financial fabrication markers (heuristic: likely hallucinated specifics)
    _risky_claims = [
        "доказано клинически", "одобрено минздравом",
        "одобрено fda", "fda approved", "гарантирует излечение",
        "полностью безопасн", "не имеет побочных", "не имеет противопоказаний",
        "юридически обязан", "по закону вы обязаны", "суд постановил",
        "центральный банк решил", "налоговая обязала",
    ]
    risky_hits = sum(1 for r in _risky_claims if r in lower_text)
    if risky_hits >= 1:
        honesty_score -= min(5, risky_hits * 3)
        reasons.append(f"honesty: потенциально выдуманные медицинские/юридические/финансовые утверждения ({risky_hits})")
    dims["honesty"] = max(0, honesty_score)

    # --- 8. DENSITY: text length and information-per-word ---
    # Target: 60-120 words (aligned with prompt instructions).
    # Posts over ~150 words are considered too long for Telegram scroll.
    density_score = 10
    if body_words < 15:
        density_score = 0
        reasons.append(f"density: текст слишком короткий ({body_words} слов)")
    elif body_words < 40:
        density_score = 2
        reasons.append(f"density: слишком короткий текст ({body_words} слов)")
    elif body_words < 60:
        density_score = 6
        reasons.append(f"density: текст на грани минимума ({body_words} слов)")
    elif body_words > 250:
        density_score -= 5
        reasons.append(f"density: слишком длинный текст ({body_words} слов), цель 60-120")
    elif body_words > 180:
        density_score -= 3
        reasons.append(f"density: текст многословен ({body_words} слов), цель 60-120")
    elif body_words > 130:
        density_score -= 1
        reasons.append(f"density: текст чуть длиннее цели ({body_words} слов)")
    # Penalize very low unique-word ratio (lots of repetition)
    if body_words > 30:
        unique_words = set(re.findall(r"[а-яёa-z]{4,}", body_lower))
        unique_ratio = len(unique_words) / body_words
        if unique_ratio < 0.25:
            density_score -= 3
            reasons.append("density: низкое разнообразие лексики")
    # Tautology penalty
    words = re.findall(r"[а-яёa-z]{4,}", lower_text)
    word_freq: dict[str, int] = {}
    for w in words:
        word_freq[w] = word_freq.get(w, 0) + 1
    repeated_words = [w for w, c in word_freq.items() if c >= 5 and w not in _COMMON_WORDS_EXCLUDE]
    if repeated_words:
        density_score -= min(3, len(repeated_words))
        reasons.append(f"density: повторяющиеся слова: {', '.join(repeated_words[:3])}")
    dims["density"] = max(0, density_score)

    # --- 9. READABILITY: sentence variety and flow ---
    read_score = 10
    sentences = [s.strip() for s in re.split(r"[.!?\n]", body or "") if len(s.strip()) > 10]
    if len(sentences) >= 3:
        lengths = [len(s.split()) for s in sentences]
        avg_len = sum(lengths) / len(lengths)
        if avg_len > 25:
            read_score -= 4
            reasons.append("readability: слишком длинные предложения")
        if all(abs(l - avg_len) < 3 for l in lengths[:6]):
            read_score -= 3
            reasons.append("readability: однообразная длина предложений")
        # Structural repetition: "не просто X, а Y" / "это не X, это Y" patterns
        _struct_patterns = [
            r"не просто .{5,30}, а ",
            r"это не .{3,20}, это ",
            r"не .{3,15}, не .{3,15}, не ",
        ]
        struct_hits = 0
        for pat in _struct_patterns:
            struct_hits += len(re.findall(pat, body_lower))
        if struct_hits >= 2:
            read_score -= 3
            reasons.append(f"readability: повторяющаяся структура предложений ({struct_hits})")
    elif len(sentences) == 1 and body_words > 60:
        read_score -= 3
        reasons.append("readability: весь текст одним абзацем")
    # Paragraph structure
    paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()] if body else []
    if len(paragraphs) <= 1 and body_words > 80:
        read_score -= 2
        reasons.append("readability: нет абзацной структуры")
    dims["readability"] = max(0, read_score)

    # --- 10. PUBLISH_READY: can be published without manual editing ---
    pub_score = 10
    # Check for missing CTA
    if not cta or len(cta.strip()) < 10:
        pub_score -= 2
        reasons.append("publish_ready: нет призыва к действию")
    # Check for markdown artifacts
    if any(mk in full_text for mk in ["```", "**", "##", "- [ ]"]):
        pub_score -= 3
        reasons.append("publish_ready: артефакты markdown в тексте")
    # Check for meta-comments about the post itself
    meta_markers = [
        "в этом посте", "в данном тексте", "в этой статье",
        "сегодня мы поговорим", "давайте разберемся",
        "давайте разберёмся", "сегодня я расскажу",
        "хочу поделиться", "думаю, многие согласятся",
    ]
    meta_hits = sum(1 for m in meta_markers if m in lower_text)
    if meta_hits:
        pub_score -= min(4, meta_hits * 2)
        reasons.append("publish_ready: мета-комментарии про сам пост")
    # Title check
    if not title or len(title.strip()) < 5:
        pub_score -= 3
        reasons.append("publish_ready: нет заголовка")
    dims["publish_ready"] = max(0, pub_score)

    # Final score = sum of all dimensions (0-100)
    total = sum(dims.values())
    total = max(0, min(100, total))

    # Hard floor: extremely low topic_fit always caps total below autopost threshold.
    # A post with topic_fit ≤ 2 is practically off-topic and must not autopublish
    # even if other dimensions score perfectly.
    TOPIC_FIT_HARD_FLOOR = 2
    if dims.get("topic_fit", 10) <= TOPIC_FIT_HARD_FLOOR and channel_topic:
        # Cap total to ensure it falls below AUTOPOST_MIN_QUALITY_SCORE
        total = min(total, 50)
        reasons.append(f"topic_fit HARD FLOOR: topic_fit={dims['topic_fit']} ≤ {TOPIC_FIT_HARD_FLOOR} — автопубликация заблокирована")

    return total, reasons, dims


# Minimum quality score for autopost — posts below this threshold get rejected.
# Total is the SUM of 10 dimensions (0-10 each, max 100). A threshold of 62
# means the post needs at least 62 total points — it does NOT require each
# dimension to reach 6.2; a post may score high on some and low on others.
AUTOPOST_MIN_QUALITY_SCORE = int(os.environ.get("AUTOPOST_MIN_QUALITY_SCORE", "62"))
# Manual generation threshold — lower than autopost since user can edit.
# NOT actively blocking in editor path — kept as a configurable reference
# for future quality-based warnings or soft-gating.
MANUAL_MIN_QUALITY_SCORE = int(os.environ.get("MANUAL_MIN_QUALITY_SCORE", "45"))
# News text quality threshold — slightly lower than autopost because news text
# is constrained to a real source and may have shorter body.  Default 55.
# ACTIVELY used in scheduler_service.py _job_news_tick to gate news posts.
NEWS_MIN_QUALITY_SCORE = int(os.environ.get("NEWS_MIN_QUALITY_SCORE", "55"))


def human_readable_quality_summary(dims: dict[str, int], reasons: list[str]) -> str:
    """Convert quality dimension scores + reasons into a short user-facing summary.

    Returns a single human-readable sentence describing the main quality problem.
    Used for manual generation error messages (not logged as-is in autopost).
    """
    # Find the weakest dimensions (score <= 3 out of 10)
    weak = [(k, v) for k, v in dims.items() if v <= 3]
    weak.sort(key=lambda x: x[1])

    _dim_labels = {
        "hook": "слабое начало текста",
        "specificity": "текст слишком общий, мало конкретики",
        "value": "мало полезной информации",
        "naturalness": "текст звучит неестественно",
        "topic_fit": "текст не по теме канала",
        "role_fit": "не соответствует стилю канала",
        "honesty": "сомнительные утверждения в тексте",
        "density": "текст слишком короткий или многословный",
        "readability": "плохая читаемость текста",
        "publish_ready": "текст не готов к публикации",
    }

    if weak:
        label = _dim_labels.get(weak[0][0], "проблема с качеством текста")
        return label

    # Fallback: use first reason if no weak dimension
    if reasons:
        first = reasons[0]
        # Strip the dimension prefix (e.g. "hook: скучный заход" → "скучный заход")
        if ": " in first:
            first = first.split(": ", 1)[1]
        return first[:100]

    return "не удалось получить достаточно качественный текст"






def _positive_title_guardrails(family: str) -> str:
    base = (
        "заголовок должен быть цепляющим, но взрослым; без дешевой сенсационности, без крика, без манипулятивного страха, "
        "без обещаний мгновенного результата, без пустого пафоса"
    )
    family = (family or "").strip().lower()
    if family in {"massage", "health", "local_business", "beauty", "cars"}:
        extra = "; допускается мягкий кликбейт через пользу, ясность, конкретную ситуацию и понятный сценарий, но без обещаний чудо-результата"
    elif family in {"gaming", "hardware", "tech"}:
        extra = "; допускается кликбейт через актуальность, выгоду выбора, заметное изменение опыта, но без абстрактного пафоса"
    else:
        extra = "; допускается кликбейт через конкретику, выгоду, понятную интригу и пользу; нельзя строить заголовок на дешевой провокации"
    return base + extra

def _family_style_instruction(family: str) -> str:
    """Returns family-appropriate writing style instruction for injection into prompts."""
    instructions = {
        "food": (
            "СТИЛЬ: пиши как опытный гастрономический автор — живой, чувственный, практичный. "
            "Короткие ёмкие предложения. Конкретные детали вместо абстракций. "
            "Нет технического языка, корпоративных оборотов, случайных бизнес-метафор."
        ),
        "health": (
            "СТИЛЬ: пиши как доверенный специалист по здоровью — спокойно, доказательно, по-человечески. "
            "Без сенсационности, без пугающих заголовков, без медицинских обещаний. "
            "Конкретная польза важнее громкого тезиса."
        ),
        "beauty": (
            "СТИЛЬ: пиши как близкий бьюти-эксперт — дружелюбно, вдохновляюще, конкретно. "
            "Практические детали, реальные результаты, без корпоративного пресс-релиза. "
            "Живой личный тон важнее официальной подачи."
        ),
        "local_business": (
            "СТИЛЬ: пиши как честный местный мастер или владелец бизнеса — конкретно, доверительно, по делу. "
            "Простой язык без жаргона. Реальный опыт важнее красивых слов."
        ),
        "education": (
            "СТИЛЬ: пиши как хороший педагог или ментор — ясно, структурированно, поддерживающе. "
            "Объясняй сложное просто. Польза и применимость на первом месте. "
            "Не поучай свысока."
        ),
        "finance": (
            "СТИЛЬ: пиши как честный финансовый аналитик — точно, взвешенно, практично. "
            "Конкретные данные важнее эмоций. Без безответственных обещаний."
        ),
        "marketing": (
            "СТИЛЬ: пиши как опытный практикующий маркетолог — конкретно, с примерами, по делу. "
            "Реальные инструменты и результаты важнее теоретических концепций."
        ),
        "lifestyle": (
            "СТИЛЬ: пиши как живой человек, а не контент-машина — тепло, искренне, по-человечески. "
            "Личный опыт и наблюдения создают связь с читателем."
        ),
        "expert_blog": (
            "СТИЛЬ: пиши как авторитетный эксперт с личным голосом — уверенно, глубоко, оригинально. "
            "Авторская позиция важнее нейтрального пересказа."
        ),
        "massage": (
            "СТИЛЬ: пиши как спокойный профессиональный специалист по восстановлению — "
            "конкретно, по-человечески, без пафоса. Практическая польза и реальный результат "
            "важнее красивых обещаний. Никаких медицинских гарантий."
        ),
        "cars": (
            "СТИЛЬ: пиши как опытный автомобильный эксперт — по делу, с конкретикой, без споров ради споров. "
            "Практические выводы для владельца важнее эмоциональных оценок и рекламного восторга."
        ),
        "gaming": (
            "СТИЛЬ: пиши как взрослый игровой журналист — увлечённо, но без крикливости. "
            "Реальный игровой опыт важнее инфоповодов. Без фанатизма и без хейта."
        ),
        "hardware": (
            "СТИЛЬ: пиши как рациональный технический обозреватель — точно, взвешенно, по существу. "
            "Практика использования важнее сухих цифр и маркетинговых обещаний."
        ),
        "tech": (
            "СТИЛЬ: пиши как практикующий разработчик или IT-специалист — чётко, по делу, без абстрактного пафоса. "
            "Реальные инструменты и применимость важнее хайпа вокруг технологий."
        ),
        "business": (
            "СТИЛЬ: пиши как опытный предприниматель — конкретно, с примерами, без бизнес-жаргона ради жаргона. "
            "Практический результат важнее красивых теорий. Взрослый деловой тон."
        ),
    }
    return instructions.get(family, (
        "СТИЛЬ: пиши как современный Telegram-блогер — конкретно, живо, по-человечески. "
        "Короткие предложения. Факты и польза без воды. Никаких шаблонных заходов."
    ))


def _build_generation_prompt(*, today: str, channel_topic: str, requested: str, bridge_instruction: str, strategy: dict[str, str], angle: dict[str, str], family: str, recent_posts: list[str], recent_plan: list[str], recent_topics: list[str] | None = None, extra_rules: str = "", channel_style: str = "", channel_audience: str = "", post_scenarios: str = "", content_constraints: str = "", content_exclusions: str = "", author_role_type: str = "", author_role_description: str = "", author_activities: str = "", author_forbidden_claims: str = "", strategy_mode: dict[str, str] | None = None) -> str:
    history_block = recent_history_lines({"posts": recent_posts[:4], "plan": recent_plan[:4]}, limit=5)
    recent_openings = "\n".join(f"- {re.split(r'[.!?\n]', x, maxsplit=1)[0][:90]}" for x in recent_posts[:4] if x) or "- пока нет данных"
    blocked_phrases = _recent_phrases(recent_posts)
    blocked_block = "\n".join(f"- {x}" for x in blocked_phrases) or "- пока нет данных"
    plan_topics = "\n".join(f"- {x[:100]}" for x in recent_plan[:6] if x) or "- пока нет данных"
    audience_line = channel_audience or strategy.get('audience') or 'не указана'
    style_line = channel_style or strategy.get('style_text') or strategy.get('style_preset') or 'не указан'
    scenarios_line = post_scenarios or strategy.get('post_scenarios') or 'не указаны'
    constraint_line = strategy.get('constraint_line') or 'без ограничений'
    if content_constraints and content_constraints.strip() not in ('[]', ''):
        constraint_line = content_constraints.strip().strip('[]').replace('"', '').replace(',', ';')
    exclusions_block = ""
    if content_exclusions and content_exclusions.strip():
        exclusions_block = f"\nСТРОГО ЗАПРЕЩЕНО в этом посте (никогда не упоминать, не намекать, не касаться):\n{content_exclusions.strip()}\n"

    # Build anti-repetition topics block from recent_topics (channel memory)
    recent_topics_list = recent_topics or []
    if recent_topics_list:
        recent_topics_block = (
            "\nТемы, которые ты уже недавно писал для этого канала (СТРОГО НЕ ПОВТОРЯЙ — выбери совершенно новый угол или тему):\n"
            + "\n".join(f"- {t[:120]}" for t in recent_topics_list[:15])
            + "\n"
        )
    else:
        recent_topics_block = ""

    # AI-bias guard: forbid AI/neural network content unless the channel is explicitly about AI
    ai_topics = ["искусственный интеллект", "нейросет", "machine learning", "llm", "chatgpt", " ai ", "neural network"]
    channel_topic_lower = channel_topic.lower()
    is_ai_channel = any(kw in channel_topic_lower for kw in ai_topics)
    if is_ai_channel:
        ai_guard = ""
    else:
        ai_guard = (
            "\nКАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО писать про: искусственный интеллект, нейросети, ChatGPT, LLM, machine learning, "
            "«технологии будущего», «эра ИИ» или любые смежные AI-темы — если только это не прямая тема поста. "
            "Канал НЕ об ИИ. Сосредоточься исключительно на теме канала.\n"
        )

    # Build author role identity block for anti-fabrication guardrails
    author_role_block = ""
    if author_role_type or author_role_description or author_activities or author_forbidden_claims:
        _role_labels = {
            "expert": "эксперт",
            "master": "мастер / специалист",
            "business_owner": "владелец бизнеса",
            "brand": "бренд",
            "media": "медиа / редакция",
            "blogger": "блогер",
            "educator": "преподаватель / образовательный канал",
        }
        ar_parts: list[str] = ["\nРОЛЬ АВТОРА КАНАЛА (СТРОГО соблюдать — нельзя фантазировать):"]
        if author_role_type:
            ar_label = _role_labels.get(author_role_type, author_role_type)
            ar_parts.append(f"- Тип автора: {ar_label}")
        if author_role_description:
            ar_parts.append(f"- Кто автор: {author_role_description}")
        if author_activities:
            ar_parts.append(f"- Что автор реально делает: {author_activities}")
        if author_forbidden_claims:
            ar_parts.append(f"- НЕЛЬЗЯ приписывать автору: {author_forbidden_claims}")
        ar_parts.append(
            "- Категорически запрещено выдумывать услуги, компетенции, достижения, "
            "образование или опыт, которые не указаны выше."
        )
        # Role-specific voice constraints: prevent cross-persona writing
        # Detect effective role from description when type is generic default
        _role_type_l = (author_role_type or "").strip().lower()
        _role_desc_l = (author_role_description or "").strip().lower()
        # Auto-detect role type from custom description for better voice matching
        _media_keywords = ("новостн", "медиа", "редакци", "агентств", "сми", "журнал")
        _brand_keywords = ("магазин", "бренд", "компани", "команд", "маркетплейс", "сервис", "агентств")
        _blogger_keywords = ("блогер", "блог", "личн")
        if _role_type_l == "expert" and _role_desc_l:
            # Refine voice detection for custom roles defaulting to 'expert'
            if any(kw in _role_desc_l for kw in _media_keywords):
                _role_type_l = "media"
            elif any(kw in _role_desc_l for kw in _brand_keywords):
                _role_type_l = "brand"
            elif any(kw in _role_desc_l for kw in _blogger_keywords):
                _role_type_l = "blogger"
        if _role_type_l in ("media",):
            ar_parts.append(
                "- ГОЛОС: Пиши как редакция / новостная лента. НЕ используй «я», «мой опыт», "
                "«мои клиенты» — это канал медиа-формата, а не личный блог."
            )
        elif _role_type_l in ("brand", "business_owner"):
            ar_parts.append(
                "- ГОЛОС: Пиши от лица команды / бренда. Допускается «мы», «наш продукт», «наша команда». "
                "НЕ пиши как личный блогер с «я заметил» или «мой опыт», если нет явного указания."
            )
        elif _role_type_l in ("expert", "master"):
            ar_parts.append(
                "- ГОЛОС: Пиши от первого лица как практикующий специалист. Допускается «я», «мой опыт», "
                "«мои клиенты» — но ТОЛЬКО если роль автора подразумевает работу с клиентами. "
                "Если роль автора — «новостной канал» или «обзорщик», НЕ используй «мои клиенты»."
            )
        elif _role_type_l in ("blogger",):
            ar_parts.append(
                "- ГОЛОС: Пиши живо, от первого лица, с личными наблюдениями. "
                "НЕ пиши сухо как корпоративная рассылка или новостной дайджест."
            )
        elif _role_type_l in ("educator",):
            ar_parts.append(
                "- ГОЛОС: Пиши как преподаватель/наставник — ясно, структурированно, с примерами. "
                "Допускается «я», но без маркетинговых штампов и без «мои клиенты»."
            )
        author_role_block = "\n".join(ar_parts) + "\n"

    # --- Strategy mode block (generation archetype) ---
    strategy_mode_block = ""
    if strategy_mode:
        strategy_mode_block = (
            f"\nСТРАТЕГИЯ ГЕНЕРАЦИИ — {strategy_mode['label']}:\n"
            f"{strategy_mode['prompt_hint']}\n"
        )

    # --- Enhanced role-specific prompt block (detailed per-role constraints) ---
    role_block_enhanced = _get_role_prompt_block(author_role_type)
    if role_block_enhanced:
        role_block_enhanced = f"\n{role_block_enhanced}\n"

    return f"""
Напиши один короткий сильный пост для Telegram. Он должен звучать как настоящий живой эксперт-блогер, делящийся ценным инсайтом, — не шаблон нейросети и не пресс-релиз.

Дата: {today}
Тема канала: {channel_topic}
Тема поста: {requested}
{bridge_instruction}
{exclusions_block}{recent_topics_block}{ai_guard}
СТРОГО: весь текст поста обязан быть НЕПОСРЕДСТВЕННО о теме канала «{channel_topic}». Запрещено уходить в абстрактные рассуждения о «рынке в целом», «трендах», «бизнесе вообще», «технологиях будущего» или любые смежные темы, если они не являются прямым содержанием канала. Если тема канала узкая — пиши узко. Лучше глубоко по теме, чем широко мимо неё.

Профиль канала — строго соблюдай:
- Для кого пишешь (аудитория): {audience_line}
- Стиль и голос: {style_line}
- Сценарии и форматы постов: {scenarios_line}
- Жёсткие ограничения канала: {constraint_line}
- Голос этой генерации: {_voice_variant()}
{author_role_block}{role_block_enhanced}{strategy_mode_block}
Угол именно этого поста:
- anchor: {angle['opening']}
- focus: {angle['focus']}
- freshness: {_freshness_line(family, requested)}
- guardrails: {_family_guardrails(family)}
- title rules: {_positive_title_guardrails(family)}

Нельзя повторять темы, углы и структуру этих недавних идей из плана:
{plan_topics}

Нельзя повторять недавние материалы:
{history_block}

Нельзя начинать примерно так же:
{recent_openings}

Нельзя повторять эти формулировки и заходы:
{blocked_block}

Верни только JSON с полями: title, body, cta, short, button_text, image_prompt.

ПЕРВЫЙ АБЗАЦ — САМЫЙ ВАЖНЫЙ. Правила для первого абзаца:
- Первое предложение ОБЯЗАНО содержать конкретный факт, наблюдение или ситуацию. Не абстрактное рассуждение.
- Примеры хороших первых предложений: «70% клиентов с болью в шее приходят после 3+ часов за ноутбуком», «Вчера клиент принёс чек на 40 тысяч за ремонт, который не был нужен», «Один неудачный заголовок стоил каналу 500 отписок за неделю».
- Примеры ПЛОХИХ первых предложений (ЗАПРЕЩЕНЫ): «В наше время всё больше людей задумываются...», «Давайте разберёмся, почему...», «Важно понимать, что...», «Не секрет, что...», «Многие из нас сталкивались с...», «Последнее время всё чаще...»
- Первый абзац должен дать читателю повод остановиться и дочитать. Если первые 2 строки можно удалить без потери смысла — они лишние.

Требования:
- title: 4-9 слов, конкретный и цепляющий. Заголовок обязан содержать конкретное утверждение, цифру или неожиданный угол зрения — не просто перефразировать тему. Запрещены: чистое переформулирование темы («О пользе грибов»), обобщённый совет без конкретики («Как улучшить свой канал»), пустой кликбейт без содержания («ШОК! Это изменит всё»). Заголовок должен говорить напрямую о ситуации или проблеме читателя и вызывать желание читать первое предложение.
- Не используй в заголовке конструкции вроде «вся правда», «без иллюзий», «за 10 минут», «как не выбрасывать деньги», «как выбрать без сожалений», «что реально тянет», «как оживить».
- body: 2-3 коротких абзаца, целевой объём 60-120 слов. Это Telegram — читают на ходу. Одна мысль на пост, без вступлений, без «раскачки». Каждое предложение обязано добавлять новую информацию. Если фразу можно удалить без потери смысла — удали. Никаких повторов одной и той же идеи разными словами. Максимум 2 абзаца основного текста — если получается больше, режь безжалостно.
- ПРОСТОТА И ПОВЕРХНОСТЬ: Пиши на поверхности темы. Не уходи в чрезмерную экспертность, сложные термины и глубокую аналитику без явного запроса пользователя. Сначала ясно и прямо раскрой тему, потом максимум 1–2 полезных уточнения. Не превращай обычный пост в лекцию или научную статью. Telegram-стиль: просто, прямо, понятно обычному человеку. Если тема простая — текст должен быть простым.
- АБСОЛЮТНЫЙ ЗАПРЕТ — нельзя начинать текст ни с одной из этих конструкций: «В современном мире», «Сегодня все знают», «Многие задаются вопросом», «Ни для кого не секрет», «В наше время», «Последнее время всё чаще», «Все мы знаем», «В этой статье», «В этом посте», «Давайте разберёмся», «Важно понимать», «Подводя итог», «Стоит отметить», «Каждый из нас», «Не будем лукавить». Если первое предложение начинается с любой из этих фраз — это провал, перепиши.
- Начало должно цеплять и заметно отличаться от недавних постов.
- В тексте должен быть конкретный вывод, польза или понятный продающий смысл для читателя.
- {_family_style_instruction(family)}
- АБСОЛЮТНЫЙ ЗАПРЕТ на повторяющиеся структуры предложений: «не просто X, а Y», «не X, не Y, не Z», «это не про X». Одна мысль — одно предложение.
- АБСОЛЮТНЫЙ ЗАПРЕТ на фразы-клише и канцелярит — ПОЛНЫЙ СПИСОК (использование ЛЮБОЙ из них = брак текста): «В современном быстро меняющемся мире», «Привет, друзья», «Сегодня мы поговорим о», «в этой статье», «в этом посте», «давайте разберёмся», «не секрет что», «это не просто», «хочу поделиться», «думаю, многие согласятся», «важно понимать», «стоит отметить», «нужно понимать», «в заключение», «подведём итог», «подводя итог», «в наше время», «это не просто... а», «это не про...», «данный подход», «комплексный подход», «системный подход», «оптимизация процессов», «синергия», «парадигма», «в рамках данного», «ключевым аспектом является», «необходимо подчеркнуть», «резюмируя», «как бы», «так сказать», «на сегодняшний день».
- ОБЯЗАТЕЛЬНАЯ КОНЦОВКА С ВОПРОСОМ: завершай пост одним естественным вопросом, прямо вытекающим из конкретного содержания этого поста. Вопрос должен касаться именно темы и наблюдений поста — не общего мнения читателя о жизни. Запрещены дежурные призывы: «Что думаете?», «Пишите в комментарии», «А вы как?», «Расскажите о своём опыте». Вопрос должен быть конкретным — чтобы читатель сразу понял, о чём именно его спрашивают. НЕ ЗАКАНЧИВАЙ пост фразами «именно поэтому это важно», «вот почему это важно», «в конечном счёте», «задумайтесь об этом», «это не случайно», «вот в чём секрет».
- АБСОЛЮТНЫЙ ЗАПРЕТ на пренебрежительные и разрушительные для репутации фразы: запрещены любые пренебрежительные императивы, направленные на тему канала или услугу («забейте на», «плюньте на», «наплюйте на», «забудьте про», «бросьте это», «хватит ныть»). Нельзя высмеивать нишу, читателя или ожидания клиента. Текст должен уважать тему и усиливать доверие к ней, а не подрывать его.
- ЗАПРЕТ НА ВЫДУМАННЫЕ ФАКТЫ: если ты не уверен в факте, статистике, цифре или утверждении — НЕ пиши его. Лучше написать «по опыту» или «часто бывает» чем придумать «исследования показали» или «70% людей». Если тема тебе незнакома — пиши на уровне здравого смысла, без деталей, в которых ты не уверен.
- Не используй markdown, code fences, JSON внутри строк, списки с 1. 2. 3., служебные подписи.
- Не пиши мета-комментарии про автора, канал, контент-план или нейросеть.
- Не делай текст стерильным: нужна живая и взрослая подача, а не пресс-релиз и не дешёвый продающий текст.
- Не упоминай год без прямой необходимости. Если тема не требует даты, не пиши год.
- Не выдумывай спорных фактов, медицинских обещаний, юридически опасных утверждений и резких выводов без запроса.
- Для услуг и экспертных ниш усиливай доверие к автору и нише. Любое упоминание автора, мастера, сервиса или специалиста допускается только в нейтральном или положительном ключе.
- Не делай текст антирекламным для ниши, не спорь с читателем и не унижай его ожидания.
- Лучше конкретная польза, ясный сценарий и точное наблюдение, чем громкий заголовок и пустой пафос.
- Пиши так, как написал бы живой автор, хорошо знающий свою аудиторию: {audience_line}.
- СОХРАНЕНИЕ СУЩНОСТИ: Основной предмет или сущность темы поста «{requested}» должна оставаться неизменной на протяжении всего текста. Допускается менять угол, форму подачи или контекст, но НЕЛЬЗЯ заменять основной предмет на смежное или абстрактное понятие. Например, если тема «грибы» — текст должен быть именно про грибы, а не про «грибной мониторинг» или «рынок грибов». Если тема «склад» — текст про склад, а не про «складскую революцию» в целом.
- СТРОГО ЗАПРЕЩЕНО: выдумывать @упоминания пользователей, названия Telegram-каналов, ссылки, URL или источники, которые не были явно указаны в запросе. Не ссылайся ни на какие каналы, аккаунты, сайты или публикации, если они не предоставлены в данных запроса.
- image_prompt: A concise English prompt for a photorealistic AI-generated image that LITERALLY and DIRECTLY depicts the main subject of the post. If the post is about cars — show a real car. If about a server — show a real server room. If about an accountant — show a real person working with documents in an office. NEVER replace the literal subject with a metaphor, abstraction, or artistic interpretation. The image must show exactly what the post is about. Style: photorealistic, editorial quality, natural light, clean composition. No text, no logos, portrait-safe. Example for a post about "coffee shop": "Professional editorial photo of a cozy coffee shop interior, barista preparing latte, warm natural light, wooden counter, realistic, high resolution".
{extra_rules}
""".strip()


def _normalize_generated_data(data: dict[str, Any]) -> dict[str, str]:
    title = _cleanup_model_field(data.get("title"))
    body = _cleanup_model_field(data.get("body"))
    cta = _cleanup_model_field(data.get("cta"))
    short = _cleanup_model_field(data.get("short"))
    button_text = _cleanup_model_field(data.get("button_text"))
    polluted = "\n".join([title, body, cta, short, button_text]).lower()
    if any(marker in polluted for marker in ("```", '"title"', '"body"', "{", "}")):
        raise RuntimeError("Модель вернула ответ в неверном формате")
    if not title or not body:
        raise RuntimeError("Модель не заполнила title/body")
    normalized = _normalize_lengths(
        _strip_ai_cliches(_strip_dismissive_phrases(title)),
        _strip_ai_cliches(_strip_dismissive_phrases(body)),
        _strip_ai_cliches(_strip_dismissive_phrases(cta)),
        _strip_ai_cliches(_strip_dismissive_phrases(short)),
        _strip_ai_cliches(button_text),
    )
    polluted = "\n".join([normalized.get("title", ""), normalized.get("body", ""), normalized.get("cta", ""), normalized.get("short", "")]).lower()
    if any(marker in polluted for marker in ("```", '"title"', '"body"', '{', '}')):
        raise RuntimeError("Модель вернула ответ в неверном формате")
    # Extract image_prompt separately — it's English and bypasses Russian cliché filters
    raw_image_prompt = _cleanup_model_field(data.get("image_prompt"))
    if raw_image_prompt:
        normalized["image_prompt"] = raw_image_prompt[:600]
    return normalized


# ---------------------------------------------------------------------------
# Post-generation targeted rewrite pass
# ---------------------------------------------------------------------------
# When a post is CLOSE to passing quality but weak on 1-2 specific dimensions,
# a targeted rewrite prompt asks the LLM to fix only those weaknesses without
# regenerating the entire post. This is cheaper than a full retry.
#
# HEURISTIC: only triggers when total score is within REWRITE_THRESHOLD_MARGIN
# of the gate threshold AND the weak dimensions are rewritable.

REWRITE_THRESHOLD_MARGIN = int(os.environ.get("REWRITE_THRESHOLD_MARGIN", "12"))
# Dimensions that can be improved via targeted rewrite without full regeneration.
# Excluded: honesty (needs new facts), density (needs more/less content),
# value (needs substantive new info), readability (structural).
# topic_fit is now rewritable — the rewrite can refocus the text back on topic.
_REWRITABLE_DIMS = {"hook", "naturalness", "role_fit", "specificity", "publish_ready", "topic_fit"}


def _build_rewrite_prompt(
    title: str,
    body: str,
    cta: str,
    weak_dims: dict[str, int],
    channel_topic: str = "",
    author_role_type: str = "",
) -> str | None:
    """Build a targeted rewrite prompt for fixable quality weaknesses.

    Returns None if no rewrite is possible (weak dims are not in _REWRITABLE_DIMS).
    """
    fixable = {k: v for k, v in weak_dims.items() if k in _REWRITABLE_DIMS and v <= 4}
    if not fixable:
        return None

    instructions: list[str] = []
    if "hook" in fixable:
        instructions.append(
            "ПЕРЕДЕЛАЙ ПЕРВЫЙ АБЗАЦ: сделай начало цепляющим. Первое предложение — "
            "конкретный факт, наблюдение или ситуация. Никаких 'В наше время...', "
            "'Давайте разберёмся...' или абстрактных рассуждений."
        )
    if "naturalness" in fixable:
        instructions.append(
            "УБЕРИ AI-ЗВУЧАНИЕ: замени канцелярит, 'данный подход', 'системный подход', "
            "'подводя итог' на живые формулировки. Текст должен звучать как человек, а не робот."
        )
    if "role_fit" in fixable:
        role = (author_role_type or "").lower()
        if role in ("media",):
            instructions.append("ГОЛОС: убери личный тон ('я', 'мой опыт'). Пиши как редакция.")
        elif role in ("brand", "business_owner"):
            instructions.append("ГОЛОС: убери блогерский тон. Пиши от лица команды/бренда ('мы', 'наш').")
        else:
            instructions.append("ГОЛОС: приведи тон в соответствие с ролью автора канала.")
    if "specificity" in fixable:
        instructions.append(
            "ДОБАВЬ КОНКРЕТИКУ: замени водные фразы ('в наше время', 'всем известно') "
            "на конкретные факты, примеры или ситуации."
        )
    if "topic_fit" in fixable:
        instructions.append(
            f"ВЕРНИ ТЕКСТ К ТЕМЕ КАНАЛА: тема канала — «{channel_topic}». "
            "Перефокусируй текст так, чтобы он был непосредственно о теме канала, "
            "а не о смежных абстрактных вещах. Замени общие рассуждения на конкретику по теме."
        )
    if "publish_ready" in fixable:
        instructions.append(
            "ПОДГОТОВЬ К ПУБЛИКАЦИИ: убери мета-комментарии ('в этом посте'), "
            "markdown-артефакты и служебные пометки."
        )

    if not instructions:
        return None

    return f"""Перепиши этот пост, исправив ТОЛЬКО указанные проблемы. Не меняй тему, структуру и общий смысл.

Тема канала: {channel_topic}

ТЕКУЩИЙ ТЕКСТ:
Заголовок: {title}
Текст: {body}
CTA: {cta}

ЧТО НУЖНО ИСПРАВИТЬ:
{chr(10).join('- ' + i for i in instructions)}

ВАЖНО:
- Не переписывай весь текст заново. Исправь ТОЛЬКО указанные проблемы.
- Сохрани тему, основные мысли и структуру.
- Не добавляй новых фактов, которых не было.
- Не делай текст длиннее.

Верни только JSON с полями: title, body, cta, short, button_text, image_prompt."""


async def _try_rewrite_pass(
    api_key: str,
    model: str,
    normalized: dict[str, str],
    q_score: int,
    q_dims: dict[str, int],
    threshold: int,
    *,
    channel_topic: str = "",
    author_role_type: str = "",
    base_url: str | None = None,
) -> tuple[dict[str, str], int, list[str], dict[str, int]] | None:
    """Attempt a targeted rewrite of a near-miss post.

    Returns (rewritten_data, new_score, new_reasons, new_dims) or None if
    rewrite is not applicable or didn't improve quality.
    """
    # Only rewrite if score is in the "near-miss" window: [threshold - margin, threshold)
    already_passing = q_score >= threshold
    too_far_below = q_score < threshold - REWRITE_THRESHOLD_MARGIN
    if already_passing or too_far_below:
        return None

    weak = {k: v for k, v in q_dims.items() if v <= 4}
    prompt = _build_rewrite_prompt(
        normalized.get("title", ""),
        normalized.get("body", ""),
        normalized.get("cta", ""),
        weak,
        channel_topic=channel_topic,
        author_role_type=author_role_type,
    )
    if not prompt:
        return None

    logger.info(
        "REWRITE_PASS: attempting targeted rewrite score=%d/%d weak_dims=%s",
        q_score, threshold, " ".join(f"{k}={v}" for k, v in weak.items()),
    )

    try:
        raw = await ai_chat(
            api_key=api_key,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,  # Lower temp for focused rewrite
            base_url=base_url,
            max_tokens=400,
        )
        data = _extract_json_object(raw)
        if not data:
            data = _extract_partial_json_object(raw)
        if not data:
            return None
        rewritten = _normalize_generated_data(data)
    except Exception as exc:
        logger.warning("REWRITE_PASS: failed: %s", exc)
        return None

    # Assess rewritten quality
    new_score, new_reasons, new_dims = assess_text_quality(
        rewritten.get("title", ""),
        rewritten.get("body", ""),
        rewritten.get("cta", ""),
        channel_topic=channel_topic,
        author_role_type=author_role_type,
    )

    if new_score > q_score:
        logger.info(
            "REWRITE_PASS: improved score %d→%d dims=%s",
            q_score, new_score, " ".join(f"{k}={v}" for k, v in new_dims.items()),
        )
        return rewritten, new_score, new_reasons, new_dims
    else:
        logger.info("REWRITE_PASS: no improvement %d→%d, keeping original", q_score, new_score)
        return None




async def generate_post_bundle(
    api_key: str,
    model: str,
    *,
    topic: str,
    prompt: str = "",
    owner_id: int | None = None,
    channel_style: str = "",
    content_rubrics: str = "",
    post_scenarios: str = "",
    channel_audience: str = "",
    content_constraints: str = "",
    recent_posts: list[str] | None = None,
    recent_plan: list[str] | None = None,
    base_url: str | None = None,
    generation_path: str = "editor",
) -> dict[str, str]:
    no_disclaimer = (await get_setting("no_disclaimer", owner_id=owner_id) or "1") == "1"
    channel_topic = (topic or "").strip() or "без общей темы"
    requested = (prompt or "").strip() or channel_topic
    owner_settings = await _load_owner_strategy_settings(
        owner_id,
        channel_topic,
        channel_style,
        content_rubrics,
        post_scenarios,
        channel_audience,
        content_constraints,
    )
    strategy = build_generation_strategy(owner_settings)
    bridge_instruction = _blend_instruction(channel_topic, requested)
    recent_posts = recent_posts or []
    recent_plan = recent_plan or []
    today = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d")
    family = _topic_family(channel_topic, requested)

    logger.info(
        "GENERATE_POST_BUNDLE_ENTRY path=%s owner_id=%s literal_topic=%r channel_family=%s",
        generation_path, owner_id, requested[:80], family,
    )

    # Fetch recent post topics for anti-repetition channel memory
    recent_topics: list[str] = []
    try:
        recent_topics = await get_recent_post_topics(owner_id=owner_id, limit=15)
    except (OSError, ValueError, TypeError) as exc:
        logger.warning("Failed to fetch recent post topics owner=%s: %s", owner_id, exc)
        recent_topics = []

    async def _run_once(user_prompt: str, temperature: float = 0.82) -> dict[str, str]:
        raw = await ai_chat(
            api_key=api_key,
            model=model,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=temperature,
            base_url=base_url,
            max_tokens=400,
        )
        logger.info("GENERATE_POST_LLM_RAW requested=%r preview=%r", requested[:140], str(raw or "")[:400])
        data = _extract_json_object(raw)
        if not data:
            data = _extract_partial_json_object(raw)
        if not data:
            data = _extract_plain_text_post(raw)
        if not data:
            _raise_generation_error(raw)
        return _normalize_generated_data(data)

    angle_pool = _family_angles(family, requested)[:4]
    candidates: list[tuple[dict[str, str], list[str], int]] = []

    # Select a strategy mode for this generation session (adaptive per-channel)
    _strategy_mode = _pick_strategy_mode(family, recent_posts, recent_plan)
    logger.info("generate_post_bundle: strategy_mode=%s", _strategy_mode["mode"])

    for idx, angle in enumerate(angle_pool, start=1):
        extra = ""
        if idx > 1:
            extra = (
                "Сделай новый вариант с другим редакторским углом и другой логикой заголовка. "
                "Не повторяй слова и формулы из типовых заголовков. "
                "Избегай конструкции «как X без Y», если она звучит дёшево или искусственно."
            )
        prompt_text = _build_generation_prompt(
            today=today,
            channel_topic=channel_topic,
            requested=requested,
            bridge_instruction=bridge_instruction,
            strategy=strategy,
            angle=angle,
            family=family,
            recent_posts=recent_posts,
            recent_plan=recent_plan,
            recent_topics=recent_topics,
            extra_rules=extra,
            channel_style=str(owner_settings.get("channel_style") or ""),
            channel_audience=str(owner_settings.get("channel_audience") or ""),
            post_scenarios=str(owner_settings.get("post_scenarios") or ""),
            content_constraints=str(owner_settings.get("content_constraints") or ""),
            content_exclusions=str(owner_settings.get("content_exclusions") or ""),
            strategy_mode=_strategy_mode,
            **_author_role_kwargs(owner_settings),
        )
        if no_disclaimer:
            prompt_text += "\nНе добавляй дисклеймеры, извинения и служебные пометки."

        normalized = await _run_once(prompt_text, temperature=min(0.9, 0.8 + idx * 0.03))

        recent_roots = _recent_title_roots(recent_posts)
        if _title_root(normalized["title"]) in recent_roots:
            normalized["title"] = clean_text(normalized["title"] + " — по делу")[:86].strip(" \"'«»")

        current_opening = _opening_key(normalized["body"])
        if current_opening in _recent_opening_keys(recent_posts):
            parts = [seg.strip() for seg in re.split(r"\n\n+", normalized["body"]) if seg.strip()]
            tail = "\n\n".join(parts[1:]) if len(parts) > 1 else normalized["body"]
            normalized["body"] = f"{angle['opening']}\n\n{tail}".strip()

        if not normalized["body"] or _looks_like_ai_error(normalized["body"]):
            continue

        issues = _quality_issues(channel_topic, requested, normalized, recent_posts)
        score = len(issues)
        if normalized.get("title") and len(normalized["title"].split()) <= 8:
            score -= 1
        candidates.append((normalized, issues, score))

        if not issues:
            # --- Quality assessment gate (autopost + editor) ---
            _ar_kwargs = _author_role_kwargs(owner_settings)
            q_score, q_reasons, q_dims = assess_text_quality(
                normalized.get("title", ""),
                normalized.get("body", ""),
                normalized.get("cta", ""),
                channel_topic=channel_topic,
                requested=requested,
                author_role_type=_ar_kwargs.get("author_role_type", ""),
            )
            normalized["quality_score"] = str(q_score)
            normalized["quality_reasons"] = "; ".join(q_reasons) if q_reasons else ""
            normalized["quality_dims"] = "; ".join(f"{k}={v}" for k, v in q_dims.items()) if q_dims else ""

            if generation_path == "autopost":
                if q_score < AUTOPOST_MIN_QUALITY_SCORE:
                    # --- Attempt targeted rewrite pass before full rejection ---
                    rewrite_result = await _try_rewrite_pass(
                        api_key, model, normalized, q_score, q_dims,
                        AUTOPOST_MIN_QUALITY_SCORE,
                        channel_topic=channel_topic,
                        author_role_type=_ar_kwargs.get("author_role_type", ""),
                        base_url=base_url,
                    )
                    if rewrite_result:
                        normalized, q_score, q_reasons, q_dims = rewrite_result
                        normalized["quality_score"] = str(q_score)
                        normalized["quality_reasons"] = "; ".join(q_reasons) if q_reasons else ""
                        normalized["quality_dims"] = "; ".join(f"{k}={v}" for k, v in q_dims.items()) if q_dims else ""
                    if q_score < AUTOPOST_MIN_QUALITY_SCORE:
                        logger.warning(
                            "generate_post_bundle: autopost quality gate REJECT score=%d min=%d dims=%s reasons=%s title=%r",
                            q_score, AUTOPOST_MIN_QUALITY_SCORE,
                            " ".join(f"{k}={v}" for k, v in q_dims.items()),
                            "; ".join(q_reasons[:4]),
                            normalized.get("title", "")[:80],
                        )
                        issues.append(f"низкий балл качества: {q_score}/{AUTOPOST_MIN_QUALITY_SCORE}")
                        candidates.append((normalized, issues, len(issues)))
                        continue
                logger.info(
                    "generate_post_bundle: autopost quality gate PASS score=%d dims=%s",
                    q_score, " ".join(f"{k}={v}" for k, v in q_dims.items()),
                )
            elif generation_path == "editor":
                # For editor path — log quality info but don't block
                logger.info(
                    "generate_post_bundle: editor quality score=%d dims=%s",
                    q_score, " ".join(f"{k}={v}" for k, v in q_dims.items()),
                )

            normalized["post_intent"] = _derive_post_intent(channel_topic, requested, normalized.get("body", ""))
            normalized["visual_brief"] = _derive_visual_brief(
                channel_topic,
                requested,
                normalized.get("title", ""),
                normalized.get("body", ""),
                normalized["post_intent"],
            )
            _apply_fabrication_cleanup(normalized)
            _apply_safety_pass(normalized, **_author_role_kwargs(owner_settings))
            # Enforce single-message budget for autopost path
            if generation_path == "autopost":
                t, b, c = enforce_autopost_budget(
                    normalized.get("title", ""),
                    normalized.get("body", ""),
                    normalized.get("cta", ""),
                    has_media=True,
                )
                normalized["title"] = t
                normalized["body"] = b
                normalized["cta"] = c
            return normalized

    if not candidates:
        raise RuntimeError("Не удалось получить пригодный текст от модели")

    candidates.sort(key=lambda item: (item[2], -len(item[0].get("body", ""))))
    best, final_issues, _ = candidates[0]
    if final_issues:
        # For autopost path, do NOT publish content with quality issues — it
        # would result in garbage posts going live without human review.
        if generation_path == "autopost":
            # Try to find a human-readable reason from quality dims
            _best_q_dims = {}
            _best_q_reasons: list[str] = []
            try:
                _best_q_dims = {k: int(v) for k, v in (
                    pair.split("=") for pair in (best.get("quality_dims") or "").split("; ") if "=" in pair
                )}
                _best_q_reasons = [r.strip() for r in (best.get("quality_reasons") or "").split(";") if r.strip()]
            except (ValueError, AttributeError):
                pass
            _hr_reason = human_readable_quality_summary(_best_q_dims, _best_q_reasons) if _best_q_dims else ""
            logger.warning(
                "generate_post_bundle: rejecting autopost candidate due to quality issues: %s (human: %s)",
                "; ".join(final_issues[:4]), _hr_reason,
            )
            raise RuntimeError(f"Не удалось сгенерировать достаточно качественный текст для автопостинга: {_hr_reason}" if _hr_reason else "Не удалось сгенерировать достаточно качественный текст для автопостинга")
        logger.warning("generate_post_bundle: returning best candidate despite quality issues: %s", "; ".join(final_issues[:4]))

    best["post_intent"] = _derive_post_intent(channel_topic, requested, best.get("body", ""))
    best["visual_brief"] = _derive_visual_brief(
        channel_topic,
        requested,
        best.get("title", ""),
        best.get("body", ""),
        best["post_intent"],
    )
    _apply_fabrication_cleanup(best)
    _apply_safety_pass(best, **_author_role_kwargs(owner_settings))

    # Enforce single-message budget for autopost path
    if generation_path == "autopost":
        t, b, c = enforce_autopost_budget(
            best.get("title", ""),
            best.get("body", ""),
            best.get("cta", ""),
            has_media=True,  # conservative: assume media caption limit
        )
        best["title"] = t
        best["body"] = b
        best["cta"] = c

    return best


async def generate_post_text(
    api_key: str,
    model: str,
    *,
    topic: str,
    prompt: str = "",
    base_url: str | None = None,
    owner_id: int | None = None,
    recent_posts: list[str] | None = None,
    recent_plan: list[str] | None = None,
    generation_path: str = "editor",
) -> str:
    # Use channel-scoped settings for consistent per-channel generation
    ch_settings = await get_channel_settings(owner_id)
    style = (ch_settings.get("channel_style") or "").strip()
    rubrics = (ch_settings.get("content_rubrics") or "").strip()
    scenarios = (ch_settings.get("post_scenarios") or "").strip()
    audience = (ch_settings.get("channel_audience") or "").strip()
    constraints = (ch_settings.get("content_constraints") or "").strip()
    bundle = await generate_post_bundle(
        api_key,
        model,
        topic=topic,
        prompt=prompt,
        channel_style=style,
        content_rubrics=rubrics,
        post_scenarios=scenarios,
        channel_audience=audience,
        content_constraints=constraints,
        recent_posts=recent_posts or [],
        recent_plan=recent_plan or [],
        base_url=base_url,
        owner_id=owner_id,
        generation_path=generation_path,
    )
    # _apply_fabrication_cleanup already runs inside generate_post_bundle, but the
    # final joined text may still contain references if they span field boundaries.
    text = "\n\n".join(part for part in [bundle.get("title", ""), bundle.get("body", ""), bundle.get("cta", "")] if part)
    text = _strip_ai_cliches(text).strip()
    # Final safety net: remove any fabricated @mentions/URLs from the combined text
    text, _, _ = _remove_fabricated_refs(text)
    # Final safety / consistency pass on combined text
    # Author role from channel-scoped settings (already loaded above)
    ar_settings: dict[str, str] = {}
    for _ar_key in ("author_role_type", "author_role_description", "author_activities", "author_forbidden_claims"):
        ar_settings[_ar_key] = (ch_settings.get(_ar_key) or "").strip()
    text = _safety_consistency_pass(text, **_author_role_kwargs(ar_settings))
    return text
