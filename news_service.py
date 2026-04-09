from __future__ import annotations

import html
import json
import logging
import re
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse

import httpx
import trafilatura

import db
from content import generate_post_text

try:
    from topic_utils import TOPIC_FAMILY_TERMS, detect_topic_family, get_family_news_angle
    from channel_profile_resolver import resolve_channel_policy, build_news_family_block
except ModuleNotFoundError:
    # Minimal fallback covering all 15 major families — used only if topic_utils is missing.
    TOPIC_FAMILY_TERMS = {
        "food": {
            "ru": ["еда", "блюд", "рецепт", "кухн", "ресторан", "кафе", "кофе", "выпечк", "фуд", "питани"],
            "en": ["food", "recipe", "cooking", "cuisine", "restaurant", "cafe", "coffee", "bakery"],
        },
        "health": {
            "ru": ["здоровь", "медицин", "врач", "лечен", "фитнес", "wellness", "питани", "зож"],
            "en": ["health", "medicine", "doctor", "fitness", "wellness", "nutrition", "diet"],
        },
        "beauty": {
            "ru": ["красот", "косметик", "косметолог", "уход", "макияж", "маникюр", "бьюти", "салон"],
            "en": ["beauty", "cosmetics", "skincare", "makeup", "manicure", "spa"],
        },
        "local_business": {
            "ru": ["ремонт", "сервис", "мастерск", "малый бизнес", "услуг"],
            "en": ["local business", "service", "workshop", "repair", "small business"],
        },
        "education": {
            "ru": ["образован", "обучен", "курс", "школ", "репетитор", "учебн"],
            "en": ["education", "learning", "course", "school", "tutor", "training"],
        },
        "finance": {
            "ru": ["финанс", "инвестиц", "деньг", "крипт", "трейдинг", "бирж"],
            "en": ["finance", "investment", "money", "crypto", "trading", "stock"],
        },
        "marketing": {
            "ru": ["маркетинг", "smm", "реклам", "продвиж", "контент"],
            "en": ["marketing", "smm", "advertising", "promotion", "content"],
        },
        "lifestyle": {
            "ru": ["лайфстайл", "образ жизни", "путешеств", "личное развит"],
            "en": ["lifestyle", "travel", "self-development", "personal growth"],
        },
        "expert_blog": {
            "ru": ["экспертн", "авторск", "специалист", "блог"],
            "en": ["expert", "author", "specialist", "blog"],
        },
        "massage": {
            "ru": ["массаж", "самомассаж", "массажист", "шея", "спина", "осанк", "плеч", "реабил", "восстанов"],
            "en": ["massage", "massage therapist", "bodywork", "therapist", "masseur"],
        },
        "cars": {
            "ru": ["машин", "авто", "автомоб", "электромоб", "тесла", "двигател"],
            "en": ["car", "cars", "automotive", "vehicle", "dashboard"],
        },
        "gaming": {
            "ru": ["игр", "гейм", "игрок", "консоль", "геймпад", "steam", "playstation", "xbox"],
            "en": ["gaming", "game", "games", "controller", "console", "esports"],
        },
        "hardware": {
            "ru": ["компьют", "ноут", "ноутбук", "пк", "процессор", "видеокарт", "ssd", "памят", "желез"],
            "en": ["computer", "laptop", "hardware", "pc", "workstation"],
        },
        "tech": {
            "ru": ["технолог", "программ", "разработ", "ии", "нейросет", "api"],
            "en": ["technology", "software", "ai", "developer", "startup"],
        },
        "business": {
            "ru": ["бизнес", "предпринимател", "стартап", "менеджмент"],
            "en": ["business", "entrepreneur", "startup", "management"],
        },
    }

    def detect_topic_family(text: str) -> str:
        q = re.sub(r"\s+", " ", (text or "").strip().lower().replace("ё", "е"))
        priority = [
            "massage", "food", "beauty", "local_business",
            "cars", "gaming", "hardware",
            "health",
            "education", "finance", "marketing", "lifestyle", "expert_blog",
            "tech", "business",
        ]
        for family in priority:
            block = TOPIC_FAMILY_TERMS.get(family, {})
            terms = block.get("ru", []) + block.get("en", [])
            if any(token in q for token in terms):
                return family
        return "generic"

    def get_family_news_angle(family: str) -> str:
        return "Раскрой эту новость через призму темы канала. Что изменилось? Почему это важно для подписчиков?"

    async def resolve_channel_policy(owner_id: int, **kwargs):
        return None

    def build_news_family_block(policy, **kwargs) -> str:
        return ""

logger = logging.getLogger(__name__)
NEWS_MAX_AGE_DAYS = 3  # Enforce strict recency; articles older than 3 days are stale for an active channel
MIN_NEWS_RELEVANCE = 18

# Stage-1 (broad recall) uses a much lower threshold to avoid discarding
# candidates too early. Stage-2 reranking picks the best from the broader pool.
_STAGE1_MIN_RELEVANCE = 6
_STAGE1_MIN_HITS = 1  # at least 1 relevance term hit to survive stage-1


def _clean(text: str) -> str:
    text = html.unescape(text or "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower().replace("ё", "е")).strip()


def _topic_default_sources(topic: str) -> str:
    family = detect_topic_family(topic or "")
    _FAMILY_SOURCES: dict[str, list[str]] = {
        "food": ["eater.com", "foodandwine.com", "bonappetit.com", "seriouseats.com",
                 "theguardian.com", "forbes.com", "reuters.com"],
        "health": ["healthline.com", "webmd.com", "medicalnewstoday.com", "mayoclinic.org",
                   "theguardian.com", "reuters.com", "forbes.com"],
        "beauty": ["allure.com", "byrdie.com", "beautybay.com", "refinery29.com",
                   "vogue.com", "theguardian.com", "forbes.com"],
        "local_business": ["entrepreneur.com", "inc.com", "smallbiztrends.com",
                           "businessinsider.com", "forbes.com", "reuters.com"],
        "education": ["edutopia.org", "chronicle.com", "insidehighered.com",
                      "theguardian.com", "edsurge.com", "reuters.com", "forbes.com"],
        "finance": ["reuters.com", "bloomberg.com", "wsj.com", "ft.com",
                    "investopedia.com", "cnbc.com", "forbes.com"],
        "marketing": ["marketingweek.com", "adweek.com", "marketingland.com",
                      "hubspot.com", "theguardian.com", "forbes.com", "reuters.com"],
        "lifestyle": ["theguardian.com", "huffpost.com", "mindbodygreen.com",
                      "wellandgood.com", "forbes.com", "reuters.com"],
        "expert_blog": ["medium.com", "substack.com", "theguardian.com",
                        "harvard.edu", "forbes.com", "reuters.com"],
        "massage": ["massagemag.com", "amtamassage.org", "massagetoday.com",
                    "abmp.com", "healthline.com", "forbes.com", "theguardian.com"],
        "cars": ["reuters.com", "bloomberg.com", "electrek.co", "cnbc.com",
                 "caranddriver.com", "motortrend.com", "theverge.com"],
        "gaming": ["ign.com", "gamespot.com", "pcgamer.com", "eurogamer.net",
                   "kotaku.com", "polygon.com", "theverge.com"],
        "hardware": ["tomshardware.com", "arstechnica.com", "anandtech.com",
                     "theverge.com", "techcrunch.com", "pcworld.com", "reuters.com"],
        "tech": ["techcrunch.com", "theverge.com", "wired.com", "arstechnica.com",
                 "reuters.com", "bloomberg.com", "vc.ru"],
        "business": ["reuters.com", "bloomberg.com", "wsj.com", "ft.com",
                     "entrepreneur.com", "inc.com", "forbes.com"],
    }
    sources = _FAMILY_SOURCES.get(family)
    if sources:
        return ",".join(sources)
    # Safe generic fallback — not tech-biased
    return ",".join(["reuters.com", "apnews.com", "theguardian.com", "bbc.com",
                     "bloomberg.com", "forbes.com", "ap.org"])


def _extract_keywords(text: str, *, limit: int = 8) -> list[str]:
    tokens = re.findall(r"[a-zа-яё0-9][a-zа-яё0-9+-]{2,}", _normalize(text))
    stop = {
        "для", "про", "это", "как", "что", "будет", "после", "когда", "новости", "новость", "свежие",
        "канал", "канала", "телеграм", "telegram", "post", "news", "latest", "about", "with", "from",
        "that", "this", "your", "their", "через", "чтобы", "последние", "сегодня", "тема"
    }
    out = []
    seen = set()
    for token in tokens:
        if len(token) < 4 or token in stop:
            continue
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out[:limit]


def _topic_query_variants(topic: str, settings: dict[str, str]) -> list[str]:
    topic = _clean(topic)
    family = detect_topic_family(topic)
    rubrics = _clean(settings.get("content_rubrics", "") or settings.get("rubrics_schedule", ""))
    scenarios = _clean(settings.get("post_scenarios", ""))
    audience = _clean(settings.get("channel_audience", ""))
    strict_mode = str(settings.get("news_strict_mode", "1")).strip() in ("1", "true", "yes", "on")
    keywords = _extract_keywords(" ".join([topic, rubrics, scenarios, audience]), limit=10)
    joined = " ".join(keywords[:5]).strip()
    family_en = " ".join(TOPIC_FAMILY_TERMS.get(family, {}).get("en", [])[:3]).strip()
    family_ru = " ".join(TOPIC_FAMILY_TERMS.get(family, {}).get("ru", [])[:3]).strip()
    # Trending/popular filter terms (EN + RU)
    trend_clause = "trending OR popular OR top OR viral OR тренд OR популярн OR топ"
    variants = []
    if topic:
        variants.append(f'"{topic}" ({trend_clause}) when:7d')
    if joined:
        variants.append(f'{joined} ({trend_clause}) when:7d')
        variants.append(f'{joined} trend OR launch OR report OR announced when:7d')
    if family != "generic" and (family_en or family_ru):
        seed = " ".join(x for x in [family_ru, family_en] if x).strip()
        variants.append(f'{seed} ({trend_clause}) OR market OR product OR research when:7d')
    if topic and audience:
        variants.append(f'{topic} {audience} when:7d')
    # If strict mode is off, add broad trending queries to capture popular news beyond the specific topic
    if not strict_mode:
        variants.append(f'trending OR viral OR popular news when:3d')
        variants.append(f'top news today OR breaking when:2d')
    seen = set()
    out = []
    for item in variants:
        item = re.sub(r"\s+", " ", item).strip()
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out[:6] if not strict_mode else out[:4]


def _build_rss_url(query: str, sources: list[str]) -> str:
    query = (query or "latest news").strip() or "latest news"
    if sources:
        source_expr = " OR ".join([f"site:{s.strip()}" for s in sources if s.strip()])
        query = f"{query} ({source_expr})"
    encoded = urllib.parse.quote(query)
    return f"https://news.google.com/rss/search?q={encoded}&hl=ru&gl=RU&ceid=RU:ru"


def _parse_pub_date(value: str) -> datetime | None:
    raw = (value or "").strip()
    if not raw:
        return None
    try:
        dt = parsedate_to_datetime(raw)
        return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


async def _extract_article_text_and_image(url: str) -> tuple[str, str]:
    try:
        async with httpx.AsyncClient(timeout=25, follow_redirects=True) as client:
            r = await client.get(url)
        if r.status_code != 200:
            return "", ""
        html_text = r.text or ""
        article_text = trafilatura.extract(
            html_text,
            include_comments=False,
            include_tables=False,
            favor_precision=True,
            output_format="txt",
        ) or ""
        og_image = ""
        m = re.search(r"<meta[^>]+property=['\"]og:image['\"][^>]+content=['\"]([^'\"]+)['\"]", html_text, flags=re.I)
        if not m:
            m = re.search(r"<meta[^>]+content=['\"]([^'\"]+)['\"][^>]+property=['\"]og:image['\"]", html_text, flags=re.I)
        if m:
            og_image = m.group(1).strip()
        return article_text.strip(), og_image.strip()
    except Exception:
        return "", ""


def _relevance_terms(topic: str, settings: dict[str, str]) -> list[str]:
    family = detect_topic_family(topic)
    family_terms = TOPIC_FAMILY_TERMS.get(family, {}).get("ru", []) + TOPIC_FAMILY_TERMS.get(family, {}).get("en", [])
    extra = _extract_keywords(" ".join([
        topic,
        settings.get("content_rubrics", ""),
        settings.get("post_scenarios", ""),
        settings.get("channel_audience", ""),
    ]), limit=12)
    seen = set()
    out = []
    for term in family_terms + extra:
        term = _normalize(term)
        if not term or len(term) < 3 or term in seen:
            continue
        seen.add(term)
        out.append(term)
    return out[:16]


def _candidate_score(topic: str, title: str, description: str, article_text: str, domain: str, pub_date: datetime | None, settings: dict[str, str]) -> tuple[int, int]:
    score = 0
    body = _normalize(f"{title} {description} {article_text[:1800]}")
    title_l = _normalize(title)
    rel_terms = _relevance_terms(topic, settings)
    relevance_hits = 0
    for term in rel_terms:
        if term in title_l:
            score += 12
            relevance_hits += 2
        elif term in body:
            score += 7
            relevance_hits += 1
    if article_text:
        score += min(len(article_text) // 350, 8)
    if domain:
        score += 3
    high_signal = [
        "announced", "launch", "released", "report", "research", "funding", "рынок", "исслед", "релиз",
        "запуск", "анонс", "рост", "обновлен", "обновление", "тренд", "trend", "market",
        "breaking", "первый", "новый рекорд", "рекордн",
    ]
    high_signal_hits = sum(1 for x in high_signal if x in body)
    if high_signal_hits >= 2:
        score += 20  # Strongly trending signal
    elif high_signal_hits >= 1:
        score += 12
    low_signal = ["opinion", "мнение", "advertisement", "promo", "affiliate", "скидк", "coupon", "deals",
                   "sponsored", "партнерск", "реклама"]
    if any(x in body for x in low_signal):
        score -= 20
    # Weak / dubious source penalty
    dubious_domains = ["zen.yandex", "pulse.mail", "dzen.ru"]
    if domain and any(d in domain for d in dubious_domains):
        score -= 10
    # SEO / aggregator / content-farm penalty (heuristic: domains known for low-quality repackaged content)
    _seo_aggregator_domains = [
        "pulse.mail.ru", "zen.yandex", "dzen.ru", "rambler.ru/news",
        "yandex.ru/news", "news.google", "flipboard.com",
        "ria.ru/amp", "tass.ru/amp",  # AMP pages often lack full context
    ]
    if domain and any(d in domain for d in _seo_aggregator_domains):
        score -= 8  # Additional penalty on top of dubious_domains
    # Pseudo-expert / clickbait article penalty
    _clickbait_signals = [
        "вы не поверите", "шок", "сенсация", "топ-10", "top 10", "you won't believe",
        "shocking", "amazing trick",
    ]
    if sum(1 for s in _clickbait_signals if s in body) >= 2:
        score -= 15
    # Penalise off-topic articles based on family — ensures news stays on-niche
    family = detect_topic_family(topic)
    # Tokens that signal irrelevant tech/finance/corporate news for non-tech/non-finance channels
    _TECH_FINANCE_NOISE = [
        "semiconductor", "chip shortage", "ipo ", "quarterly earnings", "stock market crash",
        "federal reserve", "interest rate hike", "ФРС", "ключевая ставка",
    ]
    if family == "massage":
        off_topic = [
            "cancer", "tumor", "tumour", "diabetes", "vaccine", "covid", "surgery",
            "chemotherapy", "hiv", "aids", "oncolog", "карцином", "онколог", "химиотерап",
            "диабет", "вакцин", "хирург",
        ] + _TECH_FINANCE_NOISE
        if any(x in body for x in off_topic):
            score -= 40
    elif family == "food":
        off_topic = _TECH_FINANCE_NOISE + [
            "косметик", "маникюр", "стриж", "tattoo", "gaming", "игровой",
        ]
        if any(x in body for x in off_topic):
            score -= 30
    elif family == "health":
        off_topic = _TECH_FINANCE_NOISE + [
            "рецепт блюда", "fashion week", "gaming release", "car launch", "автомобил",
        ]
        if any(x in body for x in off_topic):
            score -= 25
    elif family == "beauty":
        off_topic = _TECH_FINANCE_NOISE + [
            "gaming", "automotive", "server infrastructure", "cloud computing",
            "chip architecture",
        ]
        if any(x in body for x in off_topic):
            score -= 30
    elif family == "local_business":
        off_topic = _TECH_FINANCE_NOISE + [
            "gaming", "fashion week", "celebrity", "movie release", "album release",
        ]
        if any(x in body for x in off_topic):
            score -= 25
    elif family == "education":
        off_topic = _TECH_FINANCE_NOISE + [
            "gaming release", "fashion week", "beauty product", "automotive",
        ]
        if any(x in body for x in off_topic):
            score -= 20
    elif family == "finance":
        # Finance channels need business/markets — penalise clearly off-topic lifestyle content
        off_topic = [
            "рецепт", "косметик", "маникюр", "шампунь", "recipe", "beauty product",
            "nail salon", "gaming", "celebrity gossip",
        ]
        if any(x in body for x in off_topic):
            score -= 25
    elif family == "marketing":
        off_topic = _TECH_FINANCE_NOISE + [
            "gaming", "automotive", "beauty product", "recipe",
        ]
        if any(x in body for x in off_topic):
            score -= 20
    elif family == "lifestyle":
        off_topic = _TECH_FINANCE_NOISE + [
            "server infrastructure", "chip architecture", "gaming tournament",
            "automotive recall",
        ]
        if any(x in body for x in off_topic):
            score -= 20
    elif family == "expert_blog":
        off_topic = _TECH_FINANCE_NOISE + [
            "celebrity gossip", "gaming release", "fashion week",
        ]
        if any(x in body for x in off_topic):
            score -= 15
    elif family in ("cars", "gaming", "hardware"):
        # These accept some tech/finance overlap; penalise clearly unrelated lifestyle
        off_topic = [
            "beauty product", "косметик", "рецепт", "маникюр", "fashion week",
        ]
        if any(x in body for x in off_topic):
            score -= 20
    elif family == "tech":
        off_topic = [
            "рецепт", "косметик", "маникюр", "beauty product", "fashion week",
            "celebrity gossip",
        ]
        if any(x in body for x in off_topic):
            score -= 20
    elif family == "business":
        off_topic = [
            "рецепт", "косметик", "gaming release", "celebrity gossip",
        ]
        if any(x in body for x in off_topic):
            score -= 15
    if title.count(':') >= 2:
        score -= 3
    if pub_date:
        age_hours = max(0.0, (datetime.now(timezone.utc) - pub_date).total_seconds() / 3600.0)
        if age_hours <= 6:
            score += 18  # Very fresh — breaking news priority
        elif age_hours <= 18:
            score += 12
        elif age_hours <= 48:
            score += 8
        elif age_hours <= 96:
            score += 4
        elif age_hours > NEWS_MAX_AGE_DAYS * 24:
            score -= 18
        # Continuous freshness decay (heuristic): small bonus for very recent articles
        # Articles under 3 hours get extra boost to prioritize breaking news
        if age_hours <= 3:
            score += 6
    else:
        score -= 4

    # Title-topic relevance bonus: extra points if topic keywords appear directly in the title
    topic_words = [w.strip() for w in re.split(r"[,\s]+", _normalize(topic)) if len(w.strip()) >= 4]
    title_topic_hits = sum(1 for tw in topic_words if tw in title_l)
    if title_topic_hits >= 2:
        score += 10  # Strong title-topic alignment
    elif title_topic_hits >= 1:
        score += 5

    return score, relevance_hits


async def fetch_news_candidates(owner_id: int = 0, limit: int = 5, *, channel_target: str = "", channel_profile_id: int | None = None) -> list[dict]:
    """Fetch news candidates using 2-stage pipeline: broad recall → rerank.

    Stage 1: Gather many raw candidates with a loose relevance threshold.
    Stage 2: Rerank by freshness, relevance, source quality, dedup.

    When *channel_profile_id* is provided, settings are read from that specific
    channel profile — not the active channel and not the owner-level flat table.
    This prevents cross-channel contamination for multi-channel users.
    """
    ch_settings = await db.get_channel_settings(owner_id, channel_profile_id=channel_profile_id)
    topic = (ch_settings.get("topic") or "").strip()

    settings: dict[str, str] = {}
    for k in (
        "topic", "channel_audience", "channel_style", "content_rubrics", "rubrics_schedule",
        "post_scenarios", "news_sources", "content_exclusions", "news_strict_mode",
    ):
        settings[k] = ch_settings.get(k, "") or ""

    if not topic:
        return []

    raw_sources = (settings.get("news_sources") or _topic_default_sources(topic))
    sources = [s.strip() for s in str(raw_sources).split(",") if s.strip()]
    queries = _topic_query_variants(topic, settings)
    exclusions_raw = (settings.get("content_exclusions") or "").strip()
    exclusion_terms = [_normalize(t) for t in re.split(r"[,\n]+", exclusions_raw) if t.strip()] if exclusions_raw else []

    # ── Stage 1: broad recall ──────────────────────────────────────────
    raw_candidates: list[dict] = []
    stage1_filtered = 0
    stage1_exclusion_filtered = 0
    stage1_dedup_filtered = 0
    stage1_used_filtered = 0
    seen_links: set[str] = set()
    seen_title_tokens: list[set[str]] = []  # For near-duplicate detection
    max_raw = max(30, limit * 6)  # Gather more raw candidates

    for query in queries:
        for scoped in (_build_rss_url(query, sources), _build_rss_url(query, [])):
            try:
                async with httpx.AsyncClient(timeout=25, follow_redirects=True) as client:
                    r = await client.get(scoped)
                if r.status_code != 200:
                    continue
                root = ET.fromstring(r.text)
            except Exception:
                continue
            channel = root.find("channel")
            if channel is None:
                continue
            for item in channel.findall("item"):
                title = _clean(item.findtext("title", ""))
                link = (item.findtext("link", "") or "").strip()
                description = _clean(item.findtext("description", ""))
                pub_date = _parse_pub_date(item.findtext("pubDate", ""))
                if not link or not title or link in seen_links:
                    continue
                seen_links.add(link)
                if await db.is_news_used(link, owner_id=owner_id, channel_target=channel_target):
                    stage1_used_filtered += 1
                    continue
                # Filter out news matching content exclusions
                if exclusion_terms:
                    body_check = _normalize(f"{title} {description}")
                    if any(term in body_check for term in exclusion_terms):
                        stage1_exclusion_filtered += 1
                        continue
                # Near-duplicate detection: skip articles with >70% title overlap
                title_tokens = set(_normalize(title).split())
                if title_tokens and any(
                    len(title_tokens & prev) / max(len(title_tokens | prev), 1) > 0.7
                    for prev in seen_title_tokens
                ):
                    stage1_dedup_filtered += 1
                    continue
                seen_title_tokens.append(title_tokens)

                article_text, image_url = await _extract_article_text_and_image(link)
                domain = urlparse(link).netloc
                score, relevance_hits = _candidate_score(topic, title, description, article_text, domain, pub_date, settings)

                # Stage-1: loose threshold — keep candidates with minimal relevance
                if relevance_hits < _STAGE1_MIN_HITS or score < _STAGE1_MIN_RELEVANCE:
                    stage1_filtered += 1
                    continue

                # Source confidence check
                _source_conf = bool(article_text and len(article_text.strip()) >= 100)
                if not _source_conf:
                    score -= 8  # Weak source — headline + snippet only
                raw_candidates.append({
                    "title": title,
                    "link": link,
                    "description": description[:400],
                    "topic": topic,
                    "sources": sources,
                    "article_text": article_text[:5000],
                    "image_url": image_url,
                    "domain": domain,
                    "score": score,
                    "pub_date": pub_date.isoformat() if pub_date else "",
                    "relevance_hits": relevance_hits,
                })
                if len(raw_candidates) >= max_raw:
                    break
            if len(raw_candidates) >= max_raw:
                break
        if len(raw_candidates) >= max_raw:
            break

    logger.info(
        "NEWS_STAGE1 raw_collected=%d filtered_score=%d filtered_exclusion=%d "
        "filtered_dedup=%d filtered_used=%d topic=%r queries=%d",
        len(raw_candidates), stage1_filtered, stage1_exclusion_filtered,
        stage1_dedup_filtered, stage1_used_filtered,
        (topic or "")[:60], len(queries),
    )

    if not raw_candidates:
        logger.info("NEWS_STAGE1_EMPTY topic=%r — no candidates survived broad recall", (topic or "")[:60])
        return []

    # ── Stage 2: rerank ────────────────────────────────────────────────
    # Separate into strong (passes original threshold) and weak candidates
    strong: list[dict] = []
    weak: list[dict] = []
    for c in raw_candidates:
        if c.get("relevance_hits", 0) >= 2 and c.get("score", 0) >= MIN_NEWS_RELEVANCE:
            strong.append(c)
        else:
            weak.append(c)

    # Sort strong by score desc, weak by score desc
    strong.sort(key=lambda x: (x.get("score", 0), x.get("pub_date", "")), reverse=True)
    weak.sort(key=lambda x: (x.get("score", 0), x.get("pub_date", "")), reverse=True)

    # Merge: strong first, then weak as fallback
    candidates = strong + weak

    logger.info(
        "NEWS_STAGE2 strong=%d weak=%d total=%d",
        len(strong), len(weak), len(candidates),
    )

    # Diversity pass: avoid returning candidates from the same domain/story cluster.
    # The top-scoring candidate is always kept regardless of domain limits,
    # so the diversity filter never kills the single best result.
    if len(candidates) > limit:
        diverse: list[dict] = []
        used_domains: dict[str, int] = {}
        for i, c in enumerate(candidates):
            d = c.get("domain", "")
            # Always keep the #1 candidate (best score)
            if i == 0 or used_domains.get(d, 0) < 2:
                diverse.append(c)
                used_domains[d] = used_domains.get(d, 0) + 1
                if len(diverse) >= limit:
                    break
        # If diversity filter is too aggressive, pad with remaining
        if len(diverse) < limit:
            for c in candidates:
                if c not in diverse:
                    diverse.append(c)
                    if len(diverse) >= limit:
                        break
        result = diverse[:limit]
    else:
        result = candidates[:limit]

    logger.info(
        "NEWS_FINAL returned=%d best_score=%d best_title=%r",
        len(result),
        result[0].get("score", 0) if result else 0,
        (result[0].get("title", "")[:60]) if result else "",
    )
    return result

    return candidates[:limit]


async def fetch_latest_news(owner_id: int = 0, *, channel_target: str = "", channel_profile_id: int | None = None) -> dict | None:
    items = await fetch_news_candidates(owner_id=owner_id, limit=1, channel_target=channel_target, channel_profile_id=channel_profile_id)
    return items[0] if items else None




def build_news_source_meta(news_item: dict) -> str:
    """Build a compact JSON string with verified source metadata for a news item.

    Fields: source_title, domain, url, published_at, original_headline.
    """
    return json.dumps({
        "source_title": str(news_item.get("title") or "")[:200],
        "domain": str(news_item.get("domain") or urlparse(news_item.get("link", "")).netloc)[:100],
        "url": str(news_item.get("link") or "")[:500],
        "published_at": str(news_item.get("pub_date") or ""),
        "original_headline": str(news_item.get("title") or "")[:300],
    }, ensure_ascii=False)


def is_source_confident(news_item: dict) -> bool:
    """Return True if the news item has enough source backing to be treated as real news.

    A confident source must have:
    - a non-empty link (URL)
    - a recognizable domain
    - article body text (not just a headline + description)
    """
    link = str(news_item.get("link") or "").strip()
    domain = str(news_item.get("domain") or "").strip()
    article_text = str(news_item.get("article_text") or "").strip()
    title = str(news_item.get("title") or "").strip()

    if not link or not title:
        return False
    if not domain:
        return False
    # Article text is the strongest signal — without it we only have headline + RSS snippet
    if len(article_text) < 100:
        return False
    return True


def plan_news_brief(items: list[dict]) -> str:
    if not items:
        return "- релевантные новости не найдены"
    lines = []
    for item in items[:5]:
        title = _clean(item.get("title", ""))
        domain = item.get("domain") or urlparse(item.get("link", "")).netloc
        desc = _clean(item.get("description", ""))[:160]
        pub = item.get("pub_date", "")
        pub_mark = ""
        if pub:
            try:
                pub_mark = f", {datetime.fromisoformat(pub).astimezone().strftime('%d.%m %H:%M')}"
            except Exception:
                pub_mark = ""
        lines.append(f"- {title} ({domain}{pub_mark}): {desc}")
    return "\n".join(lines)


async def build_news_post(config, news_item: dict, owner_id: int = 0) -> str:
    title = news_item.get("title", "")
    description = news_item.get("description", "")
    topic = news_item.get("topic", "")
    link = news_item.get("link", "")
    article_text = (news_item.get("article_text") or "").strip()
    domain = urlparse(link).netloc
    pub_date = news_item.get("pub_date", "")

    # Build source attribution line
    source_line = f"Источник: {domain}"
    if pub_date:
        try:
            pub_fmt = datetime.fromisoformat(pub_date).strftime("%d.%m.%Y")
            source_line = f"Источник: {domain} ({pub_fmt})"
        except Exception:
            pass

    # Resolve channel policy for family-aware news transformation
    try:
        policy = await resolve_channel_policy(owner_id)
        family_block = build_news_family_block(policy, news_title=title, news_topic=topic) if policy else ""
        news_angle = (policy.news_angle if policy else "") or get_family_news_angle(detect_topic_family(topic))
    except (ImportError, AttributeError, KeyError, ValueError, TypeError, RuntimeError, OSError):
        policy = None
        family_block = ""
        news_angle = get_family_news_angle(detect_topic_family(topic))

    family_instruction = f"\n\nИнструкция по подаче:\n{news_angle}\n" if news_angle else ""

    if article_text:
        prompt = (
            f"Сделай короткий сильный телеграм-пост по свежей новости на тему '{topic}'. "
            "Пиши как редактор живого тематического канала, а не как новостной агрегатор. "
            "Используй только факты из статьи, без вымысла. Не пересказывай заголовок. "
            "Вытащи один конкретный смысл для подписчика: что изменилось, почему это важно и к чему это может привести. "
            "Без воды, без канцелярита, без кликбейта. "
            "Целевой объём: 60-100 слов, максимум 2 абзаца. Убирай вводные и воду."
            f"{family_instruction}"
            f"\nЗаголовок новости: {title}\n"
            f"Текст статьи:\n{article_text}\n\n"
            f"В конце ОБЯЗАТЕЛЬНО добавь строку: {source_line}"
        )
    else:
        prompt = (
            f"Сделай короткий телеграм-пост по свежей новости на тему '{topic}'. "
            "Используй только заголовок и описание. Ничего не придумывай. "
            "Нужен не пересказ, а понятный подписчику вывод: почему это стоит заметить именно сейчас. "
            "Целевой объём: 50-80 слов, максимум 2 абзаца."
            f"{family_instruction}"
            f"\nЗаголовок: {title}\nОписание: {description}\n\n"
            f"В конце ОБЯЗАТЕЛЬНО добавь строку: {source_line}"
        )
    return await generate_post_text(
        api_key=config.openrouter_api_key,
        model=config.openrouter_model,
        topic=topic,
        prompt=prompt,
        base_url=getattr(config, "openrouter_base_url", None),
        owner_id=owner_id,
        generation_path="news",
    )
