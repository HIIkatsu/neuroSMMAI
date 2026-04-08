
from __future__ import annotations

import asyncio
import email.utils
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from html import unescape
from typing import Any
from urllib.parse import quote_plus, urljoin
from xml.etree import ElementTree as ET

import httpx

logger = logging.getLogger(__name__)

USER_AGENT = "Mozilla/5.0 NeuroSMM/1.0"
_TIMEOUT = httpx.Timeout(12.0, connect=5.0)

@dataclass
class EvidenceItem:
    title: str
    source: str
    date: str
    url: str
    relevance: float
    confidence: float
    why_it_matters: str
    domain: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "source": self.source,
            "date": self.date,
            "url": self.url,
            "relevance": round(self.relevance, 2),
            "confidence": round(self.confidence, 2),
            "why_it_matters": self.why_it_matters,
            "domain": self.domain,
        }


def clean_text(text: str) -> str:
    text = str(text or "").replace("\xa0", " ").replace("ё", "е")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def word_set(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-zа-я0-9]+", clean_text(text).lower()) if len(w) >= 3}


_STRONG_LIVE = [
    "сегодня", "сейчас", "новинки", "новости", "что нового", "свеж", "релиз", "вышло", "вышедшие",
    "распродаж", "скидк", "апдейт", "обновлен", "фестиваль", "latest", "recent", "release",
    "sale", "discount", "update", "festival", "new in",
]
_SOFT_LIVE = [
    "steam", "стим", "iphone", "telegram", "openai", "tesla", "игры 2026", "в 2026", "2026",
]
_EVERGREEN = [
    "как выбрать", "идеи для постов", "как понять", "зачем", "почему", "массаж", "ремонт", "услуга",
]

def classify_prompt_mode(channel_topic: str, requested: str) -> str:
    src = clean_text(f"{channel_topic} {requested}").lower()
    strong_live = any(m in src for m in _STRONG_LIVE)
    soft_live = any(m in src for m in _SOFT_LIVE)
    evergreen = any(m in src for m in _EVERGREEN)
    if strong_live and evergreen:
        return "mixed"
    if strong_live:
        return "live_intel"
    if soft_live and not evergreen:
        return "mixed"
    return "evergreen_editorial"


def infer_live_domain(channel_topic: str, requested: str) -> str:
    src = clean_text(f"{channel_topic} {requested}").lower()
    if any(x in src for x in ["steam", "стим", "игр", "gaming", "game", "playstation", "xbox", "epic"]):
        return "gaming"
    if any(x in src for x in ["iphone", "apple", "telegram", "openai", "chatgpt", "tesla", "android", "смартфон", "ноутбук", "macbook"]):
        return "tech"
    if any(x in src for x in ["акци", "курс", "бирж", "bitcoin", "btc", "eth", "финанс", "ставк", "инфляц", "nasdaq", "s&p"]):
        return "finance"
    if any(x in src for x in ["закон", "налог", "правил", "регуляц", "такси", "штраф", "постановлен", "минтранс"]):
        return "legal"
    if any(x in src for x in ["машин", "авто", "автомоб", "электромоб", "акб", "аккумулятор", "tesla", "bmw", "toyota"]):
        return "auto"
    if any(x in src for x in ["маркетинг", "instagram", "tiktok", "youtube", "соцсет", "реклам", "seo"]):
        return "marketing"
    return "generic"


def source_registry(domain: str) -> dict[str, Any]:
    registry = {
        "gaming": {
            "google_queries": [
                "{q} Steam release OR sale OR update",
                "{q} game official news",
                "{q} site:store.steampowered.com OR site:steamdb.info",
            ],
            "preferred_domains": ["store.steampowered.com", "steamdb.info", "steamcommunity.com", "steampowered.com"],
            "extra_fetchers": ["steam_search", "steam_specials"],
        },
        "tech": {
            "google_queries": [
                "{q} official update OR release notes",
                "{q} newsroom official",
                "{q} changelog official",
            ],
            "preferred_domains": ["apple.com", "telegram.org", "openai.com", "support.google.com", "android.com"],
            "extra_fetchers": [],
        },
        "finance": {
            "google_queries": [
                "{q} official data latest",
                "{q} regulator latest",
                "{q} market update official",
            ],
            "preferred_domains": ["federalreserve.gov", "ecb.europa.eu", "sec.gov", "cbr.ru"],
            "extra_fetchers": [],
        },
        "legal": {
            "google_queries": [
                "{q} официальный закон обновление",
                "{q} government official notice",
                "{q} regulator update official",
            ],
            "preferred_domains": [".gov", "government.ru", "consultant.ru", "publication.pravo.gov.ru"],
            "extra_fetchers": [],
        },
        "auto": {
            "google_queries": [
                "{q} official update OR recall OR release",
                "{q} brand official news",
                "{q} automotive latest news",
            ],
            "preferred_domains": ["tesla.com", "toyota.com", "bmw.com", "ford.com", "kia.com"],
            "extra_fetchers": [],
        },
        "marketing": {
            "google_queries": [
                "{q} official blog update",
                "{q} platform changelog official",
                "{q} help center update",
            ],
            "preferred_domains": ["telegram.org", "business.instagram.com", "blog.google", "support.google.com", "tiktok.com"],
            "extra_fetchers": [],
        },
        "generic": {
            "google_queries": [
                "{q} latest news",
                "{q} recent update",
                "{q} official release",
            ],
            "preferred_domains": [],
            "extra_fetchers": [],
        },
    }
    return registry.get(domain, registry["generic"])


async def _http_get(url: str) -> str:
    headers = {"User-Agent": USER_AGENT}
    async with httpx.AsyncClient(timeout=_TIMEOUT, headers=headers, follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.text


async def fetch_google_news_rss(query: str) -> list[dict[str, str]]:
    url = "https://news.google.com/rss/search?q=" + quote_plus(query) + "&hl=ru&gl=RU&ceid=RU:ru"
    try:
        xml_text = await _http_get(url)
        root = ET.fromstring(xml_text)
        items = []
        for item in root.findall(".//item")[:12]:
            title = clean_text(item.findtext("title") or "")
            link = clean_text(item.findtext("link") or "")
            pub_date = clean_text(item.findtext("pubDate") or "")
            source_el = item.find("source")
            source = clean_text(source_el.text if source_el is not None else "")
            if title and link:
                items.append({"title": title, "url": link, "date": pub_date, "source": source})
        return items
    except Exception as exc:
        logger.warning("google news rss fetch failed query=%r err=%s", query, exc)
        return []


def _parse_steam_results(html: str, source_name: str) -> list[dict[str, str]]:
    rows = re.findall(
        r'<a[^>]+href="(?P<url>https://store\.steampowered\.com/app/\d+/[^"]+/?)"[^>]*>.*?<span class="title">(?P<title>.*?)</span>.*?(?:<div class="col search_released responsive_secondrow">(?P<date>.*?)</div>)?',
        html,
        flags=re.S | re.I,
    )
    items = []
    seen = set()
    for url, title, date in rows[:12]:
        title = clean_text(unescape(re.sub(r"<.*?>", " ", title)))
        date = clean_text(unescape(re.sub(r"<.*?>", " ", date)))
        key = (url, title)
        if not title or key in seen:
            continue
        seen.add(key)
        items.append({"title": title, "url": url, "date": date, "source": source_name})
    return items


async def fetch_steam_search(requested: str) -> list[dict[str, str]]:
    q = quote_plus(requested[:80])
    urls = [
        f"https://store.steampowered.com/search/?term={q}&sort_by=Released_DESC&supportedlang=english&ndl=1",
        f"https://store.steampowered.com/search/?term={q}&specials=1&supportedlang=english&ndl=1",
    ]
    out: list[dict[str, str]] = []
    for i, url in enumerate(urls):
        try:
            html = await _http_get(url)
            out.extend(_parse_steam_results(html, "Steam Store"))
        except Exception as exc:
            logger.warning("steam search fetch failed url=%r err=%s", url, exc)
    return out


async def fetch_steam_specials() -> list[dict[str, str]]:
    url = "https://store.steampowered.com/search/?specials=1&supportedlang=english&ndl=1"
    try:
        html = await _http_get(url)
        return _parse_steam_results(html, "Steam Specials")
    except Exception as exc:
        logger.warning("steam specials fetch failed err=%s", exc)
        return []


def _parse_pub_date(date_text: str) -> datetime | None:
    raw = clean_text(date_text)
    if not raw:
        return None
    try:
        dt = email.utils.parsedate_to_datetime(raw)
        if dt and dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        pass
    for fmt in ("%b %d, %Y", "%d %b, %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
        except Exception:
            pass
    return None


def score_evidence_item(item: dict[str, str], requested: str, domain: str, preferred_domains: list[str]) -> EvidenceItem:
    title = clean_text(item.get("title") or "")
    source = clean_text(item.get("source") or "")
    url = clean_text(item.get("url") or "")
    req_words = word_set(requested)
    title_words = word_set(title)
    overlap = len(req_words & title_words)

    score = overlap * 2.2
    url_low = url.lower()
    if any(dom in url_low for dom in preferred_domains):
        score += 2.5
    dt = _parse_pub_date(item.get("date") or "")
    if dt:
        days = max((datetime.now(timezone.utc) - dt).days, 0)
        if days <= 3:
            score += 2.2
        elif days <= 14:
            score += 1.2
        elif days <= 45:
            score += 0.6

    title_low = title.lower()
    domain_boost_terms = {
        "gaming": ["steam", "sale", "discount", "fest", "demo", "update", "released", "launch", "early access", "dlc"],
        "tech": ["update", "release", "beta", "patch", "feature", "ai", "ios", "android", "mac", "iphone"],
        "finance": ["rate", "inflation", "earnings", "guidance", "price", "etf", "fed", "ecb"],
        "legal": ["law", "rule", "regulation", "official", "government", "ministry", "court", "decree"],
        "auto": ["recall", "update", "model", "launch", "battery", "ev", "hybrid"],
        "marketing": ["update", "algorithm", "ads", "feature", "policy", "changelog"],
    }
    for term in domain_boost_terms.get(domain, []):
        if term in title_low:
            score += 0.9

    if re.search(r"\b202[456]\b", requested) and re.search(r"\b202[456]\b", title):
        score += 1.0

    why = title or source
    if overlap:
        why = f"прямо связано с запросом: {title}"
    elif source:
        why = f"свежий источник по теме из {source}"
    conf = min(0.5 + score * 0.07, 0.98)
    return EvidenceItem(
        title=title,
        source=source or url,
        date=clean_text(item.get("date") or ""),
        url=url,
        relevance=score,
        confidence=conf,
        why_it_matters=why,
        domain=domain,
    )


async def build_fresh_facts(channel_topic: str, requested: str, *, limit: int = 8) -> list[dict[str, Any]]:
    domain = infer_live_domain(channel_topic, requested)
    registry = source_registry(domain)
    raw_items: list[dict[str, str]] = []

    for template in registry["google_queries"][:4]:
        query = template.format(q=requested)
        raw_items.extend(await fetch_google_news_rss(query))

    extra = registry.get("extra_fetchers", [])
    if "steam_search" in extra:
        raw_items.extend(await fetch_steam_search(requested))
    if "steam_specials" in extra and any(x in requested.lower() for x in ["скид", "sale", "распрод", "steam", "стим", "новин"]):
        raw_items.extend(await fetch_steam_specials())

    uniq: dict[str, dict[str, str]] = {}
    for item in raw_items:
        key = clean_text(item.get("url") or item.get("title") or "").lower()
        if key and key not in uniq:
            uniq[key] = item

    scored = [score_evidence_item(item, requested, domain, registry["preferred_domains"]) for item in uniq.values()]
    scored.sort(key=lambda x: (-x.relevance, -x.confidence, x.title))

    strong = [x.to_dict() for x in scored if x.relevance >= 1.0][:limit]
    if strong:
        return strong

    # Не возвращаем пустоту слишком рано: даём верхние 3-5 кандидатов, если хоть что-то нашли.
    fallback = [x.to_dict() for x in scored[: min(limit, 5)]]
    return fallback
