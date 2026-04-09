"""
image_providers.py — Provider abstraction for image pipeline v3.

Provider strategy:
  - Pexels  = PRIMARY provider
  - Pixabay = SECONDARY provider
  - Openverse = EDITOR-ONLY weak fallback (NOT in strict autopost path)

Each provider returns a list of raw candidates (URL + metadata).
The pipeline then scores/ranks them separately.

Feature flags:
  - OPENVERSE_ENABLED: env var to enable/disable Openverse entirely
  - Openverse is auto-disabled for autopost mode regardless of flag
"""
from __future__ import annotations

import logging
import os
import re
import time as _time
from dataclasses import dataclass, field
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------
PEXELS_API_KEY = (os.getenv("PEXELS_API_KEY") or "").strip()
PIXABAY_API_KEY = (os.getenv("PIXABAY_API_KEY") or "").strip()
OPENVERSE_ENABLED = os.getenv("OPENVERSE_ENABLED", "true").strip().lower() in ("1", "true", "yes")

IMAGE_HTTP_PROXY_URL = (
    os.getenv("IMAGE_HTTP_PROXY_URL")
    or os.getenv("OPENROUTER_PROXY_URL")
    or os.getenv("HTTPS_PROXY")
    or os.getenv("HTTP_PROXY")
    or os.getenv("ALL_PROXY")
    or ""
).strip() or None

CONNECT_TIMEOUT = float(os.getenv("IMAGE_CONNECT_TIMEOUT_SEC", "4.0"))
READ_TIMEOUT = float(os.getenv("IMAGE_READ_TIMEOUT_SEC", "7.0"))

BAD_URL_PARTS = ("avatar", "icon", "logo", "sprite", "thumb", "placeholder")
PER_PAGE = 15  # Candidates per query per provider


# ---------------------------------------------------------------------------
# Raw candidate from a provider
# ---------------------------------------------------------------------------
@dataclass
class RawCandidate:
    """A single image candidate returned by a provider before scoring."""
    url: str = ""
    meta_text: str = ""         # Combined metadata (alt text, tags, description)
    provider: str = ""          # "pexels", "pixabay", "openverse"
    query: str = ""             # The search query that found this
    provider_score: int = 0     # Raw provider relevance signal (if available)
    width: int = 0
    height: int = 0


# ---------------------------------------------------------------------------
# Circuit breaker — skip providers with repeated failures
# ---------------------------------------------------------------------------
_FAILURE_THRESHOLD = 3
_COOLDOWN_SECONDS = 120
_failures: dict[str, int] = {}
_tripped_at: dict[str, float] = {}


def _provider_available(name: str) -> bool:
    tripped = _tripped_at.get(name)
    if tripped is None:
        return True
    if _time.monotonic() - tripped > _COOLDOWN_SECONDS:
        _failures.pop(name, None)
        _tripped_at.pop(name, None)
        return True
    return False


def _provider_success(name: str) -> None:
    _failures.pop(name, None)
    _tripped_at.pop(name, None)


def _provider_failure(name: str) -> None:
    count = _failures.get(name, 0) + 1
    _failures[name] = count
    if count >= _FAILURE_THRESHOLD:
        _tripped_at[name] = _time.monotonic()
        logger.warning(
            "Provider %s circuit-breaker tripped after %d failures (cooldown %ds)",
            name, count, _COOLDOWN_SECONDS,
        )


# ---------------------------------------------------------------------------
# URL validation
# ---------------------------------------------------------------------------
def _url_ok(url: str) -> bool:
    if not url or not url.startswith("http"):
        return False
    lowered = url.lower()
    if any(part in lowered for part in BAD_URL_PARTS):
        return False
    try:
        return bool(urlparse(url).netloc)
    except Exception:
        return False


def _image_fingerprint(url: str) -> str:
    """Create a fingerprint from URL for dedup."""
    raw = str(url or "").strip().lower()
    if not raw:
        return ""
    m = re.search(r"/photos/(\d+)/", raw)
    if m and "pexels" in raw:
        return f"pexels:{m.group(1)}"
    m = re.search(r"pixabay\.com.*?[_/](\d{6,})", raw)
    if m:
        return f"pixabay:{m.group(1)}"
    return raw.split("?", 1)[0]


def _not_used(url: str, used_fps: set[str]) -> bool:
    fp = _image_fingerprint(url)
    return bool(fp) and fp not in used_fps


def prepare_used_refs(used_refs: set[str] | None) -> set[str]:
    """Convert user-provided used refs to fingerprints."""
    return {_image_fingerprint(x) for x in (used_refs or set()) if _image_fingerprint(x)}


# ---------------------------------------------------------------------------
# HTTP client factory
# ---------------------------------------------------------------------------
def _resolved_proxy_url() -> str | None:
    proxy = (IMAGE_HTTP_PROXY_URL or "").strip()
    if not proxy:
        return None
    if proxy.lower().startswith("socks"):
        try:
            import socksio  # noqa: F401
        except Exception:
            return None
    return proxy


def make_client() -> httpx.AsyncClient:
    """Create an httpx async client with proxy and timeout config."""
    kwargs: dict = dict(
        timeout=httpx.Timeout(
            connect=CONNECT_TIMEOUT,
            read=READ_TIMEOUT,
            write=READ_TIMEOUT,
            pool=READ_TIMEOUT,
        ),
        follow_redirects=True,
        trust_env=False,
        headers={"User-Agent": "NeuroSMM/1.0", "Accept": "application/json, */*"},
    )
    proxy = _resolved_proxy_url()
    if proxy:
        kwargs["proxy"] = proxy
    return httpx.AsyncClient(**kwargs)


# ---------------------------------------------------------------------------
# Pexels provider (PRIMARY)
# ---------------------------------------------------------------------------
async def search_pexels(
    client: httpx.AsyncClient,
    query: str,
    used_fps: set[str],
) -> list[RawCandidate]:
    """Search Pexels for image candidates."""
    if not PEXELS_API_KEY or not _provider_available("pexels"):
        return []
    try:
        r = await client.get(
            "https://api.pexels.com/v1/search",
            params={"query": query, "per_page": PER_PAGE, "orientation": "landscape"},
            headers={"Authorization": PEXELS_API_KEY},
        )
        r.raise_for_status()
        data = r.json()
        _provider_success("pexels")
    except httpx.TimeoutException as e:
        logger.warning("pexels timeout query=%r err=%s", query, e)
        _provider_failure("pexels")
        return []
    except httpx.HTTPStatusError as e:
        logger.warning("pexels HTTP %s query=%r", e.response.status_code, query)
        _provider_failure("pexels")
        return []
    except (httpx.ConnectError, OSError) as e:
        logger.warning("pexels network error query=%r err=%s", query, e)
        _provider_failure("pexels")
        return []
    except Exception as e:
        logger.error("pexels unexpected error query=%r err=%s", query, e, exc_info=True)
        _provider_failure("pexels")
        return []

    candidates: list[RawCandidate] = []
    for item in data.get("photos") or []:
        meta = " ".join([
            str(item.get("alt") or ""),
            str(item.get("photographer") or ""),
        ])
        for key in ("large2x", "large", "original", "medium"):
            url = (item.get("src") or {}).get(key) or ""
            if _url_ok(url) and _not_used(url, used_fps):
                candidates.append(RawCandidate(
                    url=url.strip(),
                    meta_text=meta,
                    provider="pexels",
                    query=query,
                    width=item.get("width", 0),
                    height=item.get("height", 0),
                ))
                break  # One URL per photo item

    return candidates


# ---------------------------------------------------------------------------
# Pixabay provider (SECONDARY)
# ---------------------------------------------------------------------------
async def search_pixabay(
    client: httpx.AsyncClient,
    query: str,
    used_fps: set[str],
) -> list[RawCandidate]:
    """Search Pixabay for image candidates."""
    if not PIXABAY_API_KEY or not _provider_available("pixabay"):
        return []
    try:
        r = await client.get(
            "https://pixabay.com/api/",
            params={
                "key": PIXABAY_API_KEY,
                "q": query,
                "image_type": "photo",
                "orientation": "horizontal",
                "per_page": PER_PAGE,
                "safesearch": "true",
                "min_width": 800,
            },
        )
        r.raise_for_status()
        data = r.json()
        _provider_success("pixabay")
    except httpx.TimeoutException as e:
        logger.warning("pixabay timeout query=%r err=%s", query, e)
        _provider_failure("pixabay")
        return []
    except httpx.HTTPStatusError as e:
        logger.warning("pixabay HTTP %s query=%r", e.response.status_code, query)
        _provider_failure("pixabay")
        return []
    except (httpx.ConnectError, OSError) as e:
        logger.warning("pixabay network error query=%r err=%s", query, e)
        _provider_failure("pixabay")
        return []
    except Exception as e:
        logger.error("pixabay unexpected error query=%r err=%s", query, e, exc_info=True)
        _provider_failure("pixabay")
        return []

    candidates: list[RawCandidate] = []
    for item in data.get("hits") or []:
        meta = " ".join([
            str(item.get("tags") or ""),
            str(item.get("user") or ""),
            str(item.get("type") or ""),
        ])
        url = str(item.get("largeImageURL") or item.get("webformatURL") or "").strip()
        if url and _url_ok(url) and _not_used(url, used_fps):
            candidates.append(RawCandidate(
                url=url,
                meta_text=meta,
                provider="pixabay",
                query=query,
                width=item.get("imageWidth", 0),
                height=item.get("imageHeight", 0),
            ))

    return candidates


# ---------------------------------------------------------------------------
# Openverse provider (EDITOR-ONLY weak fallback)
# ---------------------------------------------------------------------------
async def search_openverse(
    client: httpx.AsyncClient,
    query: str,
    used_fps: set[str],
) -> list[RawCandidate]:
    """Search Openverse for image candidates.

    WARNING: This provider is EDITOR-ONLY. Do NOT use in autopost path.
    Openverse has less consistent quality and metadata coverage.
    """
    if not OPENVERSE_ENABLED or not _provider_available("openverse"):
        return []
    try:
        r = await client.get(
            "https://api.openverse.org/v1/images/",
            params={
                "q": query,
                "page_size": min(PER_PAGE, 20),
                "mature": "false",
            },
            headers={"Accept": "application/json"},
        )
        r.raise_for_status()
        data = r.json()
        _provider_success("openverse")
    except httpx.TimeoutException as e:
        logger.warning("openverse timeout query=%r err=%s", query, e)
        _provider_failure("openverse")
        return []
    except httpx.HTTPStatusError as e:
        logger.warning("openverse HTTP %s query=%r", e.response.status_code, query)
        _provider_failure("openverse")
        return []
    except (httpx.ConnectError, OSError) as e:
        logger.warning("openverse network error query=%r err=%s", query, e)
        _provider_failure("openverse")
        return []
    except Exception as e:
        logger.error("openverse unexpected error query=%r err=%s", query, e, exc_info=True)
        _provider_failure("openverse")
        return []

    candidates: list[RawCandidate] = []
    for item in data.get("results") or []:
        meta_parts = [
            str(item.get("title") or ""),
            str(item.get("description") or ""),
        ]
        tags = item.get("tags") or []
        if isinstance(tags, list):
            meta_parts.extend(
                str(t.get("name") or t) if isinstance(t, dict) else str(t)
                for t in tags[:10]
            )
        meta = " ".join(meta_parts)
        url = str(item.get("url") or "").strip()
        if url and _url_ok(url) and _not_used(url, used_fps):
            candidates.append(RawCandidate(
                url=url,
                meta_text=meta,
                provider="openverse",
                query=query,
                width=item.get("width", 0),
                height=item.get("height", 0),
            ))

    return candidates


# ---------------------------------------------------------------------------
# Collect candidates from all providers
# ---------------------------------------------------------------------------
async def collect_candidates(
    *,
    queries: list[str],
    used_refs: set[str] | None = None,
    mode: str = "autopost",
) -> list[RawCandidate]:
    """Collect raw candidates from all enabled providers.

    Provider order:
      1. Pexels (primary) — always queried
      2. Pixabay (secondary) — always queried
      3. Openverse (editor-only) — ONLY in editor mode + feature flag

    Returns all candidates unsorted. Scoring/ranking is done by image_ranker.
    """
    used_fps = prepare_used_refs(used_refs)
    all_candidates: list[RawCandidate] = []

    providers: list[tuple[str, object]] = [
        ("pexels", search_pexels),
        ("pixabay", search_pixabay),
    ]

    # Openverse only in editor mode
    is_editor = mode == "editor"
    if is_editor and OPENVERSE_ENABLED:
        providers.append(("openverse", search_openverse))

    async with make_client() as client:
        for provider_name, search_fn in providers:
            for q in queries:
                try:
                    results = await search_fn(client, q, used_fps)
                    all_candidates.extend(results)
                except Exception as exc:
                    logger.error(
                        "Provider %s query=%r failed: %s",
                        provider_name, q[:40], exc,
                    )

    logger.info(
        "PROVIDERS_COLLECTED total=%d providers=%s queries=%d mode=%s",
        len(all_candidates),
        [p[0] for p in providers],
        len(queries),
        mode,
    )
    return all_candidates
