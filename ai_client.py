from __future__ import annotations

import asyncio
import base64
import os
import logging
from typing import Optional

import httpx
from openai import AsyncOpenAI
from openai import APITimeoutError, APIConnectionError, RateLimitError, APIError

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_IMAGE_SIZE = os.getenv("IMAGE_GENERATION_SIZE", "1024x1024")
DEFAULT_CHAT_MAX_TOKENS = int(os.getenv("OPENROUTER_MAX_TOKENS", "700"))

logger = logging.getLogger(__name__)

# Module-level cache: (api_key, base_url) → AsyncOpenAI client
# Each client owns a persistent httpx connection pool — avoids recreating
# TCP connections on every AI call.
_clients: dict[tuple[str, str], AsyncOpenAI] = {}
_clients_lock = asyncio.Lock()


def _resolve_base_url(base_url: Optional[str] = None) -> str:
    return (base_url or os.getenv("OPENAI_BASE_URL") or DEFAULT_BASE_URL).strip()


def _endpoint_kind_for_image(model: str, resolved_base_url: str) -> str:
    base = (resolved_base_url or "").lower()
    m = (model or "").strip().lower()
    if "openrouter.ai" in base:
        return "openrouter.images.generate"
    if "api.openai.com" in base:
        return "openai.images.generate"
    if m.startswith("openai/"):
        return "openai-compatible.images.generate"
    return "unknown.images.generate"


def _should_fast_skip_image_call(model: str, resolved_base_url: str) -> tuple[bool, str]:
    """Return (skip, reason) for provider/model combos that are known bad at runtime."""
    base = (resolved_base_url or "").lower()
    m = (model or "").strip().lower()
    if "openrouter.ai" in base and (m == "gpt-image-1" or m.startswith("openai/gpt-image-1")):
        return True, "provider_endpoint_mismatch_openrouter_gpt_image_1"
    return False, ""


def _error_prefix(value: object, limit: int = 160) -> str:
    try:
        txt = str(value or "").strip().replace("\n", " ")
    except Exception:
        txt = ""
    return txt[:limit]


def _resolve_proxy() -> str | None:
    for key in (
        "OPENROUTER_PROXY_URL",
        "LLM_PROXY_URL",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "ALL_PROXY",
    ):
        value = (os.getenv(key) or "").strip()
        if value:
            return value
    return None


def _clamp_max_tokens(value: Optional[int]) -> int:
    if value is None:
        return DEFAULT_CHAT_MAX_TOKENS
    try:
        n = int(value)
    except Exception:
        return DEFAULT_CHAT_MAX_TOKENS
    if n < 32:
        return 32
    if n > 1200:
        return 1200
    return n


def _make_async_http_client(proxy: str | None = None) -> httpx.AsyncClient:
    """Создаёт async httpx-клиент для OpenAI SDK."""
    timeout = httpx.Timeout(connect=20.0, read=90.0, write=60.0, pool=60.0)
    transport = httpx.AsyncHTTPTransport(retries=1)
    kwargs: dict = dict(timeout=timeout, trust_env=False, transport=transport)
    if proxy:
        kwargs["proxy"] = proxy
    return httpx.AsyncClient(**kwargs)


def make_async_client(api_key: str, base_url: Optional[str] = None) -> AsyncOpenAI:
    """Создаёт AsyncOpenAI клиент. Caller отвечает за aclose()."""
    base_url = _resolve_base_url(base_url)
    proxy = _resolve_proxy()
    http_client = _make_async_http_client(proxy)
    return AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
        http_client=http_client,
    )


# Backward-compat alias (синхронный клиент больше не нужен нигде, но если
# где-то остался импорт make_client — не падаем)
def make_client(api_key: str, base_url: Optional[str] = None) -> AsyncOpenAI:  # type: ignore[return]
    logger.warning("make_client() is deprecated; use make_async_client()")
    return make_async_client(api_key, base_url=base_url)


async def _get_shared_client(api_key: str, base_url: Optional[str] = None) -> AsyncOpenAI:
    """Return a cached AsyncOpenAI client for the given (api_key, base_url) pair.

    The client owns a persistent httpx connection pool that is reused across
    calls, avoiding TCP handshake overhead on every AI request.
    Caller must NOT call ``client.close()`` — the client is shared.
    """
    resolved_url = _resolve_base_url(base_url)
    cache_key = (api_key, resolved_url)
    async with _clients_lock:
        if cache_key not in _clients:
            http_client = _make_async_http_client(_resolve_proxy())
            _clients[cache_key] = AsyncOpenAI(
                base_url=resolved_url,
                api_key=api_key,
                http_client=http_client,
            )
        return _clients[cache_key]


async def close_all_clients() -> None:
    """Close all cached OpenAI clients. Call once on application shutdown."""
    for client in list(_clients.values()):
        try:
            await client.close()
        except Exception:
            pass
    _clients.clear()


async def ai_chat(
    api_key: str,
    model: str,
    messages: list[dict],
    temperature: float = 0.7,
    base_url: Optional[str] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """Асинхронный вызов chat/completions с retry и exponential backoff."""
    safe_max_tokens = _clamp_max_tokens(max_tokens)
    client = await _get_shared_client(api_key, base_url)

    max_retries = 3
    last_exc: Exception | None = None

    for attempt in range(max_retries):
        try:
            r = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=safe_max_tokens,
            )
            return (r.choices[0].message.content or "").strip()
        except (APITimeoutError, APIConnectionError) as exc:
            last_exc = exc
            delay = 2 ** attempt  # 1s, 2s, 4s
            logger.warning(
                "ai_chat transient error (attempt %d/%d) model=%s err=%s — retrying in %ds",
                attempt + 1, max_retries, model, exc, delay,
            )
            await asyncio.sleep(delay)
        except RateLimitError as exc:
            last_exc = exc
            delay = 2 ** attempt + 2  # 3s, 4s, 6s — longer for rate limits
            logger.warning(
                "ai_chat rate limit (attempt %d/%d) model=%s err=%s — retrying in %ds",
                attempt + 1, max_retries, model, exc, delay,
            )
            await asyncio.sleep(delay)
        except APIError as exc:
            # Non-transient API errors (e.g. bad request) — don't retry
            logger.warning("ai_chat api error model=%s err=%s", model, exc)
            return ""
        except Exception as exc:
            logger.exception("ai_chat unexpected error model=%s err=%s", model, exc)
            return ""

    logger.warning("ai_chat exhausted %d retries model=%s last_err=%s", max_retries, model, last_exc)
    return ""


async def whisper_transcribe(
    api_key: str,
    audio_bytes: bytes,
    filename: str = "voice.ogg",
    *,
    language: Optional[str] = "ru",
) -> str:
    """Transcribe audio bytes using OpenAI Whisper. Returns transcribed text or empty string on failure."""
    if not api_key or not audio_bytes:
        return ""
    # Whisper is only available on OpenAI's API, not OpenRouter
    openai_base = "https://api.openai.com/v1"
    client = await _get_shared_client(api_key, openai_base)
    try:
        transcript = await client.audio.transcriptions.create(
            model="whisper-1",
            file=(filename, audio_bytes),
            language=language,
        )
        return (transcript.text or "").strip()
    except RateLimitError as exc:
        logger.warning("whisper_transcribe rate limit err=%s", exc)
        return ""
    except APITimeoutError as exc:
        logger.warning("whisper_transcribe timeout err=%s", exc)
        return ""
    except APIConnectionError as exc:
        logger.warning("whisper_transcribe connection error err=%s", exc)
        return ""
    except APIError as exc:
        logger.warning("whisper_transcribe api error err=%s", exc)
        return ""
    except Exception as exc:
        logger.exception("whisper_transcribe unexpected error err=%s", exc)
        return ""


async def ai_image_generate(
    api_key: str,
    model: str,
    prompt: str,
    *,
    base_url: Optional[str] = None,
    size: str = DEFAULT_IMAGE_SIZE,
    quality: Optional[str] = None,
    background: Optional[str] = None,
    extra_body: Optional[dict] = None,
) -> bytes | None:
    """Generate an image via the OpenAI-compatible images.generate API.

    Requests b64_json response to avoid a URL round-trip when possible.
    Falls back to downloading the URL if the provider returns a URL instead.

    Returns raw image bytes on success, None on any failure.
    Every failure is logged with a structured reason.
    """
    if not api_key or not model or not prompt:
        logger.warning(
            "ai_image_generate skip: api_key=%s model=%s prompt_len=%d",
            bool(api_key), bool(model), len(prompt or ""),
        )
        return None

    resolved_base_url = _resolve_base_url(base_url)
    normalized_model = (model or "").strip()
    if "api.openai.com" in resolved_base_url.lower() and normalized_model.lower().startswith("openai/"):
        normalized_model = normalized_model.split("/", 1)[1]

    endpoint_kind = _endpoint_kind_for_image(normalized_model, resolved_base_url)
    should_skip, skip_reason = _should_fast_skip_image_call(normalized_model, resolved_base_url)
    if should_skip:
        logger.warning(
            "ai_image_generate SKIP reason=%s base_url=%s model=%s endpoint_kind=%s",
            skip_reason,
            resolved_base_url,
            normalized_model,
            endpoint_kind,
        )
        return None

    client = await _get_shared_client(api_key, resolved_base_url)
    try:
        kwargs: dict = {
            "model": normalized_model,
            "prompt": prompt,
            "size": size,
            "response_format": "b64_json",  # Prefer inline to avoid URL fetch
        }
        if quality:
            kwargs["quality"] = quality
        if background:
            kwargs["background"] = background
        if extra_body:
            kwargs["extra_body"] = extra_body

        logger.info(
            "ai_image_generate START base_url=%s model=%s endpoint_kind=%s size=%s prompt_len=%d",
            resolved_base_url, normalized_model, endpoint_kind, size, len(prompt),
        )

        result = await client.images.generate(**kwargs)
        data = getattr(result, "data", None) or []
        if not data:
            logger.warning("ai_image_generate EMPTY_RESPONSE model=%s", normalized_model)
            return None

        item = data[0]

        # Prefer b64_json (no extra network hop)
        b64 = getattr(item, "b64_json", None)
        if b64:
            image_bytes = base64.b64decode(b64)
            logger.info(
                "ai_image_generate SUCCESS model=%s delivery=b64_json bytes=%d",
                normalized_model, len(image_bytes),
            )
            return image_bytes

        # Fallback: download from URL
        url = getattr(item, "url", None)
        if not url:
            logger.warning("ai_image_generate NO_B64_NO_URL model=%s", normalized_model)
            return None

        logger.info("ai_image_generate FETCHING_URL model=%s url=%r", normalized_model, url[:120])
        proxy = _resolve_proxy()
        fetch_timeout = httpx.Timeout(connect=20.0, read=120.0, write=60.0, pool=60.0)
        transport = httpx.AsyncHTTPTransport(retries=1)
        fetch_kwargs: dict = dict(
            timeout=fetch_timeout,
            trust_env=False,
            transport=transport,
            follow_redirects=True,
        )
        if proxy:
            fetch_kwargs["proxy"] = proxy
        async with httpx.AsyncClient(**fetch_kwargs) as http_client:
            resp = await http_client.get(url)
            resp.raise_for_status()
            image_bytes = resp.content
            logger.info(
                "ai_image_generate SUCCESS model=%s delivery=url bytes=%d",
                normalized_model, len(image_bytes),
            )
            return image_bytes

    except (APITimeoutError, RateLimitError, APIConnectionError) as exc:
        logger.warning(
            "ai_image_generate TRANSIENT_ERROR model=%s error_type=%s error=%s",
            normalized_model, type(exc).__name__, str(exc)[:200],
        )
        return None
    except APIError as exc:
        body_prefix = _error_prefix(getattr(exc, "body", None) or exc)
        logger.warning(
            "ai_image_generate API_ERROR base_url=%s model=%s endpoint_kind=%s status=%s response_prefix=%r error=%s",
            resolved_base_url,
            normalized_model,
            endpoint_kind,
            getattr(exc, "status_code", "?"),
            body_prefix,
            str(exc)[:200],
        )
        return None
    except Exception:
        logger.exception("ai_image_generate UNEXPECTED_ERROR model=%s", normalized_model)
        return None
