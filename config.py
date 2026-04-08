from __future__ import annotations

import functools
import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse
from typing import Any

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _env_str(*names: str, default: str = "") -> str:
    for name in names:
        raw = os.getenv(name)
        if raw is not None:
            value = str(raw).strip()
            if value:
                return value
    return default


def _env_int(*names: str, default: int) -> int:
    for name in names:
        raw = (os.getenv(name) or "").strip()
        if not raw:
            continue
        try:
            return int(raw)
        except Exception:
            continue
    return default


def _env_float(*names: str, default: float) -> float:
    for name in names:
        raw = (os.getenv(name) or "").strip()
        if not raw:
            continue
        try:
            return float(raw)
        except Exception:
            continue
    return default


def _env_bool(*names: str, default: bool = False) -> bool:
    for name in names:
        raw = os.getenv(name)
        if raw is None:
            continue
        value = str(raw).strip().lower()
        if not value:
            continue
        if value in {"1", "true", "yes", "on"}:
            return True
        if value in {"0", "false", "no", "off"}:
            return False
    return default


def _parse_admin_ids(s: str) -> set[int]:
    result: set[int] = set()
    for part in str(s or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            result.add(int(part))
        except ValueError:
            pass
    return result


def _parse_csv(*names: str) -> list[str]:
    value = _env_str(*names, default="")
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _normalize_origin(url: str) -> str:
    raw = (url or "").strip().rstrip("/")
    if not raw:
        return ""
    parsed = urlparse(raw)
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}"
    return ""


def _collect_allowed_origins(miniapp_url: str) -> tuple[str, ...]:
    origins: list[str] = []
    public_origin = _normalize_origin(_env_str("MINIAPP_PUBLIC_ORIGIN", default=miniapp_url))
    env_origins = _parse_csv("MINIAPP_ALLOWED_ORIGINS")
    defaults = [
        "https://web.telegram.org",
        "https://web.telegram.org.a",
        "https://web.telegram.org.k",
        "https://web.telegram.org.z",
        "https://tg.dev",
        public_origin,
    ]
    seen = set()
    for item in [*defaults, *env_origins]:
        origin = _normalize_origin(item)
        if origin and origin not in seen:
            seen.add(origin)
            origins.append(origin)
    return tuple(origins)


def _collect_trusted_hosts(miniapp_url: str) -> tuple[str, ...]:
    hosts = ["localhost", "127.0.0.1"]
    raw_hosts = _parse_csv("MINIAPP_TRUSTED_HOSTS")
    if raw_hosts:
        hosts.extend(raw_hosts)
    public = urlparse((miniapp_url or "").strip())
    if public.netloc:
        hosts.append(public.netloc)
        hosts.append(public.hostname or "")
    out = []
    seen = set()
    for host in hosts or ["*"]:
        h = (host or "").strip()
        if h and h not in seen:
            seen.add(h)
            out.append(h)
    return tuple(out)


@dataclass
class Config:
    bot_token: str
    tz: str

    llm_api_key: str
    llm_base_url: str
    llm_model: str
    llm_provider: str = "openrouter"
    llm_fallback_models: list[str] = field(default_factory=list)
    llm_request_cooldown_seconds: float = 1.0
    llm_max_tokens: int = 700

    admin_ids: set[int] = field(default_factory=set)
    miniapp_url: str = ""
    miniapp_auth_max_age: int = 3600
    max_channels_per_user: int = 10

    upload_image_limit_mb: int = 15
    upload_video_limit_mb: int = 64
    upload_document_limit_mb: int = 32
    max_upload_mb: int = 200
    temp_media_quota_mb_per_user: int = 512

    max_active_drafts_per_user: int = 15
    telegram_media_cache_minutes: int = 120

    generated_media_retention_hours: int = 48
    uploaded_photo_retention_hours: int = 72
    uploaded_video_retention_hours: int = 24
    uploaded_document_retention_hours: int = 48

    bot_username: str = ""
    telegram_proxy_url: str = ""
    storage_chat_id: int = 0

    app_env: str = "production"
    miniapp_public_origin: str = ""
    allowed_cors_origins: tuple[str, ...] = ()
    trusted_hosts: tuple[str, ...] = ()
    enable_docs: bool = False
    allow_media_query_auth: bool = True
    enforce_origin_check: bool = True
    api_rate_limit_rpm: int = 240
    api_write_rate_limit_rpm: int = 90

    hf_api_key: str = ""
    hf_image_model: str = ""
    pexels_api_key: str = ""
    pixabay_api_key: str = ""
    unsplash_access_key: str = ""
    unsplash_app_name: str = ""

    # Web auth (standalone web mode alongside Telegram Mini App)
    web_auth_secret: str = ""       # JWT signing secret; auto-generated if empty
    web_auth_enabled: bool = False  # Enable web login (Telegram Login Widget + JWT)
    web_auth_token_ttl: int = 86400 # JWT lifetime in seconds (default 24 h)

    # Billing
    payment_mode: str = "test"      # "test" or "production"
    yoo_shop_id: str = ""
    yoo_secret_key: str = ""
    yoo_return_url: str = ""
    yoo_receipt_enabled: bool = False   # 54-FZ receipt support
    yoo_vat_code: int = 1              # VAT code for receipts (1=20%, 4=no VAT)
    stars_pro_price: int = 99    # Telegram Stars price for Pro
    stars_max_price: int = 249   # Telegram Stars price for Max

    def __getattr__(self, name: str) -> Any:
        alias_map = {
            "openrouter_api_key": self.llm_api_key,
            "openrouter_base_url": self.llm_base_url,
            "openrouter_model": self.llm_model,
            "openai_api_key": self.llm_api_key,
            "openai_base_url": self.llm_base_url,
            "openai_model": self.llm_model,
            "bot_tz": self.tz,
            "bot_proxy_url": self.telegram_proxy_url,
            "proxy_url": self.telegram_proxy_url,
            "max_upload_size_mb": self.max_upload_mb,
        }
        if name in alias_map:
            return alias_map[name]
        raise AttributeError(f"Config has no attribute {name!r}")


@functools.lru_cache(maxsize=1)
def load_config() -> Config:
    bot_token = _env_str("BOT_TOKEN")
    if not bot_token:
        raise RuntimeError("BOT_TOKEN is missing in .env")

    tz = _env_str("BOT_TZ", "TZ", default="Europe/Moscow")

    llm_api_key = _env_str("LLM_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY")
    llm_base_url = _env_str(
        "LLM_BASE_URL",
        "OPENAI_BASE_URL",
        "OPENROUTER_BASE_URL",
        default="https://openrouter.ai/api/v1",
    )
    llm_model = _env_str(
        "LLM_MODEL",
        "OPENROUTER_MODEL",
        "OPENAI_MODEL",
        default="openai/gpt-4o-mini",
    )
    llm_provider = _env_str("LLM_PROVIDER", default="openrouter").lower()
    llm_fallback_models = _parse_csv("LLM_FALLBACK_MODELS")
    llm_request_cooldown_seconds = max(0.0, _env_float("LLM_REQUEST_COOLDOWN_SECONDS", default=1.0))
    llm_max_tokens = max(64, _env_int("LLM_MAX_TOKENS", "OPENROUTER_MAX_TOKENS", default=700))

    admin_ids = _parse_admin_ids(os.getenv("ADMIN_IDS", ""))
    miniapp_url = _env_str("MINIAPP_URL")
    miniapp_auth_max_age = max(300, _env_int("MINIAPP_AUTH_MAX_AGE", default=3600))
    max_channels_per_user = max(1, _env_int("MAX_CHANNELS_PER_USER", default=10))

    upload_image_limit_mb = max(1, _env_int("UPLOAD_IMAGE_LIMIT_MB", default=15))
    upload_video_limit_mb = max(5, _env_int("UPLOAD_VIDEO_LIMIT_MB", default=64))
    upload_document_limit_mb = max(5, _env_int("UPLOAD_DOCUMENT_LIMIT_MB", default=32))
    max_upload_mb = max(
        5,
        _env_int(
            "MAX_UPLOAD_MB",
            default=max(upload_image_limit_mb, upload_video_limit_mb, upload_document_limit_mb),
        ),
    )
    temp_media_quota_mb_per_user = max(32, _env_int("TEMP_MEDIA_QUOTA_MB_PER_USER", default=512))

    max_active_drafts_per_user = max(1, _env_int("MAX_ACTIVE_DRAFTS_PER_USER", default=15))
    telegram_media_cache_minutes = max(1, _env_int("TELEGRAM_MEDIA_CACHE_MINUTES", default=120))

    generated_media_retention_hours = max(6, _env_int("GENERATED_MEDIA_RETENTION_HOURS", default=48))
    uploaded_photo_retention_hours = max(6, _env_int("UPLOADED_PHOTO_RETENTION_HOURS", default=72))
    uploaded_video_retention_hours = max(6, _env_int("UPLOADED_VIDEO_RETENTION_HOURS", default=24))
    uploaded_document_retention_hours = max(6, _env_int("UPLOADED_DOCUMENT_RETENTION_HOURS", default=48))

    bot_username = _env_str("BOT_USERNAME").lstrip("@")
    telegram_proxy_url = _env_str("TELEGRAM_PROXY_URL", "BOT_PROXY_URL")
    storage_chat_id = _env_int("STORAGE_CHAT_ID", default=0)

    app_env = _env_str("APP_ENV", "ENV", "ENVIRONMENT", default="production").lower()
    enable_docs = _env_bool("ENABLE_DOCS", "MINIAPP_ENABLE_DOCS", default=app_env not in {"prod", "production"})
    miniapp_public_origin = _normalize_origin(_env_str("MINIAPP_PUBLIC_ORIGIN", default=miniapp_url))
    allowed_cors_origins = _collect_allowed_origins(miniapp_url)
    trusted_hosts = _collect_trusted_hosts(miniapp_url)
    allow_media_query_auth = _env_bool("MINIAPP_ALLOW_MEDIA_QUERY_AUTH", default=True)
    enforce_origin_check = _env_bool("MINIAPP_ENFORCE_ORIGIN_CHECK", default=True)
    api_rate_limit_rpm = max(30, _env_int("API_RATE_LIMIT_RPM", default=240))
    api_write_rate_limit_rpm = max(20, _env_int("API_WRITE_RATE_LIMIT_RPM", default=90))

    hf_api_key = _env_str("HF_API_KEY")
    hf_image_model = _env_str("HF_IMAGE_MODEL")
    pexels_api_key = _env_str("PEXELS_API_KEY")
    pixabay_api_key = _env_str("PIXABAY_API_KEY")
    unsplash_access_key = _env_str("UNSPLASH_ACCESS_KEY")
    unsplash_app_name = _env_str("UNSPLASH_APP_NAME")

    payment_mode = _env_str("PAYMENT_MODE", default="test").lower()
    if payment_mode not in ("test", "production"):
        payment_mode = "test"
    yoo_shop_id = _env_str("YOO_SHOP_ID")
    yoo_secret_key = _env_str("YOO_SECRET_KEY")
    yoo_return_url = _env_str("YOO_RETURN_URL", default="")
    yoo_receipt_enabled = _env_bool("YOO_RECEIPT_ENABLED", default=False)
    yoo_vat_code = max(1, min(6, _env_int("YOO_VAT_CODE", default=1)))
    stars_pro_price = max(1, _env_int("STARS_PRO_PRICE", default=99))
    stars_max_price = max(1, _env_int("STARS_MAX_PRICE", default=249))

    web_auth_enabled = _env_bool("WEB_AUTH_ENABLED", default=False)
    web_auth_secret = _env_str("WEB_AUTH_SECRET")
    if web_auth_enabled and not web_auth_secret:
        if app_env in {"prod", "production"}:
            raise RuntimeError(
                "WEB_AUTH_ENABLED=true but WEB_AUTH_SECRET is not set. "
                "Set a strong random secret (e.g. `python -c \"import secrets; print(secrets.token_hex(32))\"`). "
                "Auto-generation is not allowed in production."
            )
        # Dev/test: auto-generate a transient secret (tokens won't survive restarts)
        import secrets as _secrets
        web_auth_secret = _secrets.token_hex(32)
    web_auth_token_ttl = max(300, _env_int("WEB_AUTH_TOKEN_TTL", default=86400))

    return Config(
        bot_token=bot_token,
        tz=tz,
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
        llm_model=llm_model,
        llm_provider=llm_provider,
        llm_fallback_models=llm_fallback_models,
        llm_request_cooldown_seconds=llm_request_cooldown_seconds,
        llm_max_tokens=llm_max_tokens,
        admin_ids=admin_ids,
        miniapp_url=miniapp_url,
        miniapp_auth_max_age=miniapp_auth_max_age,
        max_channels_per_user=max_channels_per_user,
        upload_image_limit_mb=upload_image_limit_mb,
        upload_video_limit_mb=upload_video_limit_mb,
        upload_document_limit_mb=upload_document_limit_mb,
        max_upload_mb=max_upload_mb,
        temp_media_quota_mb_per_user=temp_media_quota_mb_per_user,
        max_active_drafts_per_user=max_active_drafts_per_user,
        telegram_media_cache_minutes=telegram_media_cache_minutes,
        generated_media_retention_hours=generated_media_retention_hours,
        uploaded_photo_retention_hours=uploaded_photo_retention_hours,
        uploaded_video_retention_hours=uploaded_video_retention_hours,
        uploaded_document_retention_hours=uploaded_document_retention_hours,
        bot_username=bot_username,
        telegram_proxy_url=telegram_proxy_url,
        storage_chat_id=storage_chat_id,
        app_env=app_env,
        miniapp_public_origin=miniapp_public_origin,
        allowed_cors_origins=allowed_cors_origins,
        trusted_hosts=trusted_hosts,
        enable_docs=enable_docs,
        allow_media_query_auth=allow_media_query_auth,
        enforce_origin_check=enforce_origin_check,
        api_rate_limit_rpm=api_rate_limit_rpm,
        api_write_rate_limit_rpm=api_write_rate_limit_rpm,
        hf_api_key=hf_api_key,
        hf_image_model=hf_image_model,
        pexels_api_key=pexels_api_key,
        pixabay_api_key=pixabay_api_key,
        unsplash_access_key=unsplash_access_key,
        unsplash_app_name=unsplash_app_name,
        yoo_shop_id=yoo_shop_id,
        yoo_secret_key=yoo_secret_key,
        yoo_return_url=yoo_return_url,
        yoo_receipt_enabled=yoo_receipt_enabled,
        yoo_vat_code=yoo_vat_code,
        payment_mode=payment_mode,
        stars_pro_price=stars_pro_price,
        stars_max_price=stars_max_price,
        web_auth_secret=web_auth_secret,
        web_auth_enabled=web_auth_enabled,
        web_auth_token_ttl=web_auth_token_ttl,
    )
