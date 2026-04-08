from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import time
from typing import Any
from urllib.parse import parse_qsl, urlparse

from fastapi import Header, HTTPException, Request

from config import load_config

logger = logging.getLogger(__name__)

_MEDIA_QUERY_AUTH_PATHS = {"/api/media/telegram"}
# These path prefixes also accept Telegram init data via the `tgWebAppData`
# query parameter so that <img src="/uploads/...?tgWebAppData=..."> works in
# the browser without requiring custom request headers.
_MEDIA_QUERY_AUTH_PREFIXES = ("/uploads/", "/generated-images/")
_HEADER_AUTH_SCHEMES = ("tma ",)
_ALLOWED_EMBED_ORIGINS = {
    "https://web.telegram.org",
    "https://web.telegram.org.a",
    "https://web.telegram.org.k",
    "https://web.telegram.org.z",
    "https://tg.dev",
}


# ---------------------------------------------------------------------------
# JWT helpers (HS256 — no external dependency)
# ---------------------------------------------------------------------------

def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(s: str) -> bytes:
    padded = s + "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(padded)


def create_web_jwt(payload: dict[str, Any]) -> str:
    """Create an HS256 JWT signed with *web_auth_secret*."""
    cfg = load_config()
    secret = cfg.web_auth_secret
    ttl = cfg.web_auth_token_ttl
    now = int(time.time())
    full_payload = {**payload, "iat": now, "exp": now + ttl}
    header_b64 = _b64url_encode(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
    payload_b64 = _b64url_encode(json.dumps(full_payload, separators=(",", ":")).encode())
    signing_input = f"{header_b64}.{payload_b64}"
    sig = hmac.new(secret.encode("utf-8"), signing_input.encode("utf-8"), hashlib.sha256).digest()
    return f"{signing_input}.{_b64url_encode(sig)}"


def verify_web_jwt(token: str) -> dict[str, Any]:
    """Verify an HS256 JWT and return the decoded payload.

    Raises ``HTTPException(401)`` on any verification failure.
    """
    cfg = load_config()
    secret = cfg.web_auth_secret
    parts = token.split(".")
    if len(parts) != 3:
        raise HTTPException(status_code=401, detail="Invalid web token")
    header_b64, payload_b64, sig_b64 = parts
    signing_input = f"{header_b64}.{payload_b64}"
    expected_sig = hmac.new(secret.encode("utf-8"), signing_input.encode("utf-8"), hashlib.sha256).digest()
    try:
        actual_sig = _b64url_decode(sig_b64)
    except Exception as exc:
        raise HTTPException(status_code=401, detail="Invalid web token") from exc
    if not hmac.compare_digest(expected_sig, actual_sig):
        raise HTTPException(status_code=401, detail="Invalid web token signature")
    try:
        payload = json.loads(_b64url_decode(payload_b64))
    except Exception as exc:
        raise HTTPException(status_code=401, detail="Invalid web token payload") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=401, detail="Invalid web token payload")
    if payload.get("exp", 0) < time.time():
        raise HTTPException(status_code=401, detail="Web token expired")
    return payload


# ---------------------------------------------------------------------------
# Telegram Login Widget verification
# (different from Mini App initData — uses SHA256(bot_token) as secret)
# ---------------------------------------------------------------------------

def verify_telegram_login_widget(data: dict[str, Any]) -> dict[str, Any]:
    """Verify data returned by the Telegram Login Widget.

    *data* must contain at minimum ``id``, ``auth_date``, and ``hash``.
    Returns a normalised user dict ``{"id": int, "raw_user": dict, "auth_date": int}``.
    Raises ``HTTPException(401)`` on failure.
    """
    cfg = load_config()
    received_hash = str(data.get("hash") or "")
    if not received_hash:
        raise HTTPException(status_code=401, detail="Missing Telegram login hash")

    # Build check string from all fields except "hash"
    check_fields = {str(k): str(v) for k, v in data.items() if k != "hash" and v is not None}
    check_string = "\n".join(f"{k}={check_fields[k]}" for k in sorted(check_fields))

    secret_key = hashlib.sha256(cfg.bot_token.encode("utf-8")).digest()
    calculated_hash = hmac.new(secret_key, check_string.encode("utf-8"), hashlib.sha256).hexdigest()

    if not hmac.compare_digest(calculated_hash, received_hash):
        raise HTTPException(status_code=401, detail="Invalid Telegram login data")

    try:
        auth_date = int(data.get("auth_date", 0))
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=401, detail="Invalid auth_date") from exc

    max_age = max(300, int(getattr(cfg, "miniapp_auth_max_age", 3600) or 3600))
    now_ts = int(time.time())
    if auth_date <= 0 or now_ts - auth_date > max_age:
        raise HTTPException(status_code=401, detail="Telegram login data expired")

    user_id = data.get("id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Missing Telegram user id")

    raw_user = {
        "id": int(user_id),
        "first_name": str(data.get("first_name", "")),
        "last_name": str(data.get("last_name", "")),
        "username": str(data.get("username", "")),
        "photo_url": str(data.get("photo_url", "")),
    }
    return {"id": int(user_id), "raw_user": raw_user, "auth_date": auth_date}


def _build_data_check_string(data: dict[str, str]) -> str:
    return "\n".join(f"{key}={data[key]}" for key in sorted(data.keys()) if key != "hash")


def verify_telegram_webapp_init_data(init_data: str) -> dict[str, Any]:
    if not init_data:
        raise HTTPException(status_code=401, detail="Missing Telegram init data")

    parsed = dict(parse_qsl(init_data, keep_blank_values=True))
    received_hash = parsed.get("hash")
    if not received_hash:
        raise HTTPException(status_code=401, detail="Missing Telegram hash")

    cfg = load_config()
    secret_key = hmac.new(b"WebAppData", cfg.bot_token.encode("utf-8"), hashlib.sha256).digest()
    calculated_hash = hmac.new(secret_key, _build_data_check_string(parsed).encode("utf-8"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(calculated_hash, received_hash):
        raise HTTPException(status_code=401, detail="Invalid Telegram init data")

    try:
        auth_date = int(parsed.get("auth_date", "0"))
    except Exception as exc:
        raise HTTPException(status_code=401, detail="Invalid auth_date") from exc

    now_ts = int(time.time())
    max_age = max(300, int(getattr(cfg, "miniapp_auth_max_age", 3600) or 3600))
    if auth_date <= 0 or now_ts - auth_date > max_age or auth_date - now_ts > 300:
        raise HTTPException(status_code=401, detail="Telegram auth data expired")

    user_raw = parsed.get("user")
    if not user_raw:
        raise HTTPException(status_code=401, detail="Missing Telegram user")

    try:
        user = json.loads(user_raw)
    except Exception as exc:
        raise HTTPException(status_code=401, detail="Invalid Telegram user payload") from exc
    if not isinstance(user, dict):
        raise HTTPException(status_code=401, detail="Invalid Telegram user payload")

    user_id = user.get("id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Missing Telegram user id")

    return {"id": int(user_id), "raw_user": user, "auth_date": auth_date}


def _extract_header_init_data(
    x_telegram_init_data: str | None,
    authorization: str | None,
    tg_web_app_data: str | None,
) -> str | None:
    if x_telegram_init_data and x_telegram_init_data.strip():
        return x_telegram_init_data.strip()
    if tg_web_app_data and tg_web_app_data.strip():
        return tg_web_app_data.strip()
    if authorization:
        value = authorization.strip()
        lowered = value.lower()
        for prefix in _HEADER_AUTH_SCHEMES:
            if lowered.startswith(prefix):
                return value[len(prefix):].strip()
    return None


def _allow_query_auth(request: Request) -> bool:
    cfg = load_config()
    if not bool(getattr(cfg, "allow_media_query_auth", True)):
        return False
    if request.method.upper() != "GET":
        return False
    path = request.url.path.rstrip("/") or "/"
    return path in _MEDIA_QUERY_AUTH_PATHS or any(path.startswith(p) for p in _MEDIA_QUERY_AUTH_PREFIXES)


def _normalize_origin(raw: str | None) -> str:
    value = (raw or "").strip()
    if not value:
        return ""
    parsed = urlparse(value)
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}".rstrip('/')
    return ""


def _validate_request_origin(request: Request) -> None:
    cfg = load_config()
    if not getattr(cfg, "enforce_origin_check", True):
        return
    if not request.url.path.startswith('/api/'):
        return
    allowed = set(getattr(cfg, 'allowed_cors_origins', ()) or ()) | set(_ALLOWED_EMBED_ORIGINS)
    origin = _normalize_origin(request.headers.get('origin'))
    referer = _normalize_origin(request.headers.get('referer'))
    candidate = origin or referer
    if not candidate:
        return
    if candidate not in allowed:
        raise HTTPException(status_code=403, detail='Blocked request origin')


def _extract_init_data_from_request(
    request: Request,
    x_telegram_init_data: str | None,
    authorization: str | None,
    tg_web_app_data: str | None,
) -> str:
    # Note: origin validation is performed by the caller (get_current_telegram_user).
    header_value = _extract_header_init_data(x_telegram_init_data, authorization, tg_web_app_data)
    if header_value:
        return header_value
    if _allow_query_auth(request):
        qp = request.query_params.get("tgWebAppData") or request.query_params.get("init_data")
        if qp and qp.strip():
            return qp.strip()
    raise HTTPException(status_code=401, detail="Missing Telegram init data")


WEB_SESSION_COOKIE = "neurosmm_session"


async def get_current_telegram_user(
    request: Request,
    x_telegram_init_data: str | None = Header(default=None, alias="X-Telegram-Init-Data"),
    authorization: str | None = Header(default=None, alias="Authorization"),
    tg_web_app_data: str | None = Header(default=None, alias="TG-WebApp-Data"),
) -> dict[str, Any]:
    # Origin check applies to ALL auth paths (Telegram initData and JWT alike).
    _validate_request_origin(request)

    # --- Path 1: Telegram Mini App initData (existing flow) ----------------
    try:
        init_data = _extract_init_data_from_request(request, x_telegram_init_data, authorization, tg_web_app_data)
        return verify_telegram_webapp_init_data(init_data)
    except HTTPException:
        pass

    # --- Path 2: JWT web session via HttpOnly cookie -----------------------
    cfg = load_config()
    if cfg.web_auth_enabled:
        cookie_token = request.cookies.get(WEB_SESSION_COOKIE)
        if cookie_token:
            payload = verify_web_jwt(cookie_token)
            user_id = payload.get("sub") or payload.get("user_id")
            if not user_id:
                raise HTTPException(status_code=401, detail="Invalid web token: missing user id")
            return {
                "id": int(user_id),
                "raw_user": payload.get("user", {}),
                "auth_date": payload.get("iat", 0),
                "web_session": True,
            }

    raise HTTPException(status_code=401, detail="Authentication required")


async def get_current_user_id(
    request: Request,
    x_telegram_init_data: str | None = Header(default=None, alias="X-Telegram-Init-Data"),
    authorization: str | None = Header(default=None, alias="Authorization"),
    tg_web_app_data: str | None = Header(default=None, alias="TG-WebApp-Data"),
) -> int:
    user = await get_current_telegram_user(request, x_telegram_init_data, authorization, tg_web_app_data)
    return int(user["id"])
