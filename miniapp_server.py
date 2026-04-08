from __future__ import annotations

import json
import logging
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

import db
from auth import (
    get_current_telegram_user,
    create_web_jwt,
    verify_telegram_login_widget,
    WEB_SESSION_COOKIE,
)
from billing_service import process_successful_payment, verify_yookassa_webhook, create_yookassa_payment, notify_user_payment_success
from ai_client import close_all_clients
from config import load_config
from miniapp_shared import BASE_DIR, GENERATED_DIR, UPLOAD_DIR
from miniapp_routes_core import router as core_router
from miniapp_routes_content import router as content_router
from miniapp_routes_media import router as media_router

cfg = load_config()
logger = logging.getLogger(__name__)
app = FastAPI(
    title="NeuroSMM Mini App API",
    docs_url="/docs" if cfg.enable_docs else None,
    redoc_url="/redoc" if cfg.enable_docs else None,
    openapi_url="/openapi.json" if cfg.enable_docs else None,
)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    # Content-Security-Policy:
    # - 'unsafe-inline' REMOVED from script-src: all inline onclick handlers have been
    #   migrated to event delegation (data-action attributes). No inline <script> tags exist.
    # - 'unsafe-inline' is still required for style-src (Telegram Web App SDK injects inline styles).
    # - Images are allowed from 'self', data: URIs, blob: URIs, and any HTTPS source
    #   (stock photo providers such as Unsplash/Pexels/Pixabay return CDN URLs).
    # - connect-src 'self' https: covers API calls and the Telegram SDK.
    # CSP: restrict connect-src and img-src to known providers instead of blanket https:
    _CSP = (
        "default-src 'none'; "
        "script-src 'self' https://telegram.org https://oauth.telegram.org; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: blob: https://images.unsplash.com https://images.pexels.com https://cdn.pixabay.com https://oaidalleapiprodscus.blob.core.windows.net https://*.hf.space; "
        "connect-src 'self' https://telegram.org https://oauth.telegram.org https://api.unsplash.com https://api.pexels.com https://pixabay.com; "
        "frame-src https://oauth.telegram.org; "
        "frame-ancestors 'none'; "
        "base-uri 'self'; "
        "form-action 'self';"
    )

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "same-origin")
        response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        response.headers.setdefault("Cross-Origin-Resource-Policy", "same-site")
        # COOP: Use same-origin-allow-popups to allow Telegram Login Widget popup
        # to communicate back.  The widget opens a popup on oauth.telegram.org
        # which must be able to call window.opener._onTelegramLoginAuth().
        response.headers.setdefault("Cross-Origin-Opener-Policy", "same-origin-allow-popups")
        response.headers.setdefault("Cache-Control", "no-store")
        response.headers.setdefault("Content-Security-Policy", self._CSP)
        if request.url.scheme == "https":
            response.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
        return response


class CSRFMiddleware(BaseHTTPMiddleware):
    """CSRF protection for web mode using Origin/Referer header verification.

    For mutating requests (POST/PATCH/PUT/DELETE) to /api/ endpoints:
    - If the request has a Telegram Mini App header (X-Telegram-Init-Data),
      it's from the Mini App and doesn't need CSRF check.
    - If the request uses cookie auth (web mode), verify that the Origin
      or Referer header matches the server's host.
    - Webhooks (/api/payments/webhook) are exempted as they come from
      external payment providers.

    This works alongside SameSite=lax cookies, providing defense-in-depth.
    """
    _EXEMPT_PATHS = {"/api/payments/webhook", "/healthz"}
    _MUTATING_METHODS = {"POST", "PATCH", "PUT", "DELETE"}

    async def dispatch(self, request: Request, call_next):
        if (
            request.method in self._MUTATING_METHODS
            and request.url.path.startswith("/api/")
            and request.url.path not in self._EXEMPT_PATHS
        ):
            # Mini App requests carry their own HMAC-verified auth — no CSRF risk.
            # IMPORTANT: We only skip CSRF if a *real* Telegram header is present.
            # The actual HMAC verification happens later in get_current_telegram_user.
            # A fake header will fail at that stage with 401. The CSRF check here
            # is defense-in-depth for cookie-only (web) sessions.
            has_tg_header = bool(
                request.headers.get("x-telegram-init-data")
                or request.headers.get("tg-webapp-data")
            )
            auth_header = request.headers.get("authorization") or ""
            has_tma_auth = auth_header.lower().startswith("tma ")

            if not has_tg_header and not has_tma_auth:
                # Web mode: verify Origin/Referer — at least one MUST be present
                origin = request.headers.get("origin") or ""
                referer = request.headers.get("referer") or ""
                request_host = request.headers.get("host") or ""

                if not origin and not referer:
                    # Both missing = potential CSRF from old browser or tool
                    logger.warning(
                        "CSRF check failed: no origin/referer method=%s path=%s host=%r",
                        request.method, request.url.path, request_host,
                    )
                    return JSONResponse(
                        status_code=403,
                        content={"detail": "CSRF verification failed: missing origin header"},
                    )

                origin_ok = False
                if origin:
                    # Origin header: must match the request host
                    from urllib.parse import urlparse
                    parsed = urlparse(origin)
                    origin_ok = (parsed.netloc == request_host) if request_host else False
                elif referer:
                    from urllib.parse import urlparse
                    parsed = urlparse(referer)
                    origin_ok = (parsed.netloc == request_host) if request_host else False

                if not origin_ok:
                    logger.warning(
                        "CSRF check failed: method=%s path=%s origin=%r host=%r",
                        request.method, request.url.path, origin, request_host,
                    )
                    return JSONResponse(
                        status_code=403,
                        content={"detail": "CSRF verification failed: origin mismatch"},
                    )

        return await call_next(request)

class SimpleRateLimitMiddleware(BaseHTTPMiddleware):
    """In-memory sliding-window rate limiter.

    Note: Rate limits are per-process and reset on restart.  Without an external
    store (Redis/Memcached), this is the most practical approach for a single-
    process deployment.  For multi-process or high-availability setups, a Redis-
    backed limiter should replace this middleware.
    """
    _MAX_TRACKED_IPS = 5000  # evict oldest entries when exceeded
    _STALE_THRESHOLD_SECONDS = 120.0  # entries older than this are considered stale

    def __init__(self, app, read_rpm: int, write_rpm: int):
        super().__init__(app)
        self.read_rpm = max(30, int(read_rpm or 240))
        self.write_rpm = max(20, int(write_rpm or 90))
        self._hits: dict[tuple[str, str], deque[float]] = defaultdict(deque)

    def _evict_stale_entries(self) -> None:
        """Remove stale entries to prevent unbounded memory growth."""
        if len(self._hits) <= self._MAX_TRACKED_IPS:
            return
        now = time.monotonic()
        stale_keys = [k for k, q in self._hits.items() if not q or now - q[-1] > self._STALE_THRESHOLD_SECONDS]
        for k in stale_keys:
            del self._hits[k]
        # If still too many, drop oldest half
        if len(self._hits) > self._MAX_TRACKED_IPS:
            keys = list(self._hits.keys())
            for k in keys[: len(keys) // 2]:
                del self._hits[k]

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if not path.startswith('/api/'):
            return await call_next(request)
        self._evict_stale_entries()
        limit = self.write_rpm if request.method.upper() in {'POST', 'PATCH', 'PUT', 'DELETE'} else self.read_rpm
        # Use the direct connection IP, not X-Forwarded-For, to prevent clients
        # from spoofing their IP and bypassing rate limits.
        # If this service runs behind a trusted reverse proxy (nginx/Traefik),
        # configure the proxy to set REMOTE_ADDR correctly (e.g. via proxy_protocol
        # or real_ip_from in nginx) rather than trusting X-Forwarded-For here.
        ip = request.client.host or 'unknown'
        key = (ip, 'w' if request.method.upper() in {'POST', 'PATCH', 'PUT', 'DELETE'} else 'r')
        now = time.monotonic()
        q = self._hits[key]
        while q and now - q[0] > 60.0:
            q.popleft()
        if len(q) >= limit:
            return JSONResponse(status_code=429, content={'detail': 'Too many requests'})
        q.append(now)
        return await call_next(request)


if cfg.trusted_hosts and cfg.trusted_hosts != ('*',):
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(cfg.trusted_hosts))

app.add_middleware(SimpleRateLimitMiddleware, read_rpm=cfg.api_rate_limit_rpm, write_rpm=cfg.api_write_rate_limit_rpm)
app.add_middleware(CSRFMiddleware)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(GZipMiddleware, minimum_size=1024)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(cfg.allowed_cors_origins),
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "X-Telegram-Init-Data", "Authorization", "TG-WebApp-Data"],
    expose_headers=["Content-Length", "Content-Type"],
    allow_credentials=True,
    max_age=3600,
)


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception):
    logger.exception('Unhandled error path=%s method=%s', request.url.path, request.method)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get('/healthz')
async def healthz():
    # Only report application liveness — do NOT expose database or internal
    # service state to unauthenticated callers.
    return {'ok': True}


@app.get('/readyz')
async def readyz():
    """Readiness probe — checks that critical dependencies are accessible.

    Returns 200 if DB is reachable; 503 otherwise.
    Does NOT expose sensitive details to unauthenticated callers.
    """
    checks: dict[str, bool] = {}
    # DB check
    try:
        async with db._db_ctx() as conn:
            cur = await conn.execute("SELECT 1")
            await cur.fetchone()
        checks["db"] = True
    except Exception:
        checks["db"] = False
    all_ok = all(checks.values())
    status = 200 if all_ok else 503
    return JSONResponse(status_code=status, content={"ready": all_ok, "checks": checks})


@app.get('/api/web-auth/diag')
async def web_auth_diag(request: Request):
    """Browser auth diagnostic endpoint — helps users and admins troubleshoot
    Telegram Login Widget / VPN / network issues.

    Returns non-sensitive diagnostic information:
    - Server-side auth config (enabled, bot configured)
    - Request origin/headers analysis
    - CSP policy summary
    - Expected domains (for widget, oauth, popups)
    - Common failure causes checklist

    Does NOT expose secrets, tokens, or user data.
    """
    _cfg = load_config()
    bot_username = _cfg.bot_username or ""
    bot_id = ""
    if _cfg.bot_token and ":" in _cfg.bot_token:
        bot_id = _cfg.bot_token.split(":")[0]

    # Request analysis
    origin = request.headers.get("origin") or ""
    referer = request.headers.get("referer") or ""
    host = request.headers.get("host") or ""
    user_agent = request.headers.get("user-agent") or ""
    proto = request.url.scheme
    x_forwarded_proto = request.headers.get("x-forwarded-proto") or ""
    x_forwarded_for = request.headers.get("x-forwarded-for") or ""

    # Determine effective protocol (behind reverse proxy)
    effective_proto = x_forwarded_proto or proto
    is_https = effective_proto == "https"

    # Widget requirements check
    checks: list[dict[str, str | bool]] = []

    checks.append({
        "check": "web_auth_enabled",
        "ok": bool(_cfg.web_auth_enabled),
        "detail": "Web auth is enabled" if _cfg.web_auth_enabled else "Web auth is DISABLED in server config",
    })

    checks.append({
        "check": "bot_configured",
        "ok": bool(bot_username and bot_id),
        "detail": f"Bot: @{bot_username} (id: {bot_id})" if bot_username else "Bot username not configured",
    })

    checks.append({
        "check": "https",
        "ok": is_https,
        "detail": f"Protocol: {effective_proto}" + (" (OK)" if is_https else " — Telegram widget requires HTTPS"),
    })

    checks.append({
        "check": "host_header",
        "ok": bool(host),
        "detail": f"Host: {host}" if host else "Missing Host header",
    })

    # Domains that must be reachable for the widget to work
    required_domains = [
        "telegram.org",           # Widget JS
        "oauth.telegram.org",     # OAuth iframe/popup
    ]

    checks.append({
        "check": "csp_allows_telegram",
        "ok": True,  # Our CSP explicitly allows these
        "detail": f"CSP script-src includes telegram.org; frame-src includes oauth.telegram.org",
    })

    checks.append({
        "check": "coop_popup",
        "ok": True,
        "detail": "COOP is same-origin-allow-popups (required for widget popup fallback)",
    })

    # Common causes of widget failure
    troubleshooting = [
        {
            "issue": "Widget iframe doesn't appear",
            "possible_causes": [
                "Domain not registered in BotFather (/setdomain command)",
                "telegram.org blocked by VPN/firewall/DNS",
                "Browser privacy extensions blocking third-party iframes",
                "Page not served over HTTPS",
            ],
            "fixable_in_code": False,
            "server_action": "Ensure domain is set via /setdomain in BotFather",
        },
        {
            "issue": "Widget loads but login fails",
            "possible_causes": [
                "HMAC verification failed (clock skew > 1 hour)",
                "Bot token mismatch between BotFather and server config",
                "Cookie blocked (SameSite/Secure mismatch if not HTTPS)",
            ],
            "fixable_in_code": True,
            "server_action": "Verify bot token matches, ensure HTTPS",
        },
        {
            "issue": "Site unreachable with VPN",
            "possible_causes": [
                "VPN/proxy DNS resolution fails for server domain",
                "Cloudflare/CDN blocks VPN IP ranges",
                "Server reverse proxy (NGINX) rejects unknown IPs",
                "Mixed DNS: VPN resolves to different IP than expected",
            ],
            "fixable_in_code": False,
            "server_action": "Check NGINX/Cloudflare settings; ensure no IP-based blocking",
        },
        {
            "issue": "Popup blocked",
            "possible_causes": [
                "Browser blocks popups (user must allow)",
                "COOP policy mismatch (should be same-origin-allow-popups)",
            ],
            "fixable_in_code": True,
            "server_action": "Ensure COOP header is same-origin-allow-popups",
        },
    ]

    return JSONResponse({
        "ok": True,
        "request_info": {
            "origin": origin[:120] if origin else None,
            "host": host,
            "protocol": effective_proto,
            "is_https": is_https,
            "has_x_forwarded_proto": bool(x_forwarded_proto),
            "has_x_forwarded_for": bool(x_forwarded_for),
            "user_agent_short": user_agent[:80] if user_agent else None,
        },
        "auth_config": {
            "web_auth_enabled": bool(_cfg.web_auth_enabled),
            "bot_username": bot_username,
            "bot_id_configured": bool(bot_id),
        },
        "checks": checks,
        "required_domains": required_domains,
        "troubleshooting": troubleshooting,
    })


@app.on_event("startup")
async def startup() -> None:
    await db.init_db()


@app.on_event("shutdown")
async def shutdown() -> None:
    await close_all_clients()
    await db.close_pool()


app.include_router(core_router)
app.include_router(content_router)
app.include_router(media_router)


@app.post("/api/payments/create")
async def create_payment(
    request: Request,
    telegram_user: dict = Depends(get_current_telegram_user),
):
    """Create a YooKassa payment link for the given tier."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"detail": "Invalid JSON"})

    tier = str(body.get("tier") or "").lower()
    if tier not in ("pro", "max"):
        return JSONResponse(status_code=400, content={"detail": "Invalid tier. Use 'pro' or 'max'."})

    owner_id = int(telegram_user.get("id") or 0)
    if not owner_id:
        return JSONResponse(status_code=401, content={"detail": "Unauthorized"})

    try:
        result = await create_yookassa_payment(tier, owner_id)
        return JSONResponse(result)
    except RuntimeError:
        return JSONResponse(status_code=503, content={"detail": "Payment provider is not configured."})
    except Exception:
        logger.exception("create_payment: unexpected error owner_id=%s tier=%s", owner_id, tier)
        return JSONResponse(status_code=500, content={"detail": "Internal error"})


@app.post("/api/payments/yookassa/webhook")
async def yookassa_webhook(request: Request):
    """Receive payment notifications from YooKassa and activate subscriptions.

    Idempotent: duplicate webhooks for the same payment_id are safely ignored.
    """
    body = await request.body()
    # Use the direct connection IP, not X-Forwarded-For, for IP whitelist verification.
    # A spoofed X-Forwarded-For header could bypass the YooKassa IP whitelist.
    # Ensure your reverse proxy (nginx/Traefik) sets REMOTE_ADDR to the real client IP.
    client_ip = request.client.host or ""
    if not verify_yookassa_webhook(body, client_ip):
        logger.warning("[PAYMENT:WEBHOOK_REJECTED] ip=%s", client_ip)
        return JSONResponse(status_code=403, content={"detail": "Forbidden"})

    try:
        event = json.loads(body)
    except Exception:
        logger.warning("[PAYMENT:WEBHOOK_BAD_JSON] ip=%s", client_ip)
        return JSONResponse(status_code=400, content={"detail": "Invalid JSON"})

    event_type = event.get("event", "")
    if event_type != "payment.succeeded":
        # Acknowledge non-payment events silently (e.g. payment.canceled)
        logger.info("[PAYMENT:WEBHOOK_EVENT] type=%s (ignored)", event_type)
        return JSONResponse({"ok": True})

    payment_obj = event.get("object") or {}
    payment_id = payment_obj.get("id", "")
    metadata = payment_obj.get("metadata") or {}
    owner_id_raw = metadata.get("owner_id")
    tier = metadata.get("tier", "")
    amount_obj = payment_obj.get("amount") or {}
    amount_value = amount_obj.get("value", "")
    amount_currency = amount_obj.get("currency", "RUB")

    if not owner_id_raw or not tier:
        logger.warning(
            "[PAYMENT:WEBHOOK_BAD_METADATA] payment_id=%s owner_id=%s tier=%s",
            payment_id, owner_id_raw, tier,
        )
        return JSONResponse(status_code=400, content={"detail": "Missing metadata"})

    try:
        owner_id = int(owner_id_raw)
    except (ValueError, TypeError):
        logger.warning(
            "[PAYMENT:WEBHOOK_BAD_OWNER] payment_id=%s owner_id_raw=%r",
            payment_id, owner_id_raw,
        )
        return JSONResponse(status_code=400, content={"detail": "Invalid owner_id"})

    try:
        activated = await process_successful_payment(
            owner_id, tier,
            payment_id=payment_id,
            method="yookassa",
            amount=amount_value,
            currency=amount_currency,
        )
        if activated:
            logger.info(
                "[PAYMENT:WEBHOOK_SUCCESS] payment_id=%s owner_id=%s tier=%s",
                payment_id, owner_id, tier,
            )
            # Notify the user via bot message (best-effort, non-blocking).
            # The webhook server may not share a Bot instance with app.py,
            # so we create a lightweight Bot just for sending the notification.
            try:
                from aiogram import Bot
                from aiogram.client.default import DefaultBotProperties
                _cfg = load_config()
                _notify_bot = Bot(
                    token=_cfg.bot_token,
                    default=DefaultBotProperties(parse_mode=None),
                )
                try:
                    await notify_user_payment_success(_notify_bot, owner_id, tier)
                finally:
                    await _notify_bot.session.close()
            except Exception:
                logger.warning(
                    "[PAYMENT:NOTIFY_SKIP] Could not send bot notification "
                    "for owner_id=%s", owner_id, exc_info=True,
                )
        else:
            logger.info(
                "[PAYMENT:WEBHOOK_DUPLICATE] payment_id=%s already processed",
                payment_id,
            )
    except Exception:
        logger.exception(
            "[PAYMENT:WEBHOOK_ERROR] process_successful_payment failed "
            "payment_id=%s owner_id=%s tier=%s",
            payment_id, owner_id, tier,
        )
        return JSONResponse(status_code=500, content={"detail": "Internal error"})

    return JSONResponse({"ok": True})


import re as _re

_SAFE_MEDIA_NAME_RE = _re.compile(r'^[\w][\w.\-]{0,254}$')


# ---------------------------------------------------------------------------
# Web Auth endpoints  (Telegram Login Widget → JWT)
# ---------------------------------------------------------------------------

@app.get("/api/web-auth/config")
async def web_auth_config():
    """Public endpoint returning web auth availability and bot info."""
    _cfg = load_config()
    bot_username = _cfg.bot_username or ""
    # Extract bot_id from the token (first part before ':')
    bot_id = ""
    if _cfg.bot_token and ":" in _cfg.bot_token:
        bot_id = _cfg.bot_token.split(":")[0]
    return JSONResponse({
        "enabled": _cfg.web_auth_enabled,
        "bot_username": bot_username,
        "bot_id": bot_id,
    })


@app.post("/api/web-auth/telegram-login")
async def web_auth_telegram_login(request: Request):
    """Verify Telegram Login Widget callback data and set an HttpOnly session cookie.

    Expected JSON body: the user object returned by the Login Widget
    containing at minimum ``id``, ``auth_date``, and ``hash``.
    """
    _cfg = load_config()
    if not _cfg.web_auth_enabled:
        return JSONResponse(status_code=403, content={"detail": "Web auth is disabled"})

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"detail": "Invalid JSON"})

    if not isinstance(body, dict):
        return JSONResponse(status_code=400, content={"detail": "Expected JSON object"})

    try:
        user_info = verify_telegram_login_widget(body)
    except HTTPException as exc:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    user_id = user_info["id"]
    raw_user = user_info.get("raw_user", {})

    # Ensure user has at least a free-tier subscription record.
    try:
        await db.ensure_web_user(user_id)
    except Exception:
        logger.exception("web_auth: ensure_web_user failed for %s", user_id)

    token = create_web_jwt({
        "sub": user_id,
        "user_id": user_id,
        "user": {
            "id": user_id,
            "first_name": raw_user.get("first_name", ""),
            "last_name": raw_user.get("last_name", ""),
            "username": raw_user.get("username", ""),
            "photo_url": raw_user.get("photo_url", ""),
        },
    })

    response = JSONResponse({
        "ok": True,
        "user": {
            "id": user_id,
            "first_name": raw_user.get("first_name", ""),
            "username": raw_user.get("username", ""),
        },
    })
    response.set_cookie(
        key=WEB_SESSION_COOKIE,
        value=token,
        max_age=_cfg.web_auth_token_ttl,
        httponly=True,
        secure=True,
        samesite="lax",
        path="/",
    )
    return response


@app.get("/api/web-auth/telegram-callback")
async def web_auth_telegram_callback(request: Request):
    """Handle Telegram Login Widget redirect callback (data-auth-url mode).

    The widget redirects to this URL with auth data as query parameters:
    id, first_name, last_name, username, photo_url, auth_date, hash.
    We verify the data, set the session cookie, and redirect back to ``/``.
    """
    _cfg = load_config()
    if not _cfg.web_auth_enabled:
        return RedirectResponse(url="/", status_code=302)

    params = dict(request.query_params)
    if not params.get("hash") or not params.get("id"):
        logger.warning("web_auth_callback: missing hash or id in query params")
        return RedirectResponse(url="/", status_code=302)

    try:
        user_info = verify_telegram_login_widget(params)
    except HTTPException as exc:
        logger.warning("web_auth_callback: verification failed: %s", exc.detail)
        return RedirectResponse(url="/", status_code=302)

    user_id = user_info["id"]
    raw_user = user_info.get("raw_user", {})

    try:
        await db.ensure_web_user(user_id)
    except Exception:
        logger.exception("web_auth_callback: ensure_web_user failed for %s", user_id)

    token = create_web_jwt({
        "sub": user_id,
        "user_id": user_id,
        "user": {
            "id": user_id,
            "first_name": raw_user.get("first_name", ""),
            "last_name": raw_user.get("last_name", ""),
            "username": raw_user.get("username", ""),
            "photo_url": raw_user.get("photo_url", ""),
        },
    })

    response = RedirectResponse(url="/", status_code=302)
    response.set_cookie(
        key=WEB_SESSION_COOKIE,
        value=token,
        max_age=_cfg.web_auth_token_ttl,
        httponly=True,
        secure=True,
        samesite="lax",
        path="/",
    )
    return response


@app.post("/api/web-auth/logout")
async def web_auth_logout():
    """Clear the web session cookie."""
    response = JSONResponse({"ok": True})
    response.delete_cookie(
        key=WEB_SESSION_COOKIE,
        path="/",
        httponly=True,
        secure=True,
        samesite="lax",
    )
    return response


def _safe_media_filename(filename: str) -> str:
    """Return *filename* after strict validation against a safe character allowlist.

    Only alphanumeric characters, underscores, dots, and hyphens are permitted.
    Any other input — including path separators, null bytes, or dot-dot sequences —
    raises ``HTTPException(400)``.

    Note: FastAPI path parameters declared as ``{name}`` (without ``:path``) never
    contain forward slashes, so directory traversal via ``/`` is already prevented
    by routing.  This function guards against remaining escape vectors.
    """
    safe_name = str(filename or "")
    if not _SAFE_MEDIA_NAME_RE.match(safe_name) or ".." in safe_name:
        raise HTTPException(status_code=400, detail="Invalid filename")
    return safe_name


def _resolve_confined_path(base_dir: Path, safe_name: str) -> Path:
    """Resolve ``base_dir / safe_name`` and verify the result stays inside *base_dir*.

    Uses ``Path.resolve()`` to expand any remaining symlinks, then confirms the
    resolved path is a descendant of the resolved base directory.  This is
    defense-in-depth on top of ``_safe_media_filename``'s character validation.
    Raises ``HTTPException(400)`` on any escape attempt.
    """
    resolved_base = base_dir.resolve()
    candidate = resolved_base / safe_name
    # Resolve the candidate after joining with the validated base to catch symlinks.
    resolved_candidate = candidate.resolve()
    try:
        resolved_candidate.relative_to(resolved_base)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid filename")
    return resolved_candidate


@app.get("/uploads/{filename}")
async def serve_upload(
    filename: str,
    telegram_user: dict = Depends(get_current_telegram_user),
):
    """Serve user-uploaded files with authentication and ownership verification.

    Files are named ``{owner_id}_{uuid}.{ext}`` by save_upload_file().  We
    extract the owner prefix and compare it against the authenticated user so
    that users can only access their own uploads.
    """
    safe_name = _safe_media_filename(filename)

    owner_id = int(telegram_user.get("id") or 0)
    if not owner_id:
        raise HTTPException(status_code=403, detail="Invalid user ID")

    # Ownership check: uploaded files are prefixed with "{owner_id}_"
    if not safe_name.startswith(f"{owner_id}_"):
        raise HTTPException(status_code=403, detail="Access denied")

    file_path = _resolve_confined_path(UPLOAD_DIR, safe_name)
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(str(file_path), headers={"Cache-Control": "private, max-age=3600"})


@app.get("/generated-images/{filename}")
async def serve_generated_image(
    filename: str,
    telegram_user: dict = Depends(get_current_telegram_user),
):
    """Serve AI-generated images with authentication and ownership verification.

    New images are named ``{owner_id}_{hash}.{ext}``.  Legacy images without
    an owner prefix are served to any authenticated user for backward compat.
    """
    safe_name = _safe_media_filename(filename)

    owner_id = int(telegram_user.get("id") or 0)
    if not owner_id:
        raise HTTPException(status_code=403, detail="Invalid user ID")

    # Ownership check: new images have "{owner_id}_" prefix
    # Legacy images (hash-only names) are allowed for any authenticated user
    parts = safe_name.split("_", 1)
    if len(parts) == 2 and parts[0].isdigit():
        file_owner = int(parts[0])
        if file_owner != owner_id:
            raise HTTPException(status_code=403, detail="Access denied")

    file_path = _resolve_confined_path(GENERATED_DIR, safe_name)
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(str(file_path), headers={"Cache-Control": "private, max-age=3600"})


# ---------------------------------------------------------------------------
# Static assets served from the project root
# (app.js and styles.css live alongside the Python source files)
# ---------------------------------------------------------------------------

_ALLOWED_ROOT_ASSETS = {
    "app.js": "application/javascript; charset=utf-8",
    "styles.css": "text/css; charset=utf-8",
}

# Compute content hashes at import time for cache-busting ETag headers.
# This ensures browsers fetch fresh assets after every deploy without
# needing to manually update version strings in HTML.
_ASSET_ETAGS: dict[str, str] = {}
for _asset_name in _ALLOWED_ROOT_ASSETS:
    _asset_path = BASE_DIR / _asset_name
    if _asset_path.is_file():
        import hashlib as _hl
        _ASSET_ETAGS[_asset_name] = _hl.md5(_asset_path.read_bytes()).hexdigest()[:12]


@app.get("/app.js")
async def serve_app_js():
    path = BASE_DIR / "app.js"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    headers = {"Cache-Control": "public, max-age=300, must-revalidate"}
    etag = _ASSET_ETAGS.get("app.js")
    if etag:
        headers["ETag"] = f'"{etag}"'
    return FileResponse(str(path), media_type=_ALLOWED_ROOT_ASSETS["app.js"], headers=headers)


@app.get("/styles.css")
async def serve_styles_css():
    path = BASE_DIR / "styles.css"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    headers = {"Cache-Control": "public, max-age=300, must-revalidate"}
    etag = _ASSET_ETAGS.get("styles.css")
    if etag:
        headers["ETag"] = f'"{etag}"'
    return FileResponse(str(path), media_type=_ALLOWED_ROOT_ASSETS["styles.css"], headers=headers)
    return FileResponse(str(path), media_type=_ALLOWED_ROOT_ASSETS["styles.css"])


# Ensure the miniapp directory exists for the StaticFiles mount.
(BASE_DIR / "miniapp").mkdir(exist_ok=True)

app.mount("/", StaticFiles(directory=str(BASE_DIR / "miniapp"), html=True), name="miniapp")
