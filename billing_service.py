"""Billing service for NeuroSMM.

Supports:
  - Telegram Stars payments (XTR currency via bot send_invoice)
  - YooKassa bank-card payments (REST API v3)

Payment flow stages (for structured logging / diagnostics):
  1. INVOICE_CREATED       — invoice or payment link generated
  2. PRE_CHECKOUT          — Telegram pre_checkout_query answered
  3. PAYMENT_RECEIVED      — successful_payment or webhook received
  4. SUBSCRIPTION_ACTIVATED — DB updated, tier active
  5. USER_NOTIFIED         — confirmation message sent to user

Configuration (.env / .env.example):
  PAYMENT_MODE        — "test" or "production"
  YOO_SHOP_ID         — YooKassa shop ID
  YOO_SECRET_KEY      — YooKassa secret key (test_ or live_ prefix)
  YOO_RETURN_URL      — URL user returns to after YooKassa payment
  YOO_RECEIPT_ENABLED — "1" to include 54-FZ receipt data in payments
  YOO_VAT_CODE        — VAT code for receipts (1-6, default 1 = 20%)
  STARS_PRO_PRICE     — price in Telegram Stars for Pro (default 99)
  STARS_MAX_PRICE     — price in Telegram Stars for Max (default 249)
"""
from __future__ import annotations

import ipaddress
import logging
import os
import uuid
from datetime import datetime, timedelta
from typing import Any

import aiohttp

import db
from config import load_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tier metadata
# ---------------------------------------------------------------------------

TIER_LABELS: dict[str, str] = {
    db.TIER_PRO: "Pro ⭐",
    db.TIER_MAX: "Max 💎",
}

RUB_PRICES: dict[str, int] = {
    db.TIER_PRO: 490,
    db.TIER_MAX: 990,
}

# Product descriptions for YooKassa receipt items (54-FZ)
PRODUCT_DESCRIPTIONS: dict[str, str] = {
    db.TIER_PRO: "Подписка NeuroSMM Pro — 30 дней",
    db.TIER_MAX: "Подписка NeuroSMM Max — 30 дней",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg():
    return load_config()


def _validate_payment_mode() -> None:
    """Log a warning if YooKassa key type doesn't match PAYMENT_MODE."""
    cfg = _cfg()
    if not cfg.yoo_secret_key:
        return
    key = cfg.yoo_secret_key
    mode = cfg.payment_mode
    if mode == "production" and key.startswith("test_"):
        logger.warning(
            "PAYMENT_MODE=production but YOO_SECRET_KEY starts with 'test_'. "
            "Payments will go to YooKassa test sandbox — no real charges."
        )
    elif mode == "test" and key.startswith("live_"):
        logger.warning(
            "PAYMENT_MODE=test but YOO_SECRET_KEY starts with 'live_'. "
            "This is a LIVE key — real money will be charged! "
            "Switch to a test key or set PAYMENT_MODE=production."
        )


# ---------------------------------------------------------------------------
# Telegram Stars
# ---------------------------------------------------------------------------

async def send_stars_invoice(bot, chat_id: int, tier: str) -> None:
    """Send a Telegram Stars invoice to the user for the requested tier.

    Uses bot.send_invoice() with currency='XTR' (Telegram Stars).
    The payload encodes tier so the successful_payment handler can act on it.
    """
    cfg = _cfg()
    prices_map = {
        db.TIER_PRO: cfg.stars_pro_price,
        db.TIER_MAX: cfg.stars_max_price,
    }
    if tier not in prices_map:
        raise ValueError(f"Unknown tier for Stars invoice: {tier!r}")

    stars_amount = prices_map[tier]
    label = TIER_LABELS[tier]
    descriptions = {
        db.TIER_PRO: (
            "✅ Безлимитные генерации постов\n"
            "✅ Автопостинг по расписанию\n"
            "✅ Голос → пост (Voice-to-Post)\n"
            "✅ News Sniper — мониторинг новостей\n"
            "✅ Подписка на 30 дней"
        ),
        db.TIER_MAX: (
            "✅ Все возможности Pro + News Sniper\n"
            "✅ Шпион конкурентов\n"
            "✅ Мультиканальность (до 10 каналов)\n"
            "✅ Приоритетная поддержка\n"
            "✅ Подписка на 30 дней"
        ),
    }

    from aiogram.types import LabeledPrice
    await bot.send_invoice(
        chat_id=chat_id,
        title=f"NeuroSMM {label}",
        description=descriptions[tier],
        payload=f"tier:{tier}",
        provider_token="",          # Empty string required for Telegram Stars (XTR)
        currency="XTR",
        prices=[LabeledPrice(label=label, amount=stars_amount)],
    )
    logger.info(
        "[PAYMENT:INVOICE_CREATED] method=stars owner_id=%s tier=%s amount=%s XTR",
        chat_id, tier, stars_amount,
    )


# ---------------------------------------------------------------------------
# YooKassa
# ---------------------------------------------------------------------------

def _build_receipt_data(tier: str, amount_rub: int) -> dict[str, Any] | None:
    """Build 54-FZ receipt data for YooKassa payment, if enabled.

    Returns a receipt dict to include in the payment request, or None.
    Requires the "Чеки" (receipts) service to be activated in the YooKassa
    merchant dashboard. Without it, YooKassa ignores the receipt field.
    """
    cfg = _cfg()
    if not cfg.yoo_receipt_enabled:
        return None

    description = PRODUCT_DESCRIPTIONS.get(tier, "Подписка NeuroSMM — 30 дней")
    return {
        "customer": {},
        "items": [
            {
                "description": description,
                "quantity": "1.00",
                "amount": {
                    "value": f"{amount_rub}.00",
                    "currency": "RUB",
                },
                "vat_code": cfg.yoo_vat_code,
                "payment_subject": "service",
                "payment_mode": "full_payment",
            }
        ],
    }


async def create_yookassa_payment(
    tier: str,
    owner_id: int,
    return_url: str | None = None,
) -> dict[str, Any]:
    """Create a YooKassa payment and return the confirmation URL.

    Returns a dict with keys:
      - payment_id: str
      - confirmation_url: str
      - status: str  (e.g. 'pending')

    Raises RuntimeError when YOO_SHOP_ID or YOO_SECRET_KEY are not configured.
    """
    cfg = _cfg()
    if not cfg.yoo_shop_id or not cfg.yoo_secret_key:
        raise RuntimeError(
            "YooKassa is not configured. Set YOO_SHOP_ID and YOO_SECRET_KEY in .env"
        )

    _validate_payment_mode()

    if tier not in RUB_PRICES:
        raise ValueError(f"Unknown tier: {tier!r}")

    amount_rub = RUB_PRICES[tier]
    label = TIER_LABELS[tier]
    idempotency_key = str(uuid.uuid4())
    back_url = return_url or cfg.yoo_return_url or "https://t.me/"

    api_payload: dict[str, Any] = {
        "amount": {"value": f"{amount_rub}.00", "currency": "RUB"},
        "confirmation": {"type": "redirect", "return_url": back_url},
        "capture": True,
        "description": f"NeuroSMM {label} — 30 дней",
        "metadata": {
            "owner_id": str(owner_id),
            "tier": tier,
        },
    }

    # 54-FZ receipt data (requires "Чеки" enabled in YooKassa dashboard)
    receipt = _build_receipt_data(tier, amount_rub)
    if receipt is not None:
        api_payload["receipt"] = receipt

    logger.info(
        "[PAYMENT:CREATING] method=yookassa owner_id=%s tier=%s amount=%s RUB receipt=%s",
        owner_id, tier, amount_rub, "yes" if receipt else "no",
    )

    async with aiohttp.ClientSession() as session:
        resp = await session.post(
            "https://api.yookassa.ru/v3/payments",
            json=api_payload,
            auth=aiohttp.BasicAuth(cfg.yoo_shop_id, cfg.yoo_secret_key),
            headers={
                "Idempotence-Key": idempotency_key,
                "Content-Type": "application/json",
            },
            timeout=aiohttp.ClientTimeout(total=15),
        )
        resp_json = await resp.json()
        if resp.status not in (200, 201):
            logger.error(
                "[PAYMENT:CREATE_FAILED] method=yookassa owner_id=%s tier=%s "
                "http_status=%s error=%s",
                owner_id, tier, resp.status,
                resp_json.get("description", resp_json),
            )
            raise RuntimeError(
                f"YooKassa API error {resp.status}: {resp_json.get('description', 'unknown')}"
            )

    confirmation = resp_json.get("confirmation") or {}
    payment_id = resp_json.get("id", "")
    logger.info(
        "[PAYMENT:INVOICE_CREATED] method=yookassa owner_id=%s tier=%s "
        "payment_id=%s amount=%s RUB",
        owner_id, tier, payment_id, amount_rub,
    )
    return {
        "payment_id": payment_id,
        "confirmation_url": confirmation.get("confirmation_url", ""),
        "status": resp_json.get("status", "pending"),
    }


# YooKassa webhook IP ranges (official list, verified 2025-04-02).
# Source: https://yookassa.ru/developers/using-api/webhooks#ip
# Re-verify this list periodically — YooKassa may add new ranges without notice.
_YOOKASSA_IP_NETWORKS = [
    ipaddress.ip_network("185.71.76.0/27"),
    ipaddress.ip_network("185.71.77.0/27"),
    ipaddress.ip_network("77.75.153.0/25"),
    ipaddress.ip_network("77.75.156.0/24"),
    ipaddress.ip_network("77.75.154.128/25"),
]


def verify_yookassa_webhook(body: bytes, ip: str) -> bool:
    """Validate that the YooKassa webhook comes from a known IP range.

    YooKassa sends webhooks only from specific IP ranges. We verify the
    source IP against the official whitelist. If ``YOO_WEBHOOK_SKIP_IP_CHECK``
    env var is set to ``1`` (for local development only), the check is skipped.

    NOTE: HMAC / request-body signature verification is NOT implemented here
    because the YooKassa REST API v3 does not provide a webhook signature or
    HMAC header (there is no ``X-Signature`` or equivalent).  IP allowlist
    verification is the only mechanism available per the official YooKassa
    documentation.  See: https://yookassa.ru/developers/using-api/webhooks
    """
    cfg = _cfg()
    skip = os.getenv("YOO_WEBHOOK_SKIP_IP_CHECK", "").strip() == "1"
    if skip:
        if cfg.payment_mode == "production":
            logger.error(
                "YOO_WEBHOOK_SKIP_IP_CHECK=1 in production mode — "
                "ignoring the flag and enforcing IP check for security"
            )
            # NEVER skip IP check in production — fall through to validation
        else:
            return True
    if not cfg.yoo_shop_id:
        # YooKassa not configured — reject everything
        return False

    if not ip:
        return False

    try:
        addr = ipaddress.ip_address(ip.strip())
    except ValueError:
        logger.warning("verify_yookassa_webhook: invalid IP %r", ip)
        return False

    for network in _YOOKASSA_IP_NETWORKS:
        if addr in network:
            return True

    logger.warning("verify_yookassa_webhook: IP %s not in YooKassa whitelist", ip)
    return False


# ---------------------------------------------------------------------------
# Shared: process successful payment (called by both webhook and Stars handler)
# ---------------------------------------------------------------------------

async def process_successful_payment(
    owner_id: int,
    tier: str,
    *,
    payment_id: str = "",
    method: str = "unknown",
    amount: str = "",
    currency: str = "",
) -> bool:
    """Upgrade a user's subscription tier and set expiry to +30 days.

    Idempotent: if payment_id is provided and already recorded, this is a no-op.
    Returns True if the subscription was activated, False if it was a duplicate.
    """
    if tier not in (db.TIER_PRO, db.TIER_MAX):
        logger.warning(
            "[PAYMENT:INVALID_TIER] owner_id=%s tier=%r payment_id=%s",
            owner_id, tier, payment_id,
        )
        return False

    logger.info(
        "[PAYMENT:RECEIVED] method=%s owner_id=%s tier=%s payment_id=%s amount=%s %s",
        method, owner_id, tier, payment_id, amount, currency,
    )

    # Idempotency: check if this payment was already processed
    if payment_id:
        already = await db.is_payment_processed(payment_id)
        if already:
            logger.info(
                "[PAYMENT:DUPLICATE] payment_id=%s already processed, skipping",
                payment_id,
            )
            return False

    # Record the payment event (also serves as idempotency guard via UNIQUE index)
    if payment_id:
        inserted = await db.record_payment_event(
            payment_id=payment_id,
            owner_id=owner_id,
            tier=tier,
            method=method,
            amount=amount,
            currency=currency,
        )
        if not inserted:
            logger.info(
                "[PAYMENT:DUPLICATE] payment_id=%s concurrent duplicate, skipping",
                payment_id,
            )
            return False
    else:
        # Stars payments may not have a unique payment_id from Telegram;
        # record with a generated UUID so we still have an audit trail.
        await db.record_payment_event(
            payment_id=f"stars_{owner_id}_{uuid.uuid4().hex[:12]}",
            owner_id=owner_id,
            tier=tier,
            method=method,
            amount=amount,
            currency=currency,
        )

    # Activate subscription
    expires_at = (datetime.utcnow() + timedelta(days=30)).isoformat(timespec="seconds")
    await db.set_user_subscription(
        owner_id,
        tier,
        expires_at=expires_at,
        auto_renew=False,
    )
    logger.info(
        "[PAYMENT:SUBSCRIPTION_ACTIVATED] owner_id=%s tier=%s expires_at=%s payment_id=%s",
        owner_id, tier, expires_at, payment_id,
    )
    return True


async def notify_user_payment_success(bot, owner_id: int, tier: str) -> None:
    """Send a confirmation message to the user after YooKassa payment via webhook.

    This is called from the webhook handler so the user gets a bot message
    even though they completed payment on an external page.
    """
    tier_labels = {db.TIER_PRO: "Pro ⭐", db.TIER_MAX: "Max 💎"}
    tier_label = tier_labels.get(tier, tier.capitalize())
    text = (
        f"🎉 Оплата прошла успешно!\n\n"
        f"Тариф <b>{tier_label}</b> активирован на 30 дней.\n"
        f"Все функции тарифа уже доступны. Приятного использования!"
    )
    try:
        await bot.send_message(owner_id, text, parse_mode="HTML")
        logger.info("[PAYMENT:USER_NOTIFIED] owner_id=%s tier=%s", owner_id, tier)
    except Exception:
        logger.warning(
            "[PAYMENT:NOTIFY_FAILED] Could not send confirmation to owner_id=%s",
            owner_id,
        )
