# Payment Flow: Telegram Bot + YooKassa

## Architecture Overview

NeuroSMM supports two payment methods:

| Method | Currency | Provider | Flow |
|--------|----------|----------|------|
| **Telegram Stars** | XTR | Telegram built-in | Bot invoice → Telegram checkout → `successful_payment` handler |
| **YooKassa** | RUB | YooKassa REST API v3 | Bot/MiniApp → YooKassa checkout page → webhook callback |

Both paths converge at `process_successful_payment()` in `billing_service.py`, which:
1. Validates the tier
2. Checks for duplicate payments (idempotency)
3. Records the payment event in `payment_events` table
4. Activates the subscription (+30 days)
5. Logs all stages with structured `[PAYMENT:*]` tags

---

## Payment Flow Stages

### Telegram Stars Flow

```
User clicks "⭐ Pro — 99 ⭐"
  → cb_buy() [handlers_private_messages.py]
  → send_stars_invoice() [billing_service.py]
    → [PAYMENT:INVOICE_CREATED] logged
  → Telegram shows native payment dialog
  → User confirms payment
  → handle_pre_checkout() auto-approves
    → [PAYMENT:PRE_CHECKOUT] logged
  → handle_successful_payment()
    → [PAYMENT:RECEIVED] logged
    → process_successful_payment(method="stars")
      → idempotency check (payment_events)
      → record_payment_event()
      → set_user_subscription(tier, expires=+30d)
      → [PAYMENT:SUBSCRIPTION_ACTIVATED] logged
    → Bot sends "🎉 Оплата прошла!" message
    → [PAYMENT:USER_NOTIFIED] logged
```

### YooKassa Flow

```
User clicks "💳 Pro — 490 ₽"
  → cb_buy() or POST /api/payments/create
  → create_yookassa_payment() [billing_service.py]
    → _validate_payment_mode() — warns on key/mode mismatch
    → _build_receipt_data() — includes 54-FZ receipt if enabled
    → POST https://api.yookassa.ru/v3/payments
    → [PAYMENT:INVOICE_CREATED] logged
  → User is redirected to YooKassa checkout page
  → User completes payment
  → YooKassa sends POST /api/payments/yookassa/webhook
    → verify_yookassa_webhook() — IP whitelist check
    → [PAYMENT:RECEIVED] logged
    → process_successful_payment(method="yookassa", payment_id=...)
      → idempotency check (payment_events — UNIQUE on payment_id)
      → record_payment_event()
      → set_user_subscription(tier, expires=+30d)
      → [PAYMENT:SUBSCRIPTION_ACTIVATED] logged
    → Bot sends "🎉 Оплата прошла!" message to user
    → [PAYMENT:USER_NOTIFIED] logged
```

---

## Configuration Guide

### Credentials Separation

| Variable | What it is | Where to get it |
|----------|------------|-----------------|
| `BOT_TOKEN` | Telegram bot token | @BotFather |
| `YOO_SHOP_ID` | YooKassa merchant ID | YooKassa dashboard → Shop Settings |
| `YOO_SECRET_KEY` | YooKassa API key | YooKassa dashboard → API Keys |

**There is no `TELEGRAM_PROVIDER_TOKEN` in this project.** Telegram Stars uses
`provider_token=""` (empty string, as required by Telegram). YooKassa payments
go through the REST API directly, not through Telegram's provider system.

### Test vs Production

| Setting | `PAYMENT_MODE=test` | `PAYMENT_MODE=production` |
|---------|---------------------|---------------------------|
| YooKassa key | Must start with `test_` | Must start with `live_` |
| Real charges | No (sandbox) | Yes |
| Webhook IP check | Can be skipped via `YOO_WEBHOOK_SKIP_IP_CHECK=1` | Always enforced (warning if skipped) |
| Stars payments | Work normally (test via BotFather test env) | Work normally |

**Safety:** If `PAYMENT_MODE=test` but `YOO_SECRET_KEY` starts with `live_`,
a warning is logged. If `PAYMENT_MODE=production` but key starts with `test_`,
another warning is logged. This prevents accidental mode mismatch.

### Switching from Test to Production

1. In YooKassa dashboard, get your **live** API key (starts with `live_`)
2. In `.env`:
   ```
   PAYMENT_MODE=production
   YOO_SECRET_KEY=live_xxxxxxxxxxxxx
   ```
3. Ensure `YOO_WEBHOOK_SKIP_IP_CHECK` is **not** set to `1`
4. Ensure nginx is configured to pass real client IP (see `DEPLOYMENT.md`)
5. Set `YOO_RETURN_URL` to your bot's deep link (e.g. `https://t.me/your_bot`)
6. Restart the application

---

## Idempotency & Reliability

### Duplicate Payment Protection

Every successful payment is recorded in the `payment_events` table with a
`UNIQUE` index on `payment_id`. If the same webhook arrives twice (network
retry, etc.), the second call is a safe no-op:

1. `is_payment_processed(payment_id)` → returns True → skip
2. Even if check races, `record_payment_event()` INSERT fails on UNIQUE → skip

For Telegram Stars, the `telegram_payment_charge_id` is used as `payment_id`.

### Audit Trail

The `payment_events` table stores:
- `payment_id` — unique payment identifier
- `owner_id` — Telegram user ID
- `tier` — activated tier (pro/max)
- `method` — payment method (stars/yookassa)
- `amount`, `currency` — payment amount
- `status` — always "success" (only successful payments are recorded)
- `created_at` — ISO timestamp

### Error Recovery

- If payment succeeds but DB write fails → exception is logged with
  `[PAYMENT:WEBHOOK_ERROR]`, YooKassa will retry the webhook
- If subscription activation fails → same retry mechanism applies
- User always gets a message: success confirmation or error with support contact

---

## 54-FZ / Receipts (Чеки)

### Current Status: Code Ready, Needs Dashboard Activation

The code includes full receipt data in YooKassa payment requests when
`YOO_RECEIPT_ENABLED=1`:

```json
{
  "receipt": {
    "customer": {},
    "items": [{
      "description": "Подписка NeuroSMM Pro — 30 дней",
      "quantity": "1.00",
      "amount": {"value": "490.00", "currency": "RUB"},
      "vat_code": 1,
      "payment_subject": "service",
      "payment_mode": "full_payment"
    }]
  }
}
```

### What's Still Needed (YooKassa Dashboard)

To actually send fiscal receipts, you need **one of**:

1. **"Чеки от ЮKassa"** — YooKassa's managed receipt service
   - Enable at: YooKassa dashboard → Shop Settings → Чеки
   - No external cash register needed
   - YooKassa handles OFD integration

2. **Your own online cash register** (онлайн-касса)
   - Connected to an OFD (оператор фискальных данных)
   - Integrated with YooKassa

**Without this activation**, YooKassa will accept the `receipt` field but
will **not** generate a fiscal receipt. The code is ready — the dashboard
configuration is the remaining step.

### Configuration

```env
YOO_RECEIPT_ENABLED=1      # Enable receipt data in payments
YOO_VAT_CODE=1             # 1=НДС 20%, 2=10%, 3=0%, 4=Без НДС, 5=20/120, 6=10/110
```

---

## Smoke Test Scenario

### Prerequisites
1. Bot running with `PAYMENT_MODE=test`
2. `YOO_SHOP_ID` and `YOO_SECRET_KEY` (test key) configured
3. For Stars: use BotFather's test environment

### Test Steps

#### 1. Telegram Stars Payment
```
1. Send /tariffs to the bot
2. Click "⭐ Pro — 99 ⭐"
3. Telegram shows Stars invoice → approve
4. Bot responds: "🎉 Оплата прошла успешно! Тариф Pro ⭐ активирован"
5. Send /plan — should show "Ваш тариф: ⭐ Pro (до DD.MM.YYYY)"
```

**Verify in logs:**
```
[PAYMENT:INVOICE_CREATED] method=stars owner_id=... tier=pro amount=99 XTR
[PAYMENT:PRE_CHECKOUT] user_id=... payload=tier:pro currency=XTR
[PAYMENT:RECEIVED] method=stars owner_id=... tier=pro payment_id=...
[PAYMENT:SUBSCRIPTION_ACTIVATED] owner_id=... tier=pro expires_at=...
[PAYMENT:USER_NOTIFIED] method=stars owner_id=... tier=pro
```

#### 2. YooKassa Payment
```
1. Send /tariffs to the bot
2. Click "💳 Pro — 490 ₽"
3. Bot shows "💳 Оплатить 490 ₽" button → click
4. Complete payment on YooKassa test page (test card: 5555 5555 5555 4477)
5. Return to bot
6. Bot sends: "🎉 Оплата прошла успешно! Тариф Pro ⭐ активирован"
7. Send /plan — should show "Ваш тариф: ⭐ Pro (до DD.MM.YYYY)"
```

**Verify in logs:**
```
[PAYMENT:CREATING] method=yookassa owner_id=... tier=pro amount=490 RUB receipt=no
[PAYMENT:INVOICE_CREATED] method=yookassa owner_id=... tier=pro payment_id=...
[PAYMENT:RECEIVED] method=yookassa owner_id=... tier=pro payment_id=...
[PAYMENT:SUBSCRIPTION_ACTIVATED] owner_id=... tier=pro expires_at=...
[PAYMENT:WEBHOOK_SUCCESS] payment_id=... owner_id=... tier=pro
[PAYMENT:USER_NOTIFIED] owner_id=... tier=pro
```

#### 3. Idempotency Check
- Resend the same webhook payload → should see `[PAYMENT:WEBHOOK_DUPLICATE]`
- Subscription should not change (same expiry date)

#### 4. Verify DB Records
```sql
SELECT * FROM payment_events ORDER BY id DESC LIMIT 5;
SELECT * FROM user_subscriptions WHERE owner_id = <your_id>;
```

---

## Files Changed

| File | Changes |
|------|---------|
| `.env.example` | **New** — Complete configuration reference with payment section |
| `config.py` | Added `payment_mode`, `yoo_receipt_enabled`, `yoo_vat_code` |
| `db.py` | Added `payment_events` table + `record_payment_event()` + `is_payment_processed()` |
| `billing_service.py` | Idempotency, 54-FZ receipts, structured logging, `notify_user_payment_success()`, `_validate_payment_mode()` |
| `miniapp_server.py` | Webhook handler: idempotency, user notification, structured logging |
| `handlers_private_messages.py` | Stars handler: payment_id tracking, structured logging, duplicate handling |
| `PAYMENT_FLOW.md` | **New** — This document |

## Root Causes Found & Fixed

1. **No idempotency** — duplicate webhooks could activate subscription multiple times
   → Fixed with `payment_events` UNIQUE index on `payment_id`

2. **No payment audit trail** — no way to verify who paid what
   → Fixed with `payment_events` table recording every payment

3. **No user notification after YooKassa** — user paid on external page, returned to
   bot, but got no confirmation message
   → Fixed with `notify_user_payment_success()` called from webhook

4. **No test/prod safety** — could accidentally use live keys in test mode
   → Fixed with `PAYMENT_MODE` + `_validate_payment_mode()` warnings

5. **No 54-FZ receipt preparation** — no receipt data in payment requests
   → Fixed with `_build_receipt_data()` and `YOO_RECEIPT_ENABLED` config

6. **Silent errors** — payment failures logged but not visible to users or in structured format
   → Fixed with `[PAYMENT:*]` structured log tags at every stage

7. **No `.env.example`** — developers had no reference for payment configuration
   → Fixed with comprehensive `.env.example`
