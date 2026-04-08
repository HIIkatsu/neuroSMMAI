# Deployment: nginx + uvicorn (real client IP)

## Problem

When uvicorn runs behind nginx, `request.client.host` returns `127.0.0.1`
(the nginx loopback address) instead of the real visitor IP.

This breaks two things:
- **Rate limiting** — all users share a single bucket.
- **YooKassa webhook IP whitelist** — all webhooks get 403 (source IP is
  `127.0.0.1`, not a YooKassa range).

Auth (Telegram HMAC + owner_id checks) is **not affected** — it never uses IP.

## Fix (two parts)

### 1. nginx — pass real IP

Add these headers inside every `location` block that proxies to uvicorn.
See `nginx_security_snippet.conf` for a ready-to-include snippet.

```nginx
proxy_set_header X-Forwarded-For   $remote_addr;
proxy_set_header X-Real-IP         $remote_addr;
proxy_set_header Host              $host;
proxy_set_header X-Forwarded-Proto $scheme;
```

### 2. uvicorn — trust the proxy

Start uvicorn with two extra flags:

```bash
uvicorn miniapp_server:app \
    --host 127.0.0.1 \
    --port 8000 \
    --proxy-headers \
    --forwarded-allow-ips="127.0.0.1"
```

- `--proxy-headers` tells uvicorn to read `X-Forwarded-For`.
- `--forwarded-allow-ips="127.0.0.1"` means uvicorn only trusts that
  header when the TCP peer is nginx (loopback). External clients cannot
  spoof the header.

If you use **gunicorn** as a process manager:

```bash
gunicorn miniapp_server:app \
    -k uvicorn.workers.UvicornWorker \
    --forwarded-allow-ips="127.0.0.1"
```

### 3. systemd

An example unit file is in `deploy/neurosmm.service`. Copy it:

```bash
sudo cp deploy/neurosmm.service /etc/systemd/system/neurosmm.service
# Edit WorkingDirectory / User / paths to match your setup
sudo systemctl daemon-reload
sudo systemctl restart neurosmm
```

## Verification

After restarting both nginx and uvicorn:

```bash
# From your local machine (replace with your server URL):
curl -s https://your-domain.com/api/bootstrap | head

# In uvicorn logs you should see your real public IP, not 127.0.0.1.
# Also test YooKassa webhooks from the YooKassa dashboard — they should
# no longer return 403.
```
