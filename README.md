# neuroSMMAI

AI-powered Telegram bot with an integrated Mini App for SMM content management.

## Architecture

The project consists of two independent runtime processes:

| Process | Entry point | Description |
|---------|-------------|-------------|
| **Telegram bot** | `app.py` | aiogram 3 polling daemon — handles commands, FSM dialogs, scheduled posting |
| **Mini App server** | `miniapp_server.py` | FastAPI/uvicorn ASGI server — serves the web UI and REST API |

Both processes share the same codebase, database (`bot.db`, SQLite via aiosqlite),
and `.env` configuration.

## Project structure

```
.
├── app.py                    # Telegram bot entry point (polling)
├── miniapp_server.py         # FastAPI entry point (uvicorn)
├── config.py                 # Centralised configuration from .env
├── db.py                     # Database schema & helpers (aiosqlite)
│
├── miniapp/
│   └── index.html            # Mini App SPA entry (served by FastAPI)
├── app.js                    # Frontend bundle (served at /app.js)
├── styles.css                # Frontend styles (served at /styles.css)
│
├── handlers_private*.py      # Bot private-chat handlers
├── handlers_admin.py         # Bot admin handlers
├── handlers_chat.py          # Bot group-chat handlers
│
├── miniapp_routes_core.py    # /api/* core routes (bootstrap, settings)
├── miniapp_routes_content.py # /api/* content routes (plans, posts)
├── miniapp_routes_media.py   # /api/* media routes (images, uploads)
├── miniapp_*.py              # Mini App services & schemas
│
├── ai_client.py              # OpenAI chat completions wrapper
├── ai_image_generator.py     # Image generation (DALL-E / Flux)
├── billing_service.py        # Payments (Telegram Stars + YooKassa)
├── scheduler_service.py      # APScheduler background jobs
├── auth.py                   # Telegram HMAC & JWT auth
│
├── deploy/
│   └── neurosmm.service      # systemd unit (uvicorn behind nginx)
├── nginx_security_snippet.conf # nginx include: proxy headers, rate-limiting
├── DEPLOYMENT.md             # Deployment guide (nginx ↔ uvicorn)
├── PAYMENT_FLOW.md           # Payment integration documentation
├── requirements.txt          # Python dependencies
└── tests/                    # Automated tests (auth, web smoke)
```

### Files that live only on the server (not in git)

| Path | Purpose |
|------|---------|
| `.env` | Secrets & configuration |
| `bot.db`, `bot.db-shm`, `bot.db-wal` | SQLite database |
| `venv/` | Python virtual environment |
| `uploads/` | User-uploaded media |
| `generated_images/` | AI-generated images |

## Quick start

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in your tokens
# Start both processes:
python app.py &                          # Telegram bot
uvicorn miniapp_server:app --port 8000   # Mini App server
```

## Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for the production setup with nginx,
systemd, and proxy headers.

## Tests

```bash
pip install pytest httpx
pytest tests/
``` 
