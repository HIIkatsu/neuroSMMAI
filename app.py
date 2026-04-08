import asyncio
import logging
import os
import random
from contextlib import suppress

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import MenuButtonCommands, MenuButtonWebApp, WebAppInfo

from config import load_config
from db import init_db
from scheduler_service import SchedulerService

from handlers_private import router as private_router
from handlers_admin import router as admin_router
from handlers_chat import router as chat_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

RETRY_MIN_SECONDS = 5
RETRY_MAX_SECONDS = 60


def _telegram_proxy_url(config) -> str | None:
    candidates = [
        getattr(config, "telegram_proxy_url", ""),
        os.getenv("TELEGRAM_PROXY_URL"),
        os.getenv("HTTPS_PROXY"),
        os.getenv("https_proxy"),
        os.getenv("HTTP_PROXY"),
        os.getenv("http_proxy"),
        os.getenv("ALL_PROXY"),
        os.getenv("all_proxy"),
    ]
    for value in candidates:
        raw = str(value or "").strip()
        if raw:
            return raw
    return None


def _make_dispatcher(config) -> Dispatcher:
    dp = Dispatcher(storage=MemoryStorage())
    dp["config"] = config
    dp.include_router(admin_router)
    dp.include_router(private_router)
    dp.include_router(chat_router)
    return dp


async def _configure_menu_button(bot: Bot, miniapp_url: str) -> None:
    try:
        if miniapp_url:
            await bot.set_chat_menu_button(
                menu_button=MenuButtonWebApp(
                    text="Открыть панель",
                    web_app=WebAppInfo(url=miniapp_url),
                )
            )
            logger.info("Mini App menu button configured")
        else:
            await bot.set_chat_menu_button(menu_button=MenuButtonCommands())
    except Exception as exc:
        logger.warning("Failed to configure Mini App menu button: %r", exc)


async def _close_bot(bot: Bot | None) -> None:
    if not bot:
        return
    with suppress(Exception):
        await bot.session.close()


async def _shutdown_scheduler(scheduler: SchedulerService | None) -> None:
    if not scheduler:
        return
    with suppress(Exception):
        if scheduler.scheduler.running:
            scheduler.scheduler.shutdown(wait=False)


async def _run_bot_once(config, dp: Dispatcher) -> None:
    proxy_url = _telegram_proxy_url(config)
    session = AiohttpSession(proxy=proxy_url) if proxy_url else AiohttpSession()
    bot: Bot | None = None
    scheduler: SchedulerService | None = None
    try:
        bot = Bot(
            token=config.bot_token,
            session=session,
            default=DefaultBotProperties(parse_mode=None),
        )
        bot._config = config
        bot._dp = dp

        scheduler = SchedulerService(bot=bot, tz=config.tz)
        scheduler.start()
        dp["scheduler"] = scheduler

        await _configure_menu_button(bot, getattr(config, "miniapp_url", "") or "")

        logger.info("BOT STARTED")
        await dp.start_polling(bot)
        logger.info("Polling stopped")
    finally:
        await _shutdown_scheduler(scheduler)
        await _close_bot(bot)


async def main() -> None:
    config = load_config()
    await init_db()

    dp = _make_dispatcher(config)
    attempt = 0

    while True:
        try:
            await _run_bot_once(config, dp)
            attempt = 0
            await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Bot stopped by keyboard interrupt")
            break
        except asyncio.CancelledError:
            logger.info("Bot task cancelled")
            raise
        except Exception:
            attempt += 1
            delay = min(RETRY_MAX_SECONDS, RETRY_MIN_SECONDS * (2 ** min(attempt - 1, 3)))
            jitter = random.uniform(0, delay * 0.15)
            total_delay = delay + jitter
            logger.exception("Fatal bot error, restarting polling in %.1fs", total_delay)
            await asyncio.sleep(total_delay)


if __name__ == "__main__":
    asyncio.run(main())
