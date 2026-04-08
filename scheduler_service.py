from __future__ import annotations

import logging
import random
import types
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta

from aiogram import Bot
from aiogram.types import FSInputFile

from db import (
    list_schedules,
    list_plan_items_active_not_posted,
    get_setting,
    get_posts_enabled,
    mark_plan_posted,
    log_post,
    list_owner_ids,
    log_news,
    set_setting,
    get_draft,
    get_recent_channel_history,
    create_draft,
    expire_overdue_subscriptions,
    get_user_subscription,
    get_last_post_time,
    TIER_FREE,
    TIER_PRO,
    TIER_MAX,
)
import db  # For get_channel_settings, list_channel_profiles
from content import generate_post_text, _remove_fabricated_refs  # kept for backward compatibility / news hooks
from actions import publish_draft, resolve_post_image, generate_post_payload
from image_search import validate_image_for_autopost
from news_service import fetch_latest_news, fetch_news_candidates, build_news_post, build_news_source_meta, is_source_confident
from media_lifecycle import cleanup_temp_media
from safe_send import safe_send as _safe_send_dm

logger = logging.getLogger(__name__)

DAY_MAP = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}

# Максимальный размер кешей дедупликации (не растут бесконечно)
_DEDUP_MAX_SIZE = 2000

# ---------------------------------------------------------------------------
# Autopost pacing: minimum gap between consecutive autopost publications.
# Prevents spammy bursts even when multiple schedule slots fire close together.
# ---------------------------------------------------------------------------
_MIN_POST_GAP_MINUTES = 45  # do not publish if last post was < 45 min ago

# Per-channel recently used angle indices to avoid repeating the same angle
# back-to-back.  Maps (owner_id, channel_target) → list of last N used template indices.
_CHANNEL_RECENT_ANGLES: dict[tuple[int, str], list[int]] = {}
_RECENT_ANGLES_LIMIT = 6

# Templates for varied post angle prompts; {topic} is substituted at runtime.
# All templates are channel-agnostic — no niche-specific language.
_ANGLE_PROMPT_TEMPLATES = [
    "раскрой тему «{topic}» через конкретную ситуацию из жизни аудитории",
    "практическая польза: {topic}",
    "распространённая ошибка в теме «{topic}»",
    "как «{topic}» помогает решить конкретную задачу",
    "неочевидный совет по теме «{topic}»",
    "новый взгляд на тему «{topic}»",
    "развенчание мифа о теме «{topic}»",
    "конкретный кейс или пример по теме «{topic}»",
    "тонкости и нюансы: что часто упускают в теме «{topic}»",
    "вопрос от подписчика по теме «{topic}»: отвечаю по делу",
]


def _pick_angle_prompt(
    topic: str,
    owner_id: int = 0,
    *,
    channel_target: str = "",
    recent_titles: list[str] | None = None,
    audience: str = "",
    constraints: str = "",
    exclusions: str = "",
) -> str:
    """Return a varied angle prompt, avoiding recently used angles per channel.

    Enriches the bare angle template with channel context (audience, constraints,
    exclusions, recent titles) so the LLM has better guidance for topic selection.
    """
    key = (owner_id, channel_target or "")
    recent = _CHANNEL_RECENT_ANGLES.get(key, [])
    available = [i for i in range(len(_ANGLE_PROMPT_TEMPLATES)) if i not in recent]
    if not available:
        # All angles exhausted — reset and pick any
        available = list(range(len(_ANGLE_PROMPT_TEMPLATES)))
        recent = []

    idx = random.choice(available)
    # Track the chosen angle
    recent.append(idx)
    if len(recent) > _RECENT_ANGLES_LIMIT:
        recent = recent[-_RECENT_ANGLES_LIMIT:]
    _CHANNEL_RECENT_ANGLES[key] = recent

    base_prompt = _ANGLE_PROMPT_TEMPLATES[idx].format(topic=topic)

    # Enrich with channel context for smarter topic selection
    parts: list[str] = [base_prompt]

    if recent_titles:
        # Prevent the LLM from regenerating similar topics
        titles_block = "; ".join(t[:80] for t in recent_titles[:8])
        parts.append(f"Не повторяй темы недавних постов: [{titles_block}]")

    if audience and audience.strip():
        parts.append(f"Аудитория канала: {audience.strip()[:120]}")

    if constraints and constraints.strip():
        parts.append(f"Ограничения: {constraints.strip()[:120]}")

    if exclusions and exclusions.strip():
        parts.append(f"Исключить: {exclusions.strip()[:120]}")

    return ". ".join(parts)


def _media_is_local(media_ref: str) -> bool:
    value = (media_ref or "").strip()
    return (
        value.startswith("local:")
        or value.startswith("/")
        or value.startswith("./")
        or value.startswith("../")
    )


def _make_per_owner_cfg(api_key: str, model: str, base_url, global_cfg) -> "types.SimpleNamespace":
    """Build a minimal config-like object for generate_post_payload.

    Combines per-owner api_key/model (resolved from owner settings) with
    global config attributes.  Used by scheduler jobs so that per-owner
    API settings are honoured (same as the editor path).
    """
    return types.SimpleNamespace(
        openrouter_api_key=api_key,
        openrouter_model=model,
        openrouter_base_url=base_url,
        max_active_drafts_per_user=getattr(global_cfg, "max_active_drafts_per_user", 25),
    )


async def _load_recent_posts_and_plan(owner_id: int) -> tuple[list[str], list[str]]:
    """Load recent channel posts and plan items for anti-repeat logic in generation."""
    try:
        history = await get_recent_channel_history(owner_id=owner_id, limit=12)
        recent_posts = history.get("recent_posts", []) + history.get("recent_drafts", [])
        recent_plan = history.get("recent_plan", [])
        return recent_posts[:10], recent_plan[:10]
    except Exception:
        logger.warning("_load_recent_posts_and_plan failed for owner_id=%s", owner_id)
        return [], []


async def _is_paid_tier(owner_id: int) -> bool:
    """Return True if the user is on a paid (Pro/Max) tier or has an active trial."""
    sub = await get_user_subscription(owner_id)
    tier = sub.get("subscription_tier", TIER_FREE)
    if tier in (TIER_PRO, TIER_MAX):
        return True
    trial_ends = sub.get("trial_ends_at") or ""
    if trial_ends:
        try:
            if datetime.fromisoformat(trial_ends) > datetime.utcnow():
                return True
        except Exception:
            pass
    return False


async def _check_post_cooldown(owner_id: int, tz: str, *, channel_target: str = "") -> bool:
    """Return True if enough time has passed since the last post (pacing OK).

    If *channel_target* is provided, checks per-channel cooldown (recommended).
    Otherwise falls back to owner-level cooldown.
    """
    last_ts = await get_last_post_time(owner_id, channel_target=channel_target)
    if not last_ts:
        return True
    try:
        last_dt = datetime.fromisoformat(last_ts)
        if last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=ZoneInfo(tz))
        now = datetime.now(ZoneInfo(tz))
        gap = now - last_dt
        if gap < timedelta(minutes=_MIN_POST_GAP_MINUTES):
            logger.info(
                "_check_post_cooldown: too soon for owner_id=%s channel=%s (gap=%s min, min=%s)",
                owner_id, channel_target or "(any)", int(gap.total_seconds() / 60), _MIN_POST_GAP_MINUTES,
            )
            return False
    except Exception as exc:
        logger.warning("_check_post_cooldown: failed to parse timestamp %r for owner_id=%s: %s", last_ts, owner_id, exc)
    return True


def _local_media_path(media_ref: str) -> str:
    value = (media_ref or "").strip()
    if value.startswith("local:"):
        return value[6:]
    return value


class SchedulerService:
    def __init__(self, bot: Bot, tz: str):
        self.bot = bot
        # FIX: используем tz из конфига, не хардкодим Moscow
        self.tz = tz or "Europe/Moscow"
        self.scheduler = AsyncIOScheduler(timezone=ZoneInfo(self.tz))
        self._last_schedule_run: dict[tuple[int, int], str] = {}
        self._last_plan_run: dict[int, str] = {}
        self._jobs_synced = False

    def _ensure_started(self) -> None:
        if not self.scheduler.running:
            self.scheduler.start()

    def _sync_jobs(self) -> None:
        self.scheduler.add_job(
            self._job_news_tick,
            "interval",
            minutes=30,
            id="news_tick",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )
        self.scheduler.add_job(
            self._job_schedule_tick,
            "interval",
            minutes=1,
            id="schedule_tick",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )
        self.scheduler.add_job(
            self._job_plan_tick,
            "interval",
            minutes=1,
            id="plan_tick",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )
        self.scheduler.add_job(
            self._job_cleanup_media,
            "interval",
            hours=1,
            id="cleanup_media",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )
        self.scheduler.add_job(
            self._job_news_sniper_tick,
            "interval",
            hours=3,
            id="news_sniper_tick",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )
        self.scheduler.add_job(
            self._job_expire_subscriptions,
            "interval",
            hours=6,
            id="expire_subscriptions",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )
        self._jobs_synced = True

    def start(self) -> None:
        self._ensure_started()
        if not self._jobs_synced:
            self._sync_jobs()

    async def rebuild_jobs(self) -> None:
        self._ensure_started()
        self._sync_jobs()

    async def stop(self) -> None:
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
        self._jobs_synced = False

    def _trim_dedup_caches(self) -> None:
        """Предотвращает бесконечный рост in-memory дедуп-кешей."""
        if len(self._last_schedule_run) > _DEDUP_MAX_SIZE:
            # Удаляем первую половину (FIFO-like)
            keys = list(self._last_schedule_run.keys())
            for k in keys[: _DEDUP_MAX_SIZE // 2]:
                del self._last_schedule_run[k]
        if len(self._last_plan_run) > _DEDUP_MAX_SIZE:
            keys = list(self._last_plan_run.keys())
            for k in keys[: _DEDUP_MAX_SIZE // 2]:
                del self._last_plan_run[k]

    async def _send_with_optional_photo(self, channel: str, text: str, image_ref: str, *, is_autopost: bool = False):
        from actions import TELEGRAM_CAPTION_LIMIT, _split_media_caption_and_body, _split_text_chunks
        from content import enforce_autopost_budget, AUTOPOST_CAPTION_BUDGET, AUTOPOST_TEXT_BUDGET
        from safe_send import safe_send as _safe_send_text

        # For autopost: enforce single-message budget as a final safety net
        if is_autopost:
            budget = AUTOPOST_CAPTION_BUDGET if image_ref else AUTOPOST_TEXT_BUDGET
            if len(text) > budget:
                text = text[:budget].rstrip() + "…"

        if image_ref:
            # Single-message-first: send as one photo+caption when text fits
            if len(text) <= TELEGRAM_CAPTION_LIMIT:
                photo = FSInputFile(_local_media_path(image_ref)) if _media_is_local(image_ref) else image_ref
                msg = await self.bot.send_photo(chat_id=channel, photo=photo, caption=text)
                return msg, "photo"

            if is_autopost:
                # Autopost: NEVER split — hard-trim to caption limit (leave 1 char for ellipsis)
                text = text[:TELEGRAM_CAPTION_LIMIT - 1].rstrip() + "…"
                photo = FSInputFile(_local_media_path(image_ref)) if _media_is_local(image_ref) else image_ref
                msg = await self.bot.send_photo(chat_id=channel, photo=photo, caption=text)
                return msg, "photo"

            # Controlled fallback: split at paragraph/sentence boundary
            caption, body_chunks = _split_media_caption_and_body(text)
            photo = FSInputFile(_local_media_path(image_ref)) if _media_is_local(image_ref) else image_ref
            msg = await self.bot.send_photo(chat_id=channel, photo=photo, caption=caption)
            for chunk in body_chunks:
                await _safe_send_text(self.bot, channel, chunk)
            return msg, "photo"

        # Text-only: respect 4096 limit
        if is_autopost:
            # Single message only
            msg = await self.bot.send_message(chat_id=channel, text=text[:4096])
            return msg, "text"

        chunks = _split_text_chunks(text, 4096)
        if not chunks:
            logger.warning("_send_with_optional_photo: empty text after split, channel=%s", channel)
            msg = await self.bot.send_message(chat_id=channel, text="(пустой пост)")
            return msg, "text"
        msg = await self.bot.send_message(chat_id=channel, text=chunks[0])
        for chunk in chunks[1:]:
            await _safe_send_text(self.bot, channel, chunk)
        return msg, "text"

    async def _job_post_regular(self, owner_id: int = 0, channel_profile_id: int = 0):
        """Публикует плановый пост по расписанию для конкретного канала."""
        try:
            # Autoposting is a Pro+ feature — skip if user is on free tier
            if not await _is_paid_tier(owner_id):
                logger.info("_job_post_regular skipped owner_id=%s: free tier", owner_id)
                return

            # Load channel-specific settings
            ch_settings = await db.get_channel_settings(owner_id, channel_profile_id=channel_profile_id or None)

            posts_enabled = (ch_settings.get("posts_enabled") or "0").strip() not in ("0", "false", "no")
            if not posts_enabled:
                return
            posting_mode = (ch_settings.get("posting_mode") or "manual").strip().lower()
            if posting_mode not in ("both", "posts"):
                return
            channel = (ch_settings.get("channel_target") or "").strip()
            if not channel:
                return
            if not await _check_post_cooldown(owner_id, self.tz, channel_target=channel):
                logger.info("_job_post_regular skipped owner_id=%s channel=%s: cooldown active", owner_id, channel)
                return

            cfg = getattr(self.bot, "_config", None)
            api_key = (await get_setting("openrouter_api_key", owner_id=owner_id)) or getattr(cfg, "openrouter_api_key", "")
            model = (await get_setting("openrouter_model", owner_id=owner_id)) or getattr(cfg, "openrouter_model", "gpt-4o-mini")
            base_url = getattr(cfg, "openrouter_base_url", None)

            # FIX: не используем хардкоженный fallback — если топика нет, пост не генерируем
            topic = (ch_settings.get("topic") or "").strip()
            if not topic:
                logger.warning("_job_post_regular skipped owner_id=%s: topic not configured", owner_id)
                return
            if not api_key:
                logger.warning("_job_post_regular skipped owner_id=%s: api_key not configured", owner_id)
                return

            # Fetch recent post titles for diversity guidance
            recent_titles: list[str] = []
            try:
                recent_titles = await db.get_recent_post_topics(owner_id=owner_id, limit=10)
            except Exception:
                pass

            # Respect per-channel auto_image setting (default: enabled)
            auto_image = (ch_settings.get("auto_image") or "1").strip() not in ("0", "false", "no")

            # --- Autopost retry loop: up to 3 attempts with varied angles ---
            _AUTOPOST_MAX_ATTEMPTS = 3
            payload_data = None
            last_error = None
            for attempt in range(1, _AUTOPOST_MAX_ATTEMPTS + 1):
                # Vary the prompt angle each time to avoid monotonous posts
                prompt = _pick_angle_prompt(
                    topic,
                    owner_id=owner_id,
                    channel_target=channel,
                    recent_titles=recent_titles,
                    audience=(ch_settings.get("channel_audience") or ""),
                    constraints=(ch_settings.get("content_constraints") or ""),
                    exclusions=(ch_settings.get("content_exclusions") or ""),
                )
                logger.info(
                    "AUTOPOST_ENTRY path=autopost owner_id=%s channel_profile_id=%s attempt=%d/%d topic=%r prompt=%r",
                    owner_id, channel_profile_id, attempt, _AUTOPOST_MAX_ATTEMPTS, topic[:80], prompt[:80],
                )
                try:
                    payload_data = await generate_post_payload(
                        _make_per_owner_cfg(api_key, model, base_url, cfg),
                        prompt,
                        owner_id=owner_id,
                        force_image=auto_image,
                        generation_path="autopost",
                    )
                    text = payload_data.get("text") or ""
                    if text:
                        break  # Success — exit retry loop
                    logger.warning("_job_post_regular: empty text attempt=%d owner_id=%s", attempt, owner_id)
                    payload_data = None
                except RuntimeError as gen_err:
                    last_error = gen_err
                    logger.warning(
                        "_job_post_regular: quality gate rejected attempt=%d/%d owner_id=%s reason=%s",
                        attempt, _AUTOPOST_MAX_ATTEMPTS, owner_id, str(gen_err)[:300],
                    )
                    payload_data = None
                    continue

            if not payload_data or not (payload_data.get("text") or ""):
                logger.warning(
                    "_job_post_regular: all %d attempts exhausted owner_id=%s topic=%r last_error=%s — skipping publication",
                    _AUTOPOST_MAX_ATTEMPTS, owner_id, topic[:80],
                    str(last_error)[:300] if last_error else "empty text",
                )
                return

            text = payload_data.get("text") or ""
            image_ref = payload_data.get("media_ref") or ""

            msg, content_type = await self._send_with_optional_photo(channel, text, image_ref, is_autopost=True)
            await log_post(
                owner_id=owner_id,
                channel_target=channel,
                content_type=content_type,
                text=text,
                topic=topic,
                file_id=image_ref,
                telegram_message_id=getattr(msg, "message_id", 0),
            )
        except Exception:
            logger.exception("_job_post_regular failed owner_id=%s channel_profile_id=%s", owner_id, channel_profile_id)

    async def _job_post_plan_item(
        self,
        item_id: int,
        owner_id: int = 0,
        channel_profile_id: int = 0,
        kind: str = "",
        payload: str = "",
        prompt: str = "",
        topic_override: str = "",
    ):
        """Публикует пост из контент-плана (channel-scoped)."""
        try:
            # Load channel-specific settings (falls back to owner-level if no profile)
            ch_settings = await db.get_channel_settings(owner_id, channel_profile_id=channel_profile_id or None)

            posts_enabled = (ch_settings.get("posts_enabled") or "0").strip() not in ("0", "false", "no")
            if not posts_enabled:
                return
            posting_mode = (ch_settings.get("posting_mode") or "manual").strip().lower()
            if posting_mode not in ("both", "posts"):
                return
            channel = (ch_settings.get("channel_target") or "").strip()
            if not channel:
                return

            if kind == "draft":
                draft_id = int(payload)
                draft = await get_draft(draft_id, owner_id=owner_id)
                if not draft:
                    logger.warning("_job_post_plan_item: draft %s not found owner_id=%s", draft_id, owner_id)
                    return
                result = await publish_draft(self.bot, draft, owner_id=owner_id)
                if result.ok:
                    await mark_plan_posted(item_id, owner_id=owner_id)
                else:
                    logger.warning("_job_post_plan_item: draft publish failed item_id=%s err=%s", item_id, result)
                return

            cfg = getattr(self.bot, "_config", None)
            api_key = (await get_setting("openrouter_api_key", owner_id=owner_id)) or getattr(cfg, "openrouter_api_key", "")
            model = (await get_setting("openrouter_model", owner_id=owner_id)) or getattr(cfg, "openrouter_model", "gpt-4o-mini")
            base_url = getattr(cfg, "openrouter_base_url", None)

            # Use channel-level topic, fall back to topic_override → owner-level
            topic = (topic_override or (ch_settings.get("topic") or "") or "").strip()
            if not topic:
                logger.warning("_job_post_plan_item skipped item_id=%s owner_id=%s cpid=%s: topic not configured", item_id, owner_id, channel_profile_id)
                return
            if not api_key:
                logger.warning("_job_post_plan_item skipped item_id=%s owner_id=%s: api_key not configured", item_id, owner_id)
                return

            prompt_text = (prompt or payload or "").strip()
            logger.info(
                "AUTOPOST_ENTRY path=plan_item owner_id=%s channel_profile_id=%s item_id=%s literal_topic=%r prompt=%r",
                owner_id, channel_profile_id, item_id, topic[:80], prompt_text[:80],
            )

            # UNIFIED PATH: use generate_post_payload — same core as editor path.
            # Respect per-channel auto_image setting (default: enabled)
            auto_image = (ch_settings.get("auto_image") or "1").strip() not in ("0", "false", "no")
            payload_data = await generate_post_payload(
                _make_per_owner_cfg(api_key, model, base_url, cfg),
                prompt_text,
                owner_id=owner_id,
                force_image=auto_image,
                generation_path="autopost",
            )
            text = payload_data.get("text") or ""
            image_ref = payload_data.get("media_ref") or ""

            if not text:
                logger.warning("_job_post_plan_item: empty text generated item_id=%s owner_id=%s", item_id, owner_id)
                return

            msg, content_type = await self._send_with_optional_photo(channel, text, image_ref, is_autopost=True)
            await log_post(
                owner_id=owner_id,
                channel_target=channel,
                content_type=content_type,
                text=text,
                prompt=prompt_text,
                topic=topic,
                file_id=image_ref,
                telegram_message_id=getattr(msg, "message_id", 0),
            )
            await mark_plan_posted(item_id, owner_id=owner_id)
        except Exception:
            logger.exception("_job_post_plan_item failed item_id=%s owner_id=%s cpid=%s", item_id, owner_id, channel_profile_id)

    async def _job_schedule_tick(self):
        try:
            # FIX: используем self.tz вместо захардкоженного MOSCOW_TZ
            now = datetime.now(ZoneInfo(self.tz))
            current_hhmm = now.strftime("%H:%M")
            weekday = now.weekday()
            self._trim_dedup_caches()
            for owner_id in await list_owner_ids():
                try:
                    schedules = await list_schedules(owner_id=owner_id)
                    for row in schedules:
                        if not int(row.get("enabled", 1)):
                            continue
                        if str(row.get("time_hhmm", "")) != current_hhmm:
                            continue
                        days = str(row.get("days", "*")).strip().lower()
                        allowed = set(DAY_MAP[p] for p in days.split(",") if p in DAY_MAP) if days != "*" else set(range(7))
                        if weekday not in allowed:
                            continue
                        dedupe_key = (int(owner_id), int(row.get("id", 0)))
                        minute_key = now.strftime("%Y-%m-%d %H:%M")
                        if self._last_schedule_run.get(dedupe_key) == minute_key:
                            continue
                        self._last_schedule_run[dedupe_key] = minute_key
                        # Pass channel_profile_id so the post uses channel-specific settings
                        cpid = int(row.get("channel_profile_id", 0))
                        await self._job_post_regular(owner_id=owner_id, channel_profile_id=cpid)
                except Exception:
                    logger.exception("_job_schedule_tick inner error owner_id=%s", owner_id)
        except Exception:
            logger.exception("_job_schedule_tick outer error")

    async def _job_plan_tick(self):
        try:
            # FIX: используем self.tz
            now = datetime.now(ZoneInfo(self.tz))
            items = await list_plan_items_active_not_posted(owner_id=None)
            for it in items:
                try:
                    dt_raw = str(it.get("dt") or "").strip().replace("T", " ")
                    try:
                        due_at = datetime.strptime(dt_raw[:16], "%Y-%m-%d %H:%M").replace(tzinfo=ZoneInfo(self.tz))
                    except Exception:
                        continue
                    if due_at > now:
                        continue
                    if now - due_at > timedelta(minutes=3):
                        continue
                    minute_key = now.strftime("%Y-%m-%d %H:%M")
                    plan_id = int(it.get("id") or 0)
                    if self._last_plan_run.get(plan_id) == minute_key:
                        continue
                    self._last_plan_run[plan_id] = minute_key

                    # Cooldown check: respect per-channel post gap
                    item_owner = int(it.get("owner_id") or 0)
                    cpid = int(it.get("channel_profile_id") or 0)
                    ch_settings = await db.get_channel_settings(item_owner, channel_profile_id=cpid or None)
                    ch_target = (ch_settings.get("channel_target") or "").strip()
                    if not await _check_post_cooldown(item_owner, self.tz, channel_target=ch_target):
                        logger.info("_job_plan_tick: cooldown active, skipping plan_id=%s owner=%s channel=%s", plan_id, item_owner, ch_target)
                        continue

                    await self._job_post_plan_item(
                        item_id=int(it["id"]),
                        owner_id=item_owner,
                        channel_profile_id=cpid,
                        kind=str(it.get("kind") or ""),
                        payload=str(it.get("payload") or ""),
                        prompt=str(it.get("prompt") or ""),
                        topic_override=str(it.get("topic") or ""),
                    )
                except Exception:
                    logger.exception("_job_plan_tick inner error item=%s", it.get("id"))
        except Exception:
            logger.exception("_job_plan_tick outer error")

    async def _job_cleanup_media(self):
        cfg = getattr(self.bot, "_config", None)
        if not cfg:
            return
        try:
            await cleanup_temp_media(
                uploads_dir=(__import__("pathlib").Path(__file__).resolve().parent / "uploads"),
                generated_dir=(__import__("pathlib").Path(__file__).resolve().parent / "generated_images"),
                config=cfg,
            )
        except Exception:
            logger.exception("_job_cleanup_media error")

    async def _job_news_tick(self):
        try:
            owner_ids = await list_owner_ids()
            cfg = getattr(self.bot, "_config", None)
            for owner_id in owner_ids:
                try:
                    # News auto-posting is a Pro+ feature
                    if not await _is_paid_tier(owner_id):
                        continue

                    # Iterate all channels for this owner, not just active
                    channels = await db.list_channel_profiles(owner_id=owner_id)
                    if not channels:
                        continue

                    for ch_profile in channels:
                        try:
                            ch_settings = await db.get_channel_settings(owner_id, channel_profile_id=int(ch_profile.get("id", 0)))
                            enabled = (ch_settings.get("news_enabled") or "0").strip() not in ("0", "false", "False")
                            if not enabled:
                                continue
                            channel = (ch_settings.get("channel_target") or "").strip()
                            if not channel:
                                continue
                            interval_h = int((ch_settings.get("news_interval_hours") or "6").strip() or "6")
                            # Night guard: do not autopost news between 23:00 and 08:00 (UTC+3)
                            now_local = datetime.now(ZoneInfo(self.tz))
                            if now_local.hour >= 23 or now_local.hour < 8:
                                logger.info("_job_news_tick: skipping night hours owner_id=%s hour=%s", owner_id, now_local.hour)
                                continue
                            # Pacing: skip if last post (any type) was too recent — per-channel cooldown
                            if not await _check_post_cooldown(owner_id, self.tz, channel_target=channel):
                                logger.info("_job_news_tick: cooldown active owner_id=%s channel=%s", owner_id, channel)
                                continue
                            # Per-channel news interval: use channel-scoped setting key
                            _news_ts_key = f"news_last_posted_at:{channel}"
                            last_ts = (await get_setting(_news_ts_key, owner_id=owner_id) or "").strip()
                            if not last_ts:
                                # Fallback to owner-level for migration compatibility
                                last_ts = (await get_setting("news_last_posted_at", owner_id=owner_id) or "").strip()
                            if last_ts:
                                try:
                                    last_dt = datetime.fromisoformat(last_ts)
                                    if datetime.now(ZoneInfo(self.tz)) - last_dt.replace(tzinfo=ZoneInfo(self.tz)) < timedelta(hours=interval_h):
                                        continue
                                except Exception:
                                    pass
                            item = await fetch_latest_news(owner_id=owner_id, channel_target=channel, channel_profile_id=int(ch_profile.get("id", 0)))
                            if not item or not cfg or not getattr(cfg, "openrouter_api_key", ""):
                                continue
                            text = await build_news_post(cfg, item, owner_id=owner_id)
                            if not text:
                                logger.warning("_job_news_tick: empty text owner_id=%s channel=%s", owner_id, channel)
                                continue
                            # News text quality gate: reject low-quality news posts before publishing
                            from content import assess_text_quality, NEWS_MIN_QUALITY_SCORE
                            _news_q_score, _news_q_reasons, _news_q_dims = assess_text_quality(
                                "", text, "",
                                channel_topic=item.get("topic") or "",
                                requested=item.get("title") or "",
                            )
                            if _news_q_score < NEWS_MIN_QUALITY_SCORE:
                                logger.warning(
                                    "_job_news_tick: news text quality gate REJECT score=%d min=%d dims=%s reasons=%s owner_id=%s title=%r",
                                    _news_q_score, NEWS_MIN_QUALITY_SCORE,
                                    " ".join(f"{k}={v}" for k, v in _news_q_dims.items()),
                                    "; ".join(_news_q_reasons[:4]),
                                    owner_id, (item.get("title") or "")[:80],
                                )
                                continue
                            # Enforce single-message budget: AUTOPOST_TEXT_BUDGET is the max
                            # character count for a text-only Telegram message (under 4096).
                            from content import AUTOPOST_TEXT_BUDGET
                            if len(text) > AUTOPOST_TEXT_BUDGET:
                                text = text[:AUTOPOST_TEXT_BUDGET].rsplit("\n", 1)[0].strip()
                            # Respect per-channel auto_image setting for news posts
                            auto_image = (ch_settings.get("auto_image") or "1").strip() not in ("0", "false", "no")
                            image_ref = ""
                            if auto_image:
                                image_ref = await resolve_post_image(
                                    item.get("title") or item.get("topic") or "",
                                    owner_id=owner_id,
                                    ai_prompt=item.get("title") or "",
                                    topic=item.get("topic") or "",
                                    post_text=text,
                                    config=cfg,
                                    raw_user_query=item.get("title") or item.get("topic") or "",
                                )
                                # Autopost quality gate: reject semantically irrelevant images
                                if image_ref and not validate_image_for_autopost(
                                    image_ref,
                                    topic=item.get("topic") or "",
                                    prompt=item.get("title") or "",
                                    post_text=text,
                                ):
                                    logger.warning("_job_news_tick: image rejected by quality gate owner_id=%s url=%r", owner_id, image_ref[:80])
                                    image_ref = ""
                            msg, content_type = await self._send_with_optional_photo(channel, text, image_ref, is_autopost=True)
                            await log_post(
                                owner_id=owner_id,
                                channel_target=channel,
                                content_type=content_type,
                                text=text,
                                prompt=item.get("title", ""),
                                topic=item.get("topic", ""),
                                file_id=image_ref,
                                telegram_message_id=getattr(msg, "message_id", 0),
                            )
                            await log_news(item["link"], item.get("title", ""), owner_id=owner_id, channel_target=channel)
                            await set_setting(
                                _news_ts_key,
                                datetime.now(ZoneInfo(self.tz)).isoformat(timespec="seconds"),
                                owner_id=owner_id,
                            )
                        except Exception:
                            logger.exception("_job_news_tick channel error owner_id=%s ch=%s", owner_id, ch_profile.get("channel_target"))
                except Exception:
                    logger.exception("_job_news_tick inner error owner_id=%s", owner_id)
        except Exception:
            logger.exception("_job_news_tick outer error")

    async def _job_news_sniper_tick(self):
        """News Sniper: proactively find urgent/breaking news, create a draft and DM the user."""
        try:
            owner_ids = await list_owner_ids()
            cfg = getattr(self.bot, "_config", None)
            for owner_id in owner_ids:
                try:
                    # News Sniper is a Pro+ feature
                    if not await _is_paid_tier(owner_id):
                        continue

                    # Iterate all channels for this owner
                    channels = await db.list_channel_profiles(owner_id=owner_id)
                    if not channels:
                        continue

                    for ch_profile in channels:
                        try:
                            ch_settings = await db.get_channel_settings(owner_id, channel_profile_id=int(ch_profile.get("id", 0)))
                            enabled = (ch_settings.get("news_enabled") or "0").strip() not in ("0", "false", "False")
                            if not enabled:
                                continue

                            channel_target = (ch_settings.get("channel_target") or "").strip()
                            if not channel_target:
                                continue

                            # Night guard: don't bother users between 23:00 and 08:00
                            now_local = datetime.now(ZoneInfo(self.tz))
                            if now_local.hour >= 23 or now_local.hour < 8:
                                continue

                            # Throttle: check when we last ran the sniper for this channel
                            _sniper_ts_key = f"news_sniper_last_at:{channel_target}"
                            last_ts = (await get_setting(_sniper_ts_key, owner_id=owner_id) or "").strip()
                            if not last_ts:
                                last_ts = (await get_setting("news_sniper_last_at", owner_id=owner_id) or "").strip()
                            if last_ts:
                                try:
                                    last_dt = datetime.fromisoformat(last_ts)
                                    if now_local - last_dt.replace(tzinfo=ZoneInfo(self.tz)) < timedelta(hours=3):
                                        continue
                                except Exception:
                                    pass

                            if not cfg or not getattr(cfg, "openrouter_api_key", ""):
                                continue

                            candidates = await fetch_news_candidates(owner_id=owner_id, limit=3, channel_target=channel_target)
                            if not candidates:
                                continue

                            # Pick the first source-confident candidate
                            item = None
                            for candidate in candidates:
                                if is_source_confident(candidate):
                                    item = candidate
                                    break
                            if not item:
                                logger.info("_job_news_sniper_tick: no source-confident news for owner_id=%s channel=%s", owner_id, channel_target)
                                continue

                            title = item.get("title", "")
                            topic = item.get("topic", "")

                            # Build post text
                            text = await build_news_post(cfg, item, owner_id=owner_id)
                            if not text:
                                logger.warning("_job_news_sniper_tick: empty text owner_id=%s channel=%s", owner_id, channel_target)
                                continue

                            # Apply fabrication cleanup — same rules as main generation pipeline
                            text, _, _ = _remove_fabricated_refs(text)

                            # Build source metadata JSON
                            news_source_json = build_news_source_meta(item)

                            # Save as draft (NOT auto-publish)
                            draft_id = await create_draft(
                                owner_id=owner_id,
                                channel_target=channel_target,
                                text=text,
                                prompt=title,
                                topic=topic,
                                draft_source="news_sniper",
                                news_source_json=news_source_json,
                            )

                            # Mark this news as used — per-channel dedup
                            await log_news(item["link"], title, owner_id=owner_id, channel_target=channel_target)

                            # Update throttle timestamp — per-channel
                            await set_setting(
                                _sniper_ts_key,
                                now_local.isoformat(timespec="seconds"),
                                owner_id=owner_id,
                            )

                            # DM the user about the proactive draft
                            miniapp_url = getattr(cfg, "miniapp_url", "") or ""
                            dm_text = (
                                f"🔥 Срочная новость по вашей теме!\n\n"
                                f"Я подготовил пост: «{title[:120]}{'…' if len(title) > 120 else ''}»\n\n"
                                f"Зайдите в приложение, чтобы опубликовать."
                            )
                            await _safe_send_dm(self.bot, owner_id, dm_text)
                            logger.info("_job_news_sniper_tick: draft created and DM sent owner_id=%s channel=%s draft_id=%s", owner_id, channel_target, draft_id)

                        except Exception:
                            logger.exception("_job_news_sniper_tick channel error owner_id=%s ch=%s", owner_id, ch_profile.get("channel_target"))
                except Exception:
                    logger.exception("_job_news_sniper_tick inner error owner_id=%s", owner_id)
        except Exception:
            logger.exception("_job_news_sniper_tick outer error")


    async def _job_expire_subscriptions(self):
        """Downgrade users whose paid subscription has expired to the free tier."""
        try:
            count = await expire_overdue_subscriptions()
            if count:
                logger.info("_job_expire_subscriptions: downgraded %d expired subscriptions", count)
        except Exception:
            logger.exception("_job_expire_subscriptions error")
