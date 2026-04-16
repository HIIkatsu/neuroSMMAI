from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import asyncio
import json
import os
import re
import subprocess
import tempfile
import logging

from aiogram.types import FSInputFile

import db
from content import generate_post_bundle
from image_service import get_image, validate_image, trigger_unsplash_download, MODE_AUTOPOST, MODE_EDITOR, _LATIN_TOKEN_RE, ImageResult
from image_prompts import build_generation_prompt
from runtime_trace import new_trace_id, trace_text_generation, trace_image_selection, TraceTimer, debug_fields, is_debug_trace_enabled
from safe_send import safe_send, safe_send_document, safe_send_photo, safe_send_video
from resolved_subject import resolve_post_subject, check_subject_alignment


@dataclass
class ActionResult:
    ok: bool
    error: str | None = None
    message_id: int = 0
    channel: str = ''
    content_type: str = ''


BASE_DIR = Path(__file__).resolve().parent
logger = logging.getLogger(__name__)


def _resolve_media_ref(media_ref: str) -> str:
    media_ref = (media_ref or "").strip()
    if not media_ref:
        return ""

    if media_ref.startswith("/uploads/") or media_ref.startswith("/generated-images/"):
        mapped = media_ref.replace("/generated-images/", "/generated_images/")
        return str((BASE_DIR / mapped.lstrip("/")).resolve())

    if media_ref.startswith("uploads/") or media_ref.startswith("generated_images/"):
        return str((BASE_DIR / media_ref).resolve())

    return media_ref


def _is_local_media(media_ref: str) -> bool:
    media_ref = (media_ref or "").strip()
    if not media_ref:
        return False
    # Absolute filesystem paths (e.g. returned by image generation) are always local.
    # They must never be forwarded to Telegram as an HTTP URL.
    if os.path.isabs(media_ref):
        return True
    resolved = _resolve_media_ref(media_ref)
    return resolved != media_ref or media_ref.startswith("./") or media_ref.startswith("../")


def _is_telegram_storage_ref(media_ref: str) -> bool:
    return str(media_ref or "").strip().startswith("tgfile:")


def _parse_telegram_storage_ref(media_ref: str) -> tuple[str, str]:
    raw = str(media_ref or "").strip()
    if not raw.startswith("tgfile:"):
        return "", raw
    parts = raw.split(":", 2)
    if len(parts) != 3:
        return "", raw
    return parts[1].strip(), parts[2].strip()


def _extract_telegram_file_id(msg: Any, content_type: str) -> str:
    try:
        if content_type == "photo" and getattr(msg, "photo", None):
            return str(msg.photo[-1].file_id)
        if content_type == "video" and getattr(msg, "video", None):
            return str(msg.video.file_id)
    except Exception:
        pass
    return ""


def _cleanup_local_media(media_ref: str) -> None:
    try:
        if not _is_local_media(media_ref):
            return
        path = _resolve_media_ref(media_ref)
        if path and os.path.isfile(path):
            os.remove(path)
    except Exception:
        pass


def _cleanup_file(path: str | None) -> None:
    try:
        if path and os.path.isfile(path):
            os.remove(path)
    except Exception:
        pass


def _split_text_chunks(text: str, limit: int = 4096) -> list[str]:
    raw = str(text or '').strip()
    if not raw:
        return []
    if len(raw) <= limit:
        return [raw]

    chunks: list[str] = []
    current = ''
    paragraphs = [part.strip() for part in re.split(r"\n{2,}", raw) if part.strip()]
    if not paragraphs:
        paragraphs = [raw]

    for para in paragraphs:
        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) <= limit:
            current = candidate
            continue
        if current:
            chunks.append(current)
            current = ''
        if len(para) <= limit:
            current = para
            continue
        words = para.split()
        buf = ''
        for word in words:
            candidate = f"{buf} {word}".strip() if buf else word
            if len(candidate) <= limit:
                buf = candidate
                continue
            if buf:
                chunks.append(buf)
            if len(word) > limit:
                for i in range(0, len(word), limit):
                    chunks.append(word[i:i + limit])
                buf = ''
            else:
                buf = word
        if buf:
            current = buf
    if current:
        chunks.append(current)
    return [c for c in chunks if c]


TELEGRAM_CAPTION_LIMIT = 1024
CAPTION_LIMIT = TELEGRAM_CAPTION_LIMIT


def _fit_single_caption(text: str, limit: int = TELEGRAM_CAPTION_LIMIT) -> str | None:
    """Try to fit *entire* text within the Telegram caption limit.

    Returns the (possibly lightly trimmed) text if it fits, or ``None``
    if trimming would lose meaningful content.
    """
    raw = str(text or "").strip()
    if not raw:
        return ""
    if len(raw) <= limit:
        return raw
    return None


def _split_media_caption_and_body(
    text: str,
    caption_limit: int = TELEGRAM_CAPTION_LIMIT,
) -> tuple[str, list[str]]:
    """Split text into a caption that fits the Telegram limit + follow-up body chunks.

    The function first attempts to keep everything in a single caption.
    Only when the text exceeds *caption_limit* does it perform a controlled
    split at paragraph / sentence / word boundaries — never an arbitrary cut.
    """
    raw = str(text or "").strip()
    if not raw:
        return "", []
    if len(raw) <= caption_limit:
        return raw, []

    # --- controlled split ---
    parts = [part.strip() for part in re.split(r"\n{2,}", raw) if part.strip()]
    caption = ""
    index = 0
    for idx, part in enumerate(parts):
        candidate = f"{caption}\n\n{part}".strip() if caption else part
        if len(candidate) <= caption_limit:
            caption = candidate
            index = idx + 1
        else:
            break

    if not caption:
        # No paragraph fits — split at the last sentence/word boundary
        head = raw[:caption_limit]
        # Prefer sentence boundary first, then newline, then space
        split_at = -1
        for sep in (". ", "! ", "? ", ".\n", "!\n", "?\n"):
            pos = head.rfind(sep)
            if pos > split_at:
                split_at = pos + 1  # include the punctuation mark, split after it
        if split_at < 240:
            split_at = max(head.rfind("\n"), head.rfind(" "))
        if split_at >= 240:
            caption = raw[:split_at].strip()
            tail = raw[split_at:].strip()
        else:
            caption = raw[:caption_limit].strip()
            tail = raw[caption_limit:].strip()
        return caption, _split_text_chunks(tail, 4096)

    tail_text = "\n\n".join(parts[index:]).strip()
    return caption, _split_text_chunks(tail_text, 4096)


async def _send_followup_chunks(bot, channel: str, chunks: list[str]) -> list[Any]:
    sent = []
    for chunk in chunks:
        msg = await safe_send(bot, channel, chunk)
        sent.append(msg)
    return sent


def _ffmpeg_exists() -> bool:
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        subprocess.run(
            ["ffprobe", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return True
    except Exception:
        return False


def _normalize_video_for_telegram(src_path: str) -> str:
    if not src_path or not os.path.isfile(src_path):
        return src_path

    if not _ffmpeg_exists():
        return src_path

    fd, out_path = tempfile.mkstemp(prefix="tg_ready_", suffix=".mp4")
    os.close(fd)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", src_path,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-profile:v", "high",
        "-level", "4.1",
        "-c:a", "aac",
        "-b:a", "128k",
        "-ar", "44100",
        "-ac", "2",
        "-map_metadata", "-1",
        "-vf", "scale='min(1280,iw)':-2",
        out_path,
    ]

    try:
        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return out_path
    except Exception as e:
        logger.warning(
            "_normalize_video_for_telegram: ffmpeg failed for %r: %s",
            os.path.basename(src_path), e, exc_info=True,
        )
        _cleanup_file(out_path)
        return src_path


def _probe_video_meta(path: str) -> tuple[int | None, int | None, int | None]:
    if not path or not os.path.isfile(path):
        return None, None, None

    if not _ffmpeg_exists():
        return None, None, None

    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height:format=duration",
            "-of", "default=noprint_wrappers=1:nokey=0",
            path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = {}
        for line in result.stdout.splitlines():
            if "=" in line:
                k, v = line.strip().split("=", 1)
                data[k] = v

        width = int(float(data["width"])) if data.get("width") else None
        height = int(float(data["height"])) if data.get("height") else None
        duration = max(1, int(float(data["duration"]))) if data.get("duration") else None
        return width, height, duration
    except Exception as e:
        logger.warning(
            "_probe_video_meta: ffprobe failed for %r: %s",
            os.path.basename(path), e, exc_info=True,
        )
        return None, None, None


def _generate_video_thumbnail(path: str) -> str | None:
    if not path or not os.path.isfile(path) or not _ffmpeg_exists():
        return None

    fd, thumb_path = tempfile.mkstemp(prefix="tg_thumb_", suffix=".jpg")
    os.close(fd)

    cmd = [
        "ffmpeg",
        "-y",
        "-ss", "00:00:01.000",
        "-i", path,
        "-frames:v", "1",
        "-vf", "scale='min(640,iw)':-2",
        thumb_path,
    ]
    try:
        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        if os.path.isfile(thumb_path) and os.path.getsize(thumb_path) > 0:
            return thumb_path
    except Exception as e:
        logger.warning(
            "_generate_video_thumbnail: ffmpeg failed for %r: %s",
            os.path.basename(path), e, exc_info=True,
        )

    _cleanup_file(thumb_path)
    return None


# ---------- ХЕШТЕГИ ----------

def generate_hashtags(topic: str, text: str) -> str:
    words = []

    for w in re.findall(r"[a-zA-Zа-яА-Я0-9]+", topic + " " + text):
        w = w.lower()
        if len(w) < 4:
            continue
        words.append(w)

    uniq = []
    for w in words:
        if w not in uniq:
            uniq.append(w)

    tags = uniq[:5]

    if not tags:
        return ""

    return " ".join("#" + t for t in tags)


# ---------- КАРТИНКИ ----------

async def resolve_post_image(
    query: str,
    *,
    owner_id: int | None = 0,
    ai_prompt: str = "",
    topic: str = "",
    post_text: str = "",
    title: str = "",
    config: Any | None = None,
    used_refs: set[str] | None = None,
    raw_user_query: str = "",
    trace_id: str = "",
    mode: str = "",
) -> str:
    """Resolve the best image for a post via the generation-first image service.

    Calls image_service.get_image() which tries AI generation first,
    then falls back to stock photo search.
    """
    _tid = trace_id or new_trace_id()
    _timer = TraceTimer()
    _timer.__enter__()

    effective_title = title or raw_user_query or query or ""
    effective_body = post_text or ""
    effective_mode = mode if mode in (MODE_AUTOPOST, MODE_EDITOR) else MODE_AUTOPOST

    logger.info(
        "resolve_post_image start owner_id=%s title=%r body_excerpt=%r "
        "topic=%r mode=%s trace_id=%s",
        owner_id, effective_title[:80], effective_body[:80],
        (topic or "")[:60], effective_mode, _tid,
    )

    api_key = (getattr(config, "llm_image_api_key", "") or getattr(config, "openrouter_image_api_key", "") or getattr(config, "openrouter_api_key", "")) if config else ""
    model = (getattr(config, "llm_image_model", "") or getattr(config, "openrouter_image_model", "") or "") if config else ""
    base_url = (getattr(config, "llm_image_base_url", "") or getattr(config, "openrouter_image_base_url", "") or getattr(config, "openrouter_base_url", None)) if config else None

    try:
        result: ImageResult = await get_image(
            title=effective_title,
            body=effective_body,
            channel_topic=topic,
            llm_image_prompt=ai_prompt,
            api_key=api_key,
            model=model,
            base_url=base_url,
            owner_id=owner_id,
            mode=effective_mode,
            used_refs=used_refs,
        )
    except Exception as exc:
        logger.exception("resolve_post_image service failed: %s", exc)
        _timer.__exit__(None, None, None)
        trace_image_selection(
            trace_id=_tid, route="resolve_post_image",
            title_excerpt=effective_title[:60], body_excerpt=effective_body[:100],
            visual_subject=raw_user_query or query, built_query=query,
            provider_result_count=0, scoring_path="service_error",
            accept_outcome="no_image", reject_reason=str(exc)[:200],
            duration_ms=_timer.elapsed_ms,
        )
        return ""

    image_ref = result.media_ref or ""
    _timer.__exit__(None, None, None)

    if image_ref:
        logger.info(
            "resolve_post_image SELECTED ref=%r source=%s mode=%s family=%s",
            image_ref[:80], result.source, effective_mode, result.family,
        )
        trace_image_selection(
            trace_id=_tid, route="resolve_post_image",
            title_excerpt=effective_title[:60], body_excerpt=effective_body[:100],
            visual_subject=raw_user_query or query,
            built_query=query,
            provider_result_count=1,
            scoring_path=f"generation_first_{result.source}",
            accept_outcome="accepted", duration_ms=_timer.elapsed_ms,
        )
    else:
        logger.debug(
            "resolve_post_image EMPTY reason=%s source=%s",
            result.failure_reason, result.source,
        )
        trace_image_selection(
            trace_id=_tid, route="resolve_post_image",
            title_excerpt=effective_title[:60], body_excerpt=effective_body[:100],
            visual_subject=raw_user_query or query, built_query=query,
            provider_result_count=0,
            scoring_path="generation_first",
            accept_outcome="no_image",
            reject_reason=result.failure_reason or "no_match",
            duration_ms=_timer.elapsed_ms,
        )

    return image_ref


# ---------- PAYLOAD ----------

async def generate_post_payload(
    config: Any,
    prompt: str = "",
    *,
    owner_id: int | None = 0,
    channel_profile_id: int | None = None,
    force_image: bool = True,
    current_media_ref: str = "",
    generation_path: str = "editor",
) -> dict:
    _trace_id = new_trace_id()
    _trace_timer = TraceTimer()
    _trace_timer.__enter__()

    # Use channel-scoped settings instead of owner-level get_setting()
    # to respect per-channel audience/style/rubrics when user has multiple channels
    ch_settings = await db.get_channel_settings(owner_id, channel_profile_id=channel_profile_id)
    channel_topic = ch_settings.get("topic") or ""
    effective_prompt = (prompt or channel_topic or "").strip()

    logger.info(
        "GENERATE_POST_PAYLOAD_ENTRY path=%s owner_id=%s literal_topic=%r prompt=%r trace_id=%s",
        generation_path, owner_id, (channel_topic or "")[:80], (effective_prompt or "")[:80], _trace_id,
    )

    recent_posts: list[str] = []
    recent_plan: list[str] = []
    try:
        posts = await db.list_recent_posts(owner_id=owner_id, limit=8)
        recent_posts = [str((x or {}).get("text") or "").strip() for x in posts if str((x or {}).get("text") or "").strip()]
    except Exception:
        recent_posts = []
    try:
        drafts = await db.list_drafts(owner_id=owner_id, limit=20)
        recent_posts.extend([str((x or {}).get("text") or "").strip() for x in (drafts or [])[:8] if str((x or {}).get("text") or "").strip()])
    except Exception:
        pass
    try:
        gens = await db.list_recent_generation_history(owner_id=owner_id, limit=12)
        recent_posts.extend([str((x or {}).get("body") or "").strip() for x in (gens or [])[:8] if str((x or {}).get("body") or "").strip()])
    except Exception:
        pass
    try:
        plan = await db.list_plan_items(owner_id=owner_id, limit=8)
        recent_plan = [str((x or {}).get("prompt") or (x or {}).get("title") or (x or {}).get("idea") or "").strip() for x in plan if str((x or {}).get("prompt") or (x or {}).get("title") or (x or {}).get("idea") or "").strip()]
    except Exception:
        recent_plan = []

    bundle = await generate_post_bundle(
        api_key=config.openrouter_api_key,
        model=config.openrouter_model,
        topic=channel_topic,
        prompt=prompt,
        channel_style=(ch_settings.get("channel_style") or ""),
        content_rubrics=(ch_settings.get("content_rubrics") or ""),
        post_scenarios=(ch_settings.get("post_scenarios") or ""),
        channel_audience=(ch_settings.get("channel_audience") or ""),
        content_constraints=(ch_settings.get("content_constraints") or ""),
        content_exclusions=(ch_settings.get("content_exclusions") or ""),
        channel_formats=(ch_settings.get("channel_formats") or ""),
        recent_posts=recent_posts[:10],
        recent_plan=recent_plan[:10],
        base_url=getattr(config, "openrouter_base_url", None),
        owner_id=owner_id,
        generation_path=generation_path,
    )

    title = str(bundle.get("title") or "").strip()
    body = str(bundle.get("body") or "").strip()
    cta = str(bundle.get("cta") or "").strip()
    short = str(bundle.get("short") or "").strip()
    button_text = str(bundle.get("button_text") or "Подробнее").strip()
    text = "\n\n".join(part for part in [title, body, cta] if part).strip()

    # image_prompt from LLM — creative metaphorical English prompt for image generation
    llm_image_prompt = str(bundle.get("image_prompt") or "").strip()
    if not llm_image_prompt:
        # Build a heuristic prompt from available context
        prompt_data = build_generation_prompt(
            title=title, body=body, channel_topic=channel_topic,
        )
        llm_image_prompt = prompt_data["prompt"]

    image_ref = ""
    post_intent = str(bundle.get("post_intent") or "").strip()
    visual_brief = str(bundle.get("visual_brief") or "").strip()
    logger.info("generate_post_payload force_image=%s owner_id=%s prompt=%r topic=%r image_prompt=%r", force_image, owner_id, prompt, channel_topic, llm_image_prompt[:100])
    logger.debug("GENERATE_POST_PAYLOAD force_image=%s owner_id=%s prompt=%r topic=%r", force_image, owner_id, (prompt or "")[:200], (channel_topic or "")[:200])
    if force_image:
        try:
            _quality_reasons = str(bundle.get("quality_reasons") or "").lower()
            text_quality_flagged = any(
                token in _quality_reasons
                for token in ("off_topic", "off-topic", "out_of_topic", "irrelevant", "topic_mismatch")
            )
            text_quality_flagged = text_quality_flagged or any(
                token in _quality_reasons
                for token in (
                    "low_confidence", "low-confidence",
                    "fabricated", "fake_numeric", "fake_authority",
                )
            )
            used_refs = {current_media_ref} if str(current_media_ref or "").strip() else set()
            # Wider dedup window for autopost to prevent repetitive images
            dedup_limit = 100 if generation_path == "autopost" else 50
            try:
                recent_refs = await db.list_recent_image_refs(owner_id=owner_id, limit=dedup_limit)
                used_refs.update([str(x).strip() for x in (recent_refs or []) if str(x).strip()])
            except Exception:
                pass
            try:
                recent_draft_refs = await db.list_recent_draft_image_refs(owner_id=owner_id, limit=dedup_limit)
                used_refs.update([str(x).strip() for x in (recent_draft_refs or []) if str(x).strip()])
            except Exception:
                pass

            # Generation-first image flow via image_service
            try:
                result: ImageResult = await asyncio.wait_for(
                    get_image(
                        title=title,
                        body=body,
                        channel_topic=channel_topic,
                        llm_image_prompt=llm_image_prompt,
                        api_key=(getattr(config, "llm_image_api_key", "") or getattr(config, "openrouter_image_api_key", "") or getattr(config, "openrouter_api_key", "")),
                        model=(getattr(config, "llm_image_model", "") or getattr(config, "openrouter_image_model", "") or ""),
                        base_url=(getattr(config, "llm_image_base_url", "") or getattr(config, "openrouter_image_base_url", "") or getattr(config, "openrouter_base_url", None)),
                        owner_id=owner_id,
                        mode=generation_path,
                        used_refs=used_refs,
                        text_quality_flagged=text_quality_flagged,
                        content_mode=str(bundle.get("content_mode") or ""),
                        channel_style=str(ch_settings.get("channel_style") or ""),
                        channel_audience=str(ch_settings.get("channel_audience") or ""),
                        channel_subniche=str(ch_settings.get("content_rubrics") or ""),
                        onboarding_summary="; ".join(
                            x for x in [
                                str(ch_settings.get("author_role_type") or "").strip(),
                                str(ch_settings.get("author_role_description") or "").strip(),
                                str(ch_settings.get("author_activities") or "").strip(),
                            ] if x
                        ),
                        content_constraints=str(ch_settings.get("content_constraints") or ""),
                        content_exclusions=str(ch_settings.get("content_exclusions") or ""),
                        visual_style=str(ch_settings.get("channel_formats") or ""),
                        forbidden_visuals=str(ch_settings.get("author_forbidden_claims") or ""),
                        post_intent=visual_brief or post_intent,
                    ),
                    timeout=45.0,
                )
                image_ref = result.media_ref or ""
                if image_ref:
                    logger.info(
                        "generate_post_payload image_ref=%r source=%s family=%s",
                        image_ref[:80], result.source, result.family,
                    )
                else:
                    logger.info(
                        "generate_post_payload no_image reason=%s source=%s",
                        result.failure_reason, result.source,
                    )
            except asyncio.TimeoutError:
                logger.warning("generate_post_payload image timed out owner_id=%s", owner_id)
                image_ref = ""
            except Exception as img_exc:
                logger.info("generate_post_payload image failed: %s", img_exc)
                image_ref = ""

            logger.info("generate_post_payload image_ref=%r", image_ref)
            logger.debug("GENERATE_POST_PAYLOAD_IMAGE_RESULT image_ref=%r", image_ref or "")
        except Exception as exc:
            logger.exception("generate_post_payload image stage failed: %s", exc)
            logger.debug("GENERATE_POST_PAYLOAD_IMAGE_STAGE_FAILED type=%s msg=%r", type(exc).__name__, str(exc)[:300])
            image_ref = ""
    # --- Cross-pipeline subject alignment check ---
    try:
        _resolved = resolve_post_subject(
            title=title,
            body=body,
            channel_topic=channel_topic,
        )
        logger.info(
            "RESOLVED_SUBJECT=%s RESOLVED_SCENE=%s family=%s confidence=%s",
            _resolved.subject, _resolved.scene, _resolved.post_family, _resolved.confidence,
        )
        # Log subject for both pipelines using the same object
        if image_ref and _resolved.subject:
            logger.info(
                "CROSS_PIPELINE_CHECK text_subject=%r image_ref=%r family=%s",
                _resolved.subject, (image_ref or "")[:80], _resolved.post_family,
            )
    except Exception as _subj_exc:
        logger.debug("RESOLVED_SUBJECT_FAILED: %s", _subj_exc)
    _trace_timer.__exit__(None, None, None)
    _text_trace = trace_text_generation(
        trace_id=_trace_id,
        route="generate_post_payload",
        source_mode=generation_path,
        requested_topic=effective_prompt,
        channel_topic=channel_topic,
        author_role=str(ch_settings.get("author_role_type") or ""),
        prompt_builder="generate_post_bundle",
        planner_used=False,
        writer_used=bool(text),
        rewrite_used=False,
        final_archetype="",
        reject_reason="" if text else "empty_text",
        quality_score=float(bundle.get("quality_score") or 0) if bundle else None,
        duration_ms=_trace_timer.elapsed_ms,
        extra={"generation_path": generation_path, "has_image": bool(image_ref)},
    )
    _img_trace = trace_image_selection(
        trace_id=_trace_id,
        route="generate_post_payload",
        title_excerpt=title,
        body_excerpt=(body or "")[:100],
        visual_subject=visual_brief or post_intent or "",
        built_query=effective_prompt,
        provider_result_count=1 if image_ref else 0,
        scoring_path="ai_image+stock_search" if force_image else "skipped",
        accept_outcome="accepted" if image_ref else "no_image",
        reject_reason="" if image_ref else "no_match",
        duration_ms=_trace_timer.elapsed_ms,
    )
    _result = {
        "text": text,
        "title": title,
        "body": body,
        "cta": cta,
        "short": short,
        "button_text": button_text,
        "prompt": prompt,
        "topic": channel_topic,
        "media_type": "photo" if image_ref else "none",
        "media_ref": image_ref,
        "media_meta_json": "",
        "buttons_json": "[]",
        "pin_post": 0,
        "comments_enabled": 1,
        "ad_mark": 0,
        "first_reaction": "",
        "reply_to_message_id": 0,
        "post_intent": post_intent,
        "visual_brief": visual_brief,
        "_trace_id": _trace_id,
    }
    _dbg = debug_fields(_text_trace)
    if _dbg:
        _dbg.update(debug_fields(_img_trace) or {})
        _result["_debug"] = _dbg
    return _result


# ---------- СОЗДАНИЕ ЧЕРНОВИКА ----------

async def create_generated_draft(
    config: Any,
    prompt: str,
    *,
    owner_id: int | None = 0,
    channel_profile_id: int | None = None,
    force_image: bool = True,
) -> int:
    if await db.count_drafts(owner_id=owner_id, status="draft") >= max(1, int(getattr(config, "max_active_drafts_per_user", 25))):
        raise ValueError(f"Достигнут лимит черновиков: {getattr(config, 'max_active_drafts_per_user', 25)}")

    ch_settings = await db.get_channel_settings(owner_id, channel_profile_id=channel_profile_id)
    channel = ch_settings.get("channel_target") or ""

    payload = await generate_post_payload(
        config,
        prompt,
        owner_id=owner_id,
        channel_profile_id=channel_profile_id,
        force_image=force_image,
        generation_path="editor",
    )

    draft_id = await db.create_draft(
        owner_id=owner_id,
        channel_target=channel,
        text=payload["text"],
        prompt=payload["prompt"],
        topic=payload["topic"],
        media_type=payload["media_type"],
        media_ref=payload["media_ref"],
        buttons_json=payload["buttons_json"],
        pin_post=payload["pin_post"],
        comments_enabled=payload["comments_enabled"],
        ad_mark=payload["ad_mark"],
        first_reaction=payload["first_reaction"],
        reply_to_message_id=payload["reply_to_message_id"],
        status="draft",
    )

    try:
        await db.mark_generation_history_draft_saved(owner_id=owner_id, draft_id=draft_id, text=payload.get("text", ""))
    except Exception:
        pass

    return draft_id


# ---------- INLINE КНОПКИ ----------

def _build_reply_markup_from_buttons(buttons_json: str):
    try:
        from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton

        data = json.loads(buttons_json or "[]")

        if not data:
            return None

        rows = []

        for item in data:
            txt = (item.get("text") or "").strip()
            url = (item.get("url") or "").strip()

            if txt and url:
                rows.append([InlineKeyboardButton(text=txt, url=url)])

        if not rows:
            return None

        return InlineKeyboardMarkup(inline_keyboard=rows)

    except Exception:
        return None


# ---------- ПРЕДПРОСМОТР ----------

async def send_draft_preview(target, bot, draft: dict, *, owner_id: int | None = 0):
    reply_markup = _build_reply_markup_from_buttons(draft.get("buttons_json", "[]"))

    text = draft.get("text", "")
    media_type = (draft.get("media_type") or "none").strip()
    media_ref = (draft.get("media_ref") or "").strip()

    if media_type == "photo" and media_ref:
        if _is_telegram_storage_ref(media_ref):
            _kind, file_id = _parse_telegram_storage_ref(media_ref)
            return await target.answer_photo(
                photo=file_id,
                caption=text,
                reply_markup=reply_markup,
            )

        if _is_local_media(media_ref):
            resolved_ref = _resolve_media_ref(media_ref)
            photo = FSInputFile(resolved_ref)
            return await target.answer_photo(
                photo=photo,
                caption=text,
                reply_markup=reply_markup,
            )

        return await target.answer_photo(
            photo=media_ref,
            caption=text,
            reply_markup=reply_markup,
        )

    if media_type == "video" and media_ref:
        if _is_telegram_storage_ref(media_ref):
            _kind, file_id = _parse_telegram_storage_ref(media_ref)
            return await target.answer_video(
                video=file_id,
                caption=text,
                reply_markup=reply_markup,
                supports_streaming=True,
            )

        if _is_local_media(media_ref):
            resolved_ref = _resolve_media_ref(media_ref)
            preview_path = _normalize_video_for_telegram(resolved_ref)
            thumb_path = _generate_video_thumbnail(preview_path)
            width, height, duration = _probe_video_meta(preview_path)

            video = FSInputFile(preview_path)
            kwargs = {
                "video": video,
                "caption": text,
                "reply_markup": reply_markup,
                "supports_streaming": True,
            }
            if width:
                kwargs["width"] = width
            if height:
                kwargs["height"] = height
            if duration:
                kwargs["duration"] = duration
            if thumb_path:
                kwargs["thumbnail"] = FSInputFile(thumb_path)

            try:
                return await target.answer_video(**kwargs)
            finally:
                if preview_path != resolved_ref:
                    _cleanup_file(preview_path)
                _cleanup_file(thumb_path)

        return await target.answer_video(
            video=media_ref,
            caption=text,
            reply_markup=reply_markup,
            supports_streaming=True,
        )

    return await target.answer(
        text,
        reply_markup=reply_markup,
    )


# ---------- ПУБЛИКАЦИЯ ----------

async def publish_draft(bot, draft: dict, *, owner_id: int | None = 0) -> ActionResult:
    resolved_owner = int(owner_id or draft.get("owner_id") or 0)
    ch_settings = await db.get_channel_settings(resolved_owner)
    channel = draft.get("channel_target") or ch_settings.get("channel_target") or ""

    if not channel:
        return ActionResult(False, "Канал не привязан")

    draft_id = int(draft.get("id") or 0)

    # Atomic claim: prevents double-publishing if two jobs fire simultaneously.
    # Skip claim when the caller (e.g. the API route) already pre-claimed the draft
    # (status is already "publishing").  Only claim when we are the first caller.
    draft_status = str(draft.get("status") or "draft").strip().lower()
    if draft_id and resolved_owner and draft_status != "publishing":
        claimed = await db.claim_draft_for_publish(draft_id, resolved_owner)
        if not claimed:
            logger.info("publish_draft: draft %s already claimed/published, skipping", draft_id)
            return ActionResult(False, "Черновик уже публикуется или опубликован")

    text = str(draft.get("text", "") or "").strip()
    media_type = (draft.get("media_type") or "none").strip()
    media_ref = (draft.get("media_ref") or "").strip()
    reply_markup = _build_reply_markup_from_buttons(
        draft.get("buttons_json", "[]")
    )
    send_silent = bool(int(draft.get("send_silent", 0) or 0))

    # Append channel signature if set — from channel-scoped settings
    signature = (ch_settings.get("channel_signature") or "").strip()
    if signature and text and not text.endswith(signature):
        text = text + "\n\n" + signature

    try:
        followup_messages: list[Any] = []
        if media_type == "photo" and media_ref:
            caption, body_chunks = _split_media_caption_and_body(text)
            if _is_telegram_storage_ref(media_ref):
                _kind, file_id = _parse_telegram_storage_ref(media_ref)
                msg = await safe_send_photo(
                    bot,
                    channel,
                    photo=file_id,
                    caption=caption,
                    reply_markup=reply_markup,
                    disable_notification=send_silent,
                )
            elif _is_local_media(media_ref):
                resolved_ref = _resolve_media_ref(media_ref)
                photo = FSInputFile(resolved_ref)
                msg = await safe_send_photo(
                    bot,
                    channel,
                    photo=photo,
                    caption=caption,
                    reply_markup=reply_markup,
                    disable_notification=send_silent,
                )
            else:
                msg = await safe_send_photo(
                    bot,
                    channel,
                    photo=media_ref,
                    caption=caption,
                    reply_markup=reply_markup,
                    disable_notification=send_silent,
                )
            if body_chunks:
                followup_messages = await _send_followup_chunks(bot, channel, body_chunks)
            content_type = "photo"

        elif media_type == "video" and media_ref:
            caption, body_chunks = _split_media_caption_and_body(text)
            if _is_telegram_storage_ref(media_ref):
                _kind, file_id = _parse_telegram_storage_ref(media_ref)
                msg = await safe_send_video(
                    bot,
                    channel,
                    video=file_id,
                    caption=caption,
                    reply_markup=reply_markup,
                    supports_streaming=True,
                    disable_notification=send_silent,
                )
            elif _is_local_media(media_ref):
                resolved_ref = _resolve_media_ref(media_ref)
                video = FSInputFile(resolved_ref)
                msg = await safe_send_video(
                    bot,
                    channel,
                    video=video,
                    caption=caption,
                    reply_markup=reply_markup,
                    supports_streaming=True,
                    disable_notification=send_silent,
                )
            else:
                msg = await safe_send_video(
                    bot,
                    channel,
                    video=media_ref,
                    caption=caption,
                    reply_markup=reply_markup,
                    supports_streaming=True,
                    disable_notification=send_silent,
                )
            if body_chunks:
                followup_messages = await _send_followup_chunks(bot, channel, body_chunks)
            content_type = "video"

        elif media_type == "document" and media_ref:
            caption, body_chunks = _split_media_caption_and_body(text)
            if _is_local_media(media_ref):
                resolved_ref = _resolve_media_ref(media_ref)
                document = FSInputFile(resolved_ref)
                msg = await safe_send_document(
                    bot,
                    channel,
                    document=document,
                    caption=caption,
                    reply_markup=reply_markup,
                    disable_notification=send_silent,
                )
            else:
                if _is_telegram_storage_ref(media_ref):
                    _kind, media_ref = _parse_telegram_storage_ref(media_ref)
                msg = await safe_send_document(
                    bot,
                    channel,
                    document=media_ref,
                    caption=caption,
                    reply_markup=reply_markup,
                    disable_notification=send_silent,
                )
            if body_chunks:
                followup_messages = await _send_followup_chunks(bot, channel, body_chunks)
            content_type = "document"

        else:
            chunks = _split_text_chunks(text or '', 4096)
            if not chunks:
                return ActionResult(False, "Пустой текст поста")
            msg = await safe_send(
                bot,
                channel,
                chunks[0],
                reply_markup=reply_markup,
                disable_notification=send_silent,
            )
            if len(chunks) > 1:
                followup_messages = await _send_followup_chunks(bot, channel, chunks[1:])
            content_type = "text"

        if not msg:
            return ActionResult(False, "Ошибка отправки в Telegram")

        if int(draft.get("pin_post", 0)):
            try:
                await bot.pin_chat_message(
                    channel,
                    getattr(msg, "message_id", 0),
                    disable_notification=True,
                )
            except Exception:
                pass

        telegram_file_id = _extract_telegram_file_id(msg, content_type)
        if not telegram_file_id and _is_telegram_storage_ref(media_ref):
            _kind, telegram_file_id = _parse_telegram_storage_ref(media_ref)
        if not telegram_file_id and not _is_local_media(media_ref):
            telegram_file_id = media_ref

        await db.log_post(
            owner_id=owner_id,
            channel_target=channel,
            content_type=content_type,
            text=text,
            prompt=draft.get("prompt", ""),
            topic=draft.get("topic", ""),
            file_id=telegram_file_id or "",
            telegram_message_id=getattr(msg, "message_id", 0),
        )

        try:
            if telegram_file_id and _is_local_media(media_ref):
                await db.update_draft_field(
                    int(draft["id"]),
                    owner_id,
                    "media_ref",
                    telegram_file_id,
                )
        except Exception:
            pass

        if _is_local_media(media_ref):
            _cleanup_local_media(media_ref)

        try:
            await db.delete_draft(int(draft["id"]), owner_id=owner_id)
        except Exception:
            pass

        last_msg_id = int(getattr((followup_messages[-1] if followup_messages else msg), 'message_id', 0) or 0)
        return ActionResult(True, message_id=last_msg_id, channel=str(channel), content_type=content_type)

    except Exception as e:
        # Release the publishing claim so the draft can be retried immediately.
        # Reset to "draft" (not "failed") so the user doesn't get stuck.
        if draft_id and resolved_owner:
            try:
                await db.release_draft_claim(draft_id, resolved_owner, status="draft")
            except Exception:
                pass
        return ActionResult(False, str(e))
