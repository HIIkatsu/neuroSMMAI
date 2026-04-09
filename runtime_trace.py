"""Runtime tracing for end-to-end observability.

Enable detailed runtime tracing by setting ``DEBUG_RUNTIME_TRACE=1`` in the
environment.  When enabled every text-generation, image-selection and
channel-bootstrap request emits structured log lines containing a unique
``trace_id`` and diagnostic fields (route, source mode, prompt builder used,
scoring path, etc.).

When debug mode is on **and** the caller explicitly opts-in, a minimal
``_debug`` dict can be injected into JSON responses for local/manual
verification.  This is never exposed unless ``DEBUG_RUNTIME_TRACE=1``.
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Any

logger = logging.getLogger("runtime_trace")

# ---------------------------------------------------------------------------
# Debug flag — driven exclusively by env var, safe to check at module level
# ---------------------------------------------------------------------------

def is_debug_trace_enabled() -> bool:
    """Return True when ``DEBUG_RUNTIME_TRACE`` env var is truthy."""
    return os.getenv("DEBUG_RUNTIME_TRACE", "").strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# Trace ID generation
# ---------------------------------------------------------------------------

def new_trace_id() -> str:
    """Return a short unique trace id suitable for log correlation."""
    return uuid.uuid4().hex[:12]


# ---------------------------------------------------------------------------
# Structured trace emitters
# ---------------------------------------------------------------------------

def trace_text_generation(
    *,
    trace_id: str,
    route: str = "",
    source_mode: str = "",
    requested_topic: str = "",
    channel_topic: str = "",
    author_role: str = "",
    prompt_builder: str = "",
    planner_used: bool = False,
    writer_used: bool = False,
    rewrite_used: bool = False,
    final_archetype: str = "",
    reject_reason: str = "",
    quality_score: float | None = None,
    duration_ms: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Log a structured text-generation trace event and return the payload."""
    payload: dict[str, Any] = {
        "trace_id": trace_id,
        "event": "text_generation",
        "route": route,
        "source_mode": source_mode,
        "requested_topic": (requested_topic or "")[:120],
        "channel_topic": (channel_topic or "")[:120],
        "author_role": author_role,
        "prompt_builder": prompt_builder,
        "planner_used": planner_used,
        "writer_used": writer_used,
        "rewrite_used": rewrite_used,
        "final_archetype": final_archetype,
        "reject_reason": reject_reason,
    }
    if quality_score is not None:
        payload["quality_score"] = round(quality_score, 3)
    if duration_ms is not None:
        payload["duration_ms"] = duration_ms
    if extra:
        payload.update(extra)

    if is_debug_trace_enabled():
        logger.info("TRACE_TEXT_GENERATION %s", _fmt(payload))
    else:
        # Always log a compact one-liner even when debug is off
        logger.info(
            "trace_text tid=%s route=%s mode=%s topic=%s builder=%s planner=%s writer=%s archetype=%s reject=%s",
            trace_id, route, source_mode,
            (requested_topic or "")[:60], prompt_builder,
            planner_used, writer_used,
            final_archetype, (reject_reason or "")[:80],
        )
    return payload


def trace_image_selection(
    *,
    trace_id: str,
    route: str = "",
    title_excerpt: str = "",
    body_excerpt: str = "",
    visual_subject: str = "",
    built_query: str = "",
    provider_result_count: int = 0,
    scoring_path: str = "",
    accept_outcome: str = "",
    reject_reason: str = "",
    duration_ms: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Log a structured image-selection trace event and return the payload."""
    payload: dict[str, Any] = {
        "trace_id": trace_id,
        "event": "image_selection",
        "route": route,
        "title_excerpt": (title_excerpt or "")[:100],
        "body_excerpt": (body_excerpt or "")[:100],
        "visual_subject": (visual_subject or "")[:100],
        "built_query": (built_query or "")[:200],
        "provider_result_count": provider_result_count,
        "scoring_path": scoring_path,
        "accept_outcome": accept_outcome,
        "reject_reason": reject_reason,
    }
    if duration_ms is not None:
        payload["duration_ms"] = duration_ms
    if extra:
        payload.update(extra)

    if is_debug_trace_enabled():
        logger.info("TRACE_IMAGE_SELECTION %s", _fmt(payload))
    else:
        logger.info(
            "trace_img tid=%s route=%s query=%s results=%d scoring=%s outcome=%s reject=%s",
            trace_id, route, (built_query or "")[:60],
            provider_result_count, scoring_path,
            accept_outcome, (reject_reason or "")[:80],
        )
    return payload


def trace_channel_label(
    *,
    trace_id: str,
    channel_profile_id: int | str = "",
    channel_target: str = "",
    channel_title: str = "",
    display_label: str = "",
    route: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Log a channel-label resolution trace event and return the payload."""
    payload: dict[str, Any] = {
        "trace_id": trace_id,
        "event": "channel_label",
        "route": route,
        "channel_profile_id": str(channel_profile_id),
        "channel_target": (channel_target or "")[:80],
        "channel_title": (channel_title or "")[:80],
        "display_label": (display_label or "")[:80],
    }
    if extra:
        payload.update(extra)

    if is_debug_trace_enabled():
        logger.info("TRACE_CHANNEL_LABEL %s", _fmt(payload))
    else:
        logger.info(
            "trace_ch tid=%s route=%s cpid=%s target=%s title=%s label=%s",
            trace_id, route, channel_profile_id,
            (channel_target or "")[:40], (channel_title or "")[:40],
            (display_label or "")[:40],
        )
    return payload


# ---------------------------------------------------------------------------
# Debug-response helper
# ---------------------------------------------------------------------------

def debug_fields(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Return a trimmed ``_debug`` dict if debug mode is on, else None.

    Callers should merge this into their JSON response only when non-None::

        dbg = debug_fields(trace_payload)
        if dbg:
            response["_debug"] = dbg
    """
    if not is_debug_trace_enabled():
        return None
    # Only include safe subset — never expose API keys / full prompts
    safe_keys = {
        "trace_id", "event", "route", "source_mode",
        "requested_topic", "channel_topic", "author_role",
        "prompt_builder", "planner_used", "writer_used", "rewrite_used",
        "final_archetype", "reject_reason", "quality_score", "duration_ms",
        "visual_subject", "built_query", "provider_result_count",
        "scoring_path", "accept_outcome",
        "channel_profile_id", "channel_target", "channel_title", "display_label",
    }
    return {k: v for k, v in payload.items() if k in safe_keys}


# ---------------------------------------------------------------------------
# Timer context helper
# ---------------------------------------------------------------------------

class TraceTimer:
    """Simple context-manager that measures elapsed wall-time in milliseconds."""

    def __init__(self) -> None:
        self._start: float = 0.0
        self.elapsed_ms: int = 0

    def __enter__(self) -> "TraceTimer":
        self._start = time.monotonic()
        return self

    def __exit__(self, *_: object) -> None:
        self.elapsed_ms = int((time.monotonic() - self._start) * 1000)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fmt(d: dict[str, Any]) -> str:
    """Format a dict as a compact key=value string for log lines."""
    parts: list[str] = []
    for k, v in d.items():
        if v is None or v == "":
            continue
        parts.append(f"{k}={v!r}" if isinstance(v, str) else f"{k}={v}")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Preview media trace
# ---------------------------------------------------------------------------

def trace_preview_media(
    *,
    trace_id: str,
    media_ref: str = "",
    render_path: str = "",
    resolved_url: str = "",
    is_stale: bool = False,
    needs_auth: bool = False,
    error: str = "",
) -> dict[str, Any]:
    """Log a structured preview media resolution trace event."""
    payload: dict[str, Any] = {
        "trace_id": trace_id,
        "event": "preview_media",
        "media_ref": (media_ref or "")[:120],
        "render_path": render_path,
        "resolved_url": (resolved_url or "")[:120],
        "is_stale": is_stale,
        "needs_auth": needs_auth,
        "error": error,
    }

    if error:
        logger.warning(
            "PREVIEW_MEDIA_RESOLVE_FAIL tid=%s ref=%s error=%s",
            trace_id, (media_ref or "")[:60], error,
        )
    else:
        logger.info(
            "PREVIEW_MEDIA_RESOLVE_OK tid=%s path=%s ref=%s",
            trace_id, render_path, (media_ref or "")[:60],
        )

    if is_stale:
        logger.warning("PREVIEW_MEDIA_STALE_REF tid=%s ref=%s", trace_id, (media_ref or "")[:60])

    return payload


# ---------------------------------------------------------------------------
# Text validation trace
# ---------------------------------------------------------------------------

def trace_text_validation(
    *,
    trace_id: str,
    source_fit_score: int = 10,
    request_fit_score: int = 10,
    fake_numeric_count: int = 0,
    fake_personal_count: int = 0,
    template_repeat_count: int = 0,
    drift_reasons: list[str] | None = None,
    total_risk: int = 0,
    rejected: bool = False,
) -> dict[str, Any]:
    """Log a structured text validation trace event."""
    payload: dict[str, Any] = {
        "trace_id": trace_id,
        "event": "text_validation",
        "source_fit_score": source_fit_score,
        "request_fit_score": request_fit_score,
        "fake_numeric_count": fake_numeric_count,
        "fake_personal_count": fake_personal_count,
        "template_repeat_count": template_repeat_count,
        "drift_reasons": drift_reasons or [],
        "total_risk": total_risk,
        "rejected": rejected,
    }

    logger.info(
        "TEXT_SOURCE_FIT_SCORE=%d TEXT_REQUEST_FIT_SCORE=%d "
        "fake_numeric=%d fake_personal=%d template_repeat=%d risk=%d rejected=%s",
        source_fit_score, request_fit_score,
        fake_numeric_count, fake_personal_count, template_repeat_count,
        total_risk, rejected,
    )

    if fake_numeric_count > 0:
        logger.warning("TEXT_FAKE_NUMERIC_CLAIM_REJECT count=%d", fake_numeric_count)
    if fake_personal_count > 0:
        logger.warning("TEXT_FAKE_PERSONAL_CLAIM_REJECT count=%d", fake_personal_count)
    if drift_reasons:
        for reason in drift_reasons:
            logger.warning("TEXT_DRIFT_REJECT reason=%s", reason)

    return payload
