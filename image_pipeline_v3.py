"""
image_pipeline_v3.py — New image pipeline orchestrator (v3).

Complete rewrite of the image selection system.

Core principle: IMAGE = f(POST), not f(CHANNEL).
Channel topic is a very weak fallback for empty posts only.

Architecture:
  1. Extract visual intent from post title + body
  2. Check imageability → skip if too abstract
  3. Collect raw candidates from providers (broad retrieval)
  4. Score each candidate against post intent
  5. Apply anti-repeat penalties
  6. Top-N reranking
  7. Mode-specific final decision
  8. Log debug trace

Two modes:
  EDITOR  — high recall, returns 6-10 candidates, tolerates weak matches
  AUTOPOST — high precision, prefers NO_IMAGE over bad image

Provider strategy:
  Pexels  = primary
  Pixabay = secondary
  Openverse = editor-only weak fallback (feature-flagged)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from urllib.parse import urlparse as _urlparse

from visual_intent_v2 import (
    VisualIntentV2,
    extract_visual_intent_v2,
    IMAGEABILITY_HIGH,
    IMAGEABILITY_MEDIUM,
    IMAGEABILITY_LOW,
    IMAGEABILITY_NONE,
)
from image_ranker import (
    CandidateScore,
    score_candidate,
    rank_candidates,
    detect_meta_family,
    AUTOPOST_MIN_SCORE,
    EDITOR_MIN_SCORE,
    TOP_N_AUTOPOST,
    TOP_N_EDITOR,
    OUTCOME_ACCEPT_BEST,
    OUTCOME_ACCEPT_FOR_EDITOR,
    OUTCOME_REJECT_WRONG_SENSE,
    OUTCOME_REJECT_GENERIC_STOCK,
    OUTCOME_REJECT_GENERIC_FILLER,
    OUTCOME_REJECT_CROSS_FAMILY,
    OUTCOME_REJECT_LOW_CONFIDENCE,
    OUTCOME_REJECT_REPEAT,
    OUTCOME_NO_IMAGE_SAFE,
    OUTCOME_NO_IMAGE_LOW_IMAGEABILITY,
    OUTCOME_NO_IMAGE_NO_CANDIDATES,
)
from image_history import (
    ImageHistory,
    get_image_history,
    extract_domain,
    url_content_hash,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline modes
# ---------------------------------------------------------------------------
MODE_AUTOPOST = "autopost"
MODE_EDITOR = "editor"


# ---------------------------------------------------------------------------
# Pipeline result
# ---------------------------------------------------------------------------
@dataclass
class PipelineResult:
    """Result of the image pipeline v3."""
    image_url: str = ""
    score: int = 0
    source_provider: str = ""
    matched_query: str = ""
    no_image_reason: str = ""
    outcome: str = ""
    mode: str = MODE_AUTOPOST
    visual_intent: VisualIntentV2 | None = None
    candidates_evaluated: int = 0
    candidates_rejected: int = 0
    reject_reasons: list[str] = field(default_factory=list)
    # Editor mode: multiple tolerable candidates
    editor_candidates: list[tuple[str, int]] = field(default_factory=list)
    # Full trace for debugging
    traces: list[CandidateScore] = field(default_factory=list)

    @property
    def has_image(self) -> bool:
        return bool(self.image_url)

    def trace_summary(self) -> dict:
        """Compact runtime trace for logging/debugging."""
        intent = self.visual_intent
        return {
            "mode": self.mode,
            "outcome": self.outcome,
            "image_url": (self.image_url or "")[:80],
            "score": self.score,
            "source": self.source_provider,
            "query": self.matched_query[:60] if self.matched_query else "",
            "no_image_reason": self.no_image_reason,
            "candidates_evaluated": self.candidates_evaluated,
            "candidates_rejected": self.candidates_rejected,
            "reject_reasons": self.reject_reasons[:5],
            "visual_subject": (intent.subject if intent else "")[:40],
            "visual_sense": (intent.sense if intent else "")[:30],
            "imageability": intent.imageability if intent else "",
            "search_queries": [q[:40] for q in (intent.query_terms if intent else [])[:3]],
            "negative_terms": (intent.negative_terms if intent else [])[:3],
            "editor_candidates_count": len(self.editor_candidates),
        }


# ---------------------------------------------------------------------------
# Debug trace
# ---------------------------------------------------------------------------
def _log_debug_trace(result: PipelineResult, title: str, body: str) -> None:
    """Log a compact but useful debug trace for the pipeline run."""
    intent = result.visual_intent
    logger.info(
        "IMAGE_PIPELINE_V3_TRACE mode=%s outcome=%s "
        "title=%r body_excerpt=%r "
        "subject=%r sense=%r scene=%r "
        "imageability=%s family=%s source=%s "
        "queries=%s "
        "candidates_evaluated=%d candidates_rejected=%d "
        "best_score=%d best_provider=%s best_query=%r "
        "no_image_reason=%s reject_reasons=%s",
        result.mode, result.outcome,
        (title or "")[:60], (body or "")[:60],
        (intent.subject if intent else "")[:40],
        (intent.sense if intent else ""),
        (intent.scene if intent else "")[:40],
        intent.imageability if intent else "",
        intent.post_family if intent else "",
        intent.source if intent else "",
        [q[:40] for q in (intent.query_terms if intent else [])[:3]],
        result.candidates_evaluated, result.candidates_rejected,
        result.score, result.source_provider,
        result.matched_query[:40] if result.matched_query else "",
        result.no_image_reason,
        result.reject_reasons[:5],
    )
    # Log top candidate traces
    for cs in result.traces[:5]:
        logger.info("IMAGE_V3_CANDIDATE %s", cs.as_log_dict())


# ---------------------------------------------------------------------------
# Record selected image in history
# ---------------------------------------------------------------------------
def _record_selection(
    cs: CandidateScore,
    intent: VisualIntentV2,
    history: ImageHistory,
) -> None:
    """Record accepted image into anti-repeat history."""
    domain = extract_domain(cs.url)
    visual_class = detect_meta_family(cs.meta_snippet)
    history.record(
        url=cs.url,
        content_hash=url_content_hash(cs.url),
        visual_class=visual_class,
        subject_bucket=intent.subject or "",
        domain=domain,
    )


# ---------------------------------------------------------------------------
# No-image reason determination
# ---------------------------------------------------------------------------
def _determine_no_image_reason(
    result: PipelineResult,
    intent: VisualIntentV2,
) -> str:
    """Determine the most specific reason for no image."""
    if intent.no_image_reason:
        return intent.no_image_reason

    reasons = result.reject_reasons
    if not reasons and result.candidates_evaluated == 0:
        return "no_candidates"

    wrong_sense_count = sum(1 for r in reasons if r.startswith("wrong_sense"))
    generic_stock_count = sum(1 for r in reasons if r == "generic_stock")
    blocked_count = sum(1 for r in reasons if r.startswith("blocked_visual"))
    cross_family_count = sum(1 for r in reasons if r.startswith("cross_family"))
    repeat_count = sum(1 for r in reasons if r == "repeat_image")
    filler_count = sum(1 for r in reasons if r == "generic_filler")

    if wrong_sense_count > 0:
        return "wrong_sense"
    if repeat_count > 0:
        return "repeat_image"
    if filler_count > 0:
        return "generic_filler"
    if generic_stock_count > 0:
        return "generic_stock"
    if blocked_count > 0:
        return "blocked_visual"
    if cross_family_count > 0:
        return "cross_family"
    if result.candidates_rejected > 0:
        return "low_subject_match"
    return "no_candidates"


def _reason_to_outcome(reason: str) -> str:
    """Map reason string to OUTCOME constant."""
    mapping = {
        "low_imageability": OUTCOME_NO_IMAGE_LOW_IMAGEABILITY,
        "low_imageability_no_subject": OUTCOME_NO_IMAGE_LOW_IMAGEABILITY,
        "no_search_queries": OUTCOME_NO_IMAGE_NO_CANDIDATES,
        "no_candidates": OUTCOME_NO_IMAGE_NO_CANDIDATES,
        "wrong_sense": OUTCOME_REJECT_WRONG_SENSE,
        "generic_stock": OUTCOME_REJECT_GENERIC_STOCK,
        "generic_filler": OUTCOME_REJECT_GENERIC_FILLER,
        "repeat_image": OUTCOME_REJECT_REPEAT,
        "blocked_visual": OUTCOME_REJECT_CROSS_FAMILY,
        "cross_family": OUTCOME_REJECT_CROSS_FAMILY,
        "low_subject_match": OUTCOME_REJECT_LOW_CONFIDENCE,
        "weak_subject": OUTCOME_NO_IMAGE_SAFE,
        "no_visual_subject": OUTCOME_NO_IMAGE_SAFE,
    }
    return mapping.get(reason, OUTCOME_NO_IMAGE_SAFE)


# ---------------------------------------------------------------------------
# Validation (post-centric)
# ---------------------------------------------------------------------------
def validate_image_post_centric_v3(
    image_ref: str,
    *,
    intent: VisualIntentV2,
    image_meta: str = "",
    mode: str = MODE_AUTOPOST,
) -> tuple[bool, str]:
    """Validate image against post's visual intent.

    Returns (is_valid, reject_reason).
    """
    if not image_ref or not image_ref.startswith("http"):
        return True, ""

    if image_meta:
        from image_ranker import check_wrong_sense as _check_ws
        wrong_sense = _check_ws(image_meta, intent)
        if wrong_sense:
            return False, wrong_sense

        blocked = get_family_blocked_visuals(intent.post_family)
        meta_lower = image_meta.lower()
        if blocked:
            import re as _re
            for cls in blocked:
                if _re.search(rf"\b{_re.escape(cls)}\b", meta_lower, _re.I):
                    return False, f"blocked_visual:{cls}"

        min_score = AUTOPOST_MIN_SCORE if mode == MODE_AUTOPOST else EDITOR_MIN_SCORE
        pc_score, reason, _cs = score_candidate(image_meta, intent)
        if pc_score < min_score:
            return False, reason or "low_score"

    return True, ""


# Need to import here to avoid circular at module level
from topic_utils import get_family_blocked_visuals  # noqa: E402


# ---------------------------------------------------------------------------
# Main pipeline entry point (v3)
# ---------------------------------------------------------------------------
async def run_pipeline_v3(
    *,
    title: str = "",
    body: str = "",
    channel_topic: str = "",
    used_refs: set[str] | None = None,
    mode: str = MODE_AUTOPOST,
) -> PipelineResult:
    """Run the full image pipeline v3.

    Flow:
    1. Extract visual intent from post
    2. Check imageability
    3. Collect candidates from providers (broad retrieval)
    4. Score each candidate post-centrically
    5. Apply anti-repeat penalties
    6. Top-N rerank
    7. Mode-specific final decision
    8. Log debug trace
    """
    result = PipelineResult(mode=mode)

    # --- Step 1: Extract visual intent ---
    intent = extract_visual_intent_v2(
        title=title,
        body=body,
        channel_topic=channel_topic,
    )
    result.visual_intent = intent

    # --- Step 2: Check imageability ---
    if intent.imageability == IMAGEABILITY_NONE:
        result.no_image_reason = "low_imageability"
        result.outcome = OUTCOME_NO_IMAGE_LOW_IMAGEABILITY
        logger.info(
            "IMAGE_V3 outcome=%s reason=low_imageability mode=%s title=%r",
            result.outcome, mode, (title or "")[:60],
        )
        _log_debug_trace(result, title, body)
        return result

    if intent.imageability == IMAGEABILITY_LOW and not intent.subject:
        result.no_image_reason = "low_imageability_no_subject"
        result.outcome = OUTCOME_NO_IMAGE_LOW_IMAGEABILITY
        _log_debug_trace(result, title, body)
        return result

    # --- Step 3: Get search queries ---
    queries = intent.query_terms
    if not queries:
        result.no_image_reason = "no_search_queries"
        result.outcome = OUTCOME_NO_IMAGE_NO_CANDIDATES
        _log_debug_trace(result, title, body)
        return result

    # --- Step 4: Collect candidates from providers ---
    from image_providers import collect_candidates, RawCandidate

    raw_candidates = await collect_candidates(
        queries=queries,
        used_refs=used_refs,
        mode=mode,
    )

    logger.info(
        "IMAGE_V3_RETRIEVE mode=%s family=%s subject=%r queries=%s "
        "imageability=%s raw_candidates=%d",
        mode, intent.post_family, (intent.subject or "")[:40],
        [q[:40] for q in queries[:3]], intent.imageability,
        len(raw_candidates),
    )

    # --- Step 5: Score each candidate ---
    scored: list[CandidateScore] = []
    for raw in raw_candidates:
        pc_score, pc_reason, cs = score_candidate(raw.meta_text, intent, raw.query)
        cs.url = raw.url
        cs.provider = raw.provider
        cs.query = raw.query
        scored.append(cs)
        result.candidates_evaluated += 1

        if cs.hard_reject or pc_score < EDITOR_MIN_SCORE:
            result.candidates_rejected += 1
            if pc_reason:
                result.reject_reasons.append(pc_reason)

    # --- Step 6: Apply anti-repeat + rank ---
    history = get_image_history()
    ranked = rank_candidates(
        scored,
        intent=intent,
        history=history,
        mode=mode,
    )
    result.traces = ranked

    # Update rejection counts after ranking
    for cs in ranked:
        if cs.outcome.startswith("REJECT") and cs.reject_reason:
            if cs.reject_reason not in result.reject_reasons:
                result.reject_reasons.append(cs.reject_reason)

    # --- Step 7: Top-N selection ---
    accepted = [
        cs for cs in ranked
        if not cs.hard_reject and not cs.outcome.startswith("REJECT")
    ]
    top_n_limit = TOP_N_EDITOR if mode == MODE_EDITOR else TOP_N_AUTOPOST
    top_n = accepted[:top_n_limit]

    # --- Step 8: Mode-specific final decision ---
    if top_n:
        best = top_n[0]
        min_score = AUTOPOST_MIN_SCORE if mode == MODE_AUTOPOST else EDITOR_MIN_SCORE

        if best.final_score >= min_score:
            result.image_url = best.url
            result.score = best.final_score
            result.source_provider = best.provider
            result.matched_query = best.query
            result.outcome = best.outcome

            if mode == MODE_EDITOR:
                result.editor_candidates = [
                    (cs.url, cs.final_score) for cs in top_n[:10]
                    if cs.final_score >= EDITOR_MIN_SCORE
                ]

            _record_selection(best, intent, history)
            _log_debug_trace(result, title, body)
            return result

        # Editor: accept weaker candidates
        if mode == MODE_EDITOR and best.final_score >= EDITOR_MIN_SCORE:
            result.image_url = best.url
            result.score = best.final_score
            result.outcome = OUTCOME_ACCEPT_FOR_EDITOR
            result.source_provider = best.provider
            result.matched_query = best.query
            result.editor_candidates = [
                (cs.url, cs.final_score) for cs in top_n[:10]
                if cs.final_score >= EDITOR_MIN_SCORE
            ]
            _record_selection(best, intent, history)
            _log_debug_trace(result, title, body)
            return result

    # --- Editor fallback: surface non-hard-rejected positives ---
    if mode == MODE_EDITOR and ranked:
        soft_ok = [
            cs for cs in ranked
            if not cs.hard_reject and cs.final_score > 0
        ]
        if soft_ok:
            best_soft = soft_ok[0]
            result.image_url = best_soft.url
            result.score = best_soft.final_score
            result.source_provider = best_soft.provider
            result.matched_query = best_soft.query
            result.outcome = OUTCOME_ACCEPT_FOR_EDITOR
            result.editor_candidates = [
                (cs.url, cs.final_score) for cs in soft_ok[:10]
            ]
            _record_selection(best_soft, intent, history)
            _log_debug_trace(result, title, body)
            return result

    # --- No acceptable candidate: honest no-image ---
    result.no_image_reason = _determine_no_image_reason(result, intent)
    result.outcome = _reason_to_outcome(result.no_image_reason)
    result.reject_reasons = list(set(result.reject_reasons))

    _log_debug_trace(result, title, body)
    return result
