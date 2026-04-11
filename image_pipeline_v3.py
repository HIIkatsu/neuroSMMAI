"""
image_pipeline_v3.py — Image pipeline orchestrator.

Complete replacement of the image selection system.

Core principle: IMAGE = f(POST), not f(CHANNEL).
A wrong image is an ERROR worse than no image.

Architecture:
  1. Extract visual intent from post title + body
  2. Check imageability → skip if abstract
  3. Collect raw candidates from providers
  4. Score each candidate against post intent (STRICT)
  5. Apply anti-repeat penalties
  6. Accept ONLY if score >= ACCEPT_MIN_SCORE (same for ALL modes)
  7. Log runtime proof: WHY accepted or rejected

ONE threshold. ONE path. Editor and autopost use identical scoring.
Mode only affects retrieval breadth (editor fetches more candidates).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

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
    check_wrong_sense,
    score_candidate,
    rank_candidates,
    detect_meta_family,
    ACCEPT_MIN_SCORE,
)
from image_history import (
    ImageHistory,
    get_image_history,
    extract_domain,
    url_content_hash,
)
from topic_utils import get_family_blocked_visuals

logger = logging.getLogger(__name__)

# Mode constants — kept here for backward compat with tests and callers
MODE_AUTOPOST = "autopost"
MODE_EDITOR = "editor"

# Re-export old outcome constants for backward compat
from image_ranker import (  # noqa: E402
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
    AUTOPOST_MIN_SCORE,
    EDITOR_MIN_SCORE,
    TOP_N_AUTOPOST,
    TOP_N_EDITOR,
)

# Backward compat alias for old validate_image_post_centric_v3 name
# Backward compat alias for old validate_image_post_centric_v3 name
# (set after function definition below)


# ---------------------------------------------------------------------------
# Pipeline result
# ---------------------------------------------------------------------------
@dataclass
class PipelineResult:
    """Result of the image pipeline."""
    image_url: str = ""
    score: int = 0
    source_provider: str = ""
    matched_query: str = ""
    no_image_reason: str = ""
    outcome: str = ""          # "ACCEPT" or "NO_IMAGE"
    accept_reason: str = ""    # Human-readable: WHY this image was chosen
    mode: str = "autopost"
    visual_intent: VisualIntentV2 | None = None
    candidates_evaluated: int = 0
    candidates_rejected: int = 0
    reject_reasons: list[str] = field(default_factory=list)
    # Editor: additional candidates above threshold (for carousel/choices)
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
            "accept_reason": self.accept_reason,
            "no_image_reason": self.no_image_reason,
            "candidates_evaluated": self.candidates_evaluated,
            "candidates_rejected": self.candidates_rejected,
            "reject_reasons": self.reject_reasons[:5],
            "visual_subject": (intent.subject if intent else "")[:40],
            "visual_sense": (intent.sense if intent else "")[:30],
            "imageability": intent.imageability if intent else "",
            "search_queries": [q[:40] for q in (intent.query_terms if intent else [])[:3]],
            "negative_terms": [t[:30] for t in (intent.negative_terms if intent else [])[:5]],
            "editor_candidates_count": len(self.editor_candidates),
        }


# Retrieval limits per mode (editor gets more candidates to choose from)
_RETRIEVE_QUERIES_AUTOPOST = 4
_RETRIEVE_QUERIES_EDITOR = 6

# Max candidates to consider after ranking
_TOP_N = 10


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
    scene_class = intent.scene or visual_class
    coarse_pattern = f"{visual_class}_{intent.subject or 'generic'}".lower().replace(" ", "_")
    history.record(
        url=cs.url,
        content_hash=url_content_hash(cs.url),
        visual_class=visual_class,
        subject_bucket=intent.subject or "",
        domain=domain,
        scene_class=scene_class,
        coarse_pattern=coarse_pattern,
    )


# ---------------------------------------------------------------------------
# No-image reason determination
# ---------------------------------------------------------------------------
_REASON_TO_OUTCOME: dict[str, str] = {
    "low_imageability": "NO_IMAGE",
    "no_candidates": "NO_IMAGE",
    "no_visual_subject": "NO_IMAGE",
    "weak_subject": "NO_IMAGE",
    "no_search_queries": "NO_IMAGE",
    "low_imageability_no_subject": "NO_IMAGE",
    "no_post_content": "NO_IMAGE",
    "wrong_sense": "REJECT_WRONG_SENSE",
    "generic_stock": "REJECT_GENERIC_STOCK",
    "generic_filler": "REJECT_GENERIC_FILLER",
    "repeat_image": "REJECT_REPEAT",
    "blocked_visual": "REJECT_CROSS_FAMILY",
    "cross_family": "REJECT_CROSS_FAMILY",
    "low_subject_match": "REJECT_LOW_CONFIDENCE",
    "low_confidence": "REJECT_LOW_CONFIDENCE",
    "scene_mismatch": "REJECT_LOW_CONFIDENCE",
    "subject_scene_reject": "REJECT_LOW_CONFIDENCE",
}


def _reason_to_outcome(reason: str) -> str:
    """Map a no-image reason string to an outcome constant."""
    return _REASON_TO_OUTCOME.get(reason, "REJECT_LOW_CONFIDENCE")


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

    # Priority order for reason reporting
    for prefix in ("wrong_sense", "blocked_visual", "cross_family",
                    "scene_mismatch", "subject_scene_reject",
                    "generic_filler", "generic_stock",
                    "repeat_image"):
        if any(r.startswith(prefix) for r in reasons):
            return prefix

    # low_score → more specific label
    if any(r.startswith("low_score") for r in reasons):
        return "low_subject_match"

    if result.candidates_rejected > 0:
        return "low_confidence"
    return "no_candidates"


# ---------------------------------------------------------------------------
# Runtime proof log
# ---------------------------------------------------------------------------
def _log_runtime_proof(result: PipelineResult, title: str, body: str) -> None:
    """Emit structured runtime proof: what happened and why."""
    intent = result.visual_intent
    logger.info(
        "IMAGE_PIPELINE_PROOF mode=%s outcome=%s "
        "title=%r subject=%r scene=%r imageability=%s "
        "queries=%s candidates=%d rejected=%d "
        "best_score=%d accept_reason=%s no_image_reason=%s "
        "reject_reasons=%s",
        result.mode, result.outcome,
        (title or "")[:60],
        (intent.subject if intent else "")[:40],
        (intent.scene if intent else "")[:40],
        intent.imageability if intent else "",
        [q[:40] for q in (intent.query_terms if intent else [])[:3]],
        result.candidates_evaluated, result.candidates_rejected,
        result.score,
        result.accept_reason or "none",
        result.no_image_reason or "none",
        result.reject_reasons[:5],
    )
    # Log top candidate details for debugging
    for cs in result.traces[:5]:
        logger.info("IMAGE_CANDIDATE %s", cs.as_log_dict())


# ---------------------------------------------------------------------------
# Validation (post-centric)
# ---------------------------------------------------------------------------
def validate_image_v3(
    image_ref: str,
    *,
    intent: VisualIntentV2,
    image_meta: str = "",
) -> tuple[bool, str]:
    """Validate image against post's visual intent.

    Returns (is_valid, reject_reason).
    Same strict threshold as pipeline selection.
    """
    if not image_ref or not image_ref.startswith("http"):
        return True, ""

    if image_meta:
        wrong_sense = check_wrong_sense(image_meta, intent)
        if wrong_sense:
            return False, wrong_sense

        blocked = get_family_blocked_visuals(intent.post_family)
        meta_lower = image_meta.lower()
        if blocked:
            for cls in blocked:
                if re.search(rf"\b{re.escape(cls)}\b", meta_lower, re.I):
                    return False, f"blocked_visual:{cls}"

        pc_score, reason, _cs = score_candidate(image_meta, intent)
        if pc_score < ACCEPT_MIN_SCORE:
            return False, reason or "low_score"

    return True, ""


# Backward compat alias
validate_image_post_centric_v3 = validate_image_v3
# ---------------------------------------------------------------------------
async def run_pipeline_v3(
    *,
    title: str = "",
    body: str = "",
    channel_topic: str = "",
    used_refs: set[str] | None = None,
    mode: str = "autopost",
) -> PipelineResult:
    """Run the image pipeline.

    Flow:
    1. Extract visual intent
    2. Check imageability
    3. Collect candidates from providers
    4. Score each candidate (strict, post-centric)
    5. Apply anti-repeat penalties
    6. Accept ONLY if best score >= ACCEPT_MIN_SCORE
    7. Log runtime proof
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
        result.outcome = "NO_IMAGE"
        _log_runtime_proof(result, title, body)
        return result

    if intent.imageability == IMAGEABILITY_LOW and not intent.subject:
        result.no_image_reason = "low_imageability_no_subject"
        result.outcome = "NO_IMAGE"
        _log_runtime_proof(result, title, body)
        return result

    # --- Step 3: Get search queries ---
    queries = intent.query_terms
    if not queries:
        result.no_image_reason = "no_search_queries"
        result.outcome = "NO_IMAGE"
        _log_runtime_proof(result, title, body)
        return result

    # Mode affects retrieval breadth only
    max_queries = _RETRIEVE_QUERIES_EDITOR if mode == "editor" else _RETRIEVE_QUERIES_AUTOPOST
    queries = queries[:max_queries]

    # --- Step 4: Collect candidates from providers ---
    from image_providers import collect_candidates

    raw_candidates = await collect_candidates(
        queries=queries,
        used_refs=used_refs,
        mode=mode,
    )

    logger.info(
        "IMAGE_RETRIEVE mode=%s subject=%r queries=%s raw_candidates=%d",
        mode, (intent.subject or "")[:40],
        [q[:40] for q in queries[:3]], len(raw_candidates),
    )

    # --- Step 5: Score each candidate (STRICT) ---
    scored: list[CandidateScore] = []
    for raw in raw_candidates:
        pc_score, pc_reason, cs = score_candidate(raw.meta_text, intent, raw.query)
        cs.url = raw.url
        cs.provider = raw.provider
        cs.query = raw.query
        scored.append(cs)
        result.candidates_evaluated += 1

        if cs.hard_reject or pc_score < ACCEPT_MIN_SCORE:
            result.candidates_rejected += 1
            if pc_reason:
                result.reject_reasons.append(pc_reason)

    # --- Step 6: Apply anti-repeat + rank ---
    history = get_image_history()
    ranked = rank_candidates(scored, intent=intent, history=history)
    result.traces = ranked

    # Collect all reject reasons from ranking
    for cs in ranked:
        if cs.reject_reason and cs.reject_reason not in result.reject_reasons:
            result.reject_reasons.append(cs.reject_reason)

    # --- Step 7: Accept ONLY if best >= ACCEPT_MIN_SCORE AND subject confirmed ---
    accepted = [
        cs for cs in ranked
        if not cs.hard_reject
        and cs.final_score >= ACCEPT_MIN_SCORE
        and (cs.subject_match >= 1 or cs.allowed_visual_hits >= 1)
    ]

    if accepted:
        best = accepted[0]
        result.image_url = best.url
        result.score = best.final_score
        result.source_provider = best.provider
        result.matched_query = best.query
        result.outcome = "ACCEPT"
        result.accept_reason = best.accept_reason

        # Editor gets additional candidates above threshold
        if mode == "editor":
            result.editor_candidates = [
                (cs.url, cs.final_score) for cs in accepted[:_TOP_N]
            ]

        _record_selection(best, intent, history)
        _log_runtime_proof(result, title, body)
        return result

    # --- No acceptable candidate: honest no-image ---
    result.no_image_reason = _determine_no_image_reason(result, intent)
    result.outcome = "NO_IMAGE"
    _log_runtime_proof(result, title, body)
    return result
