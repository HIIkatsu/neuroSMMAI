"""
image_service.py — Single public entry point for all image generation requests.

Generation flow:
  caller → image_service → prompt build → image generation → validation → dedup check → storage → result

Fallback flow (only if generation fails):
  caller → image_service → generation failed → fallback search → validation → storage → result

This module is the ONLY module external callers should import for image operations.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass

from image_prompts import build_generation_prompt, build_fallback_search_query, build_subject_rerank_profile
from visual_profile_layer import build_visual_profile, profile_search_queries, score_candidate, ProviderCandidate
from image_generation import generate_image
from image_validation import (
    validate_image_bytes,
    validate_image_url,
    validate_media_ref,
    validate_image_candidate,
)
from image_storage import save_generated_image
from image_history import get_image_history
from image_fallback import search_stock_photo, search_stock_candidates

logger = logging.getLogger(__name__)

# Mode constants — callers use these
MODE_AUTOPOST = "autopost"
MODE_EDITOR = "editor"

# Latin token regex — used by actions.py for query cleaning
_LATIN_TOKEN_RE = re.compile(r"^[A-Za-z0-9][\w.+-]*$")

# Generation timeout
_GENERATION_TIMEOUT = float(30.0)
_FALLBACK_TIMEOUT = float(15.0)
_IMAGE_SEARCH_ONLY = (os.getenv("IMAGE_SEARCH_ONLY", "").strip().lower() in {"1", "true", "yes", "on"})


@dataclass
class ImageResult:
    """Result of an image generation/search request."""
    media_ref: str = ""           # The usable media reference (URL or local path)
    source: str = ""              # "generation", "fallback", "none"
    prompt_used: str = ""         # The prompt that was used
    family: str = ""              # Detected topic family
    failure_reason: str = ""      # Why it failed (if it did)
    is_generated: bool = False    # True if AI-generated, False if stock/fallback


async def get_image(
    *,
    title: str = "",
    body: str = "",
    channel_topic: str = "",
    llm_image_prompt: str = "",
    api_key: str = "",
    model: str = "",
    base_url: str | None = None,
    owner_id: int | None = None,
    mode: str = MODE_EDITOR,
    used_refs: set[str] | None = None,
    text_quality_flagged: bool = False,
    content_mode: str = "",
    channel_style: str = "",
    channel_audience: str = "",
    channel_subniche: str = "",
    onboarding_summary: str = "",
    content_constraints: str = "",
    content_exclusions: str = "",
    visual_style: str = "",
    forbidden_visuals: str = "",
    post_intent: str = "",
) -> ImageResult:
    """Generate or find an image for a post.

    This is the SINGLE entry point for all image operations.
    Generation-first: tries AI generation, falls back to stock search only on failure.

    Args:
        title: Post title
        body: Post body text
        channel_topic: Channel topic for context
        llm_image_prompt: Pre-built LLM prompt for image generation
        api_key: API key for the image generation provider
        model: Model name override
        base_url: API base URL override
        owner_id: Owner ID for storage and access control
        mode: "editor" or "autopost"
        used_refs: Set of recently used image refs to avoid
        content_mode: Detected content mode for mode-aware prompt building

    Returns:
        ImageResult with media_ref, source, and metadata
    """
    normalized_mode = _normalize_mode(mode)
    history = get_image_history()

    # 1. Build prompt
    prompt_data = build_generation_prompt(
        title=title,
        body=body,
        channel_topic=channel_topic,
        llm_image_prompt=llm_image_prompt,
        content_mode=content_mode,
        channel_style=channel_style,
        channel_audience=channel_audience,
        channel_subniche=channel_subniche,
        onboarding_summary=onboarding_summary,
        content_constraints=content_constraints,
        content_exclusions=content_exclusions,
        visual_style=visual_style,
        forbidden_visuals=forbidden_visuals,
        post_intent=post_intent,
        text_quality_flagged=text_quality_flagged,
    )
    prompt = prompt_data["prompt"]
    family = prompt_data["family"]
    effective_mode = prompt_data["content_mode"]

    logger.info(
        "IMAGE_SERVICE_START mode=%s owner_id=%s family=%s title=%r",
        normalized_mode, owner_id, family, (title or "")[:60],
    )

    prompt_ok, prompt_reason = validate_image_candidate(
        prompt=prompt,
        title=title,
        body=body,
        channel_topic=channel_topic,
        content_mode=effective_mode,
        canonical_family=family,
    )
    if not prompt_ok:
        logger.warning("IMAGE_SERVICE_PROMPT_REJECT reason=%s title=%r", prompt_reason, (title or "")[:60])
        prompt_data = build_generation_prompt(
            title=title,
            body=body,
            channel_topic=channel_topic,
            content_mode=content_mode,
            channel_style=channel_style,
            channel_audience=channel_audience,
            channel_subniche=channel_subniche,
            onboarding_summary=onboarding_summary,
            content_constraints=content_constraints,
            content_exclusions=content_exclusions,
            visual_style=visual_style,
            forbidden_visuals=forbidden_visuals,
            post_intent=post_intent,
            text_quality_flagged=text_quality_flagged,
        )
        prompt = prompt_data["prompt"]

    # 2. Check prompt dedup
    if history.is_duplicate_prompt(prompt):
        logger.info("IMAGE_SERVICE_PROMPT_DEDUP prompt_sig=duplicate, proceeding with generation anyway")
        # We still generate — dedup is advisory for prompts, not blocking

    logger.info("IMAGE_SERVICE_SEARCH_FIRST mode=%s image_search_only=%s", normalized_mode, _IMAGE_SEARCH_ONLY)

    # 3. Search-first stock photo path (editor/manual/autopost unified)
    fallback_ref = await _try_fallback(
        title=title,
        body=body,
        channel_topic=channel_topic,
        history=history,
        used_refs=used_refs,
        text_quality_flagged=text_quality_flagged,
        mode=normalized_mode,
        family_context_hint=" ".join(
            x for x in [
                channel_topic,
                onboarding_summary,
                post_intent,
                title,
            ] if (x or "").strip()
        ),
        content_mode=effective_mode,
        normalized_visual_prompt="" if text_quality_flagged else prompt,
        resolved_intent=post_intent,
        canonical_family=family,
        onboarding_summary=onboarding_summary,
        content_constraints=content_constraints,
        llm_image_prompt=llm_image_prompt,
    )

    if fallback_ref:
        logger.info(
            "IMAGE_SERVICE_SUCCESS source=fallback ref=%r family=%s owner_id=%s",
            fallback_ref[:80], family, owner_id,
        )
        history.record(media_ref=fallback_ref)
        return ImageResult(
            media_ref=fallback_ref,
            source="fallback",
            prompt_used=prompt[:200],
            family=family,
            is_generated=False,
        )

    generated_ref = ""
    allow_generation_secondary = os.getenv("IMAGE_ALLOW_GENERATION_SECONDARY", "true").strip().lower() in {"1", "true", "yes", "on"}
    if allow_generation_secondary and normalized_mode != MODE_EDITOR and not history.is_duplicate_visual_pattern(prompt):
        generated_ref = await _try_generation(
            prompt=prompt,
            negative_prompt=prompt_data["negative_prompt"],
            api_key=api_key,
            model=model,
            base_url=base_url,
            owner_id=owner_id,
            history=history,
            used_refs=used_refs,
        )
    if generated_ref:
        logger.info(
            "IMAGE_SERVICE_SUCCESS source=generation_secondary ref=%r family=%s owner_id=%s",
            generated_ref[:80], family, owner_id,
        )
        return ImageResult(
            media_ref=generated_ref,
            source="generation",
            prompt_used=prompt[:200],
            family=family,
            is_generated=True,
        )

    # 5. Complete failure
    logger.warning(
        "IMAGE_SERVICE_NO_IMAGE owner_id=%s family=%s mode=%s",
        owner_id, family, normalized_mode,
    )
    return ImageResult(
        source="none",
        prompt_used=prompt[:200],
        family=family,
        failure_reason="generation_and_fallback_both_failed",
    )


async def validate_image(
    media_ref: str,
    *,
    title: str = "",
    body: str = "",
    channel_topic: str = "",
    mode: str = MODE_AUTOPOST,
) -> bool:
    """Validate an existing image reference.

    Simple validation: checks if the ref is plausible and non-empty.
    Used by scheduler_service.py as a quality gate.
    """
    if not media_ref or not media_ref.strip():
        return True  # Empty ref = no image = valid (caller decides if image is required)

    is_valid, reason = validate_media_ref(media_ref)
    if not is_valid:
        logger.info(
            "IMAGE_VALIDATE_REJECT ref=%r reason=%s mode=%s",
            media_ref[:80], reason, mode,
        )
    return is_valid


async def trigger_unsplash_download(download_location: str) -> bool:
    """No-op stub: Unsplash integration has been removed.

    This stub exists ONLY because miniapp_routes_content.py calls it on
    draft create/update/publish. Removing it would break those callers.
    The stub always returns False and logs explicitly.
    """
    logger.debug(
        "IMAGE_UNSPLASH_NOOP download_location=%r reason=unsplash_removed",
        (download_location or "")[:80],
    )
    return False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

async def _try_generation(
    *,
    prompt: str,
    negative_prompt: str,
    api_key: str,
    model: str,
    base_url: str | None,
    owner_id: int | None,
    history,
    used_refs: set[str] | None,
) -> str:
    """Attempt AI image generation. Returns media ref or empty string."""
    if not _is_valid_image_model(model):
        logger.warning(
            "IMAGE_GENERATION_SKIPPED_INVALID_MODEL model=%r owner_id=%s",
            (model or "")[:120], owner_id,
        )
        return ""
    try:
        image_bytes = await asyncio.wait_for(
            generate_image(
                api_key=api_key,
                prompt=prompt,
                negative_prompt=negative_prompt,
                model=model,
                base_url=base_url,
            ),
            timeout=_GENERATION_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.warning("IMAGE_GENERATION_TIMEOUT owner_id=%s", owner_id)
        return ""
    except Exception as exc:
        logger.error("IMAGE_GENERATION_ERROR owner_id=%s error=%s", owner_id, exc)
        return ""

    if not image_bytes:
        return ""

    # Validate generated bytes
    is_valid, reason = validate_image_bytes(image_bytes)
    if not is_valid:
        logger.warning("IMAGE_GENERATION_INVALID reason=%s owner_id=%s", reason, owner_id)
        return ""

    # Check content dedup
    if history.is_duplicate_content(image_bytes):
        logger.info("IMAGE_GENERATION_CONTENT_DEDUP owner_id=%s", owner_id)
        # Don't reject — duplicates from generation are unlikely but possible
        # Just log and continue

    # Save to storage
    media_ref = save_generated_image(image_bytes, owner_id=owner_id, prompt_hint=prompt[:50])
    if not media_ref:
        logger.error("IMAGE_STORAGE_FAILED owner_id=%s", owner_id)
        return ""

    # Check against used_refs
    if used_refs and media_ref in used_refs:
        logger.info("IMAGE_GENERATION_ALREADY_USED ref=%r owner_id=%s", media_ref[:80], owner_id)
        # Still return it — exact hash collision with used refs is extremely unlikely
        # for generated images

    # Record in history
    history.record(image_bytes=image_bytes, prompt=prompt, media_ref=media_ref)

    return media_ref


async def _try_fallback(
    *,
    title: str,
    body: str,
    channel_topic: str,
    history,
    used_refs: set[str] | None,
    text_quality_flagged: bool = False,
    mode: str = MODE_EDITOR,
    family_context_hint: str = "",
    content_mode: str = "",
    normalized_visual_prompt: str = "",
    resolved_intent: str = "",
    canonical_family: str = "",
    onboarding_summary: str = "",
    content_constraints: str = "",
    llm_image_prompt: str = "",
) -> str:
    """Search-first stock pipeline with canonical visual profile and universal scoring."""
    profile = build_visual_profile(
        title=title,
        body="" if text_quality_flagged else body,
        channel_topic=channel_topic,
        onboarding_summary=onboarding_summary,
        post_intent=resolved_intent,
        content_constraints=content_constraints,
    )
    primary_query, backup_query = profile_search_queries(profile)
    used_body = bool(body and not text_quality_flagged)
    used_llm = bool(llm_image_prompt and llm_image_prompt.strip() and not text_quality_flagged and mode != MODE_EDITOR)

    all_candidates: list[tuple[str, ProviderCandidate]] = []
    for q, qf in ((primary_query, "primary"), (backup_query, "fallback")):
        if not q:
            continue
        try:
            candidates = await asyncio.wait_for(
                search_stock_candidates(q, query_family=qf),
                timeout=_FALLBACK_TIMEOUT,
            )
            all_candidates.extend([(q, c) for c in candidates])
        except asyncio.TimeoutError:
            logger.warning("IMAGE_FALLBACK_TIMEOUT query=%r query_family=%s", q[:40], qf)
        except Exception as exc:
            logger.error("IMAGE_FALLBACK_ERROR query=%r query_family=%s error=%s", q[:40], qf, exc)
        if not any(existing.query_family == qf for _, existing in all_candidates):
            try:
                legacy_url = await asyncio.wait_for(search_stock_photo(q), timeout=_FALLBACK_TIMEOUT)
                if legacy_url:
                    all_candidates.append((q, ProviderCandidate(url=legacy_url, provider="unknown", source_query=q, query_family=qf)))
            except Exception:
                pass

    if not all_candidates:
        logger.info(
            "IMAGE_PIPELINE_DECISION final_decision=no_image reject_reason=no_candidates anchor_family=%s fallback_query_family=%s used_body_for_search=%s used_llm_image_prompt_for_search=%s",
            profile.domain_family, "fallback", used_body, used_llm,
        )
        return ""

    best_url = ""
    best_score = -9999.0
    best_reason = "all_candidates_rejected"

    for candidate_query, candidate in all_candidates:
        url = candidate.url
        is_valid, reason = validate_image_url(url)
        if not is_valid:
            continue
        if history.is_duplicate_ref(url):
            continue

        breakdown = score_candidate(
            candidate=candidate,
            profile=profile,
            used_refs=used_refs,
            history_duplicate=False,
            min_score=1.5,
        )
        if breakdown.score > best_score:
            best_score = breakdown.score
            best_url = url
            best_reason = breakdown.reason

        if breakdown.decision == "accepted":
            logger.info(
                "IMAGE_PIPELINE_DECISION final_decision=accepted reject_reason= provider=%s anchor_family=%s fallback_query_family=%s used_body_for_search=%s used_llm_image_prompt_for_search=%s score=%.2f",
                candidate.provider,
                profile.domain_family,
                candidate.query_family,
                used_body,
                used_llm,
                breakdown.score,
            )
            return url

        logger.info(
            "IMAGE_PIPELINE_DECISION final_decision=rejected reject_reason=%s provider=%s anchor_family=%s fallback_query_family=%s used_body_for_search=%s used_llm_image_prompt_for_search=%s score=%.2f",
            breakdown.reason,
            candidate.provider,
            profile.domain_family,
            candidate.query_family,
            used_body,
            used_llm,
            breakdown.score,
        )

    logger.info(
        "IMAGE_PIPELINE_DECISION final_decision=no_image reject_reason=%s provider=none anchor_family=%s fallback_query_family=%s used_body_for_search=%s used_llm_image_prompt_for_search=%s best_score=%.2f",
        best_reason,
        profile.domain_family,
        "fallback",
        used_body,
        used_llm,
        best_score,
    )
    return ""


def _rerank_candidate_by_subject(
    *,
    media_ref: str,
    candidate_query: str,
    positive_tokens: list[str],
    negative_tokens: list[str],
) -> tuple[bool, str]:
    source_url = (media_ref or "").lower()
    source_query = (candidate_query or "").lower()
    source_url_words = source_url.replace("-", " ").replace("_", " ")
    pos_hits_url = sum(1 for tok in positive_tokens[:10] if tok and tok in source_url)
    pos_hits_query = sum(1 for tok in positive_tokens[:10] if tok and tok in source_query)
    pos_hits = pos_hits_url + (1 if pos_hits_query > 0 else 0)
    neg_hits = sum(1 for tok in negative_tokens[:10] if tok and (tok in source_url or tok in source_url_words))
    score = (pos_hits * 2) - (neg_hits * 3)
    if not positive_tokens and not negative_tokens:
        return True, "subject_rerank_skip_no_profile"
    if neg_hits >= 1 and pos_hits_url == 0:
        return False, f"subject_rerank_negative_domains pos={pos_hits} neg={neg_hits}"
    if pos_hits == 0 and neg_hits == 0:
        return True, "subject_rerank_neutral"
    if score < 0:
        return False, f"subject_rerank_low_score pos={pos_hits} neg={neg_hits} score={score}"
    return True, f"subject_rerank_ok pos={pos_hits} neg={neg_hits} score={score}"


def _normalize_mode(mode: str) -> str:
    m = (mode or "").strip().lower()
    if m in {MODE_AUTOPOST, "news", "auto"}:
        return MODE_AUTOPOST
    return MODE_EDITOR


def _is_valid_image_model(model: str) -> bool:
    m = (model or "").strip().lower()
    if not m:
        # Empty model is allowed — generate_image() will use IMAGE_GENERATION_MODEL/default.
        return True
    # Conservative allow-list: common image generation families/providers.
    image_tokens = (
        "image", "flux", "recraft", "dall", "sd", "stable-diffusion", "midjourney", "imagen",
    )
    text_only_tokens = (
        "mistral", "claude", "llama", "deepseek", "qwen", "gemini", "gpt-4", "gpt-3", "command-r",
    )
    if any(tok in m for tok in image_tokens):
        return True
    if any(tok in m for tok in text_only_tokens):
        return False
    return False
