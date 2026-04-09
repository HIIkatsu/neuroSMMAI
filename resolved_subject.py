"""
resolved_subject.py — Canonical resolved subject for cross-pipeline alignment.

Provides ONE shared subject/scene object used by BOTH text and image pipelines.
This prevents drift between text content and image selection.

Usage:
    subject = resolve_post_subject(title=..., body=..., channel_topic=...)
    # Pass subject to both text generation and image pipeline
    # If text_subject != image_subject → flag mismatch
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ResolvedSubject:
    """Canonical post subject shared between text and image pipelines."""
    subject: str = ""           # Primary subject (English, for image matching)
    subject_ru: str = ""        # Primary subject (Russian, for text matching)
    scene: str = ""             # Expected visual scene
    post_family: str = "generic"  # Topic family
    source: str = "post"        # "post" | "channel_fallback" | "news"
    confidence: str = "medium"  # "high" | "medium" | "low"

    def matches(self, other: "ResolvedSubject", strict: bool = False) -> bool:
        """Check if two resolved subjects are aligned.

        In strict mode, requires exact subject match.
        In non-strict mode, allows family-level alignment.
        """
        if not self.subject or not other.subject:
            return True  # Can't compare, assume aligned

        if self.subject.lower() == other.subject.lower():
            return True

        if strict:
            return False

        # Family-level match
        return self.post_family == other.post_family and self.post_family != "generic"

    def as_log_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "subject_ru": self.subject_ru[:40] if self.subject_ru else "",
            "scene": self.scene,
            "family": self.post_family,
            "source": self.source,
            "confidence": self.confidence,
        }


def resolve_post_subject(
    *,
    title: str = "",
    body: str = "",
    channel_topic: str = "",
    news_title: str = "",
    news_summary: str = "",
) -> ResolvedSubject:
    """Resolve the canonical subject for a post.

    Uses the same intent extraction as the image pipeline, ensuring
    both text and image pipelines work with the same subject.
    """
    from visual_intent_v2 import extract_visual_intent_v2

    # Determine text source for intent extraction
    effective_title = news_title or title
    effective_body = news_summary or body

    intent = extract_visual_intent_v2(
        title=effective_title,
        body=effective_body,
        channel_topic=channel_topic,
    )

    # Build Russian subject from input text
    subject_ru = _extract_russian_subject(effective_title, effective_body)

    result = ResolvedSubject(
        subject=intent.subject,
        subject_ru=subject_ru,
        scene=intent.scene,
        post_family=intent.post_family,
        source=intent.source,
        confidence="high" if intent.subject else ("medium" if intent.post_family != "generic" else "low"),
    )

    logger.info(
        "RESOLVED_SUBJECT subject=%r scene=%r family=%s source=%s confidence=%s",
        result.subject, result.scene, result.post_family, result.source, result.confidence,
    )

    return result


def check_subject_alignment(
    text_subject: ResolvedSubject,
    image_subject: ResolvedSubject,
) -> tuple[bool, str]:
    """Check if text and image subjects are aligned.

    Returns (aligned: bool, mismatch_reason: str).
    """
    if not text_subject.subject or not image_subject.subject:
        return True, ""

    if text_subject.subject.lower() == image_subject.subject.lower():
        logger.info(
            "CROSS_PIPELINE_ALIGNMENT_OK subject=%r",
            text_subject.subject,
        )
        return True, ""

    # Check family-level alignment
    if (text_subject.post_family == image_subject.post_family
            and text_subject.post_family != "generic"):
        logger.info(
            "CROSS_PIPELINE_ALIGNMENT_FAMILY text_subj=%r image_subj=%r family=%s",
            text_subject.subject, image_subject.subject, text_subject.post_family,
        )
        return True, ""

    reason = (
        f"text_subject={text_subject.subject} "
        f"image_subject={image_subject.subject} "
        f"text_family={text_subject.post_family} "
        f"image_family={image_subject.post_family}"
    )
    logger.warning("CROSS_PIPELINE_ALIGNMENT_MISMATCH %s", reason)
    return False, reason


def _extract_russian_subject(title: str, body: str) -> str:
    """Extract the primary Russian subject phrase from post text."""
    combined = f"{title} {body}".strip()
    if not combined:
        return ""
    # Simple: use first meaningful phrase from title
    words = re.findall(r"[а-яёА-ЯЁ]+", title or body or "")
    return " ".join(words[:4])
