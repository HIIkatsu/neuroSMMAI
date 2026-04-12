"""
image_history.py — Dedup / anti-repeat memory for generated images.

Prevents the system from producing nearly identical outputs by tracking:
  - Content hashes of recently generated images
  - Prompt signatures to detect repeated prompts
  - Owner-scoped history to avoid cross-user dedup

All data is in-memory with TTL-based expiry. No persistence needed —
the dedup window is short (hours, not days).
"""
from __future__ import annotations

import hashlib
import logging
import time
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Configuration
DEFAULT_HISTORY_SIZE = 100
DEFAULT_TTL_SECONDS = 6 * 3600  # 6 hours


@dataclass(slots=True)
class _HistoryEntry:
    key: str
    ts: float


class ImageHistory:
    """In-memory dedup history for generated images.

    Tracks content hashes and prompt signatures to prevent repetition.
    Thread-safe for single-process asyncio (no locks needed).
    """

    def __init__(
        self,
        maxlen: int = DEFAULT_HISTORY_SIZE,
        ttl: float = DEFAULT_TTL_SECONDS,
    ):
        self._ttl = ttl
        self._content_hashes: deque[_HistoryEntry] = deque(maxlen=maxlen)
        self._prompt_sigs: deque[_HistoryEntry] = deque(maxlen=maxlen)
        self._media_refs: deque[_HistoryEntry] = deque(maxlen=maxlen)

    def _is_recent(self, entry: _HistoryEntry) -> bool:
        return (time.monotonic() - entry.ts) < self._ttl

    def _purge_expired(self, q: deque[_HistoryEntry]) -> None:
        while q and not self._is_recent(q[0]):
            q.popleft()

    def is_duplicate_content(self, image_bytes: bytes) -> bool:
        """Check if image content was recently generated."""
        if not image_bytes:
            return False
        content_hash = hashlib.sha256(image_bytes).hexdigest()[:32]
        self._purge_expired(self._content_hashes)
        return any(e.key == content_hash for e in self._content_hashes)

    def is_duplicate_prompt(self, prompt: str) -> bool:
        """Check if a very similar prompt was used recently."""
        if not prompt:
            return False
        sig = _prompt_signature(prompt)
        self._purge_expired(self._prompt_sigs)
        return any(e.key == sig for e in self._prompt_sigs)

    def is_duplicate_ref(self, media_ref: str) -> bool:
        """Check if a media reference was recently used."""
        if not media_ref:
            return False
        ref = media_ref.strip().lower()
        self._purge_expired(self._media_refs)
        return any(e.key == ref for e in self._media_refs)

    def record(
        self,
        *,
        image_bytes: bytes | None = None,
        prompt: str = "",
        media_ref: str = "",
    ) -> None:
        """Record a generated image in history for future dedup."""
        now = time.monotonic()
        if image_bytes:
            content_hash = hashlib.sha256(image_bytes).hexdigest()[:32]
            self._content_hashes.append(_HistoryEntry(key=content_hash, ts=now))
        if prompt:
            sig = _prompt_signature(prompt)
            self._prompt_sigs.append(_HistoryEntry(key=sig, ts=now))
        if media_ref:
            self._media_refs.append(_HistoryEntry(key=media_ref.strip().lower(), ts=now))

    @property
    def size(self) -> int:
        return len(self._content_hashes) + len(self._prompt_sigs) + len(self._media_refs)


def _prompt_signature(prompt: str) -> str:
    """Create a normalized signature from a prompt for near-duplicate detection."""
    # Normalize: lowercase, collapse whitespace, sort words
    words = sorted(set(prompt.lower().split()))
    normalized = " ".join(words)
    return hashlib.md5(normalized.encode()).hexdigest()[:16]


# Global instance — shared across the process
_global_history: ImageHistory | None = None


def get_image_history() -> ImageHistory:
    """Get the global ImageHistory singleton."""
    global _global_history
    if _global_history is None:
        _global_history = ImageHistory()
    return _global_history
