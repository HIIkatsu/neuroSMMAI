"""
image_history.py — Anti-repeat memory for image pipeline v3.

Tracks recently selected images to prevent repetition across posts.

Dimensions tracked:
  1. Exact URL / hash  → hard reject (same image)
  2. Visual class      → soft penalty (same type of image)
  3. Subject bucket    → soft penalty (same subject)
  4. Provider domain   → soft penalty (same source too often)

All entries have TTL-based expiry (default 6h).
Thread-safe for single-process asyncio.
"""
from __future__ import annotations

import hashlib
import logging
import time as _time
from collections import deque
from dataclasses import dataclass
from urllib.parse import urlparse as _urlparse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_HISTORY_SIZE = 80
DEFAULT_TTL_SECONDS = 3600 * 6  # 6 hours

# Penalty weights
P_REPEAT_EXACT_URL = -200       # Same exact image → essentially hard reject
P_REPEAT_HASH = -200            # Same content hash → hard reject
P_REPEAT_VISUAL_CLASS = -18     # Same visual class recently
P_REPEAT_SUBJECT_BUCKET = -12   # Same subject bucket recently
P_REPEAT_DOMAIN = -10           # Same provider domain recently
P_REPEAT_DOMAIN_FREQUENT = -20  # Same domain used very frequently
P_REPEAT_SCENE_CLASS = -15      # Same scene class recently


# ---------------------------------------------------------------------------
# History entry
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class _HistoryEntry:
    value: str
    ts: float


# ---------------------------------------------------------------------------
# ImageHistory — the anti-repeat memory
# ---------------------------------------------------------------------------
class ImageHistory:
    """In-memory history of recently selected images for anti-repeat.

    Tracks multiple dimensions:
    - URLs (exact match → hard reject)
    - Content hashes (for URL-independent duplicate detection)
    - Visual classes (e.g., "food", "tech" — penalizes same type)
    - Subject buckets (e.g., "coffee" — penalizes same subject)
    - Provider domains (penalizes over-reliance on one source)
    """

    def __init__(
        self,
        maxlen: int = DEFAULT_HISTORY_SIZE,
        ttl: float = DEFAULT_TTL_SECONDS,
    ):
        self._ttl = ttl
        self.urls: deque[_HistoryEntry] = deque(maxlen=maxlen)
        self.hashes: deque[_HistoryEntry] = deque(maxlen=maxlen)
        self.visual_classes: deque[_HistoryEntry] = deque(maxlen=maxlen)
        self.subject_buckets: deque[_HistoryEntry] = deque(maxlen=maxlen)
        self.domains: deque[_HistoryEntry] = deque(maxlen=maxlen)
        self.scene_classes: deque[_HistoryEntry] = deque(maxlen=maxlen)

    # -- Internal helpers --

    def _alive(self, entry: _HistoryEntry) -> bool:
        return (_time.monotonic() - entry.ts) < self._ttl

    def _count(self, q: deque[_HistoryEntry], val: str) -> int:
        if not val:
            return 0
        return sum(1 for e in q if self._alive(e) and e.value == val)

    def _append(self, q: deque[_HistoryEntry], val: str) -> None:
        if val:
            q.append(_HistoryEntry(value=val, ts=_time.monotonic()))

    # -- Public API --

    def compute_penalty(
        self,
        *,
        url: str = "",
        content_hash: str = "",
        visual_class: str = "",
        subject_bucket: str = "",
        domain: str = "",
        scene_class: str = "",
    ) -> int:
        """Compute total anti-repeat penalty for a candidate.

        Returns 0 or negative value. More negative = more repeated.
        """
        penalty = 0

        url_norm = (url or "").strip().lower()
        if url_norm and self._count(self.urls, url_norm) > 0:
            penalty += P_REPEAT_EXACT_URL

        hash_norm = (content_hash or "").strip().lower()
        if hash_norm and self._count(self.hashes, hash_norm) > 0:
            penalty += P_REPEAT_HASH

        vc = (visual_class or "").strip().lower()
        if vc:
            vc_count = self._count(self.visual_classes, vc)
            if vc_count > 0:
                penalty += P_REPEAT_VISUAL_CLASS * min(vc_count, 3)

        subj = (subject_bucket or "").strip().lower()
        if subj:
            subj_count = self._count(self.subject_buckets, subj)
            if subj_count > 0:
                penalty += P_REPEAT_SUBJECT_BUCKET * min(subj_count, 2)

        dom = (domain or "").strip().lower()
        if dom:
            dom_count = self._count(self.domains, dom)
            if dom_count >= 3:
                penalty += P_REPEAT_DOMAIN_FREQUENT
            elif dom_count >= 1:
                penalty += P_REPEAT_DOMAIN

        sc = (scene_class or "").strip().lower()
        if sc:
            sc_count = self._count(self.scene_classes, sc)
            if sc_count > 0:
                penalty += P_REPEAT_SCENE_CLASS * min(sc_count, 2)

        return penalty

    def record(
        self,
        *,
        url: str = "",
        content_hash: str = "",
        visual_class: str = "",
        subject_bucket: str = "",
        domain: str = "",
        scene_class: str = "",
    ) -> None:
        """Record a selected image into history."""
        self._append(self.urls, (url or "").strip().lower())
        self._append(self.hashes, (content_hash or "").strip().lower())
        self._append(self.visual_classes, (visual_class or "").strip().lower())
        self._append(self.subject_buckets, (subject_bucket or "").strip().lower())
        self._append(self.domains, (domain or "").strip().lower())
        self._append(self.scene_classes, (scene_class or "").strip().lower())

    def prune(self) -> None:
        """Remove expired entries from all queues."""
        now = _time.monotonic()
        for q in (self.urls, self.hashes, self.visual_classes,
                  self.subject_buckets, self.domains, self.scene_classes):
            while q and (now - q[0].ts) >= self._ttl:
                q.popleft()


# ---------------------------------------------------------------------------
# Utility: compute content hash from URL
# ---------------------------------------------------------------------------
def url_content_hash(url: str) -> str:
    """Compute a stable hash from URL for duplicate detection.

    Strips query parameters and fragments to detect same image
    served from different CDN cache keys.
    """
    if not url:
        return ""
    try:
        parsed = _urlparse(url)
        # Normalize: scheme + host + path (no query/fragment)
        canonical = f"{parsed.scheme}://{parsed.netloc}{parsed.path}".lower()
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]
    except Exception:
        return ""


def extract_domain(url: str) -> str:
    """Extract domain from URL for provider tracking."""
    if not url:
        return ""
    try:
        return _urlparse(url).netloc.lower()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------
_global_history = ImageHistory()


def get_image_history() -> ImageHistory:
    """Return the global image history singleton."""
    return _global_history
