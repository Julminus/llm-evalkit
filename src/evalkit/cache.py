"""On-disk response cache.

One JSON file per response, at ``<root>/<key[:2]>/<key>.json``. The key is a
SHA-256 of everything that decides the output: the provider identity
(provider, model, and endpoint where it matters), the exact prompt, the
settings the provider actually sends, and a sample index.

Why this shape:

* **Files, not a database.** Each entry is written to a temp file and
  renamed into place, so a crash or Ctrl-C never leaves a half-written entry,
  and two runs sharing a cache can't corrupt each other. Entries are plain
  JSON you can open and read.
* **Effective settings, not requested ones.** The Anthropic API ignores
  ``temperature``, so changing it must not miss the cache; changing a setting
  the provider does send must.
* **A sample index.** At temperature > 0 the same request has many valid
  answers. ``sample=0, 1, 2...`` stores several of them (self-consistency
  checks need that) instead of always replaying the first.

A corrupt or unreadable entry counts as a miss and is overwritten.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from evalkit.files import write_text_atomic
from evalkit.prompts import Prompt
from evalkit.providers.base import Completion

__all__ = ["CACHE_VERSION", "CacheEntry", "ResponseCache", "cache_key"]

CACHE_VERSION = 1
"""Bump when the key recipe or the entry format changes; old entries then miss."""


def cache_key(
    identity: Mapping[str, str],
    prompt: Prompt,
    params: Mapping[str, Any],
    *,
    sample: int = 0,
) -> str:
    """Stable hex digest of one request. Dict order never changes the key."""
    payload = {
        "v": CACHE_VERSION,
        "identity": dict(identity),
        "prompt": prompt.model_dump(),
        "params": dict(params),
        "sample": sample,
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class CacheEntry(BaseModel):
    """A stored completion plus the request that produced it (for humans)."""

    model_config = ConfigDict(frozen=True)

    key: str
    completion: Completion
    latency_s: float
    """How long the original request took. Cached replays cost ~0 s."""
    created_at: str
    request: dict[str, Any]


class ResponseCache:
    """A directory of :class:`CacheEntry` files."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> CacheEntry | None:
        try:
            raw = self.path(key).read_text(encoding="utf-8")
            entry = CacheEntry.model_validate_json(raw)
        except (OSError, ValueError, ValidationError):
            return None
        return entry if entry.key == key else None

    def put(
        self,
        key: str,
        completion: Completion,
        *,
        latency_s: float,
        request: Mapping[str, Any],
    ) -> CacheEntry:
        entry = CacheEntry(
            key=key,
            completion=completion,
            latency_s=latency_s,
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            request=dict(request),
        )
        target = self.path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        write_text_atomic(target, entry.model_dump_json(indent=2))
        return entry
