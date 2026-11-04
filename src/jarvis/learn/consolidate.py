"""Memory consolidation: merge near-duplicate memories (keep the newest) and drop stale untrusted ones."""

from __future__ import annotations

import time

from rapidfuzz import fuzz, utils


def consolidate(memory, *, similar: int = 92, untrusted_max_age_days: float = 30.0, now: float | None = None) -> dict[str, int]:
    now = time.time() if now is None else now
    facts = sorted(memory.recent(k=100_000), key=lambda f: -f.created)
    kept: list = []
    merged = expired = 0
    for f in facts:
        if f.trust == "untrusted" and now - f.created > untrusted_max_age_days * 86400:
            memory.forget(f.id)
            expired += 1
            continue
        if any(fuzz.token_sort_ratio(f.text, k.text, processor=utils.default_process) >= similar for k in kept):
            memory.forget(f.id)
            merged += 1
            continue
        kept.append(f)
    return {"kept": len(kept), "merged": merged, "expired": expired}
