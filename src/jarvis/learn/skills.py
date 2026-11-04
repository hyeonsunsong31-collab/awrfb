"""Skills library: tool sequences that worked before, offered back to the agent as hints.

core appends every successful multi-tool turn; the background learner de-duplicates the file; at the start
of a turn the most similar past requests are shown to the model as examples (never executed blindly - the
current task's scope and the safety policy still apply to every call)."""

from __future__ import annotations

import json
import time
from pathlib import Path

from rapidfuzz import fuzz, utils

MAX_SKILLS = 500


def record_skill(path: Path, request: str, tools: list[dict], lang: str = "") -> bool:
    names = [t["name"] for t in tools if t.get("ok")]
    if len(names) < 2 or len(names) != len(tools):  # only clean multi-step successes
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"t": time.time(), "request": request[:300], "tools": names, "lang": lang}, ensure_ascii=False) + "\n")
    return True


def load_skills(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def dedupe_skills(path: Path) -> int:
    """Keep the newest of near-identical requests with the same tool sequence. Returns how many remain."""
    kept: list[dict] = []
    for s in sorted(load_skills(path), key=lambda s: -s.get("t", 0)):
        if any(k["tools"] == s["tools"] and fuzz.token_set_ratio(k["request"], s["request"], processor=utils.default_process) >= 90 for k in kept):
            continue
        kept.append(s)
        if len(kept) >= MAX_SKILLS:
            break
    path.write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in reversed(kept)), encoding="utf-8")
    return len(kept)


def skill_hints(path: Path, request: str, k: int = 2, min_score: int = 60) -> str:
    scored = sorted(((fuzz.token_set_ratio(request, s["request"], processor=utils.default_process), s) for s in load_skills(path)), key=lambda x: -x[0])
    hits = [s for score, s in scored[:k] if score >= min_score]
    if not hits:
        return ""
    return "Similar past requests that worked (examples only): " + "; ".join(f"'{s['request']}' -> {' > '.join(s['tools'])}" for s in hits)
