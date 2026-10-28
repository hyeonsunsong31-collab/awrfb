"""VRAM tier selection.

Tiers are chosen from *measured* numbers: total VRAM and, ideally, free VRAM while Roblox is running
(``jarvis doctor --vram`` records both). A card whose free memory with Roblox open is too small for the
tier's resident models steps down.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from jarvis.common.paths import DataPaths

# Resident GPU memory each tier needs *during fights* (Whisper turbo int8 + fight models + margin).
FIGHT_RESIDENT_MB = {"t6": 1700, "t8": 1700, "t12": 2600}
# Extra needed *outside* fights for the GPU voice / local LLM.
IDLE_EXTRA_MB = {"t6": 0, "t8": 4200, "t12": 9000}

ORDER = ["t6", "t8", "t12"]


@dataclass
class TierDecision:
    tier: str
    total_mb: int
    free_mb_with_roblox: int | None
    reason: str


def choose_tier(total_mb: int, free_mb_with_roblox: int | None = None) -> TierDecision:
    if total_mb >= 11_500:
        tier = "t12"
    elif total_mb >= 7_500:
        tier = "t8"
    else:
        tier = "t6"
    reason = f"total VRAM {total_mb} MB"
    if free_mb_with_roblox is not None:
        while tier != "t6" and free_mb_with_roblox < FIGHT_RESIDENT_MB[tier] + IDLE_EXTRA_MB[tier] * 0.5:
            tier = ORDER[ORDER.index(tier) - 1]
            reason += f"; stepped down: only {free_mb_with_roblox} MB free with Roblox running"
    return TierDecision(tier, total_mb, free_mb_with_roblox, reason)


def step_down(tier: str) -> str:
    i = ORDER.index(tier)
    return ORDER[max(0, i - 1)]


def write_tier(paths: DataPaths, decision: TierDecision) -> Path:
    paths.run.mkdir(parents=True, exist_ok=True)
    f = paths.run / "tier.json"
    f.write_text(json.dumps(asdict(decision), indent=2), encoding="utf-8")
    return f
