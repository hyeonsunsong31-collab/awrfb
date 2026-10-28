"""Tactic bundles: named parameter overrides the bandit chooses between."""

from __future__ import annotations

from typing import Any

TACTICS: dict[str, dict[str, Any]] = {
    # The calibrated base parameters, unchanged.
    "baseline": {},
    # Close the distance and keep jabbing; block a little less.
    "pressure": {
        "spacing": {"preferred_gap": 0.15},
        "attack": {"jab_interval_ms": 220, "heavy_chance": 0.22, "max_jab_chain": 4},
        "threat": {"block_p": 0.65},
    },
    # Let them come, block early, punish hard after blocked hits.
    "counter": {
        "spacing": {"preferred_gap": 0.32},
        "attack": {"jab_interval_ms": 420, "heavy_chance": 0.35, "punish_window_ms": 450},
        "threat": {"block_p": 0.45, "block_release_p": 0.25},
    },
    # Hover at the edge of range, feint to draw a swing, then punish.
    "bait_punish": {
        "spacing": {"preferred_gap": 0.4},
        "attack": {"feint_chance": 0.3, "punish_window_ms": 450, "heavy_chance": 0.25},
        "threat": {"block_p": 0.5},
    },
    # Stay out of reach; only step in to land a quick chain.
    "keep_range": {
        "spacing": {"preferred_gap": 0.5, "retreat_gap": 0.15},
        "attack": {"max_jab_chain": 2, "jab_interval_ms": 300},
        "threat": {"block_p": 0.5},
    },
    # Mix feints and heavies against opponents who block a lot.
    "feint_mix": {
        "attack": {"feint_chance": 0.35, "heavy_chance": 0.3, "guard_break_after_blocked": 2},
    },
    # Ahead on health: back off and let UBG's out-of-combat regeneration work.
    "regen_retreat": {
        "spacing": {"preferred_gap": 0.7, "retreat_gap": 0.3},
        "attack": {"jab_interval_ms": 600, "heavy_chance": 0.05},
        "threat": {"block_p": 0.4},
    },
}

DEFAULT_TACTIC = "pressure"
