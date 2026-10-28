from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from jarvis.common.config import deep_merge, read_yaml
from jarvis.common.paths import config_dir


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Spacing(_M):
    h_close: float = 0.55
    h_far: float = 0.18
    preferred_gap: float = 0.25
    attack_gap: float = 0.35
    retreat_gap: float = 0.08


class Threat(_M):
    block_p: float = 0.55
    block_release_p: float = 0.30
    dodge_p: float = 0.85
    min_block_ms: float = 180
    max_block_ms: float = 900
    hit_reaction_block_ms: float = 350
    guard_low: float = 0.2


class Stamina(_M):
    reserve: float = 0.25
    critical: float = 0.12


class Attack(_M):
    jab_interval_ms: float = 260
    max_jab_chain: int = 3
    chain_pause_ms: float = 420
    heavy_chance: float = 0.18
    feint_chance: float = 0.10
    punish_window_ms: float = 350
    guard_break_after_blocked: int = 3
    heavy_tap_ms: float = 90


class Dash(_M):
    enabled: bool = True
    min_interval_ms: float = 1100
    lockout_after_dash_ms: float = 1000


class Movement(_M):
    approach_when_gap_over: float = 0.12


class TacticsCfg(_M):
    period_ms: float = 4000
    discount: float = 0.97


class PolicyParams(_M):
    spacing: Spacing = Field(default_factory=Spacing)
    threat: Threat = Field(default_factory=Threat)
    stamina: Stamina = Field(default_factory=Stamina)
    attack: Attack = Field(default_factory=Attack)
    dash: Dash = Field(default_factory=Dash)
    movement: Movement = Field(default_factory=Movement)
    tactics: TacticsCfg = Field(default_factory=TacticsCfg)

    def with_overrides(self, over: dict[str, Any]) -> PolicyParams:
        if not over:
            return self
        return PolicyParams.model_validate(deep_merge(self.model_dump(), over))


def load_params(relative: str = "ubg/policy_v0.yaml", calibration_dir=None) -> PolicyParams:
    base = read_yaml(config_dir() / relative)
    if calibration_dir is not None:
        base = deep_merge(base, read_yaml(calibration_dir / "policy_overrides.yaml"))
    return PolicyParams.model_validate(base)
