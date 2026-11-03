"""Between-match coach (LLM). Never in the real-time loop.

It reads the last match's numbers (stocks, damage, hits, the bandit's tactic statistics, the opponent profile)
and answers with small, bounded adjustments: tactic prior nudges and parameter multipliers from an allowlist,
each clamped and kept inside absolute limits, plus one or two spoken sentences. The fighter picks them up at
the next arm (policy_overrides.yaml + tactic_priors.yaml in the calibration folder). Every decision is logged.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from jarvis.brain.prompts import COACH
from jarvis.common.config import deep_merge, read_yaml
from jarvis.fight.policy.params import PolicyParams
from jarvis.fight.policy.tactics import TACTICS

# Parameter -> absolute (min, max) the coach may move it within.
LIMITS: dict[str, tuple[float, float]] = {
    "spacing.preferred_gap": (0.10, 0.50),
    "spacing.attack_gap": (0.20, 0.50),
    "threat.block_p": (0.30, 0.80),
    "threat.block_release_p": (0.15, 0.60),
    "threat.dodge_p": (0.60, 0.95),
    "stamina.reserve": (0.10, 0.45),
    "attack.jab_interval_ms": (180, 450),
    "attack.heavy_chance": (0.05, 0.40),
    "attack.feint_chance": (0.00, 0.30),
    "attack.punish_window_ms": (200, 600),
    "dash.min_interval_ms": (700, 2000),
}
MULT = (0.8, 1.25)
PRIOR = (-0.1, 0.1)
PRIOR_KEEP = 0.5  # new priors are averaged with the previous ones: no single match swings the style


@dataclass
class CoachAdvice:
    tactic_priors: dict[str, float] = field(default_factory=dict)
    param_nudges: dict[str, float] = field(default_factory=dict)
    comment: str = ""
    rejected: list[str] = field(default_factory=list)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def match_brief(summary: dict[str, Any]) -> dict[str, Any]:
    """The numbers the coach sees (no images, no player names)."""
    pol = summary.get("policy") or {}
    return {k: summary.get(k) for k in ("result", "duration_s", "stocks_won", "stocks_lost", "damage_dealt", "damage_taken", "hits")} | {
        "bandit": pol.get("bandit"), "opponent": pol.get("opponent"), "decisions": pol.get("decisions")}


def build_messages(summary: dict[str, Any], lang: str, knowledge: str = "") -> list[dict]:
    system = COACH.format(lang=lang, tactics=", ".join(TACTICS), params=", ".join(LIMITS))
    user = json.dumps(match_brief(summary), ensure_ascii=False, default=str)
    if knowledge:
        user += "\n\nGame notes (may be wrong; prefer the numbers):\n" + knowledge[:3000]
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _first_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    blob = m.group(0)
    for end in range(len(blob), 0, -1):  # tolerate trailing chatter after the object
        try:
            v = json.loads(blob[:end])
            return v if isinstance(v, dict) else {}
        except ValueError:
            continue
    return {}


def parse_advice(text: str, lang: str) -> CoachAdvice:
    raw = _first_json(text)
    adv = CoachAdvice()
    for k, v in (raw.get("tactic_priors") or {}).items():
        if k in TACTICS and isinstance(v, int | float):
            adv.tactic_priors[k] = _clamp(float(v), *PRIOR)
        else:
            adv.rejected.append(f"prior:{k}")
    for k, v in (raw.get("param_nudges") or {}).items():
        if k in LIMITS and isinstance(v, int | float):
            adv.param_nudges[k] = _clamp(float(v), *MULT)
        else:
            adv.rejected.append(f"param:{k}")
    comment = raw.get(f"comment_{lang}") or raw.get("comment_en") or raw.get("comment") or ""
    adv.comment = str(comment).strip()[:300]
    return adv


def _get(params: PolicyParams, dotted: str) -> float:
    section, name = dotted.split(".")
    return float(getattr(getattr(params, section), name))


def apply_advice(adv: CoachAdvice, base: PolicyParams, calibration_dir: Path) -> dict[str, Any]:
    """Merge into calibration/policy_overrides.yaml and tactic_priors.yaml, inside the absolute limits."""
    calibration_dir.mkdir(parents=True, exist_ok=True)
    over_path = calibration_dir / "policy_overrides.yaml"
    overrides = read_yaml(over_path)
    changed: dict[str, Any] = {}
    for dotted, mult in adv.param_nudges.items():
        lo, hi = LIMITS[dotted]
        new = _clamp(_get(base, dotted) * mult, lo, hi)
        value: float | int = round(new) if dotted.endswith("_ms") else round(new, 4)
        section, name = dotted.split(".")
        overrides = deep_merge(overrides, {section: {name: value}})
        changed[dotted] = value
    if changed:
        over_path.write_text(yaml.safe_dump(overrides, sort_keys=True), encoding="utf-8")
    pri_path = calibration_dir / "tactic_priors.yaml"
    priors = {k: float(v) for k, v in read_yaml(pri_path).items() if k in TACTICS}
    for k, v in adv.tactic_priors.items():
        priors[k] = round(_clamp(PRIOR_KEEP * priors.get(k, 0.0) + (1 - PRIOR_KEEP) * v, *PRIOR), 4)
    if adv.tactic_priors:
        pri_path.write_text(yaml.safe_dump(priors, sort_keys=True), encoding="utf-8")
    with (calibration_dir / "coach_log.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"t": time.time(), "params": changed, "priors": adv.tactic_priors, "rejected": adv.rejected,
                            "comment": adv.comment}, ensure_ascii=False) + "\n")
    return {"params": changed, "priors": priors}


def load_coach_priors(calibration_dir: Path) -> dict[str, float]:
    return {k: float(v) for k, v in read_yaml(calibration_dir / "tactic_priors.yaml").items() if k in TACTICS}


async def coach_match(broker, summary: dict[str, Any], lang: str, calibration_dir: Path, base: PolicyParams,
                      knowledge: str = "") -> tuple[CoachAdvice, dict[str, Any]]:
    res = await broker.complete("coach", build_messages(summary, lang, knowledge))
    adv = parse_advice(res.text, lang)
    return adv, apply_advice(adv, base, calibration_dir)
