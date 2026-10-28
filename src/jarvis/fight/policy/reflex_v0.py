"""Heuristic fighter v0: works with no training data.

Per frame it decides the holds it wants (block, approach, retreat) and diffs them against what the
actuator really holds, then adds rate-limited taps (jab chains, heavies, feints, dodges, punishes).
Every few seconds the bandit scores the current tactic by real HUD outcomes (damage dealt minus
damage taken) and may switch tactics; the outcome-based opponent model seeds its priors.

Design notes grounded in UBG mechanics (community guides, unverified numbers in policy_v0.yaml):
* stamina break (~2 s stun) is lethal -> keep a reserve, block (no stamina cost) when low;
* long blocks get guard-broken -> cap block duration, prefer spacing when guard is low;
* light punches are faster than our reaction time -> block by default at close range on threat;
* heavies break guard and punish early dodges -> heavy against opponents whose blocks absorb jabs;
* no perfect dodge ~1 s after dashing -> dash lockout.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from jarvis.fight.actuator import Command
from jarvis.fight.keymap import Keymap
from jarvis.fight.perception.pipeline import Observation
from jarvis.fight.policy.bandit import DiscountedThompson
from jarvis.fight.policy.opponent import OpponentModel
from jarvis.fight.policy.params import PolicyParams
from jarvis.fight.policy.tactics import DEFAULT_TACTIC, TACTICS

NS_PER_MS = 1_000_000
HOLDABLE = ("block", "move_forward", "move_back", "move_left", "move_right")


@dataclass
class FightCtx:
    holds_since: dict[str, int] = field(default_factory=dict)
    last_tap: dict[str, int] = field(default_factory=dict)
    jab_chain: int = 0
    chain_pause_until: int = 0
    not_landed_streak: int = 0
    last_seen_opp: int = 0
    last_search: int = 0
    tactic: str = DEFAULT_TACTIC
    tactic_since: int = 0
    period_dealt: float = 0.0
    period_taken: float = 0.0
    dodge_dir: int = 1
    decisions: dict[str, int] = field(default_factory=dict)

    def ms_since_tap(self, action: str, t: int) -> float:
        last = self.last_tap.get(action)
        return float("inf") if last is None else (t - last) / NS_PER_MS


class ReflexPolicy:
    def __init__(
        self,
        base: PolicyParams,
        keymap: Keymap,
        *,
        bandit: DiscountedThompson | None = None,
        opponent: OpponentModel | None = None,
        seed: int | None = None,
        camera_relative: bool = False,
        fixed_tactic: str | None = None,
    ) -> None:
        self.base = base
        self.km = keymap
        self.rng = random.Random(seed)
        self.bandit = bandit or DiscountedThompson(list(TACTICS), discount=base.tactics.discount, seed=seed)
        self.opponent = opponent or OpponentModel()
        self.fixed_tactic = fixed_tactic
        self.ctx = FightCtx(tactic=fixed_tactic or DEFAULT_TACTIC)
        self.params = base.with_overrides(TACTICS[self.ctx.tactic])
        self.camera_relative = camera_relative and keymap.camera.mouse_relative
        self._last_hit_taken = -(10**18)
        self._last_hit_blocked = -(10**18)
        self._last_hit_dealt = -(10**18)
        self._last_dash = -(10**18)
        self._phys_to_action = self._build_phys_map()

    # ------------------------------------------------------------------ helpers
    def _build_phys_map(self) -> dict[str, str]:
        out = {}
        for name in HOLDABLE:
            b = self.km.actions.get(name)
            if b is not None and b.kind in ("key", "mouse"):
                out[f"key:{b.key}" if b.key else f"mouse:{b.mouse}"] = name
        return out

    def usable(self, action: str) -> bool:
        return self.km.usable(action)

    def reset(self, t_ns: int) -> None:
        self.ctx = FightCtx(tactic=self.ctx.tactic, tactic_since=t_ns)
        self.bandit.set_priors(self.opponent.tactic_priors(), weight=1.0)

    def _tap(self, cmds: list[Command], action: str, t: int, reason: str, dur_ms: float | None = None) -> None:
        if not self.usable(action):
            return
        cmds.append(Command(action, "tap", dur_ms=dur_ms, reason=reason))
        self.ctx.last_tap[action] = t
        self.ctx.decisions[reason] = self.ctx.decisions.get(reason, 0) + 1
        self.opponent.note_our_action(t, action)

    def _maybe_switch_tactic(self, t: int) -> None:
        P = self.base.tactics
        if self.fixed_tactic is not None:
            return
        if self.ctx.tactic_since == 0:
            self.ctx.tactic_since = t
            return
        if (t - self.ctx.tactic_since) < P.period_ms * NS_PER_MS:
            return
        reward = self.ctx.period_dealt - self.ctx.period_taken
        self.bandit.update(self.ctx.tactic, reward)
        self.ctx.period_dealt = self.ctx.period_taken = 0.0
        new = self.bandit.choose()
        if new != self.ctx.tactic:
            self.ctx.decisions[f"tactic:{new}"] = self.ctx.decisions.get(f"tactic:{new}", 0) + 1
        self.ctx.tactic = new
        self.ctx.tactic_since = t
        self.params = self.base.with_overrides(TACTICS[new])

    # ------------------------------------------------------------------ main
    def step(self, obs: Observation, held_phys: frozenset[str]) -> list[Command]:
        t = obs.t_ns
        P = self.params
        ctx = self.ctx
        cmds: list[Command] = []

        for e in obs.events:
            if e.type == "hit_taken":
                self._last_hit_taken = t
                ctx.period_taken += e.amount
            elif e.type == "hit_blocked":
                self._last_hit_blocked = t
            elif e.type == "hit_dealt":
                self._last_hit_dealt = t
                ctx.period_dealt += e.amount
                ctx.not_landed_streak = 0
            elif e.type == "stock_lost":
                ctx.period_taken += 1.0
            elif e.type == "stock_won":
                ctx.period_dealt += 1.0
        self.opponent.update(t, obs.events, obs.gap, obs.approach, obs.hud.get("opp_hp"))
        self._maybe_switch_tactic(t)

        held = {self._phys_to_action[p] for p in held_phys if p in self._phys_to_action}
        stamina = obs.hud.get("own_stamina", 1.0)
        guard = obs.hud.get("own_guard", 1.0)
        p200 = obs.threat.get(200, 0.0)
        if obs.opp is not None:
            ctx.last_seen_opp = t
        gap = obs.gap
        ms = lambda since: (t - since) / NS_PER_MS  # noqa: E731

        # ---------------------------------------------------------- 1. defence
        want: set[str] = set()
        block_since = ctx.holds_since.get("block")
        blocking = "block" in held
        if blocking and block_since is not None:
            block_ms = ms(block_since)
        else:
            # Unknown start (e.g. held before a reset): treat the minimum hold as already satisfied.
            block_ms = P.threat.min_block_ms if blocking else 0.0
        near = gap is None or gap <= P.spacing.attack_gap + 0.1
        if ms(self._last_hit_taken) < P.threat.hit_reaction_block_ms:
            want.add("block")
        if p200 >= P.threat.block_p and near:
            want.add("block")
        if blocking and p200 > P.threat.block_release_p and block_ms < P.threat.max_block_ms:
            want.add("block")
        if blocking and block_ms < P.threat.min_block_ms:
            want.add("block")
        if stamina <= P.stamina.critical:
            want.add("block")
            want.add("move_back")
        if "block" in want and block_ms >= P.threat.max_block_ms and guard <= P.threat.guard_low:
            # About to be guard-broken: let go and make space instead.
            want.discard("block")
            want.add("move_back")

        # ---------------------------------------------------------- 2. reactive dodge
        dashed_ms = ms(self._last_dash)
        if (
            self.base.dash.enabled
            and p200 >= P.threat.dodge_p
            and stamina > P.stamina.reserve
            and dashed_ms > max(P.dash.min_interval_ms, P.dash.lockout_after_dash_ms)
            and near
        ):
            action = "dash_left" if ctx.dodge_dir > 0 else "dash_right"
            if self.usable(action):
                ctx.dodge_dir *= -1
                want.discard("block")
                self._last_dash = t
                self._tap(cmds, action, t, "dodge")

        # ---------------------------------------------------------- 3. offence
        attacking = False
        in_range = gap is not None and gap <= P.spacing.attack_gap
        if gap is None and min(ms(self._last_hit_taken), ms(self._last_hit_dealt)) < 1500:
            # HUD-only fallback (no detector / opponent not located): we just traded hits, so they
            # are within reach.
            in_range = True
        can_attack = "block" not in want and stamina > P.stamina.reserve and t >= ctx.chain_pause_until
        if can_attack and in_range:
            if ms(self._last_hit_blocked) < P.attack.punish_window_ms and ctx.ms_since_tap("heavy", t) > 600:
                self._tap(cmds, "heavy", t, "punish_after_block", P.attack.heavy_tap_ms)
                attacking = True
            elif ctx.ms_since_tap("light", t) >= P.attack.jab_interval_ms and ctx.ms_since_tap("heavy", t) > 450:
                r = self.rng.random()
                if ctx.not_landed_streak >= P.attack.guard_break_after_blocked and self.usable("heavy"):
                    self._tap(cmds, "heavy", t, "guard_break", P.attack.heavy_tap_ms)
                    ctx.not_landed_streak = 0
                elif r < P.attack.feint_chance and self.usable("feint"):
                    self._tap(cmds, "feint", t, "feint")
                elif r < P.attack.feint_chance + P.attack.heavy_chance:
                    self._tap(cmds, "heavy", t, "heavy_mix", P.attack.heavy_tap_ms)
                else:
                    self._tap(cmds, "light", t, "jab")
                    ctx.jab_chain += 1
                    if ms(self._last_hit_dealt) > P.attack.jab_interval_ms * 1.5:
                        ctx.not_landed_streak += 1
                    if ctx.jab_chain >= P.attack.max_jab_chain:
                        ctx.jab_chain = 0
                        ctx.chain_pause_until = t + int(P.attack.chain_pause_ms * NS_PER_MS)
                attacking = True

        # ---------------------------------------------------------- 4. spacing
        if gap is not None and "move_back" not in want:
            if gap > P.spacing.preferred_gap + P.movement.approach_when_gap_over and not attacking:
                want.add("move_forward")
            elif gap < P.spacing.retreat_gap and stamina < P.stamina.reserve:
                want.add("move_back")
        elif gap is None and ms(ctx.last_seen_opp) > 1000 and ms(ctx.last_search) > 1500:
            # Opponent not visible: a short step forward to find them (never wander far).
            self._tap(cmds, "move_forward", t, "search", 120)
            ctx.last_search = t
        if "move_forward" in want and "move_back" in want:
            want.discard("move_forward")
        if "block" in want and "move_forward" in want:
            want.discard("move_forward")

        # ---------------------------------------------------------- 5. camera (only if calibrated)
        if self.camera_relative and obs.opp is not None and abs(obs.opp.cx - 0.5) > 0.12:
            dx = int((obs.opp.cx - 0.5) * 120)
            cmds.append(Command("camera", "move", dx=dx, reason="face_opponent"))

        # ---------------------------------------------------------- 6. reconcile holds
        for action in HOLDABLE:
            if not self.usable(action):
                continue
            if action in want and action not in held:
                cmds.insert(0, Command(action, "hold", reason="hold"))
                ctx.holds_since[action] = t
            elif action not in want and action in held:
                cmds.insert(0, Command(action, "release", reason="release"))
                ctx.holds_since.pop(action, None)
        return cmds

    def summary(self) -> dict:
        return {
            "tactic": self.ctx.tactic,
            "bandit": self.bandit.summary(),
            "opponent": self.opponent.profile.radar(),
            "decisions": dict(self.ctx.decisions),
        }
