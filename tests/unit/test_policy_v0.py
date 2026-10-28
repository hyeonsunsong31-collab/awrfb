"""Table-style tests for the v0 reflex policy: observation in -> commands out."""

from __future__ import annotations

import pytest

from jarvis.fight.events import FightEvent
from jarvis.fight.keymap import load_keymap
from jarvis.fight.perception.hud_bars import HudReading
from jarvis.fight.perception.pipeline import Observation
from jarvis.fight.perception.tracker import Track
from jarvis.fight.perception.ui_guard import UiGuardResult
from jarvis.fight.policy.bandit import DiscountedThompson
from jarvis.fight.policy.opponent import OpponentModel
from jarvis.fight.policy.params import load_params
from jarvis.fight.policy.reflex_v0 import ReflexPolicy

MS = 1_000_000
OPP = Track(2, 0.55, 0.4, 0.15, 0.45, 0, 0, 0, 0.9, 10, 0)


def obs(t_ms, *, gap=0.2, p200=0.1, stamina=1.0, guard=1.0, events=(), opp=OPP):
    return Observation(
        t_ns=int(t_ms * MS),
        index=int(t_ms),
        hud=HudReading(int(t_ms * MS), {"own_hp": 1.0, "opp_hp": 1.0, "own_stamina": stamina, "own_guard": guard}, combat_hud=True),
        ui=UiGuardResult(),
        events=[FightEvent(int(t_ms * MS), e[0], e[1] if len(e) > 1 else 0.0) for e in events],
        opp=opp,
        gap=gap,
        threat={100: p200 * 0.7, 200: p200, 300: p200, 500: p200},
    )


def make_policy(**over):
    params = load_params().with_overrides(over)
    pol = ReflexPolicy(params, load_keymap(), seed=1, fixed_tactic="baseline")
    pol.reset(1)
    return pol


def names(cmds):
    return [(c.action, c.mode) for c in cmds]


def test_threat_near_holds_block_and_no_attack():
    pol = make_policy()
    cmds = pol.step(obs(1000, p200=0.8), frozenset())
    assert ("block", "hold") in names(cmds)
    assert not any(c.action in ("light", "heavy") for c in cmds)


def test_hit_taken_triggers_block():
    pol = make_policy()
    cmds = pol.step(obs(1000, p200=0.0, events=[("hit_taken", 0.1)]), frozenset())
    assert ("block", "hold") in names(cmds)


def test_jab_when_in_range_and_safe():
    pol = make_policy(attack={"heavy_chance": 0.0, "feint_chance": 0.0})
    cmds = pol.step(obs(1000), frozenset())
    assert ("light", "tap") in names(cmds)
    # Rate limited: no second jab 100 ms later.
    assert ("light", "tap") not in names(pol.step(obs(1100), frozenset()))
    assert ("light", "tap") in names(pol.step(obs(1300), frozenset()))


def test_jab_chain_pauses():
    pol = make_policy(attack={"heavy_chance": 0.0, "feint_chance": 0.0, "max_jab_chain": 2, "guard_break_after_blocked": 99})
    t = 1000
    jabs = 0
    for _ in range(4):
        jabs += names(pol.step(obs(t), frozenset())).count(("light", "tap"))
        t += 270
    assert jabs < 4  # a pause was inserted after the chain


def test_punish_after_blocked_hit():
    pol = make_policy()
    # We were blocking (F held) and a hit landed on the block: let go and punish immediately.
    cmds = pol.step(obs(1000, events=[("hit_blocked", 0.05)], p200=0.0), frozenset({"key:f"}))
    assert ("block", "release") in names(cmds) and ("heavy", "tap") in names(cmds)
    assert names(cmds).index(("block", "release")) < names(cmds).index(("heavy", "tap"))


def test_guard_break_after_hits_not_landing():
    pol = make_policy(attack={"heavy_chance": 0.0, "feint_chance": 0.0, "guard_break_after_blocked": 2, "max_jab_chain": 10})
    t = 1000
    seen_heavy = False
    for _ in range(6):
        cmds = pol.step(obs(t), frozenset())
        seen_heavy |= ("heavy", "tap") in names(cmds)
        t += 500
    assert seen_heavy


def test_low_stamina_blocks_and_backs_off():
    pol = make_policy()
    cmds = pol.step(obs(1000, stamina=0.05), frozenset())
    assert ("block", "hold") in names(cmds) and ("move_back", "hold") in names(cmds)
    assert not any(c.action in ("light", "heavy", "feint") for c in cmds)


def test_long_block_with_low_guard_lets_go():
    pol = make_policy(dash={"enabled": False})
    pol.step(obs(1000, p200=0.9, guard=0.1), frozenset())  # starts holding block
    cmds = pol.step(obs(2000, p200=0.9, guard=0.1), frozenset({"key:f"}))
    assert ("block", "release") in names(cmds) and ("move_back", "hold") in names(cmds)


def test_approach_when_far_and_release_when_close():
    pol = make_policy()
    cmds = pol.step(obs(1000, gap=0.8, p200=0.0), frozenset())
    assert ("move_forward", "hold") in names(cmds)
    cmds = pol.step(obs(1100, gap=0.2, p200=0.0), frozenset({"key:w"}))
    assert ("move_forward", "release") in names(cmds)


def test_dodge_respects_lockout_and_keymap():
    km = load_keymap()
    pol = make_policy()
    cmds = pol.step(obs(5000, p200=0.95), frozenset())
    assert any(c.action in ("dash_left", "dash_right") for c in cmds)
    cmds = pol.step(obs(5300, p200=0.95), frozenset())
    assert not any(c.action.startswith("dash") for c in cmds)  # locked out
    km.actions["dash"].enabled = False
    pol2 = ReflexPolicy(load_params(), km, seed=1, fixed_tactic="baseline")
    pol2.reset(1)
    assert not any(c.action.startswith("dash") for c in pol2.step(obs(5000, p200=0.95), frozenset()))


def test_disabled_actions_never_emitted():
    km = load_keymap()
    km.actions["feint"].enabled = False
    params = load_params().with_overrides({"attack": {"feint_chance": 1.0}})
    pol = ReflexPolicy(params, km, seed=1, fixed_tactic="baseline")
    pol.reset(1)
    t = 1000
    for _ in range(20):
        assert not any(c.action == "feint" for c in pol.step(obs(t), frozenset()))
        t += 300


def test_hud_only_fallback_attacks_after_trading_hits():
    pol = make_policy(attack={"heavy_chance": 0.0, "feint_chance": 0.0})
    assert ("light", "tap") not in names(pol.step(obs(1000, gap=None, opp=None, p200=0.0), frozenset()))
    pol.step(obs(1100, gap=None, opp=None, p200=0.0, events=[("hit_dealt", 0.05)]), frozenset())
    assert ("light", "tap") in names(pol.step(obs(1500, gap=None, opp=None, p200=0.0), frozenset()))


def test_bandit_learns_best_tactic_and_adapts():
    b = DiscountedThompson(["a", "b", "c"], discount=0.95, seed=3)
    true = {"a": 0.0, "b": 0.3, "c": 0.1}
    picks = []
    for _ in range(300):
        arm = b.choose()
        picks.append(arm)
        b.update(arm, true[arm] + b.rng.gauss(0, 0.05))
    assert picks[-100:].count("b") > 70
    true = {"a": 0.5, "b": 0.0, "c": 0.1}  # opponent adapted
    for _ in range(300):
        arm = b.choose()
        picks.append(arm)
        b.update(arm, true[arm] + b.rng.gauss(0, 0.05))
    assert picks[-100:].count("a") > 70


def test_opponent_model_profiles():
    om = OpponentModel(ema=0.3)
    t = 0
    for _ in range(20):  # our attacks never land
        om.note_our_action(t, "light")
        t += 400 * MS
        om.update(t, [], 0.2, 0.0, 1.0)
    assert om.profile.defense > 0.8
    for _ in range(20):  # every heavy gets punished
        om.note_our_action(t, "heavy")
        t += 100 * MS
        om.update(t, [FightEvent(t, "hit_taken", 0.1)], 0.2, 0.05, 1.0)
    assert om.profile.punish > 0.7 and om.profile.aggression > 0.7
    pri = om.tactic_priors()
    assert pri["feint_mix"] > pri["pressure"]


@pytest.mark.parametrize("tactic", ["baseline", "pressure", "counter", "bait_punish", "keep_range", "feint_mix", "regen_retreat"])
def test_every_tactic_is_valid(tactic):
    from jarvis.fight.policy.tactics import TACTICS

    load_params().with_overrides(TACTICS[tactic])
