"""M6: learned fighting - metrics, promotion gate, coach, threat model (train -> ONNX -> live), detector data."""

from __future__ import annotations

import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml
from tests.fakes.llm import ScriptedProvider, text

from jarvis.brain.broker import LLMBroker
from jarvis.fight.policy.coach import LIMITS, apply_advice, coach_match, load_coach_priors, parse_advice
from jarvis.fight.policy.params import load_params
from jarvis.learn.evaluate import Gate, promotion_decision, threat_metrics
from jarvis.learn.metrics import average_precision, best_f1_threshold, roc_auc, warning_times
from jarvis.learn.train_detector import build_yolo_dataset


# ------------------------------------------------------------------------------------- metrics
def test_average_precision_and_auc_match_hand_computed_values():
    y = np.array([1, 0, 1, 0, 0, 1])
    s = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    # precision at each positive: 1/1, 2/3, 3/6 -> mean 0.7222
    assert average_precision(y, s) == pytest.approx((1 + 2 / 3 + 3 / 6) / 3)
    assert roc_auc(y, s) == pytest.approx(5 / 9)
    assert roc_auc(np.array([1, 0]), np.array([0.5, 0.5])) == pytest.approx(0.5)  # ties count half
    assert np.isnan(average_precision(np.zeros(3), np.ones(3)))
    assert best_f1_threshold(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(0.8)


def test_metrics_agree_with_scikit_learn_when_available():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(3)
    y = rng.random(500) < 0.2
    s = np.round(rng.random(500) * 0.6 + y * 0.3, 2)  # with ties
    assert average_precision(y, s) == pytest.approx(sk.average_precision_score(y, s))
    assert roc_auc(y, s) == pytest.approx(sk.roc_auc_score(y, s))


def test_warning_time_is_the_unbroken_alarm_before_a_hit():
    t = np.arange(0, 1000, 50.0)
    p = np.zeros_like(t)
    p[(t >= 600) & (t <= 800)] = 0.9  # alarm 600..800 ms
    p[t == 300] = 0.9  # an isolated earlier blip does not count
    assert warning_times(t, p, np.array([820.0]), 0.5) == [220.0]
    assert warning_times(t, p, np.array([500.0]), 0.5) == [0.0]


def test_threat_metrics_per_horizon():
    y = np.array([[0, 1], [1, 1], [0, 0], [1, 1]])
    p = np.array([[0.1, 0.9], [0.8, 0.7], [0.2, 0.1], [0.7, 0.8]])
    m = threat_metrics(y, p, (100, 300))
    assert m["ap_100"] == 1.0 and m["pos_300"] == 3 and m["n"] == 4


# ------------------------------------------------------------------------------------- promotion gate
def test_promotion_gate():
    good = {"ap_300": 0.81, "pos_300": 120, "warning_ms_median": 240}
    assert promotion_decision(good, None)[0]
    assert not promotion_decision({**good, "ap_300": 0.6}, None)[0]  # below the M6 bar
    assert "record more" in promotion_decision({**good, "pos_300": 5}, None)[1]
    ok, why = promotion_decision(good, None, Gate(latency_ms=210))
    assert not ok and "latency" in why  # 240 < 210 + 50
    assert not promotion_decision(good, {"ap_300": 0.805})[0]  # must beat the active model by 0.01
    assert promotion_decision(good, {"ap_300": 0.78})[0]


# ------------------------------------------------------------------------------------- coach
COACH_REPLY = ('Here is my analysis.\n{"tactic_priors": {"counter": 0.5, "pressure": -0.04, "rush_b": 0.1}, '
               '"param_nudges": {"threat.block_p": 0.5, "stamina.reserve": 1.1, "actuator.heartbeat_timeout_ms": 2.0}, '
               '"comment_ko": "상대가 헤비를 자주 씁니다. 더 일찍 막겠습니다.", "comment_en": "They throw heavies a lot."} Good luck!')


def test_coach_advice_is_clamped_and_allowlisted():
    adv = parse_advice(COACH_REPLY, "ko")
    assert adv.tactic_priors == {"counter": 0.1, "pressure": -0.04}  # 0.5 clamped to +0.1
    assert adv.param_nudges == {"threat.block_p": 0.8, "stamina.reserve": 1.1}  # x0.5 clamped to x0.8
    assert "param:actuator.heartbeat_timeout_ms" in adv.rejected and "prior:rush_b" in adv.rejected  # never safety settings
    assert adv.comment.startswith("상대가") and parse_advice(COACH_REPLY, "en").comment == "They throw heavies a lot."
    assert parse_advice("no json here", "en").tactic_priors == {}


def test_coach_changes_stay_inside_absolute_limits(tmp_path):
    base = load_params()
    adv = parse_advice(COACH_REPLY, "en")
    for _ in range(10):  # repeated advice compounds but never leaves the limits
        applied = apply_advice(adv, load_params(calibration_dir=tmp_path), tmp_path)
    lo, _hi = LIMITS["threat.block_p"]
    over = yaml.safe_load((tmp_path / "policy_overrides.yaml").read_text())
    assert base.threat.block_p * 0.8**10 < lo and over["threat"]["block_p"] == pytest.approx(lo)  # clamped at the floor
    assert over["stamina"]["reserve"] <= LIMITS["stamina.reserve"][1]
    assert load_params(calibration_dir=tmp_path).threat.block_p == over["threat"]["block_p"]
    assert load_coach_priors(tmp_path) == applied["priors"] and abs(applied["priors"]["counter"]) <= 0.1
    assert len((tmp_path / "coach_log.jsonl").read_text().splitlines()) == 10


async def test_coach_match_uses_the_coach_role(cfg, tmp_path):
    nv = ScriptedProvider("nvidia", [text(COACH_REPLY)])
    broker = LLMBroker(cfg.brain, {"nvidia": nv})
    summary = {"result": "likely_lost", "stocks_won": 1, "stocks_lost": 3, "damage_taken": 2.7, "hits": {"taken": 31},
               "policy": {"bandit": {"counter": {"n": 4}}, "opponent": {"aggression": 0.8}, "decisions": {"block": 40}}}
    adv, applied = await coach_match(broker, summary, "en", tmp_path, load_params(), knowledge="Heavies are slow.")
    assert adv.comment == "They throw heavies a lot." and applied["params"]["threat.block_p"] < load_params().threat.block_p
    sent = nv.calls[0]["messages"][1]["content"]
    assert '"stocks_lost": 3' in sent and "Heavies are slow." in sent and "player" not in sent.lower()


def test_coach_priors_reach_the_bandit():
    from jarvis.fight.keymap import load_keymap
    from jarvis.fight.policy.reflex_v0 import ReflexPolicy

    pol = ReflexPolicy(load_params(), load_keymap(), seed=1, coach_priors={"counter": 0.08})
    pol.reset(0)
    base = ReflexPolicy(load_params(), load_keymap(), seed=1)
    base.reset(0)
    assert pol.bandit.prior_mean["counter"] == pytest.approx(base.bandit.prior_mean["counter"] + 0.08)


# ------------------------------------------------------------------------------------- detector dataset
def test_detector_dataset_from_a_recorded_session(tmp_path):
    rec = tmp_path / "recordings" / "20261103-120000_fight"
    rec.mkdir(parents=True)
    w = cv2.VideoWriter(str(rec / "video.mkv"), cv2.VideoWriter_fourcc(*"MJPG"), 30, (320, 180))
    if not w.isOpened():
        pytest.skip("no video writer")
    rows = []
    for k in range(24):
        w.write(np.full((180, 320, 3), 40 + k, np.uint8))
        rows.append({"vi": k, "t_ns": k * 33_000_000, "combat_hud": k >= 6, "me": [0.4, 0.6, 0.1, 0.3, 1],
                     "opp": None if k == 12 else [0.6, 0.55, 0.1, 0.3, 2]})
    w.release()
    (rec / "frames.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    data = build_yolo_dataset(tmp_path / "recordings", tmp_path / "yolo", stride=3)
    imgs = list((tmp_path / "yolo" / "images").rglob("*.jpg"))
    labels = list((tmp_path / "yolo" / "labels").rglob("*.txt"))
    assert len(imgs) == len(labels) == 5  # frames 6, 9, 15, 18, 21 (12 has no opponent, <6 no combat HUD)
    assert labels[0].read_text().splitlines()[1].startswith("1 0.600000 0.550000")
    assert "names:\n  0: self\n  1: opponent" in data.read_text()


# ------------------------------------------------------------------------------------- threat model, for real
def test_threat_model_trains_exports_and_runs_live(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from jarvis.learn import train_threat as tt
    from jarvis.learn.threat_runtime import OnnxThreat

    rng = np.random.default_rng(0)

    def split(name: str, n: int) -> None:
        x = rng.integers(0, 60, (n, 8, 96, 96, 3), dtype=np.uint8)
        y = np.zeros((n, 4), np.uint8)
        for i in range(n):
            if i % 3 == 0:  # a growing bright shape before a hit
                for t in range(4, 8):
                    r = 6 + (t - 4) * 6
                    x[i, t, 48 - r : 48 + r, 48 - r : 48 + r, 2] = 230
                y[i] = [0, 1, 1, 1]
        np.save(tmp_path / f"X_{name}.npy", x)
        np.save(tmp_path / f"Y_{name}.npy", y)
        (tmp_path / f"meta_{name}.json").write_text(json.dumps([{"session": f"s{name}", "end_vi": i, "t_ms": i * 33.0} for i in range(n)]))

    for name, n in (("train", 90), ("val", 30), ("test", 30)):
        split(name, n)
    (tmp_path / "hits.json").write_text(json.dumps({"stest": [i * 33.0 + 100 for i in range(0, 30, 3)]}))
    net, hist = tt.train(tmp_path, epochs=2, batch=16, backbone="tiny", pretrained=False)
    assert len(hist) == 2 and hist[-1]["loss"] > 0
    m = tt.evaluate(net, tmp_path)
    assert m["n"] == 30 and m["pos_300"] == 10 and 0.0 <= m["ap_300"] <= 1.0
    model = OnnxThreat(tt.export_onnx(net, tmp_path / "threat.onnx", 96))
    assert (model.window, model.crop, model.horizons) == (8, 96, (100, 200, 300, 500))
    frame = rng.integers(0, 60, (540, 960, 3), dtype=np.uint8)
    opp = SimpleNamespace(cx=0.5, cy=0.5, w=0.1, h=0.3)
    est = None
    for k in range(10):
        est = model.update(frame, opp, 0.3, k)
    assert est is not None and est.source == "model" and set(est.p) == {100, 200, 300, 500}
    assert all(0.0 <= v <= 1.0 for v in est.p.values())
    assert OnnxThreat(tmp_path / "threat.onnx").update(frame, None, None, 0).source == "model:no-target"
