from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

from fixtures.hud_synth import DEFAULT_ROIS, ubg_like
from jarvis.fight.keymap import load_keymap
from jarvis.fight.perception.layout import BarSpec, ColorRange, HudLayout
from jarvis.fight.policy.params import load_params
from jarvis.fight.recorder import SessionRecorder
from jarvis.fight.replay import render_overlay, replay_session, shadow_report
from jarvis.learn.datasets import build_samples, crop_box, read_manifest, split_of, write_manifest
from jarvis.learn.hindsight import attack_outcomes, fit_range, label_session, two_means_threshold
from jarvis.learn.registry import ModelRegistry

MS = 1_000_000
FRAME_NS = 16_666_667


def write_session(d, n_frames=120, hits=(), dealt=(), inputs=(), opp=(0.55, 0.45, 0.15, 0.45), gaps=None):
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"kind": "fight"}))
    with (d / "frames.jsonl").open("w") as f:
        for i in range(n_frames):
            row = {"vi": i, "t_ns": i * FRAME_NS, "opp": list(opp), "combat_hud": True}
            if gaps is not None:
                row["gap"] = gaps(i)
            f.write(json.dumps(row) + "\n")
    with (d / "events.jsonl").open("w") as f:
        for t_ms, amount in hits:
            f.write(json.dumps({"t_ns": int(t_ms * MS), "type": "hit_taken", "amount": amount}) + "\n")
        for t_ms in dealt:
            f.write(json.dumps({"t_ns": int(t_ms * MS), "type": "hit_dealt", "amount": 0.05}) + "\n")
    with (d / "inputs.jsonl").open("w") as f:
        for t_ms, action in inputs:
            f.write(json.dumps({"t_ns": int(t_ms * MS), "action": action, "event": "down", "ok": True}) + "\n")
    return d


def test_two_means():
    assert two_means_threshold([0.04, 0.05, 0.05, 0.06, 0.2, 0.22, 0.21]) == pytest.approx(0.13, abs=0.03)
    assert two_means_threshold([0.05, 0.05, 0.05, 0.05]) is None


def test_hindsight_labels(tmp_path):
    d = write_session(tmp_path / "s_fight", hits=[(1000, 0.05), (1500, 0.05), (1700, 0.21), (1900, 0.05), (1990, 0.2)])
    lab = label_session(d, event_lag_ms=0)
    i = 1000 * MS // FRAME_NS  # frame just before the first hit (t=983 ms)
    assert lab.y[100][i] == 1 and lab.y[500][i] == 1
    far = 520 * MS // FRAME_NS  # t=517 ms: next hit ~483 ms away
    assert lab.y[300][far] == 0 and lab.y[500][far] == 1
    heavy = [h for h in lab.hits if h.severity == "heavy"]
    assert len(heavy) == 2
    st = lab.stats()
    assert st["hits"] == 5 and st["heavy_hits"] == 2


def test_range_fit(tmp_path):
    rng = np.random.default_rng(0)
    inputs, dealt = [], []
    for k in range(60):
        t = 100 + k * 400  # real jab chains are ~250-400 ms apart
        inputs.append((t, "light"))
    gaps = lambda i: (i % 50) / 50  # noqa: E731
    for t, _ in inputs:
        i = int(t * MS // FRAME_NS)
        if gaps(i) < 0.4 and rng.random() < 0.9:
            dealt.append(t + 100)
    d = write_session(tmp_path / "r_fight", n_frames=1500, dealt=dealt, inputs=inputs, gaps=gaps)
    fit = fit_range(attack_outcomes(d))
    assert fit["samples"] == 60
    assert fit["attack_gap"] is not None and 0.25 <= fit["attack_gap"] <= 0.5


def test_dataset_split_is_by_session(tmp_path):
    names = [f"2026_{i}_fight" for i in range(200)]
    splits = [split_of(n) for n in names]
    assert splits == [split_of(n) for n in names]  # deterministic
    assert 0.55 < splits.count("train") / 200 < 0.85
    d = write_session(tmp_path / "x_fight", hits=[(500, 0.05)])
    samples = build_samples(d, window=8, stride=4)
    assert samples and len({s.split for s in samples}) == 1
    assert all(len(s.vis) == 8 and s.end_vi == s.vis[-1] for s in samples)
    p = write_manifest(samples, tmp_path / "ds", "v1")
    back = read_manifest(p)
    assert len(back) == len(samples) and back[0].y == samples[0].y


def test_crop_box_edges():
    img = np.full((540, 960, 3), 200, np.uint8)
    c = crop_box(img, [0.01, 0.01, 0.2, 0.4], 64)
    assert c.shape == (64, 64, 3) and (c == 0).any()  # padded outside the frame


def test_registry_promote_and_rollback(tmp_path):
    reg = ModelRegistry(tmp_path / "models")
    m1 = tmp_path / "a.onnx"
    m1.write_bytes(b"model-one")
    m2 = tmp_path / "b.onnx"
    m2.write_bytes(b"model-two")
    e1 = reg.register("threat", m1, dataset_version="d1", metrics={"auc": 0.7})
    e2 = reg.register("threat", m2, dataset_version="d2", metrics={"auc": 0.8})
    assert (e1.version, e2.version) == (1, 2) and e1.status == "candidate"
    reg.promote("threat", 1)
    reg.promote("threat", 2)
    assert (tmp_path / "models" / "threat" / "active.onnx").read_bytes() == b"model-two"
    assert reg.active("threat").version == 2
    back = reg.rollback("threat")
    assert back.version == 1 and (tmp_path / "models" / "threat" / "active.onnx").read_bytes() == b"model-one"
    assert ModelRegistry(tmp_path / "models").active("threat").version == 1  # persisted
    (tmp_path / "models" / e2.path).write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        reg.promote("threat", 2)


GREEN_R = [ColorRange(h=(45, 75), s=(100, 255), v=(80, 255))]


def record_synthetic_fight(root, n=90):
    rec = SessionRecorder(root, "fight", width=480, height=270, encoder="libx264")
    for i in range(n):
        own = 1.0 - 0.1 * (i // 30)
        img = ubg_like(own_hp=own, opp_hp=1.0 - 0.005 * i, size=(960, 540))
        rec.add_frame(img, {"t_ns": i * FRAME_NS, "fi": i, "state": "fighting"})
    rec.log("events", {"t_ns": 30 * FRAME_NS, "type": "hit_taken", "amount": 0.1})
    rec.log("events", {"t_ns": 60 * FRAME_NS, "type": "hit_taken", "amount": 0.1})
    rec.log("inputs", {"t_ns": 10 * FRAME_NS, "action": "light", "event": "down", "ok": True})
    return rec.close({"result": "synthetic"})


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg missing")
def test_replay_is_deterministic_and_overlay_renders(tmp_path):
    d = record_synthetic_fight(tmp_path)
    lay = HudLayout(
        frame_size=(960, 540),
        bars={"own_hp": BarSpec(roi=DEFAULT_ROIS["own_hp"], fill=GREEN_R, required=True),
              "opp_hp": BarSpec(roi=DEFAULT_ROIS["opp_hp"], fill=GREEN_R, required=True)},
    )
    a = replay_session(d, lay, load_keymap(), load_params(), seed=3)
    b = replay_session(d, lay, load_keymap(), load_params(), seed=3)
    assert a.frames == 90 and a.digest() == b.digest()
    assert any(r["combat_hud"] for r in a.rows)
    rep = shadow_report(a, d)
    assert rep["hits_taken"] == 2
    out = render_overlay(d, tmp_path / "overlay.mp4", lay)
    assert out.exists() and out.stat().st_size > 1000
