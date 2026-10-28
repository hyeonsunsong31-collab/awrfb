from __future__ import annotations

import numpy as np
import pytest

from fixtures.hud_synth import scene, ubg_like
from jarvis.fight.perception.detector import Detection, decode_yolo, iou, letterbox
from jarvis.fight.perception.layout import QueueBoxSpec
from jarvis.fight.perception.queue_box import find_boxes
from jarvis.fight.perception.self_id import SelfIdentifier
from jarvis.fight.perception.threat import HeuristicThreat, gap_from_tracks
from jarvis.fight.perception.tracker import Tracker
from jarvis.fight.perception.ui_guard import UiGuard

MS = 1_000_000


# ------------------------------------------------------------------ UI guard
def test_modal_dim_detected_after_fight_baseline():
    g = UiGuard(None)
    fight = ubg_like(own_hp=0.7, opp_hp=0.6, size=(960, 540))
    for _ in range(60):
        assert g.check(fight, combat_hud=True).popup is None
    dimmed = (fight.astype(np.float32) * 0.35).astype(np.uint8)
    H, W = dimmed.shape[:2]
    dimmed[H // 3 : 2 * H // 3, W // 3 : 2 * W // 3] = (200, 200, 200)  # dialog panel
    assert g.check(dimmed, combat_hud=False).popup == "modal_dialog"


def test_popup_template(tmp_path):
    shop = scene(960, 540, seed=42)
    UiGuard.save_popup_template(shop, tmp_path / "popups", "shop")
    g = UiGuard(None, tmp_path / "popups")
    assert g.check(shop, combat_hud=False).popup == "shop"
    assert g.check(ubg_like(size=(960, 540)), combat_hud=True).popup is None


# ------------------------------------------------------------------ queue boxes
def floor_with_boxes(boxes):
    img = np.full((540, 960, 3), 90, np.uint8)
    for (cx, cy, color) in boxes:
        x, y = int(cx * 960), int(cy * 540)
        img[y - 25 : y + 25, x - 50 : x + 50] = color
    return img


def test_queue_boxes():
    blue, green = (220, 120, 30), (60, 200, 60)
    st = find_boxes(floor_with_boxes([(0.3, 0.6, blue), (0.7, 0.7, blue)]), QueueBoxSpec())
    assert len([b for b in st.boxes if b.kind == "free"]) == 2 and not st.queued
    assert st.nearest_free() is not None
    st = find_boxes(floor_with_boxes([(0.5, 0.75, green)]), QueueBoxSpec())
    assert st.queued


# ------------------------------------------------------------------ detector decode
def test_letterbox_and_decode_roundtrip():
    img = np.zeros((540, 960, 3), np.uint8)
    lb, r, pad = letterbox(img, 640)
    assert lb.shape == (640, 640, 3)
    # One box at frame (480, 270) size 96x216 -> model space
    cx, cy, w, h = 480 * r + pad[0], 270 * r + pad[1], 96 * r, 216 * r
    out = np.zeros((1, 5, 3), np.float32)
    out[0, :, 0] = [cx, cy, w, h, 0.9]
    out[0, :, 1] = [cx + 2, cy, w, h, 0.8]  # duplicate -> NMS
    out[0, :, 2] = [10, 10, 5, 5, 0.1]  # below threshold
    dets = decode_yolo(out, r, pad, img.shape[:2], 0.25, None)
    assert len(dets) == 1
    d = dets[0]
    assert d.cx == pytest.approx(0.5, abs=0.01) and d.h == pytest.approx(0.4, abs=0.01)


def test_iou():
    a = Detection(0, 0, 0.5, 0.5, 1)
    assert iou(a, a) == pytest.approx(1.0)
    assert iou(a, Detection(0.5, 0.5, 0.5, 0.5, 1)) == 0.0


# ------------------------------------------------------------------ tracking & identity
def test_tracker_keeps_ids_and_estimates_velocity():
    tr = Tracker()
    t = 0
    tracks = []
    for i in range(30):
        t += 16 * MS
        dets = [Detection(0.2 + 0.005 * i, 0.4, 0.1, 0.3, 0.9), Detection(0.45, 0.45, 0.12, 0.4, 0.9)]
        tracks = tr.update(dets, t) if i % 2 == 0 else tr.predict(t)
    assert len(tracks) == 2
    mover = min(tracks, key=lambda x: x.cx)
    assert mover.vx == pytest.approx(0.005 / 0.016, rel=0.3)


def test_self_identity_follow_cam():
    sid = SelfIdentifier("default")
    tr = Tracker()
    t = 0
    for _ in range(5):
        t += 16 * MS
        tracks = tr.update([Detection(0.45, 0.42, 0.1, 0.45, 0.9), Detection(0.6, 0.3, 0.06, 0.25, 0.9)], t)
        ident = sid.update(tracks, t)
    assert ident.me is not None and ident.opp is not None
    assert ident.me.h > ident.opp.h and ident.conf > 0.5


def test_gap_and_threat_heuristic():
    from jarvis.fight.perception.tracker import Track

    near = Track(1, 0.5, 0.4, 0.2, 0.55, 0, 0, 0, 0.9, 10, 0)
    far = Track(2, 0.5, 0.4, 0.05, 0.18, 0, 0, 0, 0.9, 10, 0)
    assert gap_from_tracks(None, near, "default", 0.55, 0.18) == pytest.approx(0.0)
    assert gap_from_tracks(None, far, "default", 0.55, 0.18) == pytest.approx(1.0)
    th = HeuristicThreat()
    frame = ubg_like(size=(960, 540))
    for _ in range(30):
        quiet = th.update(frame, near, 0.1, 0)
    burst = frame.copy()
    burst[150:400, 380:580] = 255 - burst[150:400, 380:580]  # sudden wind-up motion in the opponent box
    hot = th.update(burst, near, 0.1, 0)
    assert hot.p[200] > quiet.p[200] + 0.2
    assert hot.p[500] >= hot.p[200] >= hot.p[100]
    far_hot = HeuristicThreat()
    for _ in range(30):
        far_hot.update(frame, far, 0.95, 0)
    assert far_hot.update(burst, far, 0.95, 0).p[200] < hot.p[200]
