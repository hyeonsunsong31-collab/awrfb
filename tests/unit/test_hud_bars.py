from __future__ import annotations

import numpy as np
import pytest

from fixtures.hud_synth import BLUE, DEFAULT_ROIS, GREEN, RED, YELLOW, SynthBar, make_frame, ubg_like
from jarvis.fight.perception.hud_bars import HudReader, estimate_fill_ranges, read_bar
from jarvis.fight.perception.layout import BarSpec, ColorRange, HudLayout, roi_to_pixels

GREEN_R = [ColorRange(h=(45, 75), s=(100, 255), v=(80, 255))]
YELLOW_R = [ColorRange(h=(20, 35), s=(100, 255), v=(80, 255))]
BLUE_R = [ColorRange(h=(95, 115), s=(100, 255), v=(80, 255))]
RED_R = [ColorRange(h=(172, 8), s=(100, 255), v=(80, 255))]


def layout() -> HudLayout:
    return HudLayout(
        frame_size=(1920, 1080),
        bars={
            "own_hp": BarSpec(roi=DEFAULT_ROIS["own_hp"], fill=GREEN_R, required=True),
            "own_stamina": BarSpec(roi=DEFAULT_ROIS["own_stamina"], fill=YELLOW_R),
            "opp_hp": BarSpec(roi=DEFAULT_ROIS["opp_hp"], fill=GREEN_R, required=True),
            "own_guard": BarSpec(roi=DEFAULT_ROIS["own_guard"], fill=BLUE_R),
        },
    )


@pytest.mark.parametrize("value", [0.0, 0.07, 0.25, 0.5, 0.731, 0.96, 1.0])
@pytest.mark.parametrize("direction", ["ltr", "rtl"])
def test_values_both_directions(value, direction):
    roi = (0.1, 0.5, 0.3, 0.02)
    img = make_frame({"b": SynthBar(roi, value, GREEN, direction)})
    r = read_bar(img, BarSpec(roi=roi, fill=GREEN_R, direction="auto"))
    assert r.value == pytest.approx(value, abs=0.02)
    assert r.conf > 0.9
    if 0.05 < value < 0.95:
        assert r.direction == direction


@pytest.mark.parametrize("value,trail", [(0.4, 0.15), (0.7, 0.2), (0.1, 0.3)])
def test_damage_trail_is_not_counted(value, trail):
    roi = (0.1, 0.5, 0.3, 0.02)
    img = make_frame({"b": SynthBar(roi, value, RED, "ltr", trail=trail)})
    r = read_bar(img, BarSpec(roi=roi, fill=RED_R))
    assert r.value == pytest.approx(value, abs=0.02)
    # Bright trail columns are "unknown", so a big trail lowers confidence - but stays above the
    # default acceptance threshold (popups covering the bar must NOT look like trails).
    assert r.conf >= 0.6


@pytest.mark.parametrize("q,blur", [(50, 0.0), (35, 1.0), (80, 1.5)])
def test_compression_and_blur(q, blur):
    img = ubg_like(own_hp=0.62, opp_hp=0.33, stamina=0.8, guard=0.45, jpeg_quality=q, blur=blur)
    rd = HudReader(layout()).read(img)
    assert rd.combat_hud
    assert rd.values["own_hp"] == pytest.approx(0.62, abs=0.03)
    assert rd.values["opp_hp"] == pytest.approx(0.33, abs=0.03)
    assert rd.values["own_stamina"] == pytest.approx(0.8, abs=0.04)
    assert rd.values["own_guard"] == pytest.approx(0.45, abs=0.04)


@pytest.mark.parametrize("scale", [0.5, 0.75, 1.5])
def test_resolution_scaling(scale):
    img = ubg_like(own_hp=0.4, opp_hp=0.9, scale=scale)
    rd = HudReader(layout()).read(img)
    assert rd.size_ok and rd.combat_hud
    assert rd.values["own_hp"] == pytest.approx(0.4, abs=0.03)
    assert rd.values["opp_hp"] == pytest.approx(0.9, abs=0.03)


def test_black_loading_screen_is_not_a_hud():
    rd = HudReader(layout()).read(np.zeros((1080, 1920, 3), np.uint8))
    assert not rd.combat_hud


def test_popup_covering_bars_closes_hud():
    img = ubg_like(own_hp=0.7, opp_hp=0.7)
    x, y, w, h = roi_to_pixels(DEFAULT_ROIS["own_hp"], img.shape)
    img[y - 40 : y + h + 40, x - 40 : x + w + 400] = (230, 230, 230)  # a white dialog over the bars
    rd = HudReader(layout()).read(img)
    assert not rd.combat_hud


def test_dimmed_menu_closes_hud():
    img = (ubg_like(own_hp=0.7, opp_hp=0.7).astype(np.float32) * 0.3).astype(np.uint8)
    assert not HudReader(layout()).read(img).combat_hud


def test_wrong_aspect_ratio():
    img = ubg_like(size=(1600, 1200))
    rd = HudReader(layout()).read(img)
    assert not rd.size_ok and not rd.combat_hud


@pytest.mark.parametrize("color", [GREEN, RED, YELLOW, BLUE])
def test_estimate_fill_ranges_matches_truth(color):
    roi = (0.1, 0.5, 0.3, 0.02)
    full = make_frame({"b": SynthBar(roi, 1.0, color)})
    x, y, w, h = roi_to_pixels(roi, full.shape)
    est = estimate_fill_ranges(full[y : y + h, x : x + w])
    for v in (0.2, 0.55, 0.9):
        img = make_frame({"b": SynthBar(roi, v, color, trail=0.1)}, seed=3)
        assert read_bar(img, BarSpec(roi=roi, fill=est)).value == pytest.approx(v, abs=0.03)


def _anchor_layout_and_template():
    import cv2

    from fixtures.hud_synth import ANCHOR_ROI, ubg_with_anchor
    from jarvis.fight.perception.layout import AnchorSpec

    lay = layout()
    lay.anchors = [AnchorSpec(name="name_tag", roi=ANCHOR_ROI, template_file="anchors/name_tag.png")]
    clean = ubg_with_anchor()
    x, y, w, h = roi_to_pixels(ANCHOR_ROI, clean.shape)
    tmpl = cv2.cvtColor(clean[y : y + h, x : x + w], cv2.COLOR_BGR2GRAY)
    return lay, tmpl


def test_anchor_confirms_hud_and_dark_popup_fails():
    from fixtures.hud_synth import ubg_with_anchor
    from jarvis.fight.perception.hud_bars import anchor_texture_ok

    lay, tmpl = _anchor_layout_and_template()
    assert anchor_texture_ok(tmpl)
    rd = HudReader(lay, anchor_templates={"name_tag": tmpl})
    ok = rd.read(ubg_with_anchor(own_hp=0.5, opp_hp=0.8, seed=7))
    assert ok.combat_hud and ok.anchors["name_tag"] > 0.9
    # A DARK dialog over our bar + name tag: the bar alone would read "empty", the anchor catches it.
    img = ubg_with_anchor(own_hp=0.5, opp_hp=0.8)
    H, W = img.shape[:2]
    img[int(0.84 * H) : int(0.98 * H), int(0.02 * W) : int(0.45 * W)] = (30, 30, 34)
    bad = rd.read(img)
    assert not bad.combat_hud


@pytest.mark.parametrize("scale", [0.75, 1.25])
def test_anchor_survives_scaling(scale):
    from fixtures.hud_synth import ubg_with_anchor

    lay, tmpl = _anchor_layout_and_template()
    rd = HudReader(lay, anchor_templates={"name_tag": tmpl})
    assert rd.read(ubg_with_anchor(scale=scale)).combat_hud


def test_white_popup_over_bar_fails_even_without_anchor():
    img = ubg_like(own_hp=0.7, opp_hp=0.7)
    x, y, w, h = roi_to_pixels(DEFAULT_ROIS["own_hp"], img.shape)
    img[y - 10 : y + h + 10, x - 10 : x + w + 10] = (235, 235, 235)
    rd = HudReader(layout()).read(img)
    assert rd.conf["own_hp"] < 0.3 and not rd.combat_hud
