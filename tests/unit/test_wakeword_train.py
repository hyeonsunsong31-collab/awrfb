"""Personal wake word training: data building, thresholds, the multi-model detector, and (where torch and
openWakeWord's feature models are available) a full train -> ONNX -> live-detector round trip on a
synthetic phrase. Real accuracy is measured on the owner's PC with their own takes."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from jarvis.voice.wake import WakeDetector
from jarvis.voice.wakeword_train import (
    CLIP,
    SR,
    WakeData,
    change_speed,
    choose_threshold,
    make_clips,
    place,
    read_wav,
    take_problem,
    trim_speech,
    write_wav,
)

OWW = os.environ.get("JARVIS_TEST_OWW_MODELS", "")


def phrase(rng: np.random.Generator, f0: float = 320.0) -> np.ndarray:
    """A made-up 'word': a rising chirp, a short gap, then a two-harmonic tone (0.6 s)."""
    f0 *= rng.uniform(0.92, 1.08)
    t1 = np.arange(int(0.25 * SR)) / SR
    up = np.sin(2 * np.pi * (f0 * t1 + 900 * t1**2))
    gap = np.zeros(int(0.08 * SR))
    t2 = np.arange(int(0.27 * SR)) / SR
    tone = 0.6 * np.sin(2 * np.pi * 2.6 * f0 * t2) + 0.4 * np.sin(2 * np.pi * 5.2 * f0 * t2)
    x = np.concatenate([up, gap, tone]) * 0.5
    return (x * np.hanning(len(x)) ** 0.2).astype(np.float32)


def other(rng: np.random.Generator) -> np.ndarray:
    """Things that are not the word: falling chirps, steady tones, babble-like noise."""
    n = int(rng.uniform(0.4, 1.6) * SR)
    t = np.arange(n) / SR
    kind = rng.integers(3)
    if kind == 0:
        x = np.sin(2 * np.pi * (900 * t - 250 * t**2))
    elif kind == 1:
        x = np.sin(2 * np.pi * rng.uniform(150, 1200) * t)
    else:
        x = np.convolve(rng.standard_normal(n), np.ones(30) / 30, mode="same") * 3
    return (0.4 * x).astype(np.float32)


def data(seed: int = 0, n_pos: int = 12) -> WakeData:
    rng = np.random.default_rng(seed)
    return WakeData(pos=[phrase(rng) for _ in range(n_pos)], neg=[other(rng) for _ in range(30)],
                    bg=[np.convolve(rng.standard_normal(SR * 4), np.ones(8) / 8, mode="same").astype(np.float32) * 0.2])


def test_wav_roundtrip_and_resample(tmp_path):
    x = (0.3 * np.sin(2 * np.pi * 440 * np.arange(SR) / SR)).astype(np.float32)
    back = read_wav(write_wav(tmp_path / "a.wav", x))
    assert len(back) == SR and np.max(np.abs(back - x)) < 1e-3
    import wave

    with wave.open(str(tmp_path / "b.wav"), "wb") as w:  # a 44.1 kHz file is resampled on read
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(44100)
        w.writeframes((np.zeros((44100, 2)) * 0).astype(np.int16).tobytes())
    assert len(read_wav(tmp_path / "b.wav")) == SR


def test_trim_place_and_take_checks():
    rng = np.random.default_rng(1)
    w = phrase(rng)
    padded = np.concatenate([np.zeros(SR), w, np.zeros(SR)])
    t = trim_speech(padded)
    assert abs(len(t) - len(w)) < 0.2 * SR
    clip = place(w, rng, [], noise_p=0.0)
    assert clip.shape == (CLIP,)
    on = np.flatnonzero(np.abs(clip) > 0.05)
    assert 1.4 * SR <= on[-1] <= 1.95 * SR  # the phrase ends near the end of the window
    assert take_problem(np.zeros(SR)) and take_problem(np.ones(SR) * 0.999)
    assert take_problem(w) is None
    assert len(change_speed(w, 0.9)) > len(w) > len(change_speed(w, 1.1))


def test_make_clips_mix_and_labels():
    rng = np.random.default_rng(2)
    clips, y = make_clips(data(), rng, 50, 150)
    assert clips.shape == (200, CLIP) and y.sum() == 50
    assert np.all(np.abs(clips) <= 1.0)
    with pytest.raises(ValueError, match="record"):
        make_clips(WakeData(), rng, 1, 1)


def test_choose_threshold_respects_the_false_accept_budget():
    pos = np.array([0.95, 0.9, 0.85, 0.6, 0.97])
    neg = np.concatenate([np.full(995, 0.1), [0.7, 0.75, 0.8, 0.4, 0.5]])
    thr, recall, fpr = choose_threshold(pos, neg, max_fpr=0.002)
    # Two negatives (0.75, 0.8) may stay above the threshold: 2/1000 = the budget. 0.7 must be excluded.
    assert fpr <= 0.002 and 0.7 < thr <= 0.75 and recall == pytest.approx(0.8)


class TwoModels:
    """openWakeWord stand-in with the stock model and a personal one."""

    def __init__(self, seq):
        self.seq = list(seq)
        self.models = {"hey_jarvis_v0.1": None, "jarvis_ko_v0.1": None}

    def predict(self, frame):
        return self.seq.pop(0) if self.seq else {"hey_jarvis_v0.1": 0.0, "jarvis_ko_v0.1": 0.0}

    def reset(self):
        pass


def test_detector_runs_both_wake_words_with_their_own_thresholds():
    quiet = {"hey_jarvis_v0.1": 0.4, "jarvis_ko_v0.1": 0.7}  # personal model needs 0.8
    loud = {"hey_jarvis_v0.1": 0.4, "jarvis_ko_v0.1": 0.86}
    det = WakeDetector(TwoModels([quiet] * 3 + [loud] * 2), threshold=0.5, patience_frames=2,
                       thresholds={"jarvis_ko": 0.8}, clock=lambda: 100.0)
    frames = np.zeros(1280 * 5, np.float32)
    fired = det.feed(frames)
    assert fired == [0.86] and det.last_model == "jarvis_ko_v0.1"
    det2 = WakeDetector(TwoModels([{"hey_jarvis_v0.1": 0.6, "jarvis_ko_v0.1": 0.1}] * 2), threshold=0.5, patience_frames=2,
                        thresholds={"jarvis_ko": 0.8}, clock=lambda: 100.0)
    assert det2.feed(np.zeros(1280 * 2, np.float32)) == [0.6]  # the stock "hey jarvis" still works


def fake_features(clips: np.ndarray) -> np.ndarray:
    """Stand-in for openWakeWord's embeddings: log band energies in 16 time slices x 96 bands."""
    n = clips.shape[0]
    seg = clips.reshape(n, 16, CLIP // 16)
    spec = np.abs(np.fft.rfft(seg * np.hanning(seg.shape[-1]), axis=-1))[..., : 96 * 10]
    return np.log1p(spec.reshape(n, 16, 96, 10).mean(-1) * 10).astype(np.float32)


def test_train_and_export_with_stand_in_features(tmp_path):
    pytest.importorskip("torch")
    ort = pytest.importorskip("onnxruntime")
    from jarvis.voice.wakeword_train import train_wakeword

    m = train_wakeword(data(3), tmp_path, "test_word", fake_features, epochs=12, n_pos=300, n_neg=900)
    assert m["recall_heldout"] >= 0.75 and m["fpr_per_window"] <= 0.01, m
    path = Path(m["path"])
    assert path.name == "test_word_v0.1.onnx" and path.with_suffix(".json").exists()
    s = ort.InferenceSession(str(path))
    assert s.get_inputs()[0].shape == [1, 16, 96]
    out = s.run(None, {"x": np.zeros((1, 16, 96), np.float32)})[0]
    assert out.shape == (1, 1) and 0.0 <= float(out[0, 0]) <= 1.0


@pytest.mark.skipif(not OWW, reason="set JARVIS_TEST_OWW_MODELS to a models dir with openwakeword/*.onnx")
def test_real_openwakeword_round_trip(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("openwakeword")
    import shutil

    from jarvis.voice.wake import load_oww
    from jarvis.voice.wakeword_train import OwwFeatures, train_wakeword

    models = tmp_path / "models"
    shutil.copytree(Path(OWW) / "openwakeword", models / "openwakeword")
    m = train_wakeword(data(4, n_pos=15), models, "synthetic_word", OwwFeatures(models), epochs=20, n_pos=500, n_neg=1500)
    assert m["recall_heldout"] >= 0.8, m
    det = WakeDetector(load_oww(models, "hey_jarvis", ["synthetic_word"]), threshold=0.5, patience_frames=1,
                       thresholds={"synthetic_word": m["threshold"]})
    assert set(det.model.models) == {"hey_jarvis_v0.1", "synthetic_word_v0.1"}
    rng = np.random.default_rng(99)
    stream = np.concatenate([np.zeros(SR), phrase(rng), np.zeros(SR)])
    fired = []
    for i in range(0, len(stream), 1280):
        fired += det.feed(stream[i : i + 1280])
    assert fired and det.last_model.startswith("synthetic_word"), det.last_score
    # One detector running continuously, like the real one: ordinary sounds must never wake it.
    det2 = WakeDetector(load_oww(models, "hey_jarvis", ["synthetic_word"]), threshold=0.5, patience_frames=1,
                        thresholds={"synthetic_word": m["threshold"]})
    calm = np.concatenate([np.concatenate([other(rng), np.zeros(SR // 4)]) for _ in range(12)]).astype(np.float32)
    false = [f for i in range(0, len(calm), 1280) for f in det2.feed(calm[i : i + 1280])]
    assert false == [], (false, det2.last_model)
    # ...and right after a wake (reset), it is quiet again rather than firing on reset()'s noise.
    burst = np.concatenate([phrase(rng), np.zeros(SR // 2), other(rng), np.zeros(SR)]).astype(np.float32)
    fires = [f for i in range(0, len(burst), 1280) for f in det2.feed(burst[i : i + 1280])]
    assert len(fires) == 1
