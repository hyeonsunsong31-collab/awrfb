"""Personal wake word: teach JARVIS to wake on "자비스" (or any phrase), in your own voice.

openWakeWord turns audio into speech embeddings with two frozen, pre-trained ONNX models (they come with
`jarvis voice-models`); a wake-word model is only a small classifier over 16 consecutive embeddings, which
cover about 2 s of audio. This module trains that classifier from:

* your own takes of the phrase (`jarvis wakeword record`), heavily augmented: speed, gain, room echo, and
  background noise under it;
* optional synthetic takes from the local voice engines (several voices and speeds);
* negatives: your other speech, near-miss words ("자비", "재빨리", "제비" ...), cut-off takes (the phrase
  not finished yet), background audio (game, music, video through the speakers) and plain noise.

The result is an ONNX model openWakeWord loads like its own (input [1, 16, 96]); it runs alongside
"hey jarvis", and the second-stage check (your voice + Whisper hearing the name) still applies.
"""

from __future__ import annotations

import json
import logging
import time
import wave
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

SR = 16000
CLIP = 2 * SR  # one model window: 16 embeddings x 80 ms (+ the 760 ms receptive field)
WINDOWS_PER_HOUR = 3600 / 0.08
NEAR_MISSES_KO = ["자비", "자비로운 마음", "재빨리 가자", "제비가 날아간다", "잡아 봐", "하비스", "가비스", "자비는 없다",
                  "자빠졌어", "자리 비워", "자비스럽다", "사비로 샀어", "자막 켜 줘", "잠깐만"]
NEAR_MISSES_EN = ["Travis", "service", "jar of beans", "Jarvis Street", "harvest", "nervous", "carcass", "chopsticks"]
READ_ALOUD = [
    "오늘 날씨가 정말 좋네요. 저녁에 뭐 먹을까요?",
    "자비로운 마음으로 친구를 도와줬어요.",
    "재빨리 링으로 가서 파란 칸에 서야 해.",
    "제비 한 마리가 창문 밖으로 날아갔다.",
    "Let me check the score of the last match first.",
    "I need to buy groceries after work today.",
    "잠깐만, 자막 좀 켜 줄래? 소리가 안 들려.",
    "The service at that restaurant was surprisingly fast.",
]


@dataclass
class WakeData:
    pos: list[np.ndarray] = field(default_factory=list)  # your takes of the phrase (trimmed, 16 kHz float32)
    neg: list[np.ndarray] = field(default_factory=list)  # speech without the phrase
    bg: list[np.ndarray] = field(default_factory=list)  # background audio (game, music, room)
    synth_pos: list[np.ndarray] = field(default_factory=list)
    synth_neg: list[np.ndarray] = field(default_factory=list)


# ------------------------------------------------------------------------------------- audio files
def read_wav(p: Path) -> np.ndarray:
    with wave.open(str(p), "rb") as w:
        n, ch, width, rate = w.getnframes(), w.getnchannels(), w.getsampwidth(), w.getframerate()
        raw = w.readframes(n)
    if width != 2:
        raise ValueError(f"{p.name}: only 16-bit WAV is supported")
    x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if rate != SR:
        from jarvis.voice.resample import resample

        x = resample(x, rate, SR)
    return x.astype(np.float32)


def write_wav(p: Path, x: np.ndarray) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(np.asarray(x, np.float32), -1, 1) * 32767).astype(np.int16)
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return p


def load_dir(d: Path) -> list[np.ndarray]:
    return [read_wav(p) for p in sorted(d.glob("*.wav"))] if d.exists() else []


def load_data(data_dir: Path) -> WakeData:
    return WakeData(pos=load_dir(data_dir / "pos"), neg=load_dir(data_dir / "neg"), bg=load_dir(data_dir / "bg"),
                    synth_pos=load_dir(data_dir / "synth_pos"), synth_neg=load_dir(data_dir / "synth_neg"))


# ------------------------------------------------------------------------------------- augmentation
def trim_speech(x: np.ndarray, frame: int = 320, rel: float = 0.08, pad_s: float = 0.08) -> np.ndarray:
    """Cut leading/trailing silence (energy relative to the loudest frame)."""
    x = np.asarray(x, np.float32)
    n = len(x) // frame
    if n == 0:
        return x
    e = np.sqrt(np.mean(np.square(x[: n * frame].reshape(n, frame)), axis=1))
    on = np.flatnonzero(e > rel * max(float(e.max()), 1e-6))
    if len(on) == 0:
        return x[:0]
    pad = int(pad_s * SR)
    return x[max(0, on[0] * frame - pad) : min(len(x), (on[-1] + 1) * frame + pad)]


def change_speed(x: np.ndarray, factor: float) -> np.ndarray:
    """Faster/slower (and higher/lower) speech. The rate is snapped to a multiple of 160 Hz so the
    polyphase resampler stays small (at most 100 phases)."""
    from jarvis.voice.resample import resample

    rate = round(SR * factor / 160) * 160
    return resample(x, rate, SR).astype(np.float32) if rate != SR else x


def add_reverb(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    rt60 = rng.uniform(0.15, 0.6)
    n = int(rt60 * SR)
    ir = rng.standard_normal(n).astype(np.float32) * np.exp(-6.9 * np.arange(n) / n).astype(np.float32)
    ir[0] = 1.0
    wet = np.convolve(x, ir / np.sqrt(np.sum(ir**2)))[: len(x)]
    mix = rng.uniform(0.1, 0.45)
    return ((1 - mix) * x + mix * wet).astype(np.float32)


def crop(x: np.ndarray, rng: np.random.Generator, n: int = CLIP) -> np.ndarray:
    if len(x) >= n:
        s = int(rng.integers(0, len(x) - n + 1))
        return x[s : s + n].astype(np.float32)
    out = np.zeros(n, np.float32)
    s = int(rng.integers(0, n - len(x) + 1))
    out[s : s + len(x)] = x
    return out


def bed(n: int, rng: np.random.Generator, bg: list[np.ndarray]) -> np.ndarray:
    """Something to put under a take: real background audio when we have it, else coloured noise."""
    if bg and rng.random() < 0.75:
        return crop(bg[int(rng.integers(len(bg)))], rng, n)
    white = rng.standard_normal(n).astype(np.float32)
    k = int(rng.integers(1, 40))  # crude low-pass for pink-ish noise
    return np.convolve(white, np.ones(k, np.float32) / k, mode="same").astype(np.float32)


def mix(x: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    px = float(np.mean(x**2)) + 1e-9
    pn = float(np.mean(noise**2)) + 1e-9
    return (x + noise * np.sqrt(px / (pn * 10 ** (snr_db / 10)))).astype(np.float32)


def augment_take(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    y = change_speed(x, float(rng.uniform(0.88, 1.12)))
    if rng.random() < 0.5:
        y = add_reverb(y, rng)
    return (y * 10 ** (rng.uniform(-14, 4) / 20)).astype(np.float32)


def place(take: np.ndarray, rng: np.random.Generator, bg: list[np.ndarray], end_s: tuple[float, float] = (1.45, 1.92),
          noise_p: float = 0.7) -> np.ndarray:
    """A 2 s clip whose phrase ends ``end_s`` seconds in (the model fires as the phrase completes)."""
    take = take[-int(1.4 * SR) :] if len(take) > int(1.4 * SR) else take
    clip = np.zeros(CLIP, np.float32)
    end = int(rng.uniform(*end_s) * SR)
    start = max(0, end - len(take))
    seg = take[len(take) - (end - start) :]
    clip[start:end] = seg[: max(0, min(len(seg), CLIP - start))]
    if rng.random() < noise_p:
        clip = mix(clip, bed(CLIP, rng, bg), float(rng.uniform(4, 25)))
    peak = float(np.max(np.abs(clip))) or 1.0
    return (clip / max(peak, 1.0)).astype(np.float32)


def make_clips(data: WakeData, rng: np.random.Generator, n_pos: int, n_neg: int) -> tuple[np.ndarray, np.ndarray]:
    """Balanced-ish training clips: (N, 32000) float32 and labels (N,)."""
    positives = data.pos + data.synth_pos
    if not positives:
        raise ValueError("no takes of the wake phrase: run `jarvis wakeword record` first")
    clips, labels = [], []
    for _ in range(n_pos):
        clips.append(place(augment_take(positives[int(rng.integers(len(positives)))], rng), rng, data.bg))
        labels.append(1)
    speech = data.neg + data.synth_neg
    for i in range(n_neg):
        r = rng.random()
        if speech and r < 0.45:
            x = augment_take(speech[int(rng.integers(len(speech)))], rng)
            clip = crop(x, rng)
            if rng.random() < 0.6:
                clip = mix(clip, bed(CLIP, rng, data.bg), float(rng.uniform(5, 25)))
        elif r < 0.6:  # cut off before the end: not a wake yet
            take = augment_take(positives[int(rng.integers(len(positives)))], rng)
            cut = take[: max(1, int(len(take) * rng.uniform(0.25, 0.6)))]
            clip = place(cut, rng, data.bg)
        elif data.bg and r < 0.9:
            clip = crop(data.bg[int(rng.integers(len(data.bg)))], rng)
        else:
            clip = bed(CLIP, rng, []) * float(rng.uniform(0.001, 0.3)) if i % 7 else np.zeros(CLIP, np.float32)
        peak = float(np.max(np.abs(clip))) or 1.0
        clips.append((clip / max(peak, 1.0)).astype(np.float32))
        labels.append(0)
    return np.stack(clips), np.asarray(labels, np.float32)


# ------------------------------------------------------------------------------------- features + model
class OwwFeatures:
    """openWakeWord's frozen melspectrogram + embedding models: 2 s clips -> (16, 96) embeddings."""

    def __init__(self, models_dir: Path) -> None:
        from openwakeword.utils import AudioFeatures

        d = models_dir / "openwakeword"
        self._f = AudioFeatures(str(d / "melspectrogram.onnx"), str(d / "embedding_model.onnx"))

    def __call__(self, clips: np.ndarray, batch: int = 64) -> np.ndarray:
        pcm = (np.clip(clips, -1, 1) * 32767).astype(np.int16)
        return np.asarray(self._f.embed_clips(pcm, batch_size=batch), np.float32)


def make_head():
    import torch

    class WakeHead(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Flatten(), torch.nn.Linear(16 * 96, 128), torch.nn.LayerNorm(128), torch.nn.ReLU(),
                torch.nn.Dropout(0.2), torch.nn.Linear(128, 128), torch.nn.LayerNorm(128), torch.nn.ReLU(),
                torch.nn.Linear(128, 1))

        def forward(self, x):
            return self.net(x)

    return WakeHead()


def train_head(x: np.ndarray, y: np.ndarray, xv: np.ndarray, yv: np.ndarray, *, epochs: int = 40, seed: int = 0,
               progress: Callable[[str], None] | None = None):
    import torch

    torch.manual_seed(seed)
    net = make_head()
    pos = max(1.0, float(y.sum()))
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(y) - pos) / pos * 0.5))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    xt, yt = torch.from_numpy(x), torch.from_numpy(y).unsqueeze(1)
    xvt, yvt = torch.from_numpy(xv), torch.from_numpy(yv).unsqueeze(1)
    gen = torch.Generator().manual_seed(seed)
    best, best_state = float("inf"), None
    for ep in range(epochs):
        net.train()
        for idx in torch.randperm(len(xt), generator=gen).split(256):
            opt.zero_grad()
            loss = loss_fn(net(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(loss_fn(net(xvt), yvt).item())
        if vl < best:
            best, best_state = vl, {k: v.clone() for k, v in net.state_dict().items()}
        if progress and (ep % 10 == 9 or ep == epochs - 1):
            progress(f"epoch {ep + 1}/{epochs}: val loss {vl:.4f}")
    if best_state is not None:
        net.load_state_dict(best_state)
    return net.eval()


def scores(net, x: np.ndarray) -> np.ndarray:
    import torch

    with torch.no_grad():
        return torch.sigmoid(net(torch.from_numpy(x))).numpy().reshape(-1)


def choose_threshold(pos: np.ndarray, neg: np.ndarray, max_fpr: float = 0.002) -> tuple[float, float, float]:
    """The lowest threshold whose false-accept rate per window stays under ``max_fpr``: (thr, recall, fpr)."""
    cands = np.unique(np.concatenate([np.linspace(0.3, 0.97, 68), pos]))
    best = (0.97, float(np.mean(pos >= 0.97)), float(np.mean(neg >= 0.97)))
    for t in sorted(c for c in cands if 0.3 <= c <= 0.97):
        fpr = float(np.mean(neg >= t))
        if fpr <= max_fpr:
            return float(round(t, 3)), float(np.mean(pos >= t)), fpr
    return best


def export_onnx(net, path: Path) -> Path:
    """Same interface as openWakeWord's own models: x [1, 16, 96] float32 -> score [1, 1]."""
    import warnings

    import torch

    class WithSigmoid(torch.nn.Module):
        def __init__(self, inner) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, x):
            return torch.sigmoid(self.inner(x))

    path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kw = {"dynamo": False} if "dynamo" in torch.onnx.export.__code__.co_varnames else {}
        torch.onnx.export(WithSigmoid(net.eval()), torch.zeros(1, 16, 96), str(path), input_names=["x"],
                          output_names=["score"], opset_version=17, **kw)
    return path


# ------------------------------------------------------------------------------------- the whole job
def train_wakeword(data: WakeData, models_dir: Path, name: str, features: Callable[[np.ndarray], np.ndarray], *,
                   epochs: int = 40, seed: int = 0, n_pos: int = 2400, n_neg: int = 7200, max_fpr: float = 0.002,
                   progress: Callable[[str], None] | None = None) -> dict:
    """Train, evaluate on held-out takes, export to models/openwakeword/<name>_v0.1.onnx. Returns metrics."""
    say = progress or (lambda m: log.info(m))
    if len(data.pos) + len(data.synth_pos) < 6:
        raise ValueError("need at least 6 takes of the phrase (record more with `jarvis wakeword record`)")
    rng = np.random.default_rng(seed)
    # Your own takes are split by take, so the reported recall is on takes the model never saw.
    order = rng.permutation(len(data.pos))
    n_hold = max(1, len(data.pos) // 5) if len(data.pos) >= 5 else 0
    hold_pos = [data.pos[i] for i in order[:n_hold]]
    train = WakeData(pos=[data.pos[i] for i in order[n_hold:]], neg=data.neg, bg=data.bg,
                     synth_pos=data.synth_pos, synth_neg=data.synth_neg)
    if not train.pos and not train.synth_pos:
        train.pos = hold_pos
    say(f"building {n_pos + n_neg} training clips")
    clips, y = make_clips(train, rng, n_pos, n_neg)
    say("computing speech embeddings")
    x = features(clips)
    vi = rng.permutation(len(x))
    nv = len(x) // 8
    val, tr = vi[:nv], vi[nv:]
    net = train_head(x[tr], y[tr], x[val], y[val], epochs=epochs, seed=seed, progress=say)
    # Held-out evaluation: clean and noisy placements of unseen takes, fresh negatives.
    eval_pos_src = hold_pos or data.pos or data.synth_pos
    ev_rng = np.random.default_rng(seed + 1)
    ep_clips = np.stack([place(t, ev_rng, data.bg, noise_p=p) for t in eval_pos_src for p in (0.0, 1.0)])
    en_clips, _ = make_clips(WakeData(pos=eval_pos_src, neg=data.neg, bg=data.bg, synth_neg=data.synth_neg), ev_rng, 0, 1500)
    sp, sn = scores(net, features(ep_clips)), scores(net, features(en_clips))
    thr, recall, fpr = choose_threshold(sp, sn, max_fpr)
    out = models_dir / "openwakeword" / f"{name}_v0.1.onnx"
    export_onnx(net, out)
    metrics = {"name": name, "threshold": thr, "recall_heldout": round(recall, 3), "fpr_per_window": round(fpr, 5),
               "false_wakes_per_hour_est": round(fpr * WINDOWS_PER_HOUR, 1), "takes": len(data.pos),
               "heldout_takes": len(hold_pos), "synthetic_takes": len(data.synth_pos), "negatives": len(data.neg),
               "background_clips": len(data.bg), "trained": time.strftime("%Y-%m-%dT%H:%M:%S"), "path": str(out)}
    out.with_suffix(".json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    say(f"held-out recall {recall:.0%} at threshold {thr} ({metrics['false_wakes_per_hour_est']} false wakes/h before "
        "the second-stage check)")
    return metrics


def synthesize_takes(cfg, phrases: list[str], voices: list[str] | None = None,
                     speeds: tuple[float, ...] = (0.9, 1.0, 1.15)) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Extra takes from the local voice (Supertonic, several voice styles): (positives, near-miss negatives)."""
    from jarvis.voice.resample import resample
    from jarvis.voice.tts.engines import SupertonicEngine

    pos: list[np.ndarray] = []
    neg: list[np.ndarray] = []
    models = cfg.paths().ensure().models
    for v in voices or ["M1", "M2", "F1", "F2"]:
        for sp in speeds:
            try:
                eng = SupertonicEngine(cfg.voice.tts.model_copy(update={"supertonic_voice": v, "speed": sp}), models)
            except Exception as e:
                log.info("voice %s unavailable: %s", v, e)
                break
            for text in phrases:
                a = eng.synth(text, "ko")
                pos.append(trim_speech(resample(a.pcm, a.rate, SR)))
            for text in NEAR_MISSES_KO + NEAR_MISSES_EN:
                a = eng.synth(text, "en" if text in NEAR_MISSES_EN else "ko")
                neg.append(resample(a.pcm, a.rate, SR).astype(np.float32))
    return pos, neg


# ------------------------------------------------------------------------------------- recording (Windows/mic)
def slug(text: str) -> str:
    import hashlib

    return hashlib.sha1(text.encode()).hexdigest()[:6]


def take_problem(x: np.ndarray) -> str | None:
    """Why a recorded take is unusable, or None."""
    if len(x) == 0 or float(np.max(np.abs(x))) < 0.02:
        return "I didn't hear anything - closer to the mic, please"
    if float(np.mean(np.abs(x) > 0.99)) > 0.01:
        return "too loud (clipping) - a little further from the mic, please"
    d = len(x) / SR
    if d < 0.25:
        return "too short"
    if d > 1.8:
        return "too long - just the wake phrase, please"
    return None


def record_mic(seconds: float, device=None) -> np.ndarray:
    import sounddevice as sd

    from jarvis.voice.resample import resample

    rate = int(sd.query_devices(device, kind="input")["default_samplerate"])
    rec = sd.rec(int(seconds * rate), samplerate=rate, channels=1, dtype="float32", device=device)
    sd.wait()
    x = rec.reshape(-1)
    return (resample(x, rate, SR) if rate != SR else x).astype(np.float32)


def beep(device=None) -> None:
    import sounddevice as sd

    t = np.arange(int(0.15 * 44100)) / 44100
    sd.play((0.2 * np.sin(2 * np.pi * 880 * t)).astype(np.float32), 44100, device=device)
    sd.wait()


def record_loopback(seconds: float) -> np.ndarray | None:
    """What the speakers play (game, music, video) via WASAPI loopback; None where unavailable."""
    try:
        import pyaudiowpatch as pyaudio
    except ImportError:
        return None
    from jarvis.voice.resample import resample

    pa = pyaudio.PyAudio()
    try:
        dev = pa.get_default_wasapi_loopback()
        rate, ch = int(dev["defaultSampleRate"]), int(dev["maxInputChannels"])
        stream = pa.open(format=pyaudio.paInt16, channels=ch, rate=rate, input=True, input_device_index=dev["index"],
                         frames_per_buffer=1024)
        chunks = [stream.read(1024, exception_on_overflow=False) for _ in range(int(seconds * rate / 1024))]
        stream.stop_stream()
        stream.close()
    except Exception as e:
        log.warning("loopback recording failed: %s", e)
        return None
    finally:
        pa.terminate()
    x = np.frombuffer(b"".join(chunks), np.int16).astype(np.float32).reshape(-1, ch).mean(axis=1) / 32768.0
    return resample(x, rate, SR).astype(np.float32)
