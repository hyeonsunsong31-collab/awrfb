"""Audio plumbing without hardware: resampler, VAD segmenter, player engine, echo canceller wiring,
wake detector, STT filtering and the short-clip owner check."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from jarvis.safety.speaker_gate import SpeakerGate
from jarvis.voice.aec import FRAME, EchoCanceller
from jarvis.voice.audio_io import PlayerCore
from jarvis.voice.resample import Resampler, resample
from jarvis.voice.stt import WhisperStt, reject_reason
from jarvis.voice.vad import CHUNK, EnergyVad, VadSegmenter
from jarvis.voice.wake import WakeDetector, heard_wake_word


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


# ------------------------------------------------------------------------------------- resampler
def _tone(rate: int, f: float, secs: float = 0.5) -> np.ndarray:
    t = np.arange(int(rate * secs)) / rate
    return (0.5 * np.sin(2 * np.pi * f * t)).astype(np.float32)


def _peak_hz(x: np.ndarray, rate: int) -> float:
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return float(np.fft.rfftfreq(len(x), 1 / rate)[spec.argmax()])


@pytest.mark.parametrize(("a", "b"), [(48000, 16000), (44100, 16000), (16000, 48000), (24000, 48000)])
def test_resampler_keeps_pitch_and_level_and_streams_exactly(a, b):
    x = _tone(a, 1000)
    y = resample(x, a, b)
    assert len(y) == round(len(x) * b / a)
    mid = y[len(y) // 4 : 3 * len(y) // 4]
    assert abs(_peak_hz(mid, b) - 1000) < 5
    assert abs(float(np.abs(mid).max()) - 0.5) < 0.01
    r, one = Resampler(a, b), Resampler(a, b).process(x)
    rng = np.random.default_rng(0)
    parts, i = [], 0
    while i < len(x):
        n = int(rng.integers(1, 700))
        parts.append(r.process(x[i : i + n]))
        i += n
    streamed = np.concatenate(parts)
    assert np.allclose(streamed, one[: len(streamed)], atol=1e-5)


def test_resampler_rejects_aliases():
    y = resample(_tone(48000, 10000), 48000, 16000)  # 10 kHz cannot exist at 16 kHz
    assert float(np.abs(y[300:-300]).max()) < 1e-3


# ------------------------------------------------------------------------------------- VAD
class ScriptVad:
    def __init__(self, probs):
        self.probs = list(probs)
        self.i = 0

    def reset(self):
        pass

    def prob(self, chunk):
        p = self.probs[self.i] if self.i < len(self.probs) else 0.0
        self.i += 1
        return p


def _feed(seg: VadSegmenter, n: int):
    events = []
    for k in range(n):
        events += seg.feed(np.full(CHUNK, k / 1000.0, np.float32))  # chunk k carries value k/1000
    return events


def test_segmenter_onset_then_abort_for_a_click():
    seg = VadSegmenter(ScriptVad([0, 0, 1, 1, 0, 0, 0, 0]))
    kinds = [e.kind for e in _feed(seg, 8)]
    assert kinds == ["onset", "abort"]


def test_segmenter_short_utterance_ends_fast_with_preroll():
    probs = [0] * 3 + [1] * 20 + [0] * 30
    seg = VadSegmenter(ScriptVad(probs), short_silence_ms=300, short_utterance_ms=900)
    ev = _feed(seg, len(probs))
    kinds = [e.kind for e in ev]
    assert kinds[0] == "onset" and "start" in kinds and kinds.count("end") == 1
    start = next(e for e in ev if e.kind == "start")
    assert start.audio is not None and len(start.audio) > 0
    end = next(e for e in ev if e.kind == "end")
    assert end.voiced_ms == 20 * 32
    assert end.t_ms == (3 + 20 + 10) * 32  # 300 ms of silence (10 chunks) is enough for a short one
    assert end.audio[0] == pytest.approx(0.0)  # pre-roll reaches back before the onset
    assert len(end.audio) == 33 * CHUNK - int(0.170 * 16000)  # trailing silence trimmed to 150 ms


def test_segmenter_long_utterance_waits_for_full_silence():
    probs = [1] * 40 + [0] * 30
    seg = VadSegmenter(ScriptVad(probs), short_silence_ms=300, short_utterance_ms=900, min_silence_ms=650)
    end = next(e for e in _feed(seg, len(probs)) if e.kind == "end")
    assert end.t_ms == (40 + 21) * 32


def test_segmenter_forces_end_at_max_length_and_respects_echo_boost():
    seg = VadSegmenter(ScriptVad([1] * 100), max_utterance_s=1.0)
    assert any(e.kind == "end" for e in _feed(seg, 100))
    seg = VadSegmenter(ScriptVad([0.6] * 20))
    seg.threshold_boost = 0.15  # while JARVIS speaks, 0.6 is treated as echo
    assert _feed(seg, 20) == []


def test_energy_vad_separates_speech_from_noise():
    v = EnergyVad()
    rng = np.random.default_rng(1)
    for _ in range(30):
        v.prob((rng.standard_normal(CHUNK) * 0.002).astype(np.float32))
    assert v.prob((rng.standard_normal(CHUNK) * 0.002).astype(np.float32)) < 0.3
    assert v.prob(_tone(16000, 300, CHUNK / 16000) * 0.4) > 0.9


# ------------------------------------------------------------------------------------- player engine
def test_player_plays_segments_in_order_with_events():
    ev = []
    p = PlayerCore(16000, on_event=lambda k, s: ev.append((k, s)))
    p.enqueue(1, np.full(800, 0.5, np.float32), "a")
    p.enqueue(2, np.full(800, -0.5, np.float32), "b")
    out = np.concatenate([p.render(160) for _ in range(12)])
    assert ev == [("start", 1), ("end", 1), ("start", 2), ("end", 2)]
    assert np.allclose(out[:800], 0.5) and np.allclose(out[800:1600], -0.5) and not out[1600:].any()
    assert not p.busy()


def test_player_ducks_with_a_ramp():
    p = PlayerCore(16000)
    p.enqueue(1, np.ones(16000, np.float32))
    p.render(160)
    p.set_gain(0.25)
    out = np.concatenate([p.render(160) for _ in range(10)])
    assert out[0] > 0.9 and out[-1] == pytest.approx(0.25, abs=1e-6)


def test_player_pause_fades_holds_and_resumes_without_skipping():
    p = PlayerCore(16000, ramp_ms=25.0)  # 400-sample ramp
    seg = np.ones(4800, np.float32)
    p.enqueue(1, seg)
    a = p.render(1600)
    p.pause()
    b = p.render(1600)
    c = p.render(1600)
    p.resume()
    d = p.render(4000)
    assert np.allclose(a, 1.0)
    assert b[0] > 0.9 and b[399] <= 1e-6 and not b[400:].any()  # faded out, then silent
    assert not c.any()  # position held while paused
    played = np.count_nonzero(np.concatenate([a, b, c, d]))
    assert played == 4800 - 1  # every sample played once (the very last ramp step hits exactly zero)


def test_player_stop_reports_what_was_heard_and_fades():
    p = PlayerCore(16000)
    p.enqueue(1, np.ones(800, np.float32))
    p.enqueue(2, np.ones(1600, np.float32))
    p.render(1200)  # segment 1 done, segment 2 a quarter in
    rep = p.stop()
    assert rep.finished == [1] and rep.current == 2 and rep.current_fraction == pytest.approx(0.25)
    tail = p.render(400)
    assert tail[0] > 0.5 and abs(tail[190]) < 0.05 and not tail[200:].any()
    assert not p.busy()


def test_player_pause_with_nothing_playing_holds_new_segments():
    p = PlayerCore(16000)
    p.pause()
    p.enqueue(1, np.ones(400, np.float32))
    assert not p.render(160).any() and p.pending() == 1
    p.resume()
    assert p.render(800).any()


# ------------------------------------------------------------------------------------- echo canceller
class FakeFrame:
    def __init__(self, data, sample_rate, num_channels, samples_per_channel):
        self._data = data

    @property
    def data(self):
        return memoryview(self._data).cast("B").cast("h")


class FakeApm:
    def __init__(self):
        self.calls = []

    def process_reverse_stream(self, f):
        self.calls.append("rev")

    def process_stream(self, f):
        self.calls.append("near")
        arr = np.frombuffer(f._data, dtype=np.int16)
        arr[:] = arr // 2  # pretend half the signal was echo

    def set_stream_delay_ms(self, ms):
        self.calls.append(f"delay{ms}")


def test_aec_feeds_far_end_before_near_end():
    clock = Clock()
    apm = FakeApm()
    aec = EchoCanceller(apm=apm, frame_cls=FakeFrame, delay_ms=60, clock=clock)
    aec.push_reference(np.zeros(FRAME, np.float32))
    aec.push_reference(np.zeros(FRAME, np.float32))
    out = aec.process(np.full(FRAME, 0.5, np.float32))
    assert apm.calls == ["rev", "rev", "delay60", "near"]
    assert np.allclose(out, 0.25, atol=1e-3)
    apm.calls.clear()
    aec.process(np.zeros(FRAME, np.float32))  # reference just paused: wait, don't invent silence
    assert apm.calls == ["delay60", "near"]
    clock.t += 0.5
    apm.calls.clear()
    aec.process(np.zeros(FRAME, np.float32))  # speakers silent for a while: feed zeros
    assert apm.calls == ["rev", "delay60", "near"]


def test_aec_reference_queue_is_bounded():
    aec = EchoCanceller(apm=FakeApm(), frame_cls=FakeFrame, max_ref_frames=50)
    for _ in range(200):
        aec.push_reference(np.zeros(FRAME, np.float32))
    aec.process(np.zeros(FRAME, np.float32))
    assert aec.ref_frames == 50


# ------------------------------------------------------------------------------------- wake word
class FakeOww:
    def __init__(self, scores):
        self.scores = list(scores)
        self.models = {"hey_jarvis_v0.1": None}
        self.resets = 0

    def predict(self, frame):
        assert frame.dtype == np.int16 and len(frame) == 1280
        return {"hey_jarvis_v0.1": self.scores.pop(0) if self.scores else 0.0}

    def reset(self):
        self.resets += 1


def test_wake_needs_patience_and_debounces():
    model = FakeOww([0.9, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9])
    w = WakeDetector(model, threshold=0.5, patience_frames=2, debounce_s=1.5, clock=Clock())
    fired = w.feed(np.zeros(1280 * 4, np.float32))
    assert len(fired) == 1 and model.resets == 1
    assert w.feed(np.zeros(1280 * 3, np.float32)) == []  # within the debounce window


def test_second_stage_needs_the_name_in_the_transcript():
    kws = ["jarvis", "jervis", "자비스"]
    assert heard_wake_word("Jarvis, open notepad", kws)
    assert heard_wake_word("자비스 불 꺼", kws)
    assert heard_wake_word("hey jervis what time is it", kws)
    assert not heard_wake_word("open the door please", kws)


# ------------------------------------------------------------------------------------- STT
@pytest.mark.parametrize(
    ("text", "nsp", "lp", "why"),
    [
        ("", 0.0, 0.0, "empty"),
        ("시청해 주셔서 감사합니다.", 0.1, -0.3, "known hallucination"),
        ("Thank you.", 0.5, -0.5, "probably noise"),
        ("Thank you.", 0.1, -0.2, None),
        ("open notepad", 0.7, -1.2, "no speech"),
        ("open notepad", 0.1, -0.3, None),
        ("ㅋㅋㅋㅋㅋㅋㅋ", 0.1, -0.3, "repetition"),
    ],
)
def test_stt_rejects_hallucinations(text, nsp, lp, why):
    assert reject_reason(text, nsp, lp) == why


class FakeWhisper:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio, language=None, **kw):
        self.calls.append(language)
        if language is None:
            info = SimpleNamespace(language="ja", language_probability=0.5,
                                   all_language_probs=[("ja", 0.5), ("ko", 0.3), ("en", 0.2)])
            return iter([SimpleNamespace(text="こんにちは", no_speech_prob=0.1, avg_logprob=-0.3)]), info
        info = SimpleNamespace(language=language, language_probability=0.9, all_language_probs=None)
        return iter([SimpleNamespace(text=" 안녕하세요 ", no_speech_prob=0.05, avg_logprob=-0.2)]), info


def test_stt_redecodes_outside_korean_english():
    fw = FakeWhisper()
    stt = WhisperStt(fw, languages=["ko", "en"], hotwords=["JARVIS", "자비스"])
    r = stt.transcribe(np.zeros(16000, np.float32))
    assert fw.calls == [None, "ko"] and r.text == "안녕하세요" and r.lang == "ko" and r.ok


# ------------------------------------------------------------------------------------- owner check
class SignEmbedder:
    """Owner audio is positive, other voices negative: the embedding is the sign mix."""

    def embed(self, audio):
        a = np.asarray(audio)
        return np.array([np.mean(a > 0), np.mean(a < 0), 0.01])


def _clip(sign: float, secs: float) -> np.ndarray:
    return np.full(int(secs * 16000), sign * 0.5, np.float32)


def test_short_command_needs_its_own_voice_and_context():
    gate = SpeakerGate(SignEmbedder(), np.array([[1.0, 0.0, 0.01]]), threshold=0.45, min_audio_s=1.5)
    owner_ctx = _clip(+1, 1.0)
    ok = gate.verify_with_context(_clip(+1, 0.6), owner_ctx)
    assert ok.is_owner and "context" in ok.reason
    tv = gate.verify_with_context(_clip(-1, 0.6), owner_ctx)  # TV says "fight" right after the owner spoke
    assert not tv.is_owner
    assert not gate.verify_with_context(_clip(+1, 0.2), owner_ctx).is_owner  # too short to judge
    assert gate.verify_with_context(_clip(+1, 2.0)).is_owner  # long clips use plain verify
    assert not gate.verify_with_context(_clip(-1, 2.0)).is_owner
    assert gate.quick_score(_clip(+1, 0.5)) == pytest.approx(1.0, abs=1e-3)


def test_unenrolled_gate_allows_but_says_why():
    gate = SpeakerGate(SignEmbedder(), None)
    r = gate.verify_with_context(_clip(-1, 0.6))
    assert r.is_owner and "not enrolled" in r.reason
    assert gate.quick_score(_clip(-1, 0.6)) is None
