"""Audio devices: microphone capture, WASAPI loopback (echo-cancellation reference) and the TTS player.

* ``MicStream``: sounddevice input, 16 kHz mono 10 ms frames (device rate + resampler if 16 kHz is
  refused). Closing it releases the device, which turns the Windows microphone indicator off (hard mute).
* ``LoopbackStream``: everything the speakers play (TTS + Roblox + media) via PyAudioWPatch, so the
  echo canceller removes all of it from the mic signal.
* ``PlayerCore``: the device-independent playback engine (segment queue, gain ramps for ducking,
  click-free pause/resume, stop with a short fade). ``Player`` drives it from a sounddevice output
  callback. Segment start/end events let the voice session know which words were actually heard.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from jarvis.voice.resample import Resampler, resample, rms, to_mono

log = logging.getLogger(__name__)

FrameSink = Callable[[np.ndarray], None]
PlayerEventSink = Callable[[str, int], None]  # ("start" | "end", segment id)


# ------------------------------------------------------------------------------------- framing
class Framer:
    """Accumulates resampled audio and emits fixed-size frames."""

    def __init__(self, in_rate: int, out_rate: int, frame: int, sink: FrameSink) -> None:
        self._rs = Resampler(in_rate, out_rate)
        self._frame = frame
        self._sink = sink
        self._acc = np.zeros(0, np.float32)

    def push(self, x: np.ndarray) -> None:
        y = self._rs.process(x)
        if len(y) == 0:
            return
        self._acc = np.concatenate([self._acc, y]) if len(self._acc) else y
        n = len(self._acc) // self._frame
        for i in range(n):
            self._sink(self._acc[i * self._frame : (i + 1) * self._frame])
        self._acc = self._acc[n * self._frame :]


# ------------------------------------------------------------------------------------- mic
class MicStream:
    def __init__(self, on_frame: FrameSink, device: str | int | None = None, rate: int = 16000, frame_ms: int = 10) -> None:
        self.on_frame = on_frame
        self.device = device
        self.rate = rate
        self.frame = rate * frame_ms // 1000
        self._stream = None
        self._framer: Framer | None = None
        self.device_rate: int | None = None
        self.overflows = 0

    @property
    def running(self) -> bool:
        return self._stream is not None

    def start(self) -> None:
        if self._stream is not None:
            return
        import sounddevice as sd

        try:
            stream = sd.InputStream(samplerate=self.rate, channels=1, dtype="float32", blocksize=self.frame,
                                    device=self.device, latency="low", callback=self._cb)
            self.device_rate = self.rate
        except Exception as e:  # e.g. WASAPI shared mode refusing 16 kHz
            info = sd.query_devices(self.device, "input")
            self.device_rate = int(info["default_samplerate"])
            log.info("mic refused %d Hz (%s); capturing at %d Hz and resampling", self.rate, e, self.device_rate)
            stream = sd.InputStream(samplerate=self.device_rate, channels=1, dtype="float32",
                                    blocksize=self.device_rate // 100, device=self.device, latency="low", callback=self._cb)
        self._framer = Framer(self.device_rate, self.rate, self.frame, self.on_frame)
        stream.start()
        self._stream = stream

    def _cb(self, indata, frames, time_info, status) -> None:  # PortAudio thread: keep it tiny
        if status and status.input_overflow:
            self.overflows += 1
        if self._framer is not None:
            self._framer.push(to_mono(indata))

    def stop(self) -> None:
        s, self._stream = self._stream, None
        if s is not None:
            try:
                s.stop()
            finally:
                s.close()


# ------------------------------------------------------------------------------------- loopback
class LoopbackStream:
    """WASAPI loopback of the default output device (Windows only, PyAudioWPatch)."""

    def __init__(self, on_frame: FrameSink, rate: int = 16000, frame_ms: int = 10) -> None:
        self.on_frame = on_frame
        self.rate = rate
        self.frame = rate * frame_ms // 1000
        self._pa = None
        self._stream = None
        self._channels = 2
        self._continue = 0
        self._framer: Framer | None = None
        self.device_name = ""

    @property
    def running(self) -> bool:
        return self._stream is not None

    def start(self) -> None:
        if self._stream is not None:
            return
        import pyaudiowpatch as pyaudio

        self._pa = pyaudio.PyAudio()
        info = self._pa.get_default_wasapi_loopback()
        self._channels = max(1, int(info["maxInputChannels"]))
        sr = int(info["defaultSampleRate"])
        self.device_name = str(info.get("name", ""))
        self._continue = pyaudio.paContinue
        self._framer = Framer(sr, self.rate, self.frame, self.on_frame)
        self._stream = self._pa.open(format=pyaudio.paFloat32, channels=self._channels, rate=sr, input=True,
                                     input_device_index=int(info["index"]), frames_per_buffer=sr // 100,
                                     stream_callback=self._cb)

    def _cb(self, in_data, frame_count, time_info, status):
        if self._framer is not None:
            x = np.frombuffer(in_data, dtype=np.float32).reshape(-1, self._channels)
            self._framer.push(x.mean(axis=1))
        return (None, self._continue)

    def stop(self) -> None:
        s, self._stream = self._stream, None
        if s is not None:
            try:
                s.stop_stream()
                s.close()
            except Exception:
                log.debug("loopback close failed", exc_info=True)
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None


# ------------------------------------------------------------------------------------- player
@dataclass
class _Segment:
    seg_id: int
    pcm: np.ndarray
    text: str
    pos: int = 0


@dataclass
class PlaybackReport:
    finished: list[int] = field(default_factory=list)
    current: int | None = None
    current_fraction: float = 0.0


class PlayerCore:
    """Device-independent playback: call ``render(n)`` from the audio callback."""

    def __init__(self, rate: int, ramp_ms: float = 25.0, on_event: PlayerEventSink | None = None) -> None:
        self.rate = rate
        self.on_event = on_event
        self._lock = threading.Lock()
        self._queue: deque[_Segment] = deque()
        self._cur: _Segment | None = None
        self._gain = 1.0
        self._duck = 1.0
        self._paused = False
        self._tail = np.zeros(0, np.float32)
        self._step = 1.0 / max(1.0, ramp_ms * rate / 1000.0)  # max gain change per sample
        self._finished: deque[int] = deque(maxlen=256)
        self.level = 0.0

    # ---------------------------------------------------------------- control (any thread)
    def enqueue(self, seg_id: int, pcm: np.ndarray, text: str = "") -> None:
        with self._lock:
            self._queue.append(_Segment(seg_id, np.asarray(pcm, np.float32).reshape(-1), text))

    def set_gain(self, gain: float) -> None:
        with self._lock:
            self._duck = float(np.clip(gain, 0.0, 1.0))

    def pause(self) -> None:
        with self._lock:
            self._paused = True

    def resume(self) -> None:
        with self._lock:
            self._paused = False

    @property
    def paused(self) -> bool:
        return self._paused

    def busy(self) -> bool:
        with self._lock:
            return self._cur is not None or bool(self._queue) or len(self._tail) > 0

    def pending(self) -> int:
        """Segments queued but not started yet."""
        with self._lock:
            return len(self._queue)

    def stop(self, fade_ms: float = 12.0) -> PlaybackReport:
        with self._lock:
            rep = PlaybackReport(list(self._finished))
            if self._cur is not None:
                c = self._cur
                rep.current = c.seg_id
                rep.current_fraction = c.pos / max(1, len(c.pcm))
                n = min(len(c.pcm) - c.pos, int(self.rate * fade_ms / 1000))
                if n > 0 and not (self._paused and self._gain <= 0.0):
                    self._tail = c.pcm[c.pos : c.pos + n] * np.linspace(self._gain, 0.0, n, dtype=np.float32)
            self._cur = None
            self._queue.clear()
            self._finished.clear()
            self._paused = False
            self._gain = self._duck
            return rep

    def clear_report(self) -> None:
        with self._lock:
            self._finished.clear()

    # ---------------------------------------------------------------- audio thread
    def render(self, frames: int) -> np.ndarray:
        out = np.zeros(frames, np.float32)
        events: list[tuple[str, int]] = []
        with self._lock:
            filled = 0
            if len(self._tail):
                n = min(frames, len(self._tail))
                out[:n] = self._tail[:n]
                self._tail = self._tail[n:]
                filled = n
            avail = frames - filled
            if self._paused:
                if self._cur is None:  # nothing playing: hold queued segments silently
                    self._gain = 0.0
                    avail = 0
                else:  # fade out while still advancing, then hold position (no audio skipped)
                    avail = min(avail, int(np.ceil(self._gain / self._step))) if self._gain > 0 else 0
            if avail > 0:
                raw = self._pull(avail, events)
                target = 0.0 if self._paused else self._duck
                ramp = self._ramp(avail, target)
                out[filled : filled + avail] += raw * ramp
        self.level = 0.8 * self.level + 0.2 * rms(out)
        if self.on_event is not None:
            for kind, sid in events:
                try:
                    self.on_event(kind, sid)
                except Exception:
                    log.exception("player event callback failed")
        return out

    def _ramp(self, n: int, target: float) -> np.ndarray:
        g0 = self._gain
        if g0 == target:
            return np.full(n, g0, np.float32)
        steps = np.arange(1, n + 1, dtype=np.float32) * self._step
        r = np.minimum(g0 + steps, target) if target > g0 else np.maximum(g0 - steps, target)
        self._gain = float(r[-1])
        return r.astype(np.float32)

    def _pull(self, n: int, events: list[tuple[str, int]]) -> np.ndarray:
        raw = np.zeros(n, np.float32)
        i = 0
        while i < n:
            if self._cur is None:
                if not self._queue:
                    break
                self._cur = self._queue.popleft()
                events.append(("start", self._cur.seg_id))
            c = self._cur
            take = min(n - i, len(c.pcm) - c.pos)
            raw[i : i + take] = c.pcm[c.pos : c.pos + take]
            c.pos += take
            i += take
            if c.pos >= len(c.pcm):
                events.append(("end", c.seg_id))
                self._finished.append(c.seg_id)
                self._cur = None
        return raw


class Player:
    """sounddevice output stream around ``PlayerCore`` (10 ms blocks so ducking reacts fast)."""

    def __init__(self, device: str | int | None = None, rate: int | None = None, block_ms: int = 10,
                 on_block: Callable[[np.ndarray, int], None] | None = None) -> None:
        self.device = device
        self._rate = rate
        self.block_ms = block_ms
        self.on_block = on_block  # echo-cancellation reference tap when no loopback is available
        self.core: PlayerCore | None = None
        self._stream = None
        self._channels = 1
        self._on_event: PlayerEventSink | None = None

    @property
    def on_event(self) -> PlayerEventSink | None:
        return self._on_event

    @on_event.setter
    def on_event(self, fn: PlayerEventSink | None) -> None:
        self._on_event = fn
        if self.core is not None:
            self.core.on_event = fn

    @property
    def rate(self) -> int:
        assert self.core is not None, "Player.start() first"
        return self.core.rate

    def start(self) -> None:
        if self._stream is not None:
            return
        import sounddevice as sd

        rate = self._rate or int(sd.query_devices(self.device, "output")["default_samplerate"])
        self.core = PlayerCore(rate, on_event=self._on_event)
        block = rate * self.block_ms // 1000
        for ch in (1, 2):
            try:
                self._stream = sd.OutputStream(samplerate=rate, channels=ch, dtype="float32", blocksize=block,
                                               device=self.device, latency="low", callback=self._cb)
                self._channels = ch
                break
            except Exception:
                if ch == 2:
                    raise
        assert self._stream is not None
        self._stream.start()

    def _cb(self, outdata, frames, time_info, status) -> None:
        assert self.core is not None
        block = self.core.render(frames)
        outdata[:] = block[:, None]
        if self.on_block is not None:
            self.on_block(block, self.core.rate)

    def close(self) -> None:
        s, self._stream = self._stream, None
        if s is not None:
            s.stop()
            s.close()

    # ---------------------------------------------------------------- PlayerLike
    def enqueue(self, seg_id: int, pcm: np.ndarray, rate: int, text: str = "") -> None:
        assert self.core is not None
        self.core.enqueue(seg_id, resample(pcm, rate, self.core.rate), text)

    def set_gain(self, gain: float) -> None:
        if self.core is not None:
            self.core.set_gain(gain)

    def pause(self) -> None:
        if self.core is not None:
            self.core.pause()

    def resume(self) -> None:
        if self.core is not None:
            self.core.resume()

    def stop(self) -> PlaybackReport:
        return self.core.stop() if self.core is not None else PlaybackReport()

    def busy(self) -> bool:
        return self.core.busy() if self.core is not None else False

    def pending(self) -> int:
        return self.core.pending() if self.core is not None else 0

    @property
    def level(self) -> float:
        return self.core.level if self.core is not None else 0.0


class NullPlayer:
    """Text-only mode (no audio output): accepts everything, plays nothing."""

    on_event: PlayerEventSink | None = None
    level = 0.0

    def enqueue(self, seg_id: int, pcm: np.ndarray, rate: int, text: str = "") -> None:
        if self.on_event is not None:  # report as played so turns finish
            self.on_event("start", seg_id)
            self.on_event("end", seg_id)

    def set_gain(self, gain: float) -> None:
        pass

    def pause(self) -> None:
        pass

    def resume(self) -> None:
        pass

    def stop(self) -> PlaybackReport:
        return PlaybackReport()

    def busy(self) -> bool:
        return False

    def pending(self) -> int:
        return 0
