"""Builds the live voice stack from config. Every optional piece degrades with a recorded reason
(``notes``) instead of failing: no loopback -> TTS-only echo reference, no Silero file -> energy VAD,
no wake model -> wake word found in transcripts, no speechbrain/enrollment -> unverified (flagged)."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import numpy as np

from jarvis.common.config import JarvisConfig
from jarvis.safety.speaker_gate import SpeakerGate
from jarvis.voice.aec import build_aec
from jarvis.voice.barge_in import BargeConfig
from jarvis.voice.frontend import AudioFrontEnd
from jarvis.voice.session import Publish, SessionOptions, Translate, TurnHandler, VoiceSession
from jarvis.voice.vad import EnergyVad, SileroVad, VadSegmenter

log = logging.getLogger(__name__)

VOICEPRINT = "voiceprint.npz"


class VoiceUnavailable(RuntimeError):
    pass


def barge_config(cfg: JarvisConfig) -> BargeConfig:
    b = cfg.voice.barge_in
    return BargeConfig(duck_gain=b.duck_gain, stop_after_ms=b.stop_after_ms, require_owner_for_stop=b.require_owner_for_stop,
                       not_owner_score=b.not_owner_score, backchannel_max_ms=b.backchannel_max_ms,
                       backchannel_words=tuple(b.backchannel_words))


def build_router(cfg: JarvisConfig):
    from jarvis.voice.tts.cache import TtsCache
    from jarvis.voice.tts.engines import EdgeEngine, Qwen3Engine, SupertonicEngine
    from jarvis.voice.tts.router import TtsRouter

    t = cfg.voice.tts
    paths = cfg.paths().ensure()
    factories = {
        "supertonic": lambda: SupertonicEngine(t, paths.models),
        "qwen3": lambda: Qwen3Engine(t),
        "edge": lambda: EdgeEngine(t),
    }
    return TtsRouter(t, TtsCache(paths.tts_cache), factories)


def build_gate(cfg: JarvisConfig, notes: list[str]) -> SpeakerGate:
    sc = cfg.voice.speaker
    models = cfg.paths().ensure().models
    owner = SpeakerGate.load_enrollment(models / VOICEPRINT)
    embedder = None
    if sc.enabled:
        try:
            import speechbrain  # noqa: F401

            from jarvis.voice.speaker_id import EcapaEmbedder

            embedder = EcapaEmbedder(sc.model, savedir=models / "spkrec", device="cpu")
        except ImportError:
            notes.append("speaker verification unavailable (install the 'speaker' extra) - anyone can command JARVIS")
    if owner is None and sc.enabled:
        notes.append("owner voice not enrolled - anyone can command JARVIS until you run `jarvis enroll`")
    return SpeakerGate(embedder, owner, threshold=sc.threshold, min_audio_s=sc.min_audio_s, enabled=sc.enabled)


def build_vad(cfg: JarvisConfig, notes: list[str]):
    v = cfg.voice.vad
    path = Path(v.model_path) if v.model_path else cfg.paths().ensure().models / "silero_vad.onnx"
    if path.exists():
        try:
            return SileroVad(path)
        except Exception as e:
            notes.append(f"Silero VAD failed to load ({e}) - using the energy VAD")
            return EnergyVad()
    notes.append("Silero VAD model missing - using the energy VAD (run `jarvis voice-models`)")
    return EnergyVad()


def build_wake(cfg: JarvisConfig, notes: list[str]):
    try:
        from jarvis.voice.wake import WakeDetector

        return WakeDetector.from_config(cfg.voice.wake, cfg.paths().ensure().models)
    except Exception as e:
        notes.append(f"wake word unavailable ({e}); listening for 'Jarvis' in transcripts instead (more GPU)")
        return None


class VoiceRuntime:
    def __init__(self, cfg: JarvisConfig, handler: TurnHandler, publish: Publish, translate: Translate | None = None) -> None:
        self.cfg = cfg
        self.handler = handler
        self.publish = publish
        self.translate = translate
        self.notes: list[str] = []
        self.session: VoiceSession | None = None
        self.player = self.mic = self.loopback = self.frontend = None
        self._bg: set[asyncio.Task] = set()

    def build(self) -> VoiceSession:
        v = self.cfg.voice
        try:
            from jarvis.voice.audio_io import LoopbackStream, MicStream, Player
            from jarvis.voice.stt import WhisperStt
        except ImportError as e:
            raise VoiceUnavailable(f"voice packages missing ({e.name}) - install the 'voice' extra") from e
        models = self.cfg.paths().ensure().models
        self.aec = build_aec(v.aec)
        if not self.aec.enabled and v.aec.enabled:
            self.notes.append("echo cancellation unavailable - use headphones so JARVIS doesn't hear itself")
        self.segmenter = VadSegmenter.from_config(build_vad(self.cfg, self.notes), v.vad)
        self.wake = build_wake(self.cfg, self.notes)
        try:
            self.stt = WhisperStt.from_config(v.stt, models)
        except Exception as e:
            raise VoiceUnavailable(f"speech recognition failed to load: {e}") from e
        self.gate = build_gate(self.cfg, self.notes)
        self.router = build_router(self.cfg)
        self.player = Player(device=v.output_device)
        self.session = VoiceSession(
            stt=self.stt, router=self.router, player=self.player, gate=self.gate, handler=self.handler,
            options=SessionOptions.from_config(v, self.cfg.safety, self.cfg.language), barge=barge_config(self.cfg),
            publish=self.publish, mic_control=self._mic_control, translate=self.translate, segmenter=self.segmenter,
            transcript_wake=self.wake is None,
        )
        self.frontend = AudioFrontEnd(aec=self.aec, segmenter=self.segmenter, wake=self.wake, emit=self.session.emit_threadsafe)
        self.mic = MicStream(self.frontend.push_mic, device=v.input_device, rate=16000)
        if self.aec.enabled and v.aec.reference == "loopback":
            self.loopback = LoopbackStream(self.aec.push_reference)
        return self.session

    def _tts_reference(self) -> None:
        """Without loopback, the echo canceller's reference is JARVIS's own output (Roblox audio is not removed)."""
        from jarvis.voice.audio_io import Framer

        tap: dict[int, Framer] = {}

        def on_block(block: np.ndarray, rate: int) -> None:
            f = tap.get(rate) or tap.setdefault(rate, Framer(rate, 16000, 160, self.aec.push_reference))
            f.push(block)

        self.player.on_block = on_block

    def _mic_control(self, on: bool) -> None:
        if on:
            self.frontend.flush()
            self.mic.start()
        else:
            self.mic.stop()

    async def start(self) -> None:
        assert self.session is not None
        self.player.start()
        await self.session.start()
        self.frontend.start()
        if self.loopback is not None:
            try:
                self.loopback.start()
            except Exception as e:
                self.notes.append(f"speaker loopback unavailable ({e}) - echo reference is JARVIS's voice only")
                self.loopback = None
        if self.aec.enabled and self.loopback is None:
            self._tts_reference()
        if self.cfg.voice.mute.start_muted:
            self.session.mute(soft=self.cfg.voice.mute.soft_mute)
            if self.cfg.voice.mute.soft_mute:
                self.mic.start()
        else:
            self.mic.start()
        t = asyncio.create_task(asyncio.to_thread(self.router.prerender))  # instant callouts later
        self._bg.add(t)
        t.add_done_callback(self._bg.discard)
        for n in self.notes:
            log.warning("voice: %s", n)
        self.publish("voice.notes", {"notes": list(self.notes)})

    async def stop(self) -> None:
        for part in (self.mic, self.loopback):
            if part is not None:
                try:
                    part.stop()
                except Exception:
                    log.debug("stopping %s failed", type(part).__name__, exc_info=True)
        if self.frontend is not None:
            self.frontend.stop()
        if self.session is not None:
            await self.session.close()
        if self.player is not None:
            self.player.close()
