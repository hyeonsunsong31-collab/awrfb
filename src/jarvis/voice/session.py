"""Voice session: the asyncio side of the voice pipeline (STS / TTS / STT orchestration).

Events from the audio front end (VAD, wake word, levels) and the player (segment start/end) arrive on
one queue. Per utterance:

  addressed?  wake word heard in it | follow-up window after a reply | JARVIS is replying (barge-in)
              | interpreter mode.  Unaddressed short utterances are still transcribed during a fight so
              "멈춰/stop" from ANY voice halts everything.
  STT -> local intent -> barge-in decision (resume / stop / re-input) -> owner gate -> turn handler.

Replies stream through ``ReplySink``: text deltas -> sentence chunker -> speech text normalization ->
TTS (worker thread, at most ~2 sentences ahead) -> player. When the owner interrupts, the history is
told exactly which words were played (``TurnHandler.interrupted``).
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import logging
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from jarvis.safety.speaker_gate import SpeakerGate, SpeakerResult, Trust
from jarvis.voice.barge_in import (
    BargeConfig,
    BargeIn,
    CancelLlm,
    Duck,
    PauseTts,
    ResumeTts,
    StopTts,
    Unduck,
)
from jarvis.voice.chunker import SentenceChunker
from jarvis.voice.frontend import LevelEvent, WakeEvent
from jarvis.voice.intents import Intent, is_kill_phrase, match_intent, strip_wake_word
from jarvis.voice.stt import SttResult
from jarvis.voice.textnorm import clean_for_speech, detect_lang
from jarvis.voice.tts.callouts import CRITICAL, callout_for_event
from jarvis.voice.vad import VadEvent
from jarvis.voice.wake import heard_wake_word

log = logging.getLogger(__name__)

Publish = Callable[[str, dict], None]
Translate = Callable[[str, str], Awaitable[tuple[str, str]]]


@dataclass
class Utterance:
    text: str  # wake word stripped
    raw: str
    lang: str
    addressed_by: str  # wake | followup | bargein | interpreter | typed
    owner: SpeakerResult
    intent: Intent | None = None
    voiced_ms: float = 0.0
    stt: SttResult | None = None


class TurnHandler(Protocol):
    async def handle(self, utt: Utterance, reply: ReplySink) -> None: ...
    async def interrupted(self, heard_text: str) -> None: ...
    def fight_active(self) -> bool: ...
    async def kill(self, reason: str) -> None: ...


@dataclass
class SessionOptions:
    followup_window_s: float = 8.0
    wake_keywords: list[str] = field(default_factory=lambda: ["jarvis", "자비스"])
    second_stage: bool = True
    kill_words: list[str] = field(default_factory=lambda: ["stop", "멈춰", "그만"])
    kill_check_max_ms: float = 1500.0
    soft_mute: bool = False
    require_owner_for_agent: bool = True
    speaking_threshold_boost: float = 0.15
    default_lang: str = "en"
    max_ahead: int = 2  # synthesized-but-not-started segments
    stale_utterance_s: float = 12.0

    @classmethod
    def from_config(cls, voice_cfg, safety_cfg=None, language: str = "auto") -> SessionOptions:
        return cls(
            followup_window_s=voice_cfg.followup_window_s,
            wake_keywords=list(voice_cfg.wake.second_stage_keywords),
            second_stage=voice_cfg.wake.second_stage,
            kill_words=list(safety_cfg.kill_words) if safety_cfg is not None else ["stop", "멈춰", "그만"],
            soft_mute=voice_cfg.mute.soft_mute,
            default_lang="ko" if language == "ko" else "en",
        )


def _cut_heard(text: str, fraction: float) -> str:
    n = int(len(text) * max(0.0, min(1.0, fraction)))
    if n <= 0:
        return ""
    if n >= len(text):
        return text
    head = text[:n]
    sp = head.rfind(" ")
    return (head[:sp] if sp > 0 else head).rstrip() + "…"


class ReplySink:
    """One turn's output channel. ``write`` streams text; ``say`` adds whole phrases."""

    def __init__(self, session: VoiceSession, turn_id: int, lang: str, *, speak: bool = True) -> None:
        self.session = session
        self.turn_id = turn_id
        self.lang = lang
        self.speak = speak
        self.cancel = asyncio.Event()
        self.text_parts: list[str] = []
        self.seg_ids: list[int] = []
        self.ended: set[int] = set()
        self.heard: str | None = None  # set when the owner interrupted: what was actually played
        self.t_end_speech: float | None = None
        self.t_stt: float | None = None
        self.t_first_text: float | None = None
        self.t_first_audio: float | None = None
        self.said_on_screen = False
        self._chunker = SentenceChunker()
        self._q: asyncio.Queue[tuple[str, str | None] | None] = asyncio.Queue()
        self._worker = asyncio.create_task(self._speak_loop()) if speak else None

    @property
    def text(self) -> str:
        return "".join(self.text_parts)

    async def write(self, delta: str) -> None:
        if self.cancel.is_set() or not delta:
            return
        if self.t_first_text is None:
            self.t_first_text = self.session.clock()
        self.text_parts.append(delta)
        self.session.publish("voice.reply", {"turn": self.turn_id, "delta": delta})
        if self.speak:
            for s in self._chunker.feed(delta):
                await self._q.put((s, None))

    async def say(self, text: str, *, key: str | None = None) -> None:
        """A complete phrase; ``key`` names a pre-rendered phrase (instant, works during fights)."""
        if self.cancel.is_set() or not text:
            return
        if self.t_first_text is None:
            self.t_first_text = self.session.clock()
        sep = "" if not self.text_parts or self.text_parts[-1].endswith((" ", "\n")) else " "
        self.text_parts.append(sep + text)
        self.session.publish("voice.reply", {"turn": self.turn_id, "delta": sep + text})
        if self.speak:
            if key:
                await self._q.put((text, key))
            else:
                ch = SentenceChunker()
                for s in ch.feed(text) + ch.flush():
                    await self._q.put((s, None))

    async def finish(self) -> None:
        if self._worker is None:
            return
        for s in self._chunker.flush():
            await self._q.put((s, None))
        await self._q.put(None)
        with contextlib.suppress(asyncio.CancelledError):
            await self._worker
        await self.session.wait_playback(self)

    def stop_worker(self) -> None:
        if self._worker is not None and not self._worker.done():
            self._worker.cancel()

    async def _speak_loop(self) -> None:
        while True:
            item = await self._q.get()
            if item is None or self.cancel.is_set():
                return
            text, key = item
            try:
                await self.session.speak_piece(self, text, key)
            except Exception:
                log.exception("speaking a reply piece failed")


@dataclass
class _UttItem:
    ev: VadEvent
    wake_hit: bool
    replying: bool
    interrupting: bool
    followup: bool
    fight: bool
    onset_wall: float
    t_end_speech: float


class VoiceSession:
    def __init__(
        self,
        *,
        stt,
        router,
        player,
        gate: SpeakerGate,
        handler: TurnHandler,
        options: SessionOptions | None = None,
        barge: BargeConfig | None = None,
        publish: Publish | None = None,
        mic_control: Callable[[bool], None] | None = None,
        translate: Translate | None = None,
        segmenter=None,
        transcript_wake: bool = False,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.stt = stt
        self.router = router
        self.player = player
        self.gate = gate
        self.handler = handler
        self.opt = options or SessionOptions()
        self.barge = BargeIn(barge)
        self._publish = publish
        self._mic_control = mic_control
        self._translate = translate
        self.segmenter = segmenter  # for the echo threshold boost while speaking
        self.transcript_wake = transcript_wake  # no wake model: look for "jarvis" in transcripts instead
        self.clock = clock
        self.lang = self.opt.default_lang
        self.muted: str | None = None  # None | "soft" | "hard"
        self.interpreter = False
        self.followup_until = float("-inf")
        self._events: asyncio.Queue[Any] = asyncio.Queue()
        self._utts: asyncio.Queue[_UttItem] = asyncio.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._tasks: list[asyncio.Task] = []
        self._bg: set[asyncio.Task] = set()
        self._wake_times: deque[float] = deque(maxlen=16)
        self._onset_wall: float | None = None
        self._ctx_audio: tuple[np.ndarray, float] | None = None
        self._seg_ids = itertools.count(1)
        self._turn_ids = itertools.count(1)
        self._seg_text: dict[int, str] = {}
        self._seg_sink: dict[int, ReplySink | None] = {}
        self._seg_dur: dict[int, float] = {}
        self._playing: int | None = None
        self._playback_evt = asyncio.Event()
        self._last_player_event = 0.0
        self.turn_sink: ReplySink | None = None
        self._turn_task: asyncio.Task | None = None
        self.stats = {"utterances": 0, "ignored": 0, "rejected_owner": 0, "false_wakes": 0, "kills": 0, "barge_ins": 0}
        if hasattr(player, "on_event"):
            player.on_event = self._player_event_threadsafe

    # ------------------------------------------------------------------ plumbing
    def publish(self, topic: str, payload: dict) -> None:
        if self._publish is not None:
            try:
                self._publish(topic, payload)
            except Exception:
                log.debug("publish failed", exc_info=True)

    def emit_threadsafe(self, ev: object) -> None:
        """Hand to the audio front end as its ``emit``."""
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._events.put_nowait, ev)

    def _player_event_threadsafe(self, kind: str, seg_id: int) -> None:
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._events.put_nowait, ("player", kind, seg_id))

    def _state(self) -> None:
        mode = "muted" if self.muted == "hard" else self.barge.mode.value
        self.publish("voice.state", {"mode": mode, "muted": self.muted, "interpreter": self.interpreter,
                                     "lang": self.lang})

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._tasks = [asyncio.create_task(self._event_loop(), name="voice-events"),
                       asyncio.create_task(self._utterance_worker(), name="voice-utterances")]
        self._state()

    async def close(self) -> None:
        await self._end_turn()
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await t
        self._tasks = []

    async def drain(self) -> None:
        """Process everything queued so far (tests)."""
        while not self._events.empty() or not self._utts.empty():
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.01)

    # ------------------------------------------------------------------ events
    async def _event_loop(self) -> None:
        while True:
            ev = await self._events.get()
            try:
                if isinstance(ev, VadEvent):
                    self._on_vad(ev)
                elif isinstance(ev, WakeEvent):
                    self._wake_times.append(ev.wall)
                    self.publish("voice.wake", {"score": round(ev.score, 3)})
                elif isinstance(ev, LevelEvent):
                    self.publish("voice.level", {"mic": round(ev.mic, 4), "out": round(float(getattr(self.player, "level", 0.0)), 4)})
                elif isinstance(ev, tuple) and ev and ev[0] == "player":
                    self._on_player(ev[1], ev[2])
            except Exception:
                log.exception("voice event %r failed", type(ev).__name__)

    def feed_event(self, ev: object) -> None:
        """Inject an event from the event-loop thread (tests, local front ends)."""
        self._events.put_nowait(ev)

    def _on_vad(self, ev: VadEvent) -> None:
        if self.muted == "hard":
            return
        if ev.kind == "onset":
            self._onset_wall = ev.wall
            self._apply(self.barge.onset())
        elif ev.kind == "abort":
            self._apply(self.barge.onset_cancelled())
        elif ev.kind == "start":
            if self._onset_wall is None:
                self._onset_wall = ev.wall - ev.speech_ms / 1000.0
            self._apply(self.barge.speech_start())
            if self.barge.replying and ev.audio is not None and self.gate.enrolled:
                t = asyncio.create_task(self._early_score(ev.audio))
                self._bg.add(t)
                t.add_done_callback(self._bg.discard)
        elif ev.kind == "tick":
            acts = self.barge.speech_tick(ev.voiced_ms)
            if acts:
                self.stats["barge_ins"] += 1
            self._apply(acts)
        elif ev.kind == "end":
            onset = self._onset_wall if self._onset_wall is not None else ev.wall - ev.speech_ms / 1000.0
            self._onset_wall = None
            item = _UttItem(
                ev=ev,
                wake_hit=any(onset - 0.6 <= w <= ev.wall for w in self._wake_times),
                replying=self.barge.replying,
                interrupting=self.barge.interrupting,
                followup=ev.wall <= self.followup_until,
                fight=self.handler.fight_active(),
                onset_wall=onset,
                t_end_speech=onset + ev.speech_ms / 1000.0,
            )
            self._utts.put_nowait(item)

    async def _early_score(self, audio: np.ndarray) -> None:
        score = await asyncio.to_thread(self.gate.quick_score, audio)
        self.barge.speaker_score(score)

    def _on_player(self, kind: str, seg_id: int) -> None:
        self._last_player_event = self.clock()
        sink = self._seg_sink.get(seg_id)
        if kind == "start":
            self._playing = seg_id
            if self.segmenter is not None:
                self.segmenter.threshold_boost = self.opt.speaking_threshold_boost
            if sink is not None:
                if sink.t_first_audio is None:
                    sink.t_first_audio = self.clock()
                    self._publish_latency(sink)
                if sink is self.turn_sink:
                    self._apply(self.barge.tts_started())
                    self._state()
        elif kind == "end":
            if sink is not None:
                sink.ended.add(seg_id)
            else:
                self._forget_segment(seg_id)
            if self._playing == seg_id:
                self._playing = None
                if self.segmenter is not None:
                    self.segmenter.threshold_boost = 0.0
        self._playback_evt.set()

    def _forget_segment(self, seg_id: int) -> None:
        self._seg_text.pop(seg_id, None)
        self._seg_sink.pop(seg_id, None)
        self._seg_dur.pop(seg_id, None)

    def _publish_latency(self, sink: ReplySink) -> None:
        if sink.t_end_speech is None:
            return
        rel = lambda t: None if t is None else round(t - sink.t_end_speech, 3)  # noqa: E731
        self.publish("voice.latency", {"turn": sink.turn_id, "stt_s": rel(sink.t_stt), "first_text_s": rel(sink.t_first_text),
                                       "first_audio_s": rel(sink.t_first_audio)})

    # ------------------------------------------------------------------ barge-in actions
    def _apply(self, acts) -> None:
        for a in acts:
            if isinstance(a, Duck):
                self.player.set_gain(a.gain)
            elif isinstance(a, Unduck):
                self.player.set_gain(1.0)
            elif isinstance(a, PauseTts):
                self.player.pause()
            elif isinstance(a, ResumeTts):
                self.player.resume()
            elif isinstance(a, StopTts):
                self._stop_playback()
            elif isinstance(a, CancelLlm) and self.turn_sink is not None:
                self.turn_sink.cancel.set()

    def _stop_playback(self) -> None:
        rep = self.player.stop()
        self.player.set_gain(1.0)
        sink = self.turn_sink
        if sink is not None and sink.speak:
            ended = sink.ended | {s for s in rep.finished if s in sink.seg_ids}
            parts = [self._seg_text.get(s, "") for s in sink.seg_ids if s in ended]
            if rep.current is not None and rep.current in sink.seg_ids:
                parts.append(_cut_heard(self._seg_text.get(rep.current, ""), rep.current_fraction))
            sink.heard = " ".join(p for p in parts if p).strip()
            sink.cancel.set()
        self._playing = None
        if self.segmenter is not None:
            self.segmenter.threshold_boost = 0.0
        self._playback_evt.set()

    # ------------------------------------------------------------------ speaking
    async def speak_piece(self, sink: ReplySink, text: str, key: str | None) -> None:
        lang = detect_lang(text) if any(c.isalpha() for c in text) else sink.lang
        fighting = self.handler.fight_active()
        if key is None:
            while self.player.pending() >= self.opt.max_ahead and not sink.cancel.is_set():
                await asyncio.sleep(0.03)
        if sink.cancel.is_set():
            return
        if key is not None:
            synth = await asyncio.to_thread(self.router.phrase, key, lang, fighting=fighting)
        else:
            synth = await asyncio.to_thread(self.router.synth, clean_for_speech(text, lang), lang, fighting=fighting)
        if synth is None and fighting and not sink.said_on_screen:
            sink.said_on_screen = True  # in-fight replies go to the HUD; say so once (pre-rendered)
            synth = await asyncio.to_thread(self.router.phrase, "on_screen", lang, fighting=True)
        if synth is None or sink.cancel.is_set():
            return
        seg = next(self._seg_ids)
        self._seg_text[seg] = text
        self._seg_sink[seg] = sink
        self._seg_dur[seg] = synth.audio.duration_s
        sink.seg_ids.append(seg)
        self.player.enqueue(seg, synth.audio.pcm, synth.audio.rate, text)

    async def wait_playback(self, sink: ReplySink) -> None:
        """Until every segment of ``sink`` has played (or it was cancelled / the device stalled)."""
        self._last_player_event = self.clock()
        while not sink.cancel.is_set():
            left = [s for s in sink.seg_ids if s not in sink.ended]
            if not left:
                return
            budget = sum(self._seg_dur.get(s, 1.0) for s in left) + 3.0
            if self.clock() - self._last_player_event > budget:
                log.warning("audio output stalled; giving up on %d segment(s)", len(left))
                return
            self._playback_evt.clear()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._playback_evt.wait(), 0.25)

    async def say_phrase(self, key: str, lang: str | None = None) -> None:
        """A standalone pre-rendered phrase (acknowledgements, fight callouts)."""
        if self.router is None:  # text-only session
            return
        lang = lang or self.lang
        synth = await asyncio.to_thread(self.router.phrase, key, lang, fighting=self.handler.fight_active())
        if synth is None:
            return
        seg = next(self._seg_ids)
        self._seg_text[seg] = key
        self._seg_sink[seg] = None
        self.player.enqueue(seg, synth.audio.pcm, synth.audio.rate, key)

    async def announce(self, text: str, lang: str | None = None) -> bool:
        """Speak one standalone sentence (coach comments). Skipped while the owner talks or a reply plays;
        during a fight with cache-only TTS it stays on the HUD. Returns whether it was queued."""
        if self.router is None or not text.strip() or self.barge.replying or self._onset_wall is not None:
            return False
        lang = lang or detect_lang(text)
        synth = await asyncio.to_thread(self.router.synth, clean_for_speech(text, lang), lang, fighting=self.handler.fight_active())
        if synth is None:
            return False
        seg = next(self._seg_ids)
        self._seg_text[seg] = text
        self._seg_sink[seg] = None
        self.player.enqueue(seg, synth.audio.pcm, synth.audio.rate, text)
        return True

    async def callout(self, key: str, data: dict | None = None) -> None:
        """Speak a fightd callout (skipped while the owner talks or a reply plays, unless critical)."""
        phrase = callout_for_event(key, data or {})
        if phrase is None:
            return
        busy = self.barge.replying or self._onset_wall is not None
        if busy and phrase not in CRITICAL:
            return
        await self.say_phrase(phrase)

    # ------------------------------------------------------------------ turns
    async def _start_turn(self, utt: Utterance, *, speak: bool = True, t_end_speech: float | None = None,
                          t_stt: float | None = None) -> ReplySink:
        await self._end_turn()
        sink = ReplySink(self, next(self._turn_ids), utt.lang, speak=speak)
        sink.t_end_speech, sink.t_stt = t_end_speech, t_stt
        self.turn_sink = sink
        self._apply(self.barge.reply_started())
        self._state()
        self._turn_task = asyncio.create_task(self._run_turn(utt, sink), name=f"voice-turn-{sink.turn_id}")
        return sink

    async def _run_turn(self, utt: Utterance, sink: ReplySink) -> None:
        try:
            if utt.addressed_by == "interpreter" and self._translate is not None:
                translated, target = await self._translate(utt.text, utt.lang)
                sink.lang = target
                await sink.say(translated)
            else:
                await self.handler.handle(utt, sink)
            if not sink.cancel.is_set():
                await sink.finish()
        except asyncio.CancelledError:
            sink.cancel.set()
        except Exception:
            log.exception("voice turn failed")
        finally:
            sink.stop_worker()
            if sink.heard is not None:
                try:
                    await self.handler.interrupted(sink.heard)
                except Exception:
                    log.exception("recording the interruption failed")
            if self.turn_sink is sink:
                self.turn_sink = None
                if self.barge.replying:
                    self._apply(self.barge.reply_finished())
                self.followup_until = self.clock() + self.opt.followup_window_s
                self._state()
            self.publish("voice.reply_done", {"turn": sink.turn_id, "text": sink.text, "heard": sink.heard,
                                              "cancelled": sink.cancel.is_set()})
            for s in sink.seg_ids:
                self._forget_segment(s)

    async def _end_turn(self, timeout: float = 2.0) -> None:
        t, sink = self._turn_task, self.turn_sink
        if t is None or t.done():
            return
        if sink is not None:
            if sink.speak and any(s not in sink.ended for s in sink.seg_ids):
                self._stop_playback()  # records what was heard; clears a paused queue for the new reply
            sink.cancel.set()
        try:
            await asyncio.wait_for(asyncio.shield(t), timeout)
        except TimeoutError:
            t.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t

    async def wait_idle(self, timeout: float = 30.0) -> None:
        """Until the current turn (if any) has finished (CLI and tests)."""
        t = self._turn_task
        if t is not None and not t.done():
            await asyncio.wait_for(asyncio.shield(t), timeout)

    # ------------------------------------------------------------------ public controls
    def stop_all(self) -> None:
        """Kill-switch hook: stop speaking now and cancel the current turn."""
        if self.turn_sink is not None:
            self._stop_playback()
            self.turn_sink.cancel.set()
        else:
            self.player.stop()
            self.player.set_gain(1.0)
        if self.barge.replying or self.barge.interrupting:
            self._apply(self.barge.reply_finished())
        self._state()

    def mute(self, soft: bool | None = None) -> None:
        soft = self.opt.soft_mute if soft is None else soft
        self.muted = "soft" if soft else "hard"
        if not soft and self._mic_control is not None:
            self._mic_control(False)  # closes the device: the Windows mic indicator goes off
        self._state()

    def unmute(self) -> None:
        was = self.muted
        self.muted = None
        if was == "hard" and self._mic_control is not None:
            self._mic_control(True)
        self._state()

    async def typed(self, text: str, *, speak: bool = True) -> ReplySink | None:
        """Typed chat (TTT) from the HUD or CLI: trusted as the owner at the keyboard."""
        text = text.strip()
        if not text:
            return None
        intent = match_intent(text)
        if (intent is not None and intent.name == "stop") or (intent is None and is_kill_phrase(text, self.opt.kill_words)):
            await self._kill("typed stop")
            return None
        owner = SpeakerResult(True, 1.0, "typed at the PC")
        utt = Utterance(text, text, detect_lang(text), "typed", owner, intent)
        return await self._start_turn(utt, speak=speak)

    async def _kill(self, reason: str) -> None:
        self.stats["kills"] += 1
        self.stop_all()
        await self.handler.kill(reason)

    # ------------------------------------------------------------------ utterances
    async def _utterance_worker(self) -> None:
        while True:
            item = await self._utts.get()
            try:
                if self.clock() - item.ev.wall > self.opt.stale_utterance_s and not item.interrupting:
                    continue  # STT fell far behind; never act on old speech
                await self._process(item)
            except Exception:
                log.exception("processing an utterance failed")

    async def _stt(self, audio: np.ndarray, *, fast: bool = False) -> SttResult:
        return await asyncio.to_thread(self.stt.transcribe, audio, fast=fast, lang_hint=None)

    def _is_stop(self, text: str, intent: Intent | None) -> bool:
        if intent is not None:
            return intent.name == "stop"
        return is_kill_phrase(text, self.opt.kill_words)

    async def _verify_owner(self, audio: np.ndarray) -> SpeakerResult:
        ctx = None
        if self._ctx_audio is not None and self.clock() - self._ctx_audio[1] <= 6.0:
            ctx = self._ctx_audio[0]
        return await asyncio.to_thread(self.gate.verify_with_context, audio, ctx)

    def _ignore(self, item: _UttItem) -> None:
        self.stats["ignored"] += 1
        self._apply(self.barge.speech_end("", item.ev.voiced_ms))

    async def _process(self, item: _UttItem) -> None:
        ev = item.ev
        if self.muted == "hard" or ev.audio is None:
            self._apply(self.barge.speech_end("", ev.voiced_ms))
            return
        if self.muted == "soft" and ev.voiced_ms > 3000:
            return self._ignore(item)  # soft mute only listens for a short "unmute"
        self.stats["utterances"] += 1
        if item.replying or item.interrupting:
            addressed: str | None = "bargein"
        elif item.wake_hit:
            addressed = "wake"
        elif item.followup:
            addressed = "followup"
        elif self.interpreter:
            addressed = "interpreter"
        else:
            addressed = None

        if addressed is None and self.muted != "soft":
            if self.transcript_wake and ev.voiced_ms <= 8000:
                stt = await self._stt(ev.audio)
                if stt.ok and heard_wake_word(stt.text, self.opt.wake_keywords):
                    addressed = "wake"
                    return await self._handle_text(item, stt, addressed)
                if stt.ok and self._is_stop(stt.text, match_intent(stt.text)) and item.fight:
                    return await self._kill("voice stop (unaddressed, during a fight)")
                return self._ignore(item)
            if item.fight and ev.voiced_ms <= self.opt.kill_check_max_ms:
                stt = await self._stt(ev.audio, fast=True)
                if stt.ok and self._is_stop(stt.text, match_intent(stt.text)):
                    return await self._kill("voice stop (unaddressed, during a fight)")
            return self._ignore(item)

        stt = await self._stt(ev.audio)
        await self._handle_text(item, stt, addressed or "softmute")

    async def _handle_text(self, item: _UttItem, stt: SttResult, addressed: str) -> None:
        ev = item.ev
        t_stt = self.clock()
        raw = stt.text if stt.ok else ""
        intent = match_intent(raw) if raw else None
        is_stop = bool(raw) and self._is_stop(raw, intent)

        if addressed == "wake" and self.opt.second_stage and raw and not is_stop \
                and not heard_wake_word(raw, self.opt.wake_keywords):
            self.stats["false_wakes"] += 1
            self.publish("voice.ignored", {"why": "wake word not in transcript", "text": raw})
            return self._ignore(item)

        acts = self.barge.speech_end(raw, ev.voiced_ms, is_stop=is_stop)
        self._apply(acts)
        if is_stop:
            self.publish("voice.transcript", {"text": raw, "lang": stt.lang, "addressed_by": addressed, "intent": "stop"})
            return await self._kill("voice stop")
        if any(isinstance(a, ResumeTts) for a in acts) or not raw:
            return  # backchannel / noise while a reply was paused: it simply continues
        if addressed == "bargein" and self.barge.is_backchannel(raw, ev.voiced_ms):
            return  # a short "응/yeah" during a reply is not a new request

        text = strip_wake_word(raw)
        lang = stt.lang if stt.lang in ("ko", "en") else detect_lang(text or raw)
        self.publish("voice.transcript", {"text": raw, "lang": lang, "addressed_by": addressed,
                                          "intent": intent.name if intent else None})

        if self.muted == "soft":
            if intent is not None and intent.name == "unmute":
                owner = await self._verify_owner(ev.audio)
                if owner.is_owner:
                    self.unmute()
                    await self.say_phrase("unmuted", lang)
                else:
                    self.stats["rejected_owner"] += 1
            return

        if not text:  # just "hey Jarvis": acknowledge and keep listening without the wake word
            self._ctx_audio = (ev.audio, self.clock())
            self.followup_until = self.clock() + self.opt.followup_window_s
            await self.say_phrase("yes", lang)
            return

        if intent is not None and intent.name in ("mute", "interpreter_off"):
            if intent.name == "mute":
                await self.say_phrase("muted", lang)
                self.mute()
            else:
                self.interpreter = False
                await self.say_phrase("interpreter_off", lang)
                self._state()
            return

        if addressed == "interpreter" and (intent is None or intent.trust == Trust.ANY):
            utt = Utterance(text, raw, lang, "interpreter", SpeakerResult(True, 0.0, "interpreter"), intent, ev.voiced_ms, stt)
            await self._start_turn(utt, t_end_speech=item.t_end_speech, t_stt=t_stt)
            return

        needs_owner = (intent is not None and intent.trust == Trust.OWNER) or (intent is None and self.opt.require_owner_for_agent)
        owner = await self._verify_owner(ev.audio) if needs_owner else SpeakerResult(True, 0.0, "not required")
        if needs_owner and not owner.is_owner:
            self.stats["rejected_owner"] += 1
            self.publish("voice.rejected", {"text": raw, "score": round(owner.score, 3), "reason": owner.reason})
            await self.say_phrase("not_owner", lang)
            return
        if owner.is_owner and needs_owner:
            self.lang = lang
        self._ctx_audio = (ev.audio, self.clock())

        if intent is not None and intent.name == "interpreter_on":
            self.interpreter = True
            await self.say_phrase("interpreter_on", lang)
            self._state()
            return

        utt = Utterance(text, raw, lang, addressed, owner, intent, ev.voiced_ms, stt)
        await self._start_turn(utt, t_end_speech=item.t_end_speech, t_stt=t_stt)
