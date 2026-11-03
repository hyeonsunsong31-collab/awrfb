"""VoiceSession end to end with fake STT/TTS/player/handler (no audio hardware, no network)."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from jarvis.safety.speaker_gate import SpeakerGate
from jarvis.voice.audio_io import PlaybackReport
from jarvis.voice.frontend import WakeEvent
from jarvis.voice.session import SessionOptions, VoiceSession
from jarvis.voice.stt import SttResult
from jarvis.voice.tts.base import Audio
from jarvis.voice.tts.router import Synth
from jarvis.voice.vad import VadEvent


class Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


class FakeStt:
    def __init__(self, texts: list[str]) -> None:
        self.texts = list(texts)
        self.calls: list[bool] = []

    def transcribe(self, audio, *, fast=False, lang_hint=None) -> SttResult:
        self.calls.append(fast)
        text = self.texts.pop(0) if self.texts else ""
        lang = "ko" if any("가" <= c <= "힣" for c in text) else "en"
        return SttResult(text=text, lang=lang, no_speech_prob=0.05, avg_logprob=-0.2)


class FakeRouter:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def synth(self, text, lang, *, fighting=False, key=None):
        self.calls.append(("synth", text, lang, fighting))
        return None if fighting else Synth(Audio(np.zeros(1600, np.float32), 16000), "fake", False)

    def phrase(self, key, lang, *, fighting=False):
        self.calls.append(("phrase", key, lang))
        return Synth(Audio(np.zeros(800, np.float32), 16000), "cache", True)


class FakePlayer:
    def __init__(self) -> None:
        self.on_event = None
        self.queue: list[tuple[int, str]] = []
        self.current: int | None = None
        self.finished: list[int] = []
        self.log: list[tuple] = []
        self.level = 0.0

    def enqueue(self, seg_id, pcm, rate, text=""):
        self.queue.append((seg_id, text))

    def set_gain(self, g):
        self.log.append(("gain", g))

    def pause(self):
        self.log.append(("pause",))

    def resume(self):
        self.log.append(("resume",))

    def stop(self):
        rep = PlaybackReport(list(self.finished), self.current, 0.5 if self.current is not None else 0.0)
        self.queue.clear()
        self.current = None
        self.log.append(("stop",))
        return rep

    def busy(self):
        return bool(self.queue) or self.current is not None

    def pending(self):
        return len(self.queue)

    # test helpers (events go through the session's thread-safe bridge)
    def start_next(self):
        sid, _ = self.queue.pop(0)
        self.current = sid
        self.on_event("start", sid)

    def finish_current(self):
        sid, self.current = self.current, None
        self.finished.append(sid)
        self.on_event("end", sid)

    def play_all(self):
        while self.queue:
            self.start_next()
            self.finish_current()


class FakeHandler:
    def __init__(self, reply: str = "Sure. Opening notepad for you right now.") -> None:
        self.reply = reply
        self.turns = []
        self.interrupts: list[str] = []
        self.kills: list[str] = []
        self.fight = False
        self.hold: asyncio.Event | None = None

    async def handle(self, utt, reply):
        self.turns.append(utt)
        await reply.write(self.reply)
        if self.hold is not None:
            await asyncio.wait({asyncio.ensure_future(self.hold.wait()), asyncio.ensure_future(reply.cancel.wait())},
                               return_when=asyncio.FIRST_COMPLETED)

    async def interrupted(self, heard):
        self.interrupts.append(heard)

    def fight_active(self):
        return self.fight

    async def kill(self, reason):
        self.kills.append(reason)


class SignEmbedder:
    def embed(self, audio):
        a = np.asarray(audio)
        return np.array([np.mean(a > 0), np.mean(a < 0), 0.01])


def owner_audio(secs=1.0):
    return np.full(int(secs * 16000), 0.3, np.float32)


def tv_audio(secs=1.0):
    return np.full(int(secs * 16000), -0.3, np.float32)


@pytest.fixture
async def rig():
    clock = Clock()
    stt = FakeStt([])
    router = FakeRouter()
    player = FakePlayer()
    handler = FakeHandler()
    gate = SpeakerGate(SignEmbedder(), np.array([[1.0, 0.0, 0.01]]), threshold=0.45, min_audio_s=1.5)
    events: list[tuple[str, dict]] = []
    mic: list[bool] = []
    s = VoiceSession(stt=stt, router=router, player=player, gate=gate, handler=handler,
                     options=SessionOptions(followup_window_s=8.0, wake_keywords=["jarvis", "자비스"],
                                            kill_words=["stop", "멈춰", "그만"]),
                     publish=lambda t, p: events.append((t, p)), mic_control=mic.append, clock=clock)
    await s.start()
    yield s, clock, stt, router, player, handler, events, mic
    await s.close()


def utter(s: VoiceSession, clock: Clock, audio: np.ndarray, *, wake: bool = False, voiced_ms: float | None = None,
          ticks: bool = False) -> None:
    voiced = voiced_ms if voiced_ms is not None else len(audio) / 16.0
    t = clock.t
    s.feed_event(VadEvent("onset", 0.0, wall=t))
    if wake:
        s.feed_event(WakeEvent(0.9, t + 0.3))
    s.feed_event(VadEvent("start", 0.0, speech_ms=256, voiced_ms=256, audio=audio[:4096], wall=t + 0.26))
    if ticks:
        v = 256.0
        while v < voiced:
            v += 32
            s.feed_event(VadEvent("tick", 0.0, speech_ms=v, voiced_ms=v, wall=t + v / 1000))
    s.feed_event(VadEvent("end", 0.0, speech_ms=voiced, voiced_ms=voiced, audio=audio, wall=t + voiced / 1000 + 0.3))


async def settle(s: VoiceSession, player: FakePlayer | None = None, rounds: int = 400) -> None:
    """Run until the session has been idle for ~100 ms (playing out any queued audio)."""
    idle = 0
    for _ in range(rounds):
        await asyncio.sleep(0.005)
        if player is not None and player.queue:
            player.play_all()
            idle = 0
            continue
        busy = not s._events.empty() or not s._utts.empty() or (s._turn_task is not None and not s._turn_task.done())
        idle = 0 if busy else idle + 1
        if idle >= 20:
            return


async def test_wake_word_alone_then_command_without_repeating_it(rig):
    s, clock, stt, router, player, handler, events, _ = rig
    stt.texts = ["Hey Jarvis", "open notepad"]
    utter(s, clock, owner_audio(1.0), wake=True)
    await settle(s, player)
    assert ("phrase", "yes", "en") in router.calls and not handler.turns
    clock.t += 2.0
    utter(s, clock, owner_audio(1.0))  # no wake word: inside the follow-up window
    await settle(s, player)
    assert [u.text for u in handler.turns] == ["open notepad"]
    assert handler.turns[0].addressed_by == "followup" and handler.turns[0].owner.is_owner
    spoken = [c[1] for c in router.calls if c[0] == "synth"]
    assert spoken == ["Sure. Opening notepad for you right now."]  # "Sure." alone is too short to send
    assert any(t == "voice.latency" for t, _ in events)


async def test_unaddressed_speech_is_ignored_without_transcribing(rig):
    s, clock, stt, _router, player, handler, _events, _ = rig
    utter(s, clock, owner_audio(2.0))
    await settle(s, player)
    assert not handler.turns and stt.calls == [] and s.stats["ignored"] == 1


async def test_barge_in_reinput_records_only_heard_words(rig):
    s, clock, stt, _router, player, handler, _events, _ = rig
    handler.reply = "The first sentence is quite long indeed. The second sentence follows it."
    handler.hold = asyncio.Event()
    stt.texts = ["Jarvis tell me a story"]
    utter(s, clock, owner_audio(1.6), wake=True)
    for _ in range(400):
        await asyncio.sleep(0.005)
        if player.queue:
            break
    player.start_next()  # first sentence starts playing
    await asyncio.sleep(0.05)
    clock.t += 3.0
    stt.texts = ["open chrome instead"]
    handler.reply = "Opening Chrome."
    handler.hold = None
    utter(s, clock, owner_audio(1.6), ticks=True)
    await settle(s, player)
    assert ("gain", 0.25) in player.log and ("pause",) in player.log and ("stop",) in player.log
    assert handler.interrupts and handler.interrupts[0].startswith("The first") and handler.interrupts[0].endswith("…")
    assert [u.text for u in handler.turns] == ["tell me a story", "open chrome instead"]
    assert handler.turns[1].addressed_by == "bargein"


async def test_tv_saying_fight_is_rejected(rig):
    s, clock, stt, router, player, handler, events, _ = rig
    stt.texts = ["Jarvis, fight!"]
    utter(s, clock, tv_audio(0.8), wake=True)
    await settle(s, player)
    assert not handler.turns and s.stats["rejected_owner"] == 1
    assert ("phrase", "not_owner", "en") in router.calls
    assert any(t == "voice.rejected" for t, _ in events)


async def test_owner_short_fight_with_wake_context_is_accepted(rig):
    s, clock, stt, _router, player, handler, _events, _ = rig
    stt.texts = ["Hey Jarvis", "3판 싸워"]
    utter(s, clock, owner_audio(1.0), wake=True)
    await settle(s, player)
    clock.t += 1.5
    utter(s, clock, owner_audio(0.7))
    await settle(s, player)
    assert len(handler.turns) == 1
    turn = handler.turns[0]
    assert turn.intent is not None and turn.intent.name == "fight" and turn.intent.args == {"matches": 3}
    assert turn.owner.is_owner


async def test_anyone_can_stop_a_fight_without_the_wake_word(rig):
    s, clock, stt, _router, player, handler, _events, _ = rig
    handler.fight = True
    stt.texts = ["멈춰"]
    utter(s, clock, tv_audio(0.5))
    await settle(s, player)
    assert handler.kills and stt.calls == [True]  # one fast transcription only


async def test_soft_mute_only_hears_unmute(rig):
    s, clock, stt, router, player, handler, _events, mic = rig
    s.mute(soft=True)
    assert s.muted == "soft" and mic == []  # soft mute keeps the device open
    stt.texts = ["what is the weather like"]
    utter(s, clock, owner_audio(1.5))
    await settle(s, player)
    assert not handler.turns and s.muted == "soft"
    stt.texts = ["음소거 해제"]
    utter(s, clock, owner_audio(1.6))
    await settle(s, player)
    assert s.muted is None and ("phrase", "unmuted", "ko") in router.calls


async def test_hard_mute_closes_the_microphone(rig):
    s, clock, stt, router, player, _handler, _events, mic = rig
    stt.texts = ["Jarvis mute"]
    utter(s, clock, owner_audio(1.0), wake=True)
    await settle(s, player)
    assert s.muted == "hard" and mic == [False] and ("phrase", "muted", "en") in router.calls
    s.unmute()
    assert mic == [False, True] and s.muted is None


async def test_typed_chat_is_owner_and_can_be_silent(rig):
    s, _clock, _stt, router, _player, handler, events, _ = rig
    await s.typed("what time is it", speak=False)
    await s.wait_idle()
    assert handler.turns[0].addressed_by == "typed" and handler.turns[0].owner.reason == "typed at the PC"
    assert not [c for c in router.calls if c[0] == "synth"]
    assert any(t == "voice.reply" for t, _ in events)


async def test_in_fight_replies_go_to_screen_with_one_cached_line(rig):
    s, _clock, _stt, router, player, handler, _events, _ = rig
    handler.fight = True
    await s.typed("how are we doing", speak=True)
    await settle(s, player)
    assert [c for c in router.calls if c[0] == "phrase"] == [("phrase", "on_screen", "en")]


async def test_callouts_respect_busy_state_except_critical(rig):
    s, _clock, _stt, router, _player, handler, _events, _ = rig
    await s.callout("stock_won", {})
    assert ("phrase", "stock_won", "en") in router.calls
    handler.hold = asyncio.Event()
    await s.typed("tell me something long", speak=False)
    await asyncio.sleep(0.05)
    router.calls.clear()
    await s.callout("stock_lost", {})
    await s.callout("panic", {})
    assert router.calls == [("phrase", "panic", "en")]
    handler.hold.set()
    await s.wait_idle()
