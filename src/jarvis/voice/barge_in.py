"""Barge-in (interruption) state machine - pure, testable with a fake clock.

"Pause, then decide":
  * speech onset while JARVIS speaks  -> DUCK the voice at once (target: < 150 ms)
  * >= stop_after_ms of voiced speech -> PAUSE the voice (it never talks over the owner)
  * when the owner's utterance ends (text from STT):
      - nothing / a backchannel ("응", "yeah") -> RESUME where it paused
      - a stop word                           -> STOP + cancel the LLM
      - anything else                         -> STOP + cancel the LLM + treat the speech as the
                                                 next request ("re-input")
A clearly-not-owner speaker score (a TV, a friend) keeps the voice ducked instead of pausing it; stop
words still work from any voice because they are decided on the transcript.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Mode(StrEnum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


@dataclass(frozen=True)
class Duck:
    gain: float


@dataclass(frozen=True)
class Unduck:
    pass


@dataclass(frozen=True)
class PauseTts:
    pass


@dataclass(frozen=True)
class ResumeTts:
    pass


@dataclass(frozen=True)
class StopTts:
    pass


@dataclass(frozen=True)
class CancelLlm:
    pass


@dataclass(frozen=True)
class CaptureAsInput:
    """The owner's utterance is the next request."""


Action = Duck | Unduck | PauseTts | ResumeTts | StopTts | CancelLlm | CaptureAsInput


@dataclass
class BargeConfig:
    duck_gain: float = 0.25
    stop_after_ms: float = 300.0
    require_owner_for_stop: bool = True
    not_owner_score: float = 0.25
    backchannel_max_ms: float = 900.0
    backchannel_words: tuple[str, ...] = ("응", "네", "예", "음", "어", "yeah", "yes", "uh-huh", "mm", "ok", "okay", "right")


class BargeIn:
    def __init__(self, cfg: BargeConfig | None = None) -> None:
        self.cfg = cfg or BargeConfig()
        self.mode = Mode.IDLE
        self.ducked = False
        self.paused = False
        self.interrupting = False  # a pause was issued for the current utterance
        self._owner_score: float | None = None

    @property
    def replying(self) -> bool:
        return self.mode in (Mode.THINKING, Mode.SPEAKING)

    # ------------------------------------------------------------------ reply lifecycle
    def reply_started(self) -> list[Action]:
        self.mode = Mode.THINKING
        return []

    def tts_started(self) -> list[Action]:
        if self.mode == Mode.THINKING:
            self.mode = Mode.SPEAKING
        return []

    def reply_finished(self) -> list[Action]:
        acts: list[Action] = [Unduck()] if self.ducked else []
        self.ducked = self.paused = self.interrupting = False
        self.mode = Mode.IDLE
        return acts

    # ------------------------------------------------------------------ owner speech
    def onset(self) -> list[Action]:
        if self.mode == Mode.SPEAKING and not self.ducked:
            self.ducked = True
            return [Duck(self.cfg.duck_gain)]
        return []

    def onset_cancelled(self) -> list[Action]:
        if self.ducked and not self.interrupting:
            self.ducked = False
            return [Unduck()]
        return []

    def speech_start(self) -> list[Action]:
        # ``interrupting`` is NOT cleared here: the previous utterance's decision may still be waiting
        # for its transcript; only ``speech_end`` consumes it.
        self._owner_score = None
        return self.onset()

    def speaker_score(self, score: float | None) -> list[Action]:
        self._owner_score = score
        return []

    def speech_tick(self, voiced_ms: float) -> list[Action]:
        if self.interrupting or not self.replying or voiced_ms < self.cfg.stop_after_ms:
            return []
        if self.cfg.require_owner_for_stop and self._owner_score is not None and self._owner_score < self.cfg.not_owner_score:
            return []  # clearly someone else: stay ducked, keep talking
        self.interrupting = True
        self.paused = True
        return [PauseTts()]

    def speech_end(self, text: str, voiced_ms: float, *, is_stop: bool = False) -> list[Action]:
        if not self.interrupting:
            if is_stop and self.replying:
                return self._stop(capture=False)
            acts: list[Action] = []
            if self.ducked:
                self.ducked = False
                acts.append(Unduck())
            return acts
        self.interrupting = False
        if is_stop:
            return self._stop(capture=False)
        if not text.strip() or self.is_backchannel(text, voiced_ms):
            self.paused = False
            acts = [ResumeTts()]
            if self.ducked:
                self.ducked = False
                acts.append(Unduck())
            return acts
        return self._stop(capture=True)

    def _stop(self, *, capture: bool) -> list[Action]:
        self.ducked = self.paused = self.interrupting = False
        self.mode = Mode.LISTENING if capture else Mode.IDLE
        acts: list[Action] = [StopTts(), CancelLlm()]
        if capture:
            acts.append(CaptureAsInput())
        return acts

    def is_backchannel(self, text: str, voiced_ms: float) -> bool:
        t = text.strip().strip(".!?~,… ").lower()
        return voiced_ms <= self.cfg.backchannel_max_ms and t in self.cfg.backchannel_words
