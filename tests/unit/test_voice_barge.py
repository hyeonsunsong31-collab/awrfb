"""Barge-in state machine: duck on onset, pause after 300 ms of voiced speech, then decide."""

from __future__ import annotations

from jarvis.voice.barge_in import (
    BargeIn,
    CancelLlm,
    CaptureAsInput,
    Duck,
    Mode,
    PauseTts,
    ResumeTts,
    StopTts,
    Unduck,
)


def speaking() -> BargeIn:
    b = BargeIn()
    b.reply_started()
    b.tts_started()
    assert b.mode == Mode.SPEAKING
    return b


def test_interruption_becomes_new_input():
    b = speaking()
    assert b.onset() == [Duck(0.25)]
    assert b.speech_start() == []  # already ducked
    assert b.speech_tick(100) == []
    assert b.speech_tick(320) == [PauseTts()]
    assert b.speech_tick(400) == []  # paused once
    assert b.speech_end("open notepad", 900) == [StopTts(), CancelLlm(), CaptureAsInput()]
    assert b.mode == Mode.LISTENING


def test_backchannel_resumes_where_it_paused():
    b = speaking()
    b.onset()
    b.speech_start()
    assert b.speech_tick(320) == [PauseTts()]
    assert b.speech_end("응", 350) == [ResumeTts(), Unduck()]
    assert b.mode == Mode.SPEAKING and not b.paused


def test_stop_word_stops_without_capture():
    b = speaking()
    b.onset()
    b.speech_start()
    b.speech_tick(330)
    assert b.speech_end("멈춰", 400, is_stop=True) == [StopTts(), CancelLlm()]
    assert b.mode == Mode.IDLE


def test_cough_is_only_a_brief_duck():
    b = speaking()
    assert b.onset() == [Duck(0.25)]
    assert b.onset_cancelled() == [Unduck()]


def test_clearly_not_owner_stays_ducked_and_keeps_talking():
    b = speaking()
    b.onset()
    b.speech_start()
    b.speaker_score(0.1)  # a TV in the room
    assert b.speech_tick(800) == []
    assert b.speech_end("some show dialogue going on and on", 2000) == [Unduck()]
    assert b.mode == Mode.SPEAKING


def test_talking_while_jarvis_thinks_holds_and_replaces_the_reply():
    b = BargeIn()
    b.reply_started()
    assert b.onset() == []  # nothing audible to duck yet
    b.speech_start()
    assert b.speech_tick(320) == [PauseTts()]  # holds audio from starting
    assert b.speech_end("actually, open chrome instead", 1500) == [StopTts(), CancelLlm(), CaptureAsInput()]


def test_idle_speech_is_not_a_barge_in():
    b = BargeIn()
    assert b.onset() == []
    assert b.speech_tick(500) == []
    assert b.speech_end("hello", 600) == []


def test_second_utterance_does_not_lose_the_pending_decision():
    b = speaking()
    b.onset()
    b.speech_start()
    assert b.speech_tick(320) == [PauseTts()]
    b.speech_start()  # the owner keeps talking before the first transcript is back
    assert b.speech_end("open notepad", 900) == [StopTts(), CancelLlm(), CaptureAsInput()]


def test_reply_finished_restores_volume():
    b = speaking()
    b.onset()
    assert b.reply_finished() == [Unduck()]
    assert b.mode == Mode.IDLE
