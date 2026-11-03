"""Speech text handling: number reading (ko/en), sentence chunking, the bilingual command grammar."""

from __future__ import annotations

import pytest

from jarvis.safety.speaker_gate import Trust
from jarvis.voice.chunker import SentenceChunker
from jarvis.voice.intents import is_kill_phrase, match_intent, strip_wake_word
from jarvis.voice.textnorm import (
    clean_for_speech,
    detect_lang,
    english_numbers,
    int_to_native,
    int_to_sino,
    korean_numbers,
)


@pytest.mark.parametrize(
    ("text", "spoken"),
    [
        ("3시에 만나요", "세 시에 만나요"),
        ("2명이 3판 싸웠어요", "두 명이 세 판 싸웠어요"),
        ("3개월 동안", "삼개월 동안"),  # 개월 is Sino-Korean
        ("10,000원", "만원"),
        ("11000", "만 천"),
        ("100000000", "일억"),
        ("20살", "스무 살"),
        ("21살", "스물한 살"),
        ("1번째 경기", "첫 번째 경기"),
        ("스톡 3-2로 이겼습니다", "스톡 삼 대 이로 이겼습니다"),
        ("5:30에 시작", "다섯 시 삼십 분에 시작"),  # digits right before Hangul (no \b boundary)
        ("13:05", "열세 시 오 분"),
        ("2026-10-28에", "이천이십육 년 시월 이십팔 일에"),
        ("6월 10월에", "유월 시월에"),
        ("010-1234-5678로 전화", "공일공, 일이삼사, 오육칠팔로 전화"),
        ("3~5개", "삼 에서 다섯 개"),
        ("50%", "오십 퍼센트"),
        ("온도는 -5도", "온도는 마이너스 오도"),
        ("3.14", "삼 점 일사"),
        ("100개", "백 개"),
    ],
)
def test_korean_numbers(text, spoken):
    assert korean_numbers(text) == spoken


@pytest.mark.parametrize(
    ("text", "spoken"),
    [
        ("I won 3-2 at 5:30", "I won three to two at five thirty"),
        ("the 1st, 2nd, 3rd, 4th, 21st and 12th", "the first, second, third, fourth, twenty-first and twelfth"),
        ("it costs 1,250 dollars", "it costs one thousand two hundred fifty dollars"),
        ("50% done", "fifty percent done"),
        ("in 2026 and 1999 and 2005", "in twenty twenty-six and nineteen ninety-nine and two thousand five"),
        ("-5 degrees", "minus five degrees"),
        ("10:00", "ten o'clock"),
        ("3~5 items", "three to five items"),
        ("1,000,000", "one million"),
    ],
)
def test_english_numbers(text, spoken):
    assert english_numbers(text) == spoken


def test_number_helpers_and_cleanup():
    assert int_to_sino(0) == "영"
    assert int_to_sino(10000) == "만"
    assert int_to_native(99) == "아흔아홉"
    assert int_to_native(100) is None
    assert clean_for_speech("**Done!** See https://x.com 😀 & 3 items") == "Done! See a link and three items"
    assert clean_for_speech("# 결과\n- 3판 중 2판 승리") == "결과 세 판 중 두 판 승리"
    assert detect_lang("안녕하세요 JARVIS") == "ko"
    assert detect_lang("Hello there") == "en"


# ------------------------------------------------------------------------------------- chunker
def test_chunker_streams_sentences_and_keeps_decimals():
    ch = SentenceChunker()
    assert ch.feed("Hello there, my good friend. How are") == ["Hello there, my good friend."]
    assert ch.feed(" you today? Fine") == ["How are you today?"]
    assert ch.flush() == ["Fine"]
    ch = SentenceChunker()
    assert ch.feed("The value 3.14 is pi. ") == ["The value 3.14 is pi."]


def test_chunker_merges_short_pieces_and_handles_korean():
    ch = SentenceChunker(min_chars=8)
    assert ch.feed("Hi. Nice to meet you. ") == ["Hi. Nice to meet you."]
    ch = SentenceChunker(min_chars=8)
    assert ch.feed("안녕하세요. 반갑습니다! 오늘은") == ["안녕하세요. 반갑습니다!"]
    assert ch.flush() == ["오늘은"]


def test_chunker_early_comma_only_for_first_piece():
    ch = SentenceChunker(first_comma_min=28)
    out = ch.feed("Today the weather in Seoul is clear and warm, so ")
    assert out == ["Today the weather in Seoul is clear and warm,"]
    assert ch.feed("bring sunglasses, and enjoy it. Next, ") == ["so bring sunglasses, and enjoy it."]


def test_chunker_splits_very_long_text():
    ch = SentenceChunker(max_chars=50)
    out = ch.feed("word " * 30)
    assert out and all(len(p) <= 51 for p in out)


# ------------------------------------------------------------------------------------- intents
@pytest.mark.parametrize(
    ("text", "name", "args"),
    [
        ("Jarvis, fight!", "fight", {}),
        ("자비스 싸워", "fight", {}),
        ("3판 싸워줘", "fight", {"matches": 3}),
        ("fight three matches", "fight", {"matches": 3}),
        ("알아서 2판 싸워", "fight", {"matches": 2, "auto_queue": True}),
        ("싸우지 마", "disarm", {}),
        ("don't fight", "disarm", {}),
        ("stop the fight", "disarm", {}),
        ("그만 싸워", "disarm", {}),
        ("stop", "stop", {}),
        ("멈춰요", "stop", {}),
        ("자비스야 멈춰", "stop", {}),
        ("enough", "stop", {}),
        ("음악 멈춰", "media_pause", {}),
        ("stop the music", "media_pause", {}),
        ("소리 좀 줄여 줘", "volume_down", {}),
        ("turn up the volume", "volume_up", {}),
        ("음소거 해 줘", "mute", {}),
        ("음소거 해제", "unmute", {}),
        ("unmute", "unmute", {}),
        ("복싱 게임 켜줘", "launch_ubg", {"private": True}),
        ("launch UBG public server", "launch_ubg", {"private": False}),
        ("what's the time?", "time", {}),
        ("몇 시야?", "time", {}),
        ("통역 모드", "interpreter_on", {}),
        ("통역 그만", "interpreter_off", {}),
        ("fight status", "fight_status", {}),
        ("몇 대 몇이야", "fight_status", {}),
    ],
)
def test_intents_match(text, name, args):
    i = match_intent(text)
    assert i is not None and i.name == name and i.args == args


@pytest.mark.parametrize(
    "text",
    [
        "I want to talk about yesterday's fight",  # long sentence mentioning "fight" must never arm
        "왜 싸워?",  # a question, not a command
        "No, fight now",  # negation
        "how do I mute discord in the settings",
        "몇 시간 걸려?",
        "light",
        "jarvis",
    ],
)
def test_intents_reject_non_commands(text):
    assert match_intent(text) is None


def test_intent_trust_is_asymmetric():
    assert match_intent("fight").trust == Trust.OWNER
    assert match_intent("unmute").trust == Trust.OWNER
    assert match_intent("launch ubg").trust == Trust.OWNER
    for t in ("stop", "mute", "싸우지 마", "통역 그만", "volume down"):
        assert match_intent(t).trust == Trust.ANY


def test_wake_word_strip_and_kill_phrase():
    assert strip_wake_word("Hey Jarvis, open notepad") == "open notepad"
    assert strip_wake_word("자비스야 멈춰") == "멈춰"
    assert strip_wake_word("Jarvisopen") == "Jarvisopen"
    assert is_kill_phrase("그만", ["그만"])
    assert not is_kill_phrase("I can't stop thinking about pizza tonight", ["stop"])
