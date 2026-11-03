"""Local Korean/English command grammar: the fast path that never needs the network.

Order matters (more specific intents first: "음악 멈춰" is media, not stop; "싸우지 마" is disarm, not
fight). Commands only match short, command-like utterances: "I want to talk about yesterday's fight"
must never arm a fight, so every rule has a word limit and starting intents ignore questions.
Trust is asymmetric: stopping things is accepted from ANY voice; starting things (fight, unmute,
launch) needs the verified OWNER.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rapidfuzz import fuzz

from jarvis.safety.speaker_gate import Trust

WAKE_WORDS = ("hey jarvis", "hi jarvis", "ok jarvis", "jarvis", "헤이 자비스", "자비스야", "자비스", "재비스", "쟈비스")
# Politeness/filler words removed before matching ("소리 좀 줄여 줘" -> "소리 줄여").
FILLERS_KO = ("좀", "지금", "빨리", "어서", "그냥")
FILLERS_EN = ("please", "now", "just", "right now", "for me", "the")
SUFFIXES_KO = ("해 주세요", "해주세요", "해 줄래", "해줄래", "해 줘", "해줘", "주세요", "줄래", "줘", "요")


@dataclass
class Intent:
    name: str
    trust: Trust
    args: dict = field(default_factory=dict)
    text: str = ""


@dataclass(frozen=True)
class _Rule:
    name: str
    trust: Trust
    phrases: tuple[str, ...]
    max_words: int
    starts_something: bool = False  # starting intents ignore questions and negations


RULES: tuple[_Rule, ...] = (
    _Rule("media_pause", Trust.ANY, ("pause music", "stop music", "pause the music", "stop the music", "pause song",
                                     "음악 멈춰", "노래 멈춰", "음악 정지", "노래 정지", "음악 꺼", "노래 꺼", "일시정지"), 5),
    _Rule("media_next", Trust.ANY, ("next song", "next track", "skip song", "skip this song", "다음 곡", "다음 노래", "노래 넘겨"), 5),
    _Rule("media_play", Trust.ANY, ("play music", "resume music", "resume the music", "음악 틀어", "노래 틀어", "음악 재생", "노래 재생"), 5),
    _Rule("disarm", Trust.ANY, ("stop fighting", "stop the fight", "stand down", "don't fight", "do not fight", "cancel the fight",
                                "싸움 멈춰", "그만 싸워", "싸우지 마", "싸우지마", "전투 중지", "경기 중지", "대전 중지", "싸움 그만"), 5),
    _Rule("fight_status", Trust.ANY, ("fight status", "match status", "how did the fight go", "how did the match go", "match result",
                                      "fight result", "전적", "경기 결과", "싸움 결과", "몇 대 몇"), 6),
    _Rule("launch_ubg", Trust.OWNER, ("launch ubg", "open ubg", "start ubg", "launch untitled boxing game", "open untitled boxing game",
                                      "start untitled boxing game", "launch boxing game", "open boxing game", "start boxing game",
                                      "복싱 게임 켜", "복싱 게임 실행", "복싱 게임 열어", "복싱 켜", "언타이틀드 복싱 켜",
                                      "언타이틀드 복싱 실행", "로블록스 복싱 켜", "로블록스 복싱 실행"), 7, starts_something=True),
    _Rule("fight", Trust.OWNER, ("fight", "start fighting", "go fight", "start the fight", "start a fight", "begin the fight",
                                 "싸워", "싸움 시작", "대전 시작", "복싱 시작", "한판 해", "한 판 해", "경기 시작", "싸우자"),
          5, starts_something=True),
    _Rule("interpreter_on", Trust.OWNER, ("interpreter mode", "translation mode", "start translating", "start interpreting",
                                          "통역 모드", "통역 시작", "통역해", "번역 모드"), 5, starts_something=True),
    _Rule("interpreter_off", Trust.ANY, ("stop translating", "stop interpreting", "interpreter off", "exit interpreter",
                                         "통역 끝", "통역 종료", "통역 그만", "번역 그만"), 5),
    _Rule("unmute", Trust.OWNER, ("unmute", "unmute yourself", "음소거 해제", "마이크 켜", "다시 들어"), 4),
    _Rule("mute", Trust.ANY, ("mute", "mute yourself", "mute the mic", "stop listening", "음소거", "마이크 꺼", "듣지 마", "듣지마"), 4),
    _Rule("volume_up", Trust.ANY, ("volume up", "louder", "turn it up", "turn up the volume", "소리 키워", "볼륨 올려", "볼륨 업", "크게 해"), 5),
    _Rule("volume_down", Trust.ANY, ("volume down", "quieter", "turn it down", "turn down the volume", "소리 줄여", "볼륨 내려",
                                     "볼륨 다운", "작게 해"), 5),
    _Rule("time", Trust.ANY, ("what time is it", "what's the time", "몇 시야", "몇시야", "몇 시지", "몇시지", "몇 시예요", "몇시예요"), 5),
    _Rule("stop", Trust.ANY, ("stop", "abort", "cancel", "shut up", "be quiet", "enough", "멈춰", "그만", "정지", "스톱",
                              "조용히", "그만해", "취소", "됐어", "닥쳐"), 3),
)

_COUNT_EN = re.compile(r"\b(\d+|one|two|three|four|five)\s+(?:matches|match|games|game|rounds|round|times)\b")
_COUNT_KO = re.compile(r"(\d+|한|두|세|네|다섯)\s*(?:판|번|경기|게임)")
_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "한": 1, "두": 2, "세": 3, "네": 4, "다섯": 5}
_NEGATION = re.compile(r"\b(don't|do not|never|no)\b|지 ?마|하지 ?마|안 ?돼")


def strip_wake_word(text: str) -> str:
    t = text.strip()
    low = t.lower()
    for w in sorted(WAKE_WORDS, key=len, reverse=True):
        if low.startswith(w):
            rest = t[len(w):]
            if not rest or not rest[0].isalnum():
                return rest.lstrip(" ,.!?~").strip()
    return t


def normalize(text: str) -> str:
    t = strip_wake_word(text).lower()
    t = re.sub(r"[^\w\s']", " ", t)
    words = [w for w in t.split() if w not in FILLERS_KO]
    t = " ".join(words)
    for f in sorted(FILLERS_EN, key=len, reverse=True):
        t = re.sub(rf"\b{re.escape(f)}\b", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    for suf in SUFFIXES_KO:
        if t.endswith(suf) and len(t) > len(suf) + 1:
            t = t[: -len(suf)].strip()
            break
    return t


def _count(t: str) -> int | None:
    for rx in (_COUNT_EN, _COUNT_KO):
        m = rx.search(t)
        if m:
            v = m.group(1)
            return int(v) if v.isdigit() else _WORDS.get(v)
    return None


def _contains(t: str, phrase: str) -> bool:
    if re.fullmatch(r"[a-z' ]+", phrase):
        return re.search(rf"(?<![\w']){re.escape(phrase)}(?![\w'])", t) is not None
    return phrase in t


# Phrases go through the same normalization as utterances ("stop the fight" -> "stop fight").
_COMPILED: tuple[tuple[_Rule, tuple[str, ...]], ...] = tuple(
    (r, tuple(dict.fromkeys(normalize(p) for p in r.phrases))) for r in RULES
)


def match_intent(text: str, fuzzy_threshold: int = 90) -> Intent | None:
    raw = text.strip()
    t = normalize(raw)
    if not t:
        return None
    words = len(t.split())
    question = raw.rstrip().endswith(("?", "？"))
    negated = _NEGATION.search(t) is not None
    for rule, phrases in _COMPILED:
        if words > rule.max_words:
            continue
        if rule.starts_something and (question or negated):
            continue
        hit = any(_contains(t, p) for p in phrases)
        if not hit and words <= 4:
            hit = any(len(p) >= 5 and fuzz.ratio(t, p) >= fuzzy_threshold for p in phrases)
        if not hit:
            continue
        args: dict = {}
        if rule.name == "fight":
            n = _count(t)
            if n:
                args["matches"] = max(1, min(10, n))
            if "auto" in t or "알아서" in t or "자동" in t:
                args["auto_queue"] = True
        elif rule.name == "launch_ubg":
            args["private"] = not any(w in t for w in ("public", "공개"))
        return Intent(rule.name, rule.trust, args, raw)
    return None


def is_kill_phrase(text: str, kill_words: list[str]) -> bool:
    """True when a short utterance is one of the configured kill words (any voice may stop everything)."""
    t = normalize(text)
    return bool(t) and len(t.split()) <= 3 and any(_contains(t, w.lower()) for w in kill_words)
