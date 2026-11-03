"""Fixed phrases: fight callouts and acknowledgements, pre-rendered outside fights so they play
instantly (and with zero GPU cost) during a match."""

from __future__ import annotations

from typing import Any

PHRASES: dict[str, dict[str, str]] = {
    # fight callouts (keys follow fightd's fight.callout events)
    "armed": {"en": "Armed. Waiting for the match to start.", "ko": "전투 준비 완료. 경기 시작을 기다립니다."},
    "match_start": {"en": "Fight.", "ko": "시작합니다."},
    "stock_won": {"en": "Stock taken.", "ko": "스톡 하나 가져왔습니다."},
    "stock_lost": {"en": "We lost a stock.", "ko": "스톡 하나 잃었습니다."},
    "final_stock_own": {"en": "Last stock. Careful now.", "ko": "마지막 스톡입니다."},
    "final_stock_opp": {"en": "They are on their last stock.", "ko": "상대가 마지막 스톡입니다."},
    "match_won": {"en": "Match won.", "ko": "경기에서 이겼습니다."},
    "match_lost": {"en": "Match lost.", "ko": "경기에서 졌습니다."},
    "match_over": {"en": "Match over.", "ko": "경기가 끝났습니다."},
    "queue_failed": {"en": "I couldn't find a free ring.", "ko": "빈 링을 찾지 못했습니다."},
    "disarmed": {"en": "Standing down.", "ko": "전투를 중지했습니다."},
    "panic": {"en": "Emergency stop. All keys released.", "ko": "긴급 정지. 모든 키를 놓았습니다."},
    # acknowledgements
    "yes": {"en": "Yes?", "ko": "네?"},
    "muted": {"en": "Microphone muted.", "ko": "마이크를 껐습니다."},
    "unmuted": {"en": "I'm listening again.", "ko": "다시 듣고 있습니다."},
    "not_owner": {"en": "Only my owner can ask me that.", "ko": "주인님 목소리로만 할 수 있는 명령입니다."},
    "stopped": {"en": "Stopped.", "ko": "멈췄습니다."},
    "launching": {"en": "Launching untitled boxing game.", "ko": "언타이틀드 복싱 게임을 실행합니다."},
    "on_screen": {"en": "It's on your screen.", "ko": "화면에 띄웠습니다."},
    "interpreter_on": {"en": "Interpreter mode on.", "ko": "통역 모드를 켭니다."},
    "interpreter_off": {"en": "Interpreter mode off.", "ko": "통역 모드를 끕니다."},
    "no_model": {"en": "I can't reach a language model right now.", "ko": "지금은 언어 모델에 연결할 수 없습니다."},
}

# Played even while the owner is talking or a reply is playing.
CRITICAL = {"panic", "disarmed", "stopped"}


def phrase(key: str, lang: str) -> str:
    p = PHRASES[key]
    return p.get(lang) or p["en"]


def callout_for_event(key: str, data: dict[str, Any]) -> str | None:
    """Map a fightd ``fight.callout`` event to a phrase key (None = nothing to say)."""
    if key == "match_end":
        result = str(data.get("result", ""))
        if result in ("won", "likely_won"):
            return "match_won"
        if result in ("lost", "likely_lost"):
            return "match_lost"
        return "match_over"
    if key == "final_stock_empty":
        return "final_stock_own" if data.get("side") == "own" else "final_stock_opp"
    return key if key in PHRASES else None
