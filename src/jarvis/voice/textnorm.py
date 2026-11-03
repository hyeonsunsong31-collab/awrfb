"""Text normalization for speech: numbers spelled out (Korean and English), markdown/URL/emoji cleanup,
language guess.

Korean has two number systems: Sino-Korean (일, 이, 삼...) for most numbers and native Korean (한, 두,
세...) before counters such as 시, 개, 명, 판. Neither Supertonic nor most small TTS models expand digits,
so every number is written out before synthesis.
"""

from __future__ import annotations

import re

DIGITS = "영일이삼사오육칠팔구"
SMALL_UNITS = ("", "십", "백", "천")
BIG_UNITS = ("", "만", "억", "조", "경")
NATIVE_ONES = ("", "한", "두", "세", "네", "다섯", "여섯", "일곱", "여덟", "아홉")
NATIVE_TENS = ("", "열", "스무", "서른", "마흔", "쉰", "예순", "일흔", "여든", "아흔")
# Counters read with native numbers (1..99). Some need a guard against Sino words that start the same
# way: 3개월 is 삼 개월, 2시즌 is 이 시즌.
NATIVE_COUNTERS = {
    "시간": "", "시": "(?!즌|리즈)", "개": "(?!월)", "명": "", "번": "", "판": "", "마리": "", "살": "",
    "잔": "", "권": "", "대": "", "장": "", "달": "", "곡": "", "가지": "", "사람": "", "군데": "",
    "벌": "", "켤레": "",
}


def int_to_sino(n: int) -> str:
    if n == 0:
        return "영"
    if n < 0:
        return "마이너스 " + int_to_sino(-n)
    parts: list[str] = []
    gi = 0
    while n > 0:
        n, group = divmod(n, 10000)
        if group:
            s = ""
            for i, unit in enumerate(SMALL_UNITS):
                d = (group // 10**i) % 10
                if d:
                    s = ("" if (d == 1 and i > 0) else DIGITS[d]) + unit + s
            if group == 1 and gi == 1:
                s = ""  # 10,000 is 만, not 일만
            parts.append(s + BIG_UNITS[gi])
        gi += 1
    return " ".join(reversed(parts))


def int_to_native(n: int) -> str | None:
    """1..99 in native Korean as used before counters (스무 살, 스물한 개). None outside that range."""
    if not 1 <= n <= 99:
        return None
    tens, ones = divmod(n, 10)
    t = "스물" if tens == 2 and ones else NATIVE_TENS[tens]
    return t + NATIVE_ONES[ones]


def _digits_ko(s: str) -> str:
    return "".join("공" if c == "0" else DIGITS[int(c)] for c in s if c.isdigit())


def number_to_korean(token: str) -> str:
    tok = token.replace(",", "")
    if "." in tok:
        whole, frac = tok.split(".", 1)
        return int_to_sino(int(whole or "0")) + " 점 " + "".join(DIGITS[int(c)] for c in frac if c.isdigit())
    if len(tok) > 1 and tok.startswith("0"):
        return _digits_ko(tok)  # phone numbers, codes: 010 -> 공일공
    return int_to_sino(int(tok))


def _month_ko(m: int) -> str:
    return {6: "유월", 10: "시월"}.get(m, f"{int_to_sino(m)} 월")


# Digit lookarounds instead of \b: Hangul counts as a word character, so "5:30에" has no boundary.
_NUM = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
_MINUS = re.compile(r"(?<![\w)])-(?=\d)")
_DATE = re.compile(r"(?<!\d)(\d{4})[-./](\d{1,2})[-./](\d{1,2})(?!\d)")
_TIME = re.compile(r"(?<![\d:])(\d{1,2}):(\d{2})(?![\d:])")
_PHONE = re.compile(r"(?<!\d)(0\d{1,2})-(\d{3,4})-(\d{4})(?!\d)")
_RANGE = re.compile(r"(?<=\d)\s*~\s*(?=\d)")
_MONTH = re.compile(r"(?<!\d)(1[0-2]|0?[1-9])\s*월(?!요일)")
_SCORE = re.compile(r"(?<![\d.,:-])(\d{1,2})\s*[-:]\s*(\d{1,2})(?![\d.,:-])")
_COUNTER = re.compile(
    r"(\d+)\s*(" + "|".join(re.escape(k) + g for k, g in sorted(NATIVE_COUNTERS.items(), key=lambda kv: -len(kv[0]))) + r")"
)


def korean_numbers(text: str) -> str:
    t = _DATE.sub(lambda m: f"{int_to_sino(int(m[1]))} 년 {_month_ko(int(m[2]))} {int_to_sino(int(m[3]))} 일", text)

    def clock(m: re.Match[str]) -> str:
        h, mi = int(m[1]), int(m[2])
        hour = f"{int_to_native(h) or int_to_sino(h)} 시"
        return hour if mi == 0 else f"{hour} {int_to_sino(mi)} 분"

    t = _TIME.sub(clock, t)
    t = _PHONE.sub(lambda m: ", ".join(_digits_ko(g) for g in m.groups()), t)
    t = _MONTH.sub(lambda m: _month_ko(int(m[1])), t)
    t = _RANGE.sub(" 에서 ", t)
    t = re.sub(r"(?<!\d)1\s*번째", "첫 번째", t)

    def counter(m: re.Match[str]) -> str:
        n = int(m[1])
        return f"{int_to_native(n) or int_to_sino(n)} {m[2]}"

    t = _COUNTER.sub(counter, t)
    t = _SCORE.sub(lambda m: f"{int_to_sino(int(m[1]))} 대 {int_to_sino(int(m[2]))}", t)
    t = t.replace("%", " 퍼센트")
    t = _MINUS.sub("마이너스 ", t)
    return _NUM.sub(lambda m: number_to_korean(m[0]), t)


# ------------------------------------------------------------------------------------------ English
_ONES_EN = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS_EN = ["_", "_", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_SCALES_EN = ((10**12, "trillion"), (10**9, "billion"), (10**6, "million"), (1000, "thousand"))
_ORD_EN = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth", "nine": "ninth", "twelve": "twelfth"}


def int_to_english(n: int) -> str:
    if n < 0:
        return "minus " + int_to_english(-n)
    if n < 20:
        return _ONES_EN[n]
    if n < 100:
        t, o = divmod(n, 10)
        return _TENS_EN[t] + (f"-{_ONES_EN[o]}" if o else "")
    if n < 1000:
        h, r = divmod(n, 100)
        return f"{_ONES_EN[h]} hundred" + (f" {int_to_english(r)}" if r else "")
    for value, name in _SCALES_EN:
        if n >= value:
            q, r = divmod(n, value)
            return f"{int_to_english(q)} {name}" + (f" {int_to_english(r)}" if r else "")
    return str(n)


def ordinal_english(n: int) -> str:
    words = int_to_english(n)
    head, sep, last = words.rpartition("-") if "-" in words.split(" ")[-1] else words.rpartition(" ")
    if last in _ORD_EN:
        last = _ORD_EN[last]
    elif last.endswith("y"):
        last = last[:-1] + "ieth"
    else:
        last += "th"
    return head + sep + last


def _year_english(y: int) -> str:
    if 2000 <= y <= 2009:
        return int_to_english(y)
    hi, lo = divmod(y, 100)
    if lo == 0:
        return f"{int_to_english(hi)} hundred"
    return f"{int_to_english(hi)} " + (f"oh {int_to_english(lo)}" if lo < 10 else int_to_english(lo))


def number_to_english(token: str) -> str:
    tok = token.replace(",", "")
    if "." in tok:
        whole, frac = tok.split(".", 1)
        return int_to_english(int(whole or "0")) + " point " + " ".join(_ONES_EN[int(c)] for c in frac if c.isdigit())
    if len(tok) > 1 and tok.startswith("0"):
        return " ".join("oh" if c == "0" else _ONES_EN[int(c)] for c in tok)
    n = int(tok)
    if len(tok) == 4 and 1100 <= n <= 2099 and "," not in token:
        return _year_english(n)
    return int_to_english(n)


_ORDINAL = re.compile(r"\b(\d+)(st|nd|rd|th)\b", re.I)


def english_numbers(text: str) -> str:
    def clock(m: re.Match[str]) -> str:
        h, mi = int(m[1]), int(m[2])
        if mi == 0:
            return f"{int_to_english(h)} o'clock"
        return f"{int_to_english(h)} " + (f"oh {int_to_english(mi)}" if mi < 10 else int_to_english(mi))

    t = _TIME.sub(clock, text)
    t = _RANGE.sub(" to ", t)
    t = _ORDINAL.sub(lambda m: ordinal_english(int(m[1])), t)
    t = _SCORE.sub(lambda m: f"{int_to_english(int(m[1]))} to {int_to_english(int(m[2]))}", t)
    t = re.sub(r"(\d)\s*%", r"\1 percent", t)
    t = _MINUS.sub("minus ", t)
    return _NUM.sub(lambda m: number_to_english(m[0]), t)


# -------------------------------------------------------------------------------------------- shared
_URL = re.compile(r"https?://\S+|www\.\S+")
_MD = re.compile(r"(\*\*|__|`{1,3}|^#{1,6}\s*|^>\s*|^\s*[-*+]\s+|^\s*\d+\.\s+)", re.M)
_EMOJI = re.compile("[\U0001f300-\U0001faff☀-➿️]")


def is_korean(text: str, threshold: float = 0.3) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    hangul = sum("가" <= c <= "힣" or "ㄱ" <= c <= "ㆎ" for c in letters)
    return hangul / len(letters) >= threshold


def detect_lang(text: str) -> str:
    return "ko" if is_korean(text) else "en"


def clean_for_speech(text: str, lang: str | None = None) -> str:
    lang = lang or detect_lang(text)
    t = _URL.sub("링크" if lang == "ko" else "a link", text)
    t = _MD.sub("", t)
    t = _EMOJI.sub("", t)
    t = t.replace("&", " 그리고 " if lang == "ko" else " and ")
    t = korean_numbers(t) if lang == "ko" else english_numbers(t)
    return re.sub(r"\s+", " ", t).strip()
