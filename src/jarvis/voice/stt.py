"""Speech to text: faster-whisper large-v3-turbo (CUDA int8 on the 6 GB tier), Korean/English only.

* Language is restricted to the configured set (ko/en): if Whisper detects anything else, the clip is
  re-decoded with the most likely allowed language.
* Bilingual hotwords and an initial prompt bias spelling ("JARVIS", "자비스", "Roblox"...).
* Whisper invents text on silence/noise ("시청해 주셔서 감사합니다", "thanks for watching"); such
  results are rejected so they never become commands.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

# Phrases Whisper produces on silence or background audio (subtitle credits from its training data).
ALWAYS_REJECT = (
    "시청해 주셔서 감사합니다", "시청해주셔서 감사합니다", "구독과 좋아요", "구독 좋아요", "좋아요와 구독", "mbc 뉴스",
    "자막 제공", "자막 by", "다음 영상에서", "thank you for watching", "thanks for watching", "please subscribe",
    "subtitles by", "amara.org", "like and subscribe",
)
# Common but also plausible: rejected only when Whisper itself thinks there was probably no speech.
SUSPICIOUS = ("you", "thank you", "thanks", "bye", "감사합니다", "고맙습니다", "네", "아", "음", "우", "어")


@dataclass
class SttResult:
    text: str
    lang: str
    lang_prob: float = 0.0
    no_speech_prob: float = 0.0
    avg_logprob: float = 0.0
    duration_s: float = 0.0
    elapsed_s: float = 0.0
    rejected: str | None = None

    @property
    def ok(self) -> bool:
        return self.rejected is None and bool(self.text)


def reject_reason(text: str, no_speech_prob: float, avg_logprob: float) -> str | None:
    t = text.strip()
    if not t or not re.search(r"\w", t):
        return "empty"
    low = t.lower()
    if any(p in low for p in ALWAYS_REJECT):
        return "known hallucination"
    if no_speech_prob > 0.6 and avg_logprob < -1.0:
        return "no speech"
    if low.strip(" .!?~,") in SUSPICIOUS and no_speech_prob > 0.35:
        return "probably noise"
    if re.fullmatch(r"(.)\1{5,}", low.replace(" ", "")):
        return "repetition"
    return None


def cuda_available() -> bool:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


class WhisperStt:
    def __init__(self, model: Any, *, languages: list[str], beam_size: int = 3, hotwords: list[str] | None = None,
                 initial_prompt: str | None = None) -> None:
        self.model = model
        self.languages = list(languages)
        self.beam_size = beam_size
        self.hotwords = " ".join(hotwords) if hotwords else None
        self.initial_prompt = initial_prompt

    @classmethod
    def from_config(cls, cfg, models_dir: Path) -> WhisperStt:
        from faster_whisper import WhisperModel

        device = cfg.device
        if device == "auto":
            device = "cuda" if cuda_available() else "cpu"
        compute = cfg.compute_type if device == "cuda" else cfg.cpu_compute_type
        log.info("loading Whisper %s on %s (%s)", cfg.model, device, compute)
        model = WhisperModel(cfg.model, device=device, compute_type=compute, download_root=str(models_dir / "whisper"))
        return cls(model, languages=cfg.languages, beam_size=cfg.beam_size, hotwords=cfg.hotwords,
                   initial_prompt=cfg.initial_prompt)

    def _run(self, audio: np.ndarray, language: str | None, fast: bool):
        segs, info = self.model.transcribe(
            audio,
            language=language,
            beam_size=1 if fast else self.beam_size,
            temperature=0.0 if fast else [0.0, 0.2, 0.4],
            condition_on_previous_text=False,
            initial_prompt=self.initial_prompt,
            hotwords=self.hotwords,
            without_timestamps=True,
            vad_filter=False,
        )
        return list(segs), info

    def transcribe(self, audio: np.ndarray, *, fast: bool = False, lang_hint: str | None = None) -> SttResult:
        t0 = time.perf_counter()
        audio = np.asarray(audio, np.float32).reshape(-1)
        lang = lang_hint if lang_hint in self.languages else None
        segs, info = self._run(audio, lang, fast)
        if lang is None and self.languages and info.language not in self.languages:
            probs = dict(info.all_language_probs or [])
            best = max(self.languages, key=lambda code: probs.get(code, 0.0))
            log.debug("whisper detected %s; re-decoding as %s", info.language, best)
            segs, info = self._run(audio, best, fast)
        text = " ".join(s.text.strip() for s in segs).strip()
        no_speech = max((s.no_speech_prob for s in segs), default=1.0)
        avg_lp = float(np.mean([s.avg_logprob for s in segs])) if segs else -10.0
        return SttResult(
            text=text,
            lang=info.language if info.language in self.languages or not self.languages else self.languages[0],
            lang_prob=float(info.language_probability or 0.0),
            no_speech_prob=float(no_speech),
            avg_logprob=avg_lp,
            duration_s=len(audio) / 16000.0,
            elapsed_s=time.perf_counter() - t0,
            rejected=reject_reason(text, no_speech, avg_lp),
        )
