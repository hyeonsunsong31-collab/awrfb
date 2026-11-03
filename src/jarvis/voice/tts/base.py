"""Common TTS types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class TtsError(RuntimeError):
    pass


@dataclass
class Audio:
    pcm: np.ndarray  # float32 mono
    rate: int

    @property
    def duration_s(self) -> float:
        return len(self.pcm) / float(self.rate) if self.rate else 0.0


class TtsEngine(Protocol):
    name: str

    def synth(self, text: str, lang: str) -> Audio: ...


LANG_NAMES = {"ko": "Korean", "en": "English"}
