"""Chooses the TTS engine per request: the tier's engine, then the fallback chain; cached audio first.

During a fight the tier's ``fight_engine`` rule applies: ``cache_only`` (6/8 GB cards) means only
pre-rendered phrases are spoken - nothing is synthesized while the GPU/CPU serve Roblox and the fighter.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from jarvis.voice.tts.base import Audio, TtsEngine
from jarvis.voice.tts.cache import TtsCache
from jarvis.voice.tts.callouts import PHRASES

log = logging.getLogger(__name__)

ENGINE_RETRY_S = 300.0  # an engine that failed to load is retried after this long
SYNTH_RETRY_S = 60.0  # an engine that failed a synthesis is skipped for this long


@dataclass
class Synth:
    audio: Audio
    engine: str
    cached: bool


class TtsRouter:
    def __init__(self, cfg, cache: TtsCache, factories: dict[str, Callable[[], TtsEngine]], *,
                 clock: Callable[[], float] = time.monotonic, cache_max_chars: int = 48) -> None:
        self.cfg = cfg
        self.cache = cache
        self._factories = factories
        self._engines: dict[str, TtsEngine] = {}
        self._down_until: dict[str, float] = {}
        self._locks: dict[str, threading.Lock] = {n: threading.Lock() for n in factories}
        self._load_lock = threading.Lock()
        self._clock = clock
        self.cache_max_chars = cache_max_chars
        self.last_error: str | None = None

    # ------------------------------------------------------------------ engines
    def chain(self) -> list[str]:
        names = ([] if self.cfg.engine == "auto" else [self.cfg.engine]) + list(self.cfg.fallback_chain)
        return [n for i, n in enumerate(names) if n in self._factories and n not in names[:i]]

    def primary(self) -> str:
        ch = self.chain()
        return ch[0] if ch else "none"

    def voice_id(self, name: str | None = None) -> str:
        """Identifies the voice so cached audio is re-rendered when the voice changes."""
        name = name or self.primary()
        c = self.cfg
        if name == "supertonic":
            return f"supertonic:{c.supertonic_model}:{c.supertonic_voice}:{c.supertonic_steps}:{c.speed}"
        if name == "qwen3":
            ref = c.qwen_voice_ref_wav or ""
            try:
                stamp = str(Path(ref).stat().st_mtime_ns) if ref else ""
            except OSError:
                stamp = ""
            return f"qwen3:{c.qwen_base_model}:{hashlib.sha1((ref + stamp).encode()).hexdigest()[:10]}:{c.speed}"
        if name == "edge":
            return f"edge:{c.edge_voice_en}:{c.edge_voice_ko}:{c.speed}"
        return name

    def _engine(self, name: str) -> TtsEngine | None:
        with self._load_lock:
            if name in self._engines:
                return self._engines[name]
            if self._down_until.get(name, 0.0) > self._clock():
                return None
            try:
                eng = self._factories[name]()
            except Exception as e:
                self.last_error = f"{name}: {e}"
                log.warning("TTS engine %s unavailable: %s", name, e)
                self._down_until[name] = self._clock() + ENGINE_RETRY_S
                return None
            self._engines[name] = eng
            return eng

    def available(self) -> list[str]:
        return [n for n in self.chain() if self._engine(n) is not None]

    # ------------------------------------------------------------------ keys
    def phrase_key(self, key: str, lang: str) -> str:
        return TtsCache.make_key("phrase", key, lang, self.voice_id())

    def _text_key(self, text: str, lang: str) -> str:
        return TtsCache.make_key("text", lang, text, self.voice_id())

    # ------------------------------------------------------------------ synthesis
    def synth(self, text: str, lang: str, *, fighting: bool = False, key: str | None = None) -> Synth | None:
        text = text.strip()
        if not text:
            return None
        k = key or (self._text_key(text, lang) if len(text) <= self.cache_max_chars else None)
        if k:
            cached = self.cache.get(k)
            if cached is not None:
                return Synth(cached, "cache", True)
        if fighting:
            if self.cfg.fight_engine == "cache_only":
                return None
            names = [self.cfg.fight_engine]
        else:
            names = self.chain()
        for name in names:
            if self._down_until.get(name, 0.0) > self._clock() and name in self._engines:
                continue
            eng = self._engine(name)
            if eng is None:
                continue
            try:
                with self._locks.setdefault(name, threading.Lock()):
                    audio = eng.synth(text, lang)
            except Exception as e:
                self.last_error = f"{name}: {e}"
                log.warning("TTS %s failed (%s); trying the next engine", name, e)
                self._down_until[name] = self._clock() + SYNTH_RETRY_S
                continue
            if len(audio.pcm) == 0:
                continue
            if k:
                self.cache.put(k, audio)
            return Synth(audio, name, False)
        return None

    def phrase(self, key: str, lang: str, *, fighting: bool = False) -> Synth | None:
        text = PHRASES[key].get(lang) or PHRASES[key]["en"]
        return self.synth(text, lang, fighting=fighting, key=self.phrase_key(key, lang))

    def prerender(self, keys: Iterable[str] | None = None, langs: Iterable[str] = ("ko", "en")) -> int:
        """Render fixed phrases into the cache (call outside fights). Returns how many were rendered."""
        n = 0
        for key in keys or PHRASES:
            for lang in langs:
                k = self.phrase_key(key, lang)
                if self.cache.has(k):
                    continue
                if self.phrase(key, lang) is not None:
                    n += 1
        return n
