"""Giving the GPU back to Roblox during a fight.

The VRAM tier is chosen at setup with Roblox running, but a heavy map, a browser or another app can eat
the margin later. During a fight core watches free VRAM (NVML) and the Roblox frame rate fightd reports.
When either stays low it steps down, one level at a time:

  level 1: JARVIS's GPU voice (Qwen3-TTS) is unloaded - the CPU voice and pre-rendered callouts speak -
           and models the local LLM server holds in VRAM are unloaded;
  level 2: speech recognition moves to the CPU too (slower replies; "stop" stays fast).

After the fight ends, and a quiet minute, everything is restored. The decision logic is pure and tested;
core performs the actions.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

VOICE_OFF_GPU = "voice_off_gpu"
STT_OFF_GPU = "stt_off_gpu"
RESTORE = "restore"


@dataclass
class GuardConfig:
    min_free_mb: int = 400
    fps_floor: float = 30.0  # below this the fight suffers whatever the baseline
    fps_drop: float = 0.8  # or below 80% of this fight's own early frame rate
    baseline_s: float = 20.0  # the first seconds of a fight set its frame-rate baseline
    sustain: int = 3  # consecutive bad samples before acting (one sample ~ 2 s)
    min_step_s: float = 10.0
    recover_after_s: float = 60.0


@dataclass
class GpuGuard:
    cfg: GuardConfig = field(default_factory=GuardConfig)
    level: int = 0
    reason: str = ""

    def __post_init__(self) -> None:
        self._bad = 0
        self._last_step = float("-inf")
        self._fight_start: float | None = None
        self._fight_end: float | None = None
        self._early: list[float] = []
        self._recent: deque[float] = deque(maxlen=3)

    @property
    def baseline_fps(self) -> float | None:
        if len(self._early) < 3:
            return None
        s = sorted(self._early)
        return s[int(0.9 * (len(s) - 1))]

    def _pressure(self, free_mb: int | None, fps: float | None) -> str:
        c = self.cfg
        if free_mb is not None and free_mb < c.min_free_mb:
            return f"only {free_mb} MB of VRAM free"
        if fps is None or not self._recent:
            return ""
        med = sorted(self._recent)[len(self._recent) // 2]
        if med < c.fps_floor:
            return f"Roblox at {med:.0f} fps"
        base = self.baseline_fps
        if base and med < c.fps_drop * base:
            return f"Roblox dropped to {med:.0f} fps from {base:.0f}"
        return ""

    def update(self, now: float, fight_active: bool, free_mb: int | None, fps: float | None) -> list[str]:
        """One sample (every ~2 s). Returns the actions to perform now."""
        c = self.cfg
        if not fight_active:
            self._fight_start = None
            self._early.clear()
            self._recent.clear()
            self._bad = 0
            if self.level == 0:
                return []
            if self._fight_end is None:
                self._fight_end = now
            if now - self._fight_end >= c.recover_after_s:
                log.info("fight over: giving JARVIS its GPU models back")
                self.level, self.reason, self._fight_end = 0, "", None
                return [RESTORE]
            return []
        self._fight_end = None
        if self._fight_start is None:
            self._fight_start = now
        if fps is not None and fps > 0:
            if now - self._fight_start <= c.baseline_s:
                self._early.append(fps)
            self._recent.append(fps)
        why = self._pressure(free_mb, fps)
        self._bad = self._bad + 1 if why else 0
        if self._bad < c.sustain or self.level >= 2 or now - self._last_step < c.min_step_s:
            return []
        self.level += 1
        self._last_step = now
        self._bad = 0
        self.reason = why
        log.warning("GPU guard level %d: %s", self.level, why)
        return [VOICE_OFF_GPU if self.level == 1 else STT_OFF_GPU]


def unload_local_models(base_url: str, timeout: float = 3.0) -> list[str]:
    """Ask a local Ollama server to drop every model it holds in memory (keep_alive 0)."""
    import httpx

    root = base_url.rstrip("/").removesuffix("/v1")
    out = []
    with httpx.Client(timeout=timeout, trust_env=False) as http:  # a local server: never through a proxy
        try:
            loaded = [m["name"] for m in http.get(f"{root}/api/ps").json().get("models", [])]
        except Exception as e:
            log.info("local LLM server not reachable (%s)", e)
            return []
        for name in loaded:
            try:
                http.post(f"{root}/api/generate", json={"model": name, "keep_alive": 0})
                out.append(name)
            except Exception as e:
                log.warning("could not unload %s: %s", name, e)
    return out
