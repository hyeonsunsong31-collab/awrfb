"""System numbers for the HUD stats panel (CPU, RAM, GPU, VRAM)."""

from __future__ import annotations

import time

import psutil

from jarvis.common.gpu import primary_gpu


class StatsSampler:
    def __init__(self, gpu_every_s: float = 2.0) -> None:
        self._gpu_every = gpu_every_s
        self._gpu_t = float("-inf")
        self._gpu: dict | None = None
        psutil.cpu_percent(None)  # prime the counter

    def sample(self) -> dict:
        now = time.monotonic()
        if now - self._gpu_t >= self._gpu_every:  # nvidia-smi fallback is slow: sample less often
            self._gpu_t = now
            g = primary_gpu()
            self._gpu = None if g is None else {"name": g.name, "util": g.util_pct, "vram_used_mb": g.used_mb,
                                                "vram_total_mb": g.total_mb}
        vm = psutil.virtual_memory()
        return {"cpu": psutil.cpu_percent(None), "ram_used_gb": round(vm.used / 2**30, 1),
                "ram_total_gb": round(vm.total / 2**30, 1), "gpu": self._gpu}
