"""NVIDIA GPU inventory via NVML (nvidia-ml-py), falling back to ``nvidia-smi``."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass


@dataclass
class GpuInfo:
    index: int
    name: str
    total_mb: int
    free_mb: int
    used_mb: int
    driver: str = ""
    util_pct: int | None = None


def _nvml() -> list[GpuInfo] | None:
    try:
        import pynvml  # provided by nvidia-ml-py
    except ImportError:
        return None
    try:
        pynvml.nvmlInit()
    except Exception:
        return None
    try:
        driver = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(driver, bytes):
            driver = driver.decode()
        out = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode()
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            try:
                util = int(pynvml.nvmlDeviceGetUtilizationRates(h).gpu)
            except Exception:
                util = None
            out.append(
                GpuInfo(
                    index=i,
                    name=str(name),
                    total_mb=int(mem.total // 2**20),
                    free_mb=int(mem.free // 2**20),
                    used_mb=int(mem.used // 2**20),
                    driver=str(driver),
                    util_pct=util,
                )
            )
        return out
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def _smi() -> list[GpuInfo]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        res = subprocess.run(
            [
                exe,
                "--query-gpu=index,name,memory.total,memory.free,memory.used,driver_version,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    out = []
    for line in res.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:
            continue
        try:
            out.append(
                GpuInfo(
                    index=int(parts[0]),
                    name=parts[1],
                    total_mb=int(float(parts[2])),
                    free_mb=int(float(parts[3])),
                    used_mb=int(float(parts[4])),
                    driver=parts[5],
                    util_pct=int(float(parts[6])) if parts[6] not in ("", "[N/A]") else None,
                )
            )
        except ValueError:
            continue
    return out


def gpus() -> list[GpuInfo]:
    found = _nvml()
    return found if found is not None else _smi()


def primary_gpu() -> GpuInfo | None:
    g = gpus()
    return max(g, key=lambda x: x.total_mb) if g else None
