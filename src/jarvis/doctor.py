"""`jarvis doctor`: what this PC can run, what's missing, and which VRAM tier to use."""

from __future__ import annotations

import importlib
import json
import os
import platform
import shutil
import struct
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx
import psutil

from jarvis import __version__
from jarvis.common.config import JarvisConfig
from jarvis.common.gpu import gpus
from jarvis.common.tiers import choose_tier, write_tier

OPTIONAL_MODULES = {
    "windows_capture": "Roblox window capture (win extra)",
    "pyaudiowpatch": "speaker loopback for echo cancellation (win extra)",
    "uiautomation": "GUI control via UI Automation (win extra)",
    "pycaw": "volume control (win extra)",
    "keyring": "secure NVIDIA API key storage (win extra)",
    "pystray": "tray icon (win extra)",
    "sounddevice": "microphone/speaker audio (voice extra)",
    "openwakeword": "'hey jarvis' wake word (voice extra)",
    "faster_whisper": "speech-to-text (voice extra)",
    "livekit.rtc": "echo cancellation (voice extra)",
    "supertonic": "CPU text-to-speech (voice extra)",
    "edge_tts": "cloud TTS fallback (voice extra)",
    "onnxruntime": "ONNX inference (voice/ml extra)",
    "torch": "GPU inference/training (gpu extra)",
    "qwen_tts": "GPU text-to-speech (tts-qwen extra)",
    "speechbrain": "owner voice verification (speaker extra)",
    "rapidocr": "Korean/English OCR (ocr extra)",
    "ddgs": "web search (web extra)",
    "trafilatura": "web page text extraction (web extra)",
    "yt_dlp": "video ingestion (web extra)",
    "sqlite_vec": "vector memory (kb extra)",
    "ultralytics": "detector training (detect extra)",
}


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    level: str = "info"  # info | warn | error


@dataclass
class DoctorReport:
    version: str
    created: str
    checks: list[Check] = field(default_factory=list)
    gpu: list[dict[str, Any]] = field(default_factory=list)
    tier: dict[str, Any] | None = None

    def add(self, name: str, ok: bool, detail: str = "", level: str | None = None) -> None:
        self.checks.append(Check(name, ok, detail, level or ("info" if ok else "warn")))


def _module(name: str) -> tuple[bool, str]:
    try:
        m = importlib.import_module(name)
    except Exception as e:  # ImportError or a broken native dependency
        return False, f"{type(e).__name__}: {e}"[:160]
    return True, str(getattr(m, "__version__", ""))


def roblox_installed() -> tuple[bool, str]:
    if sys.platform != "win32":
        return False, "not Windows"
    local = Path(os.environ.get("LOCALAPPDATA", ""))
    hits = sorted((local / "Roblox" / "Versions").glob("*/RobloxPlayerBeta.exe")) if local else []
    if hits:
        return True, str(hits[-1])
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, r"roblox-player\shell\open\command") as k:
            cmd, _ = winreg.QueryValueEx(k, "")
            return True, f"protocol handler: {cmd}"
    except OSError:
        pass
    return False, "Roblox Player not found (install it from roblox.com and sign in once)"


def roblox_running() -> bool:
    for p in psutil.process_iter(["name"]):
        if (p.info.get("name") or "").lower() in ("robloxplayerbeta.exe", "windows10universal.exe"):
            return True
    return False


def ffmpeg_status(exe: str) -> tuple[bool, str]:
    path = shutil.which(exe)
    if not path:
        return False, "ffmpeg not found (needed to record proof videos)"
    try:
        out = subprocess.run([path, "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError) as e:
        return False, str(e)
    return True, "h264_nvenc available" if "h264_nvenc" in out else "no NVENC (will use libx264)"


def api_key_present(cfg: JarvisConfig) -> tuple[bool, str]:
    ep = cfg.brain.endpoints.get("nvidia")
    if ep is None:
        return False, "no nvidia endpoint configured"
    if ep.api_key_env and os.environ.get(ep.api_key_env):
        return True, f"from ${ep.api_key_env}"
    if ep.keyring_service:
        try:
            import keyring

            if keyring.get_password(ep.keyring_service, ep.keyring_user or "default"):
                return True, "from Windows Credential Manager"
        except Exception:
            pass
    return False, "not set - run `jarvis setup key` (never paste it into chat)"


def ollama_status(cfg: JarvisConfig) -> tuple[bool, str]:
    ep = next((e for e in cfg.brain.endpoints.values() if e.kind == "ollama"), None)
    if ep is None:
        return False, "no local endpoint configured"
    base = ep.base_url.rstrip("/").removesuffix("/v1")
    try:
        r = httpx.get(f"{base}/api/tags", timeout=2.0)
        r.raise_for_status()
        names = [m.get("name") for m in r.json().get("models", [])]
        return True, ", ".join(n for n in names if n) or "running, no models pulled"
    except Exception as e:
        return False, f"not reachable ({type(e).__name__}) - install Ollama for offline fallback"


def run_doctor(cfg: JarvisConfig, *, vram: bool = False) -> DoctorReport:
    rep = DoctorReport(__version__, time.strftime("%Y-%m-%dT%H:%M:%S"))
    rep.add("os", sys.platform == "win32", f"{platform.platform()}", None if sys.platform == "win32" else "warn")
    rep.add("python", sys.version_info >= (3, 11), sys.version.split()[0])
    rep.add("64-bit", struct.calcsize("P") == 8, f"{struct.calcsize('P') * 8}-bit", "error" if struct.calcsize("P") != 8 else None)
    rep.add("config", True, f"tier {cfg.tier}, data dir {cfg.paths().root}")

    try:
        from jarvis.safety.policy import load_policy

        pol = load_policy()
        rep.add(
            "safety policy",
            pol.protected,
            f"{pol.source} ({'admin-owned, read-only' if pol.protected else 'NOT protected - run scripts/install.ps1 as the owner'})",
        )
    except Exception as e:
        rep.add("safety policy", False, str(e), "error")

    for mod, why in OPTIONAL_MODULES.items():
        ok, detail = _module(mod)
        rep.add(f"module {mod}", ok, f"{why}: {detail}" if detail else why)

    ok, detail = roblox_installed()
    rep.add("roblox installed", ok, detail)
    rep.add("roblox running", roblox_running(), "")
    ok, detail = ffmpeg_status(cfg.fight.recording.ffmpeg)
    rep.add("ffmpeg", ok, detail)
    ok, detail = api_key_present(cfg)
    rep.add("nvidia api key", ok, detail)
    ok, detail = ollama_status(cfg)
    rep.add("ollama (local fallback)", ok, detail)

    try:
        import sounddevice as sd

        dev_in = sd.query_devices(kind="input")
        dev_out = sd.query_devices(kind="output")
        rep.add("audio", True, f"in: {dev_in['name']} | out: {dev_out['name']}")
    except Exception as e:
        rep.add("audio", False, f"{type(e).__name__}: {e}"[:160])

    found = gpus()
    rep.gpu = [asdict(g) for g in found]
    if not found:
        rep.add("nvidia gpu", False, "no NVIDIA GPU detected (vision/STT will be slow)", "warn")
    else:
        g = max(found, key=lambda x: x.total_mb)
        rep.add("nvidia gpu", True, f"{g.name}, {g.total_mb} MB total, {g.free_mb} MB free, driver {g.driver}")
        if vram:
            free_with_roblox = g.free_mb if roblox_running() else None
            decision = choose_tier(g.total_mb, free_with_roblox)
            write_tier(cfg.paths().ensure(), decision)
            rep.tier = asdict(decision)
            note = "" if free_with_roblox is not None else " (start Roblox and re-run for a measured tier)"
            rep.add("vram tier", True, f"{decision.tier}: {decision.reason}{note}")
    return rep


def save_report(rep: DoctorReport, reports_dir: Path) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    p = reports_dir / f"doctor-{time.strftime('%Y%m%d-%H%M%S')}.json"
    p.write_text(json.dumps(asdict(rep), indent=2, ensure_ascii=False), encoding="utf-8")
    return p
