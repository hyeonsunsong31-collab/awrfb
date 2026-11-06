"""`jarvis selftest`: every check that can run on this PC, in one command, with one report.

It turns "verify on your PC" into something you run and send back. Without flags it needs no game and no
microphone; ``--interactive`` adds the checks that need you (speak into the mic); ``--live`` adds the ones
that need Roblox open on UBG. Each check passes, warns (works but degraded or needs attention), fails, or
is skipped with the reason. The JSON report lands in ``reports/`` and goes into ``jarvis report``.

Nothing here sends a single key to Roblox: the live checks watch the window and prove that the input path
refuses to act while disarmed.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import platform
import re
import socket
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from jarvis.common.config import JarvisConfig

PASS, WARN, FAIL, SKIP = "pass", "warn", "fail", "skip"
SENTENCES = {
    "ko": "안녕하세요, 저는 자비스입니다. 무엇을 도와드릴까요?",
    "en": "Hello, I am Jarvis. How can I help you today?",
}
WAKE_PHRASE = "Hey Jarvis."
SR = 16000

# Win32 extended window styles the HUD must have.
WS_EX_TOPMOST = 0x00000008
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_LAYERED = 0x00080000
WS_EX_NOACTIVATE = 0x08000000


@dataclass
class Result:
    name: str
    status: str
    detail: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0


class Skip(Exception):
    """The check does not apply here (and says why)."""


@dataclass
class Ctx:
    cfg: JarvisConfig
    live: bool = False
    interactive: bool = False
    say: Callable[[str], None] = print
    factories: dict[str, Callable[..., Any]] = field(default_factory=dict)  # tests inject fakes here
    shared: dict[str, Any] = field(default_factory=dict)
    windows: bool = sys.platform == "win32"

    def make(self, name: str, default: Callable[..., Any], *args: Any) -> Any:
        return self.factories.get(name, default)(*args)


@dataclass(frozen=True)
class Check:
    name: str
    fn: Callable[[Ctx], Result | list[Result]]
    title: str
    windows_only: bool = False
    live: bool = False
    interactive: bool = False


# ------------------------------------------------------------------------------------- helpers
def squash(text: str) -> str:
    return re.sub(r"[\W_]+", "", text.lower())


def similarity(expected: str, heard: str) -> float:
    """0-100: how close a transcript is to the sentence spoken (spacing and punctuation ignored)."""
    from rapidfuzz import fuzz

    return float(fuzz.ratio(squash(expected), squash(heard)))


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def supervisor_state(cfg: JarvisConfig) -> dict | None:
    """The running supervisor's last snapshot, or None when JARVIS is not running."""
    p = cfg.paths().run / "supervisor.json"
    try:
        snap = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    try:
        import psutil

        if not psutil.pid_exists(int(snap.get("pid", -1))):
            return None
    except Exception:
        return None
    return snap if time.time() - float(snap.get("t", 0)) < 30 else None


def hud_flags(exstyle: int) -> dict[str, bool]:
    return {
        "always_on_top": bool(exstyle & WS_EX_TOPMOST),
        "never_takes_focus": bool(exstyle & WS_EX_NOACTIVATE),
        "click_through": bool(exstyle & WS_EX_TRANSPARENT) and bool(exstyle & WS_EX_LAYERED),
    }


def hud_verdict(connected: bool, flags: dict[str, bool] | None, focus_kept: bool | None) -> tuple[str, str]:
    if not connected:
        return FAIL, "the HUD never connected to the hub (see logs/hud.out.log, or run `jarvis hud`)"
    if flags is None:
        return WARN, "connected, but its window was not found to check its flags"
    missing = [k for k, v in flags.items() if not v]
    if missing:
        return FAIL, f"HUD window is missing: {', '.join(missing)}"
    if focus_kept is False:
        return FAIL, "the HUD took keyboard focus when it appeared"
    return PASS, "connected; always on top, click-through, never takes focus"


def _speech(ctx: Ctx, lang: str) -> np.ndarray | None:
    return ctx.shared.get(f"speech_{lang}")


# ------------------------------------------------------------------------------------- checks: this PC
def check_doctor(ctx: Ctx) -> Result:
    from jarvis.doctor import run_doctor

    rep = run_doctor(ctx.cfg)
    errors = [c for c in rep.checks if not c.ok and c.level == "error"]
    warns = [c for c in rep.checks if not c.ok and c.level != "error"]
    status = FAIL if errors else (WARN if warns else PASS)
    look = errors + warns
    detail = (f"{len(look)} to look at: " + ", ".join(c.name for c in look[:8]) + (" ..." if len(look) > 8 else "")
              + " (details in the report / `jarvis doctor`)") if look else "all checks ok"
    return Result("doctor", status, detail, {"checks": [asdict(c) for c in rep.checks], "gpu": rep.gpu})


def check_ipc(ctx: Ctx) -> Result:
    """Event bus + token RPC round trip on spare ports (the plumbing every JARVIS process uses)."""
    from jarvis.common.ipc import AsyncSubscriber, BusProxy, Publisher, RpcClient, RpcServer

    async def go() -> tuple[Any, Any]:
        p_in, p_out, p_rpc = free_port(), free_port(), free_port()
        proxy = BusProxy(p_in, p_out)
        proxy.start()
        srv = RpcServer("selftest", p_rpc, "selftest-token")
        srv.add("ping", lambda p: {"pong": p.get("n")})
        await srv.start()
        client = RpcClient(p_rpc, "selftest-token", timeout=3.0)
        sub = AsyncSubscriber(p_out, ["selftest."])
        pub = Publisher(p_in)
        try:
            rpc = await client.call("ping", {"n": 7})
            got = None
            for _ in range(50):  # PUB/SUB joins asynchronously: repeat until heard
                pub.publish("selftest.ping", {"n": 1})
                try:
                    got = await asyncio.wait_for(sub.recv(), 0.1)
                    break
                except TimeoutError:
                    continue
            return rpc, got
        finally:
            pub.close()
            sub.close()
            await client.close()
            await srv.close()
            proxy.stop()

    rpc, got = asyncio.run(go())
    if rpc != {"pong": 7}:
        return Result("ipc", FAIL, f"RPC round trip returned {rpc!r}")
    if got is None:
        return Result("ipc", FAIL, "event bus delivered nothing within 5 s")
    return Result("ipc", PASS, "RPC and event bus round trips work")


def check_gpu_runtime(ctx: Ctx) -> Result:
    from jarvis.common.cuda_dlls import setup_cuda_dlls
    from jarvis.common.gpu import primary_gpu

    if primary_gpu() is None:
        raise Skip("no NVIDIA GPU detected")
    setup_cuda_dlls()
    data: dict[str, Any] = {}
    problems: list[str] = []
    try:
        import onnxruntime as ort

        data["onnxruntime_providers"] = ort.get_available_providers()
        probe = next((p for p in (ctx.cfg.paths().models / "silero_vad.onnx",
                                  ctx.cfg.paths().models / "openwakeword" / "melspectrogram.onnx") if p.exists()), None)
        if probe is not None:  # a real session: CUDA DLL problems only show up here
            sess = ort.InferenceSession(str(probe), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            data["onnxruntime_session"] = sess.get_providers()
            if "CUDAExecutionProvider" not in sess.get_providers():
                problems.append("ONNX Runtime cannot use CUDA (fighter models run on the CPU)")
        elif "CUDAExecutionProvider" not in data["onnxruntime_providers"]:
            problems.append("onnxruntime-gpu is not installed")
    except ImportError:
        problems.append("onnxruntime is not installed")
    try:
        import ctranslate2

        data["ctranslate2_cuda_devices"] = ctranslate2.get_cuda_device_count()
        if not data["ctranslate2_cuda_devices"]:
            problems.append("Whisper (CTranslate2) sees no CUDA device")
    except ImportError:
        problems.append("faster-whisper/CTranslate2 is not installed")
    with contextlib.suppress(ImportError):
        import torch

        data["torch_cuda"] = bool(torch.cuda.is_available())
        data["torch_version"] = torch.__version__
    return Result("gpu_runtime", WARN if problems else PASS, "; ".join(problems) or "CUDA works for ONNX Runtime and Whisper", data)


def check_tts(ctx: Ctx) -> Result:
    from jarvis.voice.resample import resample
    from jarvis.voice.runtime import build_router

    router = ctx.make("router", build_router, ctx.cfg)
    data: dict[str, Any] = {}
    for key, (text, lang) in {"ko": (SENTENCES["ko"], "ko"), "en": (SENTENCES["en"], "en"), "wake": (WAKE_PHRASE, "en")}.items():
        for name in router.chain():
            eng = router.engine(name)
            if eng is None:
                continue
            t0 = time.perf_counter()
            try:
                audio = eng.synth(text, lang)
            except Exception as e:
                data.setdefault("errors", []).append(f"{name}/{lang}: {type(e).__name__}: {e}"[:200])
                continue
            dt = time.perf_counter() - t0
            if len(audio.pcm) == 0:
                continue
            pcm = resample(np.asarray(audio.pcm, np.float32), audio.rate, SR) if audio.rate != SR else audio.pcm
            ctx.shared[f"speech_{key}"] = np.asarray(pcm, np.float32)
            data[key] = {"engine": name, "audio_s": round(audio.duration_s, 2), "synth_s": round(dt, 2),
                         "rtf": round(dt / max(audio.duration_s, 1e-3), 2)}
            break
    voiced = [k for k in ("ko", "en") if k in data]
    if not voiced:
        return Result("tts", FAIL, "no voice engine produced audio: " + "; ".join(data.get("errors", [])[:3]), data)
    engines = {data[k]["engine"] for k in voiced}
    if len(voiced) < 2:
        return Result("tts", WARN, f"only {', '.join(voiced)} speech works", data)
    if engines == {"edge"}:
        return Result("tts", WARN, "only the cloud voice (Edge-TTS) works; fights use pre-rendered callouts", data)
    return Result("tts", PASS, f"Korean and English speech ({', '.join(sorted(engines))})", data)


def check_stt(ctx: Ctx) -> Result:
    from jarvis.voice.stt import WhisperStt

    clips = {lang: _speech(ctx, lang) for lang in ("ko", "en") if _speech(ctx, lang) is not None}
    if not clips:
        raise Skip("no synthesized speech to transcribe (the TTS check failed)")
    stt = ctx.make("stt", lambda cfg: WhisperStt.from_config(cfg.voice.stt, cfg.paths().ensure().models), ctx.cfg)
    data: dict[str, Any] = {}
    worst = 100.0
    for lang, pcm in clips.items():
        res = stt.transcribe(pcm)  # no hint: language detection is part of the test
        score = similarity(SENTENCES[lang], res.text)
        worst = min(worst, score if res.lang == lang else min(score, 60.0))  # right words, wrong language label
        data[lang] = {"heard": res.text, "lang": res.lang, "similarity": round(score), "stt_s": round(res.elapsed_s, 2),
                      "rtf": round(res.elapsed_s / max(len(pcm) / SR, 1e-3), 2)}
    if worst >= 70:
        return Result("stt", PASS, "Whisper understood both languages", data)
    if worst >= 45:
        return Result("stt", WARN, "Whisper understood, but not cleanly (accent of the synthetic voice?)", data)
    return Result("stt", FAIL, "Whisper did not understand the test sentences", data)


def check_wake(ctx: Ctx) -> Result:
    from jarvis.voice.wake import FRAME, WakeDetector

    wake = _speech(ctx, "wake")
    if wake is None:
        raise Skip("no synthesized 'Hey Jarvis' (the TTS check failed)")
    make = lambda cfg: WakeDetector.from_config(cfg.voice.wake, cfg.paths().ensure().models)  # noqa: E731

    def run(pcm: np.ndarray) -> tuple[list[float], float, float]:
        det = ctx.make("wake", make, ctx.cfg)
        pad = np.zeros(SR // 2, np.float32)
        x = np.concatenate([pad, pcm, pad, pad])
        peak, fired, t0 = 0.0, [], time.perf_counter()
        for i in range(0, len(x), FRAME):
            fired += det.feed(x[i : i + FRAME])
            peak = max(peak, float(det.last_score))
        per_frame_ms = (time.perf_counter() - t0) * 1000 / max(1, len(x) // FRAME)
        return fired, peak, per_frame_ms

    fired, peak, ms = run(wake)
    data: dict[str, Any] = {"peak_score": round(peak, 3), "fired": bool(fired), "cpu_ms_per_80ms_frame": round(ms, 2)}
    other = _speech(ctx, "ko")
    if other is not None:
        f2, p2, _ = run(other)
        data["false_fire_on_other_speech"] = bool(f2)
        data["other_peak_score"] = round(p2, 3)
        if f2:
            return Result("wake_word", WARN, "the wake word fired on a sentence without it (raise voice.wake.threshold?)", data)
    if not fired:
        return Result("wake_word", WARN, "a synthetic 'Hey Jarvis' did not wake it; try your own voice with --interactive", data)
    return Result("wake_word", PASS, f"'Hey Jarvis' detected (score {peak:.2f}), {ms:.1f} ms CPU per frame", data)


def check_vad(ctx: Ctx) -> Result:
    from jarvis.voice.runtime import build_vad
    from jarvis.voice.vad import EnergyVad

    speech = _speech(ctx, "en") if _speech(ctx, "en") is not None else _speech(ctx, "ko")
    if speech is None:
        raise Skip("no synthesized speech (the TTS check failed)")
    notes: list[str] = []
    vad = ctx.make("vad", build_vad, ctx.cfg, notes)

    def frac(x: np.ndarray) -> float:
        vad.reset()
        probs = [vad.prob(x[i : i + 512]) for i in range(0, len(x) - 511, 512)]
        return float(np.mean([p > 0.5 for p in probs])) if probs else 0.0

    rng = np.random.default_rng(0)
    quiet = (rng.standard_normal(SR) * 0.002).astype(np.float32)
    s, q = frac(speech), frac(quiet)
    data = {"speech_frames_voiced": round(s, 2), "quiet_frames_voiced": round(q, 2), "engine": type(vad).__name__, "notes": notes}
    if s < 0.3 or q > 0.1:
        return Result("vad", FAIL, f"speech detection is off (voiced: speech {s:.0%}, quiet {q:.0%})", data)
    if isinstance(vad, EnergyVad):
        return Result("vad", WARN, "using the simple energy VAD (run `jarvis voice-models` for Silero)", data)
    return Result("vad", PASS, f"Silero VAD: speech {s:.0%} voiced, quiet {q:.0%}", data)


def check_speaker(ctx: Ctx) -> Result:
    from jarvis.voice.runtime import build_gate

    notes: list[str] = []
    gate = ctx.make("gate", build_gate, ctx.cfg, notes)
    if not gate.enabled:
        raise Skip("speaker verification is switched off (voice.speaker.enabled)")
    if not gate.enrolled:
        return Result("speaker_gate", WARN, "your voice is not enrolled: anyone can command JARVIS (run `jarvis enroll`)", {"notes": notes})
    clip = _speech(ctx, "ko") if _speech(ctx, "ko") is not None else _speech(ctx, "en")
    if clip is None:
        raise Skip("no synthesized speech to test with")
    res = gate.verify(clip)
    data = {"score": round(float(res.score), 3), "threshold": gate.threshold, "reason": res.reason, "notes": notes}
    if res.is_owner:
        return Result("speaker_gate", FAIL, "a synthetic voice passed as you: re-run `jarvis enroll` or raise voice.speaker.threshold", data)
    return Result("speaker_gate", PASS, f"a stranger's (synthetic) voice is rejected (score {res.score:.2f} < {gate.threshold})", data)


def check_audio_devices(ctx: Ctx) -> Result:
    try:
        import sounddevice as sd
    except ImportError as e:
        raise Skip("sounddevice is not installed (voice extra)") from e
    data: dict[str, Any] = {}
    try:
        data["input"] = sd.query_devices(kind="input")["name"]
        data["output"] = sd.query_devices(kind="output")["name"]
    except Exception as e:
        return Result("audio_devices", FAIL, f"no default microphone or speaker: {e}", data)
    if sys.platform == "win32":
        try:
            import pyaudiowpatch as pyaudio

            pa = pyaudio.PyAudio()
            try:
                data["loopback"] = pa.get_default_wasapi_loopback()["name"]
            finally:
                pa.terminate()
        except Exception as e:
            data["loopback_error"] = str(e)[:160]
            return Result("audio_devices", WARN, "no WASAPI loopback: echo cancellation hears only JARVIS's own voice, not the game", data)
    return Result("audio_devices", PASS, f"mic: {data['input']} | speakers: {data['output']}", data)


def check_policy(ctx: Ctx) -> Result:
    from jarvis.safety.policy import PolicyChecker, load_policy

    pol = load_policy()
    chk = PolicyChecker(pol)
    cases = {
        "delete C:\\Windows\\System32": chk.check_path_mutation("C:\\Windows\\System32", "delete"),
        "encoded PowerShell": chk.check_shell("powershell -EncodedCommand ZQBjAGgAbwAgAGgAaQA="),
        "'Buy 400 Robux' button": chk.check_ui_element("Buy 400 Robux"),
        "typing a card number": chk.check_typed_text("4111 1111 1111 1111"),
    }
    allowed = [k for k, d in cases.items() if bool(d)]
    data = {"source": pol.source, "protected": pol.protected, "refused": [k for k in cases if k not in allowed]}
    if allowed:
        return Result("safety_policy", FAIL, f"the policy ALLOWED: {', '.join(allowed)}", data)
    if not pol.protected:
        return Result("safety_policy", WARN, "rules hold, but the policy file is not admin-protected (run scripts/install.ps1)", data)
    return Result("safety_policy", PASS, "protected policy refuses deletes of system folders, encoded commands, buy buttons, card numbers", data)


def check_audit(ctx: Ctx) -> Result:
    from jarvis.common.audit import verify_chain

    d = ctx.cfg.paths().audit
    origins = sorted({p.name.rsplit("-", 1)[0] for p in d.glob("*.jsonl")}) if d.exists() else []
    if not origins:
        raise Skip("no audit logs yet")
    data = {}
    broken = []
    for o in origins:
        res = verify_chain(sorted(d.glob(f"{o}-*.jsonl")))
        data[o] = {"ok": res.ok, "records": res.records, "reason": res.reason}
        if not res.ok:
            broken.append(o)
    if broken:
        return Result("audit", FAIL, f"audit chain BROKEN for {', '.join(broken)} (edited or truncated)", data)
    return Result("audit", PASS, f"hash chains intact ({sum(v['records'] for v in data.values())} records)", data)


def check_llm(ctx: Ctx) -> Result:
    from jarvis.core import build_brain

    broker, _agent, memory = ctx.make("brain", build_brain, ctx.cfg)

    async def go():
        return await asyncio.wait_for(
            broker.complete("chat", [{"role": "user", "content": "Reply with the single word OK."}]), 90)

    t0 = time.perf_counter()
    try:
        res = asyncio.run(go())
    except Exception as e:
        return Result("llm", FAIL, f"no language model answered: {type(e).__name__}: {e}"[:300])
    finally:
        with contextlib.suppress(Exception):
            memory.close()
    dt = time.perf_counter() - t0
    data = {"endpoint": res.endpoint, "model": res.model, "seconds": round(dt, 2), "reply": res.text[:80]}
    if "ok" not in res.text.lower():
        return Result("llm", WARN, f"{res.endpoint}:{res.model} answered oddly: {res.text[:60]!r}", data)
    local = res.endpoint == "local"
    return Result("llm", WARN if local else PASS,
                  f"{res.endpoint}:{res.model} answered in {dt:.1f} s" + (" (cloud unavailable: local fallback)" if local else ""), data)


# ------------------------------------------------------------------------------------- checks: Windows
def check_hotkeys(ctx: Ctx) -> Result:
    if supervisor_state(ctx.cfg) is not None:
        raise Skip("JARVIS is running and holds its hotkeys (stop it to test for conflicts)")
    from jarvis.platform.factory import get_platform

    c = ctx.cfg
    combos = list(dict.fromkeys([*c.safety.panic_hotkeys, c.voice.mute.hotkey, c.hud.interactive_hotkey]))
    hk = ctx.make("hotkeys", lambda: get_platform().new_hotkeys())
    for combo in combos:
        hk.register(combo, lambda: None)
    hk.start()
    time.sleep(0.5)  # registration happens on the hotkey thread
    failed = list(getattr(hk, "failed", []))
    hk.stop()
    if failed:
        return Result("hotkeys", FAIL, f"already taken by another app: {', '.join(failed)} (change them in config.yaml)",
                      {"combos": combos, "failed": failed})
    return Result("hotkeys", PASS, f"all free: {', '.join(combos)}", {"combos": combos})


def _ex_style(hwnd: int) -> int:
    import ctypes

    fn = ctypes.windll.user32.GetWindowLongPtrW
    fn.restype = ctypes.c_ssize_t
    fn.argtypes = (ctypes.c_void_p, ctypes.c_int)
    return int(fn(ctypes.c_void_p(hwnd), -20)) & 0xFFFFFFFF  # GWL_EXSTYLE


def _hud_window(pids: set[int]):
    from jarvis.platform.win.window import WinWindowManager

    wins = [w for w in WinWindowManager().find_windows() if w.pid in pids]
    return next((w for w in wins if "JARVIS HUD" in (w.title or "")), wins[0] if wins else None)


def _tree(pid: int) -> set[int]:
    import psutil

    try:
        p = psutil.Process(pid)
        return {pid, *(c.pid for c in p.children(recursive=True))}
    except Exception:
        return {pid}


def check_hud(ctx: Ctx) -> Result:
    from jarvis.platform.win.window import WinWindowManager

    snap = supervisor_state(ctx.cfg)
    hud_row = next((p for p in (snap or {}).get("procs", []) if p.get("name") == "hud"), None)
    if hud_row and hud_row.get("status") == "running" and hud_row.get("pid"):  # inspect the running HUD
        w = _hud_window(_tree(int(hud_row["pid"])))
        flags = hud_flags(_ex_style(w.hwnd)) if w else None
        status, detail = hud_verdict(True, flags, None)
        return Result("hud", status, detail + " (the HUD JARVIS is running)", {"flags": flags})
    return _launch_temp_hud(ctx, WinWindowManager())


def _launch_temp_hud(ctx: Ctx, wm) -> Result:
    import secrets
    import subprocess
    import threading

    from jarvis.common.paths import repo_root
    from jarvis.hud_bridge.hub import HudHub

    hud_dir = repo_root() / ctx.cfg.hud.electron_dir
    if not (hud_dir / "out" / "main" / "main.js").exists():
        return Result("hud", FAIL, f"the HUD is not built: run `npm ci` and `npm run build` in {hud_dir}")
    run = ctx.cfg.paths().ensure().run
    token = secrets.token_urlsafe(24)
    token_file = run / "selftest-hud.token"
    token_file.write_text(token, encoding="utf-8")
    port = free_port()
    h = ctx.cfg.hud
    conf = run / "selftest-hud.json"
    conf.write_text(json.dumps({"host": "127.0.0.1", "port": port, "token_file": str(token_file), "hotkey": h.interactive_hotkey,
                                "content_protection": h.content_protection, "stream_mode": h.stream_mode,
                                "combat_mode": h.combat_mode, "combat_fps_cap": h.combat_fps_cap,
                                "prefer_secondary_monitor": h.prefer_secondary_monitor}), encoding="utf-8")
    hub = HudHub("127.0.0.1", port, token)
    loop = asyncio.new_event_loop()
    started = threading.Event()

    def serve() -> None:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(hub.start())
        started.set()
        loop.run_forever()

    threading.Thread(target=serve, name="selftest-hub", daemon=True).start()
    started.wait(10)
    before = wm.foreground()
    npm = "npm.cmd" if sys.platform == "win32" else "npm"
    proc = subprocess.Popen([npm, "run", "start"], cwd=str(hud_dir), env={**os.environ, "JARVIS_HUD_CONFIG": str(conf)},
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        t_end = time.monotonic() + 40
        while time.monotonic() < t_end and hub.clients == 0:
            time.sleep(0.25)
        connected = hub.clients > 0
        time.sleep(1.0)
        w = _hud_window(_tree(proc.pid)) if connected else None
        flags = hud_flags(_ex_style(w.hwnd)) if w else None
        after = wm.foreground()
        focus_kept = None if before is None else (after is not None and after.hwnd == before.hwnd)
        status, detail = hud_verdict(connected, flags, focus_kept)
        return Result("hud", status, detail, {"flags": flags, "focus_kept": focus_kept})
    finally:
        import psutil

        for pid in sorted(_tree(proc.pid), reverse=True):
            with contextlib.suppress(Exception):
                psutil.Process(pid).kill()
        loop.call_soon_threadsafe(loop.stop)
        for f in (conf, token_file):
            f.unlink(missing_ok=True)


# ------------------------------------------------------------------------------------- checks: with you
def check_mic(ctx: Ctx) -> Result:
    import sounddevice as sd

    from jarvis.voice.resample import resample, rms
    from jarvis.voice.runtime import build_vad
    from jarvis.voice.stt import WhisperStt
    from jarvis.voice.wake import FRAME, WakeDetector, heard_wake_word

    rate = int(sd.query_devices(kind="input")["default_samplerate"])
    ctx.say("Say: \"Hey Jarvis, what time is it?\" (or \"헤이 자비스, 몇 시야?\") - recording 5 seconds...")
    rec = sd.rec(int(5 * rate), samplerate=rate, channels=1, dtype="float32")
    sd.wait()
    pcm = resample(rec.reshape(-1), rate, SR) if rate != SR else rec.reshape(-1)
    data: dict[str, Any] = {"rms": round(rms(pcm), 4), "peak": round(float(np.max(np.abs(pcm))), 3)}
    if data["peak"] < 0.01:
        return Result("mic", FAIL, "the microphone recorded silence (muted, or the wrong default device?)", data)
    vad = build_vad(ctx.cfg, [])
    vad.reset()
    voiced = [vad.prob(pcm[i : i + 512]) > 0.5 for i in range(0, len(pcm) - 511, 512)]
    data["voiced"] = round(float(np.mean(voiced)), 2)
    fired = False
    with contextlib.suppress(Exception):
        det = WakeDetector.from_config(ctx.cfg.voice.wake, ctx.cfg.paths().ensure().models)
        for i in range(0, len(pcm), FRAME):
            fired |= bool(det.feed(pcm[i : i + FRAME]))
    data["wake_fired"] = fired
    stt = WhisperStt.from_config(ctx.cfg.voice.stt, ctx.cfg.paths().ensure().models)
    res = stt.transcribe(pcm)
    data["heard"], data["lang"] = res.text, res.lang
    named = heard_wake_word(res.text, ctx.cfg.voice.wake.second_stage_keywords)
    if fired and named:
        return Result("mic", PASS, f"woke on your voice and heard: {res.text!r}", data)
    if named:
        return Result("mic", WARN, f"heard {res.text!r}, but the wake-word model did not fire (transcript fallback works)", data)
    return Result("mic", WARN, f"heard {res.text!r} - say 'Hey Jarvis' clearly at the start", data)


# ------------------------------------------------------------------------------------- checks: in the game
def check_live_game(ctx: Ctx) -> list[Result]:
    """Roblox window, real capture FPS, HUD reading, and proof that nothing is sent while disarmed."""
    from jarvis.fight.actuator import Command
    from jarvis.fight.gameio import GameIO
    from jarvis.fight.perception.hud_bars import HudReader
    from jarvis.fight.perception.layout import load_layout
    from jarvis.fight.states import FightState
    from jarvis.platform.factory import get_platform

    cfg = ctx.cfg
    paths = cfg.paths().ensure()
    io = ctx.make("gameio", lambda: GameIO(cfg, get_platform()))
    try:
        w = io.attach()
    except Exception as e:
        return [Result("roblox_window", FAIL, f"Roblox window not found: {e} (open UBG first)")]
    out = [Result("roblox_window", PASS, f"{w.title!r} (pid {w.pid}), DPI {io.plat.windows.dpi(w.hwnd)}",
                  {"foreground": io.plat.windows.is_foreground(w.hwnd)})]
    layout = load_layout(paths.calibration)
    reader = HudReader(layout, calibration_dir=paths.calibration) if layout else None
    io.start()
    try:
        assert io.sm.state == FightState.DISARMED
        frames, lat, confs, combat = 0, [], [], 0
        last = 0
        t_end = time.monotonic() + 6.0
        sent_before = io.actuator.stats["sent"]
        io.actuator.submit([Command("light", reason="selftest: must be refused while disarmed")])
        while time.monotonic() < t_end:
            f = io.frames.get_newer(last, 0.5)
            if f is None:
                continue
            last = f.index
            frames += 1
            if f.source_t_ns and 0 <= (f.t_ns - f.source_t_ns) / 1e6 < 1000:
                lat.append((f.t_ns - f.source_t_ns) / 1e6)
            if reader is not None and frames % 3 == 0:
                rd = reader.read(f.image, f.t_ns)
                confs.append(rd.hud_conf)
                combat += bool(rd.combat_hud)
        fps = frames / 6.0
        st = WARN if 30 <= fps < 55 else (FAIL if fps < 30 else PASS)
        out.append(Result("capture", st, f"{fps:.0f} fps from Windows Graphics Capture"
                          + (f", {np.median(lat):.1f} ms capture latency" if lat else ""),
                          {"fps": round(fps, 1), "latency_ms_median": round(float(np.median(lat)), 2) if lat else None}))
        sent = io.actuator.stats["sent"] - sent_before
        out.append(Result("no_input_when_disarmed", PASS if sent == 0 else FAIL,
                          "a fight key submitted while disarmed was refused; nothing reached Roblox" if sent == 0 else
                          f"{sent} input(s) were SENT while disarmed - report this", dict(io.actuator.stats)))
        if reader is None:
            out.append(Result("hud_reader", WARN, "HUD not calibrated yet: run `jarvis calibrate hud` during a match"))
        else:
            share = combat / max(1, len(confs))
            data = {"combat_hud_share": round(share, 2), "hud_conf_median": round(float(np.median(confs)), 2) if confs else None}
            if not confs:
                out.append(Result("hud_reader", FAIL, "no frames to read", data))
            elif share == 0:
                out.append(Result("hud_reader", WARN, "no combat HUD seen (fine in the lobby; re-run during a match)", data))
            else:
                out.append(Result("hud_reader", PASS if data["hud_conf_median"] >= cfg.fight.perception.hud_conf_min else WARN,
                                  f"combat HUD read in {share:.0%} of frames, confidence {data['hud_conf_median']}", data))
        problems = io_problems(io)
        out.append(Result("fight_preflight", WARN if problems else PASS,
                          "; ".join(problems) or "a fight could be armed here (private server permitting)", {"problems": problems}))
    finally:
        io.stop()
    return out


def io_problems(io) -> list[str]:
    from jarvis.fight.loop import FightLoop

    try:
        return FightLoop(io.cfg, io, record=False).preflight()
    except Exception as e:
        return [f"preflight failed: {e}"]


CHECKS: tuple[Check, ...] = (
    Check("doctor", check_doctor, "This PC (GPU, Roblox, audio, models, API key, policy)"),
    Check("ipc", check_ipc, "Process messaging (event bus + RPC)"),
    Check("gpu_runtime", check_gpu_runtime, "CUDA for ONNX Runtime and Whisper"),
    Check("tts", check_tts, "Voice: Korean and English speech"),
    Check("stt", check_stt, "Speech recognition round trip"),
    Check("wake_word", check_wake, "Wake word"),
    Check("vad", check_vad, "Voice activity detection"),
    Check("speaker_gate", check_speaker, "Owner-only commands"),
    Check("audio_devices", check_audio_devices, "Microphone, speakers, loopback"),
    Check("safety_policy", check_policy, "Safety policy refusals"),
    Check("audit", check_audit, "Tamper-evident audit log"),
    Check("llm", check_llm, "Language model answers"),
    Check("hotkeys", check_hotkeys, "Global hotkeys are free", windows_only=True),
    Check("hud", check_hud, "Holographic HUD window", windows_only=True),
    Check("mic", check_mic, "Your voice through the microphone", interactive=True),
    Check("game", check_live_game, "Roblox / UBG (window, capture, HUD, no input when disarmed)", windows_only=True, live=True),
)


def run_selftest(ctx: Ctx, only: set[str] | None = None, on_result: Callable[[Result], None] | None = None,
                 checks: tuple[Check, ...] = CHECKS) -> list[Result]:
    results: list[Result] = []

    def emit(r: Result) -> None:
        results.append(r)
        if on_result:
            on_result(r)

    for c in checks:
        if only and c.name not in only:
            continue
        if c.windows_only and not ctx.windows:
            emit(Result(c.name, SKIP, "Windows only"))
            continue
        if c.live and not ctx.live:
            emit(Result(c.name, SKIP, "needs Roblox open on UBG: add --live"))
            continue
        if c.interactive and not ctx.interactive:
            emit(Result(c.name, SKIP, "needs you at the mic: add --interactive"))
            continue
        t0 = time.perf_counter()
        try:
            out = c.fn(ctx)
        except Skip as e:
            out = Result(c.name, SKIP, str(e))
        except Exception as e:
            out = Result(c.name, FAIL, f"{type(e).__name__}: {e}"[:400], {"traceback": traceback.format_exc()[-2000:]})
        for r in out if isinstance(out, list) else [out]:
            r.seconds = round(time.perf_counter() - t0, 2)
            emit(r)
    return results


def summary(results: list[Result]) -> dict[str, int]:
    return {s: sum(r.status == s for r in results) for s in (PASS, WARN, FAIL, SKIP)}


def save_report(results: list[Result], ctx: Ctx) -> Path:
    from jarvis import __version__

    d = ctx.cfg.paths().ensure().reports
    p = d / f"selftest-{time.strftime('%Y%m%d-%H%M%S')}.json"
    p.write_text(json.dumps({"version": __version__, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                             "platform": platform.platform(), "python": sys.version.split()[0], "tier": ctx.cfg.tier,
                             "live": ctx.live, "interactive": ctx.interactive, "summary": summary(results),
                             "results": [asdict(r) for r in results]}, ensure_ascii=False, indent=1, default=str),
                 encoding="utf-8")
    return p
