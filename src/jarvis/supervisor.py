"""supervisor: starts and watches every JARVIS process; owns the event bus, the tray icon and a second panic
hotkey.

Started at logon by Task Scheduler (``pythonw -m jarvis.supervisor``) with normal user rights - not as a
Windows service, because services cannot see the desktop, audio devices or input. It:

* runs the event bus, so the bus survives restarts of any other process;
* starts core, toolhost, fightd, learnerd and the HUD, and restarts a process that exits with backoff
  (a crash loop is reported on the tray/HUD and retried slowly - never hidden);
* if fightd ever dies on its own, sends key-up for every key and mouse button the keymap could hold
  (the third stuck-key layer, after fightd's own release-on-exit and the actuator heartbeat);
* lends the GPU to one side at a time: while a fight is on, any learning job still alive after a short
  grace period is ended (learnerd normally ends it itself the moment the fight starts);
* panic: its own hotkey (separate from fightd's) and the tray broadcast ``kill`` on the bus, and a fightd
  that no longer answers is killed and its keys released.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jarvis.common.config import JarvisConfig
from jarvis.common.ipc import (
    AsyncSubscriber,
    BusProxy,
    Publisher,
    RpcClient,
    RpcError,
    RpcServer,
    load_or_create_token,
)

log = logging.getLogger(__name__)

ACTIVE_FIGHT = {"armed", "navigating", "queued", "fighting", "post_match", "calibrating"}
STOP_GRACE_S = 6.0
GPU_GRACE_S = 1.5


# ------------------------------------------------------------------------------------- restart policy
@dataclass
class Backoff:
    base_s: float = 1.0
    max_s: float = 60.0
    healthy_after_s: float = 60.0  # a run at least this long resets the backoff
    window_s: float = 300.0
    max_crashes: int = 5  # this many exits inside the window = crash loop
    loop_retry_s: float = 300.0  # a crash-looping process is retried this rarely


class RestartPolicy:
    """When to start a process again after it exited (pure, tested with a fake clock)."""

    def __init__(self, b: Backoff | None = None) -> None:
        self.b = b or Backoff()
        self.exits: deque[float] = deque()
        self.delay = 0.0

    def on_exit(self, started_at: float, now: float) -> float:
        if now - started_at >= self.b.healthy_after_s:
            self.delay = 0.0
            self.exits.clear()
        self.exits.append(now)
        while self.exits and now - self.exits[0] > self.b.window_s:
            self.exits.popleft()
        if self.crash_looping():
            return self.b.loop_retry_s
        self.delay = self.b.base_s if self.delay == 0.0 else min(self.b.max_s, self.delay * 2)
        return self.delay

    def crash_looping(self) -> bool:
        return len(self.exits) >= self.b.max_crashes


# ------------------------------------------------------------------------------------- children
@dataclass
class ProcSpec:
    name: str
    argv: list[str]
    enabled: bool = True
    gui: bool = False  # the HUD: may open windows (still no console window)


@dataclass
class Child:
    spec: ProcSpec
    policy: RestartPolicy = field(default_factory=RestartPolicy)
    proc: Any = None  # subprocess.Popen
    job: Any = None  # Windows Job object (kills the whole tree with the supervisor)
    status: str = "pending"  # pending | running | backoff | crash_loop | stopped | disabled
    started_at: float = 0.0
    next_start: float = 0.0
    restarts: int = 0
    last_exit: int | None = None
    stopping: bool = False
    reported_loop: bool = False

    @property
    def pid(self) -> int | None:
        return None if self.proc is None else self.proc.pid

    def row(self) -> dict:
        return {"name": self.spec.name, "status": self.status, "pid": self.pid, "restarts": self.restarts,
                "last_exit": self.last_exit, "up_s": round(time.monotonic() - self.started_at, 1) if self.status == "running" else 0}


def default_specs(cfg: JarvisConfig, python: str | None = None, *, voice: bool = True) -> list[ProcSpec]:
    py = python or sys.executable
    j = [py, "-m", "jarvis"]
    return [
        ProcSpec("toolhost", [*j, "toolhost"]),
        ProcSpec("fightd", [*j, "fightd"]),
        ProcSpec("core", [*j, "core"] + ([] if voice else ["--no-voice"])),
        ProcSpec("learnerd", [*j, "learnerd"], enabled=cfg.learn.enabled),
        ProcSpec("hud", [*j, "hud"], enabled=cfg.hud.enabled, gui=True),
    ]


def _descendants(proc) -> list:
    try:
        return proc.children(recursive=True)
    except Exception:
        return []


def _rotate(p: Path, max_bytes: int = 5 * 1024 * 1024) -> None:
    with contextlib.suppress(OSError):
        if p.exists() and p.stat().st_size > max_bytes:
            os.replace(p, p.with_suffix(p.suffix + ".1"))


# ------------------------------------------------------------------------------------- supervisor
class Supervisor:
    def __init__(
        self,
        cfg: JarvisConfig,
        specs: list[ProcSpec] | None = None,
        *,
        backoff: Backoff | None = None,
        clock: Callable[[], float] = time.monotonic,
        input_sink_factory: Callable[[], Any] | None = None,
        serve: bool = True,
        tray: bool = True,
        hotkeys: bool = True,
    ) -> None:
        self.cfg = cfg
        self.paths = cfg.paths().ensure()
        self.clock = clock
        self.children = {s.name: Child(s, RestartPolicy(backoff)) for s in (specs if specs is not None else default_specs(cfg))}
        self.input_sink_factory = input_sink_factory
        self.fight_state = "detached"
        self.fight_since = 0.0
        self.shutting_down = False
        self.notices: deque[str] = deque(maxlen=20)
        self.keys_released: list[str] = []
        self._want_tray = tray
        self._want_hotkeys = hotkeys
        self._tray = None
        self._hotkeys = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_event: asyncio.Event | None = None
        self.token = load_or_create_token(self.paths.run)
        self.pub: Publisher | None = None
        self.server: RpcServer | None = None
        self.bus: BusProxy | None = None
        if serve:
            self.server = RpcServer("supervisor", cfg.ipc.rpc_ports["supervisor"], self.token, cfg.ipc.host)
            for name in ("status", "restart", "shutdown", "panic"):
                self.server.add(name, getattr(self, f"rpc_{name}"))

    # ------------------------------------------------------------------ process control
    def _env(self) -> dict[str, str]:
        return {**os.environ, "JARVIS_SUPERVISED": "1", "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}

    def start_child(self, c: Child) -> None:
        c.started_at = self.clock()  # a failed start counts as an immediate exit for the backoff
        log_path = self.paths.logs / f"{c.spec.name}.out.log"
        _rotate(log_path)
        out = log_path.open("ab")
        kw: dict[str, Any] = {"stdin": subprocess.DEVNULL, "stdout": out, "stderr": subprocess.STDOUT, "env": self._env(),
                              "cwd": str(self.paths.root)}
        try:
            if sys.platform == "win32":
                kw["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
                c.proc = subprocess.Popen(c.spec.argv, **kw)
                with contextlib.suppress(Exception):
                    from jarvis.platform.win.job import JobObject

                    c.job = JobObject()
                    c.job.assign(c.proc)
            else:
                c.proc = subprocess.Popen(c.spec.argv, start_new_session=True, **kw)
        except OSError as e:
            log.error("could not start %s: %s", c.spec.name, e)
            c.proc = None
            self._after_exit(c, -1, self.clock())
            return
        finally:
            out.close()  # the child holds its own handle
        c.status = "running"
        c.stopping = False
        log.info("started %s (pid %s)", c.spec.name, c.proc.pid)

    def _kill_tree(self, c: Child, *, force: bool) -> None:
        p = c.proc
        if p is None:
            return
        if c.job is not None and force:
            with contextlib.suppress(Exception):
                c.job.terminate()
            return
        try:
            import psutil

            parent = psutil.Process(p.pid)
            procs = [*parent.children(recursive=True), parent]
        except Exception:
            procs = []
        for q in procs:
            with contextlib.suppress(Exception):
                q.kill() if force else q.terminate()
        if not procs:
            with contextlib.suppress(Exception):
                p.kill() if force else p.terminate()

    def stop_child(self, c: Child, grace_s: float = STOP_GRACE_S) -> None:
        p = c.proc
        if p is None:
            c.status = "stopped"
            return
        c.stopping = True
        try:
            self._kill_tree(c, force=False)
            try:
                p.wait(timeout=grace_s)
            except subprocess.TimeoutExpired:
                self._kill_tree(c, force=True)
                with contextlib.suppress(subprocess.TimeoutExpired):
                    p.wait(timeout=3)
            c.last_exit = p.returncode
            self._close_job(c)
            c.proc = None
            c.status = "stopped"
        finally:
            c.stopping = False

    def _close_job(self, c: Child) -> None:
        if c.job is not None:
            with contextlib.suppress(Exception):
                c.job.close()
            c.job = None

    def _after_exit(self, c: Child, code: int | None, now: float) -> None:
        c.last_exit = code
        c.proc = None
        self._close_job(c)
        if self.shutting_down or c.stopping:
            c.status = "stopped"
            return
        c.restarts += 1
        delay = c.policy.on_exit(c.started_at, now)
        c.next_start = now + delay
        if c.policy.crash_looping():
            c.status = "crash_loop"
            if not c.reported_loop:
                c.reported_loop = True
                self.notice(f"{c.spec.name} keeps crashing (exit {code}); retrying every {int(delay)} s. "
                            f"See logs/{c.spec.name}.out.log")
        else:
            c.status = "backoff"
            c.reported_loop = False
        log.warning("%s exited with %s; restart in %.0f s", c.spec.name, code, delay)

    def poll(self) -> None:
        """One supervision step: notice exits, start what is due. Called twice a second."""
        now = self.clock()
        for c in self.children.values():
            if not c.spec.enabled:
                c.status = "disabled"
                continue
            if c.stopping:  # stop_child (another thread) owns it right now
                continue
            if c.proc is not None:
                code = c.proc.poll()
                if code is not None:
                    if c.spec.name == "fightd" and not (c.stopping or self.shutting_down):
                        self.release_game_keys(f"fightd exited unexpectedly ({code})")
                    self._after_exit(c, code, now)
            elif not self.shutting_down and c.status in ("pending", "backoff", "crash_loop") and now >= c.next_start:
                self.start_child(c)
        self.lend_gpu(now)

    # ------------------------------------------------------------------ safety layers
    def release_game_keys(self, reason: str) -> list[str]:
        """Key-up for every key/button the keymap could hold. Releasing is always safe."""
        from jarvis.fight.keymap import load_keymap
        from jarvis.platform.base import MouseButton

        try:
            km = load_keymap(self.paths.calibration)
            keys, buttons = sorted(km.primitive_keys(False)), sorted(km.mouse_buttons(False))
        except Exception:
            log.exception("keymap unreadable; releasing the default fight keys")
            keys, buttons = ["w", "a", "s", "d", "space", "f", "q", "e", "r", "lshift"], ["left", "right"]
        try:
            factory = self.input_sink_factory
            if factory is None:
                from jarvis.platform.factory import get_platform

                factory = get_platform().new_input_sink
            sink = factory()
        except Exception as e:
            log.error("cannot release keys (%s): %s", reason, e)
            return []
        released = []
        for k in keys:
            with contextlib.suppress(Exception):
                sink.key_up(k)
                released.append(k)
        for b in buttons:
            with contextlib.suppress(Exception):
                sink.mouse_up(MouseButton(b))
                released.append(f"mouse:{b}")
        log.warning("released %d game inputs: %s", len(released), reason)
        self.keys_released = released
        return released

    def on_fight_state(self, new: str) -> None:
        active = new in ACTIVE_FIGHT
        if active and self.fight_state not in ACTIVE_FIGHT:
            self.fight_since = self.clock()
        self.fight_state = new

    def lend_gpu(self, now: float) -> int:
        """While a fight is on, no learning job may keep running (learnerd stops its own; this is the backstop)."""
        c = self.children.get("learnerd")
        if self.fight_state not in ACTIVE_FIGHT or c is None or c.proc is None or now - self.fight_since < GPU_GRACE_S:
            return 0
        try:
            import psutil

            kids = psutil.Process(c.proc.pid).children(recursive=True)
        except Exception:
            return 0
        # Only the job processes: on Windows a venv's python.exe is a launcher whose child is learnerd itself.
        jobs = []
        for k in kids:
            with contextlib.suppress(Exception):
                if "jarvis.learn.jobs" in " ".join(k.cmdline()):
                    jobs.append(k)
        victims = {k.pid: k for j in jobs for k in [j, *_descendants(j)]}
        for k in victims.values():
            with contextlib.suppress(Exception):
                k.kill()
        if jobs:
            log.warning("fight active: ended %d learning job(s) learnerd left running", len(jobs))
        return len(jobs)

    def recover_after_crash(self) -> bool:
        """At start: if the previous supervisor died with fightd running, release every game key first."""
        try:
            prev = json.loads((self.paths.run / "supervisor.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        if any(p.get("name") == "fightd" and p.get("status") == "running" for p in prev.get("procs", [])):
            self.release_game_keys("the previous JARVIS run ended without a clean shutdown")
            return True
        return False

    def publish(self, topic: str, payload: dict) -> None:
        """Thread-safe: hotkey and tray callbacks run on their own threads; the socket lives on the loop's."""
        if self.pub is None or self._loop is None:
            return
        self._loop.call_soon_threadsafe(self.pub.publish, topic, payload)

    def panic(self, reason: str) -> None:
        """Anyone may stop everything. Broadcast kill; if fightd does not answer, kill it and release keys."""
        log.warning("PANIC: %s", reason)
        self.publish("kill", {"reason": reason, "t_ns": time.perf_counter_ns(), "by": "supervisor"})
        if self._loop is not None:
            asyncio.run_coroutine_threadsafe(self._check_fightd_after_panic(reason), self._loop)

    async def _check_fightd_after_panic(self, reason: str) -> None:
        c = self.children.get("fightd")
        if c is None or c.proc is None:
            return
        client = RpcClient(self.cfg.ipc.rpc_ports["fightd"], self.token, self.cfg.ipc.host, timeout=1.5)
        try:
            await client.call("panic", {"reason": reason})
        except RpcError:
            # Hung or gone: kill it. poll() then sees the exit, releases every game key and restarts it
            # (a fresh fightd starts disarmed).
            log.error("fightd did not answer the panic: killing it")
            self._kill_tree(c, force=True)
        finally:
            await client.close()

    # ------------------------------------------------------------------ notices / status
    def notice(self, text: str) -> None:
        self.notices.append(text)
        log.warning("notice: %s", text)
        self.publish("sys.notice", {"text": text})
        if self._tray is not None:
            with contextlib.suppress(Exception):
                self._tray.notify(text)

    def snapshot(self) -> dict:
        return {"pid": os.getpid(), "fight_state": self.fight_state, "procs": [c.row() for c in self.children.values()],
                "notices": list(self.notices), "t": time.time()}

    def write_state(self) -> None:
        with contextlib.suppress(OSError):
            p = self.paths.run / "supervisor.json"
            tmp = p.with_name(p.name + ".tmp")
            tmp.write_text(json.dumps(self.snapshot()), encoding="utf-8")
            os.replace(tmp, p)

    def health(self) -> str:
        rows = [c for c in self.children.values() if c.spec.enabled]
        if any(c.status == "crash_loop" for c in rows):
            return "degraded"
        if all(c.status == "running" for c in rows):
            return "ok"
        return "starting"

    # ------------------------------------------------------------------ RPC
    def rpc_status(self, _p: dict) -> dict:
        return {**self.snapshot(), "health": self.health()}

    async def rpc_restart(self, p: dict) -> dict:
        names = list(self.children) if p.get("name") in (None, "", "all") else [str(p["name"])]
        unknown = [n for n in names if n not in self.children]
        if unknown:
            return {"ok": False, "error": f"unknown process {unknown[0]!r}"}
        for n in names:
            c = self.children[n]
            if n == "fightd" and c.proc is not None:
                await self._fightd_disarm("restart")
            await asyncio.to_thread(self.stop_child, c)
            if n == "fightd":
                self.release_game_keys("fightd restarted")
            c.policy = RestartPolicy(c.policy.b)
            c.status, c.next_start, c.reported_loop = "pending", 0.0, False
        return {"ok": True, "restarted": names}

    async def _fightd_disarm(self, why: str) -> None:
        client = RpcClient(self.cfg.ipc.rpc_ports["fightd"], self.token, self.cfg.ipc.host, timeout=2.0)
        try:
            await client.call("disarm", {"reason": f"JARVIS {why}"})
        except RpcError:
            pass
        finally:
            await client.close()

    def rpc_shutdown(self, _p: dict) -> dict:
        if self._stop_event is not None and self._loop is not None:
            self._loop.call_soon_threadsafe(self._stop_event.set)
        return {"ok": True}

    def rpc_panic(self, p: dict) -> dict:
        self.panic(str(p.get("reason", "panic (rpc)")))
        return {"ok": True}

    # ------------------------------------------------------------------ desktop integration
    def hotkey_bindings(self) -> dict[str, Callable[[], None]]:
        combos = self.cfg.safety.panic_hotkeys[1:]  # fightd registers the first one itself: two independent paths
        out: dict[str, Callable[[], None]] = {c: (lambda c=c: self.panic(f"panic hotkey {c}")) for c in combos}
        mute = self.cfg.voice.mute.hotkey
        if mute and mute not in out:
            out[mute] = lambda: self._core_call("toggle_mute")
        return out

    def _start_hotkeys(self) -> None:
        bindings = self.hotkey_bindings()
        if not bindings:
            return
        try:
            from jarvis.platform.factory import get_platform

            hk = get_platform().new_hotkeys()
            for combo, fn in bindings.items():
                hk.register(combo, fn)
            hk.start()
            self._hotkeys = hk
        except Exception as e:
            log.warning("global hotkeys unavailable here: %s", e)

    def _core_call(self, method: str, params: dict | None = None) -> None:
        async def go() -> None:
            client = RpcClient(self.cfg.ipc.rpc_ports["core"], self.token, self.cfg.ipc.host, timeout=3.0)
            try:
                await client.call(method, params or {})
            except RpcError as e:
                self.notice(f"core did not answer ({method}): {e}")
            finally:
                await client.close()

        if self._loop is not None:
            asyncio.run_coroutine_threadsafe(go(), self._loop)

    def _fightd_call(self, method: str, params: dict | None = None) -> None:
        async def go() -> None:
            client = RpcClient(self.cfg.ipc.rpc_ports["fightd"], self.token, self.cfg.ipc.host, timeout=3.0)
            try:
                await client.call(method, params or {})
            except RpcError as e:
                self.notice(f"fightd did not answer ({method}): {e}")
            finally:
                await client.close()

        if self._loop is not None:
            asyncio.run_coroutine_threadsafe(go(), self._loop)

    def _start_tray(self) -> None:
        try:
            from jarvis.tray import Tray
        except ImportError as e:
            log.info("tray icon unavailable: %s", e)
            return
        actions = {
            "mute": lambda: self._core_call("mute", {"soft": False}),
            "unmute": lambda: self._core_call("unmute"),
            "disarm": lambda: self._fightd_call("disarm", {"reason": "owner (tray)"}),
            "panic": lambda: self.panic("tray"),
            "restart": lambda: self._loop and asyncio.run_coroutine_threadsafe(self.rpc_restart({}), self._loop),
            "logs": lambda: _open_folder(self.paths.logs),
            "quit": lambda: self.rpc_shutdown({}),
        }
        try:
            self._tray = Tray(actions, status=lambda: (self.health(), self.fight_state))
            self._tray.start()
        except Exception as e:
            log.warning("tray icon could not start: %s", e)
            self._tray = None

    # ------------------------------------------------------------------ main
    async def _bus_listener(self) -> None:
        sub = AsyncSubscriber(self.cfg.ipc.bus_out_port, ["fight.state"], self.cfg.ipc.host)
        try:
            while True:
                _topic, msg = await sub.recv()
                if isinstance(msg, dict):
                    self.on_fight_state(str(msg.get("new", "")))
        finally:
            sub.close()

    async def run(self) -> None:
        from jarvis.core import bus_port_free

        self._loop = asyncio.get_running_loop()
        self._stop_event = asyncio.Event()
        ipc = self.cfg.ipc
        if not bus_port_free(ipc.host, ipc.bus_in_port):
            raise RuntimeError(f"event bus port {ipc.bus_in_port} is in use - is JARVIS already running (`jarvis status`)?")
        assert self.server is not None
        await self.server.start()  # single instance: a second supervisor fails to bind here
        self.recover_after_crash()
        self.bus = BusProxy(ipc.bus_in_port, ipc.bus_out_port, ipc.host)
        self.bus.start()
        self.pub = Publisher(ipc.bus_in_port, ipc.host)
        listener = asyncio.create_task(self._bus_listener())
        if self._want_hotkeys:
            self._start_hotkeys()
        if self._want_tray:
            self._start_tray()
        log.info("supervisor ready: %s", ", ".join(n for n, c in self.children.items() if c.spec.enabled))
        try:
            while not self._stop_event.is_set():
                try:
                    self.poll()
                    self.write_state()
                except Exception:  # the supervisor must outlive any single failure
                    log.exception("supervision step failed")
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self._stop_event.wait(), 0.5)
        finally:
            await self.shutdown()
            listener.cancel()

    async def shutdown(self) -> None:
        if self.shutting_down:
            return
        self.shutting_down = True
        log.info("supervisor shutting down")
        fd = self.children.get("fightd")
        if fd is not None and fd.proc is not None:  # disarm first, so no key is ever left held
            await self._fightd_disarm("quitting")
        for name in ("hud", "core", "learnerd", "toolhost", "fightd"):
            c = self.children.get(name)
            if c is not None:
                await asyncio.to_thread(self.stop_child, c)
        for c in self.children.values():  # anything not in the fixed order
            if c.proc is not None:
                await asyncio.to_thread(self.stop_child, c)
        if fd is not None:
            self.release_game_keys("JARVIS quitting")
        for comp in (self._hotkeys, self._tray):
            if comp is not None:
                with contextlib.suppress(Exception):
                    comp.stop()
        if self.server is not None:
            await self.server.close()
        if self.pub is not None:
            self.pub.close()
        if self.bus is not None:
            self.bus.stop()
        self.write_state()


def _open_folder(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        os.startfile(str(p))  # type: ignore[attr-defined]
    else:
        subprocess.Popen(["xdg-open", str(p)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main(cfg: JarvisConfig | None = None, *, voice: bool = True) -> int:
    from jarvis.common.config import load_config
    from jarvis.common.logging_setup import setup_logging

    cfg = cfg or load_config()
    setup_logging("supervisor", cfg.paths().ensure().logs, cfg.log_level, console=sys.stderr is not None and sys.stderr.isatty())
    sup = Supervisor(cfg, default_specs(cfg, voice=voice))
    try:
        asyncio.run(sup.run())
    except RuntimeError as e:
        log.error("%s", e)
        return 1
    except Exception as e:  # zmq "address in use" = another supervisor owns the port
        if "Address already in use" in str(e) or "address in use" in str(e).lower():
            log.error("another JARVIS supervisor is already running")
            return 0
        raise
    return 0


if __name__ == "__main__":
    threading.current_thread().name = "supervisor"
    sys.exit(main(voice="--no-voice" not in sys.argv[1:]))
