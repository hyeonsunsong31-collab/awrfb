"""learnerd: the background self-education process.

It never touches the PC or the game: it has no tool backend, and its only LLM use (knowledge-base refresh)
goes through the broker without tools. Jobs run as child process trees; a fight starting (``fight.state``),
fightd claiming the GPU (``gpu.claim``) or the kill switch (``kill``) ends the running job at once. Results
go out on the bus (``learn.*``) so core can show them on the HUD and say when the HUD must be recalibrated
after a game update.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import sys
import time

import psutil

from jarvis.common.config import JarvisConfig
from jarvis.common.ipc import AsyncSubscriber, Publisher, RpcServer, load_or_create_token
from jarvis.learn.jobs import JOBS, list_sessions
from jarvis.learn.scheduler import LearnState, Status, next_job
from jarvis.tools.procs import ProcessRunner

log = logging.getLogger(__name__)

ACTIVE_FIGHT = {"armed", "navigating", "queued", "fighting", "post_match", "calibrating"}
JOB_TIMEOUT_S = {"post_fight": 600, "nightly": 6 * 3600, "video": 3 * 3600, "kb": 1800, "consolidate": 600}
TICK_S = 15.0


def user_idle_s() -> float | None:
    """Seconds since the owner last touched keyboard or mouse (Windows), else None."""
    if sys.platform != "win32":
        return None
    import ctypes

    class LASTINPUTINFO(ctypes.Structure):
        _fields_ = (("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint))

    li = LASTINPUTINFO(ctypes.sizeof(LASTINPUTINFO), 0)
    if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(li)):
        return None
    tick = ctypes.windll.kernel32.GetTickCount() & 0xFFFFFFFF
    return ((tick - li.dwTime) & 0xFFFFFFFF) / 1000.0  # both wrap every 49.7 days


class Learnerd:
    def __init__(self, cfg: JarvisConfig, *, runner: ProcessRunner | None = None, serve: bool = True,
                 python: str = sys.executable) -> None:
        self.cfg = cfg
        self.paths = cfg.paths().ensure()
        self.state_file = self.paths.run / "learnerd_state.json"
        self.state = self._load()
        self.runner = runner or ProcessRunner()
        self.python = python
        self.fight_active = False
        self.fight_ended = time.time()
        self.running: str | None = None
        self.last_result: dict | None = None
        self._stop_reason = ""
        self._job_task: asyncio.Task | None = None
        self.pub = None
        self.server = None
        if serve:
            self.pub = Publisher(cfg.ipc.bus_in_port, cfg.ipc.host)
            self.server = RpcServer("learnerd", cfg.ipc.rpc_ports["learnerd"], load_or_create_token(self.paths.run), cfg.ipc.host)
            for name in ("status", "request", "cancel"):
                self.server.add(name, getattr(self, f"rpc_{name}"))

    # ------------------------------------------------------------------ state
    def _load(self) -> LearnState:
        try:
            return LearnState.from_dict(json.loads(self.state_file.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError):
            return LearnState()

    def _save(self) -> None:
        tmp = self.state_file.with_name(self.state_file.name + ".tmp")
        tmp.write_text(json.dumps(self.state.to_dict()), encoding="utf-8")
        tmp.replace(self.state_file)

    def publish(self, topic: str, payload: dict) -> None:
        if self.pub is not None:
            self.pub.publish(topic, payload)

    def status(self) -> Status:
        g = None
        with contextlib.suppress(Exception):
            from jarvis.common.gpu import primary_gpu

            gi = primary_gpu()
            g = gi.util_pct if gi else None
        idle = time.time() - self.fight_ended
        owner = user_idle_s()
        if owner is not None:
            idle = min(idle, owner)
        return Status(fight_active=self.fight_active, idle_s=idle, cpu_pct=psutil.cpu_percent(None), gpu_util_pct=g,
                      sessions=[s.name for s in list_sessions(self.paths.recordings)], running=self.running)

    def videos_left_today(self) -> int:
        today = time.strftime("%Y-%m-%d")
        used = self.state.videos_today if self.state.videos_day == today else 0
        return max(0, self.cfg.learn.video_ingest.max_videos_per_day - used)

    # ------------------------------------------------------------------ fights always win
    def on_fight_state(self, new: str) -> None:
        was = self.fight_active
        self.fight_active = new in ACTIVE_FIGHT
        if self.fight_active:
            self.stop_job(f"fight {new}")
        if was and not self.fight_active:
            self.fight_ended = time.time()

    def stop_job(self, why: str) -> int:
        if self.running is None:
            return 0
        log.info("stopping %s: %s", self.running, why)
        self._stop_reason = why
        return self.runner.kill_all()

    # ------------------------------------------------------------------ jobs
    async def run_job(self, job: str) -> dict:
        self.running = job
        self._stop_reason = ""
        self.publish("learn.status", {"running": job})
        out = self.paths.run / f"learn_{job}.json"
        out.unlink(missing_ok=True)
        argv = [self.python, "-m", "jarvis.learn.jobs", job, "--budget", str(self.videos_left_today()), "--out", str(out)]
        try:
            r = await self.runner.run(argv, timeout_s=JOB_TIMEOUT_S.get(job, 3600))
            if r.killed:
                res = {"job": job, "ok": False, "error": f"stopped ({self._stop_reason or 'killed'})"}
            elif r.timed_out:
                res = {"job": job, "ok": False, "error": "timed out"}
            else:
                res = self._read_result(job, out, r.out, r.err)
        except Exception as e:  # could not even start the child
            log.exception("job %s could not run", job)
            res = {"job": job, "ok": False, "error": f"{type(e).__name__}: {e}"}
        finally:
            self.running = None
            self.publish("learn.status", {"running": None})
        self.absorb(job, res)
        self.last_result = res
        self.publish("learn.result", res)
        if res.get("drift"):
            self.publish("learn.drift", {"reason": res.get("drift_reason", "")})
        log.info("job %s -> %s", job, {k: v for k, v in res.items() if k not in ("processed", "labels")})
        return res

    @staticmethod
    def _read_result(job: str, out, stdout: str, stderr: str) -> dict:
        try:
            return json.loads(out.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        for line in reversed(stdout.strip().splitlines()):
            if line.startswith("{"):
                with contextlib.suppress(ValueError):
                    return json.loads(line)
        return {"job": job, "ok": False, "error": (stderr.strip()[-300:] or "no result")}

    def absorb(self, job: str, res: dict) -> None:
        now = time.time()
        st = self.state
        if not res.get("ok"):
            self._save()
            return
        if job in st.requested:
            st.requested.remove(job)
        if job == "post_fight":
            st.processed_sessions = list(res.get("processed", st.processed_sessions))
        elif job == "nightly":
            st.last_nightly = now
            if res.get("trained"):
                st.sessions_at_last_train = int(res.get("sessions", st.sessions_at_last_train))
        elif job == "kb":
            st.last_kb = now
        elif job == "consolidate":
            st.last_consolidate = now
        elif job == "video":
            today = time.strftime("%Y-%m-%d")
            st.videos_today = (st.videos_today if st.videos_day == today else 0) + int(res.get("tried", 0))
            st.videos_day = today
            if not res.get("tried"):  # nothing to do (no detector, no layout, disk full): don't retry all day
                st.videos_today = max(st.videos_today, self.cfg.learn.video_ingest.max_videos_per_day)
        self._save()

    async def tick(self) -> str | None:
        if self._job_task is not None and not self._job_task.done():
            return None
        job = next_job(self.state, self.status(), self.cfg.learn)
        if job is not None and not self.fight_active:
            self.running = job  # set before the task starts, so a fight beginning right now still stops it
            self._job_task = asyncio.create_task(self.run_job(job))
        return job

    # ------------------------------------------------------------------ RPC
    def rpc_status(self, _p: dict) -> dict:
        return {"running": self.running, "fight_active": self.fight_active, "enabled": self.cfg.learn.enabled,
                "state": self.state.to_dict(), "last_result": self.last_result}

    def rpc_request(self, p: dict) -> dict:
        job = str(p.get("job", ""))
        if job not in JOBS:
            return {"ok": False, "error": f"unknown job {job!r} (one of {', '.join(JOBS)})"}
        if job not in self.state.requested:
            self.state.requested.append(job)
            self._save()
        return {"ok": True, "queued": list(self.state.requested)}

    def rpc_cancel(self, _p: dict) -> dict:
        job = self.running
        if job in self.state.requested:
            self.state.requested.remove(job)
            self._save()
        return {"ok": True, "stopped": job, "killed": self.stop_job("owner cancelled")}

    # ------------------------------------------------------------------ main
    async def _bus(self) -> None:
        sub = AsyncSubscriber(self.cfg.ipc.bus_out_port, ["fight.state", "gpu.claim", "kill"], self.cfg.ipc.host)
        try:
            while True:
                topic, msg = await sub.recv()
                if topic == "fight.state" and isinstance(msg, dict):
                    self.on_fight_state(str(msg.get("new", "")))
                elif topic in ("gpu.claim", "kill"):
                    self.stop_job(topic)
        finally:
            sub.close()

    async def run(self) -> None:
        assert self.server is not None
        await self.server.start()
        bus = asyncio.create_task(self._bus())
        psutil.cpu_percent(None)  # prime the CPU sampler
        log.info("learnerd ready (learning %s)", "on" if self.cfg.learn.enabled else "off")
        last = 0.0
        try:
            while True:
                if self.fight_active and self.running is not None:  # belt and braces for a job spawned mid-claim
                    self.stop_job("fight active")
                if time.monotonic() - last >= TICK_S:
                    last = time.monotonic()
                    await self.tick()
                await asyncio.sleep(1.0)
        finally:
            bus.cancel()
            self.stop_job("learnerd exiting")
            await self.server.close()


def main(cfg: JarvisConfig) -> None:
    asyncio.run(Learnerd(cfg).run())
