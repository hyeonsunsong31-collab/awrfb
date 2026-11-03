"""Child processes the kill switch can end completely: on Windows each run gets a Job object with
kill-on-close (the whole process tree dies with it); elsewhere each run gets its own session/process group."""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import subprocess
import sys
import threading
from dataclasses import dataclass


@dataclass
class ProcResult:
    code: int | None
    out: str
    err: str
    timed_out: bool = False
    killed: bool = False


class ProcessRunner:
    def __init__(self, max_output: int = 64_000) -> None:
        self.max_output = max_output
        self._active: dict[int, tuple[subprocess.Popen, object | None]] = {}
        self._lock = threading.Lock()
        self._killed: set[int] = set()

    @property
    def active(self) -> int:
        with self._lock:
            return len(self._active)

    def _spawn(self, argv: list[str], cwd: str | None, env: dict | None, stdin: bool) -> tuple[subprocess.Popen, object | None]:
        kw: dict = {"cwd": cwd, "env": env, "stdout": subprocess.PIPE, "stderr": subprocess.PIPE,
                    "stdin": subprocess.PIPE if stdin else subprocess.DEVNULL}
        job = None
        if sys.platform == "win32":
            kw["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
            proc = subprocess.Popen(argv, **kw)
            from jarvis.platform.win.job import JobObject

            job = JobObject()
            job.assign(proc)
        else:
            proc = subprocess.Popen(argv, start_new_session=True, **kw)
        return proc, job

    def _kill(self, proc: subprocess.Popen, job: object | None) -> None:
        if job is not None:
            with contextlib.suppress(Exception):
                job.terminate()  # type: ignore[attr-defined]
        elif proc.poll() is None:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(proc.pid, signal.SIGKILL)

    def run_sync(self, argv: list[str], *, timeout_s: float = 60.0, cwd: str | None = None, env: dict | None = None,
                 input_text: str | None = None) -> ProcResult:
        proc, job = self._spawn(argv, cwd, env, input_text is not None)
        with self._lock:
            self._active[proc.pid] = (proc, job)
        timed_out = False
        try:
            try:
                out, err = proc.communicate(input_text.encode() if input_text is not None else None, timeout=timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                self._kill(proc, job)
                out, err = proc.communicate()
            else:
                self._kill(proc, job)  # the parent finished: end anything it left running in its tree
        finally:
            with self._lock:
                self._active.pop(proc.pid, None)
                killed = proc.pid in self._killed
                self._killed.discard(proc.pid)
            if job is not None:
                with contextlib.suppress(Exception):
                    job.close()  # type: ignore[attr-defined]
        dec = lambda b: (b or b"").decode("utf-8", errors="replace")[: self.max_output]  # noqa: E731
        return ProcResult(proc.returncode, dec(out), dec(err), timed_out, killed)

    async def run(self, argv: list[str], **kw) -> ProcResult:
        return await asyncio.to_thread(self.run_sync, argv, **kw)

    def kill_all(self) -> int:
        """Kill switch: end every running process tree. Returns how many trees were killed."""
        with self._lock:
            items = list(self._active.items())
            self._killed.update(pid for pid, _ in items)
        for _pid, (proc, job) in items:
            self._kill(proc, job)
        return len(items)
