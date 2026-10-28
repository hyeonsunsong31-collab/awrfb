"""Windows Job objects: every child process a tool starts is assigned to a job so the kill switch can
end the whole process tree at once (and closing the job handle kills everything in it)."""

from __future__ import annotations

import ctypes
import subprocess

from jarvis.platform.win.api import kernel32, last_error
from jarvis.platform.win.structs import (
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
    JobObjectExtendedLimitInformation,
)


class JobObject:
    def __init__(self) -> None:
        k = kernel32()
        self._h = k.CreateJobObjectW(None, None)
        if not self._h:
            raise OSError(f"CreateJobObjectW failed ({last_error()})")
        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not k.SetInformationJobObject(
            self._h, JobObjectExtendedLimitInformation, ctypes.byref(info), ctypes.sizeof(info)
        ):
            raise OSError(f"SetInformationJobObject failed ({last_error()})")

    def assign(self, proc: subprocess.Popen) -> None:
        handle = int(proc._handle)  # type: ignore[attr-defined]  # Windows Popen exposes the process handle
        if not kernel32().AssignProcessToJobObject(self._h, handle):
            raise OSError(f"AssignProcessToJobObject failed ({last_error()})")

    def terminate(self, exit_code: int = 1) -> None:
        if self._h:
            kernel32().TerminateJobObject(self._h, exit_code)

    def close(self) -> None:
        if self._h:
            kernel32().CloseHandle(self._h)
            self._h = None

    def __del__(self) -> None:  # pragma: no cover - best effort
        try:
            self.close()
        except Exception:
            pass
