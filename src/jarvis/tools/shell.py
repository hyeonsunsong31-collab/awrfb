"""PowerShell tool. Each command is parsed for real (pwsh AST) and checked against the protected policy
before it runs; it runs hidden, in its own Job object, with a timeout. Deletes must use fs_delete (Recycle
Bin) and the game is off limits: Roblox is driven only by the fight system."""

from __future__ import annotations

import shutil
import sys

from jarvis.safety.capabilities import Cap
from jarvis.tools.base import Refused, Tool, ToolContext, ToolResult, clip, i, s

GAME_WORDS = ("roblox", "robloxplayer")


def powershell_exe() -> str | None:
    if sys.platform == "win32":
        return shutil.which("pwsh") or shutil.which("powershell") or "powershell.exe"
    return shutil.which("pwsh")


def check_command(ctx: ToolContext, command: str) -> None:
    if not command.strip():
        raise Refused("empty command")
    low = command.lower()
    if any(w in low for w in GAME_WORDS):
        raise Refused("Roblox is controlled only by the fight system (say 'launch UBG' / 'fight'), not by shell commands")
    d = ctx.policy.check_shell(command)
    if not d:
        raise Refused(d.reason)


async def powershell(ctx: ToolContext, a: dict) -> ToolResult:
    command = str(a.get("command", ""))
    check_command(ctx, command)
    exe = powershell_exe()
    if exe is None:
        return ToolResult(False, "PowerShell is not available on this system")
    timeout = max(1, min(int(a.get("timeout_s", 60)), int(getattr(ctx.policy.policy, "shell_timeout_s", 120))))
    r = await ctx.runner.run([exe, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", command], timeout_s=timeout)
    if r.killed:
        return ToolResult(False, "stopped by the kill switch")
    out = (r.out + ("\n[stderr]\n" + r.err if r.err.strip() else "")).strip()
    status = "timed out" if r.timed_out else f"exit code {r.code}"
    return ToolResult(r.code == 0 and not r.timed_out, clip(f"{status}\n{out}" if out else status, 8000), "untrusted", "powershell")


TOOLS = [
    Tool("powershell", "Run a PowerShell command (hidden window, timeout). Use fs_delete for deletions.", Cap.SHELL, powershell,
         {"command": s("PowerShell command or script"), "timeout_s": i("Timeout in seconds (default 60)")}, ("command",),
         trust="untrusted", timeout_s=130.0),
]

