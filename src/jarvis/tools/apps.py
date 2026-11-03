"""Apps and processes. Launched apps are normal apps (the kill switch does not close them); the game and
critical system processes are off limits."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
import webbrowser

from jarvis.safety.capabilities import Cap
from jarvis.tools.base import Refused, Tool, ToolContext, ToolResult, i, ok, s

PROTECTED_PROCS = {
    "system", "registry", "idle", "smss.exe", "csrss.exe", "wininit.exe", "winlogon.exe", "services.exe", "lsass.exe",
    "svchost.exe", "fontdrvhost.exe", "dwm.exe", "msmpeng.exe", "securityhealthservice.exe", "explorer.exe",
    "systemd", "init", "kthreadd",
}
GAME_REFUSAL = "Roblox is controlled only by the fight system (say 'launch UBG' or 'fight')"

_start_apps: tuple[float, list[dict]] = (0.0, [])


def is_game(ctx: ToolContext, name: str) -> bool:
    return "roblox" in name.lower() or ctx.policy.is_roblox(None, name)


async def start_apps(ctx: ToolContext) -> list[dict]:
    """Installed apps from the Start menu (Get-StartApps), cached for five minutes."""
    global _start_apps
    if time.monotonic() - _start_apps[0] < 300 and _start_apps[1]:
        return _start_apps[1]
    from jarvis.tools.shell import powershell_exe

    exe = powershell_exe()
    if exe is None or ctx.runner is None:
        return []
    r = await ctx.runner.run([exe, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command",
                              "Get-StartApps | Select-Object Name, AppID | ConvertTo-Json -Compress"], timeout_s=30)
    try:
        apps = json.loads(r.out) if r.out.strip() else []
    except ValueError:
        apps = []
    apps = apps if isinstance(apps, list) else [apps]
    _start_apps = (time.monotonic(), apps)
    return apps


def best_app(apps: list[dict], query: str) -> dict | None:
    from rapidfuzz import fuzz, process

    names = [str(a.get("Name", "")) for a in apps]
    hit = process.extractOne(query, names, scorer=fuzz.WRatio, score_cutoff=70)
    return apps[hit[2]] if hit else None


async def app_open(ctx: ToolContext, a: dict) -> ToolResult:
    target = str(a.get("name", "")).strip()
    if not target:
        raise Refused("which app or file?")
    if is_game(ctx, target):
        raise Refused(GAME_REFUSAL)
    if target.startswith(("http://", "https://")):
        webbrowser.open(target, new=2)
        return ok(f"opened {target}")
    path = os.path.expandvars(os.path.expanduser(target))
    if os.path.exists(path):
        if sys.platform == "win32":
            os.startfile(path)  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return ok(f"opened {path}")
    if sys.platform != "win32":
        return ToolResult(False, "opening apps by name works on Windows only")
    app = best_app(await start_apps(ctx), target)
    if app is None:
        return ToolResult(False, f"no installed app matches {target!r}")
    if is_game(ctx, str(app.get("Name", ""))):
        raise Refused(GAME_REFUSAL)
    subprocess.Popen(["explorer.exe", f"shell:AppsFolder\\{app['AppID']}"])
    return ok(f"opened {app['Name']}")


async def app_list(ctx: ToolContext, a: dict) -> ToolResult:
    q = str(a.get("filter", "")).lower()
    names = sorted({str(x.get("Name", "")) for x in await start_apps(ctx)})
    names = [n for n in names if q in n.lower()][:300]
    return ok("\n".join(names) if names else "no apps found", trust="untrusted", source="start menu")


async def process_list(ctx: ToolContext, a: dict) -> ToolResult:
    import psutil

    q = str(a.get("filter", "")).lower()
    n = max(1, min(int(a.get("limit", 25)), 200))

    def collect() -> list[tuple[float, str]]:
        rows = []
        for p in psutil.process_iter(["pid", "name", "memory_info"]):
            name = p.info.get("name") or "?"
            if q and q not in name.lower():
                continue
            mem = (p.info.get("memory_info").rss if p.info.get("memory_info") else 0) / 2**20
            rows.append((mem, f"{name}  pid {p.info['pid']}  {mem:.0f} MB"))
        return sorted(rows, reverse=True)[:n]

    rows = await asyncio.to_thread(collect)
    return ok("\n".join(r for _, r in rows) or "no matching processes", trust="untrusted", source="processes")


def _is_self(p) -> bool:
    if p.pid in (os.getpid(), os.getppid()):
        return True
    try:
        return "jarvis" in " ".join(p.cmdline()).lower()
    except Exception:
        return False


async def process_kill(ctx: ToolContext, a: dict) -> ToolResult:
    import psutil

    target = str(a.get("name") or a.get("pid") or "").strip()
    if not target:
        raise Refused("give a process name or pid")
    if not target.isdigit():
        low = target.lower()
        if low in PROTECTED_PROCS or low + ".exe" in PROTECTED_PROCS:
            raise Refused(f"refusing to end {target}: Windows needs it")
        if is_game(ctx, target):
            raise Refused(GAME_REFUSAL)
    procs = [psutil.Process(int(target))] if target.isdigit() else [
        p for p in psutil.process_iter(["name"]) if (p.info.get("name") or "").lower() in (target.lower(), target.lower() + ".exe")
    ]
    if not procs:
        return ToolResult(False, f"no process named {target!r}")
    for p in procs:
        name = (p.name() or "").lower()
        if name in PROTECTED_PROCS:
            raise Refused(f"refusing to end {name}: Windows needs it")
        if is_game(ctx, name):
            raise Refused(GAME_REFUSAL)
        if _is_self(p):
            raise Refused("refusing to end JARVIS's own processes (say 'stop' instead)")

    def kill() -> int:
        for p in procs:
            p.terminate()
        _gone, alive = psutil.wait_procs(procs, timeout=3)
        for p in alive:
            p.kill()
        return len(procs)

    n = await asyncio.to_thread(kill)
    return ok(f"ended {n} process(es) named {procs[0].name()}")


TOOLS = [
    Tool("app_open", "Open an installed app by name (Start menu), a file/folder path, or a web address.", Cap.APPS, app_open,
         {"name": s("App name (e.g. 'Notepad', '카카오톡'), path, or URL")}, ("name",)),
    Tool("app_list", "List installed apps (Start menu), optionally filtered.", Cap.APPS, app_list,
         {"filter": s("Part of the name")}, trust="untrusted", windows_only=True),
    Tool("process_list", "List running processes by memory use.", Cap.SYSTEM, process_list,
         {"filter": s("Part of the process name"), "limit": i("Max rows (default 25)")}, trust="untrusted"),
    Tool("process_kill", "End a process by name or pid (not system processes, not the game).", Cap.SHELL, process_kill,
         {"name": s("Process name (e.g. notepad.exe)"), "pid": i("Process id")}),
]
