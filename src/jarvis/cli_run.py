"""Running JARVIS: start/stop/status of the whole system (the supervisor) and logon autostart."""

from __future__ import annotations

import asyncio
import subprocess
import sys
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

console = Console()
TASK_NAME = "JARVIS"


def _cfg():
    from jarvis.common.config import load_config

    return load_config()


def supervisor_call(cfg, method: str, params: dict | None = None, timeout: float = 3.0) -> dict | None:
    from jarvis.common.ipc import RpcClient, RpcError, load_or_create_token

    async def go() -> dict | None:
        c = RpcClient(cfg.ipc.rpc_ports["supervisor"], load_or_create_token(cfg.paths().ensure().run), cfg.ipc.host, timeout=timeout)
        try:
            return dict(await c.call(method, params or {}) or {})
        except RpcError:
            return None
        finally:
            await c.close()

    return asyncio.run(go())


def gui_python() -> str:
    """pythonw.exe on Windows (no console window), else the current interpreter."""
    exe = Path(sys.executable)
    if sys.platform == "win32":
        w = exe.with_name("pythonw.exe")
        if w.exists():
            return str(w)
    return str(exe)


def start(
    foreground: bool = typer.Option(False, "--foreground", "-f", help="Run the supervisor in this terminal (logs to the console)."),
    no_voice: bool = typer.Option(False, "--no-voice", help="Start core without audio devices (typed chat only)."),
) -> None:
    """Start JARVIS: the supervisor launches core, toolhost, fightd, learnerd and the HUD, and keeps them running."""
    cfg = _cfg()
    if supervisor_call(cfg, "status", timeout=1.5) is not None:
        console.print("JARVIS is already running (`jarvis status`).")
        return
    if foreground:
        from jarvis.common.logging_setup import setup_logging
        from jarvis.supervisor import Supervisor, default_specs

        setup_logging("supervisor", cfg.paths().ensure().logs, cfg.log_level)
        try:
            asyncio.run(Supervisor(cfg, default_specs(cfg, voice=not no_voice)).run())
        except RuntimeError as e:
            console.print(f"[red]{e}[/]")
            raise typer.Exit(1) from None
        return
    argv = [gui_python(), "-m", "jarvis.supervisor"] + (["--no-voice"] if no_voice else [])
    kw: dict = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL,
                "cwd": str(cfg.paths().ensure().root)}
    if sys.platform == "win32":
        kw["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    else:
        kw["start_new_session"] = True
    subprocess.Popen(argv, **kw)
    for _ in range(40):
        time.sleep(0.25)
        if supervisor_call(cfg, "status", timeout=1.0) is not None:
            console.print("JARVIS started. Say 'Hey Jarvis' (or 자비스). `jarvis status` shows the processes.")
            return
    console.print(f"[yellow]the supervisor did not answer yet - check {cfg.paths().logs / 'supervisor.log'}[/]")


def stop() -> None:
    """Quit JARVIS (disarms the fighter first, releases every game key, stops all processes)."""
    cfg = _cfg()
    if supervisor_call(cfg, "shutdown") is None:
        console.print("JARVIS is not running.")
        return
    for _ in range(80):
        time.sleep(0.25)
        if supervisor_call(cfg, "status", timeout=0.5) is None:
            console.print("JARVIS stopped.")
            return
    console.print("[yellow]still shutting down...[/]")


def status() -> None:
    """Show every JARVIS process: running, restarting or crash-looping."""
    cfg = _cfg()
    r = supervisor_call(cfg, "status")
    if r is None:
        console.print("JARVIS is not running. Start it with `jarvis start`.")
        raise typer.Exit(1)
    t = Table("process", "status", "pid", "restarts", "last exit", "up (s)", title=f"JARVIS - {r.get('health')} - fight: {r.get('fight_state')}")
    colors = {"running": "green", "backoff": "yellow", "crash_loop": "red", "disabled": "dim", "stopped": "dim"}
    for p in r.get("procs", []):
        c = colors.get(p["status"], "white")
        t.add_row(p["name"], f"[{c}]{p['status']}[/]", str(p.get("pid") or "-"), str(p["restarts"]),
                  "-" if p.get("last_exit") is None else str(p["last_exit"]), str(p.get("up_s", 0)))
    console.print(t)
    for n in r.get("notices", [])[-5:]:
        console.print(f"[yellow]! {n}[/]")


def restart(name: str = typer.Argument("all", help="core | toolhost | fightd | learnerd | hud | all")) -> None:
    """Restart one JARVIS process (or all of them)."""
    cfg = _cfg()
    r = supervisor_call(cfg, "restart", {"name": name}, timeout=30.0)
    if r is None:
        console.print("JARVIS is not running.")
        raise typer.Exit(1)
    console.print(f"restarted: {', '.join(r['restarted'])}" if r.get("ok") else f"[red]{r.get('error')}[/]")


def panic() -> None:
    """Emergency stop from a terminal: disarm the fighter, release keys, cancel the agent, stop speech."""
    cfg = _cfg()
    if supervisor_call(cfg, "panic", {"reason": "panic (terminal)"}) is None:
        from jarvis.common.ipc import Publisher

        pub = Publisher(cfg.ipc.bus_in_port, cfg.ipc.host)  # no supervisor: shout on the bus directly
        time.sleep(0.3)
        pub.publish("kill", {"reason": "panic (terminal)"})
        time.sleep(0.2)
        pub.close()
    console.print("panic sent.")


# ------------------------------------------------------------------------------------- autostart
def autostart_command(enable: bool) -> list[str]:
    """The Task Scheduler call (logon task, normal rights, restart on failure)."""
    script = Path(__file__).resolve().parents[2] / "scripts" / ("register_autostart.ps1" if enable else "unregister_autostart.ps1")
    args = ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)]
    if enable:
        args += ["-Python", gui_python(), "-WorkDir", str(Path(__file__).resolve().parents[2])]
    return args


def autostart(
    enable: bool = typer.Option(True, "--enable/--disable", help="Register or remove the logon task."),
) -> None:
    """Start JARVIS automatically when you log on (Task Scheduler, your normal user rights)."""
    if sys.platform != "win32":
        console.print("[red]autostart uses Windows Task Scheduler[/]")
        raise typer.Exit(1)
    r = subprocess.run(autostart_command(enable), check=False)
    if r.returncode:
        raise typer.Exit(r.returncode)
    console.print("autostart enabled: JARVIS starts at logon." if enable else "autostart removed.")


def register(app: typer.Typer) -> None:
    for fn in (start, stop, status, restart, panic, autostart):
        app.command()(fn)

