"""`jarvis hud`: start the holographic overlay (Electron). The supervisor will do this at logon."""

from __future__ import annotations

import contextlib
import json
import os
import secrets
import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console

console = Console()


def _cfg():
    from jarvis.common.config import load_config

    return load_config()


def write_hud_config(cfg, *, port: int, token_file: Path) -> Path:
    """What the Electron main process needs: where the hub is, the token file, window behaviour."""
    h = cfg.hud
    p = cfg.paths().ensure().run / "hud.json"
    p.write_text(json.dumps({
        "host": cfg.ipc.host, "port": port, "token_file": str(token_file), "hotkey": h.interactive_hotkey,
        "content_protection": h.content_protection, "stream_mode": h.stream_mode, "combat_mode": h.combat_mode,
        "combat_fps_cap": h.combat_fps_cap, "prefer_secondary_monitor": h.prefer_secondary_monitor,
    }, indent=2), encoding="utf-8")
    return p


def _token_file(run: Path, token: str, name: str) -> Path:
    f = run / name
    f.write_text(token, encoding="utf-8")
    with contextlib.suppress(OSError):
        os.chmod(f, 0o600)
    return f


def hud(
    demo: bool = typer.Option(False, help="Drive the HUD with a scripted demo hub (no core, no Roblox)."),
    dev: bool = typer.Option(False, help="electron-vite dev mode (hot reload)."),
) -> None:
    """Show the holographic HUD (transparent, click-through, never takes focus; Ctrl+Alt+J = interact)."""
    from jarvis.common.ipc import load_or_create_token
    from jarvis.common.paths import repo_root

    cfg = _cfg()
    d = repo_root() / cfg.hud.electron_dir
    if not (d / "node_modules").exists():
        console.print(f"[red]HUD dependencies missing[/] - run `npm ci` in {d}")
        raise typer.Exit(1)
    run = cfg.paths().ensure().run
    helpers: list[subprocess.Popen] = []
    if demo:
        token = secrets.token_urlsafe(24)
        port = cfg.ipc.hud_ws_port + 9
        helpers.append(subprocess.Popen([sys.executable, "-m", "jarvis.hud_bridge.demo", "--port", str(port), "--token", token, "--forever"]))
        token_file = _token_file(run, token, "hud-demo.token")
    else:
        port = cfg.ipc.hud_ws_port
        token = load_or_create_token(run)
        token_file = run / "ipc.token" if (run / "ipc.token").exists() else _token_file(run, token, "hud.token")
    conf = write_hud_config(cfg, port=port, token_file=token_file)
    env = {**os.environ, "JARVIS_HUD_CONFIG": str(conf)}
    npm = "npm.cmd" if sys.platform == "win32" else "npm"
    # Started by the supervisor (no console): never let npm's cmd.exe pop up a console window.
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" and os.environ.get("JARVIS_SUPERVISED") == "1" else 0
    try:
        if dev:
            subprocess.run([npm, "run", "dev"], cwd=d, env=env, check=False, creationflags=flags)
        else:
            if not (d / "out" / "main" / "main.js").exists():
                subprocess.run([npm, "run", "build"], cwd=d, env=env, check=True, creationflags=flags)
            code = subprocess.run([npm, "run", "start"], cwd=d, env=env, check=False, creationflags=flags).returncode
            if code:
                raise typer.Exit(code)
    finally:
        for p in helpers:
            p.terminate()


def register(app: typer.Typer) -> None:
    app.command()(hud)
