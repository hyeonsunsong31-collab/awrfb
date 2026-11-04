"""`jarvis setup`: the first-run wizard. Each step runs the same command you can run on its own later, asks
before it starts, and can be skipped. Nothing here handles your Roblox password: you sign in to the Roblox
app yourself."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass

import typer
from rich.console import Console
from rich.panel import Panel

console = Console()


@dataclass(frozen=True)
class Step:
    key: str
    title: str
    why: str
    commands: tuple[tuple[str, ...], ...]
    windows_only: bool = False


ROBLOX_CHECKLIST = """\
Roblox / UBG settings for a bot account (do these yourself in the Roblox app):
  1. Use an ALT account with 0 Robux, no payment method, and a Parental Controls spending limit of 0.
  2. Sign in to the Roblox desktop app yourself - JARVIS never sees or stores your password.
  3. Fight only in a PRIVATE server with people who agreed to it. UBG's rules name auto-dodging as
     bannable and Roblox prohibits automation that gives an unfair advantage. Put your private
     server's link code in fight.game.private_server_link_code (config.yaml); public servers stay
     off unless you set fight.game.allowed_servers: public AND fight.game.i_accept_ban_risk: true.
  4. Roblox settings: windowed/borderless, graphics fixed (not Automatic), Camera Shake reduced.
  5. In UBG: HUD on, default keybinds (or calibrate yours), stand in a ring's blue box to queue."""

STEPS = (
    Step("doctor", "Check this PC", "GPU/VRAM tier, Roblox install, audio devices, models, safety policy.",
         (("doctor",),)),
    Step("api", "NVIDIA API key", "Stored in Windows Credential Manager (never printed). Then every model is probed.",
         (("api-key",), ("probe",))),
    Step("models", "Voice models", "Wake word, VAD, Whisper, Supertonic and the speaker model (pinned, SHA-256 checked).",
         (("voice-models", "--all"),)),
    Step("audio", "Microphone and speakers", "Say a few sentences in Korean and English; check what JARVIS hears.",
         (("listen", "--seconds", "20"), ("say", "안녕하세요, 저는 자비스입니다. Nice to meet you."))),
    Step("enroll", "Your voiceprint", "60 s of your voice (both languages) so only you can start fights or give PC commands.",
         (("enroll",),)),
    Step("voice", "JARVIS's own voice (8 GB+ GPU)", "Designs an original voice once with Qwen3-TTS (skip on 6 GB cards).",
         (("setup-voice",),)),
    Step("roblox", "Roblox and UBG", ROBLOX_CHECKLIST, (("launch",),), windows_only=True),
    Step("calibrate", "Calibrate to UBG", "Start a private-server match first. HUD bars, popups, keybinds, then latency.",
         (("calibrate", "hud"), ("calibrate", "popup"), ("calibrate", "keys"), ("calibrate", "latency")), windows_only=True),
    Step("vram", "VRAM tier with Roblox running", "Measures free VRAM next to Roblox and picks the model tier.",
         (("doctor", "--vram"),), windows_only=True),
    Step("autostart", "Start at logon", "Task Scheduler logon task with your normal rights (not a service).",
         (("autostart", "--enable"),), windows_only=True),
)


def run_cmd(args: tuple[str, ...]) -> int:
    return subprocess.run([sys.executable, "-m", "jarvis", *args], check=False).returncode


def setup(
    only: str = typer.Option("", help=f"Run just these steps (comma-separated): {', '.join(s.key for s in STEPS)}"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Run every step without asking."),
) -> None:
    """First-run wizard: API key, voice models, audio check, voiceprint, voice, Roblox checklist, calibration, autostart."""
    wanted = {k.strip() for k in only.split(",") if k.strip()}
    unknown = wanted - {s.key for s in STEPS}
    if unknown:
        console.print(f"[red]unknown step(s)[/]: {', '.join(sorted(unknown))}")
        raise typer.Exit(2)
    console.print(Panel("JARVIS setup - every step can be skipped and re-run later on its own.\n"
                        "Safety rules you agree to by using the fighter are in SAFETY.md.", title="JARVIS"))
    failed = []
    for s in STEPS:
        if wanted and s.key not in wanted:
            continue
        if s.windows_only and sys.platform != "win32":
            console.print(f"[dim]- {s.title}: Windows only, skipped[/]")
            continue
        console.print(Panel(s.why, title=s.title))
        if not yes and not typer.confirm(f"Run '{s.title}' now?", default=True):
            continue
        for c in s.commands:
            console.print(f"[cyan]$ jarvis {' '.join(c)}[/]")
            if run_cmd(c) != 0:
                failed.append(f"jarvis {' '.join(c)}")
                console.print("[yellow]that step did not finish - you can re-run it later[/]")
                break
    console.print(Panel(("Done. " if not failed else "Done, but re-run: " + "; ".join(failed) + ". ")
                        + "Start JARVIS with `jarvis start` and say 'Hey Jarvis' (or 자비스).", title="JARVIS"))


def register(app: typer.Typer) -> None:
    app.command()(setup)
