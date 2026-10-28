"""`jarvis` command line."""

from __future__ import annotations

import json
import sys

import typer
from rich.console import Console
from rich.table import Table

from jarvis import __version__

app = typer.Typer(add_completion=False, no_args_is_help=True, help="JARVIS - voice agent + real UBG fighter")
console = Console()


def _cfg():
    from jarvis.common.config import load_config

    return load_config()


@app.command()
def version() -> None:
    """Print the version."""
    console.print(f"jarvis {__version__}")


@app.command()
def doctor(
    vram: bool = typer.Option(False, help="Measure VRAM (start Roblox first) and pick the tier."),
    as_json: bool = typer.Option(False, "--json", help="Print JSON."),
) -> None:
    """Check this PC: GPU/VRAM tier, Roblox, audio, models, API key, safety policy."""
    from jarvis.doctor import run_doctor, save_report

    cfg = _cfg()
    rep = run_doctor(cfg, vram=vram)
    path = save_report(rep, cfg.paths().ensure().reports)
    if as_json:
        console.print_json(path.read_text(encoding="utf-8"))
        return
    t = Table(title=f"jarvis doctor {rep.version}")
    t.add_column("check")
    t.add_column("ok")
    t.add_column("detail", overflow="fold")
    for c in rep.checks:
        mark = "[green]yes[/]" if c.ok else ("[red]NO[/]" if c.level == "error" else "[yellow]no[/]")
        t.add_row(c.name, mark, c.detail)
    console.print(t)
    console.print(f"report saved: {path}")


@app.command("input-test")
def input_test(repeats: int = typer.Option(3, help="How many times to type the phrase.")) -> None:
    """Type a test phrase into Notepad with real SendInput; Alt-Tab away to prove it stops."""
    from jarvis.fight.input_test import run_input_test
    from jarvis.platform.factory import get_platform

    cfg = _cfg()
    plat = get_platform()
    console.print("Typing into Notepad. Alt-Tab away or press the panic hotkey to prove input stops.")
    res = run_input_test(plat, cfg.safety.panic_hotkeys[0], repeats=repeats)
    out = cfg.paths().ensure().reports / "input-test.json"
    out.write_text(json.dumps(res.to_dict(), indent=2), encoding="utf-8")
    console.print_json(json.dumps(res.to_dict()))
    if res.keys_still_down:
        console.print("[red]keys still held down - please report this[/]")
        raise typer.Exit(1)


@app.command("audit-verify")
def audit_verify() -> None:
    """Verify the tamper-evident audit logs (hash chains)."""
    from jarvis.common.audit import verify_chain

    cfg = _cfg()
    d = cfg.paths().audit
    origins = sorted({p.name.rsplit("-", 1)[0] for p in d.glob("*.jsonl")})
    bad = False
    for o in origins:
        res = verify_chain(sorted(d.glob(f"{o}-*.jsonl")))
        color = "green" if res.ok else "red"
        console.print(f"[{color}]{o}: {'ok' if res.ok else 'BROKEN'}[/] records={res.records} {res.reason}")
        bad |= not res.ok
    if bad:
        raise typer.Exit(1)


def _register_subcommands() -> None:
    """Later milestones attach their command groups here (fight, calibrate, voice, ...)."""
    import importlib

    for mod in ("jarvis.cli_fight", "jarvis.cli_voice", "jarvis.cli_learn", "jarvis.cli_setup", "jarvis.cli_run"):
        try:
            m = importlib.import_module(mod)
        except ModuleNotFoundError as e:
            if e.name == mod:
                continue
            raise
        m.register(app)


def main() -> None:
    _register_subcommands()
    try:
        app()
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()

