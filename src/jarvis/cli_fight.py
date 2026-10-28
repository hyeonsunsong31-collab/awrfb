"""Game commands: launch, capture test, calibration, fight, shadow, replay, report."""

from __future__ import annotations

import json
import time
import zipfile

import typer
from rich.console import Console

console = Console()
calibrate_app = typer.Typer(help="Calibrate JARVIS to UBG on this PC.", no_args_is_help=True)


def _cfg():
    from jarvis.common.config import load_config

    return load_config()


def _io(cfg, **kw):
    from jarvis.common.audit import AuditLog
    from jarvis.fight.gameio import GameIO
    from jarvis.platform.factory import get_platform

    audit = AuditLog(cfg.paths().ensure().audit, "fightd") if cfg.safety.audit_enabled else None
    return GameIO(cfg, get_platform(), audit=audit, **kw)


def launch(
    private: bool = typer.Option(True, "--private/--public", help="Private server (default) or a public server."),
) -> None:
    """Launch untitled boxing game in the Roblox app (you must be signed in already)."""
    from jarvis.fight.launcher import GameLauncher, LaunchError
    from jarvis.platform.factory import get_platform

    cfg = _cfg()
    gl = GameLauncher(get_platform(), cfg.fight.game, cfg.paths().ensure().run)
    try:
        s = gl.launch(private=private)
    except LaunchError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1) from e
    console.print(f"UBG running: pid {s.pid}, server: {s.server_kind}")


def capture_test(seconds: float = typer.Option(10.0, help="How long to capture.")) -> None:
    """Capture the real Roblox window, record a proof clip and report FPS/latency."""
    from jarvis.fight.perception.hud_bars import HudReader
    from jarvis.fight.perception.layout import load_layout
    from jarvis.fight.recorder import SessionRecorder

    cfg = _cfg()
    paths = cfg.paths().ensure()
    io = _io(cfg)
    io.attach()
    layout = load_layout(paths.calibration)
    hud = HudReader(layout, calibration_dir=paths.calibration) if layout else None
    rc = cfg.fight.recording
    rec = SessionRecorder(paths.recordings, "capture_test", width=rc.width, height=rc.height, ffmpeg=rc.ffmpeg,
                          encoder=rc.encoder, meta={"tier": cfg.tier, "window": io.window.title if io.window else ""})
    io.start()
    lat = []
    n = 0
    last = 0
    t_end = time.monotonic() + seconds
    try:
        while time.monotonic() < t_end:
            f = io.frames.get_newer(last, 0.5)
            if f is None:
                continue
            last = f.index
            n += 1
            row = {"fi": f.index, "t_ns": f.t_ns}
            if f.source_t_ns:
                d = (f.t_ns - f.source_t_ns) / 1e6
                if 0 <= d < 1000:
                    lat.append(d)
                    row["capture_latency_ms"] = d
            if hud is not None:
                rd = hud.read(f.image, f.t_ns)
                row["hud"] = rd.values
                row["combat_hud"] = rd.combat_hud
            rec.add_frame(f.image, row)
            if n == 1:
                import cv2

                cv2.imwrite(str(rec.dir / "first_frame.png"), f.image)
    finally:
        io.stop()
    summary = {
        "frames": n,
        "seconds": seconds,
        "fps": n / seconds,
        "dropped_in_slot": io.frames.dropped,
        "capture_latency_ms_median": sorted(lat)[len(lat) // 2] if lat else None,
        "frame_size": None if io.frames.peek() is None else list(io.frames.peek().image.shape[1::-1]),
    }
    d = rec.close(summary)
    console.print_json(json.dumps(summary))
    console.print(f"proof bundle: {d}")


@calibrate_app.command("hud")
def calibrate_hud_cmd(
    camera_mode: str = typer.Option("default", help="UBG camera setting in use (default/side)."),
) -> None:
    """Drag boxes around the HUD bars of a live UBG match; verifies them live and saves the layout."""
    from jarvis.fight.calibrate.fingerprint import Fingerprint
    from jarvis.fight.calibrate.hud_wizard import calibrate_hud
    from jarvis.fight.calibrate.roi_ui import TkRoiPicker

    cfg = _cfg()
    paths = cfg.paths().ensure()
    io = _io(cfg)
    io.attach()
    io.start()
    try:
        console.print("Be IN a UBG match (fight HUD visible). Focusing Roblox for 3 seconds...")
        io.focus_game()
        time.sleep(3)
        cursor_visible = io.plat.cursor.cursor_visible()
        res = calibrate_hud(lambda: io.frames.get_newer(0, 1.0), TkRoiPicker(), paths.calibration,
                            cursor_visible_in_fight=cursor_visible, camera_mode=camera_mode)
    finally:
        io.stop()
    Fingerprint.make(res.layout, io.keymap).save(paths.calibration)
    console.print(f"HUD recognized on {res.live_combat_fraction:.0%} of live frames; confidence {res.live_conf}")
    for w in res.warnings:
        console.print(f"[yellow]warning:[/] {w}")
    console.print(f"saved: {paths.calibration / 'hud_layout.yaml'}  snapshot: {res.snapshot}")


def _run_key_verification(actions: list[str] | None, repeats: int, kind: str) -> list:
    from jarvis.fight.calibrate.fingerprint import Fingerprint
    from jarvis.fight.calibrate.keys import KeyVerifier
    from jarvis.fight.keymap import save_keymap
    from jarvis.fight.perception.hud_bars import HudReader
    from jarvis.fight.perception.layout import load_layout
    from jarvis.fight.recorder import SessionRecorder

    cfg = _cfg()
    paths = cfg.paths().ensure()
    rc = cfg.fight.recording
    rec = SessionRecorder(paths.recordings, kind, width=rc.width, height=rc.height, ffmpeg=rc.ffmpeg, encoder=rc.encoder,
                          meta={"tier": cfg.tier})
    io = _io(cfg, input_log=lambda r: rec.log("inputs", r))
    layout = load_layout(paths.calibration)
    hud = HudReader(layout, calibration_dir=paths.calibration) if layout else None
    acts = actions or [a for a in io.keymap.actions if io.keymap.usable(a)]
    io.attach()
    io.start()
    console.print(f"Pressing {', '.join(acts)} x{repeats} in Roblox in 3 s. Panic hotkey: {cfg.safety.panic_hotkeys[0]}")
    time.sleep(3)
    try:
        reports = KeyVerifier(io, hud, log_event=lambda e: rec.log("events", e), add_frame=rec.add_frame).run(acts, repeats)
    finally:
        io.stop()
    km = io.keymap.model_copy(deep=True)
    for r in reports:
        km.actions[r.action].verified = r.verified
    save_keymap(km, paths.calibration)
    if layout is not None:
        Fingerprint.make(layout, km).save(paths.calibration)
    summary = {"actions": [r.to_dict() for r in reports]}
    d = rec.close(summary)
    for r in reports:
        color = "green" if r.verified else "red"
        lat = r.median_latency_ms
        console.print(f"[{color}]{r.action:<13} {r.success_rate:4.0%} seen[/]  median latency {lat if lat is None else round(lat)} ms")
    console.print(f"proof bundle: {d}")
    return reports


@calibrate_app.command("popup")
def calibrate_popup_cmd(name: str = typer.Argument(..., help="e.g. shop, spin, reward_wheel, purchase")) -> None:
    """Capture the UI currently open in UBG (shop/spin/wheel...) so fights never act on that screen."""
    from jarvis.fight.perception.ui_guard import UiGuard

    cfg = _cfg()
    paths = cfg.paths().ensure()
    io = _io(cfg)
    io.attach()
    io.start(capture=True)
    try:
        console.print(f"Open the '{name}' screen in UBG now. Capturing in 4 seconds...")
        io.focus_game()
        time.sleep(4)
        f = io.frames.get_newer(0, 2.0)
        if f is None:
            console.print("[red]no frame captured[/]")
            raise typer.Exit(1)
        p = UiGuard.save_popup_template(f.image, paths.calibration / "popups", name)
    finally:
        io.stop()
    console.print(f"saved popup template: {p}")


@calibrate_app.command("keys")
def calibrate_keys_cmd(
    actions: str = typer.Option("", help="Comma-separated actions (default: every enabled action)."),
    repeats: int = typer.Option(10, help="Presses per action."),
) -> None:
    """Press every keymap action in UBG and confirm the game reacts (private server / training spot!)."""
    _run_key_verification([a.strip() for a in actions.split(",") if a.strip()] or None, repeats, "calibrate_keys")


@calibrate_app.command("latency")
def calibrate_latency_cmd(repeats: int = typer.Option(20)) -> None:
    """Measure input-to-screen latency with repeated blocks."""
    _run_key_verification(["block"], repeats, "calibrate_latency")


def report(
    last: int = typer.Option(5, help="How many recent sessions to include."),
    videos: bool = typer.Option(False, help="Include video files (large)."),
) -> None:
    """Zip recent proof bundles, calibration, doctor reports and logs to send back for tuning."""
    cfg = _cfg()
    paths = cfg.paths().ensure()
    out = paths.reports / f"jarvis-report-{time.strftime('%Y%m%d-%H%M%S')}.zip"
    sessions = sorted([p for p in paths.recordings.iterdir() if p.is_dir()])[-last:]
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for s in sessions:
            for f in s.rglob("*"):
                if f.is_file() and (videos or f.suffix not in (".mkv", ".mp4")):
                    z.write(f, f"sessions/{s.name}/{f.relative_to(s)}")
        for d, prefix in ((paths.calibration, "calibration"), (paths.reports, "reports"), (paths.logs, "logs")):
            for f in d.rglob("*"):
                if f.is_file() and f != out and f.suffix != ".zip" and f.stat().st_size < 50 * 2**20:
                    z.write(f, f"{prefix}/{f.relative_to(d)}")
    console.print(f"report: {out} ({out.stat().st_size / 2**20:.1f} MB) - contains no API keys or passwords")


def fight(
    matches: int = typer.Option(1, help="How many matches to fight before disarming."),
    auto_queue: bool = typer.Option(False, help="Walk to a free ring box by itself (WASD only)."),
    hotkey: str = typer.Option("ctrl+alt+f", help="Arm/disarm toggle hotkey."),
    arm_now: bool = typer.Option(False, "--arm-now", help="Arm immediately instead of waiting for the hotkey."),
) -> None:
    """Fight in UBG (owner command). Stand in a ring box in your private server, then press the hotkey."""
    from jarvis.fight.loop import ArmRefused, FightLoop
    from jarvis.fight.states import FightState

    cfg = _cfg()
    io = _io(cfg)
    io.attach()
    io.start()
    loop = FightLoop(cfg, io, on_callout=lambda k, d: console.print(f"[cyan]{k}[/] {d if d else ''}"))
    loop.start()

    def toggle() -> None:
        if io.sm.state == FightState.DISARMED:
            try:
                io.focus_game()
                loop.arm(matches, auto_queue)
                console.print("[green]ARMED[/] - waiting for the match")
            except ArmRefused as e:
                console.print(f"[red]refused:[/] {e}")
        else:
            loop.disarm("owner hotkey")
            console.print("[yellow]DISARMED[/]")

    hk = io.plat.new_hotkeys()
    hk.register(hotkey, toggle)
    hk.start()
    console.print(f"Arm/disarm: {hotkey}   Panic: {', '.join(cfg.safety.panic_hotkeys)}   Ctrl+C quits")
    problems = loop.preflight()
    for pr in problems:
        console.print(f"[yellow]preflight:[/] {pr}")
    if arm_now:
        toggle()
    try:
        last = None
        while True:
            time.sleep(0.5)
            st = io.sm.state.value
            if st != last:
                console.print(f"state: {st}")
                last = st
    except KeyboardInterrupt:
        pass
    finally:
        hk.stop()
        loop.stop()
        io.stop()


def fightd() -> None:
    """Run the fight daemon (normally started by the supervisor)."""
    from jarvis.common.logging_setup import setup_logging
    from jarvis.fight.fightd import main as fightd_main

    cfg = _cfg()
    setup_logging("fightd", cfg.paths().ensure().logs, cfg.log_level)
    fightd_main(cfg)


def register(app: typer.Typer) -> None:
    app.command()(launch)
    app.command("capture-test")(capture_test)
    app.add_typer(calibrate_app, name="calibrate")
    app.command()(report)
    app.command()(fight)
    app.command()(fightd)

