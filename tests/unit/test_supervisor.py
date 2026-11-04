"""Supervisor: restart policy, crash handling, the stuck-key release when fightd dies, GPU lending and the
real bus/RPC lifecycle. Child processes are small Python stand-ins (no Roblox, no GPU)."""

from __future__ import annotations

import asyncio
import sys
import time
import types

import psutil
import pytest

from jarvis.supervisor import Backoff, ProcSpec, RestartPolicy, Supervisor, default_specs

PY = sys.executable
SLEEP = [PY, "-c", "import time; time.sleep(60)"]
CRASH = [PY, "-c", "import sys; sys.exit(3)"]
FAST = Backoff(base_s=0.05, max_s=0.2, healthy_after_s=30, window_s=60, max_crashes=3, loop_retry_s=0.3)


class FakeSink:
    def __init__(self):
        self.calls = []

    def key_up(self, k):
        self.calls.append(("key_up", k))

    def mouse_up(self, b):
        self.calls.append(("mouse_up", str(b.value)))


def make_sup(cfg, specs, sink=None, backoff=FAST):
    return Supervisor(cfg, specs, backoff=backoff, input_sink_factory=(lambda: sink) if sink else None,
                      serve=False, tray=False, hotkeys=False)


def wait_for(pred, timeout=10.0, step=0.05, poll=None):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if poll:
            poll()
        if pred():
            return True
        time.sleep(step)
    return False


# ------------------------------------------------------------------------------------- policy
def test_restart_policy_backoff_reset_and_crash_loop():
    p = RestartPolicy(Backoff(base_s=1, max_s=8, healthy_after_s=60, window_s=300, max_crashes=5, loop_retry_s=300))
    t = 1000.0
    delays = []
    for _ in range(4):
        delays.append(p.on_exit(started_at=t, now=t + 2))
        t += 10
    assert delays == [1, 2, 4, 8] and not p.crash_looping()
    assert p.on_exit(started_at=t, now=t + 1) == 300 and p.crash_looping()  # 5 quick exits: crash loop
    assert p.on_exit(started_at=t, now=t + 120) == 1 and not p.crash_looping()  # a healthy run resets it


def test_default_specs_follow_config(cfg):
    names = {s.name: s for s in default_specs(cfg, "py")}
    assert set(names) == {"toolhost", "fightd", "core", "learnerd", "hud"}
    assert names["core"].argv == ["py", "-m", "jarvis", "core"]
    assert "--no-voice" in default_specs(cfg, "py", voice=False)[2].argv
    cfg.learn.enabled = False
    assert not {s.name: s for s in default_specs(cfg, "py")}["learnerd"].enabled


# ------------------------------------------------------------------------------------- children
def test_starts_restarts_and_reports_a_crash_loop(cfg):
    sup = make_sup(cfg, [ProcSpec("core", SLEEP), ProcSpec("toolhost", CRASH), ProcSpec("hud", SLEEP, enabled=False)])
    try:
        sup.poll()
        core, th = sup.children["core"], sup.children["toolhost"]
        assert core.status == "running" and core.pid and sup.children["hud"].status == "disabled"
        assert wait_for(lambda: th.status == "crash_loop", poll=sup.poll), th
        assert th.restarts >= 3 and th.last_exit == 3
        assert sup.health() == "degraded" and any("toolhost keeps crashing" in n for n in sup.notices)
        assert core.status == "running" and core.restarts == 0  # the healthy process was never touched
        assert (cfg.paths().logs / "core.out.log").exists()
        sup.stop_child(core)
        assert core.status == "stopped" and core.proc is None and core.last_exit is not None  # stopped and reaped
        sup.poll()
        assert core.status == "stopped"  # a deliberate stop is not restarted
    finally:
        asyncio.run(sup.shutdown())
    assert all(c.proc is None for c in sup.children.values())


def test_fightd_dying_releases_every_game_key(cfg):
    sink = FakeSink()
    sup = make_sup(cfg, [ProcSpec("fightd", CRASH)], sink=sink, backoff=Backoff(base_s=30))
    try:
        sup.poll()
        assert wait_for(lambda: sup.children["fightd"].status == "backoff", poll=sup.poll)
        released = {k for _, k in sink.calls}
        assert {"w", "a", "s", "d", "f", "space"} <= released and {"left", "right"} <= released
        assert all(kind in ("key_up", "mouse_up") for kind, _ in sink.calls)  # releasing only, never pressing
    finally:
        asyncio.run(sup.shutdown())


def test_key_release_failure_never_takes_the_supervisor_down(cfg):
    def broken():
        raise RuntimeError("no input on this platform")

    sup = Supervisor(cfg, [ProcSpec("fightd", CRASH)], backoff=FAST, input_sink_factory=broken, serve=False, tray=False, hotkeys=False)
    try:
        sup.poll()
        assert wait_for(lambda: sup.children["fightd"].restarts >= 2, poll=sup.poll)
    finally:
        asyncio.run(sup.shutdown())


def test_keys_are_released_after_an_unclean_previous_run(cfg):
    import json

    run = cfg.paths().ensure().run
    sink = FakeSink()
    sup = make_sup(cfg, [], sink=sink)
    (run / "supervisor.json").write_text(json.dumps({"procs": [{"name": "fightd", "status": "stopped"}]}))
    assert sup.recover_after_crash() is False and sink.calls == []
    (run / "supervisor.json").write_text(json.dumps({"procs": [{"name": "fightd", "status": "running"}]}))
    assert sup.recover_after_crash() is True and ("key_up", "w") in sink.calls


def test_deliberate_fightd_stop_does_not_count_as_a_crash(cfg):
    sink = FakeSink()
    sup = make_sup(cfg, [ProcSpec("fightd", SLEEP)], sink=sink)
    sup.poll()
    sup.stop_child(sup.children["fightd"])
    sup.poll()
    assert sink.calls == [] and sup.children["fightd"].restarts == 0
    asyncio.run(sup.shutdown())


@pytest.mark.skipif(sys.platform == "win32", reason="psutil child listing is enough here; Windows uses Job objects")
def test_gpu_is_lent_to_the_fight(cfg):
    # learnerd stand-in with one "job" child (its command line names jarvis.learn.jobs, like the real one)
    learner = [PY, "-c", "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)',"
               "'jarvis.learn.jobs']); time.sleep(60)"]
    sup = make_sup(cfg, [ProcSpec("learnerd", learner)])
    try:
        sup.poll()
        pid = sup.children["learnerd"].pid
        assert wait_for(lambda: len(psutil.Process(pid).children()) == 1)
        job = psutil.Process(pid).children()[0]
        assert sup.lend_gpu(sup.clock()) == 0  # no fight: learning may use the GPU
        sup.on_fight_state("fighting")
        assert sup.lend_gpu(sup.clock()) == 0  # grace period: learnerd stops its own job first
        sup.fight_since -= 5
        assert sup.lend_gpu(sup.clock()) == 1
        assert wait_for(lambda: not job.is_running() or job.status() == psutil.STATUS_ZOMBIE)
        assert sup.children["learnerd"].status == "running"  # learnerd itself keeps running
    finally:
        asyncio.run(sup.shutdown())


# ------------------------------------------------------------------------------------- live bus + RPC
def test_supervisor_lifecycle_over_real_bus_and_rpc(data_dir):
    from jarvis.common.config import load_config
    from jarvis.common.ipc import Publisher, RpcClient, load_or_create_token

    cfg = load_config(environ={"JARVIS__IPC__BUS_IN_PORT": "48810", "JARVIS__IPC__BUS_OUT_PORT": "48811",
                               "JARVIS__IPC__RPC_PORTS__SUPERVISOR": "48820", "JARVIS__IPC__RPC_PORTS__FIGHTD": "48823"})
    sup = Supervisor(cfg, [ProcSpec("core", SLEEP)], backoff=FAST, input_sink_factory=FakeSink, tray=False, hotkeys=False)

    async def go():
        task = asyncio.create_task(sup.run())
        client = RpcClient(48820, load_or_create_token(cfg.paths().run), timeout=1.0)
        try:
            st = None
            for _ in range(100):
                await asyncio.sleep(0.05)
                try:
                    st = await client.call("status")
                    if st["procs"][0]["status"] == "running":
                        break
                except Exception:
                    continue
            assert st and st["procs"][0]["name"] == "core" and st["health"] == "ok"
            pub = Publisher(48810)
            for _ in range(60):  # PUB/SUB join is asynchronous: repeat until the supervisor heard it
                pub.publish("fight.state", {"old": "disarmed", "new": "fighting"})
                await asyncio.sleep(0.05)
                if sup.fight_state == "fighting":
                    break
            pub.close()
            assert sup.fight_state == "fighting"
            assert (await client.call("restart", {"name": "nope"}))["ok"] is False
            await client.call("shutdown")
            await asyncio.wait_for(task, 20)
        finally:
            await client.close()

    asyncio.run(go())
    assert sup.children["core"].proc is None and sup.children["core"].status == "stopped"
    assert (cfg.paths().run / "supervisor.json").exists()


# ------------------------------------------------------------------------------------- tray + CLI
def test_tray_icon_and_menu(monkeypatch):
    pytest.importorskip("PIL")  # Pillow comes with the Windows extra (tray icon)
    from jarvis import tray as tray_mod

    img = tray_mod.icon_image("fighting", 64)
    assert img.size == (64, 64) and img.getpixel((32, 32))[:3] == tray_mod.COLORS["fighting"]
    assert img.getpixel((0, 0))[3] == 0  # transparent corners
    assert tray_mod.tray_state("ok", "fighting") == "fighting" and tray_mod.tray_state("degraded", "disarmed") == "degraded"

    class Item:
        def __init__(self, text, action, enabled=True):
            self.text, self.action, self.enabled = text, action, enabled

    class Menu:
        SEPARATOR = object()

        def __init__(self, *items):
            self.items = items

    class Icon:
        def __init__(self, name, icon, title, menu):
            self.icon, self.title, self.menu = icon, title, menu

    monkeypatch.setitem(sys.modules, "pystray", types.SimpleNamespace(MenuItem=Item, Menu=Menu, Icon=Icon))
    hits = []
    t = tray_mod.Tray({k: (lambda k=k: hits.append(k)) for k in tray_mod.LABELS}, status=lambda: ("ok", "fighting"))
    items = [i for i in t.icon.menu.items if i is not Menu.SEPARATOR]
    assert items[0].enabled is False and "fight: fighting" in items[0].text(None)
    by_label = {i.text: i for i in items[1:]}
    by_label[tray_mod.LABELS["panic"]].action()
    assert hits == ["panic"]
    assert t.refresh() is True and t.refresh() is False  # icon redrawn only when the state changes


def test_hotkeys_second_panic_and_mute_toggle(cfg):
    sup = make_sup(cfg, [])
    b = sup.hotkey_bindings()
    assert cfg.safety.panic_hotkeys[0] not in b  # fightd's own
    assert set(cfg.safety.panic_hotkeys[1:]) <= set(b) and cfg.voice.mute.hotkey in b
    published = []
    sup.publish = lambda topic, payload: published.append(topic)
    b[cfg.safety.panic_hotkeys[1]]()
    assert published == ["kill"]


def test_autostart_command_uses_task_scheduler_script():
    from jarvis.cli_run import autostart_command

    on = autostart_command(True)
    assert on[0] == "powershell.exe" and on[-4] == "-Python" and on[5].endswith("register_autostart.ps1")
    assert autostart_command(False)[5].endswith("unregister_autostart.ps1")


def test_gameio_leaves_the_second_panic_hotkey_to_the_supervisor(cfg, monkeypatch):
    from fakes.platform import ROBLOX, FakeHotkeys, FakeWindowManager, fake_platform
    from jarvis.fight.gameio import GameIO

    made = []

    def hk():
        made.append(FakeHotkeys())
        return made[-1]

    plat = fake_platform(windows=FakeWindowManager([ROBLOX], foreground=ROBLOX.hwnd), new_hotkeys=hk)
    for supervised, expect in (("1", cfg.safety.panic_hotkeys[:1]), ("0", cfg.safety.panic_hotkeys)):
        monkeypatch.setenv("JARVIS_SUPERVISED", supervised)
        io = GameIO(cfg, plat)
        io.attach()
        io.start(capture=False)
        try:
            assert list(made[-1].registered) == expect
        finally:
            io.stop()
