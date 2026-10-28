from __future__ import annotations

import json
import shutil
import subprocess
import threading

import numpy as np
import pytest

from fakes.platform import ROBLOX, FakeWindowManager, fake_platform
from jarvis.common.config import GameConfig
from jarvis.fight.frames import FpsMeter, LatestFrame
from jarvis.fight.launcher import GameLauncher, GameSession, LaunchError, fight_allowed_in, launch_url
from jarvis.fight.recorder import SessionRecorder, read_jsonl
from jarvis.platform.base import Frame


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg missing")
def test_recorder_writes_video_and_rows(tmp_path):
    rec = SessionRecorder(tmp_path, "test", width=320, height=180, encoder="libx264", meta={"tier": "t6"})
    for i in range(30):
        img = np.full((360, 640, 3), i * 8, np.uint8)
        assert rec.add_frame(img, {"fi": i + 100, "t_ns": i * 16_666_667}) == i
        rec.log("inputs", {"t_ns": i, "action": "block", "event": "down"})
    rec.log("events", {"t_ns": 5, "type": "hit_taken", "amount": 0.1})
    d = rec.close(summary={"result": "test"})
    frames = read_jsonl(d / "frames.jsonl")
    assert [f["vi"] for f in frames] == list(range(30))
    assert frames[0]["fi"] == 100 and frames[0]["v"] == 1
    assert len(read_jsonl(d / "inputs.jsonl")) == 30
    assert read_jsonl(d / "events.jsonl")[0]["type"] == "hit_taken"
    meta = json.loads((d / "meta.json").read_text())
    assert meta["frames_written"] == 30 and meta["tier"] == "t6" and meta["encoder"] == "libx264"
    assert json.loads((d / "summary.json").read_text())["result"] == "test"
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
         "stream=nb_read_frames,width,height", "-of", "json", str(d / "video.mkv")],
        capture_output=True, text=True, check=True,
    )
    st = json.loads(probe.stdout)["streams"][0]
    assert int(st["nb_read_frames"]) == 30 and (st["width"], st["height"]) == (320, 180)


def test_recorder_without_ffmpeg_saves_keyframes(tmp_path):
    rec = SessionRecorder(tmp_path, "nofff", width=64, height=36, ffmpeg="definitely-not-ffmpeg")
    for i in range(61):
        rec.add_frame(np.zeros((36, 64, 3), np.uint8), {"fi": i})
    d = rec.close()
    assert len(list((d / "keyframes").glob("*.jpg"))) == 3
    assert "ffmpeg not found" in json.loads((d / "meta.json").read_text())["errors"][0]


def test_launch_urls_and_server_policy():
    g = GameConfig()
    assert launch_url(g, private=False) == "roblox://experiences/start?placeId=13621938427"
    with pytest.raises(LaunchError):
        launch_url(g, private=True)
    g.private_server_link_code = "1234abcd"
    assert launch_url(g, private=True) == "roblox://placeId=13621938427&linkCode=1234abcd"
    g.private_server_link_code = "https://www.roblox.com/share?code=xyz&type=Server"
    assert launch_url(g, private=True).startswith("https://www.roblox.com/share")

    private = GameSession(1, 2, "private", True, 0.0)
    public = GameSession(1, 2, "public", True, 0.0)
    unknown = GameSession(1, 2, "unknown", False, 0.0)
    assert fight_allowed_in(private, g)[0]
    assert not fight_allowed_in(public, g)[0]
    assert not fight_allowed_in(unknown, g)[0]
    assert not fight_allowed_in(None, g)[0]
    g.allowed_servers = "public"
    assert not fight_allowed_in(public, g)[0]  # still needs the explicit ban-risk acknowledgement
    g.i_accept_ban_risk = True
    assert fight_allowed_in(unknown, g)[0]


def test_attach_unknown_server_and_persistence(tmp_path):
    plat = fake_platform(windows=FakeWindowManager([ROBLOX], foreground=ROBLOX.hwnd))
    gl = GameLauncher(plat, GameConfig(), tmp_path)
    s = gl.attach()
    assert s.server_kind == "unknown" and not s.launched_by_jarvis
    assert GameLauncher(plat, GameConfig(), tmp_path).session.pid == ROBLOX.pid
    empty = fake_platform(windows=FakeWindowManager([], None))
    with pytest.raises(LaunchError):
        GameLauncher(empty, GameConfig(), tmp_path / "x").attach()


def test_latest_frame_always_newest():
    slot = LatestFrame()
    for i in range(1, 6):
        slot.put(Frame(np.zeros((2, 2, 3), np.uint8), t_ns=i, index=i))
    f = slot.get_newer(0, timeout=0.01)
    assert f.index == 5 and slot.dropped == 4
    assert slot.get_newer(5, timeout=0.01) is None
    t = threading.Timer(0.02, lambda: slot.put(Frame(np.zeros((2, 2, 3), np.uint8), t_ns=6, index=6)))
    t.start()
    assert slot.get_newer(5, timeout=1.0).index == 6


def test_fps_meter():
    m = FpsMeter(alpha=1.0)
    for i in range(10):
        m.tick(i * 16_666_667)
    assert m.fps == pytest.approx(60, rel=0.01)
