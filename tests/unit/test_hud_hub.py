"""HUD WebSocket hub: token auth, snapshot for late joiners, throttling, commands, slow clients."""

from __future__ import annotations

import asyncio
import json

import pytest
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

from jarvis.core import Core
from jarvis.hud_bridge import hub as hubmod
from jarvis.hud_bridge.hub import HudHub


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


async def _open(hub: HudHub, token: str = "secret"):
    ws = await connect(f"ws://127.0.0.1:{hub.port}")
    await ws.send(json.dumps({"type": "auth", "token": token}))
    return ws


async def _recv(ws, timeout: float = 2.0) -> dict:
    return json.loads(await asyncio.wait_for(ws.recv(), timeout))


@pytest.fixture
async def hub():
    commands: list[dict] = []

    async def on_command(cmd):
        commands.append(cmd)
        return {"ok": True, "echo": cmd.get("text")}

    clock = Clock()
    h = HudHub("127.0.0.1", 0, "secret", on_command, clock=clock, queue_size=8)
    await h.start()
    h.commands = commands
    h.test_clock = clock
    yield h
    await h.stop()


async def test_bad_token_is_rejected(hub):
    ws = await _open(hub, token="wrong")
    with pytest.raises(ConnectionClosed) as e:
        await _recv(ws)
    assert e.value.rcvd.code == 4401


async def test_missing_auth_times_out(hub, monkeypatch):
    monkeypatch.setattr(hubmod, "AUTH_TIMEOUT_S", 0.2)
    ws = await connect(f"ws://127.0.0.1:{hub.port}")
    with pytest.raises(ConnectionClosed) as e:
        await _recv(ws)
    assert e.value.rcvd.code == 4401


async def test_hello_carries_sticky_state_then_events_flow(hub):
    hub.broadcast("voice.state", {"mode": "idle"})
    hub.broadcast("voice.transcript", {"text": "not sticky"})
    ws = await _open(hub)
    hello = await _recv(ws)
    assert hello["type"] == "hello" and hello["snapshot"] == {"voice.state": {"mode": "idle"}}
    await asyncio.sleep(0.05)
    hub.broadcast("voice.reply", {"turn": 1, "delta": "안녕하세요"})
    ev = await _recv(ws)
    assert ev == {"type": "event", "topic": "voice.reply", "data": {"turn": 1, "delta": "안녕하세요"}}
    await ws.close()


async def test_high_rate_topics_are_throttled(hub):
    ws = await _open(hub)
    await _recv(ws)
    await asyncio.sleep(0.05)
    for _ in range(10):
        hub.broadcast("fight.telemetry", {"x": 1})
    hub.test_clock.t += 0.1  # > 1/15 s later
    hub.broadcast("fight.telemetry", {"x": 2})
    got = [await _recv(ws), await _recv(ws)]
    assert [g["data"]["x"] for g in got] == [1, 2]
    with pytest.raises(TimeoutError):
        await _recv(ws, timeout=0.2)
    await ws.close()


async def test_commands_reach_core_and_get_results(hub):
    ws = await _open(hub)
    await _recv(ws)
    await ws.send(json.dumps({"type": "chat", "id": 7, "text": "hello"}))
    assert await _recv(ws) == {"type": "result", "id": 7, "ok": True, "echo": "hello"}
    await ws.send(json.dumps({"type": "format_disk", "id": 8}))
    assert await _recv(ws) == {"type": "result", "id": 8, "ok": False, "error": "unknown command"}
    assert hub.commands == [{"type": "chat", "id": 7, "text": "hello"}]
    await ws.close()


async def test_slow_client_drops_old_events_instead_of_blocking(hub):
    ws = await _open(hub)
    await _recv(ws)
    await asyncio.sleep(0.05)
    client = next(iter(hub._clients))
    for i in range(50):  # nobody reads: the bounded queue keeps only the newest
        client.offer(json.dumps({"type": "event", "topic": "voice.reply", "data": {"i": i}}))
    assert client.dropped >= 40
    await ws.close()


async def test_core_handles_hud_commands(cfg):
    core = Core(cfg, voice=False)
    core.session = core._text_session()
    await core.session.start()
    events: list[tuple[str, dict]] = []
    core.hub = None
    core.publish = lambda t, p: events.append((t, p))  # type: ignore[method-assign]
    core.session._publish = core.publish
    kills: list[str] = []
    core.kill_switch.on_kill(kills.append)
    try:
        assert await core._hud_command({"type": "mute"}) == {"ok": True} and core.session.muted == "hard"
        assert await core._hud_command({"type": "unmute"}) == {"ok": True} and core.session.muted is None
        await core._hud_command({"type": "interpreter", "on": True})
        assert core.session.interpreter and events[-1][0] == "voice.state" and events[-1][1]["interpreter"]
        await core._hud_command({"type": "stop"})
        assert kills == ["HUD stop button"]
    finally:
        await core.session.close()
