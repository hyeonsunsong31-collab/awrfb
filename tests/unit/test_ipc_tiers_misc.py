from __future__ import annotations

import asyncio
import socket

import numpy as np
import pytest

from jarvis.common.ipc import (
    AsyncSubscriber,
    BusProxy,
    Publisher,
    RpcClient,
    RpcError,
    RpcServer,
    load_or_create_token,
)
from jarvis.common.tiers import choose_tier
from jarvis.platform.base import PlatformUnavailable
from jarvis.safety.killswitch import KillSwitch
from jarvis.safety.speaker_gate import SpeakerGate


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def test_rpc_roundtrip_and_auth():
    port = free_port()
    srv = RpcServer("t", port, "secret")

    @srv.method("add")
    async def add(p):
        return p["a"] + p["b"]

    srv.add("boom", lambda p: 1 / 0)
    await srv.start()
    good = RpcClient(port, "secret")
    bad = RpcClient(port, "wrong")
    try:
        assert await good.call("add", {"a": 2, "b": 3}) == 5
        with pytest.raises(RpcError, match="ZeroDivisionError"):
            await good.call("boom")
        with pytest.raises(RpcError, match="unknown method"):
            await good.call("nope")
        with pytest.raises(RpcError, match="bad token"):
            await bad.call("add", {"a": 1, "b": 1})
    finally:
        await good.close()
        await bad.close()
        await srv.close()


async def test_event_bus():
    pin, pout = free_port(), free_port()
    proxy = BusProxy(pin, pout)
    proxy.start()
    sub = AsyncSubscriber(pout, ["fight."])
    pub = Publisher(pin)
    try:
        got = None
        for _ in range(50):  # PUB/SUB slow-joiner: keep publishing until the subscriber is connected
            pub.publish("fight.telemetry", {"hp": 0.5})
            pub.publish("voice.state", {"x": 1})
            try:
                got = await asyncio.wait_for(sub.recv(), 0.1)
                break
            except TimeoutError:
                continue
        assert got == ("fight.telemetry", {"hp": 0.5})
    finally:
        pub.close()
        sub.close()
        proxy.stop()


def test_token_file(tmp_path, monkeypatch):
    monkeypatch.delenv("JARVIS_IPC_TOKEN", raising=False)
    t1 = load_or_create_token(tmp_path)
    assert len(t1) > 20 and load_or_create_token(tmp_path) == t1


def test_tiers():
    assert choose_tier(6144).tier == "t6"
    assert choose_tier(8192).tier == "t8"
    assert choose_tier(12288).tier == "t12"
    assert choose_tier(12288, free_mb_with_roblox=3000).tier == "t6"
    assert choose_tier(8192, free_mb_with_roblox=4200).tier == "t8"


def test_killswitch_runs_all_callbacks():
    ks = KillSwitch(on_broadcast=lambda m: calls.append(("bcast", m["reason"])))
    calls: list = []
    ks.on_kill(lambda r: calls.append(("a", r)))
    ks.on_kill(lambda r: 1 / 0)  # a failing callback must not stop the rest
    ks.on_kill(lambda r: calls.append(("b", r)))
    ks.trigger("hotkey")
    assert calls == [("a", "hotkey"), ("b", "hotkey"), ("bcast", "hotkey")]


class FixedEmbedder:
    def __init__(self, vec):
        self.vec = np.asarray(vec, dtype=np.float32)

    def embed(self, audio):
        return self.vec


def test_speaker_gate(tmp_path):
    owner = np.array([[1.0, 0.0, 0.0], [0.9, 0.1, 0.0]], dtype=np.float32)
    gate = SpeakerGate(FixedEmbedder([1, 0.05, 0]), owner, threshold=0.8, min_audio_s=1.0)
    audio = np.zeros(16000 * 2, dtype=np.float32)
    assert gate.verify(audio).is_owner
    stranger = SpeakerGate(FixedEmbedder([0, 1, 0]), owner, threshold=0.8, min_audio_s=1.0)
    assert not stranger.verify(audio).is_owner
    assert not gate.verify(audio[:8000]).is_owner  # too short to verify
    not_enrolled = SpeakerGate(None, None)
    r = not_enrolled.verify(audio)
    assert r.is_owner and "not enrolled" in r.reason
    p = tmp_path / "owner.npz"
    SpeakerGate.save_enrollment(p, owner)
    assert np.allclose(SpeakerGate.load_enrollment(p), owner)


def test_factory_refuses_fakes_outside_test_mode(monkeypatch):
    from fakes.platform import fake_platform
    from jarvis.platform import factory

    monkeypatch.setenv("JARVIS_TEST", "0")
    with pytest.raises(PlatformUnavailable):
        factory.register_test_platform(fake_platform())
    factory._cached = None
    with pytest.raises(PlatformUnavailable):
        factory.get_platform()
    monkeypatch.setenv("JARVIS_TEST", "1")
    p = fake_platform()
    factory.register_test_platform(p)
    assert factory.get_platform() is p
    factory.register_test_platform(None)
