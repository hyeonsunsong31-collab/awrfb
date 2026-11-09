"""GPU guard: when core gives the GPU back to Roblox during a fight, and when it takes it again."""

from __future__ import annotations

from types import SimpleNamespace

from jarvis.gpu_guard import RESTORE, STT_OFF_GPU, VOICE_OFF_GPU, GpuGuard, GuardConfig, unload_local_models


def run(g: GpuGuard, samples, t0: float = 0.0, dt: float = 2.0):
    out = []
    for i, (fight, free, fps) in enumerate(samples):
        out += [(t0 + i * dt, a) for a in g.update(t0 + i * dt, fight, free, fps)]
    return out


def test_nothing_happens_outside_fights_or_when_healthy():
    g = GpuGuard()
    assert run(g, [(False, 100, None)] * 10) == []  # low VRAM outside a fight is not the guard's business
    assert run(g, [(True, 2000, 60.0)] * 30) == [] and g.level == 0


def test_short_dips_are_ignored():
    g = GpuGuard()
    assert run(g, [(True, 300, 60.0), (True, 300, 60.0), (True, 2000, 60.0)] * 5) == []


def test_low_vram_steps_down_one_level_at_a_time():
    g = GpuGuard()
    acts = run(g, [(True, 250, 60.0)] * 20)
    assert [a for _, a in acts] == [VOICE_OFF_GPU, STT_OFF_GPU] and g.level == 2
    (t1, _), (t2, _) = acts
    assert t1 == 4.0 and t2 - t1 >= 10.0  # three bad samples, then at least 10 s before the next step
    assert "250 MB" in g.reason


def test_frame_rate_drop_against_the_fights_own_baseline():
    g = GpuGuard(GuardConfig(baseline_s=10))
    acts = run(g, [(True, 3000, 60.0)] * 6 + [(True, 3000, 42.0)] * 6)
    assert [a for _, a in acts] == [VOICE_OFF_GPU] and "from 60" in g.reason
    g2 = GpuGuard()
    assert [a for _, a in run(g2, [(True, 3000, 25.0)] * 4)] == [VOICE_OFF_GPU]  # below the floor from the start


def test_restores_a_quiet_minute_after_the_fight():
    g = GpuGuard()
    run(g, [(True, 200, 60.0)] * 4)
    assert g.level == 1
    acts = run(g, [(False, 200, None)] * 20, t0=100.0)  # 40 s after: not yet
    assert acts == [] and g.level == 1
    acts = run(g, [(True, 3000, 60.0)] + [(False, 3000, None)] * 32, t0=200.0)  # a new fight resets the timer
    assert [a for _, a in acts] == [RESTORE] and acts[0][0] >= 202.0 + 60.0 and g.level == 0


def test_unload_local_models_is_harmless_without_a_server():
    assert unload_local_models("http://127.0.0.1:9/v1", timeout=0.5) == []


async def test_core_performs_the_steps(cfg, monkeypatch):
    from jarvis.core import Core

    core = Core(cfg, voice=False)
    calls, notices = [], []
    core.publish = lambda topic, payload: notices.append(payload["text"])
    router = SimpleNamespace(set_gpu_allowed=lambda allowed: calls.append(("gpu", allowed)) or (["qwen3"] if not allowed else []))
    core.runtime = SimpleNamespace(router=router, stt="gpu-whisper")
    core.session = SimpleNamespace(stt="gpu-whisper")
    monkeypatch.setattr("jarvis.gpu_guard.unload_local_models", lambda url, timeout=3.0: ["qwen3:4b"])
    loaded = []

    def fake_whisper(stt_cfg, models_dir):
        loaded.append(stt_cfg.device)
        return f"{stt_cfg.device}-whisper"

    monkeypatch.setattr("jarvis.voice.stt.WhisperStt.from_config", staticmethod(fake_whisper))
    core.gpu_guard.reason = "only 250 MB of VRAM free"
    await core._gpu_action(VOICE_OFF_GPU)
    assert calls == [("gpu", False)] and "qwen3, qwen3:4b" in notices[-1]
    await core._gpu_action(STT_OFF_GPU)
    assert core.session.stt == core.runtime.stt == "cpu-whisper" and core._stt_moved
    await core._gpu_action(RESTORE)
    assert calls[-1] == ("gpu", True) and loaded == ["cpu", cfg.voice.stt.device]
    assert core.session.stt == f"{cfg.voice.stt.device}-whisper" and not core._stt_moved
    assert "restored" in notices[-1]
