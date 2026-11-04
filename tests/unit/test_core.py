"""core routing without devices: intents -> fightd RPC, status/time phrasing, typed chat through the brain."""

from __future__ import annotations

import time

import numpy as np

from fakes.llm import ScriptedProvider, text
from jarvis.brain.agent import Agent
from jarvis.brain.broker import LLMBroker
from jarvis.brain.providers import build_providers
from jarvis.common.config import update_user_config
from jarvis.core import Core, NullToolBackend, clock_text
from jarvis.safety.speaker_gate import SpeakerResult
from jarvis.voice.intents import match_intent
from jarvis.voice.models import VOICE_MODELS, ModelFile, is_valid, missing, sha256_file
from jarvis.voice.session import Utterance


class FakeFightd:
    def __init__(self, replies: dict[str, dict]) -> None:
        self.replies = replies
        self.calls: list[tuple[str, dict]] = []

    async def call(self, method, params=None, timeout=None):
        self.calls.append((method, params or {}))
        return self.replies.get(method, {"ok": True})


class Reply:
    def __init__(self) -> None:
        import asyncio

        self.cancel = asyncio.Event()
        self.said: list[tuple[str, str | None]] = []
        self.written: list[str] = []

    async def say(self, text, *, key=None):
        self.said.append((text, key))

    async def write(self, delta):
        self.written.append(delta)


def utt(text: str, lang: str = "en", by: str = "wake") -> Utterance:
    return Utterance(text, text, lang, by, SpeakerResult(True, 0.9, "ok"), match_intent(text))


async def test_fight_intent_arms_fightd(cfg):
    core = Core(cfg, voice=False)
    core.fightd = FakeFightd({"arm": {"ok": True, "state": "armed"}})
    r = Reply()
    await core.handle(utt("3판 싸워", "ko"), r)
    assert core.fightd.calls == [("arm", {"matches": 3, "auto_queue": False, "origin": "wake"})]
    assert r.said == [("전투 준비 완료. 경기 시작을 기다립니다.", "armed")]


async def test_refused_fight_says_why(cfg):
    core = Core(cfg, voice=False)
    core.fightd = FakeFightd({"arm": {"ok": False, "error": "I only fight in a private server by default."}})
    r = Reply()
    await core.handle(utt("fight"), r)
    assert r.said[0][0].startswith("I can't start the fight.") and "private server" in r.said[0][0]


async def test_disarm_and_status(cfg):
    core = Core(cfg, voice=False)
    core.fightd = FakeFightd({"status": {"connected": True, "state": "disarmed"}})
    r = Reply()
    await core.handle(utt("stop the fight"), r)
    assert core.fightd.calls[0][0] == "disarm" and r.said == [("Standing down.", "disarmed")]
    core.last_match = {"stocks_won": 3, "stocks_lost": 1, "result": "likely_won"}
    r = Reply()
    await core.handle(utt("fight status"), r)
    assert r.said[0][0] == "Last match: 3-1 in stocks, likely a win. Current state: disarmed."
    assert core.status_text({"connected": False}, "ko").startswith("지난 경기는 스톡 3-1, 승리로 보입니다.")


def test_clock_text_both_languages():
    t = time.struct_time((2026, 11, 3, 14, 5, 0, 1, 307, 0))
    assert clock_text("ko", t) == "지금은 오후 2시 5분입니다."
    assert clock_text("en", t) == "It's 2:05 PM."
    assert clock_text("ko", time.struct_time((2026, 11, 3, 0, 0, 0, 1, 307, 0))) == "지금은 오전 12시입니다."


async def test_typed_chat_runs_an_agent_turn(cfg):
    core = Core(cfg, voice=False)
    nv = ScriptedProvider("nvidia", [text("[]"), text("Hello", " there, sir.")])  # classifier, then chat
    core.broker = LLMBroker(cfg.brain, {"nvidia": nv})
    core.agent = Agent(core.broker, NullToolBackend(), cfg.brain, memory=core.memory)
    core.session = core._text_session()
    await core.session.start()
    try:
        res = await core.rpc_chat({"text": "say hello", "speak": False})
    finally:
        await core.session.close()
    assert res == {"ok": True, "text": "Hello there, sir."}
    assert core.agent.conv.messages[-1]["content"] == "Hello there, sir."


def test_build_providers_skips_cloud_without_key(cfg, monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setattr("jarvis.brain.providers.resolve_api_key", lambda ep: None)
    assert set(build_providers(cfg.brain)) == {"local"}
    monkeypatch.setattr("jarvis.brain.providers.resolve_api_key", lambda ep: "nvapi-test")
    assert set(build_providers(cfg.brain)) == {"nvidia", "local"}


def test_user_config_update_merges(cfg):
    p = update_user_config(cfg.paths(), {"voice": {"tts": {"qwen_voice_ref_text": "hello"}}})
    update_user_config(cfg.paths(), {"voice": {"tts": {"qwen_voice_ref_wav": "a.wav"}}})
    import yaml

    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert data["voice"]["tts"] == {"qwen_voice_ref_text": "hello", "qwen_voice_ref_wav": "a.wav"}


def test_model_manifest_checks_hashes(tmp_path):
    payload = np.arange(100, dtype=np.uint8).tobytes()
    p = tmp_path / "x" / "m.onnx"
    p.parent.mkdir()
    p.write_bytes(payload)
    m = ModelFile("x/m.onnx", "https://example.invalid/m.onnx", sha256_file(p), len(payload))
    assert is_valid(tmp_path, m) and missing(tmp_path, (m,)) == []
    p.write_bytes(payload[:-1] + b"\x00")
    assert not is_valid(tmp_path, m)
    assert {f.rel for f in VOICE_MODELS} == {"silero_vad.onnx", "openwakeword/hey_jarvis_v0.1.onnx",
                                            "openwakeword/melspectrogram.onnx", "openwakeword/embedding_model.onnx"}
