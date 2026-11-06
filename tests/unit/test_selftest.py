"""jarvis selftest: the checks' verdicts with stand-in engines, the real IPC/policy/audit checks, and the
live-game path against a simulated window (plumbing only - the real run happens on the owner's PC)."""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

from jarvis.safety.speaker_gate import SpeakerResult
from jarvis.selftest import (
    FAIL,
    PASS,
    SENTENCES,
    SKIP,
    WARN,
    WS_EX_LAYERED,
    WS_EX_NOACTIVATE,
    WS_EX_TOPMOST,
    WS_EX_TRANSPARENT,
    Check,
    Ctx,
    Result,
    hud_flags,
    hud_verdict,
    run_selftest,
    save_report,
    similarity,
    summary,
)
from jarvis.voice.stt import SttResult
from jarvis.voice.tts.base import Audio

LENGTHS = {"ko": 32000, "en": 24000, "wake": 12000}


def tone(n: int, amp: float = 0.3) -> np.ndarray:
    t = np.arange(n) / 16000.0
    return (amp * np.sin(2 * np.pi * 220 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)


class FakeEngine:
    name = "fake"

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    def synth(self, text: str, lang: str) -> Audio:
        if self.fail:
            raise RuntimeError("model download blocked")
        key = "wake" if "Hey Jarvis" in text else lang
        return Audio(tone(LENGTHS[key]), 16000)


class FakeRouter:
    def __init__(self, fail: bool = False) -> None:
        self.eng = FakeEngine(fail)

    def chain(self):
        return ["supertonic"]

    def engine(self, name):
        return self.eng


class FakeStt:
    def __init__(self, heard: dict[str, tuple[str, str]] | None = None) -> None:
        self.heard = heard or {"ko": (SENTENCES["ko"], "ko"), "en": (SENTENCES["en"], "en")}

    def transcribe(self, pcm, **kw):
        lang = next(k for k, n in LENGTHS.items() if n == len(pcm))
        text, detected = self.heard[lang]
        return SttResult(text=text, lang=detected, elapsed_s=0.2)


class FakeWake:
    """Fires when a loud stretch ends that is wake-phrase sized (or after any stretch with fire_on_anything)."""

    def __init__(self, fire_on_anything: bool = False) -> None:
        self.any = fire_on_anything
        self.run = 0
        self.last_score = 0.0

    def feed(self, chunk):
        loud = len(chunk) > 0 and float(np.sqrt(np.mean(np.square(chunk)))) > 0.05
        if loud:
            self.run += len(chunk)
            self.last_score = 0.3
            return []
        fired = self.run > 0 and (self.any or self.run < 20000)
        self.run = 0
        self.last_score = 0.9 if fired else 0.0
        return [0.9] if fired else []


class FakeVad:
    def reset(self):
        pass

    def prob(self, chunk):
        return 1.0 if float(np.sqrt(np.mean(np.square(chunk)))) > 0.01 else 0.0


class FakeGate:
    enabled, enrolled, threshold = True, True, 0.45

    def __init__(self, owner: bool = False) -> None:
        self.owner = owner

    def verify(self, clip):
        return SpeakerResult(self.owner, 0.8 if self.owner else 0.12, "")


def voice_ctx(cfg, **over) -> Ctx:
    f = {"router": lambda c: FakeRouter(), "stt": lambda c: FakeStt(), "wake": lambda c: FakeWake(),
         "vad": lambda c, notes: FakeVad(), "gate": lambda c, notes: FakeGate()}
    f.update(over)
    return Ctx(cfg, factories=f, say=lambda t: None)


VOICE = {"tts", "stt", "wake_word", "vad", "speaker_gate"}


def by_name(results):
    return {r.name: r for r in results}


def test_similarity_ignores_spacing_and_punctuation():
    assert similarity("안녕하세요, 저는 자비스입니다.", "안녕하세요 저는 자비스 입니다") == 100
    assert similarity("Hello, I am Jarvis.", "hello i am jarvis") == 100
    assert similarity(SENTENCES["ko"], "오늘 날씨가 좋네요") < 40


def test_voice_chain_passes_with_working_engines(cfg):
    r = by_name(run_selftest(voice_ctx(cfg), VOICE))
    assert {n: r[n].status for n in VOICE} == {n: PASS for n in VOICE}, {n: r[n].detail for n in VOICE}
    assert r["stt"].data["ko"]["similarity"] == 100 and r["tts"].data["ko"]["engine"] == "supertonic"


def test_misheard_or_wrong_language_is_reported(cfg):
    bad = voice_ctx(cfg, stt=lambda c: FakeStt({"ko": ("오늘 날씨가 좋네요", "ko"), "en": (SENTENCES["en"], "en")}))
    assert by_name(run_selftest(bad, VOICE))["stt"].status == FAIL
    wrong_lang = voice_ctx(cfg, stt=lambda c: FakeStt({"ko": (SENTENCES["ko"], "ja"), "en": (SENTENCES["en"], "en")}))
    assert by_name(run_selftest(wrong_lang, VOICE))["stt"].status == WARN


def test_broken_tts_fails_and_dependents_skip(cfg):
    r = by_name(run_selftest(voice_ctx(cfg, router=lambda c: FakeRouter(fail=True)), VOICE))
    assert r["tts"].status == FAIL and "model download blocked" in r["tts"].detail
    assert r["stt"].status == r["wake_word"].status == r["vad"].status == SKIP


def test_synthetic_voice_passing_as_owner_fails(cfg):
    r = by_name(run_selftest(voice_ctx(cfg, gate=lambda c, n: FakeGate(owner=True)), VOICE))
    assert r["speaker_gate"].status == FAIL and "re-run `jarvis enroll`" in r["speaker_gate"].detail


def test_wake_word_that_fires_on_anything_warns(cfg):
    r = by_name(run_selftest(voice_ctx(cfg, wake=lambda c: FakeWake(fire_on_anything=True)), VOICE))
    assert r["wake_word"].status == WARN and r["wake_word"].data["false_fire_on_other_speech"]


def test_framework_skips_and_catches(cfg):
    def boom(ctx):
        raise ValueError("kaput")

    checks = (Check("win", lambda c: Result("win", PASS), "w", windows_only=True),
              Check("live", lambda c: Result("live", PASS), "l", live=True),
              Check("me", lambda c: Result("me", PASS), "i", interactive=True),
              Check("boom", boom, "b"),
              Check("many", lambda c: [Result("a", PASS), Result("b", WARN)], "m"))
    ctx = Ctx(cfg, windows=False)
    r = by_name(run_selftest(ctx, checks=checks))
    assert r["win"].status == r["live"].status == r["me"].status == SKIP and "--live" in r["live"].detail
    assert r["boom"].status == FAIL and "kaput" in r["boom"].detail and "traceback" in r["boom"].data
    assert r["a"].status == PASS and r["b"].status == WARN
    assert summary(list(r.values())) == {"pass": 1, "warn": 1, "fail": 1, "skip": 3}


def test_real_ipc_policy_and_audit_checks(cfg):
    from jarvis.common.audit import AuditLog

    r = by_name(run_selftest(Ctx(cfg), {"ipc", "safety_policy", "audit"}))
    assert r["ipc"].status == PASS
    assert r["safety_policy"].status in (PASS, WARN) and len(r["safety_policy"].data["refused"]) == 4
    assert r["audit"].status == SKIP
    log = AuditLog(cfg.paths().ensure().audit, "selftest", background=False)
    for i in range(3):
        log.append("tool_call", {"i": i})
    log.close()
    assert by_name(run_selftest(Ctx(cfg), {"audit"}))["audit"].status == PASS
    f = next(cfg.paths().audit.glob("selftest-*.jsonl"))
    lines = f.read_text(encoding="utf-8").splitlines()
    lines[1] = lines[1].replace('"i":1', '"i":9')
    assert '"i":9' in lines[1]
    f.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert by_name(run_selftest(Ctx(cfg), {"audit"}))["audit"].status == FAIL


def test_hud_window_flags_and_verdict():
    good = hud_flags(WS_EX_TOPMOST | WS_EX_NOACTIVATE | WS_EX_TRANSPARENT | WS_EX_LAYERED)
    assert all(good.values())
    assert hud_verdict(True, good, True)[0] == PASS
    assert hud_verdict(False, None, None)[0] == FAIL
    assert hud_verdict(True, hud_flags(WS_EX_TOPMOST | WS_EX_LAYERED), True) == (FAIL, "HUD window is missing: never_takes_focus, click_through")
    assert hud_verdict(True, good, False)[0] == FAIL
    assert hud_verdict(True, None, None)[0] == WARN


def test_report_is_saved_for_jarvis_report(cfg):
    ctx = Ctx(cfg)
    p = save_report([Result("ipc", PASS, "ok"), Result("tts", FAIL, "자비스 voice missing")], ctx)
    d = json.loads(p.read_text(encoding="utf-8"))
    assert p.parent == cfg.paths().reports and d["summary"] == {"pass": 1, "warn": 0, "fail": 1, "skip": 0}
    assert d["results"][1]["detail"] == "자비스 voice missing"


def test_live_game_checks_against_a_simulated_window(data_dir):
    from fakes.platform import ROBLOX, FakeWindowManager, fake_platform
    from fakes.sim_game import SimFight, SimFrameSource, SimSink
    from jarvis.common.config import load_config
    from jarvis.fight.gameio import GameIO

    cfg = load_config(environ={})
    game = SimFight(size=(960, 540))
    plat = fake_platform(windows=FakeWindowManager([ROBLOX], foreground=ROBLOX.hwnd),
                         new_input_sink=lambda forbidden_keys=(): SimSink(game, forbidden_keys),
                         new_frame_source=lambda **kw: SimFrameSource(game, **kw))
    ctx = Ctx(cfg, live=True, windows=True, factories={"gameio": lambda: GameIO(cfg, plat)})
    t0 = time.monotonic()
    r = by_name(run_selftest(ctx, {"game"}))
    assert time.monotonic() - t0 < 30
    assert r["roblox_window"].status == PASS
    assert r["capture"].status in (PASS, WARN) and r["capture"].data["fps"] > 20
    assert r["no_input_when_disarmed"].status == PASS and r["no_input_when_disarmed"].data["denied"] >= 1
    assert game.presses == []  # nothing reached the "game"
    assert r["hud_reader"].status == WARN and "calibrate" in r["hud_reader"].detail
    assert r["fight_preflight"].status == WARN and r["fight_preflight"].data["problems"]


@pytest.mark.parametrize("bad", ["bogus", "doctor,bogus"])
def test_cli_rejects_unknown_checks(bad, data_dir):
    from typer.testing import CliRunner

    from jarvis.cli import _register_subcommands, app

    _register_subcommands()
    res = CliRunner().invoke(app, ["selftest", "--only", bad])
    assert res.exit_code == 2 and "unknown check" in res.output
