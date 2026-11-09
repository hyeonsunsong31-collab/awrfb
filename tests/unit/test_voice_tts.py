"""TTS routing: tier engine -> fallback chain, cached phrases, and the in-fight cache-only rule."""

from __future__ import annotations

import numpy as np

from jarvis.common.config import TtsConfig
from jarvis.voice.tts.base import Audio, TtsError
from jarvis.voice.tts.cache import TtsCache
from jarvis.voice.tts.callouts import PHRASES, callout_for_event
from jarvis.voice.tts.router import TtsRouter


class FakeEngine:
    def __init__(self, name: str, fail: bool = False) -> None:
        self.name = name
        self.fail = fail
        self.calls: list[tuple[str, str]] = []

    def synth(self, text: str, lang: str) -> Audio:
        self.calls.append((text, lang))
        if self.fail:
            raise RuntimeError("synth failed")
        return Audio(np.full(160, 0.1, np.float32), 16000)


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def _router(tmp_path, engines: dict[str, FakeEngine | Exception], **cfg):
    def factory(item):
        def make():
            if isinstance(item, Exception):
                raise item
            return item

        return make

    conf = TtsConfig(**{"engine": "qwen3", "fallback_chain": ["qwen3", "supertonic", "edge"], **cfg})
    clock = Clock()
    r = TtsRouter(conf, TtsCache(tmp_path / "tts"), {n: factory(e) for n, e in engines.items()}, clock=clock)
    return r, clock


def test_falls_back_when_the_tier_engine_cannot_load(tmp_path):
    sup = FakeEngine("supertonic")
    r, _ = _router(tmp_path, {"qwen3": TtsError("no reference voice"), "supertonic": sup, "edge": FakeEngine("edge")})
    s = r.synth("A longer sentence that is not cached because of its length.", "en")
    assert s is not None and s.engine == "supertonic" and not s.cached
    assert "qwen3" in (r.last_error or "")


def test_falls_back_when_synthesis_fails_and_retries_later(tmp_path):
    sup, edge = FakeEngine("supertonic", fail=True), FakeEngine("edge")
    r, clock = _router(tmp_path, {"supertonic": sup, "edge": edge}, engine="supertonic", fallback_chain=["supertonic", "edge"])
    long = "This sentence is long enough that the router will not cache it at all."
    assert r.synth(long, "en").engine == "edge"
    assert r.synth(long + " Again.", "en").engine == "edge" and len(sup.calls) == 1  # skipped while cooling down
    clock.t += 61
    sup.fail = False
    assert r.synth(long + " Third.", "en").engine == "supertonic"


def test_short_phrases_are_cached(tmp_path):
    sup = FakeEngine("supertonic")
    r, _ = _router(tmp_path, {"supertonic": sup}, engine="supertonic", fallback_chain=["supertonic"])
    assert r.synth("네, 주인님.", "ko").cached is False
    again = r.synth("네, 주인님.", "ko")
    assert again.cached and again.engine == "cache" and len(sup.calls) == 1


def test_fight_cache_only_speaks_only_prerendered_phrases(tmp_path):
    sup = FakeEngine("supertonic")
    r, _ = _router(tmp_path, {"supertonic": sup}, engine="supertonic", fallback_chain=["supertonic"], fight_engine="cache_only")
    assert r.synth("Something new to say during the match.", "en", fighting=True) is None
    assert sup.calls == []  # nothing synthesized while fighting
    assert r.prerender(keys=["armed", "stock_won"], langs=["ko", "en"]) == 4
    n = len(sup.calls)
    hit = r.phrase("stock_won", "ko", fighting=True)
    assert hit is not None and hit.cached and len(sup.calls) == n
    assert r.prerender(keys=["armed"]) == 0  # already cached


def test_voice_change_invalidates_cached_phrases(tmp_path):
    r, _ = _router(tmp_path, {"supertonic": FakeEngine("supertonic")}, engine="supertonic", fallback_chain=["supertonic"])
    k1 = r.phrase_key("armed", "en")
    r.cfg.supertonic_voice = "F2"
    assert r.phrase_key("armed", "en") != k1


def test_cache_roundtrip_and_prune(tmp_path):
    c = TtsCache(tmp_path / "c", max_mb=0.001)
    a = Audio(np.linspace(-1, 1, 400, dtype=np.float32), 24000)
    c.put("k1", a)
    got = c.get("k1")
    assert got is not None and got.rate == 24000 and np.allclose(got.pcm, a.pcm)
    for i in range(5):
        c.put(f"k{i + 2}", a)
    c.prune()
    assert sum(f.stat().st_size for f in (tmp_path / "c").glob("*.npz")) <= 1100


def test_callout_mapping_covers_fightd_events():
    assert callout_for_event("match_end", {"result": "likely_won"}) == "match_won"
    assert callout_for_event("match_end", {"result": "lost"}) == "match_lost"
    assert callout_for_event("match_end", {"result": "unknown"}) == "match_over"
    assert callout_for_event("final_stock_empty", {"side": "own"}) == "final_stock_own"
    assert callout_for_event("final_stock_empty", {"side": "opp"}) == "final_stock_opp"
    assert callout_for_event("armed", {}) == "armed"
    assert callout_for_event("something_else", {}) is None
    assert all({"ko", "en"} <= set(v) for v in PHRASES.values())


class ClosableEngine(FakeEngine):
    def __init__(self, name: str, fail: bool = False) -> None:
        super().__init__(name, fail)
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_fallback_audio_is_cached_as_the_voice_that_spoke(tmp_path):
    qwen, sup = FakeEngine("qwen3", fail=True), FakeEngine("supertonic")
    r, clock = _router(tmp_path, {"qwen3": qwen, "supertonic": sup, "edge": FakeEngine("edge")})
    assert r.synth("네, 주인님.", "ko").engine == "supertonic"  # qwen3 failed: the CPU voice spoke
    clock.t += 61
    qwen.fail = False
    again = r.synth("네, 주인님.", "ko")  # the configured voice works again: it speaks, not the cached fallback
    assert again.engine == "qwen3" and not again.cached
    assert r.synth("네, 주인님.", "ko").cached


def test_gpu_voice_can_be_unloaded_for_a_fight_and_restored(tmp_path):
    qwen, sup = ClosableEngine("qwen3"), FakeEngine("supertonic")
    r, _ = _router(tmp_path, {"qwen3": qwen, "supertonic": sup, "edge": FakeEngine("edge")}, fight_engine="cache_only")
    assert r.prerender(keys=["stock_won"], langs=["en"]) == 1 and qwen.calls  # rendered in the GPU voice
    assert r.synth("A long sentence spoken before the fight by the GPU voice.", "en").engine == "qwen3"
    assert r.set_gpu_allowed(False) == ["qwen3"] and qwen.closed
    assert "qwen3" not in r.chain() and r.primary() == "qwen3"  # identity stays the configured voice
    assert r.synth("Another long sentence while the GPU voice is unloaded.", "en").engine == "supertonic"
    hit = r.phrase("stock_won", "en", fighting=True)  # the pre-rendered GPU-voice callout still plays
    assert hit is not None and hit.cached
    n = len(qwen.calls)
    r.set_gpu_allowed(True)
    assert "qwen3" in r.chain()
    assert r.synth("Back to the GPU voice after the fight, with a long sentence.", "en").engine == "qwen3"
    assert len(qwen.calls) == n + 1
