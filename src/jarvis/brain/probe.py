"""Startup probe: which configured models answer, support native tool calls and images, and how fast.
Cached for ``probe_cache_hours`` in ``run/probe.json`` (each probe costs a few requests)."""

from __future__ import annotations

import base64
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from jarvis.brain.broker import ProbeInfo
from jarvis.brain.providers import Provider, ProviderError
from jarvis.common.config import BrainConfig

PING_TOOL = {
    "type": "function",
    "function": {
        "name": "report_number",
        "description": "Report a number.",
        "parameters": {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]},
    },
}


@dataclass
class ProbeResult:
    key: str
    ok: bool
    tools: bool | None = None
    vision: bool | None = None
    ttft_ms: float | None = None
    tokens_per_s: float | None = None
    korean_ok: bool | None = None
    error: str = ""


def _tiny_png() -> str:
    img = np.zeros((64, 64, 3), np.uint8)
    img[:, :32] = (0, 0, 255)  # left half red, right half black
    _ok, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


async def _timed_text(p: Provider, model: str, messages, **kw) -> tuple[str, float | None, float | None, list]:
    t0 = time.perf_counter()
    first = None
    text = []
    calls = []
    n_chunks = 0
    async for d in p.stream(model, messages, **kw):
        if d.kind == "text":
            if first is None:
                first = time.perf_counter()
            text.append(d.text)
            n_chunks += 1
        elif d.kind == "done":
            calls = d.tool_calls
    t1 = time.perf_counter()
    ttft = None if first is None else (first - t0) * 1000
    tps = None if first is None or t1 <= first else n_chunks / (t1 - first)
    return "".join(text), ttft, tps, calls


async def probe_model(p: Provider, model: str) -> ProbeResult:
    key = f"{p.name}:{model}"
    r = ProbeResult(key, ok=False)
    try:
        text, r.ttft_ms, r.tokens_per_s, _ = await _timed_text(
            p, model, [{"role": "user", "content": "한국어로 한 문장만: 오늘 기분 어때? Then one English sentence."}],
            max_tokens=80, timeout=40)
        r.ok = bool(text.strip())
        r.korean_ok = any("가" <= ch <= "힣" for ch in text)
    except ProviderError as e:
        r.error = f"{e.kind}: {e}"[:200]
        return r
    try:
        _, _, _, calls = await _timed_text(
            p, model, [{"role": "user", "content": "Call report_number with n=7."}], tools=[PING_TOOL], max_tokens=60, timeout=40)
        r.tools = any(c.name == "report_number" and c.args().get("n") == 7 for c in calls)
    except ProviderError as e:
        r.tools = False if e.kind in ("unsupported", "bad_request") else None
    try:
        msg = [{"role": "user", "content": [
            {"type": "text", "text": "What color is the left half of this image? One word."},
            {"type": "image_url", "image_url": {"url": _tiny_png()}},
        ]}]
        text, _, _, _ = await _timed_text(p, model, msg, max_tokens=20, timeout=40)
        r.vision = "red" in text.lower() or "빨" in text
    except ProviderError as e:
        r.vision = False if e.kind in ("unsupported", "bad_request") else None
    return r


def load_cache(path: Path, max_age_h: float) -> dict[str, ProbeResult] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if time.time() - data.get("t", 0) > max_age_h * 3600:
        return None
    return {k: ProbeResult(**v) for k, v in data.get("results", {}).items()}


def save_cache(path: Path, results: dict[str, ProbeResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"t": time.time(), "results": {k: asdict(v) for k, v in results.items()}}, indent=2), encoding="utf-8")


async def probe_all(cfg: BrainConfig, providers: dict[str, Provider]) -> dict[str, ProbeResult]:
    keys = {item for role in cfg.roles.values() for item in role.chain}
    out = {}
    for key in sorted(keys):
        ep, _, model = key.partition(":")
        if ep in providers:
            out[key] = await probe_model(providers[ep], model)
    return out


def to_probe_info(results: dict[str, ProbeResult]) -> dict[str, ProbeInfo]:
    return {k: ProbeInfo(tools=v.tools, vision=v.vision) for k, v in results.items() if v.ok}
