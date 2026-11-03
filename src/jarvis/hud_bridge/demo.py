"""Demo hub: drives the HUD with a scripted sequence (HUD development and the end-to-end test).

    python -m jarvis.hud_bridge.demo --port 47839 --token demo

Every command the HUD sends is printed to stdout as one JSON line. Nothing here touches Roblox or the
microphone; it only exercises the HUD protocol.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys

from jarvis.hud_bridge.hub import HudHub

SCRIPT: list[tuple[float, str, dict]] = [
    (0.0, "voice.state", {"mode": "idle", "muted": None, "interpreter": False, "lang": "ko"}),
    (0.0, "voice.notes", {"notes": ["demo mode - no microphone, no Roblox"]}),
    (0.0, "system.stats", {"cpu": 23.0, "ram_used_gb": 9.4, "ram_total_gb": 31.8,
                           "gpu": {"name": "NVIDIA GeForce RTX (demo)", "util": 41, "vram_used_mb": 3900, "vram_total_mb": 8192}}),
    (0.4, "voice.wake", {"score": 0.93}),
    (0.6, "voice.state", {"mode": "listening", "muted": None, "interpreter": False, "lang": "ko"}),
    (1.2, "voice.transcript", {"text": "자비스, 3판 싸워", "lang": "ko", "addressed_by": "wake", "intent": "fight"}),
    (1.3, "voice.state", {"mode": "thinking", "muted": None, "interpreter": False, "lang": "ko"}),
    (1.5, "voice.reply", {"turn": 1, "delta": "전투 준비 완료. "}),
    (1.6, "voice.state", {"mode": "speaking", "muted": None, "interpreter": False, "lang": "ko"}),
    (1.7, "voice.reply", {"turn": 1, "delta": "경기 시작을 기다립니다."}),
    (2.2, "voice.reply_done", {"turn": 1, "text": "전투 준비 완료. 경기 시작을 기다립니다.", "heard": None, "cancelled": False}),
    (2.3, "voice.state", {"mode": "idle", "muted": None, "interpreter": False, "lang": "ko"}),
    (2.5, "fight.state", {"old": "disarmed", "new": "fighting", "reason": "match started"}),
]


def telemetry(t: float) -> dict:
    return {
        "state": "fighting", "gate_open": True, "gate": "ok", "combat_hud": True,
        "hud": {"own_hp": 0.55 + 0.4 * math.cos(t / 3), "opp_hp": 0.5 + 0.45 * math.sin(t / 4), "own_stamina": 0.5 + 0.5 * math.sin(t),
                "own_guard": 0.8},
        "threat": {"100": 0.1, "200": 0.2 + 0.2 * math.sin(t * 2) ** 2, "300": 0.35, "500": 0.5},
        "gap": 0.4 + 0.2 * math.sin(t), "opp": [0.6, 0.55, 0.08, 0.3], "me": [0.4, 0.6, 0.08, 0.32],
        "fps": 59.8, "perception_ms": 6.1, "tactic": "counter",
        "opponent": {"aggression": 0.7, "defense": 0.4, "punish": 0.55, "range_pref": 0.3, "regen": 0.2},
        "inputs": {"press": 120, "denied": 0}, "match": {"won": 1, "lost": 0},
    }


async def run(port: int, token: str, loop_forever: bool) -> None:
    async def on_command(cmd: dict) -> dict:
        print(json.dumps(cmd, ensure_ascii=False), flush=True)
        if cmd.get("type") == "chat":
            hub.broadcast("voice.transcript", {"text": cmd.get("text", ""), "lang": "en", "addressed_by": "typed", "intent": None})
            hub.broadcast("voice.reply", {"turn": 99, "delta": f"(demo) you typed: {cmd.get('text', '')}"})
            hub.broadcast("voice.reply_done", {"turn": 99, "text": "", "heard": None, "cancelled": False})
        return {"ok": True}

    hub = HudHub("127.0.0.1", port, token, on_command, rates_hz={"fight.telemetry": 15.0})
    await hub.start()
    print(json.dumps({"ready": True, "port": hub.port}), flush=True)
    while True:
        while hub.clients == 0:
            await asyncio.sleep(0.05)
        t0 = asyncio.get_running_loop().time()
        for at, topic, data in SCRIPT:
            await asyncio.sleep(max(0.0, t0 + at - asyncio.get_running_loop().time()))
            hub.broadcast(topic, data)
        for i in range(400 if not loop_forever else 10**9):
            t = i / 15.0
            hub.broadcast("fight.telemetry", telemetry(t))
            hub.broadcast("voice.level", {"mic": abs(math.sin(t * 3)) * 0.1, "out": abs(math.sin(t * 5)) * 0.2})
            await asyncio.sleep(1 / 15)
        if not loop_forever:
            return


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=47839)
    ap.add_argument("--token", default="demo")
    ap.add_argument("--forever", action="store_true")
    a = ap.parse_args()
    try:
        asyncio.run(run(a.port, a.token, a.forever))
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
