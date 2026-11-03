"""core process: the voice session, the LLM brain and command routing.

* Local intents go straight to their owner: fight/disarm/launch/status -> fightd (RPC, token-protected;
  "fight" only arrives here after owner verification or from the keyboard), volume -> Windows endpoint
  volume, time -> local clock. Everything else is an agent turn on the brain.
* Kill switch: stop speech, cancel the agent, tell fightd to panic, broadcast ``kill`` on the bus.
* RPC for the HUD/CLI: status, chat (typed TTT), mute/unmute, stop.
* Bus: publishes voice.* events; consumes fight.* (state, callouts spoken from the cache) and kill.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import socket
import sys
import time
from typing import Any

from jarvis.brain.agent import Agent, AgentTurn, ToolOutcome
from jarvis.brain.broker import LLMBroker
from jarvis.brain.memory import MemoryStore
from jarvis.brain.probe import load_cache, to_probe_info
from jarvis.brain.providers import build_providers
from jarvis.brain.scope import scope_request
from jarvis.common.config import JarvisConfig
from jarvis.common.ipc import (
    AsyncSubscriber,
    BusProxy,
    Publisher,
    RpcClient,
    RpcError,
    RpcServer,
    RpcTimeout,
    load_or_create_token,
)
from jarvis.safety.capabilities import Cap
from jarvis.safety.killswitch import KillSwitch
from jarvis.safety.speaker_gate import SpeakerGate
from jarvis.voice.session import ReplySink, SessionOptions, Utterance, VoiceSession
from jarvis.voice.tts.callouts import phrase

log = logging.getLogger(__name__)

ACTIVE_FIGHT_STATES = {"armed", "navigating", "queued", "fighting", "post_match", "calibrating"}
STATE_KO = {"disarmed": "대기", "armed": "전투 준비", "navigating": "링으로 이동 중", "queued": "경기 대기", "fighting": "전투 중",
            "post_match": "경기 종료 처리 중", "calibrating": "보정 중", "detached": "연결 안 됨"}


class NullToolBackend:
    async def schemas(self, caps: set[Cap]) -> list[dict]:
        return []

    async def open_scope(self, task_id: str, caps: set[Cap], origin: str, request: str) -> None:
        return None

    async def call(self, task_id: str, name: str, args: dict[str, Any]) -> ToolOutcome:
        return ToolOutcome(False, "PC tools are not running yet")


class RpcToolBackend:
    """toolhost over RPC. While toolhost is down the agent simply has no tools (plain conversation)."""

    def __init__(self, client: RpcClient) -> None:
        self.client = client
        self._down_until = 0.0

    def _down(self) -> bool:
        return time.monotonic() < self._down_until

    async def schemas(self, caps: set[Cap]) -> list[dict]:
        if self._down():
            return []
        try:
            return list(await self.client.call("schemas", {"caps": sorted(c.value for c in caps)}, timeout=2.0) or [])
        except RpcError:
            self._down_until = time.monotonic() + 30.0
            return []

    async def open_scope(self, task_id: str, caps: set[Cap], origin: str, request: str) -> None:
        if self._down():
            return
        with contextlib.suppress(RpcError):
            await self.client.call("open_scope", {"task_id": task_id, "caps": sorted(c.value for c in caps),
                                                  "origin": origin, "request": request}, timeout=2.0)

    async def call(self, task_id: str, name: str, args: dict[str, Any]) -> ToolOutcome:
        try:
            r = await self.client.call("call", {"task_id": task_id, "name": name, "args": args}, timeout=180.0)
        except RpcError as e:
            return ToolOutcome(False, f"toolhost unavailable: {e}")
        return ToolOutcome(bool(r.get("ok")), str(r.get("text", "")), r.get("trust", "trusted"), r.get("source", name),
                           r.get("image_data_url"))


def build_brain(cfg: JarvisConfig, tools=None) -> tuple[LLMBroker, Agent, MemoryStore]:
    paths = cfg.paths().ensure()
    providers = build_providers(cfg.brain)
    probe = load_cache(paths.run / "probe.json", cfg.brain.probe_cache_hours) or {}
    broker = LLMBroker(cfg.brain, providers, to_probe_info(probe))
    memory = MemoryStore(paths.db / "memory.sqlite")
    agent = Agent(broker, tools or NullToolBackend(), cfg.brain, memory=memory)
    return broker, agent, memory


def bus_port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
        except OSError:
            return False
    return True


def clock_text(lang: str, now: time.struct_time | None = None) -> str:
    t = now or time.localtime()
    h, m = t.tm_hour, t.tm_min
    if lang == "ko":
        part = "오전" if h < 12 else "오후"
        h12 = h % 12 or 12
        return f"지금은 {part} {h12}시 {m}분입니다." if m else f"지금은 {part} {h12}시입니다."
    h12 = h % 12 or 12
    return f"It's {h12}:{m:02d} {'AM' if h < 12 else 'PM'}."


class Core:
    def __init__(self, cfg: JarvisConfig, *, voice: bool = True) -> None:
        self.cfg = cfg
        self.paths = cfg.paths().ensure()
        self.token = load_or_create_token(self.paths.run)
        host, ports = cfg.ipc.host, cfg.ipc.rpc_ports
        self.pub = Publisher(cfg.ipc.bus_in_port, host)
        from jarvis.tools.local import CompositeToolBackend

        extras: dict[str, Any] = {"core": self}
        self.tools = CompositeToolBackend(extras, RpcToolBackend(RpcClient(ports["toolhost"], self.token, host,
                                                                           timeout=cfg.ipc.request_timeout_s)))
        self.broker, self.agent, self.memory = build_brain(cfg, self.tools)
        extras["memory"] = self.memory
        self.fightd = RpcClient(ports["fightd"], self.token, host, timeout=cfg.ipc.request_timeout_s)
        self.kill_switch = KillSwitch(on_broadcast=lambda m: self.pub.publish("kill", m))
        self.kill_switch.on_kill(self._on_kill_local)
        self.fight_state = "detached"
        self.last_match: dict[str, Any] | None = None
        self.want_voice = voice and cfg.voice.enabled
        self.runtime = None
        self.session: VoiceSession | None = None
        self.voice_notes: list[str] = []
        self.server = RpcServer("core", ports["core"], self.token, host)
        for name in ("status", "chat", "mute", "unmute", "stop"):
            self.server.add(name, getattr(self, f"rpc_{name}"))
        self.hub = None
        if cfg.hud.enabled:
            from jarvis.hud_bridge.hub import HudHub

            self.hub = HudHub(host, cfg.ipc.hud_ws_port, self.token, self._hud_command,
                              rates_hz={"fight.telemetry": cfg.hud.telemetry_hz, "fight.frame": max(cfg.hud.pip_fps, 0.1)})
        self._bg: set[asyncio.Task] = set()

    # ------------------------------------------------------------------ helpers
    def publish(self, topic: str, payload: dict) -> None:
        self.pub.publish(topic, payload)
        if self.hub is not None:
            self.hub.broadcast(topic, payload)

    def _spawn(self, coro) -> None:
        t = asyncio.get_running_loop().create_task(coro)
        self._bg.add(t)
        t.add_done_callback(self._bg.discard)

    async def _fightd(self, method: str, params: dict | None = None, timeout: float | None = None) -> dict:
        try:
            return dict(await self.fightd.call(method, params or {}, timeout=timeout) or {})
        except RpcTimeout:
            return {"ok": False, "error": "fightd did not answer in time - is it running? (`jarvis fightd`)"}
        except RpcError as e:
            return {"ok": False, "error": f"fightd error: {e}"}

    def _text_session(self) -> VoiceSession:
        from jarvis.voice.audio_io import NullPlayer

        return VoiceSession(stt=None, router=None, player=NullPlayer(), gate=SpeakerGate(None, None), handler=self,
                            options=SessionOptions.from_config(self.cfg.voice, self.cfg.safety, self.cfg.language),
                            publish=self.publish)

    # ------------------------------------------------------------------ TurnHandler
    async def handle(self, utt: Utterance, reply: ReplySink) -> None:
        it = utt.intent
        fn = getattr(self, f"_intent_{it.name}", None) if it is not None else None
        if fn is not None:
            await fn(utt, reply)
        else:
            await self._agent_turn(utt, reply)

    async def interrupted(self, heard_text: str) -> None:
        self.agent.note_interrupted(heard_text)

    def fight_active(self) -> bool:
        return self.fight_state in ACTIVE_FIGHT_STATES

    async def kill(self, reason: str) -> None:
        self.kill_switch.trigger(reason)

    def _on_kill_local(self, reason: str) -> None:
        if self.session is not None:
            self.session.stop_all()
        with contextlib.suppress(RuntimeError):
            self._spawn(self._fightd("panic", {"reason": reason}, timeout=2.0))
            self._spawn(self._toolhost_kill(reason))

    async def _toolhost_kill(self, reason: str) -> None:
        with contextlib.suppress(RpcError):
            await self.tools.client.call("kill", {"reason": reason}, timeout=2.0)

    # ------------------------------------------------------------------ intents
    async def _agent_turn(self, utt: Utterance, reply: ReplySink) -> None:
        caps = await scope_request(utt.text, self.broker)
        origin = "owner_text" if utt.addressed_by == "typed" else "owner_voice"
        res = await self.agent.run(AgentTurn(text=utt.text, origin=origin, lang=utt.lang, caps=caps),
                                   on_text=reply.write, cancel=reply.cancel)
        if res.error:
            log.info("agent turn ended with error: %s", res.error)
        self._spawn(self.agent.maybe_summarize())

    async def _intent_fight(self, utt: Utterance, reply: ReplySink) -> None:
        args = utt.intent.args if utt.intent else {}
        r = await self._fightd("arm", {"matches": args.get("matches", 1), "auto_queue": args.get("auto_queue", False),
                                       "origin": utt.addressed_by})
        if r.get("ok"):
            await reply.say(phrase("armed", utt.lang), key="armed")
        else:
            lead = "전투를 시작할 수 없습니다." if utt.lang == "ko" else "I can't start the fight."
            await reply.say(f"{lead} {r.get('error', '')}".strip())

    async def _intent_disarm(self, utt: Utterance, reply: ReplySink) -> None:
        await self._fightd("disarm", {"reason": f"owner command ({utt.addressed_by})"}, timeout=3.0)
        await reply.say(phrase("disarmed", utt.lang), key="disarmed")

    async def _intent_launch_ubg(self, utt: Utterance, reply: ReplySink) -> None:
        private = bool(utt.intent.args.get("private", True)) if utt.intent else True
        await reply.say(phrase("launching", utt.lang), key="launching")
        r = await self._fightd("launch", {"private": private}, timeout=self.cfg.fight.game.launch_timeout_s + 20)
        if not r.get("connected") and not r.get("ok"):
            lead = "게임을 실행하지 못했습니다." if utt.lang == "ko" else "The game did not start."
            await reply.say(f"{lead} {r.get('error', '')}".strip())

    async def _intent_fight_status(self, utt: Utterance, reply: ReplySink) -> None:
        st = await self._fightd("status", timeout=3.0)
        await reply.say(self.status_text(st, utt.lang))

    def status_text(self, st: dict, lang: str) -> str:
        ko = lang == "ko"
        parts: list[str] = []
        m = self.last_match
        if m:
            won, lost, res = m.get("stocks_won", 0), m.get("stocks_lost", 0), str(m.get("result", "unknown"))
            if ko:
                verdict = {"won": "승리", "likely_won": "승리로 보입니다", "lost": "패배", "likely_lost": "패배로 보입니다"}.get(res, "결과 불확실")
                parts.append(f"지난 경기는 스톡 {won}-{lost}, {verdict}.")
            else:
                verdict = {"won": "a win", "likely_won": "likely a win", "lost": "a loss", "likely_lost": "likely a loss"}.get(res, "result unclear")
                parts.append(f"Last match: {won}-{lost} in stocks, {verdict}.")
        if not st.get("connected"):
            parts.append("로블록스에 연결되어 있지 않습니다." if ko else "Roblox isn't connected.")
        else:
            state = str(st.get("state", "unknown"))
            parts.append(f"현재 상태는 {STATE_KO.get(state, state)}입니다." if ko else f"Current state: {state.replace('_', ' ')}.")
        return " ".join(parts)

    async def _volume(self, utt: Utterance, reply: ReplySink, delta: float) -> None:
        if sys.platform != "win32":
            await reply.say("이 시스템에서는 볼륨을 바꿀 수 없습니다." if utt.lang == "ko" else "I can't change the volume on this system.")
            return
        from jarvis.platform.win.volume import change_volume

        try:
            level = await asyncio.to_thread(change_volume, delta)
        except Exception as e:
            log.warning("volume change failed: %s", e)
            await reply.say("볼륨을 바꾸지 못했습니다." if utt.lang == "ko" else "I couldn't change the volume.")
            return
        pct = round(level * 100)
        await reply.say(f"볼륨 {pct}%." if utt.lang == "ko" else f"Volume {pct}%.")

    async def _intent_volume_up(self, utt: Utterance, reply: ReplySink) -> None:
        await self._volume(utt, reply, +0.1)

    async def _intent_volume_down(self, utt: Utterance, reply: ReplySink) -> None:
        await self._volume(utt, reply, -0.1)

    async def _intent_time(self, utt: Utterance, reply: ReplySink) -> None:
        await reply.say(clock_text(utt.lang))

    async def _intent_mute(self, utt: Utterance, reply: ReplySink) -> None:  # typed "mute"
        if self.session is not None:
            await reply.say(phrase("muted", utt.lang), key="muted")
            self.session.mute()

    async def _intent_unmute(self, utt: Utterance, reply: ReplySink) -> None:
        if self.session is not None:
            self.session.unmute()
            await reply.say(phrase("unmuted", utt.lang), key="unmuted")

    async def _intent_interpreter_on(self, utt: Utterance, reply: ReplySink) -> None:
        if self.session is not None:
            self.session.interpreter = True
            await reply.say(phrase("interpreter_on", utt.lang), key="interpreter_on")

    async def _intent_interpreter_off(self, utt: Utterance, reply: ReplySink) -> None:
        if self.session is not None:
            self.session.interpreter = False
            await reply.say(phrase("interpreter_off", utt.lang), key="interpreter_off")

    async def _hud_panels(self, utt: Utterance, reply: ReplySink, action: str) -> None:
        panel = utt.intent.args.get("panel", "all") if utt.intent else "all"
        self.publish("hud.command", {"action": action, "panel": panel, "t": time.time()})
        await reply.say("알겠습니다." if utt.lang == "ko" else "Done.")

    async def _intent_hud_show(self, utt: Utterance, reply: ReplySink) -> None:
        await self._hud_panels(utt, reply, "show")

    async def _intent_hud_hide(self, utt: Utterance, reply: ReplySink) -> None:
        await self._hud_panels(utt, reply, "hide")

    async def _hud_command(self, cmd: dict) -> dict:
        """Commands from the HUD (token-authenticated, typed by the owner at this PC)."""
        kind = cmd.get("type")
        s = self.session
        if kind == "stop":
            self.kill_switch.trigger("HUD stop button")
        elif kind == "disarm":
            return await self._fightd("disarm", {"reason": "HUD"}, timeout=3.0)
        elif s is None:
            return {"ok": False, "error": "core is starting"}
        elif kind == "chat":
            text = str(cmd.get("text", ""))[:2000]
            await s.typed(text, speak=bool(cmd.get("speak", self.runtime is not None)))
        elif kind == "mute":
            s.mute(soft=bool(cmd.get("soft", False)))
        elif kind == "unmute":
            s.unmute()
        elif kind == "interpreter":
            s.interpreter = bool(cmd.get("on", False))
            s.publish("voice.state", {"mode": s.barge.mode.value, "muted": s.muted, "interpreter": s.interpreter, "lang": s.lang})
        return {"ok": True}

    async def _stats_loop(self) -> None:
        from jarvis.hud_bridge.stats import StatsSampler

        sampler = StatsSampler()
        while True:
            if self.hub is not None and self.hub.clients:
                self.hub.broadcast("system.stats", await asyncio.to_thread(sampler.sample))
            await asyncio.sleep(1.0)

    # ------------------------------------------------------------------ RPC
    def rpc_status(self, _p: dict) -> dict:
        s = self.session
        return {
            "voice": self.runtime is not None,
            "voice_notes": self.voice_notes,
            "muted": s.muted if s else None,
            "mode": s.barge.mode.value if s else "idle",
            "interpreter": s.interpreter if s else False,
            "fight_state": self.fight_state,
            "broker": self.broker.status(),
            "stats": dict(s.stats) if s else {},
        }

    async def rpc_chat(self, p: dict) -> dict:
        assert self.session is not None
        sink = await self.session.typed(str(p.get("text", "")), speak=bool(p.get("speak", self.runtime is not None)))
        if sink is None:
            return {"ok": True, "text": "", "stopped": True}
        await self.session.wait_idle(timeout=float(p.get("timeout", 180.0)))
        return {"ok": True, "text": sink.text}

    def rpc_mute(self, p: dict) -> dict:
        if self.session is not None:
            self.session.mute(soft=bool(p.get("soft", False)))
        return {"ok": True}

    def rpc_unmute(self, _p: dict) -> dict:
        if self.session is not None:
            self.session.unmute()
        return {"ok": True}

    def rpc_stop(self, p: dict) -> dict:
        self.kill_switch.trigger(str(p.get("reason", "stop requested")))
        return {"ok": True}

    # ------------------------------------------------------------------ bus
    async def _bus_listener(self) -> None:
        sub = AsyncSubscriber(self.cfg.ipc.bus_out_port, ["fight.", "kill"], self.cfg.ipc.host)
        try:
            while True:
                topic, msg = await sub.recv()
                try:
                    await self._on_bus(topic, msg if isinstance(msg, dict) else {})
                except Exception:
                    log.exception("bus event %s failed", topic)
        finally:
            sub.close()

    async def _on_bus(self, topic: str, msg: dict) -> None:
        if self.hub is not None and topic.startswith("fight."):
            self.hub.broadcast(topic, msg)
        if topic == "fight.state":
            self.fight_state = str(msg.get("new", self.fight_state))
        elif topic == "fight.callout":
            key = str(msg.get("key", ""))
            if key == "match_end":
                self.last_match = msg
            if self.session is not None and key != "armed":  # "armed" is confirmed by the fight intent itself
                await self.session.callout(key, msg)
        elif topic == "fight.panic":
            if self.session is not None:
                await self.session.callout("panic", msg)
        elif topic == "kill" and self.session is not None:
            self.session.stop_all()  # someone else pulled the kill switch; never re-broadcast

    # ------------------------------------------------------------------ main
    async def run(self) -> None:
        await self.server.start()
        if self.want_voice:
            from jarvis.voice.interpreter import Interpreter
            from jarvis.voice.runtime import VoiceRuntime, VoiceUnavailable

            rt = VoiceRuntime(self.cfg, self, self.publish, Interpreter(self.broker).translate)
            try:
                self.session = rt.build()
                await rt.start()
                self.runtime = rt
                self.voice_notes = rt.notes
            except VoiceUnavailable as e:
                log.error("voice disabled: %s", e)
                self.voice_notes = [str(e)]
                self.session = None
        if self.session is None:
            self.session = self._text_session()
            await self.session.start()
        self._spawn(self._bus_listener())
        if self.hub is not None:
            try:
                await self.hub.start()
                self._spawn(self._stats_loop())
            except OSError as e:
                log.warning("HUD hub could not listen on port %s: %s", self.cfg.ipc.hud_ws_port, e)
                self.hub = None
        log.info("core ready (voice=%s) on port %s", self.runtime is not None, self.server.port)
        try:
            while True:
                await asyncio.sleep(3600)
        finally:
            if self.runtime is not None:
                await self.runtime.stop()
            elif self.session is not None:
                await self.session.close()
            for t in list(self._bg):
                t.cancel()
            if self.hub is not None:
                await self.hub.stop()
            await self.server.close()
            self.memory.close()
            self.pub.close()


def main(cfg: JarvisConfig, *, voice: bool = True, standalone_bus: bool = False) -> None:
    proxy = None
    if standalone_bus and bus_port_free(cfg.ipc.host, cfg.ipc.bus_in_port):
        proxy = BusProxy(cfg.ipc.bus_in_port, cfg.ipc.bus_out_port, cfg.ipc.host)
        proxy.start()
    try:
        asyncio.run(Core(cfg, voice=voice).run())
    finally:
        if proxy is not None:
            proxy.stop()
