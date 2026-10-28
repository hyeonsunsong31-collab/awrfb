"""fightd: the process that owns the game connection. It is the ONLY process that sends input to
Roblox. Commands arrive over token-authenticated RPC from core (voice/HUD) or the CLI; telemetry and
callouts go out on the event bus; a `kill` broadcast from anyone disarms immediately."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import Any

from jarvis.common.audit import AuditLog
from jarvis.common.config import JarvisConfig
from jarvis.common.ipc import AsyncSubscriber, Publisher, RpcServer, load_or_create_token
from jarvis.fight.gameio import GameIO
from jarvis.fight.launcher import LaunchError
from jarvis.fight.loop import ArmRefused, FightLoop
from jarvis.fight.states import FightState
from jarvis.platform.factory import get_platform

log = logging.getLogger(__name__)


class FightDaemon:
    def __init__(self, cfg: JarvisConfig) -> None:
        self.cfg = cfg
        self.paths = cfg.paths().ensure()
        self.token = load_or_create_token(self.paths.run)
        self.plat = get_platform()
        self.audit = AuditLog(self.paths.audit, "fightd") if cfg.safety.audit_enabled else None
        self.io: GameIO | None = None
        self.loop: FightLoop | None = None
        self._outbox: asyncio.Queue[tuple[str, Any]] | None = None
        self._aloop: asyncio.AbstractEventLoop | None = None
        self.server = RpcServer("fightd", cfg.ipc.rpc_ports["fightd"], self.token, cfg.ipc.host)
        for name in ("status", "attach", "launch", "arm", "disarm", "panic", "detach"):
            self.server.add(name, getattr(self, f"rpc_{name}"))

    # ------------------------------------------------------------------ outbound events (thread-safe)
    def emit(self, topic: str, payload: Any) -> None:
        if self._aloop is not None and self._outbox is not None:
            self._aloop.call_soon_threadsafe(self._outbox.put_nowait, (topic, payload))

    async def _publisher(self) -> None:
        assert self._outbox is not None
        pub = Publisher(self.cfg.ipc.bus_in_port, self.cfg.ipc.host)
        try:
            while True:
                topic, payload = await self._outbox.get()
                pub.publish(topic, payload)
        finally:
            pub.close()

    async def _kill_listener(self) -> None:
        sub = AsyncSubscriber(self.cfg.ipc.bus_out_port, ["kill"], self.cfg.ipc.host)
        try:
            while True:
                _topic, msg = await sub.recv()
                if self.io is not None:
                    self.io.panic(f"kill broadcast: {msg.get('reason', '')}")
        finally:
            sub.close()

    # ------------------------------------------------------------------ connection
    def _connect(self) -> None:
        if self.io is not None:
            return
        io = GameIO(self.cfg, self.plat, audit=self.audit, on_panic=lambda r: self.emit("fight.panic", {"reason": r}))
        io.attach()
        io.start()
        io.sm.on_change(lambda ch: self.emit("fight.state", {"old": ch.old.value, "new": ch.new.value, "reason": ch.reason}))
        loop = FightLoop(
            self.cfg,
            io,
            on_telemetry=lambda d: self.emit("fight.telemetry", d),
            on_callout=lambda k, d: self.emit("fight.callout", {"key": k, **d}),
        )
        loop.start()
        self.io, self.loop = io, loop

    def _disconnect(self, reason: str) -> None:
        if self.loop is not None:
            self.loop.stop()
        if self.io is not None:
            self.io.stop()
        self.io = self.loop = None
        self.emit("fight.state", {"old": "?", "new": "detached", "reason": reason})

    # ------------------------------------------------------------------ RPC
    def rpc_status(self, _p: dict) -> dict:
        if self.io is None or self.loop is None:
            return {"connected": False, "state": "detached"}
        return {
            "connected": True,
            "state": self.io.sm.state.value,
            "server_kind": getattr(self.io.session, "server_kind", None),
            "fps": round(self.io.fps.fps, 1),
            "last_refusal": self.loop.last_refusal,
            "preflight": self.loop.preflight() if self.io.sm.state == FightState.DISARMED else [],
            "inputs": dict(self.io.actuator.stats),
        }

    def rpc_attach(self, _p: dict) -> dict:
        try:
            self._connect()
        except LaunchError as e:
            return {"ok": False, "error": str(e)}
        return self.rpc_status({})

    def rpc_launch(self, p: dict) -> dict:
        from jarvis.fight.launcher import GameLauncher

        self._disconnect("relaunch")
        gl = GameLauncher(self.plat, self.cfg.fight.game, self.paths.run)
        try:
            gl.launch(private=bool(p.get("private", True)))
        except LaunchError as e:
            return {"ok": False, "error": str(e)}
        time.sleep(2.0)
        return self.rpc_attach({})

    def rpc_arm(self, p: dict) -> dict:
        """Only reachable with the IPC token; core calls it only for an owner-verified command."""
        if self.loop is None:
            att = self.rpc_attach({})
            if not att.get("connected"):
                return {"ok": False, "error": att.get("error", "not connected to Roblox")}
        assert self.loop is not None and self.io is not None
        self.io.focus_game()
        try:
            self.loop.arm(int(p.get("matches", 1)), bool(p.get("auto_queue", False)))
        except ArmRefused as e:
            return {"ok": False, "error": str(e)}
        if self.audit is not None:
            self.audit.append("arm", {"origin": p.get("origin", "?"), "matches": p.get("matches", 1)})
        return {"ok": True, "state": self.io.sm.state.value}

    def rpc_disarm(self, p: dict) -> dict:
        if self.loop is not None:
            self.loop.disarm(str(p.get("reason", "owner command")))
        return {"ok": True}

    def rpc_panic(self, p: dict) -> dict:
        if self.io is not None:
            self.io.panic(str(p.get("reason", "panic")))
        return {"ok": True}

    def rpc_detach(self, _p: dict) -> dict:
        self._disconnect("owner command")
        return {"ok": True}

    # ------------------------------------------------------------------ main
    async def run(self) -> None:
        self._aloop = asyncio.get_running_loop()
        self._outbox = asyncio.Queue(maxsize=1000)
        await self.server.start()
        tasks = [asyncio.create_task(self._publisher()), asyncio.create_task(self._kill_listener())]
        log.info("fightd ready on port %s", self.server.port)
        try:
            while True:
                await asyncio.sleep(1.0)
                # Roblox closed? Drop the connection cleanly (keys released by io.stop()).
                if self.io is not None and self.io.window is not None and not self.plat.windows.exists(self.io.window.hwnd):
                    self._disconnect("Roblox window closed")
        finally:
            for t in tasks:
                t.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await t
            self._disconnect("fightd exiting")
            await self.server.close()


def main(cfg: JarvisConfig) -> None:
    asyncio.run(FightDaemon(cfg).run())
