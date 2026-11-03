"""WebSocket hub for the holographic HUD (127.0.0.1 only).

Protocol (JSON text frames):
  HUD -> hub  first message {"type": "auth", "token": "<run/ipc.token>"}; anything else closes with 4401.
  hub -> HUD  {"type": "hello", "snapshot": {topic: last payload}} so a (re)connecting HUD shows current state,
              then {"type": "event", "topic": "...", "data": {...}} for every event.
  HUD -> hub  commands {"type": "chat"|"mute"|"unmute"|"stop"|"disarm"|"interpreter", "id": n, ...};
              the reply is {"type": "result", "id": n, ...}.

High-rate topics are throttled per topic; each client has a bounded queue, so a slow or frozen HUD
drops old events instead of slowing core down.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import logging
import time
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Any

log = logging.getLogger(__name__)

COMMANDS = frozenset({"chat", "mute", "unmute", "stop", "disarm", "interpreter"})
# Latest value of these is replayed to every new client.
STICKY = frozenset({"voice.state", "voice.notes", "fight.state", "fight.telemetry", "system.stats", "hud.command"})
# The recent conversation is replayed too, so a restarted HUD shows the last exchanges.
HISTORY = frozenset({"voice.transcript", "voice.reply_done", "voice.rejected"})
HISTORY_LEN = 24
AUTH_TIMEOUT_S = 3.0

CommandHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any] | None]]


class _Client:
    def __init__(self, ws, maxsize: int) -> None:
        self.ws = ws
        self.queue: asyncio.Queue[str] = asyncio.Queue(maxsize=maxsize)
        self.dropped = 0

    def offer(self, msg: str) -> None:
        if self.queue.full():
            with contextlib.suppress(asyncio.QueueEmpty):
                self.queue.get_nowait()  # drop the oldest: the HUD wants the newest state
            self.dropped += 1
        self.queue.put_nowait(msg)


class HudHub:
    def __init__(
        self,
        host: str,
        port: int,
        token: str,
        on_command: CommandHandler | None = None,
        *,
        rates_hz: dict[str, float] | None = None,
        queue_size: int = 256,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.host = host
        self.port = port
        self._token = token
        self._on_command = on_command
        self._rates = {"fight.telemetry": 15.0, "fight.frame": 5.0, "voice.level": 30.0, "system.stats": 2.0, **(rates_hz or {})}
        self._queue_size = queue_size
        self._clock = clock
        self._clients: set[_Client] = set()
        self._last: dict[str, float] = {}
        self.snapshot: dict[str, Any] = {}
        self.history: deque[tuple[str, Any]] = deque(maxlen=HISTORY_LEN)
        self._server = None

    @property
    def clients(self) -> int:
        return len(self._clients)

    async def start(self) -> None:
        from websockets.asyncio.server import serve

        self._server = await serve(self._handler, self.host, self.port, max_size=1 << 20, ping_interval=20)
        if self.port == 0:  # tests: pick a free port
            self.port = next(iter(self._server.sockets)).getsockname()[1]

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    # ------------------------------------------------------------------ outbound
    def broadcast(self, topic: str, data: Any) -> None:
        """Call from the event-loop thread. Cheap when no HUD is connected."""
        if topic in STICKY:
            self.snapshot[topic] = data
        elif topic in HISTORY:
            self.history.append((topic, data))
        if not self._clients:
            return
        rate = self._rates.get(topic)
        if rate:
            now = self._clock()
            if now - self._last.get(topic, float("-inf")) < 1.0 / rate:
                return
            self._last[topic] = now
        msg = json.dumps({"type": "event", "topic": topic, "data": data}, ensure_ascii=False, default=str)
        for c in list(self._clients):
            c.offer(msg)

    # ------------------------------------------------------------------ connection
    async def _handler(self, ws) -> None:
        try:
            raw = await asyncio.wait_for(ws.recv(), AUTH_TIMEOUT_S)
            msg = json.loads(raw)
        except Exception:
            await ws.close(4401, "auth required")
            return
        if not isinstance(msg, dict) or msg.get("type") != "auth" or not hmac.compare_digest(str(msg.get("token", "")), self._token):
            await ws.close(4401, "bad token")
            return
        client = _Client(ws, self._queue_size)
        hello = {"type": "hello", "snapshot": self.snapshot, "history": [{"topic": t, "data": d} for t, d in self.history]}
        await ws.send(json.dumps(hello, ensure_ascii=False, default=str))
        self._clients.add(client)
        sender = asyncio.create_task(self._sender(client))
        try:
            async for raw in ws:
                await self._command(client, raw)
        except Exception:
            log.debug("HUD connection ended", exc_info=True)
        finally:
            self._clients.discard(client)
            sender.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await sender

    async def _sender(self, client: _Client) -> None:
        while True:
            msg = await client.queue.get()
            await client.ws.send(msg)

    async def _command(self, client: _Client, raw: str | bytes) -> None:
        try:
            cmd = json.loads(raw)
        except ValueError:
            return
        if not isinstance(cmd, dict) or cmd.get("type") not in COMMANDS:
            reply: dict[str, Any] | None = {"ok": False, "error": "unknown command"}
        elif self._on_command is None:
            reply = {"ok": False, "error": "no command handler"}
        else:
            try:
                reply = await self._on_command(cmd)
            except Exception as e:
                log.exception("HUD command %s failed", cmd.get("type"))
                reply = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        if isinstance(cmd, dict) and cmd.get("id") is not None:
            client.offer(json.dumps({"type": "result", "id": cmd["id"], **(reply or {"ok": True})}, ensure_ascii=False, default=str))
