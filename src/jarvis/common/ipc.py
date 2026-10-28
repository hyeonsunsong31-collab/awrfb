"""Inter-process communication over ZeroMQ on loopback.

* RPC: each service binds a ROUTER socket; clients use DEALER. Every request carries the session
  token (random, user-private file) - requests without it are rejected. Commands that change state
  (arming a fight, running a tool) only ever travel over RPC.
* Events: an XSUB/XPUB proxy (run by the supervisor) fans out informational events (telemetry,
  transcripts, state changes) plus the always-safe ``kill`` broadcast.
Payloads are msgpack-encoded with msgspec.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import inspect
import itertools
import logging
import os
import secrets
import threading
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import msgspec
import zmq
import zmq.asyncio

log = logging.getLogger(__name__)

_enc = msgspec.msgpack.Encoder()
_dec = msgspec.msgpack.Decoder()


def encode(obj: Any) -> bytes:
    return _enc.encode(obj)


def decode(data: bytes) -> Any:
    return _dec.decode(data)


class RpcError(RuntimeError):
    pass


class RpcTimeout(RpcError):
    pass


class RpcUnauthorized(RpcError):
    pass


def load_or_create_token(run_dir: Path) -> str:
    env = os.environ.get("JARVIS_IPC_TOKEN")
    if env:
        return env
    run_dir.mkdir(parents=True, exist_ok=True)
    f = run_dir / "ipc.token"
    if f.exists():
        tok = f.read_text(encoding="utf-8").strip()
        if tok:
            return tok
    tok = secrets.token_urlsafe(32)
    f.write_text(tok, encoding="utf-8")
    with contextlib.suppress(OSError):
        os.chmod(f, 0o600)
    return tok


Handler = Callable[[dict[str, Any]], Any | Awaitable[Any]]


class RpcServer:
    def __init__(self, name: str, port: int, token: str, host: str = "127.0.0.1") -> None:
        self.name = name
        self.port = port
        self.host = host
        self._token = token
        self._handlers: dict[str, Handler] = {}
        self._ctx = zmq.asyncio.Context.instance()
        self._sock: zmq.asyncio.Socket | None = None
        self._task: asyncio.Task | None = None
        self._inflight: set[asyncio.Task] = set()

    def method(self, name: str) -> Callable[[Handler], Handler]:
        def deco(fn: Handler) -> Handler:
            self._handlers[name] = fn
            return fn

        return deco

    def add(self, name: str, fn: Handler) -> None:
        self._handlers[name] = fn

    async def start(self) -> None:
        sock = self._ctx.socket(zmq.ROUTER)
        sock.setsockopt(zmq.LINGER, 0)
        sock.bind(f"tcp://{self.host}:{self.port}")
        self._sock = sock
        self._task = asyncio.create_task(self._loop(), name=f"rpc-{self.name}")

    async def _loop(self) -> None:
        assert self._sock is not None
        while True:
            try:
                frames = await self._sock.recv_multipart()
            except (asyncio.CancelledError, zmq.ContextTerminated):
                return
            if len(frames) < 2:
                continue
            ident, payload = frames[0], frames[-1]
            t = asyncio.create_task(self._handle(ident, payload))
            self._inflight.add(t)
            t.add_done_callback(self._inflight.discard)

    async def _handle(self, ident: bytes, payload: bytes) -> None:
        rid = None
        try:
            req = decode(payload)
            rid = req.get("id")
            if not hmac.compare_digest(str(req.get("tok", "")), self._token):
                raise RpcUnauthorized("bad token")
            fn = self._handlers.get(req.get("m", ""))
            if fn is None:
                raise RpcError(f"unknown method {req.get('m')!r}")
            res = fn(req.get("p") or {})
            if inspect.isawaitable(res):
                res = await res
            reply = {"id": rid, "ok": True, "r": res}
        except Exception as e:  # report every failure to the caller, never crash the server
            if not isinstance(e, RpcError):
                log.exception("rpc %s handler failed", self.name)
            reply = {"id": rid, "ok": False, "e": f"{type(e).__name__}: {e}"}
        if self._sock is not None:
            try:
                await self._sock.send_multipart([ident, encode(reply)])
            except zmq.ZMQError:
                pass

    async def close(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        for t in list(self._inflight):
            t.cancel()
        if self._sock is not None:
            self._sock.close(0)
            self._sock = None


class RpcClient:
    def __init__(self, port: int, token: str, host: str = "127.0.0.1", timeout: float = 10.0) -> None:
        self._addr = f"tcp://{host}:{port}"
        self._token = token
        self._timeout = timeout
        self._ctx = zmq.asyncio.Context.instance()
        self._sock: zmq.asyncio.Socket | None = None
        self._pending: dict[int, asyncio.Future] = {}
        self._ids = itertools.count(1)
        self._reader: asyncio.Task | None = None

    async def _ensure(self) -> zmq.asyncio.Socket:
        if self._sock is None:
            s = self._ctx.socket(zmq.DEALER)
            s.setsockopt(zmq.LINGER, 0)
            s.connect(self._addr)
            self._sock = s
            self._reader = asyncio.create_task(self._read())
        return self._sock

    async def _read(self) -> None:
        assert self._sock is not None
        while True:
            try:
                frames = await self._sock.recv_multipart()
            except (asyncio.CancelledError, zmq.ContextTerminated):
                return
            try:
                rep = decode(frames[-1])
            except Exception:
                continue
            fut = self._pending.pop(rep.get("id"), None)
            if fut is not None and not fut.done():
                fut.set_result(rep)

    async def call(self, method: str, params: dict[str, Any] | None = None, timeout: float | None = None) -> Any:
        sock = await self._ensure()
        rid = next(self._ids)
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[rid] = fut
        await sock.send_multipart([encode({"id": rid, "m": method, "p": params or {}, "tok": self._token})])
        try:
            rep = await asyncio.wait_for(fut, timeout or self._timeout)
        except TimeoutError as e:
            self._pending.pop(rid, None)
            raise RpcTimeout(f"{method} timed out") from e
        if not rep.get("ok"):
            raise RpcError(rep.get("e", "rpc failed"))
        return rep.get("r")

    async def close(self) -> None:
        if self._reader:
            self._reader.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reader
        if self._sock is not None:
            self._sock.close(0)
            self._sock = None


# ------------------------------------------------------------------------------------- event bus
class BusProxy:
    """XSUB (publishers connect) <-> XPUB (subscribers connect), on its own thread."""

    def __init__(self, in_port: int, out_port: int, host: str = "127.0.0.1") -> None:
        self._ctx = zmq.Context()
        self._in = f"tcp://{host}:{in_port}"
        self._out = f"tcp://{host}:{out_port}"
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        def run() -> None:
            xsub = self._ctx.socket(zmq.XSUB)
            xpub = self._ctx.socket(zmq.XPUB)
            xsub.bind(self._in)
            xpub.bind(self._out)
            try:
                zmq.proxy(xsub, xpub)
            except zmq.ContextTerminated:
                pass
            finally:
                xsub.close(0)
                xpub.close(0)

        self._thread = threading.Thread(target=run, name="bus-proxy", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._ctx.term()


class Publisher:
    """Thread-confined PUB socket (use one per thread)."""

    def __init__(self, port: int, host: str = "127.0.0.1", ctx: zmq.Context | None = None) -> None:
        self._sock = (ctx or zmq.Context.instance()).socket(zmq.PUB)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.setsockopt(zmq.SNDHWM, 1000)
        self._sock.connect(f"tcp://{host}:{port}")

    def publish(self, topic: str, payload: Any) -> None:
        try:
            self._sock.send_multipart([topic.encode("utf-8"), encode(payload)], flags=zmq.NOBLOCK)
        except zmq.Again:
            pass  # drop telemetry rather than block a real-time loop

    def close(self) -> None:
        self._sock.close(0)


class AsyncSubscriber:
    def __init__(self, port: int, topics: list[str], host: str = "127.0.0.1") -> None:
        self._sock = zmq.asyncio.Context.instance().socket(zmq.SUB)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.connect(f"tcp://{host}:{port}")
        for t in topics or [""]:
            self._sock.setsockopt(zmq.SUBSCRIBE, t.encode("utf-8"))

    async def recv(self) -> tuple[str, Any]:
        topic, data = await self._sock.recv_multipart()
        return topic.decode("utf-8"), decode(data)

    def close(self) -> None:
        self._sock.close(0)
