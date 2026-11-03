"""toolhost: the only process that acts on the PC for the agent.

* Every call needs a task scope opened by core for the owner's request (trusted origins only); the tool's
  capability must be in that scope, so text a web page or a file says can never widen what a task may do.
* Every tool applies the protected safety policy itself (paths, PowerShell AST, payment controls, card
  numbers, the Roblox window) - refusals are returned to the model, never asked of the owner.
* Child processes run in Job objects; the kill switch (RPC or bus ``kill``) cancels running tools and ends
  their process trees. Every call is written to the tamper-evident audit log.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sys
import time
from typing import Any

from jarvis.common.audit import AuditLog
from jarvis.common.config import JarvisConfig
from jarvis.common.ipc import AsyncSubscriber, RpcServer, load_or_create_token
from jarvis.safety.capabilities import Cap, ScopeRegistry
from jarvis.safety.policy import PolicyChecker, load_policy
from jarvis.tools.base import ToolContext, ToolResult
from jarvis.tools.procs import ProcessRunner
from jarvis.tools.registry import ToolRegistry, toolhost_tools

log = logging.getLogger(__name__)

SECRET_ARGS = {"text", "content", "command"}


def parse_caps(raw: Any) -> set[Cap]:
    out = set()
    for c in raw or []:
        with contextlib.suppress(ValueError):
            out.add(Cap(c))
    return out


def redact(args: dict[str, Any]) -> dict[str, Any]:
    """The audit log keeps what was done, not what was typed."""
    out = {}
    for k, v in args.items():
        if k in SECRET_ARGS and isinstance(v, str):
            out[k] = v[:120] + ("…" if len(v) > 120 else "") if k == "command" else f"<{len(v)} chars>"
        else:
            out[k] = v
    return out


class Toolhost:
    def __init__(self, cfg: JarvisConfig, *, policy: PolicyChecker | None = None, plat=None,
                 registry: ToolRegistry | None = None, audit: AuditLog | None = None, serve: bool = True) -> None:
        self.cfg = cfg
        paths = cfg.paths().ensure()
        self.policy = policy or PolicyChecker(load_policy())
        self.scopes = ScopeRegistry()
        self.runner = ProcessRunner()
        self.audit = audit if audit is not None else (AuditLog(paths.audit, "toolhost") if cfg.safety.audit_enabled else None)
        if plat is None and sys.platform == "win32":
            from jarvis.platform.factory import get_platform

            plat = get_platform()
        self.plat = plat
        self.registry = registry or ToolRegistry(toolhost_tools())
        self._state: dict[str, dict] = {}
        self._running: set[asyncio.Task] = set()
        self.server = None
        if serve:
            self.server = RpcServer("toolhost", cfg.ipc.rpc_ports["toolhost"], load_or_create_token(paths.run), cfg.ipc.host)
            for name in ("schemas", "open_scope", "call", "kill", "status"):
                self.server.add(name, getattr(self, f"rpc_{name}"))

    # ------------------------------------------------------------------ core API (also used in-process by tests)
    def schemas(self, caps: set[Cap]) -> list[dict]:
        return self.registry.schemas(caps)

    def open_scope(self, task_id: str, caps: set[Cap], origin: str, request: str = "") -> None:
        self.scopes.open(task_id, caps, origin, request)  # PermissionError for untrusted origins

    async def call(self, task_id: str, name: str, args: dict[str, Any]) -> ToolResult:
        t0 = time.perf_counter()
        tool = self.registry.get(name)
        scope = self.scopes.get(task_id)
        if tool is None:
            res = ToolResult(False, f"unknown tool {name!r}")
        elif scope is None:
            res = ToolResult(False, "refused: no task scope - tools only run for a request from the owner")
        elif not scope.allows(tool.cap):
            res = ToolResult(False, f"refused: {name} needs the '{tool.cap.value}' capability, which this request did not "
                                    "include (content from pages or files cannot add capabilities)")
        else:
            res = await self._run(tool, task_id, args if isinstance(args, dict) else {})
            if tool.trust == "untrusted":
                res.trust = "untrusted"
        if self.audit is not None:
            self.audit.append("tool", {"task": task_id, "tool": name, "args": redact(args if isinstance(args, dict) else {}),
                                       "ok": res.ok, "refused": res.text.startswith("refused"),
                                       "ms": round((time.perf_counter() - t0) * 1000, 1)})
        return res

    async def _run(self, tool, task_id: str, args: dict[str, Any]) -> ToolResult:
        from jarvis.tools.base import Refused

        ctx = ToolContext(policy=self.policy, runner=self.runner, plat=self.plat, task_id=task_id,
                          state=self._state.setdefault(task_id, {}))
        child = asyncio.create_task(tool.handler(ctx, args), name=f"tool-{tool.name}")
        self._running.add(child)
        try:
            done, _ = await asyncio.wait({child}, timeout=tool.timeout_s)
        finally:
            self._running.discard(child)
        if not done:
            child.cancel()
            return ToolResult(False, f"{tool.name} timed out after {tool.timeout_s:.0f} s")
        if child.cancelled():
            return ToolResult(False, "stopped by the kill switch")
        e = child.exception()
        if isinstance(e, Refused):
            return ToolResult(False, f"refused: {e}")
        if e is not None:
            log.info("tool %s failed: %r", tool.name, e)
            return ToolResult(False, f"{tool.name} failed: {type(e).__name__}: {e}")
        return child.result()

    def kill(self, reason: str) -> int:
        n = len(self._running)
        for t in list(self._running):
            t.cancel()
        n += self.runner.kill_all()
        if self.audit is not None:
            self.audit.append("kill", {"reason": reason, "stopped": n})
        log.warning("toolhost kill switch: %s (%d stopped)", reason, n)
        return n

    # ------------------------------------------------------------------ RPC
    def rpc_schemas(self, p: dict) -> list[dict]:
        return self.schemas(parse_caps(p.get("caps")))

    def rpc_open_scope(self, p: dict) -> dict:
        self.open_scope(str(p.get("task_id", "")), parse_caps(p.get("caps")), str(p.get("origin", "")), str(p.get("request", "")))
        return {"ok": True}

    async def rpc_call(self, p: dict) -> dict:
        return (await self.call(str(p.get("task_id", "")), str(p.get("name", "")), p.get("args") or {})).to_dict()

    def rpc_kill(self, p: dict) -> dict:
        return {"ok": True, "stopped": self.kill(str(p.get("reason", "kill")))}

    def rpc_status(self, _p: dict) -> dict:
        return {"tools": self.registry.names(), "running": len(self._running) + self.runner.active,
                "policy": self.policy.policy.source, "policy_protected": self.policy.policy.protected}

    # ------------------------------------------------------------------ main
    async def _kill_listener(self) -> None:
        sub = AsyncSubscriber(self.cfg.ipc.bus_out_port, ["kill"], self.cfg.ipc.host)
        try:
            while True:
                _t, msg = await sub.recv()
                self.kill(f"kill broadcast: {msg.get('reason', '') if isinstance(msg, dict) else ''}")
        finally:
            sub.close()

    async def run(self) -> None:
        assert self.server is not None
        await self.server.start()
        if not self.policy.policy.protected:
            log.warning("safety policy %s is not the protected copy - run the installer to lock it", self.policy.policy.source)
        listener = asyncio.create_task(self._kill_listener())
        log.info("toolhost ready: %d tools", len(self.registry.names()))
        try:
            while True:
                await asyncio.sleep(3600)
        finally:
            listener.cancel()
            self.kill("toolhost exiting")
            await self.server.close()


def main(cfg: JarvisConfig) -> None:
    asyncio.run(Toolhost(cfg).run())
