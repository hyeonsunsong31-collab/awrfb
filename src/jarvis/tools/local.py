"""Tools that live in core (they need core's state): long-term memory, fight status/stop, HUD notices -
plus the backend that merges them with toolhost's tools for the agent."""

from __future__ import annotations

from typing import Any

from jarvis.brain.agent import ToolOutcome
from jarvis.safety.capabilities import Cap
from jarvis.tools.base import Refused, Tool, ToolContext, ToolResult, i, ok, s
from jarvis.tools.registry import ToolRegistry


async def memory_remember(ctx: ToolContext, a: dict) -> ToolResult:
    text = str(a.get("text", "")).strip()
    if not text:
        raise Refused("nothing to remember")
    fid = ctx.extras["memory"].remember(text, source="agent", trust="owner")
    return ok(f"remembered (#{fid})")


async def memory_recall(ctx: ToolContext, a: dict) -> ToolResult:
    facts = ctx.extras["memory"].recall(str(a.get("query", "")), k=max(1, min(int(a.get("k", 5)), 20)))
    return ok("\n".join(f"#{f.id} {f.text}" for f in facts) or "nothing relevant remembered")


async def memory_forget(ctx: ToolContext, a: dict) -> ToolResult:
    fid = int(a.get("id", 0))
    return ok(f"forgot #{fid}") if ctx.extras["memory"].forget(fid) else ToolResult(False, f"no memory #{fid}")


async def ubg_status(ctx: ToolContext, a: dict) -> ToolResult:
    core = ctx.extras["core"]
    st = await core._fightd("status", timeout=3.0)
    return ok(core.status_text(st, "en") + (f" Last refusal: {st['last_refusal']}." if st.get("last_refusal") else ""))


async def ubg_stop(ctx: ToolContext, a: dict) -> ToolResult:
    r = await ctx.extras["core"]._fightd("disarm", {"reason": "agent"}, timeout=3.0)
    return ok("fight stopped") if r.get("ok", True) else ToolResult(False, str(r.get("error", "could not reach fightd")))


async def notify(ctx: ToolContext, a: dict) -> ToolResult:
    text = str(a.get("text", "")).strip()[:500]
    if not text:
        raise Refused("empty notice")
    ctx.extras["core"].publish("hud.notify", {"text": text})
    return ok("shown on the HUD")


LOCAL_TOOLS = [
    Tool("memory_remember", "Remember a fact or preference about the owner for later conversations.", Cap.MEMORY, memory_remember,
         {"text": s("What to remember, as one sentence")}, ("text",)),
    Tool("memory_recall", "Search long-term memory.", Cap.MEMORY, memory_recall, {"query": s("What to look for"), "k": i("Max results")},
         ("query",)),
    Tool("memory_forget", "Forget one remembered item by its #id.", Cap.MEMORY, memory_forget, {"id": i("Memory id")}, ("id",)),
    Tool("ubg_status", "Status of the untitled boxing game fighter (connection, state, last match). Read-only.", Cap.SYSTEM, ubg_status),
    Tool("ubg_stop", "Stop (disarm) the fighter. Starting fights is only possible by the owner's own command.", Cap.SYSTEM, ubg_stop),
    Tool("notify", "Show a short notice on the holographic HUD.", Cap.SYSTEM, notify, {"text": s("Notice text")}, ("text",)),
]


class CompositeToolBackend:
    """The agent's ToolBackend: core-local tools plus toolhost's (over RPC)."""

    def __init__(self, extras: dict[str, Any], remote) -> None:
        self.local = ToolRegistry(LOCAL_TOOLS, windows=True)
        self.extras = extras
        self.remote = remote
        self.client = getattr(remote, "client", None)  # kill-switch access to toolhost
        self._caps: dict[str, set[Cap]] = {}

    async def schemas(self, caps: set[Cap]) -> list[dict]:
        local = self.local.schemas(caps)
        names = {t["function"]["name"] for t in local}
        return local + [t for t in await self.remote.schemas(caps) if t["function"]["name"] not in names]

    async def open_scope(self, task_id: str, caps: set[Cap], origin: str, request: str) -> None:
        self._caps[task_id] = set(caps)
        if len(self._caps) > 256:
            self._caps.pop(next(iter(self._caps)))
        await self.remote.open_scope(task_id, caps, origin, request)

    async def call(self, task_id: str, name: str, args: dict[str, Any]) -> ToolOutcome:
        tool = self.local.get(name)
        if tool is None:
            return await self.remote.call(task_id, name, args)
        if tool.cap not in self._caps.get(task_id, set()):
            return ToolOutcome(False, f"refused: {name} needs the '{tool.cap.value}' capability, which this request did not include")
        ctx = ToolContext(policy=None, task_id=task_id, extras=self.extras)
        try:
            r = await tool.handler(ctx, args)
        except Refused as e:
            return ToolOutcome(False, f"refused: {e}")
        return ToolOutcome(r.ok, r.text, r.trust, r.source or name, r.image_data_url)
