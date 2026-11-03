"""Desktop windows: list, focus, minimize/maximize/restore/close. The Roblox window is off limits."""

from __future__ import annotations

from jarvis.safety.capabilities import Cap
from jarvis.tools.base import Refused, Tool, ToolContext, ToolResult, ok, s

GAME_REFUSAL = "the Roblox window belongs to the fight system; I don't touch it"
ACTIONS = {"minimize": 6, "maximize": 3, "restore": 9}
WM_CLOSE = 0x0010


def _wm(ctx: ToolContext):
    if ctx.plat is None:
        raise Refused("window control is not available on this system")
    return ctx.plat.windows


def _is_game(ctx: ToolContext, w) -> bool:
    return ctx.policy.is_roblox(w.title, w.process_name)


def find_window(ctx: ToolContext, title: str):
    wins = [w for w in _wm(ctx).find_windows() if w.title.strip()]
    q = title.strip().lower()
    if not q:
        raise Refused("which window?")
    hits = [w for w in wins if q in w.title.lower()] or [w for w in wins if q in (w.process_name or "").lower()]
    if not hits:
        from rapidfuzz import fuzz, process

        best = process.extractOne(q, [w.title for w in wins], scorer=fuzz.WRatio, score_cutoff=75)
        hits = [wins[best[2]]] if best else []
    if not hits:
        raise Refused(f"no open window matches {title!r}")
    w = hits[0]
    if _is_game(ctx, w):
        raise Refused(GAME_REFUSAL)
    return w


def win_action(hwnd: int, action: str) -> bool:
    from jarvis.platform.win.api import user32

    u = user32()
    if action == "close":
        return bool(u.PostMessageW(hwnd, WM_CLOSE, 0, 0))
    return bool(u.ShowWindow(hwnd, ACTIONS[action]))


async def window_list(ctx: ToolContext, a: dict) -> ToolResult:
    rows = []
    for w in _wm(ctx).find_windows():
        if not w.title.strip():
            continue
        tag = "  (game - off limits)" if _is_game(ctx, w) else ""
        rows.append(f"{w.title}  [{w.process_name or '?'}]{tag}")
    return ok("\n".join(rows) or "no windows", trust="untrusted", source="windows")


async def window_focus(ctx: ToolContext, a: dict) -> ToolResult:
    w = find_window(ctx, str(a.get("title", "")))
    return ok(f"focused {w.title}") if _wm(ctx).focus(w.hwnd) else ToolResult(False, f"could not focus {w.title}")


async def window_action(ctx: ToolContext, a: dict) -> ToolResult:
    action = str(a.get("action", "")).lower()
    if action not in (*ACTIONS, "close"):
        raise Refused("action must be minimize, maximize, restore or close")
    w = find_window(ctx, str(a.get("title", "")))
    return ok(f"{action} {w.title}") if win_action(w.hwnd, action) else ToolResult(False, f"could not {action} {w.title}")


TOOLS = [
    Tool("window_list", "List open windows.", Cap.WINDOWS, window_list, trust="untrusted", windows_only=True),
    Tool("window_focus", "Bring a window to the front (by part of its title or app name).", Cap.WINDOWS, window_focus,
         {"title": s("Part of the window title or process name")}, ("title",), windows_only=True),
    Tool("window_action", "Minimize, maximize, restore or close a window.", Cap.WINDOWS, window_action,
         {"title": s("Part of the window title"), "action": s("minimize | maximize | restore | close")}, ("title", "action"),
         windows_only=True),
]
