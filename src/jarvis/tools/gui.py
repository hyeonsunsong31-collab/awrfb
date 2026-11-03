"""GUI control of other apps. UI Automation first: invoking a button or setting a field's value needs no
synthetic mouse or keyboard input at all. When the model needs to see, it gets a screenshot with numbered
marks over the controls (set-of-marks) and clicks by number; OCR reads text UIA cannot.

Refused: payment/purchase controls, typing card numbers, the Roblox window, and any input while Roblox has
focus (input would reach the game)."""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
from typing import Any

from jarvis.safety.capabilities import Cap
from jarvis.tools.base import Refused, Tool, ToolContext, ToolResult, b, clip, i, ok, s
from jarvis.tools.windows import find_window

CLICKABLE = {"ButtonControl", "MenuItemControl", "HyperlinkControl", "ListItemControl", "TabItemControl", "CheckBoxControl",
             "RadioButtonControl", "TreeItemControl", "ComboBoxControl", "SplitButtonControl", "EditControl", "DocumentControl"}
MAX_MARKS = 60


class _UiaThread:
    """uiautomation is COM: every call runs on one dedicated, COM-initialized thread."""

    def __init__(self) -> None:
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="uia", initializer=self._init)

    @staticmethod
    def _init() -> None:
        import uiautomation as auto

        _UiaThread._keep = auto.UIAutomationInitializerInThread()  # type: ignore[attr-defined]

    async def run(self, fn, *args):
        return await asyncio.get_running_loop().run_in_executor(self._pool, fn, *args)


_uia: _UiaThread | None = None


def uia() -> _UiaThread:
    global _uia
    if _uia is None:
        _uia = _UiaThread()
    return _uia


# ---------------------------------------------------------------------------------- policy gates
def check_name(ctx: ToolContext, name: str) -> None:
    d = ctx.policy.check_ui_element(name or "")
    if not d:
        raise Refused(d.reason)


def guard_input(ctx: ToolContext) -> None:
    fg = ctx.plat.windows.foreground() if ctx.plat is not None else None
    if fg is not None and ctx.policy.is_roblox(fg.title, fg.process_name):
        raise Refused("the game has focus; any input now would reach Roblox, so I won't")


def target_window(ctx: ToolContext, title: str):
    w = find_window(ctx, title)  # refuses the Roblox window
    return w


# ---------------------------------------------------------------------------------- UIA helpers (uia thread)
def _walk(hwnd: int, max_depth: int = 14, limit: int = 500) -> list[Any]:
    import uiautomation as auto

    root = auto.ControlFromHandle(hwnd)
    out, frontier = [], [(root, 0)]
    while frontier and len(out) < limit:
        c, depth = frontier.pop(0)
        out.append(c)
        if depth < max_depth:
            try:
                frontier.extend((ch, depth + 1) for ch in c.GetChildren())
            except Exception:
                continue
    return out


def _label(c: Any) -> str:
    kind = c.ControlTypeName.replace("Control", "")
    aid = f" id={c.AutomationId}" if getattr(c, "AutomationId", "") else ""
    return f"{kind} '{c.Name}'{aid}"


def _visible(c: Any) -> bool:
    try:
        r = c.BoundingRectangle
        return not c.IsOffscreen and r.right > r.left and r.bottom > r.top
    except Exception:
        return False


def _activate(c: Any) -> str:
    """Prefer patterns that move no mouse and send no keys."""
    for getter, action in (("GetInvokePattern", "Invoke"), ("GetTogglePattern", "Toggle"),
                           ("GetSelectionItemPattern", "Select"), ("GetExpandCollapsePattern", "Expand")):
        try:
            pat = getattr(c, getter)()
            if pat:
                getattr(pat, action)()
                return action.lower()
        except Exception:
            continue
    c.Click(simulateMove=False)
    return "click"


def _find(hwnd: int, name: str, kinds: set[str] | None = None) -> Any | None:
    q = name.strip().lower()
    cands = [c for c in _walk(hwnd) if (kinds is None or c.ControlTypeName in kinds) and q in (c.Name or "").lower() and _visible(c)]
    cands.sort(key=lambda c: ((c.Name or "").lower() != q, c.ControlTypeName not in CLICKABLE, len(c.Name or "")))
    return cands[0] if cands else None


# ---------------------------------------------------------------------------------- tools
async def ui_find(ctx: ToolContext, a: dict) -> ToolResult:
    w = target_window(ctx, str(a.get("window", "")))
    q = str(a.get("contains", "")).lower()
    kind = str(a.get("control_type", "")).strip()

    def run() -> list[Any]:
        found = [c for c in _walk(w.hwnd) if _visible(c) and (c.Name or kind)]
        if q:
            found = [c for c in found if q in (c.Name or "").lower()]
        if kind:
            found = [c for c in found if c.ControlTypeName.lower().startswith(kind.lower())]
        return found[:MAX_MARKS]

    found = await uia().run(run)
    ctx.state["marks"] = {k + 1: (w.hwnd, c) for k, c in enumerate(found)}
    lines = [f"{k + 1}: {_label(c)}" for k, c in enumerate(found)]
    return ok(f"{w.title}:\n" + ("\n".join(lines) or "no matching controls"), trust="untrusted", source=w.title)


async def ui_click(ctx: ToolContext, a: dict) -> ToolResult:
    name = str(a.get("name", ""))
    check_name(ctx, name)
    w = target_window(ctx, str(a.get("window", "")))
    guard_input(ctx)

    def run() -> str:
        c = _find(w.hwnd, name)
        if c is None:
            return ""
        check_name(ctx, c.Name or "")  # the real control's name, not just the request
        return f"{_activate(c)}ed {_label(c)}"

    did = await uia().run(run)
    return ok(did) if did else ToolResult(False, f"no visible control named {name!r} in {w.title}")


async def ui_type(ctx: ToolContext, a: dict) -> ToolResult:
    text = str(a.get("text", ""))
    d = ctx.policy.check_typed_text(text)
    if not d:
        raise Refused(d.reason)
    field = str(a.get("field", ""))
    w = target_window(ctx, str(a.get("window", "")))
    guard_input(ctx)
    submit = bool(a.get("submit", False))

    def run() -> str:
        c = _find(w.hwnd, field, {"EditControl", "DocumentControl", "ComboBoxControl"}) if field else None
        if c is None and field:
            return ""
        try:
            pat = c.GetValuePattern() if c is not None else None
            if pat and not submit:
                pat.SetValue(text)
                return f"set {_label(c)}"
        except Exception:
            pass
        ctx.plat.windows.focus(w.hwnd)
        guard_input(ctx)  # re-check right before keystrokes
        if c is not None:
            c.SetFocus()
        import uiautomation as auto

        auto.SendKeys(text.replace("{", "{{}").replace("}", "{}}") + ("{Enter}" if submit else ""), interval=0.01, waitTime=0.05)
        return f"typed {len(text)} characters into {(_label(c) if c is not None else w.title)}"

    did = await uia().run(run)
    return ok(did) if did else ToolResult(False, f"no text field named {field!r} in {w.title}")


def _grab(rect) -> Any:
    import numpy as np
    from PIL import ImageGrab

    img = ImageGrab.grab(bbox=(rect.left, rect.top, rect.right, rect.bottom), all_screens=True) if rect else ImageGrab.grab(all_screens=True)
    return np.asarray(img.convert("RGB"))[:, :, ::-1].copy()  # BGR for OpenCV


def _jpeg_data_url(bgr: Any, max_w: int = 1280) -> str:
    import cv2

    h, w = bgr.shape[:2]
    if w > max_w:
        bgr = cv2.resize(bgr, (max_w, round(h * max_w / w)), interpolation=cv2.INTER_AREA)
    okk, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 72])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if okk else ""


async def screen_capture(ctx: ToolContext, a: dict) -> ToolResult:
    import cv2

    title = str(a.get("window", "")).strip()
    w = target_window(ctx, title) if title else None
    rect = ctx.plat.windows.client_rect_screen(w.hwnd) if (w and ctx.plat) else None

    def run() -> tuple[Any, dict, list[str]]:
        img = _grab(rect)
        marks: dict[int, tuple[int, Any]] = {}
        lines: list[str] = []
        if w is not None:
            ox, oy = (rect.left, rect.top) if rect else (0, 0)
            for c in [c for c in _walk(w.hwnd) if c.ControlTypeName in CLICKABLE and _visible(c)][:MAX_MARKS]:
                k = len(marks) + 1
                r = c.BoundingRectangle
                x0, y0, x1, y1 = r.left - ox, r.top - oy, r.right - ox, r.bottom - oy
                cv2.rectangle(img, (x0, y0), (x1, y1), (0, 200, 255), 2)
                cv2.putText(img, str(k), (x0 + 2, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
                cv2.putText(img, str(k), (x0 + 2, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1)
                marks[k] = (w.hwnd, c)
                lines.append(f"{k}: {_label(c)}")
        return img, marks, lines

    img, marks, lines = await uia().run(run)
    ctx.state["marks"] = marks
    note = ("Numbered controls (click with ui_click_mark):\n" + "\n".join(lines)) if lines else "Screenshot of the whole screen."
    return ok(note, trust="untrusted", source=w.title if w else "screen", image=_jpeg_data_url(img))


async def ui_click_mark(ctx: ToolContext, a: dict) -> ToolResult:
    n = int(a.get("number", 0))
    mark = ctx.state.get("marks", {}).get(n)
    if mark is None:
        return ToolResult(False, f"no mark {n}; take a new screen_capture first")
    _hwnd, c = mark
    guard_input(ctx)

    def run() -> str:
        check_name(ctx, c.Name or "")
        return f"{_activate(c)}ed {_label(c)}"

    return ok(await uia().run(run))


async def screen_ocr(ctx: ToolContext, a: dict) -> ToolResult:
    title = str(a.get("window", "")).strip()
    w = target_window(ctx, title) if title else None
    rect = ctx.plat.windows.client_rect_screen(w.hwnd) if (w and ctx.plat) else None
    img = await asyncio.to_thread(_grab, rect)
    text = await asyncio.to_thread(ocr_text, img)
    return ok(clip(text or "(no text found)", 20000), trust="untrusted", source=w.title if w else "screen")


_ocr_engine: Any = None


def ocr_text(bgr: Any) -> str:
    """RapidOCR (Korean + English); supports the 3.x result object and the older list format."""
    global _ocr_engine
    if _ocr_engine is None:
        try:
            from rapidocr import RapidOCR
        except ImportError as e:
            raise Refused("OCR is not installed (install the 'ocr' extra)") from e
        _ocr_engine = RapidOCR()
    out = _ocr_engine(bgr)
    txts = getattr(out, "txts", None)
    if txts is not None:
        return "\n".join(t for t in txts if t)
    result = out[0] if isinstance(out, tuple) else out
    return "\n".join(str(r[1]) for r in (result or []) if len(r) > 1)


TOOLS = [
    Tool("ui_find", "List the controls (buttons, fields, menu items) of an app window, numbered.", Cap.SCREEN, ui_find,
         {"window": s("Part of the window title"), "contains": s("Only controls whose name contains this"),
          "control_type": s("e.g. Button, Edit, MenuItem")}, ("window",), trust="untrusted", windows_only=True),
    Tool("ui_click", "Click/activate a control by its name in an app window (UI Automation).", Cap.GUI, ui_click,
         {"window": s("Part of the window title"), "name": s("The control's visible name")}, ("window", "name"), windows_only=True),
    Tool("ui_type", "Type text into a field of an app window (optionally press Enter).", Cap.GUI, ui_type,
         {"window": s("Part of the window title"), "field": s("The field's name (empty: the focused field)"),
          "text": s("Text to type"), "submit": b("Press Enter afterwards")}, ("window", "text"), windows_only=True),
    Tool("screen_capture", "Screenshot of a window (with numbered controls) or of the whole screen, for you to look at.",
         Cap.SCREEN, screen_capture, {"window": s("Part of the window title (empty: whole screen)")}, trust="untrusted",
         windows_only=True),
    Tool("ui_click_mark", "Click the control with this number from the last screen_capture/ui_find.", Cap.GUI, ui_click_mark,
         {"number": i("Mark number")}, ("number",), windows_only=True),
    Tool("screen_ocr", "Read the text on screen or in a window (OCR, Korean and English).", Cap.SCREEN, screen_ocr,
         {"window": s("Part of the window title (empty: whole screen)")}, trust="untrusted", windows_only=True),
]

