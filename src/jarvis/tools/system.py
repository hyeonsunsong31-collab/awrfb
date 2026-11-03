"""System tools: volume, media keys, clipboard, system information."""

from __future__ import annotations

import asyncio
import time

from jarvis.safety.capabilities import Cap
from jarvis.tools.base import Refused, Tool, ToolContext, ToolResult, clip, i, ok, s

MEDIA = {"play_pause": 0xB3, "next": 0xB0, "previous": 0xB1, "stop": 0xB2, "mute": 0xAD}


def roblox_in_front(ctx: ToolContext) -> bool:
    if ctx.plat is None:
        return False
    fg = ctx.plat.windows.foreground()
    return fg is not None and ctx.policy.is_roblox(fg.title, fg.process_name)


async def volume(ctx: ToolContext, a: dict) -> ToolResult:
    from jarvis.platform.win.volume import change_volume, get_volume

    if "level" in a and a["level"] is not None:
        target = max(0, min(int(a["level"]), 100)) / 100
        cur = await asyncio.to_thread(get_volume)
        new = await asyncio.to_thread(change_volume, target - cur)
    else:
        new = await asyncio.to_thread(change_volume, max(-100, min(int(a.get("delta", 0)), 100)) / 100)
    return ok(f"volume {round(new * 100)}%")


async def media(ctx: ToolContext, a: dict) -> ToolResult:
    action = str(a.get("action", "")).lower()
    if action not in MEDIA:
        raise Refused(f"action must be one of {', '.join(MEDIA)}")
    if roblox_in_front(ctx):
        # A synthetic key would be delivered to the game window first; never send input to Roblox.
        raise Refused("the game has focus; media keys would go to Roblox, so I won't send them now")
    from jarvis.platform.win.keys import send_vk

    await asyncio.to_thread(send_vk, MEDIA[action])
    return ok(f"media: {action}")


async def clipboard_get(ctx: ToolContext, a: dict) -> ToolResult:
    from jarvis.platform.win.clipboard import get_text

    text = await asyncio.to_thread(get_text)
    return ok(clip(text or "(clipboard is empty)", 20000), trust="untrusted", source="clipboard")


async def clipboard_set(ctx: ToolContext, a: dict) -> ToolResult:
    text = str(a.get("text", ""))
    d = ctx.policy.check_typed_text(text)
    if not d:
        raise Refused(d.reason)
    from jarvis.platform.win.clipboard import set_text

    await asyncio.to_thread(set_text, text)
    return ok(f"copied {len(text)} characters to the clipboard")


async def system_info(ctx: ToolContext, a: dict) -> ToolResult:
    import psutil

    from jarvis.common.gpu import primary_gpu

    def collect() -> str:
        vm = psutil.virtual_memory()
        lines = [f"CPU {psutil.cpu_percent(0.3):.0f}% of {psutil.cpu_count()} threads",
                 f"RAM {vm.used / 2**30:.1f} / {vm.total / 2**30:.1f} GB ({vm.percent:.0f}%)"]
        g = primary_gpu()
        if g:
            util = f", {g.util_pct}% busy" if g.util_pct is not None else ""
            lines.append(f"GPU {g.name}: {g.used_mb / 1024:.1f} / {g.total_mb / 1024:.1f} GB VRAM{util}")
        for part in psutil.disk_partitions(all=False):
            try:
                u = psutil.disk_usage(part.mountpoint)
            except OSError:
                continue
            lines.append(f"disk {part.mountpoint} {u.free / 2**30:.0f} GB free of {u.total / 2**30:.0f} GB")
        bat = psutil.sensors_battery() if hasattr(psutil, "sensors_battery") else None
        if bat:
            lines.append(f"battery {bat.percent:.0f}%{' (charging)' if bat.power_plugged else ''}")
        up = time.time() - psutil.boot_time()
        lines.append(f"up {int(up // 3600)} h {int(up % 3600 // 60)} min")
        return "\n".join(lines)

    return ok(await asyncio.to_thread(collect), source="system")


TOOLS = [
    Tool("volume", "Set the speaker volume (level 0-100) or change it (delta, e.g. -10).", Cap.SYSTEM, volume,
         {"level": i("Target volume 0-100"), "delta": i("Change in percent points")}, windows_only=True),
    Tool("media", "Media keys: play_pause, next, previous, stop, mute.", Cap.SYSTEM, media,
         {"action": s("play_pause | next | previous | stop | mute")}, ("action",), windows_only=True),
    Tool("clipboard_get", "Read the clipboard text (untrusted).", Cap.CLIPBOARD, clipboard_get, trust="untrusted", windows_only=True),
    Tool("clipboard_set", "Put text on the clipboard.", Cap.CLIPBOARD, clipboard_set, {"text": s("Text to copy")}, ("text",),
         windows_only=True),
    Tool("system_info", "CPU, memory, GPU, disks, battery and uptime.", Cap.SYSTEM, system_info),
]
