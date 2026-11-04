"""Tray icon (pystray, Windows notification area): status at a glance, mute/unmute, stop the fight, PANIC,
logs, restart and quit. The icon is drawn here (an arc-reactor ring) and recoloured by state."""

from __future__ import annotations

import contextlib
import logging
import threading
from collections.abc import Callable

log = logging.getLogger(__name__)

ACTIVE_FIGHT = {"armed", "navigating", "queued", "fighting", "post_match", "calibrating"}
COLORS = {
    "ok": (0, 229, 255),  # cyan: everything running
    "starting": (110, 150, 180),
    "degraded": (255, 160, 0),  # a process keeps crashing
    "fighting": (255, 64, 64),
}
LABELS = {
    "mute": "Mute microphone / 마이크 끄기",
    "unmute": "Unmute microphone / 마이크 켜기",
    "disarm": "Stop the fight / 전투 중지",
    "panic": "PANIC STOP / 긴급 정지",
    "logs": "Open logs / 로그 폴더",
    "restart": "Restart JARVIS / 다시 시작",
    "quit": "Quit JARVIS / 종료",
}


def tray_state(health: str, fight_state: str) -> str:
    return "fighting" if fight_state in ACTIVE_FIGHT else (health if health in COLORS else "starting")


def icon_image(state: str, size: int = 64):
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    c = COLORS.get(state, COLORS["starting"])
    m = max(1, size // 12)
    d.ellipse((m, m, size - 1 - m, size - 1 - m), outline=(*c, 255), width=max(2, size // 10))
    r = size // 5
    mid = size // 2
    d.ellipse((mid - r, mid - r, mid + r, mid + r), fill=(*c, 255))
    return img


class Tray:
    def __init__(self, actions: dict[str, Callable[[], object]], status: Callable[[], tuple[str, str]],
                 refresh_s: float = 2.0) -> None:
        import pystray

        self._pystray = pystray
        self.actions = actions
        self.status = status
        self.refresh_s = refresh_s
        self._state = ""
        self._stop = threading.Event()
        self.icon = pystray.Icon("jarvis", icon_image("starting"), "JARVIS", menu=self.menu())

    def _do(self, name: str) -> Callable:
        def run(_icon=None, _item=None) -> None:
            try:
                self.actions[name]()
            except Exception:
                log.exception("tray action %s failed", name)

        return run

    def status_text(self) -> str:
        health, fight = self.status()
        return f"JARVIS: {health} · fight: {fight}"

    def menu(self):
        Item, Menu = self._pystray.MenuItem, self._pystray.Menu
        return Menu(
            Item(lambda _item: self.status_text(), None, enabled=False),
            Menu.SEPARATOR,
            Item(LABELS["panic"], self._do("panic")),
            Item(LABELS["disarm"], self._do("disarm")),
            Menu.SEPARATOR,
            Item(LABELS["mute"], self._do("mute")),
            Item(LABELS["unmute"], self._do("unmute")),
            Menu.SEPARATOR,
            Item(LABELS["logs"], self._do("logs")),
            Item(LABELS["restart"], self._do("restart")),
            Item(LABELS["quit"], self._do("quit")),
        )

    def refresh(self) -> bool:
        st = tray_state(*self.status())
        if st == self._state:
            return False
        self._state = st
        self.icon.icon = icon_image(st)
        self.icon.title = self.status_text()
        return True

    def _refresher(self) -> None:
        while not self._stop.wait(self.refresh_s):
            with contextlib.suppress(Exception):
                self.refresh()
                self.icon.update_menu()

    def start(self) -> None:
        def setup(icon) -> None:
            icon.visible = True
            self.refresh()

        threading.Thread(target=self.icon.run, kwargs={"setup": setup}, name="tray", daemon=True).start()
        threading.Thread(target=self._refresher, name="tray-refresh", daemon=True).start()

    def notify(self, text: str) -> None:
        with contextlib.suppress(Exception):
            self.icon.notify(text[:250], "JARVIS")

    def stop(self) -> None:
        self._stop.set()
        with contextlib.suppress(Exception):
            self.icon.stop()
