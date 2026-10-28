"""A tiny simulated 'game window' for PIPELINE tests: it reacts to the fake input sink and produces
frames through a capture-like thread. It tests our plumbing (actuator -> input -> frames -> analysis),
never fighting skill, and is never used outside tests."""

from __future__ import annotations

import threading
import time

import numpy as np

from fakes.platform import FakeInputSink
from fixtures.hud_synth import draw_anchor, ubg_like
from jarvis.platform.base import Frame


class SimGame:
    def __init__(self, react: set[str] | None = None, size=(960, 540)) -> None:
        self.react = react  # physical inputs the "game" reacts to (None = all)
        self.size = size
        self.stamina = 1.0
        self.own_hp = 1.0
        self.opp_hp = 1.0
        self.anim_until = 0.0
        self.lock = threading.Lock()
        self._base = ubg_like(size=size)
        draw_anchor(self._base)

    def on_input(self, kind: str, phys: str) -> None:
        if kind != "down":
            return
        if self.react is not None and phys not in self.react:
            return
        with self.lock:
            self.anim_until = time.monotonic() + 0.25
            if phys in ("key:space", "mouse:left", "mouse:right"):
                self.stamina = max(0.0, self.stamina - 0.1)

    def render(self) -> np.ndarray:
        with self.lock:
            animating = time.monotonic() < self.anim_until
            st = self.stamina
            self.stamina = min(1.0, self.stamina + 0.002)
        img = ubg_like(own_hp=self.own_hp, opp_hp=self.opp_hp, stamina=st, size=self.size)
        draw_anchor(img)
        if animating:
            H, W = img.shape[:2]
            cx = int(W * (0.45 + 0.1 * np.sin(time.monotonic() * 40)))
            img[int(H * 0.4) : int(H * 0.8), cx - 40 : cx + 40] = (250, 250, 250)
        return img


class SimSink(FakeInputSink):
    def __init__(self, game: SimGame, forbidden_keys=()) -> None:
        super().__init__(forbidden_keys)
        self.game = game

    def key_down(self, key: str) -> None:
        super().key_down(key)
        self.game.on_input("down", self.events[-1][1])

    def mouse_down(self, button) -> None:
        super().mouse_down(button)
        self.game.on_input("down", self.events[-1][1])


class SimFrameSource:
    def __init__(self, game: SimGame, fps: int = 60, **_kw) -> None:
        self.game = game
        self.fps = fps
        self._stop = threading.Event()
        self._t: threading.Thread | None = None

    def start(self, window, on_frame) -> None:
        def run() -> None:
            i = 0
            while not self._stop.is_set():
                i += 1
                on_frame(Frame(self.game.render(), time.perf_counter_ns(), i))
                time.sleep(1 / self.fps)

        self._t = threading.Thread(target=run, daemon=True)
        self._t.start()

    def stop(self) -> None:
        self._stop.set()
        if self._t:
            self._t.join(timeout=1)

    def running(self) -> bool:
        return self._t is not None and self._t.is_alive()
