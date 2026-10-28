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


class SimFight(SimGame):
    """Adds a match: hits we 'land' lower the opponent's bar, the opponent hits back periodically,
    bars refill per stock, and the HUD disappears after the 3rd stock. Plumbing tests only."""

    def __init__(self, size=(960, 540), opp_hit_every_s: float = 1.2, start_delay_s: float = 0.3) -> None:
        super().__init__(size=size)
        self.t0 = time.monotonic()
        self.start_delay_s = start_delay_s
        self.opp_hit_every_s = opp_hit_every_s
        self.next_opp_hit = self.t0 + start_delay_s + opp_hit_every_s
        self.stocks_won = 0
        self.stocks_lost = 0
        self.over = False
        self.presses: list[str] = []

    def on_input(self, kind: str, phys: str) -> None:
        super().on_input(kind, phys)
        if kind != "down" or self.over:
            return
        self.presses.append(phys)
        dmg = {"mouse:left": 0.12, "mouse:right": 0.25}.get(phys, 0.0)
        with self.lock:
            self.opp_hp = max(0.0, self.opp_hp - dmg)

    def render(self):
        now = time.monotonic()
        with self.lock:
            if not self.over and now > self.next_opp_hit:
                self.own_hp = max(0.0, self.own_hp - 0.05)
                self.next_opp_hit = now + self.opp_hit_every_s
            if self.opp_hp <= 0.0 and not self.over and getattr(self, "_refill_at", None) is None \
                    and getattr(self, "_over_at", None) is None:
                self.stocks_won += 1
                if self.stocks_won >= 3:
                    self._over_at = now + 0.4  # the KO'd bar stays on screen briefly
                else:
                    self._refill_at = now + 0.15
            if getattr(self, "_refill_at", None) and now > self._refill_at:
                self.opp_hp = 1.0
                self._refill_at = None
            if getattr(self, "_over_at", None) and now > self._over_at:
                self.over = True
        if self.over or now - self.t0 < self.start_delay_s:
            return scene_only(self.size)
        return super().render()


def scene_only(size):
    from fixtures.hud_synth import scene

    return scene(size[0], size[1], seed=9)
