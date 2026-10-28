"""Launch or attach to the real untitled boxing game in the Roblox desktop app.

The owner signs in to the Roblox app themselves; JARVIS only opens the documented ``roblox://``
deep link and then finds the player window. It never sees or stores Roblox credentials.

Server policy: bots in public servers risk a ban (UBG's rules name auto-dodging; Roblox's ToS bans
unfair automation). Fighting is therefore only allowed in a private server JARVIS itself launched in
this Roblox session, unless the owner explicitly opted into public servers in config.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from jarvis.common.config import GameConfig
from jarvis.platform.base import Platform, WindowRef

log = logging.getLogger(__name__)


class LaunchError(RuntimeError):
    pass


@dataclass
class GameSession:
    pid: int
    hwnd: int
    server_kind: str  # "private" | "public" | "unknown"
    launched_by_jarvis: bool
    started_at: float
    link_code_sha: str | None = None


def launch_url(game: GameConfig, *, private: bool) -> str:
    if private:
        code = (game.private_server_link_code or "").strip()
        if not code:
            raise LaunchError(
                "No private server configured. Put your UBG private server share link (or its linkCode) in "
                "fight.game.private_server_link_code."
            )
        if code.startswith(("https://", "roblox://", "roblox-player:")):
            return code  # a full share link opens Roblox directly
        return game.private_server_url_template.format(place_id=game.place_id, link_code=code)
    return game.launch_url_template.format(place_id=game.place_id)


def fight_allowed_in(session: GameSession | None, game: GameConfig) -> tuple[bool, str]:
    """Server policy check used when arming a fight."""
    if game.allowed_servers == "public" and game.i_accept_ban_risk:
        return True, "public servers allowed by owner (ban risk accepted)"
    if session is not None and session.server_kind == "private" and session.launched_by_jarvis:
        return True, "private server launched by JARVIS"
    return False, (
        "I only fight in a private server by default - bots in public servers can get your account "
        "banned. Say 'launch UBG private server' (needs fight.game.private_server_link_code), or set "
        "fight.game.allowed_servers: public and i_accept_ban_risk: true to override."
    )


def open_url(url: str) -> None:
    if sys.platform == "win32":
        os.startfile(url)  # type: ignore[attr-defined]  # protocol handler registered by Roblox
    else:
        raise LaunchError("launching Roblox needs Windows")


def find_player_window(plat: Platform, game: GameConfig) -> WindowRef | None:
    for w in plat.windows.find_windows(process_name=game.process_name):
        if w.title.strip().lower() == game.window_title.lower():
            return w
    # Microsoft Store build / renamed process: fall back to an exact title match.
    for w in plat.windows.find_windows(title_substring=game.window_title):
        if w.title.strip().lower() == game.window_title.lower():
            return w
    return None


class GameLauncher:
    def __init__(self, plat: Platform, game: GameConfig, state_dir: Path) -> None:
        self._plat = plat
        self._game = game
        self._state_file = state_dir / "game_session.json"
        self.session: GameSession | None = self._load()

    def _load(self) -> GameSession | None:
        try:
            data = json.loads(self._state_file.read_text(encoding="utf-8"))
            return GameSession(**data)
        except (OSError, ValueError, TypeError):
            return None

    def _save(self) -> None:
        if self.session is None:
            return
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        self._state_file.write_text(json.dumps(asdict(self.session)), encoding="utf-8")

    def launch(self, *, private: bool, wait: bool = True) -> GameSession:
        url = launch_url(self._game, private=private)
        before = find_player_window(self._plat, self._game)
        log.info("launching UBG (%s server)", "private" if private else "public")
        open_url(url)
        if not wait:
            raise LaunchError("launch requested without waiting; attach later")
        deadline = time.monotonic() + self._game.launch_timeout_s
        while time.monotonic() < deadline:
            w = find_player_window(self._plat, self._game)
            # Joining a new server replaces the player process; an unchanged PID means we cannot
            # know which server the window is in, so we never label it "private" by guesswork.
            if w is not None and (before is None or w.pid != before.pid):
                self.session = GameSession(
                    pid=w.pid,
                    hwnd=w.hwnd,
                    server_kind="private" if private else "public",
                    launched_by_jarvis=True,
                    started_at=time.time(),
                    link_code_sha=hashlib.sha256(self._game.private_server_link_code.encode()).hexdigest()[:12]
                    if private and self._game.private_server_link_code
                    else None,
                )
                self._save()
                return self.session
            time.sleep(0.5)
        if before is not None:
            raise LaunchError(
                "Roblox was already running and did not switch servers. Close Roblox and launch again "
                "so I know which server we're in."
            )
        raise LaunchError("the Roblox player window did not appear - is Roblox installed and signed in?")

    def attach(self) -> GameSession:
        w = find_player_window(self._plat, self._game)
        if w is None:
            raise LaunchError("Roblox is not running. Start UBG first (or say 'launch UBG').")
        if self.session is not None and self.session.pid == w.pid:
            if self.session.hwnd != w.hwnd:
                self.session.hwnd = w.hwnd
                self._save()
            return self.session
        # A Roblox we didn't launch: we cannot know which server it is.
        self.session = GameSession(w.pid, w.hwnd, "unknown", False, time.time())
        self._save()
        return self.session

    def window(self) -> WindowRef | None:
        if self.session is None:
            return None
        w = find_player_window(self._plat, self._game)
        if w is None or w.pid != self.session.pid:
            return None
        return w
