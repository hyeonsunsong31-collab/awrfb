"""Filesystem locations. Nothing here creates secrets; tokens live in ``run_dir`` (user-private)."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from platformdirs import user_data_dir

APP_NAME = "Jarvis"


def repo_root() -> Path:
    """The source checkout root (``src/jarvis/common/paths.py`` -> three levels up)."""
    return Path(__file__).resolve().parents[3]


def config_dir() -> Path:
    env = os.environ.get("JARVIS_CONFIG_DIR")
    if env:
        return Path(env)
    return repo_root() / "configs"


def default_data_dir() -> Path:
    env = os.environ.get("JARVIS_DATA_DIR")
    if env:
        return Path(env)
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / APP_NAME
    return Path(user_data_dir(APP_NAME.lower(), appauthor=False))


def protected_policy_path() -> Path:
    """Where the installer writes the read-only safety policy (admin-owned, user read-only)."""
    env = os.environ.get("JARVIS_POLICY_PATH")
    if env:
        return Path(env)
    if sys.platform == "win32":
        base = os.environ.get("PROGRAMDATA") or r"C:\ProgramData"
        return Path(base) / APP_NAME / "safety.yaml"
    return Path("/etc/jarvis/safety.yaml")


@dataclass(frozen=True)
class DataPaths:
    root: Path

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def recordings(self) -> Path:
        return self.root / "recordings"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def audit(self) -> Path:
        return self.root / "audit"

    @property
    def run(self) -> Path:
        return self.root / "run"

    @property
    def db(self) -> Path:
        return self.root / "db"

    @property
    def datasets(self) -> Path:
        return self.root / "datasets"

    @property
    def calibration(self) -> Path:
        return self.root / "calibration"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def tts_cache(self) -> Path:
        return self.root / "tts_cache"

    @property
    def user_config(self) -> Path:
        return self.root / "config.yaml"

    def ensure(self) -> DataPaths:
        for p in (
            self.root,
            self.models,
            self.recordings,
            self.logs,
            self.audit,
            self.run,
            self.db,
            self.datasets,
            self.calibration,
            self.reports,
            self.tts_cache,
        ):
            p.mkdir(parents=True, exist_ok=True)
        return self


def data_paths(root: Path | None = None) -> DataPaths:
    return DataPaths(root or default_data_dir())
