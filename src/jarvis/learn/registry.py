"""Model registry: candidate -> shadow -> active -> retired, with one-step rollback.

A model is only promoted when it measured better on held-out sessions (see jarvis.learn.evaluate).
The active file is copied to ``models/<name>/active.onnx`` (copy, not symlink, for Windows), which is
what fightd loads. Writes are atomic (temp file + os.replace).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class ModelEntry:
    name: str
    version: int
    path: str  # relative to the models dir
    sha256: str
    dataset_version: str
    metrics: dict[str, Any]
    status: str  # candidate | shadow | active | retired
    created: float = field(default_factory=time.time)
    game_fingerprint: str | None = None
    notes: str = ""


class ModelRegistry:
    def __init__(self, models_dir: Path) -> None:
        self.dir = models_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.file = self.dir / "registry.json"
        self.entries: list[ModelEntry] = self._load()

    def _load(self) -> list[ModelEntry]:
        if not self.file.exists():
            return []
        data = json.loads(self.file.read_text(encoding="utf-8"))
        return [ModelEntry(**e) for e in data.get("models", [])]

    def _save(self) -> None:
        payload = json.dumps({"v": 1, "models": [asdict(e) for e in self.entries]}, indent=2)
        fd, tmp = tempfile.mkstemp(dir=self.dir, prefix=".registry-", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
        os.replace(tmp, self.file)

    def versions(self, name: str) -> list[ModelEntry]:
        return sorted((e for e in self.entries if e.name == name), key=lambda e: e.version)

    def active(self, name: str) -> ModelEntry | None:
        return next((e for e in self.entries if e.name == name and e.status == "active"), None)

    def register(self, name: str, file: Path, *, dataset_version: str, metrics: dict[str, Any],
                 game_fingerprint: str | None = None, notes: str = "") -> ModelEntry:
        version = (self.versions(name)[-1].version + 1) if self.versions(name) else 1
        dest = self.dir / name / f"v{version}{file.suffix}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, dest)
        e = ModelEntry(name, version, str(dest.relative_to(self.dir)), sha256_file(dest), dataset_version, metrics,
                       "candidate", game_fingerprint=game_fingerprint, notes=notes)
        self.entries.append(e)
        self._save()
        return e

    def set_status(self, name: str, version: int, status: str) -> ModelEntry:
        e = next(x for x in self.entries if x.name == name and x.version == version)
        e.status = status
        self._save()
        return e

    def promote(self, name: str, version: int) -> ModelEntry:
        target = next(x for x in self.entries if x.name == name and x.version == version)
        src = self.dir / target.path
        if sha256_file(src) != target.sha256:
            raise RuntimeError(f"{name} v{version} file changed on disk (hash mismatch); refusing to promote")
        for x in self.entries:
            if x.name == name and x.status == "active":
                x.status = "retired"
        target.status = "active"
        active_path = self.dir / name / f"active{src.suffix}"
        tmp = active_path.with_suffix(active_path.suffix + ".tmp")
        shutil.copy2(src, tmp)
        os.replace(tmp, active_path)
        self._save()
        return target

    def rollback(self, name: str) -> ModelEntry | None:
        """Re-activate the most recent retired version (e.g. a promoted model misbehaves live)."""
        cur = self.active(name)
        retired = [e for e in self.versions(name) if e.status == "retired" and (cur is None or e.version != cur.version)]
        if not retired:
            return None
        prev = retired[-1]
        if cur is not None:
            cur.status = "rejected"
        return self.promote(name, prev.version)
