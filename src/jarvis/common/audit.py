"""Append-only, hash-chained audit log (JSON Lines).

Every record carries ``prev`` (the previous record's hash) and ``hash`` = sha256 of the canonical JSON
of the record without ``hash``. Editing or deleting any line breaks the chain, which ``verify_chain``
detects. Each process writes its own file per day; the chain continues across daily files.
Secrets are redacted before hashing so they never reach disk.
"""

from __future__ import annotations

import hashlib
import json
import queue
import re
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

GENESIS = "0" * 64
_SECRET_KEYS = re.compile(r"(api[_-]?key|token|password|passwd|secret|authorization|cookie)", re.I)
_NVAPI = re.compile(r"nvapi-[A-Za-z0-9_\-]{8,}")


def redact(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: ("***" if _SECRET_KEYS.search(str(k)) else redact(v)) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [redact(v) for v in obj]
    if isinstance(obj, str):
        return _NVAPI.sub("nvapi-***", obj)
    return obj


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def record_hash(rec: dict[str, Any]) -> str:
    body = {k: v for k, v in rec.items() if k != "hash"}
    return hashlib.sha256(canonical(body).encode("utf-8")).hexdigest()


def _last_record(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("rb") as f:
            f.seek(0, 2)
            end = f.tell()
            if end == 0:
                return None
            chunk = min(end, 65536)
            f.seek(end - chunk)
            lines = f.read(chunk).splitlines()
    except OSError:
        return None
    for raw in reversed(lines):
        raw = raw.strip()
        if not raw:
            continue
        try:
            return json.loads(raw)
        except ValueError:
            return None  # torn last line: verify_chain will report it
    return None


class AuditLog:
    def __init__(self, directory: Path, origin: str, *, background: bool = True) -> None:
        self._dir = directory
        self._dir.mkdir(parents=True, exist_ok=True)
        self._origin = origin
        self._lock = threading.Lock()
        self._day = ""
        self._path: Path | None = None
        self._seq = 0
        self._prev = GENESIS
        self._resume()
        self._q: queue.Queue[tuple[str, dict[str, Any], int, float] | None] | None = None
        self._thread: threading.Thread | None = None
        if background:
            self._q = queue.Queue(maxsize=200_000)
            self._thread = threading.Thread(target=self._writer, name=f"audit-{origin}", daemon=True)
            self._thread.start()

    def _file_for(self, day: str) -> Path:
        return self._dir / f"{self._origin}-{day}.jsonl"

    def _resume(self) -> None:
        files = sorted(self._dir.glob(f"{self._origin}-*.jsonl"))
        if files:
            last = _last_record(files[-1])
            if last and "hash" in last:
                self._seq = int(last.get("seq", 0))
                self._prev = str(last["hash"])

    def append(self, kind: str, data: dict[str, Any] | None = None) -> None:
        item = (kind, redact(data or {}), time.perf_counter_ns(), time.time())
        if self._q is not None:
            try:
                self._q.put_nowait(item)
            except queue.Full:  # never block the caller (e.g. the fight loop)
                pass
        else:
            self._write(*item)

    def _write(self, kind: str, data: dict[str, Any], t_ns: int, t_wall: float) -> None:
        with self._lock:
            day = datetime.fromtimestamp(t_wall, tz=UTC).strftime("%Y%m%d")
            if day != self._day:
                self._day = day
                self._path = self._file_for(day)
            self._seq += 1
            rec: dict[str, Any] = {
                "v": 1,
                "seq": self._seq,
                "t": datetime.fromtimestamp(t_wall, tz=UTC).isoformat(timespec="milliseconds"),
                "t_ns": t_ns,
                "origin": self._origin,
                "kind": kind,
                "data": data,
                "prev": self._prev,
            }
            rec["hash"] = record_hash(rec)
            assert self._path is not None
            with self._path.open("a", encoding="utf-8") as f:
                f.write(canonical(rec) + "\n")
            self._prev = rec["hash"]

    def _writer(self) -> None:
        assert self._q is not None
        while True:
            item = self._q.get()
            if item is None:
                self._q.task_done()
                return
            try:
                self._write(*item)
            finally:
                self._q.task_done()

    def flush(self, timeout: float = 5.0) -> None:
        if self._q is None:
            return
        deadline = time.monotonic() + timeout
        while self._q.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(0.005)

    def close(self) -> None:
        if self._q is not None and self._thread is not None:
            self.flush()
            self._q.put(None)
            self._thread.join(timeout=2.0)
            self._q = None


@dataclass
class VerifyResult:
    ok: bool
    records: int
    first_bad_seq: int | None = None
    reason: str = ""


def verify_chain(files: list[Path], start_prev: str = GENESIS) -> VerifyResult:
    """Verify one origin's files in chronological order."""
    prev = start_prev
    n = 0
    expected_seq: int | None = None
    for path in files:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    return VerifyResult(False, n, (expected_seq or 0), f"unparseable line in {path.name}")
                seq = int(rec.get("seq", -1))
                if expected_seq is not None and seq != expected_seq:
                    return VerifyResult(False, n, seq, f"sequence gap: expected {expected_seq}, got {seq}")
                if rec.get("prev") != prev:
                    return VerifyResult(False, n, seq, "prev hash does not match previous record")
                if record_hash(rec) != rec.get("hash"):
                    return VerifyResult(False, n, seq, "record hash mismatch (edited record)")
                prev = rec["hash"]
                expected_seq = seq + 1
                n += 1
    return VerifyResult(True, n)
