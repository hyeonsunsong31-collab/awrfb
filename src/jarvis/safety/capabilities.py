"""Task-scoped capabilities.

Each owner request opens (or extends) a task scope: the set of capability classes that request
plausibly needs. The scope is decided ONLY from the owner's own words (verified voice or the owner's
HUD chat), before any web page, OCR text or file content is read. Tool calls outside the scope are
refused by toolhost, so injected instructions inside untrusted content cannot widen what JARVIS may
do - without ever asking for confirmation.
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from enum import StrEnum


class Cap(StrEnum):
    FILES_READ = "files.read"
    FILES_WRITE = "files.write"
    FILES_DELETE = "files.delete"
    SHELL = "shell"
    APPS = "apps"
    WINDOWS = "windows"
    GUI = "gui"
    SCREEN = "screen"
    WEB = "web"
    SYSTEM = "system"
    CLIPBOARD = "clipboard"
    MEMORY = "memory"
    GAME = "game"


# Granted to every owner request: read-only or easily reversible actions.
BASELINE: frozenset[Cap] = frozenset(
    {Cap.FILES_READ, Cap.APPS, Cap.WINDOWS, Cap.SCREEN, Cap.WEB, Cap.SYSTEM, Cap.CLIPBOARD, Cap.MEMORY}
)

_KEYWORDS: dict[Cap, list[str]] = {
    Cap.FILES_DELETE: [
        r"\bdelete\b", r"\bremove\b", r"\berase\b", r"\bclean ?up\b", r"\bclear\b", r"\btrash\b",
        "지워", "지우", "삭제", "정리", "비워",
    ],
    Cap.FILES_WRITE: [
        r"\bwrite\b", r"\bsave\b", r"\bcreate\b", r"\bedit\b", r"\brename\b", r"\bmove\b", r"\bcopy\b",
        r"\borganize\b", r"\bmake (a|an|the)? ?(file|folder|note|document)", r"\bdownload\b",
        "저장", "만들", "작성", "수정", "이름 바꿔", "옮겨", "복사", "정리", "다운로드", "메모",
    ],
    Cap.SHELL: [
        r"\brun\b", r"\bcommand\b", r"\bterminal\b", r"\bpowershell\b", r"\bscript\b", r"\binstall\b",
        r"\buninstall\b", r"\bupdate\b", r"\bping\b", r"\bprocess(es)?\b", r"\bkill\b",
        "실행", "명령", "설치", "삭제해줘 프로그램", "업데이트", "터미널", "스크립트", "프로세스",
    ],
    Cap.GUI: [
        r"\bclick\b", r"\btype\b", r"\bpress\b", r"\bsettings?\b", r"\bbutton\b", r"\bfill\b",
        r"\bsend\b", r"\bmessage\b", r"\breply\b", r"\bpost\b", r"\bscroll\b", r"\bselect\b",
        "클릭", "눌러", "입력", "설정", "버튼", "보내", "메시지", "답장", "스크롤", "선택",
    ],
    Cap.GAME: [r"\bfight\b", r"\bbox(ing)?\b", r"\broblox\b", r"\bubg\b", "싸워", "싸움", "복싱", "로블록스", "대전"],
}


def classify_heuristic(text: str) -> set[Cap]:
    low = text.lower()
    caps: set[Cap] = set(BASELINE)
    for cap, pats in _KEYWORDS.items():
        if any(re.search(p, low) for p in pats):
            caps.add(cap)
    # Writing usually implies GUI or shell to do it; deleting implies writing (moves) as well.
    if Cap.FILES_DELETE in caps:
        caps.add(Cap.FILES_WRITE)
    return caps


CLASSIFIER_PROMPT = (
    "You decide which capability classes a computer assistant needs to fulfil the OWNER's request. "
    "Answer with JSON only: {\"capabilities\": [...]} using any of: "
    + ", ".join(c.value for c in Cap)
    + ". Be generous for what the owner asked and include nothing unrelated. The request may be Korean or English."
)


def parse_classifier_output(raw: str) -> set[Cap]:
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return set()
    try:
        data = json.loads(m.group(0))
    except ValueError:
        return set()
    out: set[Cap] = set()
    for v in data.get("capabilities", []):
        try:
            out.add(Cap(str(v).strip().lower()))
        except ValueError:
            continue
    return out


@dataclass
class TaskScope:
    task_id: str
    caps: set[Cap]
    origin: str  # "owner_voice" | "owner_text" | "system"
    created_ns: int = field(default_factory=time.perf_counter_ns)
    requests: list[str] = field(default_factory=list)

    def allows(self, cap: Cap) -> bool:
        return cap in self.caps


class ScopeRegistry:
    """Held by toolhost. Only core (with the IPC token) can open or extend scopes."""

    TRUSTED_ORIGINS = frozenset({"owner_voice", "owner_text", "system"})

    def __init__(self, ttl_s: float = 3600.0) -> None:
        self._scopes: dict[str, TaskScope] = {}
        self._lock = threading.Lock()
        self._ttl_ns = int(ttl_s * 1e9)

    def open(self, task_id: str, caps: set[Cap], origin: str, request: str = "") -> TaskScope:
        if origin not in self.TRUSTED_ORIGINS:
            raise PermissionError(f"untrusted origin {origin!r} cannot open a task scope")
        if Cap.GAME in caps and origin not in ("owner_voice", "owner_text"):
            caps = set(caps) - {Cap.GAME}
        with self._lock:
            self._gc()
            sc = self._scopes.get(task_id)
            if sc is None:
                sc = TaskScope(task_id, set(caps), origin)
                self._scopes[task_id] = sc
            else:
                sc.caps |= set(caps)
            if request:
                sc.requests.append(request)
            return sc

    def get(self, task_id: str) -> TaskScope | None:
        with self._lock:
            return self._scopes.get(task_id)

    def close(self, task_id: str) -> None:
        with self._lock:
            self._scopes.pop(task_id, None)

    def _gc(self) -> None:
        now = time.perf_counter_ns()
        for k in [k for k, s in self._scopes.items() if now - s.created_ns > self._ttl_ns]:
            del self._scopes[k]
