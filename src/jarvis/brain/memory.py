"""Memory: the rolling conversation and a long-term store.

Long-term memory is SQLite with an FTS5 *trigram* index - substring matching works for Korean, where
particles attach to words (e.g. "로블록스를" still matches "로블록스"). Every fact keeps its source
and trust level; recalled facts are presented to the model as quoted reference material.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from jarvis.brain.types import Message


@dataclass
class Fact:
    id: int
    text: str
    source: str
    trust: str  # "owner" | "system" | "untrusted"
    created: float


class MemoryStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS facts (id INTEGER PRIMARY KEY, text TEXT NOT NULL, source TEXT, trust TEXT, created REAL)"
            )
            try:
                self._db.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(text, content='facts', content_rowid='id', tokenize='trigram')"
                )
                self.fts = True
            except sqlite3.OperationalError:  # very old SQLite without trigram
                self.fts = False
            self._db.commit()

    def remember(self, text: str, source: str = "owner", trust: str = "owner") -> int:
        text = text.strip()
        if not text:
            raise ValueError("empty fact")
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO facts(text, source, trust, created) VALUES (?, ?, ?, ?)", (text, source, trust, time.time())
            )
            fid = int(cur.lastrowid)
            if self.fts:
                self._db.execute("INSERT INTO facts_fts(rowid, text) VALUES (?, ?)", (fid, text))
            self._db.commit()
        return fid

    def forget(self, fid: int) -> bool:
        with self._lock:
            row = self._db.execute("SELECT text FROM facts WHERE id=?", (fid,)).fetchone()
            if row is None:
                return False
            if self.fts:
                self._db.execute("INSERT INTO facts_fts(facts_fts, rowid, text) VALUES('delete', ?, ?)", (fid, row[0]))
            self._db.execute("DELETE FROM facts WHERE id=?", (fid,))
            self._db.commit()
            return True

    def recall(self, query: str, k: int = 5) -> list[Fact]:
        q = query.strip()
        if not q:
            return []
        with self._lock:
            rows = []
            if self.fts and len(q) >= 3:
                terms = []
                for t in q.replace('"', " ").split():
                    terms.extend(v for v in _variants(t) if v not in terms)
                if terms:
                    match = " OR ".join(f'"{t}"' for t in terms[:24])
                    rows = self._db.execute(
                        "SELECT f.id, f.text, f.source, f.trust, f.created FROM facts_fts JOIN facts f ON f.id = facts_fts.rowid "
                        "WHERE facts_fts MATCH ? ORDER BY rank LIMIT ?",
                        (match, k),
                    ).fetchall()
            if not rows:
                rows = self._db.execute(
                    "SELECT id, text, source, trust, created FROM facts WHERE text LIKE ? ORDER BY created DESC LIMIT ?",
                    (f"%{q[:40]}%", k),
                ).fetchall()
        return [Fact(*r) for r in rows]

    def recent(self, k: int = 10) -> list[Fact]:
        with self._lock:
            rows = self._db.execute(
                "SELECT id, text, source, trust, created FROM facts ORDER BY created DESC LIMIT ?", (k,)
            ).fetchall()
        return [Fact(*r) for r in rows]

    def close(self) -> None:
        with self._lock:
            self._db.close()


def _has_hangul(t: str) -> bool:
    return any("\uac00" <= ch <= "\ud7a3" for ch in t)


def _variants(term: str) -> list[str]:
    """Search variants of a query word. Korean particles attach to words ("로블록스를"), and the
    trigram index matches whole substrings, so also try the stem with 1-2 trailing syllables removed."""
    out = [term] if len(term) >= 3 else []
    if _has_hangul(term):
        for cut in (1, 2):
            stem = term[:-cut]
            if len(stem) >= 3:
                out.append(stem)
    return out


def memory_block(facts: list[Fact]) -> str:
    if not facts:
        return ""
    lines = [f"- ({f.trust}, {f.source}) {f.text}" for f in facts]
    return "Things you remember (reference, may be outdated):\n" + "\n".join(lines)


class Conversation:
    """Rolling chat history with a running summary of older turns (summarized by the local model)."""

    def __init__(self, max_messages: int = 40) -> None:
        self.max_messages = max_messages
        self.messages: list[Message] = []
        self.summary = ""

    def add(self, msg: Message) -> None:
        self.messages.append(msg)

    def needs_summary(self) -> bool:
        return len(self.messages) > self.max_messages

    def split_for_summary(self) -> tuple[list[Message], list[Message]]:
        """Older half to summarize, newer half to keep (never splitting a tool call from its results)."""
        cut = len(self.messages) // 2
        while cut < len(self.messages) and self.messages[cut].get("role") == "tool":
            cut += 1
        return self.messages[:cut], self.messages[cut:]

    def apply_summary(self, summary: str, keep: list[Message]) -> None:
        self.summary = summary.strip()
        self.messages = list(keep)

    def context(self) -> list[Message]:
        out: list[Message] = []
        if self.summary:
            out.append({"role": "system", "content": f"Summary of the earlier conversation:\n{self.summary}"})
        out.extend(self.messages)
        return out
