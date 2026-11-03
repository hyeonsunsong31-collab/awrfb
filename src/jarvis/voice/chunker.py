"""Streaming sentence chunker: LLM token stream -> speakable pieces, as early as possible.

A boundary is sentence punctuation FOLLOWED by whitespace (so "3.14" or "e.g." mid-token are not
split), or a newline. Very short pieces are merged into the next one; very long ones are split at a
comma/space. The first piece of a reply may also end at a comma once it is long enough, which cuts the
time to first audio on long opening sentences.
"""

from __future__ import annotations

import re

_END = re.compile(r"([.!?。！？…]+[\"')\]]*)(\s+)|\n+")
_COMMA = re.compile(r"[,，、;:]\s+")


class SentenceChunker:
    def __init__(self, min_chars: int = 8, max_chars: int = 180, first_comma_min: int = 28) -> None:
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.first_comma_min = first_comma_min
        self.buf = ""
        self.carry = ""
        self.emitted = 0

    def _emit(self, out: list[str], piece: str) -> None:
        piece = piece.strip()
        if piece:
            out.append(piece)
            self.emitted += 1

    def feed(self, text: str) -> list[str]:
        self.buf += text
        out: list[str] = []
        while True:
            m = _END.search(self.buf)
            if not m:
                break
            end = m.end(1) if m.group(1) else m.start()
            piece = (self.carry + self.buf[:end]).strip()
            self.buf = self.buf[m.end() :]
            if len(piece) < self.min_chars:
                self.carry = piece + " " if piece else ""
                continue
            self.carry = ""
            self._emit(out, piece)
        if self.emitted == 0 and not out:
            pending = self.carry + self.buf
            for c in _COMMA.finditer(pending):
                if c.start() >= self.first_comma_min:
                    self._emit(out, pending[: c.start() + 1])
                    self.carry, self.buf = "", pending[c.end() :]
                    break
        while len(self.buf) > self.max_chars:
            cut = max(self.buf.rfind(",", 0, self.max_chars), self.buf.rfind(" ", 0, self.max_chars))
            if cut <= 0:
                cut = self.max_chars
            self._emit(out, self.carry + self.buf[: cut + 1])
            self.carry = ""
            self.buf = self.buf[cut + 1 :]
        return out

    def flush(self) -> list[str]:
        out: list[str] = []
        self._emit(out, self.carry + self.buf)
        self.buf = ""
        self.carry = ""
        return out
