"""File tools. Reading is free (files.read). Writing refuses protected folders and moves the previous
version of an overwritten file to the Recycle Bin first (files.write). Deleting always goes to the Recycle
Bin (files.delete). File names and contents are untrusted text for the model."""

from __future__ import annotations

import asyncio
import os
import shutil
import time
from pathlib import Path

from jarvis.safety.capabilities import Cap
from jarvis.tools.base import Refused, Tool, ToolContext, ToolResult, b, clip, i, ok, s

SKIP_DIRS = {"$recycle.bin", "system volume information", "node_modules", ".git", "__pycache__", "windows"}


def trash(path: str) -> None:
    from send2trash import send2trash

    send2trash(path)


def _path(raw: str | None) -> Path:
    if not raw or not str(raw).strip():
        raise Refused("a path is required")
    return Path(os.path.expandvars(os.path.expanduser(str(raw).strip().strip('"')))).absolute()


def _check(ctx: ToolContext, p: Path, op: str, raw: str | None = None) -> None:
    """Check every form of the path: as given, absolute, and with links resolved (a junction must not
    smuggle a protected folder past the policy)."""
    forms = [str(p)]
    if raw:
        forms.append(str(raw).strip().strip('"'))
    try:
        forms.append(str(p.resolve(strict=False)))
    except (OSError, RuntimeError):
        pass
    for form in forms:
        d = ctx.policy.check_path_mutation(form, op)
        if not d:
            raise Refused(d.reason)


def _size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return str(n)


async def fs_list(ctx: ToolContext, a: dict) -> ToolResult:
    p = _path(a.get("path") or "~")
    if not p.exists():
        return ToolResult(False, f"{p} does not exist")
    if p.is_file():
        st = p.stat()
        return ok(f"{p} (file, {_size(st.st_size)}, modified {time.strftime('%Y-%m-%d %H:%M', time.localtime(st.st_mtime))})", trust="untrusted", source=str(p))
    limit = max(1, min(int(a.get("limit", 200)), 1000))
    rows = []
    for e in sorted(p.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))[:limit]:
        try:
            st = e.stat()
        except OSError:
            continue
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime))
        rows.append(f"[dir]  {e.name}" if e.is_dir() else f"       {e.name}  ({_size(st.st_size)}, {when})")
    return ok(f"{p}:\n" + ("\n".join(rows) or "(empty)"), trust="untrusted", source=str(p))


def _decode(data: bytes) -> str:
    for enc in ("utf-8", "cp949", "utf-16"):  # cp949: Korean Windows text files
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


async def fs_read(ctx: ToolContext, a: dict) -> ToolResult:
    p = _path(a.get("path"))
    if not p.is_file():
        return ToolResult(False, f"{p} is not a file")
    if p.stat().st_size > 20 * 1024 * 1024:
        return ToolResult(False, f"{p} is larger than 20 MB; read a smaller file or search inside it")
    data = await asyncio.to_thread(p.read_bytes)
    if b"\x00" in data[:4096] and not data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return ok(f"{p} is a binary file ({_size(len(data))})", source=str(p))
    n = max(200, min(int(a.get("max_chars", 20000)), 100_000))
    return ok(clip(_decode(data), n), trust="untrusted", source=str(p))


async def fs_search(ctx: ToolContext, a: dict) -> ToolResult:
    import fnmatch

    root = _path(a.get("root") or "~")
    pat = str(a.get("pattern", "")).strip()
    if not pat:
        raise Refused("a pattern is required (e.g. *.pdf or a part of the name)")
    glob = pat if any(c in pat for c in "*?[") else f"*{pat}*"
    limit = max(1, min(int(a.get("limit", 50)), 500))

    def walk() -> list[str]:
        found: list[str] = []
        deadline = time.monotonic() + 8.0
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d.lower() not in SKIP_DIRS and not d.startswith(".")]
            for name in dirnames + filenames:
                if fnmatch.fnmatch(name.lower(), glob.lower()):
                    found.append(os.path.join(dirpath, name))
                    if len(found) >= limit:
                        return found
            if time.monotonic() > deadline:
                found.append("... (search stopped after 8 s; narrow the folder)")
                return found
        return found

    hits = await asyncio.to_thread(walk)
    return ok("\n".join(hits) if hits else f"nothing matching {pat!r} under {root}", trust="untrusted", source=str(root))


async def fs_write(ctx: ToolContext, a: dict) -> ToolResult:
    p = _path(a.get("path"))
    _check(ctx, p, "write", a.get("path"))
    content = str(a.get("content", ""))
    cap_mb = getattr(ctx.policy.policy, "max_file_write_mb", 50.0)
    if len(content.encode("utf-8")) > cap_mb * 1024 * 1024:
        raise Refused(f"refusing to write more than {cap_mb:.0f} MB")
    append = bool(a.get("append", False))
    note = ""
    if p.exists() and p.is_dir():
        return ToolResult(False, f"{p} is a folder")
    if p.exists() and not append:
        await asyncio.to_thread(trash, str(p))  # overwrites stay reversible
        note = " (the previous version is in the Recycle Bin)"
    p.parent.mkdir(parents=True, exist_ok=True)

    def write() -> None:
        with open(p, "a" if append else "w", encoding="utf-8", newline="") as f:
            f.write(content)

    await asyncio.to_thread(write)
    return ok(f"{'appended' if append else 'wrote'} {len(content)} characters to {p}{note}")


async def fs_move(ctx: ToolContext, a: dict) -> ToolResult:
    src, dst = _path(a.get("src")), _path(a.get("dst"))
    _check(ctx, src, "move", a.get("src"))
    _check(ctx, dst, "write", a.get("dst"))
    if not src.exists():
        return ToolResult(False, f"{src} does not exist")
    if dst.exists() and not dst.is_dir():
        return ToolResult(False, f"{dst} already exists; delete or rename it first")
    final = await asyncio.to_thread(shutil.move, str(src), str(dst))
    return ok(f"moved {src} -> {final}")


async def fs_delete(ctx: ToolContext, a: dict) -> ToolResult:
    p = _path(a.get("path"))
    _check(ctx, p, "delete", a.get("path"))
    if not p.exists():
        return ToolResult(False, f"{p} does not exist")
    await asyncio.to_thread(trash, str(p))
    return ok(f"moved {p} to the Recycle Bin")


async def fs_mkdir(ctx: ToolContext, a: dict) -> ToolResult:
    p = _path(a.get("path"))
    _check(ctx, p, "write", a.get("path"))
    p.mkdir(parents=True, exist_ok=True)
    return ok(f"folder ready: {p}")


TOOLS = [
    Tool("fs_list", "List a folder (or describe a file). Paths may use ~ and %VARS%.", Cap.FILES_READ, fs_list,
         {"path": s("Folder path, default: home"), "limit": i("Max entries (default 200)")}, trust="untrusted"),
    Tool("fs_read", "Read a text file.", Cap.FILES_READ, fs_read,
         {"path": s("File path"), "max_chars": i("Max characters (default 20000)")}, ("path",), trust="untrusted"),
    Tool("fs_search", "Find files or folders by name (glob like *.pdf or part of the name) under a folder.", Cap.FILES_READ,
         fs_search, {"root": s("Folder to search, default: home"), "pattern": s("Name pattern"), "limit": i("Max results")},
         ("pattern",), trust="untrusted"),
    Tool("fs_write", "Write (or append) a text file. Overwriting keeps the old version in the Recycle Bin.", Cap.FILES_WRITE,
         fs_write, {"path": s("File path"), "content": s("Text to write"), "append": b("Append instead of overwrite")},
         ("path", "content")),
    Tool("fs_move", "Move or rename a file or folder.", Cap.FILES_WRITE, fs_move,
         {"src": s("Source path"), "dst": s("Destination path or folder")}, ("src", "dst")),
    Tool("fs_mkdir", "Create a folder (and parents).", Cap.FILES_WRITE, fs_mkdir, {"path": s("Folder path")}, ("path",)),
    Tool("fs_delete", "Delete a file or folder by moving it to the Recycle Bin (restorable).", Cap.FILES_DELETE, fs_delete,
         {"path": s("Path to delete")}, ("path",)),
]
