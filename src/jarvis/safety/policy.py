"""The safety floor JARVIS cannot cross, even though it never asks for confirmation.

Loaded from the admin-owned, read-only ``%ProgramData%\\Jarvis\\safety.yaml`` written by the installer.
If that file is missing (dev machines, CI) the repo default is used and ``protected`` is False, which
``jarvis doctor`` reports loudly.
"""

from __future__ import annotations

import ntpath
import os
import posixpath
import re
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from jarvis.common.paths import config_dir, protected_policy_path
from jarvis.safety import ps_ast


class SafetyPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = 1
    protected_roots: list[str] = Field(default_factory=list)
    protected_trees: list[str] = Field(default_factory=list)
    denied_commands: list[str] = Field(default_factory=list)
    denied_patterns: list[str] = Field(default_factory=list)
    payment_keywords: list[str] = Field(default_factory=list)
    detect_card_numbers: bool = True
    roblox_window_titles: list[str] = Field(default_factory=lambda: ["Roblox"])
    roblox_process_names: list[str] = Field(default_factory=lambda: ["RobloxPlayerBeta.exe"])
    shell_timeout_s: float = 120.0
    max_file_write_mb: float = 50.0

    # runtime-only
    protected: bool = False
    source: str = ""


@dataclass
class Decision:
    allowed: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.allowed


ALLOW = Decision(True)


def deny(reason: str) -> Decision:
    return Decision(False, reason)


def _file_is_read_only_for_user(path: Path) -> bool:
    try:
        return not os.access(path, os.W_OK)
    except OSError:
        return False


def load_policy(path: Path | None = None) -> SafetyPolicy:
    candidates = [path] if path else [protected_policy_path(), config_dir() / "safety.default.yaml"]
    for p in candidates:
        if p and p.exists():
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            pol = SafetyPolicy.model_validate(data)
            pol.source = str(p)
            pol.protected = (p == protected_policy_path()) and _file_is_read_only_for_user(p)
            return pol
    raise FileNotFoundError("no safety policy found (expected configs/safety.default.yaml)")


# ------------------------------------------------------------------------------------- paths
_ENV = re.compile(r"%([A-Za-z0-9_()]+)%")


def expand_env(p: str, environ: dict[str, str] | None = None) -> str:
    env = environ if environ is not None else os.environ
    lower = {k.lower(): v for k, v in env.items()}

    def sub(m: re.Match[str]) -> str:
        return lower.get(m.group(1).lower(), m.group(0))

    return os.path.expanduser(_ENV.sub(sub, p))


def _is_windows_path(p: str) -> bool:
    return bool(re.match(r"^[A-Za-z]:[\\/]", p)) or p.startswith("\\\\")


def norm_path(p: str, environ: dict[str, str] | None = None) -> str:
    """Normalize for comparison: expand %VARS%, collapse ``..``, unify separators, case-fold."""
    p = expand_env(p.strip().strip('"'), environ)
    if _is_windows_path(p):
        p = ntpath.normpath(p).lower()
        if len(p) > 3:
            p = p.rstrip("\\")
        return p
    p = posixpath.normpath(p)
    return p.rstrip("/") or "/"


def _is_within(child: str, parent: str) -> bool:
    if child == parent:
        return True
    sep = "\\" if "\\" in parent or _is_windows_path(parent) else "/"
    prefix = parent if parent.endswith(sep) else parent + sep
    return child.startswith(prefix)


class PolicyChecker:
    def __init__(self, policy: SafetyPolicy, environ: dict[str, str] | None = None) -> None:
        self.policy = policy
        self._env = environ
        self._roots = {norm_path(r, environ) for r in policy.protected_roots}
        self._trees = [norm_path(t, environ) for t in policy.protected_trees]
        self._denied_cmds = {ps_ast.canonical_name(c) for c in policy.denied_commands}
        self._patterns = [re.compile(p, re.I) for p in policy.denied_patterns]
        self._pay = [k.lower() for k in policy.payment_keywords]

    # ------------------------------------------------------------------ filesystem
    def check_path_mutation(self, path: str, op: str = "delete") -> Decision:
        n = norm_path(path, self._env)
        if n in self._roots:
            return deny(f"refusing to {op} protected folder {path!r}")
        for t in self._trees:
            if _is_within(n, t):
                return deny(f"refusing to {op} inside protected folder {t!r}")
        return ALLOW

    # ------------------------------------------------------------------ shell
    def check_shell(self, script: str, *, use_pwsh: bool = True) -> Decision:
        flat = " ".join(script.split())
        for pat in self._patterns:
            if pat.search(flat):
                return deny(f"command matches a denied pattern ({pat.pattern[:40]}...)")
        analysis = ps_ast.analyze(script, use_pwsh=use_pwsh)
        for cmd in analysis.commands:
            name = cmd.canonical
            if name in self._denied_cmds:
                if name == "remove-item":
                    return deny("deletes must use the files.delete tool (Recycle Bin), not the shell")
                return deny(f"command {cmd.name!r} is not allowed")
            if name in ps_ast.LAUNCHERS:
                target = _launch_target(cmd)
                if target and ps_ast.canonical_name(target) in self._denied_cmds:
                    return deny(f"launching {target!r} is not allowed")
        return ALLOW

    # ------------------------------------------------------------------ UI / typing
    def check_ui_element(self, name: str) -> Decision:
        low = name.lower()
        for k in self._pay:
            if k in low:
                return deny(f"refusing to activate a payment/purchase control ({name!r})")
        return ALLOW

    def check_typed_text(self, text: str) -> Decision:
        if self.policy.detect_card_numbers and contains_card_number(text):
            return deny("refusing to type something that looks like a payment card number")
        return ALLOW

    def is_roblox(self, title: str | None, process_name: str | None) -> bool:
        t = (title or "").lower()
        p = (process_name or "").lower()
        if any(p == x.lower() for x in self.policy.roblox_process_names):
            return True
        # The player window is titled exactly "Roblox"; browser tabs about Roblox must not match.
        return any(t == x.lower() for x in self.policy.roblox_window_titles)


def _launch_target(cmd: ps_ast.PsCommand) -> str | None:
    """The program a launcher command starts: ``-FilePath X`` or the first positional argument."""
    args = cmd.args
    for i, a in enumerate(args):
        if a.lower() in ("-filepath", "-filename", "-path", "-fp") and i + 1 < len(args):
            return args[i + 1]
    for a in args:
        if not a.startswith("-") and not a.startswith("/") and "://" not in a:
            return a
    return None


_DIGIT_RUN = re.compile(r"(?:\d[ -]?){13,19}")


def luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = ord(ch) - 48
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def contains_card_number(text: str) -> bool:
    for m in _DIGIT_RUN.finditer(text):
        digits = re.sub(r"\D", "", m.group(0))
        if 13 <= len(digits) <= 19 and luhn_ok(digits):
            return True
    return False
