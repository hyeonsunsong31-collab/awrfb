"""PowerShell command analysis.

Preferred: parse with PowerShell's own parser (``System.Management.Automation.Language.Parser``)
through ``pwsh``/``powershell.exe`` and list every ``CommandAst`` with its arguments - no regex games.
Fallback (no PowerShell available, e.g. Linux CI): a conservative tokenizer that splits statements and
pipelines, unwraps nested ``powershell``/``cmd /c`` invocations and normalizes aliases.
"""

from __future__ import annotations

import json
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass, field

ALIASES = {
    "rm": "remove-item",
    "ri": "remove-item",
    "del": "remove-item",
    "erase": "remove-item",
    "rd": "remove-item",
    "rmdir": "remove-item",
    "iex": "invoke-expression",
    "iwr": "invoke-webrequest",
    "curl": "invoke-webrequest",
    "wget": "invoke-webrequest",
    "irm": "invoke-restmethod",
    "saps": "start-process",
    "start": "start-process",
    "ii": "invoke-item",
    "sc": "set-content",
    "ac": "add-content",
}

LAUNCHERS = {
    "start-process",
    "invoke-item",
    "cmd",
    "powershell",
    "pwsh",
    "wsl",
    "bash",
    "runas",
    "schtasks",
    "wmic",
    "invoke-command",
    "start-job",
}

_ANALYZER = r"""
$src = [Console]::In.ReadToEnd()
$tokens = $null; $errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseInput($src, [ref]$tokens, [ref]$errors)
$cmds = @($ast.FindAll({ param($n) $n -is [System.Management.Automation.Language.CommandAst] }, $true) | ForEach-Object {
  $name = $_.GetCommandName()
  $a = @($_.CommandElements | Select-Object -Skip 1 | ForEach-Object { $_.Extent.Text })
  [pscustomobject]@{ name = $name; args = $a; dynamic = ($null -eq $name) }
})
[pscustomobject]@{ commands = $cmds; errors = @($errors | ForEach-Object { $_.Message }) } | ConvertTo-Json -Depth 6 -Compress
"""


@dataclass
class PsCommand:
    name: str
    args: list[str] = field(default_factory=list)
    dynamic: bool = False

    @property
    def canonical(self) -> str:
        return canonical_name(self.name)


@dataclass
class PsAnalysis:
    commands: list[PsCommand]
    errors: list[str]
    method: str  # "pwsh" | "fallback"


def canonical_name(name: str) -> str:
    n = name.strip().strip("'\"").replace("/", "\\")
    n = n.rsplit("\\", 1)[-1].lower()
    for ext in (".exe", ".com", ".bat", ".cmd", ".ps1"):
        if n.endswith(ext):
            n = n[: -len(ext)]
    return ALIASES.get(n, n)


def find_powershell() -> str | None:
    return shutil.which("pwsh") or shutil.which("powershell")


def _analyze_with_pwsh(script: str, exe: str, timeout: float = 15.0) -> PsAnalysis | None:
    try:
        res = subprocess.run(
            [exe, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", _ANALYZER],
            input=script,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if res.returncode != 0 or not res.stdout.strip():
        return None
    try:
        data = json.loads(res.stdout.strip().splitlines()[-1])
    except ValueError:
        return None
    raw_cmds = data.get("commands") or []
    if isinstance(raw_cmds, dict):  # ConvertTo-Json unwraps single-element arrays
        raw_cmds = [raw_cmds]
    cmds = []
    for c in raw_cmds:
        args = c.get("args") or []
        if isinstance(args, str):
            args = [args]
        cmds.append(PsCommand(name=c.get("name") or "", args=[str(a) for a in args], dynamic=bool(c.get("dynamic"))))
    errors = data.get("errors") or []
    if isinstance(errors, str):
        errors = [errors]
    # Nested shells inside the script: analyse their command strings too.
    nested: list[PsCommand] = []
    for c in cmds:
        nested.extend(_nested_commands(c))
    return PsAnalysis(cmds + nested, [str(e) for e in errors], "pwsh")


_SPLIT = re.compile(r"\|\||&&|[;\n|]|(?<![<>])&(?![>&])")


def _tokens(segment: str) -> list[str]:
    try:
        lex = shlex.shlex(segment, posix=True, punctuation_chars=False)
        lex.whitespace_split = True
        lex.commenters = ""
        return list(lex)
    except ValueError:
        return segment.split()


def _nested_commands(cmd: PsCommand) -> list[PsCommand]:
    name = cmd.canonical
    if name in ("powershell", "pwsh"):
        args = list(cmd.args)
        for i, a in enumerate(args):
            if a.lower() in ("-command", "-c") and i + 1 < len(args):
                return analyze_fallback(" ".join(args[i + 1 :]).strip("'\"")).commands
        return []
    if name == "cmd":
        args = list(cmd.args)
        for i, a in enumerate(args):
            if a.lower() in ("/c", "/k") and i + 1 < len(args):
                return analyze_fallback(" ".join(args[i + 1 :]).strip("'\"")).commands
    return []


def analyze_fallback(script: str) -> PsAnalysis:
    cmds: list[PsCommand] = []
    for seg in _SPLIT.split(script):
        seg = seg.strip()
        if not seg or seg.startswith("#"):
            continue
        toks = _tokens(seg)
        if not toks:
            continue
        # Skip call operators and assignment prefixes:  & "x.exe" ...   $a = Get-Thing ...
        while toks and toks[0] in ("&", "."):
            toks = toks[1:]
        if len(toks) >= 2 and toks[0].startswith("$") and toks[1] == "=":
            toks = toks[2:]
        if not toks:
            continue
        head = toks[0]
        dynamic = head.startswith("$") or head.startswith("(")
        cmd = PsCommand(name=head, args=toks[1:], dynamic=dynamic)
        cmds.append(cmd)
        cmds.extend(_nested_commands(cmd))
    return PsAnalysis(cmds, [], "fallback")


def analyze(script: str, *, use_pwsh: bool = True) -> PsAnalysis:
    if use_pwsh:
        exe = find_powershell()
        if exe:
            got = _analyze_with_pwsh(script, exe)
            if got is not None:
                return got
    return analyze_fallback(script)
