from __future__ import annotations

import pytest

from jarvis.safety.capabilities import (
    BASELINE,
    Cap,
    ScopeRegistry,
    classify_heuristic,
    parse_classifier_output,
)


def test_heuristic_baseline_and_keywords():
    caps = classify_heuristic("what's the weather tomorrow?")
    assert caps == set(BASELINE)
    assert Cap.FILES_DELETE in classify_heuristic("delete the old screenshots in Downloads")
    assert Cap.FILES_DELETE in classify_heuristic("다운로드 폴더 정리하고 오래된 파일 지워줘")
    assert Cap.SHELL in classify_heuristic("install the latest python")
    assert Cap.GUI in classify_heuristic("디스코드에서 친구한테 메시지 보내줘")
    assert Cap.GAME in classify_heuristic("자비스, 싸워")
    assert Cap.FILES_DELETE not in classify_heuristic("open spotify and play my playlist")


def test_parse_classifier_output():
    raw = 'Sure: {"capabilities": ["files.read", "shell", "bogus"]}'
    assert parse_classifier_output(raw) == {Cap.FILES_READ, Cap.SHELL}
    assert parse_classifier_output("no json") == set()


def test_scope_registry_trust():
    reg = ScopeRegistry()
    with pytest.raises(PermissionError):
        reg.open("t1", {Cap.SHELL}, origin="web_page")
    sc = reg.open("t1", {Cap.FILES_READ}, origin="owner_voice", request="read my notes")
    assert sc.allows(Cap.FILES_READ) and not sc.allows(Cap.SHELL)
    reg.open("t1", {Cap.SHELL}, origin="owner_voice", request="and run the script")
    assert reg.get("t1").allows(Cap.SHELL)
    sys_scope = reg.open("t2", {Cap.GAME, Cap.WEB}, origin="system")
    assert not sys_scope.allows(Cap.GAME)  # only the owner may start fights
