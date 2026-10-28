from __future__ import annotations

import json

from jarvis.common.audit import AuditLog, redact, verify_chain


def files(d, origin="toolhost"):
    return sorted(d.glob(f"{origin}-*.jsonl"))


def test_chain_verifies_and_detects_tampering(tmp_path):
    log = AuditLog(tmp_path, "toolhost", background=False)
    for i in range(5):
        log.append("tool", {"tool": "shell", "i": i})
    res = verify_chain(files(tmp_path))
    assert res.ok and res.records == 5

    f = files(tmp_path)[0]
    lines = f.read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[2])
    rec["data"]["tool"] = "something-else"
    lines[2] = json.dumps(rec)
    f.write_text("\n".join(lines) + "\n", encoding="utf-8")
    bad = verify_chain(files(tmp_path))
    assert not bad.ok and bad.first_bad_seq == 3


def test_deleting_a_line_breaks_chain(tmp_path):
    log = AuditLog(tmp_path, "fightd", background=False)
    for i in range(4):
        log.append("game_input", {"i": i})
    f = files(tmp_path, "fightd")[0]
    lines = f.read_text(encoding="utf-8").splitlines()
    del lines[1]
    f.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert not verify_chain(files(tmp_path, "fightd")).ok


def test_chain_resumes_across_instances(tmp_path):
    AuditLog(tmp_path, "core", background=False).append("a", {})
    AuditLog(tmp_path, "core", background=False).append("b", {})
    res = verify_chain(files(tmp_path, "core"))
    assert res.ok and res.records == 2


def test_background_writer_and_redaction(tmp_path):
    log = AuditLog(tmp_path, "core")
    log.append("probe", {"api_key": "nvapi-SECRETSECRET", "note": "key nvapi-ABCDEFGHIJKL used"})
    log.close()
    text = files(tmp_path, "core")[0].read_text(encoding="utf-8")
    assert "SECRET" not in text and "ABCDEFGHIJKL" not in text
    assert redact({"Authorization": "Bearer x"}) == {"Authorization": "***"}
