"""Background-learner jobs. Each runs in its own process (``python -m jarvis.learn.jobs <job> --out f.json``)
so that a fight starting can end it instantly (learnerd kills the whole process tree). The result is a
JSON object written to ``--out`` (and printed as the last stdout line)."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

log = logging.getLogger(__name__)

SESSION_KINDS = ("_fight", "_shadow", "_video")
JOBS = ("post_fight", "nightly", "video", "kb", "consolidate")
KB_QUERIES = ["untitled boxing game guide", "untitled boxing game tips combos", "untitled boxing game how to dodge",
              "언타이틀드 복싱 게임 공략"]


def list_sessions(recordings: Path) -> list[Path]:
    if not recordings.exists():
        return []
    return sorted(p for p in recordings.iterdir() if p.is_dir() and any(k in p.name for k in SESSION_KINDS)
                  and (p / "frames.jsonl").exists())


# ------------------------------------------------------------------------------------- post-fight
def job_post_fight(cfg) -> dict:
    """Label statistics for the newest sessions + game-update drift check over our recent fights."""
    from jarvis.learn.drift import detect_drift, session_hud_conf, write_drift
    from jarvis.learn.hindsight import label_session

    paths = cfg.paths().ensure()
    sessions = list_sessions(paths.recordings)
    stats = {s.name: label_session(s).stats() for s in sessions[-5:]}
    own = [s for s in sessions if "_video" not in s.name]
    confs = [c for c in (session_hud_conf(s) for s in own[-10:]) if c is not None]
    drifting, reason = detect_drift(confs)
    write_drift(paths.run, drifting, reason)
    return {"processed": [s.name for s in sessions], "drift": drifting, "drift_reason": reason, "labels": stats}


# ------------------------------------------------------------------------------------- nightly retrain
def _train_candidate(cfg, samples, data_dir: Path, test_dir: Path | None, epochs: int, **train_kw):
    from jarvis.learn import train_threat as tt

    if not (data_dir / "info.json").exists():
        tt.prepare_crops(samples, cfg.paths().recordings, data_dir, cfg.fight.perception.crop_size)
    net, _ = tt.train(data_dir, epochs=epochs, window=cfg.fight.perception.threat_window, **train_kw)
    return net, tt.evaluate(net, test_dir or data_dir)


def _measured_latency(calibration: Path) -> float | None:
    try:
        return float(json.loads((calibration / "latency.json").read_text(encoding="utf-8"))["median_ms"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def job_nightly(cfg, epochs: int = 15, **train_kw) -> dict:
    """Dataset -> train (own fights only, and own + video labels) -> test on held-out own fights -> gate ->
    promote. Video-derived labels are kept only if they do not hurt the score on our own fights."""
    from jarvis.common.config import update_user_config
    from jarvis.learn import train_threat as tt
    from jarvis.learn.datasets import build_samples, write_manifest
    from jarvis.learn.evaluate import Gate, promotion_decision
    from jarvis.learn.registry import ModelRegistry

    paths = cfg.paths().ensure()
    sessions = list_sessions(paths.recordings)
    window = cfg.fight.perception.threat_window
    samples = [s for d in sessions for s in build_samples(d, window=window)]
    own = [s for s in samples if s.source != "video"]
    if not own or not any(s.split == "train" for s in own):
        return {"trained": False, "why": "not enough recorded fights with a visible opponent yet", "sessions": len(sessions)}
    version = time.strftime("%Y%m%d-%H%M%S")
    write_manifest(samples, paths.datasets, version)
    own_dir = paths.datasets / f"threat-{version}-own"
    all_dir = paths.datasets / f"threat-{version}-all"
    tmp_onnx = paths.datasets / f"threat-{version}.onnx"
    try:
        net, metrics = _train_candidate(cfg, own, own_dir, None, epochs, **train_kw)
        used_video = False
        if len(own) < len(samples):  # video labels must earn their place on our own fights' test set
            net_v, m_v = _train_candidate(cfg, samples, all_dir, own_dir, epochs, **train_kw)
            if m_v.get("ap_300") is not None and m_v["ap_300"] >= metrics.get("ap_300", 0.0) - 0.005:
                net, metrics, used_video = net_v, m_v, True
        tt.export_onnx(net, tmp_onnx, cfg.fight.perception.crop_size)
        reg = ModelRegistry(paths.models)
        active = reg.active("threat")
        entry = reg.register("threat", tmp_onnx, dataset_version=version, metrics={**metrics, "video_labels": used_video})
        ok, why = promotion_decision(metrics, active.metrics if active else None, Gate(latency_ms=_measured_latency(paths.calibration)))
        if ok:
            reg.promote("threat", entry.version)
            update_user_config(paths, {"fight": {"perception": {"threat_model": "threat/active.onnx"}}})
    finally:  # crops are large and rebuilt from the recordings each night
        for d in (own_dir, all_dir):
            shutil.rmtree(d, ignore_errors=True)
        tmp_onnx.unlink(missing_ok=True)
    return {"trained": True, "version": entry.version, "promoted": ok, "why": why, "video_labels": used_video,
            "ap_300": metrics.get("ap_300"), "sessions": len(sessions)}


# ------------------------------------------------------------------------------------- video ingestion
def job_video(cfg, budget: int) -> dict:
    from jarvis.fight.loop import load_detector
    from jarvis.fight.perception.detector import NullDetector
    from jarvis.fight.perception.layout import load_layout
    from jarvis.learn.video_ingest import find_local_videos, folder_gb, ingest_video, ytdlp_download

    paths = cfg.paths().ensure()
    vi = cfg.learn.video_ingest
    layout = load_layout(paths.calibration)
    if layout is None:
        return {"tried": 0, "why": "calibrate the HUD first (videos are screened with your calibrated reader)"}
    detector = load_detector(cfg)
    if isinstance(detector, NullDetector):
        return {"tried": 0, "why": "no fighter detector yet (`jarvis train detector`): videos would give no training windows"}
    if folder_gb(paths.recordings) > cfg.fight.recording.max_disk_gb:
        return {"tried": 0, "why": f"recordings use more than {cfg.fight.recording.max_disk_gb} GB"}
    seen_file = paths.datasets / "videos_seen.json"
    try:
        seen = set(json.loads(seen_file.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        seen = set()
    candidates = [p for p in find_local_videos(vi.local_folders) if str(p) not in seen][:budget]
    raw = paths.datasets / "video_raw"
    for q in vi.queries:
        if len(candidates) >= budget or folder_gb(raw) > vi.max_gb:
            break
        candidates += [p for p in ytdlp_download(q, raw, budget - len(candidates)) if str(p) not in seen]
    accepted = rejected = 0
    why = ""
    for v in candidates[:budget]:
        sd = None
        try:
            sd = ingest_video(v, paths.recordings, layout, paths.calibration, detector=detector, source=str(v),
                              ffmpeg=cfg.fight.recording.ffmpeg, encoder=cfg.fight.recording.encoder)
        except RuntimeError as e:  # e.g. ffmpeg missing: not this video's fault, stop the batch
            why = str(e)
            break
        except Exception:  # an unreadable file is rejected (and never retried), not fatal
            log.exception("could not ingest %s", v)
        finally:
            if raw in v.parents:  # our own download: the session keeps the re-encoded copy
                v.unlink(missing_ok=True)
        seen.add(str(v))
        accepted += sd is not None
        rejected += sd is None
        seen_file.write_text(json.dumps(sorted(seen)), encoding="utf-8")
    res = {"ingested": accepted, "rejected": rejected, "tried": accepted + rejected}
    return {**res, "why": why} if why else res


# ------------------------------------------------------------------------------------- knowledge base
def job_kb(cfg) -> dict:
    from jarvis.common.paths import config_dir
    from jarvis.core import build_brain
    from jarvis.learn.kb import ensure_kb, merge_facts, refresh_from_web, save_kb

    paths = cfg.paths().ensure()
    kb_dir = paths.root / "kb"
    facts = ensure_kb(kb_dir, config_dir())
    broker, _agent, memory = build_brain(cfg)  # no tool backend: the learner can only read the web
    try:
        new = asyncio.run(refresh_from_web(broker, KB_QUERIES))
    finally:
        memory.close()
    facts, added = merge_facts(facts, new)
    save_kb(kb_dir, facts)
    return {"facts": len(facts), "added": added}


# ------------------------------------------------------------------------------------- consolidation
def job_consolidate(cfg) -> dict:
    from jarvis.brain.memory import MemoryStore
    from jarvis.learn.consolidate import consolidate
    from jarvis.learn.skills import dedupe_skills

    paths = cfg.paths().ensure()
    mem = MemoryStore(paths.db / "memory.sqlite")
    try:
        res = consolidate(mem)
    finally:
        mem.close()
    skills = paths.db / "skills.jsonl"
    res["skills"] = dedupe_skills(skills) if skills.exists() else 0
    return res


def run_job(cfg, name: str, budget: int = 5) -> dict:
    if name == "post_fight":
        return job_post_fight(cfg)
    if name == "nightly":
        return job_nightly(cfg)
    if name == "video":
        return job_video(cfg, budget)
    if name == "kb":
        return job_kb(cfg)
    if name == "consolidate":
        return job_consolidate(cfg)
    raise ValueError(f"unknown job {name}")


def write_result(path: Path | None, res: dict) -> None:
    text = json.dumps(res, default=str, ensure_ascii=False)
    if path is not None:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    print(text)


def main(argv: list[str] | None = None) -> int:
    from jarvis.common.config import load_config

    ap = argparse.ArgumentParser(prog="python -m jarvis.learn.jobs")
    ap.add_argument("job", choices=JOBS)
    ap.add_argument("--budget", type=int, default=5)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from jarvis.common.cuda_dlls import setup_cuda_dlls

    setup_cuda_dlls()
    try:
        res = {"job": a.job, "ok": True, **run_job(load_config(), a.job, a.budget)}
    except Exception as e:
        log.exception("job %s failed", a.job)
        write_result(a.out, {"job": a.job, "ok": False, "error": f"{type(e).__name__}: {e}"})
        return 1
    write_result(a.out, res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
