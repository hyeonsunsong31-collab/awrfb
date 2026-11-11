"""Learning commands: labels, datasets, model registry, training, range calibration."""

from __future__ import annotations

import json
import time

import typer
from rich.console import Console
from rich.table import Table

console = Console()
dataset_app = typer.Typer(help="Training datasets from your recordings.", no_args_is_help=True)
models_app = typer.Typer(help="Model registry (candidate -> active, rollback).", no_args_is_help=True)
train_app = typer.Typer(help="Train fight models from your recordings.", no_args_is_help=True)
learn_app = typer.Typer(help="Background self-education (learnerd): status, run a job now, cancel.", no_args_is_help=True)


def _cfg():
    from jarvis.common.config import load_config

    return load_config()


def _sessions(cfg, kinds=("fight", "shadow", "video", "calibrate_keys")):
    rec = cfg.paths().ensure().recordings
    return sorted(p for p in rec.iterdir() if p.is_dir() and any(f"_{k}" in p.name for k in kinds) and (p / "frames.jsonl").exists())


def labels(session: str = typer.Argument("", help="Session dir name (default: all).")) -> None:
    """Show hindsight label statistics (hits, heavy threshold, positive rates)."""
    from jarvis.learn.hindsight import label_session

    cfg = _cfg()
    dirs = [cfg.paths().recordings / session] if session else _sessions(cfg)
    t = Table("session", "frames", "hits", "heavy", "pos@300ms")
    for d in dirs:
        st = label_session(d).stats()
        t.add_row(d.name, str(st["frames"]), str(st["hits"]), str(st["heavy_hits"]), f"{st['pos_rate_300ms']:.3f}")
    console.print(t)


@dataset_app.command("build")
def dataset_build(window: int = 8, stride: int = 2) -> None:
    """Build the threat-model dataset manifest (split by session)."""
    from jarvis.learn.datasets import build_samples, write_manifest

    cfg = _cfg()
    samples = []
    for d in _sessions(cfg, ("fight", "shadow", "video")):
        samples += build_samples(d, window=window, stride=stride)
    version = time.strftime("%Y%m%d-%H%M%S")
    p = write_manifest(samples, cfg.paths().ensure().datasets, version)
    by = {}
    for s in samples:
        by[s.split] = by.get(s.split, 0) + 1
    console.print(f"{len(samples)} samples {by} -> {p}")


@models_app.command("list")
def models_list() -> None:
    from jarvis.learn.registry import ModelRegistry

    reg = ModelRegistry(_cfg().paths().ensure().models)
    t = Table("name", "version", "status", "metrics", "dataset")
    for e in reg.entries:
        t.add_row(e.name, str(e.version), e.status, json.dumps(e.metrics)[:60], e.dataset_version)
    console.print(t)


@models_app.command("promote")
def models_promote(name: str, version: int) -> None:
    from jarvis.learn.registry import ModelRegistry

    e = ModelRegistry(_cfg().paths().ensure().models).promote(name, version)
    console.print(f"{e.name} v{e.version} is now active")


@models_app.command("rollback")
def models_rollback(name: str) -> None:
    from jarvis.learn.registry import ModelRegistry

    e = ModelRegistry(_cfg().paths().ensure().models).rollback(name)
    console.print(f"rolled back to {e.name} v{e.version}" if e else "nothing to roll back to")


def calibrate_range() -> None:
    """Fit where your punches actually land (P(hit | gap)) from recorded fights; updates the policy."""
    import yaml

    from jarvis.common.config import read_yaml
    from jarvis.learn.hindsight import attack_outcomes, fit_range

    cfg = _cfg()
    outcomes = []
    for d in _sessions(cfg, ("fight",)):
        outcomes += attack_outcomes(d)
    fit = fit_range(outcomes)
    console.print_json(json.dumps(fit))
    if fit.get("attack_gap") is not None:
        cal = cfg.paths().ensure().calibration
        f = cal / "policy_overrides.yaml"
        over = read_yaml(f)
        over.setdefault("spacing", {})["attack_gap"] = round(float(fit["attack_gap"]), 3)
        f.write_text(yaml.safe_dump(over, sort_keys=False), encoding="utf-8")
        console.print(f"attack_gap -> {fit['attack_gap']:.3f} ({f})")


def _newest_manifest(datasets, version: str):
    pats = [f"manifest-{version}.*"] if version else ["manifest-*.parquet", "manifest-*.jsonl"]
    found = sorted({p for pat in pats for p in datasets.glob(pat)}, key=lambda p: p.stat().st_mtime)
    return found[-1] if found else None


def _measured_latency(cal) -> float | None:
    f = cal / "latency.json"
    try:
        return float(json.loads(f.read_text(encoding="utf-8"))["median_ms"])
    except (OSError, ValueError, KeyError):
        return None


@train_app.command("threat")
def train_threat_cmd(
    dataset: str = typer.Option("", help="Dataset version (default: the newest `jarvis dataset build`)."),
    epochs: int = typer.Option(20),
    batch: int = typer.Option(32),
    backbone: str = typer.Option("mobilenet_v3_small", help="mobilenet_v3_small | tiny"),
    device: str = typer.Option("auto"),
    promote: bool = typer.Option(False, help="Make it active if it passes the promotion gate."),
) -> None:
    """Train the threat predictor (hindsight labels), test it on unseen fights, register it."""
    from jarvis.common.config import update_user_config
    from jarvis.learn import train_threat as tt
    from jarvis.learn.datasets import read_manifest
    from jarvis.learn.evaluate import Gate, promotion_decision
    from jarvis.learn.registry import ModelRegistry

    cfg = _cfg()
    paths = cfg.paths().ensure()
    manifest = _newest_manifest(paths.datasets, dataset)
    if manifest is None:
        console.print("[red]no dataset[/] - record fights, then run `jarvis dataset build`")
        raise typer.Exit(1)
    version = manifest.stem.removeprefix("manifest-")
    crop = cfg.fight.perception.crop_size
    data_dir = paths.datasets / f"threat-{version}"
    if not (data_dir / "info.json").exists():
        console.print(f"cutting crops for {manifest.name} ...")
        tt.prepare_crops(read_manifest(manifest), paths.recordings, data_dir, crop, progress=console.print)
    net, _hist = tt.train(data_dir, epochs=epochs, batch=batch, backbone=backbone, device=device, progress=console.print)
    metrics = tt.evaluate(net, data_dir, device)
    onnx = tt.export_onnx(net, data_dir / "threat.onnx", crop)
    reg = ModelRegistry(paths.models)
    entry = reg.register("threat", onnx, dataset_version=version, metrics=metrics, notes=backbone)
    active = reg.active("threat")
    ok, why = promotion_decision(metrics, active.metrics if active else None, Gate(latency_ms=_measured_latency(paths.calibration)))
    console.print(f"threat v{entry.version}: {json.dumps({k: metrics.get(k) for k in ('n', 'ap_300', 'auc_300', 'pos_300', 'warning_ms_median')})}")
    console.print(f"gate: {'[green]pass[/]' if ok else '[yellow]hold[/]'} - {why}")
    if promote and ok:
        reg.promote("threat", entry.version)
        update_user_config(paths, {"fight": {"perception": {"threat_model": "threat/active.onnx"}}})
        console.print("promoted: the next arm uses it")


@train_app.command("wakeword")
def train_wakeword_cmd(
    name: str = typer.Option("jarvis_ko", help="The name used with `jarvis wakeword record`."),
    epochs: int = typer.Option(40, help="Training epochs."),
    tts: bool = typer.Option(True, help="Add synthetic takes from the local voice (several voices and speeds)."),
    activate: bool = typer.Option(True, help="Use it right away (adds it to voice.wake.extra_models)."),
) -> None:
    """Train your personal wake word (e.g. 자비스) from your recorded takes; runs next to "hey jarvis"."""
    from jarvis.common.config import update_user_config
    from jarvis.voice.wakeword_train import (
        OwwFeatures,
        load_data,
        synthesize_takes,
        train_wakeword,
        write_wav,
    )

    cfg = _cfg()
    paths = cfg.paths().ensure()
    d = paths.datasets / "wakeword" / name
    if not (d / "pos").exists():
        console.print(f"[red]no takes yet[/] - run `jarvis wakeword record --name {name}` first")
        raise typer.Exit(1)
    if tts and not (d / "synth_pos").exists():
        phrases = json.loads((d / "phrases.json").read_text(encoding="utf-8")) if (d / "phrases.json").exists() else ["자비스"]
        console.print("synthesizing extra takes with the local voice...")
        try:
            pos, neg = synthesize_takes(cfg, phrases)
            for i, x in enumerate(pos):
                write_wav(d / "synth_pos" / f"s{i:04d}.wav", x)
            for i, x in enumerate(neg):
                write_wav(d / "synth_neg" / f"s{i:04d}.wav", x)
        except Exception as e:
            console.print(f"[yellow]no synthetic takes ({e}); training on your recordings only[/]")
    m = train_wakeword(load_data(d), paths.models, name, OwwFeatures(paths.models), epochs=epochs, progress=console.print)
    console.print(f"model: {m['path']}\nheld-out recall {m['recall_heldout']:.0%}, threshold {m['threshold']}, "
                  f"~{m['false_wakes_per_hour_est']} false wakes/h before the second-stage check (your voice + the name)")
    if activate:
        extra = sorted({*cfg.voice.wake.extra_models, name})
        update_user_config(paths, {"voice": {"wake": {"extra_models": extra, "model_thresholds": {name: m["threshold"]}}}})
        console.print("activated - restart the voice with `jarvis restart core` (or `jarvis start`)")


@train_app.command("detector")
def train_detector_cmd(
    epochs: int = typer.Option(60),
    imgsz: int = typer.Option(640),
    base: str = typer.Option("yolo11n.pt"),
    promote: bool = typer.Option(False, help="Make it the active detector."),
) -> None:
    """Fine-tune the self/opponent detector on your recordings (pseudo-labels from tracking)."""
    from jarvis.common.config import update_user_config
    from jarvis.learn.registry import ModelRegistry
    from jarvis.learn.train_detector import build_yolo_dataset, train_detector

    cfg = _cfg()
    paths = cfg.paths().ensure()
    data = build_yolo_dataset(paths.recordings, paths.datasets / "detector", progress=console.print)
    onnx, metrics = train_detector(data, paths.datasets / "detector-runs", epochs=epochs, imgsz=imgsz, base=base)
    reg = ModelRegistry(paths.models)
    entry = reg.register("detector", onnx, dataset_version=time.strftime("%Y%m%d-%H%M%S"), metrics=metrics, notes=base)
    console.print(f"detector v{entry.version}: {metrics}")
    if promote:
        reg.promote("detector", entry.version)
        update_user_config(paths, {"fight": {"perception": {"detector_model": "detector/active.onnx", "detector_input": imgsz}}})
        console.print("promoted: the next arm uses it")


def coach(session: str = typer.Argument("", help="Fight session dir (default: the newest).")) -> None:
    """Run the between-match coach on a recorded match (bounded tactic/parameter nudges + a comment)."""
    import asyncio

    from jarvis.core import build_brain
    from jarvis.fight.policy.coach import coach_match
    from jarvis.fight.policy.params import load_params

    cfg = _cfg()
    paths = cfg.paths().ensure()
    fights = [d for d in _sessions(cfg, ("fight",)) if (d / "summary.json").exists() and (not session or d.name == session)]
    if not fights:
        console.print("[red]no recorded match with a summary[/]")
        raise typer.Exit(1)
    summary = json.loads((fights[-1] / "summary.json").read_text(encoding="utf-8"))
    broker, _agent, memory = build_brain(cfg)
    try:
        adv, applied = asyncio.run(coach_match(broker, summary, "ko" if cfg.language == "ko" else "en", paths.calibration,
                                               load_params(cfg.fight.policy_file, paths.calibration)))
    finally:
        memory.close()
    console.print(f"comment: {adv.comment}\nparams: {applied['params']}\npriors: {applied['priors']}"
                  + (f"\nignored: {adv.rejected}" if adv.rejected else ""))


def _learnerd_call(method: str, params: dict | None = None) -> dict | None:
    import asyncio

    from jarvis.common.ipc import RpcClient, RpcError, load_or_create_token

    cfg = _cfg()
    paths = cfg.paths().ensure()

    async def go() -> dict | None:
        client = RpcClient(cfg.ipc.rpc_ports["learnerd"], load_or_create_token(paths.run), cfg.ipc.host, timeout=3.0)
        try:
            return dict(await client.call(method, params or {}) or {})
        except RpcError:
            return None
        finally:
            await client.close()

    return asyncio.run(go())


@learn_app.command("status")
def learn_status() -> None:
    """What the background learner is doing and when each job last ran."""
    r = _learnerd_call("status")
    if r is None:
        console.print("learnerd is not running (the supervisor starts it; or run `jarvis learnerd`)")
        raise typer.Exit(1)
    st = r.get("state", {})
    fmt = lambda t: time.strftime("%Y-%m-%d %H:%M", time.localtime(t)) if t else "never"  # noqa: E731
    console.print(f"running: {r.get('running') or 'nothing'}   fight active: {r.get('fight_active')}   enabled: {r.get('enabled')}")
    console.print(f"nightly retrain: {fmt(st.get('last_nightly'))}   knowledge refresh: {fmt(st.get('last_kb'))}   "
                  f"memory consolidation: {fmt(st.get('last_consolidate'))}")
    console.print(f"videos today: {st.get('videos_today', 0)}   queued: {st.get('requested') or '-'}")
    if r.get("last_result"):
        console.print(f"last result: {json.dumps(r['last_result'], ensure_ascii=False, default=str)[:400]}")


@learn_app.command("now")
def learn_now(
    job: str = typer.Argument(..., help="post_fight | nightly | video | kb | consolidate"),
    here: bool = typer.Option(False, "--here", help="Run in this terminal instead of queueing it in learnerd."),
) -> None:
    """Run a learning job now (queued in learnerd; it still never runs during a fight)."""
    from jarvis.learn.jobs import JOBS

    if job not in JOBS:
        console.print(f"[red]unknown job[/] {job!r}; one of: {', '.join(JOBS)}")
        raise typer.Exit(2)
    if here:
        from jarvis.learn.jobs import run_job

        console.print(json.dumps(run_job(_cfg(), job), ensure_ascii=False, indent=1, default=str))
        return
    r = _learnerd_call("request", {"job": job})
    if r is None:
        console.print("learnerd is not running - use --here to run the job in this terminal")
        raise typer.Exit(1)
    console.print(f"queued: {r.get('queued')}" if r.get("ok") else f"[red]{r.get('error')}[/]")


@learn_app.command("cancel")
def learn_cancel() -> None:
    """Stop the running learning job."""
    r = _learnerd_call("cancel")
    console.print("learnerd is not running" if r is None else f"stopped: {r.get('stopped') or 'nothing was running'}")


def learnerd() -> None:
    """The background-learner process (started by the supervisor)."""
    from jarvis.common.logging_setup import setup_logging
    from jarvis.learnerd import main as learnerd_main

    cfg = _cfg()
    setup_logging("learnerd", cfg.paths().ensure().logs, cfg.log_level)
    learnerd_main(cfg)


def register(app: typer.Typer) -> None:
    app.command()(labels)
    app.add_typer(learn_app, name="learn")
    app.command()(learnerd)
    app.add_typer(dataset_app, name="dataset")
    app.add_typer(models_app, name="models")
    app.add_typer(train_app, name="train")
    app.command()(coach)
    from jarvis.cli_fight import calibrate_app

    calibrate_app.command("range")(calibrate_range)
