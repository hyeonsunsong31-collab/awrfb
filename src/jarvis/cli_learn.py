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


def register(app: typer.Typer) -> None:
    app.command()(labels)
    app.add_typer(dataset_app, name="dataset")
    app.add_typer(models_app, name="models")
    from jarvis.cli_fight import calibrate_app

    calibrate_app.command("range")(calibrate_range)
