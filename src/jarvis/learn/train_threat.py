"""Train the threat predictor on your own recordings and export the streaming ONNX model.

Labels come from hindsight (our HP/guard dropped within h ms of a frame); splits are by session hash, so
the test score is on fights the model never saw. The result is registered as a *candidate*; it becomes
active only through the promotion gate (jarvis.learn.evaluate) or an explicit ``jarvis models promote``.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from itertools import groupby
from pathlib import Path

import numpy as np

from jarvis.fight.recorder import read_jsonl
from jarvis.learn.datasets import Sample, VideoFrames, crop_window
from jarvis.learn.evaluate import threat_metrics
from jarvis.learn.hindsight import label_session
from jarvis.learn.metrics import warning_times

log = logging.getLogger(__name__)

HORIZONS = (100, 200, 300, 500)
SPLITS = ("train", "val", "test")


# ------------------------------------------------------------------------------------- data
def prepare_crops(samples: list[Sample], recordings: Path, out_dir: Path, crop: int = 96,
                  progress: Callable[[str], None] | None = None) -> dict[str, int]:
    """Cut opponent crop windows out of the recorded videos once (training reads them memory-mapped)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    xs: dict[str, list[np.ndarray]] = {k: [] for k in SPLITS}
    ys: dict[str, list[list[int]]] = {k: [] for k in SPLITS}
    meta: dict[str, list[dict]] = {k: [] for k in SPLITS}
    hits: dict[str, list[float]] = {}
    for session, items in groupby(sorted(samples, key=lambda s: (s.session, s.end_vi)), key=lambda s: s.session):
        sd = recordings / session
        try:
            frames = VideoFrames(sd)
        except Exception as e:
            log.info("skipping %s (no readable video: %s)", session, e)
            continue
        t_of = {int(r.get("vi", i)): int(r["t_ns"]) for i, r in enumerate(read_jsonl(sd / "frames.jsonl")) if "t_ns" in r}
        hits[session] = [h.t_ns / 1e6 for h in label_session(sd).hits if h.kind == "hit_taken"]
        n = 0
        try:
            for s in items:
                x = crop_window(frames, s, crop)
                if x is None:
                    continue
                xs[s.split].append(x)
                ys[s.split].append([int(s.y[str(h)]) for h in HORIZONS])
                meta[s.split].append({"session": session, "end_vi": s.end_vi, "t_ms": t_of.get(s.end_vi, 0) / 1e6})
                n += 1
        finally:
            frames.close()
        if progress:
            progress(f"{session}: {n} windows")
    counts = {}
    for k in SPLITS:
        counts[k] = len(xs[k])
        if xs[k]:
            np.save(out_dir / f"X_{k}.npy", np.stack(xs[k]).astype(np.uint8))
            np.save(out_dir / f"Y_{k}.npy", np.asarray(ys[k], np.uint8))
        (out_dir / f"meta_{k}.json").write_text(json.dumps(meta[k]), encoding="utf-8")
    (out_dir / "hits.json").write_text(json.dumps(hits), encoding="utf-8")
    (out_dir / "info.json").write_text(json.dumps({"crop": crop, "horizons": HORIZONS, "counts": counts}), encoding="utf-8")
    return counts


def load_split(data_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray, list[dict]] | None:
    x = data_dir / f"X_{split}.npy"
    if not x.exists():
        return None
    return (np.load(x, mmap_mode="r"), np.load(data_dir / f"Y_{split}.npy"),
            json.loads((data_dir / f"meta_{split}.json").read_text(encoding="utf-8")))


# ------------------------------------------------------------------------------------- training
def _device(name: str | None):
    import torch

    if name and name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _augment(xb, rng: np.random.Generator):
    """Mirror (threat is left/right symmetric) and brightness/contrast jitter, per window."""
    import torch

    flip = torch.from_numpy(rng.random(xb.shape[0]) < 0.5)
    xb = torch.where(flip.view(-1, 1, 1, 1, 1), xb.flip(3), xb)
    gain = torch.from_numpy(rng.uniform(0.8, 1.2, (xb.shape[0], 1, 1, 1, 1)).astype(np.float32))
    bias = torch.from_numpy(rng.uniform(-20, 20, (xb.shape[0], 1, 1, 1, 1)).astype(np.float32))
    return (xb.float() * gain + bias).clamp(0, 255).to(torch.uint8)


def predict(net, x: np.ndarray, device, batch: int = 64) -> np.ndarray:
    import torch

    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = torch.from_numpy(np.array(x[i : i + batch])).to(device)  # copy: memory-mapped input is read-only
            out.append(torch.sigmoid(net(xb)).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(HORIZONS)), np.float32)


def train(data_dir: Path, *, epochs: int = 20, batch: int = 32, lr: float = 3e-4, backbone: str = "mobilenet_v3_small",
          pretrained: bool = True, device: str | None = None, seed: int = 0, window: int = 8,
          progress: Callable[[str], None] | None = None):
    import torch

    from jarvis.learn.threat_net import ThreatNet

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tr = load_split(data_dir, "train")
    if tr is None or len(tr[0]) == 0:
        raise ValueError("no training windows - record fights (`jarvis fight`) and run `jarvis dataset build` first")
    va = load_split(data_dir, "val")
    xtr, ytr, _ = tr
    dev = _device(device)
    net = ThreatNet(window=xtr.shape[1], backbone=backbone, pretrained=pretrained).to(dev)
    pos = ytr.mean(0).clip(1e-3, 1)
    pos_weight = torch.tensor(((1 - pos) / pos).clip(1, 20), dtype=torch.float32, device=dev)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    steps = max(1, epochs * int(np.ceil(len(xtr) / batch)))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.15)
    amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best, best_state, history = -1.0, None, []
    k300 = HORIZONS.index(300)
    for ep in range(epochs):
        net.train()
        order = rng.permutation(len(xtr))
        total = 0.0
        for i in range(0, len(order), batch):
            idx = np.sort(order[i : i + batch])
            xb = _augment(torch.from_numpy(np.array(xtr[idx])), rng).to(dev)
            yb = torch.from_numpy(ytr[idx].astype(np.float32)).to(dev)
            warm = torch.from_numpy(np.where(rng.random(len(idx)) < 0.2, rng.integers(1, xtr.shape[1], len(idx)), 0)).to(dev)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
                loss = loss_fn(net(xb, warm), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item() * len(idx)
        score = float("nan")
        if va is not None and len(va[0]):
            m = threat_metrics(va[1], predict(net, va[0], dev), HORIZONS)
            score = m[f"ap_{HORIZONS[k300]}"]
        history.append({"epoch": ep + 1, "loss": round(total / len(xtr), 4), "val_ap_300": score})
        if progress:
            progress(f"epoch {ep + 1}/{epochs}: loss {total / len(xtr):.4f}, val AP@300 {score:.3f}")
        key = score if not np.isnan(score) else -total
        if key > best:
            best, best_state = key, {k: v.detach().clone() for k, v in net.state_dict().items()}
    if best_state is not None:
        net.load_state_dict(best_state)
    return net, history


def evaluate(net, data_dir: Path, device=None) -> dict:
    dev = _device(device) if not hasattr(device, "type") else device
    te = load_split(data_dir, "test")
    if te is None or len(te[0]) == 0:
        return {"n": 0}
    x, y, meta = te
    p = predict(net, x, dev)
    m = threat_metrics(y, p, HORIZONS)
    hits = json.loads((data_dir / "hits.json").read_text(encoding="utf-8"))
    k = HORIZONS.index(300)
    leads: list[float] = []
    for session, rows in groupby(sorted(range(len(meta)), key=lambda i: (meta[i]["session"], meta[i]["t_ms"])), key=lambda i: meta[i]["session"]):
        idx = list(rows)
        leads += warning_times(np.array([meta[i]["t_ms"] for i in idx]), p[idx, k], np.array(hits.get(session, [])), m["thr_300"])
    if leads:
        m["warning_ms_median"] = float(np.median(leads))
        m["warned_share"] = round(float(np.mean([v > 0 for v in leads])), 3)
    return m


# ------------------------------------------------------------------------------------- export
def export_onnx(net, path: Path, crop: int) -> Path:
    import onnx
    import torch

    from jarvis.learn.threat_net import StreamingThreat

    net = net.cpu().eval()
    stream = StreamingThreat(net).eval()
    prev = torch.zeros(1, net.window - 1, net.dim)
    frame = torch.zeros(1, crop, crop, 3, dtype=torch.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    kw = dict(input_names=["prev", "frame"], output_names=["emb", "logits"], opset_version=17,
              dynamic_axes={"prev": {0: "batch"}, "frame": {0: "batch"}, "emb": {0: "batch"}, "logits": {0: "batch"}})
    import warnings

    try:
        with warnings.catch_warnings():  # the TorchScript exporter is deprecated but still the most portable
            warnings.simplefilter("ignore", DeprecationWarning)
            torch.onnx.export(stream, (prev, frame), str(path), dynamo=False, **kw)
    except TypeError:  # older torch without the dynamo switch
        torch.onnx.export(stream, (prev, frame), str(path), **kw)
    except Exception:  # newer torch without the legacy exporter (needs onnxscript): static batch of 1
        torch.onnx.export(stream, (prev, frame), str(path), dynamo=True, input_names=kw["input_names"],
                          output_names=kw["output_names"], opset_version=18)
    model = onnx.load(str(path))
    for k, v in {"window": str(net.window), "crop": str(crop), "horizons": ",".join(map(str, HORIZONS)),
                 "created": time.strftime("%Y-%m-%d %H:%M:%S")}.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = k, v
    onnx.save(model, str(path))
    return path
