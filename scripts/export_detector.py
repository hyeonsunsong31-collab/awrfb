"""Export a zero-shot character detector (YOLO-World) to ONNX for fightd - run once on your PC.

    uv pip install -e ".[detect]"        # ultralytics (AGPL-3.0; fine for personal use)
    uv run python scripts/export_detector.py

The model file lands in %LOCALAPPDATA%\\Jarvis\\models\\detector\\zero_shot.onnx and your user config
is updated to use it. Later, `jarvis train detector` fine-tunes a smaller model on your own
recordings and the model registry swaps it in only if it measures better.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml

from jarvis.common.config import load_config, read_yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolov8s-worldv2.pt")
    ap.add_argument("--classes", default="roblox character,person,roblox avatar")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    from ultralytics import YOLO

    cfg = load_config()
    paths = cfg.paths().ensure()
    model = YOLO(args.weights)
    model.set_classes([c.strip() for c in args.classes.split(",") if c.strip()])
    out = Path(model.export(format="onnx", imgsz=args.imgsz, simplify=True))
    dest = paths.models / "detector" / "zero_shot.onnx"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(out), dest)

    user = read_yaml(paths.user_config)
    user.setdefault("fight", {}).setdefault("perception", {})["detector_model"] = "detector/zero_shot.onnx"
    paths.user_config.write_text(yaml.safe_dump(user, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"detector exported to {dest}; config updated ({paths.user_config})")


if __name__ == "__main__":
    main()
