#!/usr/bin/env python3
"""Convert a trained YOLOv8 detection .pt checkpoint to ONNX + X-AnyLabeling config.yaml.

X-AnyLabeling expects fixed-size ONNX (no dynamic batch). Load the YAML via
AI -> ... -> Load Custom Model.

Requires: pip install -r requirements.txt (ultralytics, PyYAML).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any, Optional

import yaml
from ultralytics import YOLO  # type: ignore[union-attr]


def normalize_weights_path(raw: str) -> Path:
    cleaned = raw.strip().strip('"').strip("'")
    cleaned = cleaned.replace("\\", "/")
    return Path(cleaned).resolve()


def _coerce_imgsz(val: Any) -> Optional[int]:
    if val is None:
        return None
    if isinstance(val, (list, tuple)):
        if not val:
            return None
        return int(max(int(x) for x in val))
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _imgsz_from_checkpoint(weights: Path) -> Optional[int]:
    try:
        import torch  # type: ignore[import-untyped]
    except Exception:
        return None
    try:
        try:
            ckpt = torch.load(weights, map_location="cpu", weights_only=True)
        except TypeError:
            ckpt = torch.load(weights, map_location="cpu")
    except Exception:
        return None
    if not isinstance(ckpt, dict):
        return None
    ta = ckpt.get("train_args")
    if ta is None:
        return None
    if isinstance(ta, dict):
        return _coerce_imgsz(ta.get("imgsz"))
    return _coerce_imgsz(getattr(ta, "imgsz", None))


def resolve_imgsz(model: YOLO, weights: Path, override: Optional[int]) -> int:
    if override is not None:
        v = _coerce_imgsz(override)
        if v is None or v <= 0:
            raise ValueError("--imgsz must be a positive integer")
        return v

    args = getattr(model.model, "args", None)
    if isinstance(args, dict):
        v = _coerce_imgsz(args.get("imgsz"))
        if v:
            return v
    elif args is not None:
        v = _coerce_imgsz(getattr(args, "imgsz", None))
        if v:
            return v

    v = _imgsz_from_checkpoint(weights)
    if v:
        return v
    return 640


def class_list_from_model(model: YOLO) -> list[str]:
    names = model.names
    if isinstance(names, dict):
        return [names[i] for i in sorted(names)]
    return list(names)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert YOLOv8 detection .pt weights to ONNX + X-AnyLabeling config.yaml"
        )
    )
    parser.add_argument(
        "weights",
        type=str,
        help="Path to trained .pt weights (detection task)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory (default: <weights_dir>/<stem>_xanylabeling)"
        ),
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Square ONNX input size (default: from checkpoint, else 640)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="conf_threshold written to config.yaml (default: 0.25)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="iou_threshold written to config.yaml (default: 0.45)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="config name field (default: weights file stem)",
    )
    parser.add_argument(
        "--display-name",
        type=str,
        default=None,
        help="config display_name (default: same as --name)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for ONNX export (e.g. cpu, 0). Default: ultralytics default",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    weights = normalize_weights_path(args.weights)
    if not weights.is_file():
        print(f"[ERROR] Weights not found: {weights}", file=sys.stderr)
        return 1
    if weights.suffix.lower() != ".pt":
        print(f"[WARNING] Expected a .pt file, got: {weights.suffix}")

    if not (0.0 <= args.conf <= 1.0):
        print(f"[ERROR] --conf must be between 0.0 and 1.0, got: {args.conf}", file=sys.stderr)
        return 1
    if not (0.0 <= args.iou <= 1.0):
        print(f"[ERROR] --iou must be between 0.0 and 1.0, got: {args.iou}", file=sys.stderr)
        return 1

    print(f"\n[Load] {weights}")
    model = YOLO(str(weights))

    if getattr(model, "task", None) != "detect":
        print(
            f"[ERROR] Detection checkpoints only (task={model.task!r}).",
            file=sys.stderr,
        )
        return 1

    stem = weights.stem
    out_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (weights.parent / f"{stem}_xanylabeling")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        imgsz = resolve_imgsz(model, weights, args.imgsz)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    conf_name = args.name if args.name else stem
    display_name = args.display_name if args.display_name else conf_name
    classes = class_list_from_model(model)

    onnx_name = f"{stem}.onnx"
    target_onnx = out_dir / onnx_name

    print(f"\n[Export] ONNX  imgsz={imgsz}  dynamic=False  -> {target_onnx}")
    export_kw: dict[str, Any] = {
        "format": "onnx",
        "imgsz": imgsz,
        "dynamic": False,
        "simplify": True,
        "half": False,
    }
    if args.device is not None:
        export_kw["device"] = args.device

    exported_path = Path(model.export(**export_kw)).resolve()
    if exported_path != target_onnx.resolve():
        if target_onnx.is_file():
            target_onnx.unlink()
        shutil.move(str(exported_path), str(target_onnx))

    cfg_path = out_dir / "config.yaml"
    config_doc = {
        "type": "yolov8",
        "name": conf_name,
        "display_name": display_name,
        "provider": "Ultralytics",
        "model_path": onnx_name,
        "input_width": imgsz,
        "input_height": imgsz,
        "conf_threshold": float(args.conf),
        "iou_threshold": float(args.iou),
        "classes": classes,
    }
    with open(cfg_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(
            config_doc,
            handle,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )

    print(f"\n[OK] Wrote {target_onnx}")
    print(f"      Wrote {cfg_path}")
    print(f"      Classes ({len(classes)}): {classes}")
    print("\n  In X-AnyLabeling: AI -> ... -> Load Custom Model -> pick config.yaml\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
