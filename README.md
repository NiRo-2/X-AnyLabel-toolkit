# X-AnyLabel-toolkit

Helper scripts for working with [X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling): JSON annotation utilities and YOLOv8 model export for custom AI models.

## Requirements

- Python 3.9+
- `pip install -r requirements.txt` for `yolov8_pt_to_xanylabeling_onnx.py` (ultralytics, PyYAML). The JSON annotation scripts use only the standard library.

## Setup

```bash
pip install -r requirements.txt
```

## Scripts

### Annotation tools (`rename_label.py`, `export_classes.py`)

Both scripts:

- Recursively scan all `*.json` files under `--in-dir`
- **Never modify JSON files under `--in-dir`** (read-only input)
- **Validate everything before writing** — if validation fails, no output is created or changed
- Print a `--- Summary ---` block at the end of every run

Annotation labels live in `shapes[].label` in each JSON file.

### rename_label.py

Batch-rename one or more labels across all JSON files. Output mirrors the input folder layout under `--out-dir`.

| Flag | Description |
|------|-------------|
| `--in-dir` | Input directory with JSON annotation files |
| `--out-dir` | Output directory for rewritten JSON files (must be outside `--in-dir`) |
| `--rename FROM TO` | Rename `FROM` to `TO`; repeat for multiple renames |

**Behavior:**

- Exact, case-sensitive label match (no partial matches)
- Each label is looked up once in the rename map (no chaining: `--rename A B --rename B C` does not turn `A` into `C`)
- Writes to a staging directory first, then promotes to `--out-dir` on success
- Warns if a `--rename FROM` label matches no shapes

Single rename:

```bash
python scripts/rename_label.py \
  --in-dir ./annotations \
  --out-dir ./annotations-renamed \
  --rename old_class new_class
```

Multiple renames in one run:

```bash
python scripts/rename_label.py \
  --in-dir ./annotations \
  --out-dir ./annotations-renamed \
  --rename old_class_a new_class_a \
  --rename old_class_b new_class_b
```

### export_classes.py

Build a YOLO-compatible `classes.txt` from all unique labels found in JSON files under `--in-dir`.

| Flag | Description |
|------|-------------|
| `--in-dir` | Input directory with JSON annotation files |
| `--out-classes` | Output path for `classes.txt` (may be inside or outside `--in-dir`) |

**Behavior:**

- One class name per line (line index = class ID, 0-based)
- Classes sorted alphabetically for stable output
- Only writes `classes.txt`; JSON annotation files are never modified
- Atomic write: existing `classes.txt` is replaced only after validation passes

```bash
python scripts/export_classes.py \
  --in-dir ./annotations \
  --out-classes ./classes.txt
```

### yolov8_pt_to_xanylabeling_onnx.py

Convert a trained YOLOv8 **detection** `.pt` checkpoint to fixed-size ONNX plus an X-AnyLabeling `config.yaml` (load via **AI → … → Load Custom Model**).

| Argument / flag | Description |
|-----------------|-------------|
| `weights` | Path to trained `.pt` weights |
| `--output-dir` | Output folder (default: `<weights_dir>/<stem>_xanylabeling`) |
| `--imgsz` | Square ONNX input size (default: from checkpoint, else 640) |
| `--conf` | `conf_threshold` in config.yaml (default: 0.25) |
| `--iou` | `iou_threshold` in config.yaml (default: 0.45) |
| `--name` | `name` field in config (default: weights stem) |
| `--display-name` | `display_name` in config (default: same as `--name`) |
| `--device` | Export device, e.g. `cpu`, `0`, `cuda:0` |

**Behavior:**

- Exports ONNX with `dynamic=False` (required by X-AnyLabeling)
- Writes `config.yaml` with class names from the checkpoint
- Detection task only; segmentation/classification checkpoints are rejected

```bash
python scripts/yolov8_pt_to_xanylabeling_onnx.py path/to/runs/detect/train/weights/best.pt
```

Custom output location and thresholds:

```bash
python scripts/yolov8_pt_to_xanylabeling_onnx.py best.pt \
  --output-dir ./models/my_detector \
  --imgsz 640 \
  --conf 0.3 \
  --iou 0.5 \
  --display-name "My detector"
```

## Suggested workflow

1. Rename labels if needed → `rename_label.py`
2. Export class list → `export_classes.py`
3. Train YOLOv8 (or use existing weights), then export for labeling → `yolov8_pt_to_xanylabeling_onnx.py`
4. Use `classes.txt` with YOLO export or training; load the ONNX bundle in X-AnyLabeling for auto-labeling

## Format reference

JSON annotations follow the [X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling) format. See the upstream [user guide](https://github.com/CVHub520/X-AnyLabeling/blob/main/docs/en/user_guide.md) for details.

## License

MIT — see [LICENSE](LICENSE).
