# X-AnyLabel-toolkit

Helper scripts for working with [X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling) JSON annotation files.

## Requirements

- Python 3.9+

## Setup

```bash
pip install -r requirements.txt
```

No external pip packages are required today; the file is included for a consistent setup workflow.

## Scripts

Both scripts:

- Recursively scan all `*.json` files under `--in-dir`
- **Never modify `--in-dir`** (read-only input; all output goes elsewhere)
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
| `--out-classes` | Output path for `classes.txt` (must be outside `--in-dir`) |

**Behavior:**

- One class name per line (line index = class ID, 0-based)
- Classes sorted alphabetically for stable output
- Atomic write: existing `classes.txt` is replaced only after validation passes

```bash
python scripts/export_classes.py \
  --in-dir ./annotations \
  --out-classes ./classes.txt
```

## Suggested workflow

1. Rename labels if needed → `rename_label.py`
2. Export class list → `export_classes.py`
3. Use `classes.txt` with X-AnyLabeling YOLO export or training pipelines

## Format reference

JSON annotations follow the [X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling) format. See the upstream [user guide](https://github.com/CVHub520/X-AnyLabeling/blob/main/docs/en/user_guide.md) for details.

## License

MIT — see [LICENSE](LICENSE).
