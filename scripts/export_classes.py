#!/usr/bin/env python3
"""Export unique class labels from X-AnyLabeling JSON files to classes.txt."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _common import (
    collect_labels_from_data,
    find_json_files,
    print_summary,
    print_validation_errors,
    validate_json_files,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a YOLO-compatible classes.txt from labels in X-AnyLabeling "
            "JSON files. Input directory is read-only."
        )
    )
    parser.add_argument(
        "--in-dir",
        required=True,
        type=Path,
        help="Input directory containing JSON annotation files",
    )
    parser.add_argument(
        "--out-classes",
        required=True,
        type=Path,
        help="Output path for classes.txt",
    )
    return parser


def validate_paths(in_dir: Path, out_classes: Path) -> list[str]:
    errors: list[str] = []

    if not in_dir.exists():
        errors.append(f"--in-dir does not exist: {in_dir}")
    elif not in_dir.is_dir():
        errors.append(f"--in-dir is not a directory: {in_dir}")

    if not errors:
        try:
            out_classes.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            errors.append(f"cannot create output parent directory: {exc}")

    return errors


def atomic_write_text(path: Path, content: str) -> None:
    temp_path = Path(str(path) + ".tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            handle.write(content)
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    in_dir = args.in_dir.resolve()
    out_classes = args.out_classes.resolve()

    path_errors = validate_paths(in_dir, out_classes)
    if path_errors:
        print_validation_errors(path_errors)
        return 1

    json_files = find_json_files(in_dir)
    if not json_files:
        print_validation_errors([f"no JSON files found under --in-dir: {in_dir}"])
        return 1

    parsed, validation_errors = validate_json_files(in_dir, json_files)
    if validation_errors:
        print_validation_errors(validation_errors)
        return 1

    all_labels: set[str] = set()
    files_without_shapes = 0
    files_with_labels = 0

    for data in parsed:
        if isinstance(data, dict) and "shapes" not in data:
            files_without_shapes += 1

        labels = collect_labels_from_data(data)
        if labels:
            files_with_labels += 1
        all_labels.update(labels)

    sorted_labels = sorted(all_labels)
    if not sorted_labels:
        print(
            "Warning: no labels found in any JSON file; writing empty classes.txt",
            file=sys.stderr,
        )

    content = "\n".join(sorted_labels)
    if content:
        content += "\n"
    else:
        content = "\n"

    execute_errors = 0
    try:
        atomic_write_text(out_classes, content)
    except OSError as exc:
        execute_errors += 1
        print(f"Error during write: {exc}", file=sys.stderr)

    summary_lines = [
        f"Input directory:     {in_dir}",
        f"Output file:         {out_classes}",
        f"Files scanned:       {len(json_files)}",
        f"Files with labels:   {files_with_labels}",
        f"Files without shapes: {files_without_shapes}",
        f"Unique classes:      {len(sorted_labels)}",
        f"Errors:              {execute_errors}",
        "Classes (class_id: name):",
    ]
    if sorted_labels:
        for index, label in enumerate(sorted_labels):
            summary_lines.append(f"  {index}: {label}")
    else:
        summary_lines.append("  (none)")

    ok = execute_errors == 0
    print_summary(summary_lines, ok=ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
