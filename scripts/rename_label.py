#!/usr/bin/env python3
"""Rename labels in X-AnyLabeling JSON annotation files."""

from __future__ import annotations

import argparse
import shutil
import sys
from copy import deepcopy
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _common import (
    collect_labels_from_data,
    find_json_files,
    is_inside,
    print_summary,
    print_validation_errors,
    validate_json_files,
    write_json_file,
)


def parse_rename_pairs(raw_pairs: list[list[str]] | None) -> dict[str, str]:
    if not raw_pairs:
        raise argparse.ArgumentTypeError("at least one --rename pair is required")

    mapping: dict[str, str] = {}
    for pair in raw_pairs:
        if len(pair) != 2:
            raise argparse.ArgumentTypeError(
                "--rename requires exactly two arguments: FROM TO"
            )
        source, target = pair
        if source == target:
            raise argparse.ArgumentTypeError(
                f"--rename '{source}' '{target}' is a no-op and not allowed"
            )
        if source in mapping:
            raise argparse.ArgumentTypeError(
                f"duplicate --rename source label: '{source}'"
            )
        mapping[source] = target
    return mapping


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rename labels in X-AnyLabeling JSON files. "
            "Input directory is read-only; output is written to --out-dir."
        )
    )
    parser.add_argument(
        "--in-dir",
        required=True,
        type=Path,
        help="Input directory containing JSON annotation files",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Output directory for rewritten JSON files",
    )
    parser.add_argument(
        "--rename",
        action="append",
        nargs=2,
        metavar=("FROM", "TO"),
        required=True,
        help="Rename FROM label to TO label (repeat for multiple renames)",
    )
    return parser


def validate_paths(in_dir: Path, out_dir: Path, staging_dir: Path) -> list[str]:
    errors: list[str] = []

    if not in_dir.exists():
        errors.append(f"--in-dir does not exist: {in_dir}")
    elif not in_dir.is_dir():
        errors.append(f"--in-dir is not a directory: {in_dir}")

    if in_dir.exists() and is_inside(out_dir, in_dir):
        errors.append(
            f"--out-dir must be outside --in-dir (input is read-only): {out_dir}"
        )

    if staging_dir.exists():
        errors.append(
            f"staging directory already exists (remove it and retry): {staging_dir}"
        )

    if not errors:
        out_parent = out_dir.parent
        try:
            out_parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            errors.append(f"cannot create output parent directory: {exc}")

    return errors


def apply_renames(data: object, rename_map: dict[str, str]) -> tuple[object, int, dict[str, int]]:
    updated = deepcopy(data)
    total = 0
    per_mapping = {source: 0 for source in rename_map}

    if not isinstance(updated, dict):
        return updated, total, per_mapping

    shapes = updated.get("shapes")
    if not isinstance(shapes, list):
        return updated, total, per_mapping

    for shape in shapes:
        if not isinstance(shape, dict):
            continue
        label = shape.get("label")
        if label in rename_map:
            shape["label"] = rename_map[label]
            total += 1
            per_mapping[label] += 1

    return updated, total, per_mapping


def promote_staging(staging_dir: Path, out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    staging_dir.rename(out_dir)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        rename_map = parse_rename_pairs(args.rename)
    except argparse.ArgumentTypeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    in_dir = args.in_dir.resolve()
    out_dir = args.out_dir.resolve()
    staging_dir = Path(str(out_dir) + ".staging")

    path_errors = validate_paths(in_dir, out_dir, staging_dir)
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

    label_hits = {source: 0 for source in rename_map}
    files_without_shapes = 0
    for data in parsed:
        if isinstance(data, dict) and "shapes" not in data:
            files_without_shapes += 1
        found = collect_labels_from_data(data)
        for source in rename_map:
            if source in found:
                label_hits[source] += 1

    for source, file_count in label_hits.items():
        if file_count == 0:
            target = rename_map[source]
            print(
                f"Warning: --rename '{source}' '{target}' matched no labels",
                file=sys.stderr,
            )

    files_scanned = len(json_files)
    files_modified = 0
    total_renames = 0
    per_mapping = {source: 0 for source in rename_map}
    execute_errors = 0

    try:
        staging_dir.mkdir(parents=True, exist_ok=False)

        for json_path, data in zip(json_files, parsed):
            relative = json_path.relative_to(in_dir)
            output_path = staging_dir / relative

            updated, count, mapping_counts = apply_renames(data, rename_map)
            if count > 0:
                files_modified += 1
            total_renames += count
            for source, mapping_count in mapping_counts.items():
                per_mapping[source] += mapping_count

            write_json_file(output_path, updated)

        promote_staging(staging_dir, out_dir)
    except OSError as exc:
        execute_errors += 1
        print(f"Error during write: {exc}", file=sys.stderr)
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)

    summary_lines = [
        f"Input directory:     {in_dir}",
        f"Output directory:    {out_dir}",
        f"Rename mappings:     {len(rename_map)}",
        f"Files scanned:       {files_scanned}",
        f"Files modified:      {files_modified}",
        f"Total renames:       {total_renames}",
    ]
    for source, target in rename_map.items():
        summary_lines.append(f"  {source} -> {target}: {per_mapping[source]}")
    summary_lines.extend(
        [
            f"Files without shapes: {files_without_shapes}",
            f"Errors:              {execute_errors}",
        ]
    )

    ok = execute_errors == 0
    print_summary(summary_lines, ok=ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
