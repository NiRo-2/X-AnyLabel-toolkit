"""Shared helpers for X-AnyLabel-toolkit scripts."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def is_inside(path: Path, parent: Path) -> bool:
    """Return True if path is equal to or nested inside parent."""
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def find_json_files(in_dir: Path) -> list[Path]:
    """Return all JSON files under in_dir, sorted for deterministic order."""
    return sorted(in_dir.rglob("*.json"))


def load_json_file(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def validate_json_structure(data: Any, relative_path: str) -> list[str]:
    """Validate X-AnyLabeling JSON structure. Return list of error messages."""
    errors: list[str] = []

    if not isinstance(data, dict):
        errors.append(f"{relative_path}: root must be a JSON object")
        return errors

    if "shapes" not in data:
        return errors

    shapes = data["shapes"]
    if not isinstance(shapes, list):
        errors.append(f"{relative_path}: 'shapes' must be a list")
        return errors

    for index, shape in enumerate(shapes):
        if not isinstance(shape, dict):
            errors.append(f"{relative_path}: shapes[{index}] must be an object")
            continue
        if "label" in shape and not isinstance(shape["label"], str):
            errors.append(
                f"{relative_path}: shapes[{index}].label must be a string"
            )

    return errors


def validate_json_files(in_dir: Path, json_files: list[Path]) -> tuple[list[Any], list[str]]:
    """
    Load and validate all JSON files.

    Returns (parsed_data_list, errors). parsed_data_list aligns with json_files.
    On any error, parsed entries may be partial — caller must abort before writes.
    """
    parsed: list[Any] = []
    errors: list[str] = []

    for json_path in json_files:
        relative = json_path.relative_to(in_dir).as_posix()
        try:
            data = load_json_file(json_path)
        except json.JSONDecodeError as exc:
            errors.append(f"{relative}: invalid JSON ({exc.msg} at line {exc.lineno})")
            parsed.append(None)
            continue
        except OSError as exc:
            errors.append(f"{relative}: cannot read file ({exc})")
            parsed.append(None)
            continue

        structure_errors = validate_json_structure(data, relative)
        errors.extend(structure_errors)
        parsed.append(data)

    return parsed, errors


def print_validation_errors(errors: list[str]) -> None:
    print("Validation errors:", file=sys.stderr)
    for index, error in enumerate(errors, start=1):
        print(f"  {index}. {error}", file=sys.stderr)
    print("Validation failed — no files written.", file=sys.stderr)


def collect_labels_from_data(data: Any) -> set[str]:
    labels: set[str] = set()
    if not isinstance(data, dict):
        return labels
    shapes = data.get("shapes")
    if not isinstance(shapes, list):
        return labels
    for shape in shapes:
        if not isinstance(shape, dict):
            continue
        label = shape.get("label")
        if isinstance(label, str) and label:
            labels.add(label)
    return labels


def write_json_file(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def print_summary(lines: list[str], *, ok: bool) -> None:
    print("--- Summary ---")
    for line in lines:
        print(line)
    status = "ok" if ok else "completed with errors"
    print(f"Status:              {status}")
