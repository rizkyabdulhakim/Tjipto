from __future__ import annotations

import json
from pathlib import Path


# Six decimal places preserve sub-point PDF geometry while avoiding runtime-dependent float spellings.
CANONICAL_FLOAT_DIGITS = 6


def _canonicalize(value):
    if isinstance(value, float):
        return float(f"{value:.{CANONICAL_FLOAT_DIGITS}f}")
    if isinstance(value, dict):
        return {key: _canonicalize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_canonicalize(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_canonicalize(item) for item in value)
    return value


def write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(_canonicalize(data), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            json.dump(_canonicalize(row), handle, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
