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
    path.write_bytes((json.dumps(_canonicalize(data), ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_bytes("".join(json.dumps(_canonicalize(row), ensure_ascii=False, allow_nan=False) + "\n" for row in rows).encode("utf-8"))
