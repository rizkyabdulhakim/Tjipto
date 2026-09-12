from __future__ import annotations

import json
from pathlib import Path
from json.encoder import encode_basestring, encode_basestring_ascii
from typing import Any

from json.encoder import _make_iterencode  # type: ignore[attr-defined]


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


class _CanonicalJSONEncoder(json.JSONEncoder):
    """Canonicalize float spelling while streaming the original object graph."""

    def iterencode(self, value, _one_shot=False):
        markers: dict[int, Any] | None = {} if self.check_circular else None
        encoder = encode_basestring_ascii if self.ensure_ascii else encode_basestring

        def floatstr(number):
            if number != number:
                text = "NaN"
            elif number == float("inf"):
                text = "Infinity"
            elif number == -float("inf"):
                text = "-Infinity"
            else:
                return str(float(f"{number:.{CANONICAL_FLOAT_DIGITS}f}"))
            if not self.allow_nan:
                raise ValueError("Out of range float values are not JSON compliant: " + repr(number))
            return text

        iterator = _make_iterencode(
            markers,
            self.default,
            encoder,
            self.indent,
            floatstr,
            self.key_separator,
            self.item_separator,
            self.sort_keys,
            self.skipkeys,
            _one_shot,
        )
        return iterator(value, 0)


def write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(_canonicalize(data), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def write_json_in_place(path: Path, data: dict) -> None:
    """Write a large generated JSON object without a second object graph."""
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, cls=_CanonicalJSONEncoder, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            json.dump(_canonicalize(row), handle, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
