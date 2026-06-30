from __future__ import annotations

import json
from pathlib import Path


def write_json(path: Path, data: dict) -> None:
    path.write_bytes((json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_bytes("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8"))
