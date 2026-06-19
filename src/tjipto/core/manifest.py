from __future__ import annotations

import hashlib
import json
from pathlib import Path


FORBIDDEN_ACTIVE_PATH_PARTS = (
    "candidate",
    "batch",
    "manual_review",
    "dry_run",
    "pilot",
    "remaining",
    "supplemental",
    "v1",
    "v2",
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest(final_dir: Path) -> tuple[str, ...]:
    manifest_path = final_dir / "manifest.json"
    manifest = read_json(manifest_path)
    errors: list[str] = []
    for key in ("source_documents", "evidence_registry", "bbox_registry", "graph_nodes", "graph_edges"):
        rel = manifest[key]
        lowered = rel.casefold()
        for part in FORBIDDEN_ACTIVE_PATH_PARTS:
            if part in lowered:
                errors.append(f"forbidden active path part:{rel}:{part}")
        path = final_dir / rel
        if not path.exists():
            errors.append(f"missing:{rel}")
        elif file_sha256(path) != manifest["files"][rel]["sha256"]:
            errors.append(f"sha256_mismatch:{rel}")
    return tuple(errors)
