from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path

from tjipto.core.manifest import file_sha256, read_jsonl


def write_json(path: Path, data: dict) -> None:
    path.write_bytes((json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_bytes("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8"))


def refresh_manifest(final_dir: Path, manifest: dict) -> None:
    counts = manifest.setdefault("counts", {})
    count_files = {
        "document_metadata": "document_metadata.jsonl",
        "legal_units": "legal_units.jsonl",
        "chunks": "chunks.jsonl",
        "evidence_records": "evidence_registry.jsonl",
        "bbox_records": "bbox_registry.jsonl",
        "metadata_grounding": "metadata_grounding.jsonl",
        "metadata_grounding_records": "metadata_grounding_registry.jsonl",
        "graph_nodes": "graph_nodes.jsonl",
        "graph_edges": "graph_edges.jsonl",
        "page_text_spans": "page_text_spans.jsonl",
        "retrieval_units": "retrieval_units.jsonl",
    }
    for key, filename in count_files.items():
        counts[key] = len(read_jsonl(final_dir / filename))
    for rel in manifest["files"]:
        path = final_dir / rel
        if path.exists():
            manifest["files"][rel]["bytes"] = path.stat().st_size
            manifest["files"][rel]["sha256"] = file_sha256(path)
    write_json(final_dir / "manifest.json", manifest)


def atomic_promote_artifacts(
    *,
    final_dir: Path,
    build: Callable[[Path], None],
    validate: Callable[[Path], tuple[str, ...]],
) -> None:
    final_dir = final_dir.resolve()
    with tempfile.TemporaryDirectory(prefix=".uud-stage-", dir=final_dir.parent) as tmp:
        tmp_dir = Path(tmp)
        stage_dir = tmp_dir / "stage"
        snapshot_dir = tmp_dir / "snapshot"
        shutil.copytree(final_dir, stage_dir)
        shutil.copytree(final_dir, snapshot_dir)
        build(stage_dir)
        errors = validate(stage_dir)
        if errors:
            raise ValueError(";".join(errors))
        promoted: list[str] = []
        try:
            for path in sorted(stage_dir.iterdir()):
                if path.is_file():
                    target = final_dir / path.name
                    path.replace(target)
                    promoted.append(path.name)
        except Exception:
            for name in promoted:
                (snapshot_dir / name).replace(final_dir / name)
            raise
