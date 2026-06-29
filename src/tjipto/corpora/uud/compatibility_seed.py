from __future__ import annotations

from pathlib import Path

from tjipto.core.manifest import read_json, read_jsonl


def load_compatibility_seed(stage_dir: Path) -> dict:
    # ponytail: compatibility bridge until UUD extraction rebuilds these rows from source specs.
    return {
        "legal_units": read_jsonl(stage_dir / "legal_units.jsonl"),
        "chunks": read_jsonl(stage_dir / "chunks.jsonl"),
        "evidence": read_jsonl(stage_dir / "evidence_registry.jsonl"),
        "bbox_rows": read_jsonl(stage_dir / "bbox_registry.jsonl"),
        "retrieval_units": read_jsonl(stage_dir / "retrieval_units.jsonl"),
        "metadata_assertions": read_jsonl(stage_dir / "metadata.jsonl"),
        "metadata_graph_edges": read_jsonl(stage_dir / "metadata_graph_edges.jsonl"),
        "validation_report": read_json(stage_dir / "validation_report.json"),
    }
