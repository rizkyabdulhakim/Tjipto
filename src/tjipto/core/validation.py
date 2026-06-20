from __future__ import annotations

from .config import CorpusConfig
from .manifest import read_json


def validate_counts(final_dir) -> dict[str, int]:
    manifest_path = final_dir / "manifest.json"
    manifest = read_json(manifest_path)
    config = CorpusConfig(manifest["corpus_id"], manifest_path, manifest)
    return {
        "source_documents": len(config.jsonl("source_documents")),
        "evidence_records": len(config.jsonl("evidence")),
        "bbox_records": len(config.jsonl("bbox")),
        "graph_nodes": len(config.jsonl("graph_nodes")),
        "graph_edges": len(config.jsonl("graph_edges")),
    }
