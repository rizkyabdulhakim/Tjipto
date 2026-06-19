from __future__ import annotations

from .manifest import read_jsonl


def validate_counts(final_dir) -> dict[str, int]:
    return {
        "source_documents": len(read_jsonl(final_dir / "source_documents.jsonl")),
        "evidence_records": len(read_jsonl(final_dir / "evidence_registry.jsonl")),
        "bbox_records": len(read_jsonl(final_dir / "bbox_registry.jsonl")),
        "graph_nodes": len(read_jsonl(final_dir / "graph_nodes.jsonl")),
        "graph_edges": len(read_jsonl(final_dir / "graph_edges.jsonl")),
    }
