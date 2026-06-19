from __future__ import annotations

from pathlib import Path

from tjipto.core.config import CorpusConfig


def config_for(corpus_id: str, repo_root: Path | None = None) -> CorpusConfig | None:
    root = repo_root or Path(__file__).resolve().parents[3]
    final_dir = root / "data" / "final" / corpus_id
    if not (final_dir / "manifest.json").exists():
        return None
    return CorpusConfig(
        corpus_id=corpus_id,
        final_dir=final_dir,
        manifest_path=final_dir / "manifest.json",
        source_documents_path=final_dir / "source_documents.jsonl",
        evidence_registry_path=final_dir / "evidence_registry.jsonl",
        bbox_registry_path=final_dir / "bbox_registry.jsonl",
        graph_nodes_path=final_dir / "graph_nodes.jsonl",
        graph_edges_path=final_dir / "graph_edges.jsonl",
    )
