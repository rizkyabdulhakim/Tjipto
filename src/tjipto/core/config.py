from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CorpusConfig:
    corpus_id: str
    final_dir: Path
    manifest_path: Path
    source_documents_path: Path
    evidence_registry_path: Path
    bbox_registry_path: Path
    graph_nodes_path: Path
    graph_edges_path: Path
