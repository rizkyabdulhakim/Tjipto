from __future__ import annotations

from tjipto.core.manifest import read_jsonl


class GraphStore:
    def __init__(self, config):
        self.config = config

    def counts(self) -> dict[str, int]:
        return {
            "nodes": len(read_jsonl(self.config.graph_nodes_path)),
            "edges": len(read_jsonl(self.config.graph_edges_path)),
        }
