from __future__ import annotations

class GraphStore:
    def __init__(self, config):
        self.config = config

    def counts(self) -> dict[str, int]:
        return {
            "nodes": len(self.config.jsonl("graph_nodes")),
            "edges": len(self.config.jsonl("graph_edges")),
        }
