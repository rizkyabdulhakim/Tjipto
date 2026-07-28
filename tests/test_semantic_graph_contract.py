from __future__ import annotations

from pathlib import Path
import unittest

from tjipto.evidence.store import EvidenceStore
from tjipto.corpora.registry import CorpusRegistry


ROOT = Path(__file__).resolve().parents[1]


class SemanticGraphContractTest(unittest.TestCase):
    def test_runtime_graph_uses_persisted_semantic_edges_only(self) -> None:
        config = CorpusRegistry(ROOT).resolve("uud")
        assert config is not None
        store = EvidenceStore(config)
        edges = store.semantic_graph_edges
        self.assertTrue(edges)
        self.assertTrue(all(edge.get("runtime_loadable") is True for edge in edges))
        self.assertTrue(all(edge["edge_type"] not in {"PAGE_GROUNDED_AT", "HAS_BBOX", "USES_SOURCE_PDF"} for edge in edges))
        self.assertTrue(all(not edge["edge_id"].startswith("relation::") for edge in edges))


if __name__ == "__main__":
    unittest.main()
