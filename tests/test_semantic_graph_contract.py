from __future__ import annotations

from pathlib import Path
import unittest

from tjipto.contracts.relations import descriptor_for
from tjipto.evidence.store import EvidenceStore
from tjipto.corpora.registry import CorpusRegistry
from tjipto.retrieval.relations import amendment_relation_lookup


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

    def test_declared_inverse_edges_are_persisted_and_directed(self) -> None:
        config = CorpusRegistry(ROOT).resolve("uud")
        assert config is not None
        edges = EvidenceStore(config).semantic_graph_edges
        by_id = {edge["edge_id"]: edge for edge in edges}
        inverses = [edge for edge in edges if edge.get("derived_from_edge_id")]
        self.assertTrue(inverses)
        for edge in inverses:
            with self.subTest(edge=edge["edge_id"]):
                origin = by_id[edge["derived_from_edge_id"]]
                descriptor = descriptor_for(origin["edge_type"])
                assert descriptor is not None
                self.assertEqual(edge["edge_type"], descriptor.inverse)
                self.assertEqual((edge["source_id"], edge["target_id"]), (origin["target_id"], origin["source_id"]))

    def test_document_amendment_query_selects_persisted_graph_edges(self) -> None:
        config = CorpusRegistry(ROOT).resolve("uud")
        assert config is not None
        store = EvidenceStore(config)
        target, edges = amendment_relation_lookup(store, "UUD 1945 diubah oleh amandemen berapa")
        self.assertEqual(target, {"mode": "document", "role": "original_historical"})
        self.assertEqual(len(edges), 4)
        self.assertTrue(all(edge["edge_type"] == "AMENDED_BY" for edge in edges))
        self.assertTrue(all(edge["relation_id"].startswith("uud_document_relation::") for edge in edges))


if __name__ == "__main__":
    unittest.main()
