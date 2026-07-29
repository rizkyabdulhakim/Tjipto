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
                projection = edge.get("relation_projection")
                if not projection or not projection.get("source_legal_unit_id"):
                    continue
                self.assertEqual(
                    (edge["source_id"], edge["target_id"]),
                    (
                        f"legal_unit::{projection['source_legal_unit_id']}",
                        f"legal_unit::{projection['target_legal_unit_id']}",
                    ),
                )
                self.assertEqual(projection["projection_direction"], "inverse")
                self.assertEqual(projection["relation_type"], edge["edge_type"])
                self.assertEqual(
                    (
                        projection["source_document_id"],
                        projection["target_document_id"],
                        projection["source_label"],
                        projection["target_label"],
                    ),
                    (
                        origin["relation_projection"]["target_document_id"],
                        origin["relation_projection"]["source_document_id"],
                        origin["relation_projection"]["target_label"],
                        origin["relation_projection"]["source_label"],
                    ),
                )

    def test_document_amendment_query_selects_persisted_graph_edges(self) -> None:
        config = CorpusRegistry(ROOT).resolve("uud")
        assert config is not None
        store = EvidenceStore(config)
        target, edges = amendment_relation_lookup(store, "UUD 1945 diubah oleh amandemen berapa")
        self.assertEqual(target, {"mode": "document", "role": "original_historical"})
        self.assertEqual(len(edges), 4)
        self.assertTrue(all(edge["edge_type"] == "AMENDED_BY" for edge in edges))
        self.assertTrue(all(edge["relation_id"].startswith("uud_document_relation::") for edge in edges))

    def test_runtime_relation_edges_embed_their_audited_projection(self) -> None:
        config = CorpusRegistry(ROOT).resolve("uud")
        assert config is not None
        projection = config.json("runtime_projection")["artifacts"]
        self.assertNotIn("document_relations", projection)
        self.assertNotIn("article_amendment_relations", projection)
        relation_edges = [edge for edge in projection["graph_edges"] if edge.get("relation_id")]
        self.assertTrue(relation_edges)
        for edge in relation_edges:
            relation = edge.get("relation_projection") or {}
            self.assertEqual(relation.get("relation_id"), edge["relation_id"])
            self.assertEqual(relation.get("relation_type"), edge["edge_type"])
            if relation.get("source_legal_unit_id"):
                self.assertEqual(edge["source_id"], f"legal_unit::{relation['source_legal_unit_id']}")
                self.assertEqual(edge["target_id"], f"legal_unit::{relation['target_legal_unit_id']}")

    def test_relation_schema_declares_canonical_addition_inverse(self) -> None:
        self.assertEqual(descriptor_for("ADDS").inverse, "INSERTED_BY")  # type: ignore[union-attr]
        self.assertEqual(descriptor_for("INSERTED_BY").inverse, "ADDS")  # type: ignore[union-attr]
        self.assertIsNone(descriptor_for("INSERTS"))

    def test_relation_projection_health_has_no_direction_or_endpoint_mismatch(self) -> None:
        config = CorpusRegistry(ROOT).resolve("uud")
        assert config is not None
        health = config.json("validation_report")["legal_graph_authority_health"]
        self.assertEqual(health["forward_relation_projection_endpoint_mismatch_count"], 0)
        self.assertEqual(health["inverse_relation_projection_endpoint_mismatch_count"], 0)
        self.assertEqual(health["relation_direction_mismatch_count"], 0)


if __name__ == "__main__":
    unittest.main()
