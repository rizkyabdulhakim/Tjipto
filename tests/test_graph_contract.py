from __future__ import annotations

from pathlib import Path
import unittest

from tjipto.corpora.adapter import config_for
from tjipto.graph.store import GraphStore
from tjipto.core.manifest import read_jsonl


ROOT = Path(__file__).resolve().parents[1]


class GraphContractTest(unittest.TestCase):
    def test_graph_lite_counts_are_preserved(self) -> None:
        graph = GraphStore(config_for("uud", ROOT))
        self.assertEqual(graph.counts(), {"nodes": 2339, "edges": 3150})

    def test_article_versions_are_source_scoped_not_equivalence_claims(self) -> None:
        rows = read_jsonl(ROOT / "data/final/uud/article_versions.jsonl")
        self.assertEqual(len(rows), 218)
        for row in rows:
            self.assertIn("not_accepted_legal_equivalence", row["cross_source_equivalence_status"])
            self.assertTrue(row["members"])
            for member in row["members"]:
                self.assertIn("chunk_id", member)
                self.assertNotIn("chunk_candidate_id", member)

    def test_source_conflicts_reference_source_documents(self) -> None:
        source_ids = {
            row["source_document_id"]
            for row in read_jsonl(ROOT / "data/final/uud/source_documents.jsonl")
        }
        rows = read_jsonl(ROOT / "data/final/uud/source_conflicts.jsonl")
        self.assertEqual(len(rows), 1)
        for row in rows:
            self.assertIn(row["source_document_id"], source_ids)
            self.assertNotEqual(row["status"], "unresolved_review_required_required")

    def test_validation_artifacts_resolve_refs(self) -> None:
        source_ids = {
            row["source_document_id"]
            for row in read_jsonl(ROOT / "data/final/uud/source_documents.jsonl")
        }
        chunk_ids = {
            row["chunk_id"]
            for row in read_jsonl(ROOT / "data/final/uud/chunks.jsonl")
        }
        unit_ids = {
            row["legal_unit_id"]
            for row in read_jsonl(ROOT / "data/final/uud/legal_units.jsonl")
        }
        alignment = read_jsonl(ROOT / "data/final/uud/validation_alignment_results.jsonl")
        self.assertEqual(len(alignment), 610)
        for row in alignment:
            if row.get("legal_unit_id"):
                self.assertIsNotNone(row.get("source_document_id"))
            if row.get("chunk_id"):
                self.assertIn(row["chunk_id"], chunk_ids)
            if row.get("legal_unit_id"):
                self.assertIn(row["legal_unit_id"], unit_ids)
            if row.get("source_document_id"):
                self.assertIn(row["source_document_id"], source_ids)
        exceptions = read_jsonl(ROOT / "data/final/uud/validation_exceptions.jsonl")
        self.assertEqual(len(exceptions), 19)
        for row in exceptions:
            if row.get("source_document_id"):
                self.assertIn(row["source_document_id"], source_ids)
            if row.get("chunk_id"):
                self.assertIn(row["chunk_id"], chunk_ids)

    def test_metadata_graph_edges_exclude_source_role_level_amends(self) -> None:
        edges = read_jsonl(ROOT / "data/final/uud/metadata_graph_edges.jsonl")
        self.assertEqual(len(edges), 449)
        self.assertFalse(
            [edge for edge in edges if edge["edge_type"] in {"AMENDS", "AMENDED_BY"}]
        )
        self.assertTrue(all(edge["status"] == "accepted" for edge in edges))
        self.assertTrue(all(edge["runtime_loadable"] is False for edge in edges))

    def test_amends_edges_are_preserved_as_not_promoted_exceptions(self) -> None:
        exceptions = read_jsonl(ROOT / "data/final/uud/validation_exceptions.jsonl")
        amends = [
            row for row in exceptions
            if row.get("edge_type") in {"AMENDS", "AMENDED_BY"}
        ]
        self.assertEqual(len(amends), 8)
        for row in amends:
            self.assertEqual(row["status"], "not_promoted_source_role_level_only")
            self.assertFalse(row["runtime_loadable"])


if __name__ == "__main__":
    unittest.main()
