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


if __name__ == "__main__":
    unittest.main()
