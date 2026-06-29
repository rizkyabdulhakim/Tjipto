from __future__ import annotations

from pathlib import Path
import unittest

from tjipto.core.manifest import read_jsonl


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "data/final/uud"


class MetadataGraphIntegrityContractTest(unittest.TestCase):
    def test_metadata_graph_edges_resolve_for_all_edge_types(self) -> None:
        metadata_ids = {row["metadata_id"] for row in read_jsonl(FINAL / "metadata.jsonl")}
        graph_node_ids = {row["node_id"] for row in read_jsonl(FINAL / "graph_nodes.jsonl")}
        edge_types = set()
        for edge in read_jsonl(FINAL / "metadata_graph_edges.jsonl"):
            edge_types.add(edge["edge_type"])
            self.assertIn(edge["source_id"], metadata_ids | graph_node_ids, edge["edge_id"])
            self.assertIn(edge["target_id"], metadata_ids | graph_node_ids, edge["edge_id"])
            self.assertEqual(edge["status"], "accepted")
            self.assertFalse(edge["runtime_loadable"])
        self.assertEqual(edge_types, {"HAS_METADATA", "ISSUED_BY", "SIGNED_BY", "DECIDED_BY", "SOURCE_PUBLISHED_BY"})


if __name__ == "__main__":
    unittest.main()
