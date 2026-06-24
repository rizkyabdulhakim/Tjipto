from __future__ import annotations

from pathlib import Path
import unittest

from tjipto.core.manifest import read_jsonl


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "data/final/uud"


class MetadataGraphIntegrityContractTest(unittest.TestCase):
    def test_metadata_graph_targets_resolve_to_metadata_assertions(self) -> None:
        metadata_ids = {row["metadata_id"] for row in read_jsonl(FINAL / "metadata.jsonl")}
        for edge in read_jsonl(FINAL / "metadata_graph_edges.jsonl"):
            if edge["edge_type"] == "HAS_METADATA":
                self.assertIn(edge["target_id"], metadata_ids, edge["edge_id"])
            self.assertEqual(edge["status"], "accepted")
            self.assertFalse(edge["runtime_loadable"])


if __name__ == "__main__":
    unittest.main()
