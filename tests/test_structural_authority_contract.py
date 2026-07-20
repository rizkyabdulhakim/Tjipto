from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import unittest

from tjipto.core.manifest import read_jsonl
from tjipto.corpora.uud.validation import _structural_authority_contract_health


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "data/final/uud"


class StructuralAuthorityContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.units = read_jsonl(FINAL / "legal_units.jsonl")
        self.chunks = read_jsonl(FINAL / "chunks.jsonl")
        self.nodes = read_jsonl(FINAL / "graph_nodes.jsonl")
        self.edges = read_jsonl(FINAL / "graph_edges.jsonl")
        self.retrieval = read_jsonl(FINAL / "retrieval_units.jsonl")

    def test_rebuilt_contract_is_complete(self) -> None:
        self.assertEqual(
            _structural_authority_contract_health(self.units, self.chunks, self.nodes, self.edges, self.retrieval)["status"], "complete"
        )

    def test_corrupt_parent_and_edge_fail(self) -> None:
        units = deepcopy(self.units)
        edges = deepcopy(self.edges)
        units[1]["parent_legal_unit_ids"] = ["missing-parent"]
        edges[0]["support_kind"] = None
        health = _structural_authority_contract_health(units, self.chunks, self.nodes, edges, self.retrieval)
        self.assertGreater(health["bad_parent_count"], 0)
        self.assertGreater(health["bad_graph_edge_count"], 0)


if __name__ == "__main__":
    unittest.main()
