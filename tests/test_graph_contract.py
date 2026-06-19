from __future__ import annotations

from pathlib import Path
import unittest

from tjipto.corpora.adapter import config_for
from tjipto.graph.store import GraphStore


ROOT = Path(__file__).resolve().parents[1]


class GraphContractTest(unittest.TestCase):
    def test_graph_lite_counts_are_preserved(self) -> None:
        graph = GraphStore(config_for("uud", ROOT))
        self.assertEqual(graph.counts(), {"nodes": 2339, "edges": 3150})


if __name__ == "__main__":
    unittest.main()
