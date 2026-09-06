from __future__ import annotations

import unittest
from unittest.mock import patch

from scripts.attest_dense_runtime import _hybrid_activation_valid
from scripts.evaluate_dense_retrieval import _metrics, _runtime_identity


class DenseEvaluationContractTest(unittest.TestCase):
    def test_missed_ranking_cannot_receive_perfect_ndcg(self) -> None:
        cases = [{"query": "q", "gold_groups": [["gold"]]}]
        metrics = _metrics(cases, [[{"evidence_id": "other"}]])
        self.assertEqual(metrics["hit_rate_at_k"], 0.0)
        self.assertEqual(metrics["ndcg_at_k"], 0.0)
        self.assertEqual(metrics["support_group_recall_at_k"], 0.0)
        self.assertEqual(metrics["relevant_item_denominator"], 1)

    def test_ndcg_uses_ideal_cutoff_independent_of_returned_length(self) -> None:
        cases = [{"query": "q", "gold_groups": [["gold-a", "gold-b"]]}]
        metrics = _metrics(cases, [[{"evidence_id": "gold-a"}]])
        self.assertLess(metrics["ndcg_at_k"], 1.0)
        self.assertEqual(metrics["relevant_item_denominator"], 2)

    def test_gitless_identity_is_typed_unavailable(self) -> None:
        class Args:
            runtime_commit = None
            runtime_tree = None
            identity_sidecar = None

        with patch("scripts.evaluate_dense_retrieval._git_optional", return_value="unavailable"):
            identity = _runtime_identity(Args())
        self.assertEqual(identity["runtime_commit"], "unavailable")
        self.assertEqual(identity["runtime_tree_sha"], "unavailable")

    def test_true_hybrid_contract_requires_both_nonempty_lanes(self) -> None:
        activation = {
            "dense_configured": True,
            "dense_runtime_available": True,
            "hybrid_active": True,
            "route": "hybrid",
            "contributing_lanes": ("bm25", "dense"),
            "fusion": {"algorithm": "rrf_rank_only", "lane_candidate_counts": {"bm25": 1, "dense": 1}},
        }
        self.assertTrue(_hybrid_activation_valid(activation))
        activation["fusion"]["lane_candidate_counts"]["dense"] = 0
        self.assertFalse(_hybrid_activation_valid(activation))


if __name__ == "__main__":
    unittest.main()
