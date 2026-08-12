from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tjipto.retrieval.hybrid import RetrievalHit, hybrid_search, reciprocal_rank_fusion
from tjipto.retrieval.research import ResearchIntent, execute_research, plan_research
from tjipto.retrieval.sufficiency import EvidenceRequirement, assess_sufficiency, collect_evidence_set


def _hit(evidence_id: str, lane: str, rank: int, score: float) -> RetrievalHit:
    return RetrievalHit(evidence_id, {"evidence_id": evidence_id}, lane, rank, score, lane)


class HybridResearchContractTest(unittest.TestCase):
    def test_rrf_deduplicates_and_never_adds_raw_scores(self) -> None:
        sparse = (_hit("b", "bm25", 1, 100.0), _hit("a", "bm25", 2, 1.0))
        dense = (_hit("a", "dense", 1, 0.99), _hit("b", "dense", 2, 0.1))
        fused = reciprocal_rank_fusion({"bm25": sparse, "dense": dense}, k=1)
        self.assertEqual([hit.evidence_id for hit in fused], ["a", "b"])
        self.assertEqual(fused[0].raw_score, None)
        self.assertEqual(fused[0].lane_provenance, (("bm25", 2, 1.0, "bm25"), ("dense", 1, 0.99, "dense")))
        self.assertEqual(fused[0].fused_score, fused[1].fused_score)

    def test_rrf_deduplicates_repeated_evidence_within_one_lane(self) -> None:
        fused = reciprocal_rank_fusion({"bm25": (_hit("same", "bm25", 3, 1.0), _hit("same", "bm25", 1, 2.0))})
        self.assertEqual(fused[0].lane_provenance, (("bm25", 1, 2.0, "bm25"),))

    def test_hybrid_falls_back_to_sparse_with_typed_reason(self) -> None:
        class Index:
            def search(self, query, limit):
                return [{"evidence_id": "sparse", "quoted_text": query, "_bm25_provenance": {"raw_score": 2.0, "rank": 1}}]

        with patch("tjipto.retrieval.hybrid.sparse_index_for_store", return_value=Index()), patch(
            "tjipto.retrieval.hybrid.dense_search",
            return_value={"status": "dense_unavailable", "reason": "worker_timeout", "matches": ()},
        ):
            result = hybrid_search(SimpleNamespace(), "query", 2)
        self.assertEqual(result["status"], "found")
        self.assertEqual(result["retrieval_degraded_reason"], "worker_timeout")
        self.assertEqual(result["matches"][0]["evidence_id"], "sparse")
        self.assertNotIn("raw_score", repr(result))

    def test_requirement_assignment_is_verified_and_non_overlapping(self) -> None:
        rows = (
            {"evidence_id": "current", "source_role": "current", "temporal_context": "current"},
            {"evidence_id": "history", "source_role": "historical", "temporal_context": "historical"},
        )
        store = SimpleNamespace()
        with patch("tjipto.retrieval.sufficiency.validate_answer_candidate", return_value=(True, "answer_evidence")):
            evidence = collect_evidence_set(
                store,
                rows,
                (
                    EvidenceRequirement("current", source_role="current"),
                    EvidenceRequirement("history", temporal_context="historical"),
                ),
            )
        assessment = assess_sufficiency(evidence, (EvidenceRequirement("current"), EvidenceRequirement("history")))
        self.assertTrue(evidence.complete)
        self.assertEqual(assessment.status, "complete")
        self.assertEqual(evidence.assignment_map(), {"current": ("current",), "history": ("history",)})

    def test_sufficiency_reports_partial_and_retry_without_fabrication(self) -> None:
        requirement = EvidenceRequirement("missing", allow_partial=True)
        evidence = collect_evidence_set(SimpleNamespace(), (), (requirement,))
        assessment = assess_sufficiency(evidence, (requirement,), partial_allowed=True, retry_budget=1)
        self.assertEqual(assessment.status, "insufficient")
        self.assertEqual(assessment.missing_requirement_ids, ("missing",))
        self.assertTrue(assessment.retry_allowed)

    def test_planner_retains_original_and_rejects_scope_changes(self) -> None:
        class Provider:
            def propose(self, request):
                return {"variants": [
                    {"query": "rewritten", "explicit_references": ("Pasal 1",)},
                    {"query": "same"},
                ]}

        intent = ResearchIntent(comparison=True, decomposition=True, max_variants=3)
        plan = plan_research("original", intent, provider=Provider(), explicit_references=("Pasal 2",))
        self.assertEqual(plan.variants[0].query, "original")
        self.assertEqual(plan.provider_status, "degraded")
        self.assertIn("scope_invariant_violation", plan.rejection_reasons)

    def test_simple_queries_do_not_invoke_provider_or_decompose(self) -> None:
        class Provider:
            def propose(self, request):
                raise AssertionError("simple query must not call planner")

        plan = plan_research("exact query", ResearchIntent(), provider=Provider())
        self.assertEqual(tuple(variant.query for variant in plan.variants), ("exact query",))
        seen = []
        _, rows = execute_research("original", lambda query, variant: seen.append(query) or (), intent=ResearchIntent())
        self.assertEqual(seen, ["original"])
        self.assertEqual(rows, ())


if __name__ == "__main__":
    unittest.main()
