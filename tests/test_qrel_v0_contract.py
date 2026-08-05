from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.evaluate_qrel import REQUIRED_FIELDS, evaluate, evaluate_rows


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "data" / "final" / "uud"
QREL = ROOT / "evaluation" / "uud" / "qrel_v0.jsonl"


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class QrelV0ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.qrels = _rows(QREL)
        cls.supports = _rows(FINAL / "meaningful_support_units.jsonl")
        cls.report = evaluate(QREL, repo_root=ROOT)
        cls.identity = cls.qrels[0]["corpus_identity"]

    def test_candidate_catalog_is_complete_but_not_expert_accepted(self) -> None:
        self.assertEqual(self.report["framework_status"], "PASS", self.report["counters"])
        self.assertEqual(self.report["acceptance_status"], "BLOCKED")
        self.assertGreaterEqual(self.report["primary_count"], 50)
        self.assertGreaterEqual(self.report["adversarial_count"], 20)
        self.assertEqual(self.report["candidate_count"], len(self.qrels))
        self.assertEqual(self.report["adjudicated_count"], 0)
        self.assertEqual(self.report["unreviewed_counted_toward_acceptance"], 0)
        self.assertTrue(all(REQUIRED_FIELDS == row.keys() for row in self.qrels))
        tags = {tag for row in self.qrels for tag in row["coverage_tags"]}
        self.assertTrue({
            "exact_reference", "historical_text", "current_text", "amendment_instrument",
            "metadata_or_structure", "source_annotation_or_anomaly", "alternative_valid_evidence",
            "hard_negative", "unsupported_or_out_of_corpus", "signatories", "institutions_and_sessions",
            "decision_clause", "clarification",
        } <= tags)
        self.assertGreater(sum(row["expected_recovery_behavior"] == "clarify" for row in self.qrels), 0)

    def test_duplicate_unresolved_and_unreviewed_rows_fail_closed(self) -> None:
        duplicate = [deepcopy(self.qrels[0]), deepcopy(self.qrels[0])]
        report = evaluate_rows(duplicate, self.supports, self.identity)
        self.assertGreater(report["counters"]["duplicate_case_id_count"], 0)
        unresolved = [deepcopy(self.qrels[0])]
        unresolved[0]["gold_support_ids"] = ["missing-support"]
        report = evaluate_rows(unresolved, self.supports, self.identity)
        self.assertGreater(report["counters"]["unresolved_support_id_count"], 0)
        candidate = [deepcopy(self.qrels[0]) for _ in range(70)]
        for index, row in enumerate(candidate):
            row["case_id"] = f"candidate-{index}"
            row["case_kind"] = "primary" if index < 50 else "adversarial"
        report = evaluate_rows(candidate, self.supports, self.identity)
        self.assertEqual(report["adjudicated_count"], 0)
        self.assertEqual(report["acceptance_status"], "BLOCKED")

    def test_metrics_accept_alternative_support_and_detect_forbidden_support(self) -> None:
        row = deepcopy(next(item for item in self.qrels if len(item["alternative_valid_support_ids"]) == 1))
        row.update({"review_status": "adjudicated", "reviewer_role": "indonesian_constitutional_law_adjudicator", "reviewed_at": "2026-08-05T00:00:00Z"})
        alternative = row["alternative_valid_support_ids"][0]
        prediction = [{
            "case_id": row["case_id"],
            "retrieved_support_ids": [alternative, *row["forbidden_support_ids"][:1]],
            "minimal_span_ids": next(item["text_span_ids"] for item in row["minimal_relevant_spans"] if item["support_id"] == alternative),
            "highlighted_span_ids": next(item["text_span_ids"] for item in row["minimal_relevant_spans"] if item["support_id"] == alternative),
            "public_targets": [next(item for item in row["expected_public_targets"] if item["support_id"] == alternative)],
            "source_role": row["source_role"],
            "temporal_scope": row["temporal_scope"],
            "recovery_behavior": "retrieve",
        }]
        report = evaluate_rows([row], self.supports, self.identity, prediction, k=2)
        self.assertEqual(report["framework_status"], "PASS", report["counters"])
        self.assertEqual(report["metrics"]["recall@2"], 0.5)
        self.assertEqual(report["metrics"]["precision@2"], 0.5)
        adversarial = deepcopy(next(item for item in self.qrels if item["case_kind"] == "adversarial"))
        adversarial.update({"review_status": "adjudicated", "reviewer_role": "indonesian_constitutional_law_adjudicator", "reviewed_at": "2026-08-05T00:00:00Z"})
        prediction = [{"case_id": adversarial["case_id"], "retrieved_support_ids": adversarial["forbidden_support_ids"], "recovery_behavior": "retrieve"}]
        report = evaluate_rows([adversarial], self.supports, self.identity, prediction)
        self.assertEqual(report["metrics"]["hard_negative_false_positive_rate"], 1.0)

    def test_self_comparison_and_qrel_label_predictions_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "same.jsonl"
            path.write_text(json.dumps(self.qrels[0]) + "\n", encoding="utf-8")
            self.assertEqual(evaluate(path, path, ROOT)["counters"]["self_comparison_count"], 1)
        row = deepcopy(self.qrels[0])
        row.update({"review_status": "adjudicated", "reviewer_role": "expert", "reviewed_at": "2026-08-05T00:00:00Z"})
        report = evaluate_rows([row], self.supports, self.identity, [deepcopy(row)])
        self.assertGreater(report["counters"]["prediction_contains_qrel_label_count"], 0)

    def test_production_has_no_qrel_dependency(self) -> None:
        consumers = [
            path
            for package in (ROOT / "src/tjipto/retrieval", ROOT / "src/tjipto/runtime")
            for path in package.rglob("*.py")
            if "qrel" in path.read_text(encoding="utf-8").casefold()
        ]
        self.assertEqual(consumers, [])


if __name__ == "__main__":
    unittest.main()
