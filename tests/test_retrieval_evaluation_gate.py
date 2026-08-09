from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "tests/fixtures/uud/retrieval_cases.jsonl"
RUNNER = ROOT / "scripts/evaluate_uud_retrieval.py"
BASELINE = ROOT / "tests/fixtures/uud/retrieval_baseline.json"
SPEC = importlib.util.spec_from_file_location("evaluate_uud_retrieval", RUNNER)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class RetrievalEvaluationGateTest(unittest.TestCase):
    def test_fixture_rows_are_explicit(self) -> None:
        required = {
            "id",
            "query",
            "corpus_id",
            "expected_status",
            "expected_support_type",
            "expected_legal_unit_ids",
            "expected_evidence_ids",
            "forbidden_legal_unit_ids",
            "forbidden_evidence_ids",
            "expected_claims",
            "expected_claim_support",
            "expected_predicate",
            "expected_polarity",
            "expected_modality",
            "expected_reason_code",
            "expected_source_role",
            "expected_temporal_context",
            "expected_needed_corpora",
            "forbidden_support_ids",
            "risk_family",
            "notes",
            "category",
            "expected_behavior",
            "gold_support_ids",
            "alternative_support_ids",
            "minimal_span_ids",
            "expected_source_roles",
            "expected_temporal_scopes",
            "expected_public_targets",
        }
        rows = [json.loads(line) for line in CASES.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertGreaterEqual(len(rows), 40)
        self.assertEqual(len({row["id"] for row in rows}), len(rows))
        for row in rows:
            self.assertLessEqual(required, set(row), row["id"])
            self.assertEqual(row["gold_support_ids"], row["expected_evidence_ids"], row["id"])
            self.assertFalse(set(row["gold_support_ids"]) & set(row["forbidden_support_ids"]), row["id"])
        ids = {row["id"] for row in rows}
        families = {row["risk_family"] for row in rows}
        self.assertIn("criminal_law_out_of_corpus", families)
        self.assertIn("explicit_article_wrong_function", families)
        self.assertIn("pasal_7a_removal_ground_positive", families)
        self.assertIn("deletion_relation_synonyms", families)
        self.assertIn("relation_vs_exact_article_arbitration", families)
        self.assertIn("metadata_non_final", families)
        self.assertIn("source_anomaly_non_final", families)
        self.assertIn("current_fact_unsupported", families)
        self.assertIn("criminal_law_sanksi_korupsi_pasal_7a", ids)
        self.assertIn("article_relation_exact_pasal_16_delete_no_source", ids)
        behaviors = {behavior: sum(row["expected_behavior"] == behavior for row in rows) for behavior in ("retrieve", "abstain", "clarify")}
        self.assertTrue(all(behaviors.values()), behaviors)
        self.assertTrue(any(row["category"] == "source_annotation" for row in rows))

    def test_empty_denominators_are_not_applicable(self) -> None:
        metrics = runner._retrieval_metrics([], [])
        self.assertIsNone(metrics["precision"])
        self.assertIsNone(metrics["ndcg"])
        self.assertIsNone(metrics["clarification_accuracy"])

    def test_missed_ranking_has_zero_ndcg_and_missed_clarification_is_not_perfect(self) -> None:
        cases = [
            {
                "id": "missed",
                "expected_behavior": "retrieve",
                "gold_support_ids": ["gold"],
                "alternative_support_ids": [],
                "minimal_span_ids": [],
                "expected_source_roles": [],
                "expected_temporal_scopes": [],
                "expected_public_targets": [],
            },
            {
                "id": "ambiguous",
                "expected_behavior": "clarify",
            },
        ]
        results = [
            {"id": "missed", "actual": {"support_ids": [], "text_span_ids": [], "source_roles": [], "temporal_scopes": [], "public_targets": [], "status": "no_results"}},
            {"id": "ambiguous", "actual": {"support_ids": [], "status": "answer_ready"}},
        ]
        metrics = runner._retrieval_metrics(cases, results)
        self.assertEqual(metrics["ndcg"], 0.0)
        self.assertEqual(metrics["clarification_accuracy"], 0.0)

    def test_runner_reports_no_known_gaps(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report = Path(temp_dir) / "report.json"
            with contextlib.redirect_stdout(io.StringIO()):
                result = runner.main(["--report", str(report)])
            self.assertEqual(result, 0)
            data = json.loads(report.read_text(encoding="utf-8"))
        expected = len([line for line in CASES.read_text(encoding="utf-8").splitlines() if line.strip()])
        self.assertEqual(data["counts"]["pass"], expected)
        self.assertEqual(data["counts"]["fail"], 0)
        self.assertEqual(data["counts"]["known_gap"], 0)
        self.assertTrue(all(value == 0 for value in data["acceptance_counters"].values()))
        self.assertEqual(data["metrics"]["hard_negative_false_positive_rate"], 0.0)

    def test_baseline_binds_the_same_cases_and_evaluator(self) -> None:
        baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
        self.assertEqual(baseline["case_set_sha256"], hashlib.sha256(CASES.read_bytes()).hexdigest())
        self.assertEqual(baseline["evaluator_sha256"], hashlib.sha256(RUNNER.read_bytes()).hexdigest())
        rows = [json.loads(line) for line in CASES.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(baseline["case_count"], len(rows))
        self.assertEqual(
            baseline["behavior_counts"],
            {behavior: sum(row["expected_behavior"] == behavior for row in rows) for behavior in ("retrieve", "abstain", "clarify")},
        )
        self.assertTrue(all(value > 0 for value in baseline["denominators"].values()))


if __name__ == "__main__":
    unittest.main()
