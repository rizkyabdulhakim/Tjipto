from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "tests/fixtures/uud/retrieval_eval_cases.jsonl"
RUNNER = ROOT / "scripts/evaluate_uud_retrieval.py"
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
            "case_status",
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
        }
        rows = [json.loads(line) for line in CASES.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertGreaterEqual(len(rows), 40)
        self.assertEqual(len({row["id"] for row in rows}), len(rows))
        for row in rows:
            self.assertLessEqual(required, set(row), row["id"])
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

    def test_strict_known_gap_mode_passes(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = runner.main(["--strict-known-gaps"])
        self.assertEqual(result, 0)
        self.assertIn("KNOWN_GAP=0", output.getvalue())


if __name__ == "__main__":
    unittest.main()
