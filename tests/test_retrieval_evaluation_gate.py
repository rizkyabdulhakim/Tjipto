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
            "notes",
        }
        rows = [json.loads(line) for line in CASES.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertGreaterEqual(len(rows), 9)
        self.assertEqual(len({row["id"] for row in rows}), len(rows))
        for row in rows:
            self.assertLessEqual(required, set(row), row["id"])
        self.assertTrue(any(row["id"] == "unsafe_three_period_word" and row["case_status"] == "known_gap" for row in rows))

    def test_runner_reports_known_gap_without_failing_default_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report = Path(temp_dir) / "report.json"
            with contextlib.redirect_stdout(io.StringIO()):
                result = runner.main(["--report", str(report)])
            self.assertEqual(result, 0)
            data = json.loads(report.read_text(encoding="utf-8"))
        self.assertEqual(data["counts"]["pass"], 8)
        self.assertEqual(data["counts"]["fail"], 0)
        self.assertEqual(data["counts"]["known_gap"], 1)
        gap = next(row for row in data["results"] if row["outcome"] == "KNOWN_GAP")
        self.assertEqual(gap["id"], "unsafe_three_period_word")
        self.assertIn("forbidden_evidence_returned", ";".join(gap["errors"]))

    def test_strict_known_gap_mode_fails(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = runner.main(["--strict-known-gaps"])
        self.assertEqual(result, 1)
        self.assertIn("KNOWN_GAP: unsafe_three_period_word", output.getvalue())


if __name__ == "__main__":
    unittest.main()
