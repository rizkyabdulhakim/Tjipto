from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "evaluate_semantic_generalization.py"
CASES = ROOT / "tests" / "fixtures" / "uud" / "semantic_generalization_cases.jsonl"
BASELINE = ROOT / "tests" / "fixtures" / "uud" / "semantic_generalization_baseline.json"
SPEC = importlib.util.spec_from_file_location("evaluate_semantic_generalization", RUNNER)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class SemanticGeneralizationEvaluationTest(unittest.TestCase):
    def test_frozen_held_out_suite_covers_required_families(self) -> None:
        rows = [json.loads(line) for line in CASES.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(len(rows), len({row["case_id"] for row in rows}))
        self.assertEqual(
            {
                "paraphrase_lexical_mismatch", "realistic_user_scenario", "ambiguity", "negation_modality",
                "comparison", "multi_hop_relation", "historical_vs_current", "out_of_corpus_hard_negative",
            },
            {row["family"] for row in rows},
        )

    def test_v0_and_orchestrated_contract_is_valid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "semantic-generalization.json"
            self.assertEqual(runner.main(["--report", str(report)]), 0)
            data = json.loads(report.read_text(encoding="utf-8"))
        baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
        self.assertEqual(data["status"], "valid")
        self.assertEqual(data["evaluation_identity"], baseline["evaluation_identity"])
        self.assertEqual(data["metrics"]["hard_negative_fp"], 0)
        self.assertEqual(data["metrics"]["query_drift_rate"], 0)
        self.assertGreaterEqual(
            data["metrics"]["orchestrated_required_support_recall"],
            data["metrics"]["v0_required_support_recall"],
        )


if __name__ == "__main__":
    unittest.main()
