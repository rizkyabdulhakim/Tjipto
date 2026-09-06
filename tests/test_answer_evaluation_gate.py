from __future__ import annotations

import importlib.util
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "tests/fixtures/uud/answer_cases.jsonl"
RUNNER = ROOT / "scripts/evaluate_uud_answers.py"
BASELINE = ROOT / "tests/fixtures/uud/answer_baseline.json"
SPEC = importlib.util.spec_from_file_location("evaluate_uud_answers", RUNNER)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class AnswerEvaluationGateTest(unittest.TestCase):
    def test_cases_cover_answer_boundaries(self) -> None:
        rows = [json.loads(line) for line in CASES.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(len(rows), len({row["case_id"] for row in rows}))
        self.assertLessEqual(
            {"exact_reference", "paraphrase", "concept_synonym", "person_institution", "date_metadata", "historical", "relation", "multi_support", "ambiguity_source_scope", "ambiguity_legal_target", "ambiguity_relation_operation", "ambiguity_concept", "ambiguity_entity", "typo_noise", "out_of_corpus", "source_annotation", "source_discrepancy", "proposition", "proposition_contradiction"},
            {row["category"] for row in rows},
        )
        self.assertEqual({"answer", "abstain"}, {row["behavior"] for row in rows})

    def test_answer_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report = Path(temp_dir) / "answers.json"
            result = runner.main(["--report", str(report)])
            data = json.loads(report.read_text(encoding="utf-8"))
        self.assertEqual(result, 0)
        self.assertEqual(data["counts"]["fail"], 0)
        self.assertEqual(data["metrics"]["forbidden_support_false_positive_rate"], 0.0)

    def test_baseline_is_bound_to_the_frozen_evaluator_and_cases(self) -> None:
        baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
        self.assertEqual(baseline["case_set_sha256"], sha256(CASES.read_bytes()).hexdigest())
        self.assertEqual(baseline["evaluator_sha256"], sha256(RUNNER.read_bytes()).hexdigest())
        self.assertEqual(baseline["metrics"]["denominators"]["case_pass_rate"], baseline["counts"]["pass"] + baseline["counts"]["fail"])


if __name__ == "__main__":
    unittest.main()
