from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "tests/fixtures/uud/research_retrieval_cases.jsonl"
RUNNER = ROOT / "scripts/evaluate_uud_research_retrieval.py"
BASELINE = ROOT / "tests/fixtures/uud/research_retrieval_baseline.json"
SPEC = importlib.util.spec_from_file_location("evaluate_uud_research_retrieval", RUNNER)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class ResearchRetrievalEvaluationTest(unittest.TestCase):
    def test_candidate_pool_presence_cannot_replace_assigned_published_support(self) -> None:
        case = {
            "expected_status": "answer_ready",
            "gold_support_groups": [["gold-a", "gold-b"]],
        }
        response = {"status": "answer_ready", "evidence_set": {"assignments": (("r", ("unrelated",)),)}}
        errors = runner._compare(case, response, ["unrelated"], ["unrelated"], [])
        self.assertIn("support_group_missing", errors)

    def test_frozen_families_and_identity(self) -> None:
        rows = [json.loads(line) for line in CASES.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(len(rows), len({row["case_id"] for row in rows}))
        self.assertEqual(
            {"structural_navigation", "semantic_paraphrase", "layperson_legal_language", "broad_underspecified_concept", "comparison", "multi_support", "multi_hop_procedure", "hard_negative_out_of_corpus"},
            {row["family"] for row in rows},
        )
        self.assertIn("apa ketentuan sebelum Pasal 28?", {row["query"] for row in rows})
        baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
        self.assertEqual(baseline["evaluation_identity"]["case_set_sha256"], hashlib.sha256(CASES.read_bytes()).hexdigest())
        self.assertEqual(baseline["evaluation_identity"]["evaluator_sha256"], hashlib.sha256(RUNNER.read_bytes()).hexdigest())
        self.assertEqual(baseline["evaluation_identity"]["case_count"], len(rows))
        self.assertTrue(baseline["evaluation_identity"]["base_commit"])

    def test_current_benchmark_is_truthful(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report = Path(temp_dir) / "research.json"
            self.assertEqual(runner.main(["--report", str(report)]), 0)
            data = json.loads(report.read_text(encoding="utf-8"))
        self.assertEqual(data["counts"]["cases"], len(_read_cases()))
        self.assertEqual(data["execution_status"], "valid")
        self.assertEqual(data["blocking_failures"], 0)
        self.assertEqual(data["gap"], 0)
        self.assertEqual(data["met"], data["counts"]["cases"])


def _read_cases() -> list[dict]:
    return [json.loads(line) for line in CASES.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    unittest.main()
