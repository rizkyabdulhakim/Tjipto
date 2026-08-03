from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import unittest

from scripts.evaluate_meaningful_support import evaluate_rows, load_artifacts
from tjipto.corpora.uud.meaningful_support_builder import build_meaningful_support_units


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "data" / "final" / "uud"
ORACLE_PATH = ROOT / "tests" / "fixtures" / "uud" / "meaningful_support_oracle.json"


def _rows(name: str) -> list[dict]:
    return [json.loads(line) for line in (FINAL / name).read_text(encoding="utf-8").splitlines() if line]


class MeaningfulSupportContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = _rows("meaningful_support_units.jsonl")
        cls.artifacts = load_artifacts(FINAL)
        cls.oracle = json.loads(ORACLE_PATH.read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls) -> None:
        del cls.rows, cls.artifacts, cls.oracle

    def test_projection_is_deterministic_and_matches_exact_artifact(self) -> None:
        kwargs = {
            "page_text_spans": self.artifacts["spans"],
            "raw_source_spans": self.artifacts["raw"],
            "evidence": self.artifacts["evidence"],
            "metadata_grounding": self.artifacts["metadata"],
            "source_conflicts": self.artifacts["conflicts"],
            "bbox_registry": self.artifacts["bboxes"],
            "word_bboxes": self.artifacts["words"],
        }
        first = build_meaningful_support_units(**kwargs)
        second = build_meaningful_support_units(**kwargs)
        self.assertEqual(first, second)
        self.assertEqual(first, self.rows)

    def test_independent_evaluator_accepts_all_reviewed_decisions(self) -> None:
        report = evaluate_rows(self.rows, self.artifacts, self.oracle)
        self.assertEqual(report["status"], "PASS", report)
        self.assertEqual(report["meaningful_span_count"], 1837)
        self.assertEqual(report["canonical_owner_span_count"], 1835)
        self.assertTrue(all(report["audited_spans"].values()))
        self.assertTrue(all(value == 0 for value in report["counters"].values()), report)

    def test_independent_evaluator_rejects_owner_lineage_and_policy_mutations(self) -> None:
        mutations = {
            "missing_owner": lambda rows: rows[0].update(owner_id="missing-owner"),
            "duplicate_owner": lambda rows: rows[0]["text_span_ids"].append(rows[1]["text_span_ids"][0]),
            "wrong_source_role": lambda rows: rows[0].update(source_role="current_consolidated" if rows[0]["source_role"] != "current_consolidated" else "original_historical"),
            "wrong_legal_force": lambda rows: rows[0].update(legal_force="metadata_only" if rows[0]["legal_force"] != "metadata_only" else "canonical_normative"),
            "invalid_selector": lambda rows: rows[0]["selector_refs"].__setitem__(0, "missing-selector"),
            "invalid_bbox": lambda rows: rows[0]["bbox_refs"].__setitem__(0, "missing-bbox"),
            "altered_hash": lambda rows: rows[0].update(quoted_text_sha256="0" * 64),
            "fabricated_finality": lambda rows: rows[0].update(citation_final=not rows[0]["citation_final"]),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                rows = deepcopy(self.rows)
                mutate(rows)
                self.assertEqual(evaluate_rows(rows, self.artifacts, self.oracle)["status"], "FAIL")

    def test_evaluator_does_not_import_or_call_projection_builder(self) -> None:
        source = (ROOT / "scripts" / "evaluate_meaningful_support.py").read_text(encoding="utf-8")
        self.assertNotIn("meaningful_support_builder", source)
        self.assertNotIn("build_meaningful_support_units", source)


if __name__ == "__main__":
    unittest.main()
