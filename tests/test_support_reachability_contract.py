from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import unittest

from scripts.evaluate_meaningful_support import evaluate_rows, load_artifacts
from scripts.evaluate_support_reachability import evaluate
from tjipto.runtime.api import handle_request
from tjipto.runtime.service import LegalRuntimeService


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "data" / "final" / "uud"


def _rows(name: str) -> list[dict]:
    return [json.loads(line) for line in (FINAL / name).read_text(encoding="utf-8").splitlines() if line]


class SupportReachabilityContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = _rows("meaningful_support_units.jsonl")
        cls.artifacts = load_artifacts(FINAL)
        cls.oracle = json.loads((ROOT / "tests/fixtures/uud/meaningful_support_oracle.json").read_text(encoding="utf-8"))
        cls.report = evaluate(ROOT)

    def test_every_decision_has_a_deterministic_terminal_route(self) -> None:
        self.assertEqual(self.report["status"], "PASS", self.report["counters"])
        self.assertEqual(self.report["decision_count"], len(self.rows))
        self.assertEqual(self.report["resolved_viewer_count"], sum(row["viewer_eligible"] for row in self.rows))
        self.assertEqual(self.report["exact_highlight_count"], sum(row["bbox_precision"] == "exact" for row in self.rows))
        self.assertEqual(self.report["page_grounded_count"], sum(row["bbox_precision"] == "page_grounded_only" for row in self.rows))
        self.assertEqual(self.report["typed_exclusion_count"], sum(row["decision_kind"] == "typed_exclusion" for row in self.rows))
        self.assertTrue(all(value == 0 for value in self.report["counters"].values()), self.report["counters"])

    def test_public_targets_resolve_exact_page_only_and_exclusion_policies(self) -> None:
        service = LegalRuntimeService(ROOT)
        exact = next(row for row in self.rows if row["bbox_precision"] == "exact")
        page_only = next(row for row in self.rows if row["bbox_precision"] == "page_grounded_only")
        exclusion = next(row for row in self.rows if row["decision_kind"] == "typed_exclusion")
        exact_target = service.register_public_target("uud", {"support_unit_id": exact["support_unit_id"]})
        exact_result = handle_request("uud", "viewer", {"target": exact_target}, service=service)
        self.assertEqual(exact_result["status"], "viewer_payload_ready")
        self.assertTrue(exact_result["viewer_highlightable"])
        self.assertGreater(len(exact_result["bbox_rectangles"]), 0)
        self.assertLessEqual(len(exact_result["bbox_rectangles"]), len(exact["bbox_refs"]))
        page_target = service.register_public_target("uud", {"support_unit_id": page_only["support_unit_id"]})
        page_result = handle_request("uud", "viewer", {"target": page_target}, service=service)
        self.assertEqual(page_result["status"], "viewer_payload_ready")
        self.assertTrue(page_result["pdf_access_available"])
        self.assertFalse(page_result["viewer_highlightable"])
        self.assertFalse(page_result["bbox_rectangles"])
        excluded = service.viewer("uud", support_unit_id=exclusion["support_unit_id"])
        self.assertEqual((excluded["status"], excluded["reason"]), ("not_found", "invalid_support_target"))

    def test_mutations_fail_independent_owner_page_quote_geometry_and_authority_checks(self) -> None:
        mutations = (
            ("owner_id", "missing-owner"),
            ("page_numbers", [999]),
            ("quoted_text_sha256", "0" * 64),
            ("authority_kind", "normative_legal_text"),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                rows = deepcopy(self.rows)
                target = next(row for row in rows if row["decision_kind"] == "canonical_owner_support")
                target[field] = value
                self.assertEqual(evaluate_rows(rows, self.artifacts, self.oracle)["status"], "FAIL")
        rows = deepcopy(self.rows)
        target = next(row for row in rows if row["bbox_precision"] == "exact")
        target["bbox_refs"].append("foreign-character")
        self.assertEqual(evaluate_rows(rows, self.artifacts, self.oracle)["status"], "FAIL")

    def test_evaluator_has_no_builder_or_runtime_projection_oracle(self) -> None:
        source = (ROOT / "scripts/evaluate_support_reachability.py").read_text(encoding="utf-8")
        self.assertNotIn("meaningful_support_builder", source)
        self.assertNotIn("build_meaningful_support_units", source)
        self.assertNotIn("runtime_projection.json", source)


if __name__ == "__main__":
    unittest.main()
