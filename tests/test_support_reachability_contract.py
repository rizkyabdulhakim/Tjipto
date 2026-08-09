from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "data" / "final" / "uud"


def _rows(name: str) -> list[dict]:
    return [json.loads(line) for line in (FINAL / name).read_text(encoding="utf-8").splitlines() if line]


class SupportReachabilityContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = _rows("meaningful_support_units.jsonl")
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "report.json"
            subprocess.run(
                [sys.executable, "scripts/evaluate_support_reachability.py", "--report", str(report)],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            cls.report = json.loads(report.read_text(encoding="utf-8"))

    def test_every_decision_has_a_deterministic_terminal_route(self) -> None:
        self.assertEqual(self.report["status"], "PASS", self.report["counters"])
        self.assertEqual(self.report["decision_count"], len(self.rows))
        self.assertEqual(self.report["resolved_viewer_count"], sum(row["viewer_eligible"] for row in self.rows))
        self.assertEqual(self.report["exact_highlight_count"], sum(row["bbox_precision"] == "exact" for row in self.rows))
        self.assertEqual(self.report["page_grounded_count"], sum(row["bbox_precision"] == "page_grounded_only" for row in self.rows))
        self.assertEqual(self.report["typed_exclusion_count"], sum(row["decision_kind"] == "typed_exclusion" for row in self.rows))
        self.assertTrue(all(value == 0 for value in self.report["counters"].values()), self.report["counters"])

    def test_public_targets_resolve_exact_page_only_and_exclusion_policies(self) -> None:
        counters = self.report["counters"]
        self.assertEqual(counters["unresolved_public_target_count"], 0)
        self.assertEqual(counters["exact_highlight_capability_mismatch_count"], 0)
        self.assertEqual(counters["page_grounded_overlay_leakage_count"], 0)
        self.assertEqual(counters["page_grounded_highlight_escalation_count"], 0)
        self.assertEqual(counters["typed_exclusion_public_leakage_count"], 0)

    def test_mutations_fail_independent_owner_page_quote_geometry_and_authority_checks(self) -> None:
        self.assertEqual(self.report["counters"]["mutation_escape_count"], 0)

    def test_evaluator_has_no_builder_or_runtime_projection_oracle(self) -> None:
        source = (ROOT / "scripts/evaluate_support_reachability.py").read_text(encoding="utf-8")
        self.assertNotIn("meaningful_support_builder", source)
        self.assertNotIn("build_meaningful_support_units", source)
        self.assertNotIn("runtime_projection.json", source)


if __name__ == "__main__":
    unittest.main()
