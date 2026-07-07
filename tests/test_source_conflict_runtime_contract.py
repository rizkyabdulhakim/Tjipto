from __future__ import annotations

import json
from pathlib import Path
import unittest

from tjipto.runtime.service import LegalRuntimeService


ROOT = Path(__file__).resolve().parents[1]


class SourceConflictRuntimeContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = LegalRuntimeService(ROOT)

    def test_known_source_conflicts_expose_safe_provenance(self) -> None:
        for case in _source_conflict_cases():
            query = case["query"]
            result = self.service.ask("uud", query)
            self.assertEqual(result["status"], case["expected_status"], query)
            self.assertEqual(result["route"], "source_anomaly_explanation", query)
            self.assertEqual(result["source_conflict"]["source_conflict_id"], case["source_conflict_id"], query)
            self.assertEqual(result["source_conflict"]["type"], case["type"], query)
            self.assertEqual(result["source_conflict"]["classification"], case["classification"], query)
            self.assertEqual(result["source_conflict"]["source_document_id"], case["source_document_id"], query)
            for reason in case["expected_insufficient_reasons"]:
                self.assertIn(reason, result["insufficient_reasons"], query)
            for text in case["answer_contains"]:
                self.assertIn(text.casefold(), result["answer"].casefold(), query)
            for text in case.get("answer_not_contains", ()):
                self.assertNotIn(text.casefold(), result["answer"].casefold(), query)
            self.assertEqual(bool(result["citations"]), case["has_citations"], query)
            self.assertEqual(bool(result["viewer_refs"]), case["has_viewer_refs"], query)
            self.assertEqual(len(result.get("trace_support", ())), case["trace_support_count"], query)
            self.assertIn("source_conflict_not_final_legal_authority", result["warnings"], query)

    def test_trace_only_source_conflicts_do_not_create_fake_citations(self) -> None:
        result = self.service.ask("uud", "Apa konflik sumber Aturan Tambahan Pasal III Perubahan Keempat?")
        self.assertEqual(result["status"], "limited_answer")
        self.assertFalse(result["citations"])
        self.assertFalse(result["viewer_refs"])
        self.assertTrue(result["trace_support"])
        support = result["trace_support"][0]
        self.assertEqual(support["support_class"], "source_conflict_trace")
        self.assertFalse(support["citation_available"])
        self.assertFalse(support["viewer_highlightable"])
        self.assertIsNone(support["viewer_ref"])

    def test_exact_source_conflict_provenance_can_resolve_existing_viewer_policy(self) -> None:
        result = self.service.ask("uud", "Apa konflik sumber Pasal 25E dan Pasal 25A Perubahan Kedua?")
        self.assertEqual(result["status"], "limited_answer")
        self.assertTrue(result["evidence"])
        self.assertTrue(result["citations"])
        self.assertTrue(result["viewer_refs"])
        self.assertFalse(result["trace_support"])
        self.assertTrue(all(row["can_resolve"] for row in result["viewer_refs"]))
        self.assertNotIn("Reviewer decision: .", result["answer"])

    def test_vague_source_conflict_query_fails_closed(self) -> None:
        for case in _source_conflict_negative_cases():
            result = self.service.ask("uud", case["query"])
            self.assertEqual(result["status"], case["status"], case["query"])
            self.assertEqual(result["route"], case["route"], case["query"])
            for reason in case["insufficient_reasons"]:
                self.assertIn(reason, result["insufficient_reasons"], case["query"])
            self.assertEqual(result["source_conflict"], case["source_conflict"], case["query"])
            for field in case["expect_empty"]:
                self.assertFalse(result[field], case["query"])


def _source_conflict_cases() -> tuple[dict, ...]:
    path = ROOT / "tests/fixtures/uud/source_conflict_cases.jsonl"
    return tuple(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _source_conflict_negative_cases() -> tuple[dict, ...]:
    path = ROOT / "tests/fixtures/uud/source_conflict_negative_cases.jsonl"
    return tuple(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


if __name__ == "__main__":
    unittest.main()
