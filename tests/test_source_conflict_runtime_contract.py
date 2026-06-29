from __future__ import annotations

import json
from pathlib import Path
import unittest

from tjipto.runtime.service import LegalRuntimeService


ROOT = Path(__file__).resolve().parents[1]


class SourceConflictRuntimeContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = LegalRuntimeService(ROOT)

    def test_known_source_conflicts_are_explained_without_promotion(self) -> None:
        for case in _source_conflict_cases():
            query = case["query"]
            result = self.service.ask("uud", query)
            self.assertEqual(result["status"], "insufficient_evidence", query)
            self.assertEqual(result["route"], "source_anomaly_explanation", query)
            for reason in case["expected_insufficient_reasons"]:
                self.assertIn(reason, result["insufficient_reasons"], query)
            self.assertEqual(result["source_conflict"]["source_conflict_id"], case["source_conflict_id"], query)
            self.assertEqual(result["source_conflict"]["type"], case["type"], query)
            self.assertEqual(result["source_conflict"]["classification"], case["classification"], query)
            self.assertEqual(result["source_conflict"]["source_document_id"], case["source_document_id"], query)
            for text in case["answer_contains"]:
                self.assertIn(text.casefold(), result["answer"].casefold(), query)
            self.assertFalse(result["citations"], query)
            self.assertFalse(result["viewer_refs"], query)

    def test_vague_source_conflict_query_fails_closed(self) -> None:
        result = self.service.ask("uud", "Apa konflik sumber UUD?")
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(result["route"], "source_anomaly_explanation")
        self.assertIn("source_anomaly_unresolved", result["insufficient_reasons"])
        self.assertIsNone(result["source_conflict"])
        self.assertFalse(result["citations"])
        self.assertFalse(result["viewer_refs"])


def _source_conflict_cases() -> tuple[dict, ...]:
    path = ROOT / "tests/fixtures/uud/source_conflict_cases.jsonl"
    return tuple(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


if __name__ == "__main__":
    unittest.main()
