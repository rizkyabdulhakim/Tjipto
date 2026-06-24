from __future__ import annotations

from pathlib import Path
import unittest

from tjipto.runtime.service import LegalRuntimeService


ROOT = Path(__file__).resolve().parents[1]


class SourceConflictRuntimeContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = LegalRuntimeService(ROOT)

    def test_known_source_conflicts_are_explained_without_promotion(self) -> None:
        cases = (
            "Apa konflik sumber Aturan Tambahan Pasal III Perubahan Keempat?",
            "Apa konflik sumber Pasal 25E dan Pasal 25A Perubahan Kedua?",
            "Apa status anomali sumber Perubahan Keempat UUD?",
        )
        for query in cases:
            result = self.service.ask("uud", query)
            self.assertEqual(result["status"], "insufficient_evidence", query)
            self.assertEqual(result["route"], "source_anomaly_explanation", query)
            self.assertIn("source_anomaly", result["insufficient_reasons"], query)
            self.assertFalse(result["citations"], query)
            self.assertFalse(result["viewer_refs"], query)


if __name__ == "__main__":
    unittest.main()
