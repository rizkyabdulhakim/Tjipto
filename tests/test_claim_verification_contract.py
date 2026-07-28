from __future__ import annotations

from pathlib import Path
import unittest

from tjipto.runtime.service import LegalRuntimeService


ROOT = Path(__file__).resolve().parents[1]


class ClaimVerificationContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.service = LegalRuntimeService(ROOT)

    def test_normative_tokens_are_not_substantive_support(self) -> None:
        for query in (
            "Pasal 28A mengatur pajak?",
            "Pasal 28A melarang hidup?",
            "Pasal 28A mewajibkan hidup?",
            "Pasal 28A memperbolehkan hidup?",
            "Pasal 27 mewajibkan pekerjaan?",
            "Pasal 28J mewajibkan agama?",
            "Pasal 20 memperbolehkan rancangan undangundang diajukan?",
            "Pasal 27 mewajibkan hukum menjunjung warga negara?",
        ):
            with self.subTest(query=query):
                result = self.service.ask("uud", query)
                self.assertEqual((result["status"], result["claim_support"][0]["status"]), ("insufficient_evidence", "insufficient"))
                self.assertFalse(result["citations"])

    def test_textual_support_is_one_atomic_grounded_segment(self) -> None:
        for query in (
            "Pasal 12 menyebut keadaan bahaya syaratsyarat dan?",
            "Pasal 1 menyebut Negara Kesatuan yang berbentuk Republik?",
            "Pasal 7B menyebut memeriksa mengadili?",
        ):
            with self.subTest(query=query):
                result = self.service.ask("uud", query)
                self.assertEqual(result["claim_support"][0]["status"], "insufficient")
                self.assertFalse(result["citations"])

    def test_internal_negation_retains_exact_source_segment(self) -> None:
        for query in (
            "Pasal 27 menyebut tidak ada kecualinya?",
            "Apakah Pasal 28I menyebut tidak dapat dikurangi?",
        ):
            with self.subTest(query=query):
                result = self.service.ask("uud", query)
                segment = result["claim_support"][0]["support_segments"][0]
                self.assertEqual((result["status"], result["claim_support"][0]["status"]), ("answer_ready", "supported"))
                self.assertTrue(segment["text_span_ids"])
                self.assertTrue(segment["bbox_refs"])
                self.assertTrue(segment["exact_quote"])


if __name__ == "__main__":
    unittest.main()
