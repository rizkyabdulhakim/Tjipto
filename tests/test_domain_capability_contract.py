from __future__ import annotations

from pathlib import Path
import unittest

from tjipto.runtime.service import LegalRuntimeService


ROOT = Path(__file__).resolve().parents[1]


class DomainCapabilityContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.service = LegalRuntimeService(ROOT)

    def test_content_and_location_tokens_do_not_invent_a_domain_or_corpus(self) -> None:
        for query in (
            "Tanah disebut dalam pasal berapa?",
            "Apa aturan tentang bumi dan tanah?",
            "Apa aturan tentang tanah di Jakarta?",
            "Berapa harga makanan di Jakarta?",
            "Apa aturan pendidikan di Jakarta?",
            "Apa aturan pemilu di Jakarta?",
        ):
            with self.subTest(query=query):
                result = self.service.ask("uud", query)
                self.assertNotEqual(result["route"], "missing_corpus")
                self.assertNotEqual(result.get("legal_domain"), "land")


if __name__ == "__main__":
    unittest.main()
