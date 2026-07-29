from __future__ import annotations

from pathlib import Path
import unittest

from tjipto.corpora.verified import VerifiedCorpusRepository
from tjipto.evidence.store import EvidenceStore
from tjipto.runtime.query_semantics import interpret_query
from tjipto.runtime.service import LegalRuntimeService


ROOT = Path(__file__).resolve().parents[1]


class SourceTemporalArbitrationContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.service = LegalRuntimeService(ROOT)
        cls.store = cls.service._store("uud")
        assert cls.store is not None

    @classmethod
    def tearDownClass(cls) -> None:
        cls.store = None
        cls.service = None
        EvidenceStore.clear_shared_cache()
        VerifiedCorpusRepository.clear_shared_cache()

    def test_named_source_precedes_generic_post_amendment_wording(self) -> None:
        result = self.service.ask("uud", "Apa isi Pasal 7 setelah Perubahan Pertama?")
        self.assertEqual(result["citations"][0]["source_role"], "amendment_1_historical")

    def test_temporal_language_never_activates_navigation(self) -> None:
        for suffix in ("setelah perubahan", "sesudah perubahan", "pasca amandemen", "naskah konsolidasi"):
            with self.subTest(suffix=suffix):
                result = self.service.ask("uud", f"Apa isi Pasal 7 {suffix}?")
                self.assertEqual((result["route"], result["citations"][0]["citation"]), ("legal_reference", "Pasal 7"))
        self.assertEqual(self.service.ask("uud", "Pasal berikutnya setelah Pasal 7")["route"], "structural_navigation")
        self.assertNotEqual(interpret_query(self.store, "uud", "Pasal 7 setelah Pasal 6").requested_function, "structural_navigation")


if __name__ == "__main__":
    unittest.main()
