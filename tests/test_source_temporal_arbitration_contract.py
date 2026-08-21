from __future__ import annotations

from pathlib import Path
import unittest

from tjipto.corpora.verified import VerifiedCorpusRepository
from tjipto.corpora.source_arbitration import source_roles_for_query
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

    def test_coordinated_role_alias_requires_an_explicit_instrument_anchor(self) -> None:
        config = self.store.config
        roles = source_roles_for_query("pilihan pertama dan pendidikan", strategy=config.query_strategy, config=config)
        self.assertEqual(roles, ())
        for query in ("amandemen pertama, kedua", "amandemen kedua/pertama"):
            with self.subTest(query=query):
                roles = source_roles_for_query(query, strategy=config.query_strategy, config=config)
                self.assertEqual(set(roles), {"amendment_1_historical", "amendment_2_historical"})

    def test_temporal_language_never_activates_navigation(self) -> None:
        for suffix in (
            "setelah perubahan",
            "sesudah perubahan",
            "pasca perubahan",
            "setelah diubah",
            "sesudah diubah",
            "setelah diamandemen",
            "sesudah diamandemen",
            "pasca amandemen",
            "saat ini",
            "naskah konsolidasi",
        ):
            with self.subTest(suffix=suffix):
                result = self.service.ask("uud", f"Apa isi Pasal 7 {suffix}?")
                self.assertEqual((result["route"], result["citations"][0]["citation"]), ("legal_reference", "Pasal 7"))
        self.assertEqual(self.service.ask("uud", "Pasal berikutnya setelah Pasal 7")["route"], "structural_navigation")
        self.assertNotEqual(interpret_query(self.store, "uud", "Pasal 7 setelah Pasal 6").requested_function, "structural_navigation")


if __name__ == "__main__":
    unittest.main()
