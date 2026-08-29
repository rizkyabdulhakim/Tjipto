from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
import unittest

from tjipto.corpora.capabilities import resolve_capability
from tjipto.corpora.verified import VerifiedCorpusRepository
from tjipto.evidence.store import EvidenceStore
from tjipto.runtime.intent import classify_relation_intent
from tjipto.runtime.query_semantics import interpret_query
from tjipto.runtime.service import LegalRuntimeService


ROOT = Path(__file__).resolve().parents[1]


class RuntimeIntentArbitrationContractTest(unittest.TestCase):
    service: LegalRuntimeService
    store: EvidenceStore

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

    def test_criminal_punishment_function_blocks_article_citation(self) -> None:
        for query in (
            "denda korupsi presiden",
            "apa pidana korupsi menurut Pasal 7A",
            "pidana korupsi presiden menurut Pasal 7A",
            "apa sanksi korupsi menurut pasal 7A?",
            "Pasal 7A mengatur sanksi apa",
            "Pasal 7A pidana korupsi",
            "Pasal 7A menjatuhkan hukuman apa",
            "ancaman pidana menurut Pasal 7A",
            "hukuman korupsi Presiden menurut Pasal 7A",
            "korupsi dalam pasal 7A hukuman apa",
        ):
            decision = resolve_capability(self.store.config, query, "direct_quotation", ("uud",))
            self.assertEqual(decision.requested_operation, "external_legal_domain_research", query)
            self.assertEqual(decision.missing_capabilities, ("criminal_punishment",), query)
            result = self.service.ask("uud", query)
            self.assertEqual(result["status"], "insufficient_evidence", query)
            self.assertEqual(result["route"], "unsupported_scope", query)
            self.assertFalse(result["citations"], query)
            self.assertFalse(result["viewer_refs"], query)
            citation = self.service.citation("uud", query)
            self.assertEqual(citation["status"], "citation_not_found", query)
            self.assertFalse(citation["citation_payloads"], query)

    def test_pasal_7a_removal_ground_context_is_not_blocked(self) -> None:
        for query in (
            "korupsi dalam Pasal 7A maksudnya apa?",
            "tindak pidana berat dalam Pasal 7A maksudnya apa?",
            "alasan Presiden dapat diberhentikan menurut Pasal 7A",
        ):
            self.assertIsNone(resolve_capability(self.store.config, query, "direct_quotation", ("uud",)).reason_code, query)
            result = self.service.ask("uud", query)
            self.assertEqual(result["status"], "answer_ready", query)
            self.assertEqual(result["route"], "legal_reference", query)

    def test_delete_relation_family_outranks_exact_article_reference(self) -> None:
        for query in (
            "perubahan keempat mencabut pasal 16?",
            "pasal 16 dicabut perubahan keempat?",
            "pencabutan pasal 16 oleh perubahan keempat",
            "penghapusan pasal 16",
        ):
            intent = classify_relation_intent(self.store, query)
            self.assertEqual(intent.requested_function, "amendment_relation", query)
            self.assertEqual(intent.relation_type, "DELETE_OR_REMOVE_PROVISION", query)
            result = self.service.ask("uud", query)
            self.assertEqual(result["route"], "document_relation", query)
            self.assertNotEqual(result["intent"], "exact_citation", query)

    def test_deterministic_routing_precedence_preserves_valid_recall(self) -> None:
        exact = self.service.ask("uud", "Pasal 33 ayat (3) tentang tanah")
        self.assertEqual((exact["status"], exact["route"]), ("answer_ready", "legal_reference"))
        self.assertTrue(exact["citations"])

        metadata = self.service.ask("uud", "Jakarta dan tanggal Perubahan Pertama")
        self.assertEqual((metadata["status"], metadata["route"]), ("answer_ready", "metadata_fact"))
        self.assertEqual(metadata["metadata_facts"][0]["answer"], "19 Oktober 1999")

        navigation = self.service.ask("uud", "pasal berikutnya setelah Pasal 7")
        self.assertEqual(
            (navigation["status"], navigation["route"], navigation["citations"][0]["citation"]),
            ("answer_ready", "structural_navigation", "Pasal 7A"),
        )

    def test_navigation_requires_an_explicit_adjacency_operation(self) -> None:
        for query in (
            "Pasal berikutnya setelah Pasal 7",
            "Pasal apa berikutnya setelah Pasal 7",
            "Setelah Pasal 7 pasal berapa?",
            "Sesudah Pasal 7 apa?",
            "apa ketentuan sebelum Pasal 28?",
        ):
            with self.subTest(query=query):
                self.assertEqual(self.service.ask("uud", query)["route"], "structural_navigation")
        previous = self.service.ask("uud", "apa ketentuan sebelum Pasal 28?")
        self.assertEqual(previous["citations"][0]["citation"], "Pasal 27")
        for query in ("Pasal 7 setelah Pasal 6", "Siapa Presiden setelah Pasal 7?"):
            with self.subTest(query=query):
                self.assertNotEqual(self.service.ask("uud", query)["route"], "structural_navigation")

    def test_corpus_routes_compact_and_article_scoped_queries(self) -> None:
        self.assertIn("BAB XA", self.service.ask("uud", "isi babxa uud")["answer"])
        previous = self.service.ask("uud", "bab sebelum BAB XA")
        self.assertEqual(previous["route"], "structural_navigation")
        # The source text orders BAB X immediately before BAB XA; BAB IXA is
        # a separate inserted chapter that precedes BAB X.
        self.assertIn("BAB X", previous["answer"])
        ayat_count = self.service.ask("uud", "berapa ayat Pasal 29")
        self.assertEqual(ayat_count["route"], "structure_count")
        self.assertIn("Pasal 29 dengan 2 ayat", ayat_count["answer"])
        natural_number = self.service.ask("uud", "pasal dua puluh delapan")
        self.assertEqual(natural_number["route"], "legal_reference")
        self.assertIn("Pasal 28", natural_number["answer"])

    def test_document_and_relation_phrasing_uses_existing_corpus_routes(self) -> None:
        collection = self.service.ask("uud", "berikan semua dokumen UUD")
        self.assertEqual(collection["route"], "source_document_collection")
        identity = self.service.ask("uud", "apa itu naskah konsolidasi")
        self.assertEqual(identity["route"], "document_relation")
        self.assertIn("naskah konsolidasi", identity["answer"])
        current = self.service.ask("uud", "naskah setelah semua amandemen")
        self.assertEqual(current["route"], "source_document")
        additions = self.service.ask("uud", "ketentuan apa yang ditambahkan Perubahan Kedua")
        self.assertEqual(additions["route"], "document_relation")
        self.assertIn("ditambahkan", additions["answer"])
        deletion = self.service.ask("uud", "ketentuan apa yang dihapus Perubahan Keempat")
        self.assertEqual(deletion["route"], "document_relation")
        self.assertIn("Pasal 16", deletion["answer"])
        renumbering = self.service.ask("uud", "Pasal 25E UUD")
        self.assertEqual(renumbering["route"], "document_relation")
        self.assertIn("Pasal 25A", renumbering["answer"])
        occurrence = self.service.ask("uud", "kebebasan berbicara ada di naskah apa saja")
        self.assertEqual(occurrence["route"], "lexical_fallback")
        self.assertEqual(
            {row["source_role"] for row in occurrence["evidence"]},
            {"original_historical", "amendment_2_historical", "current_consolidated"},
        )
        conflict = self.service.ask("uud", "Pasal III PDF sama dengan Pasal II naskah satu")
        self.assertEqual(conflict["route"], "source_anomaly_explanation")
        self.assertIn("typo penomoran tercetak", conflict["answer"])

    def test_interpreted_query_is_immutable(self) -> None:
        semantics = interpret_query(self.store, "uud", "Pasal berikutnya setelah Pasal 7")
        with self.assertRaises(FrozenInstanceError):
            semantics.requested_function = "other"  # type: ignore[misc]

    def test_current_fact_precedes_legal_reference_and_discovery_precedes_domain(self) -> None:
        for query in ("siapa Presiden setelah Pasal 7?", "siapa presiden menurut Pasal 7?"):
            with self.subTest(query=query):
                result = self.service.ask("uud", query)
                self.assertEqual((result["route"], result["reason_code"]), ("current_fact_unsupported", "current_fact_unsupported"))
                self.assertFalse(result["citations"])
        result = self.service.ask("uud", "Tanah disebut dalam pasal berapa?")
        self.assertNotEqual(result["route"], "capability_unavailable")


if __name__ == "__main__":
    unittest.main()
