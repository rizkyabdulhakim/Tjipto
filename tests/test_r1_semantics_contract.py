from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
import unittest

from tjipto.runtime.query_semantics import interpret_query
from tjipto.runtime.service import LegalRuntimeService


ROOT = Path(__file__).resolve().parents[1]


class R1SemanticsContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.service = LegalRuntimeService(ROOT)
        cls.store = cls.service._store("uud")
        assert cls.store is not None

    def test_proposition_requires_substantive_support(self) -> None:
        result = self.service.ask("uud", "Pasal 28A mengatur pajak?")
        self.assertEqual((result["status"], result["reason_code"]), ("insufficient_evidence", "claim_support_insufficient"))
        self.assertFalse(result["citations"])
        self.assertEqual(result["claim_support"][0]["status"], "insufficient")

    def test_negated_proposition_without_matching_predicate_is_insufficient(self) -> None:
        result = self.service.ask("uud", "Pasal 7 tidak mengatur masa jabatan?")
        self.assertEqual((result["status"], result["reason_code"]), ("insufficient_evidence", "claim_support_insufficient"))
        self.assertEqual(result["claim_support"][0]["status"], "insufficient")
        self.assertFalse(result["citations"])

    def test_shared_object_does_not_support_a_different_normative_predicate(self) -> None:
        cases = (
            ("Pasal 28A melarang hidup?", "prohibits", "prohibition"),
            ("Pasal 28A mewajibkan hidup?", "requires", "obligation"),
            ("Pasal 28A memperbolehkan hidup?", "permits", "permission"),
            ("Pasal 7 melarang presiden?", "prohibits", "prohibition"),
        )
        for query, predicate, modality in cases:
            with self.subTest(query=query):
                result = self.service.ask("uud", query)
                claim = result["claim_support"][0]
                self.assertEqual((result["status"], claim["status"]), ("insufficient_evidence", "insufficient"))
                self.assertEqual((claim["predicate"], claim["polarity"], claim["modality"]), (predicate, "positive", modality))
                self.assertFalse(result["citations"])

    def test_temporal_qualification_precedes_navigation(self) -> None:
        for suffix in (
            "setelah perubahan", "sesudah perubahan", "pasca perubahan", "setelah diubah",
            "sesudah diubah", "setelah diamandemen", "sesudah diamandemen", "pasca amandemen", "saat ini", "naskah konsolidasi",
        ):
            with self.subTest(suffix=suffix):
                result = self.service.ask("uud", f"Apa isi Pasal 7 {suffix}?")
                self.assertEqual((result["status"], result["route"]), ("answer_ready", "legal_reference"))
                self.assertEqual(result["citations"][0]["citation"], "Pasal 7")
                self.assertEqual(result["citations"][0]["source_role"], "current_consolidated")
        navigation = self.service.ask("uud", "Pasal berikutnya setelah Pasal 7")
        self.assertEqual((navigation["status"], navigation["route"]), ("answer_ready", "structural_navigation"))
        self.assertEqual(navigation["citations"][0]["citation"], "Pasal 7A")

    def test_scope_follows_retrieval_and_reports_missing_corpus(self) -> None:
        result = self.service.ask("uud", "Apa aturan tentang tanah di Jakarta?")
        self.assertEqual((result["status"], result["route"], result["reason_code"]), (
            "insufficient_evidence", "missing_corpus", "missing_corpus_support"
        ))
        self.assertTrue(result["retrieval_attempted"])
        self.assertEqual(result["available_corpora"], ("uud",))
        self.assertEqual(result["missing_corpora"], ("land_law",))
        self.assertEqual(result["required_capabilities"], ("land_regulation",))
        self.assertGreater(result["retrieval_candidate_count"], 0)

    def test_partial_land_retrieval_is_evaluated_before_missing_corpus(self) -> None:
        result = self.service.ask("uud", "Apa aturan tentang bumi dan tanah?")
        self.assertEqual((result["status"], result["route"]), ("insufficient_evidence", "missing_corpus"))
        self.assertEqual(result["missing_corpora"], ("land_law",))
        self.assertEqual(result["retrieval_route"], "bm25")

    def test_source_discrepancy_uses_trace_not_substantive_authority(self) -> None:
        result = self.service.ask(
            "uud", "Kenapa Amandemen 4 Aturan Tambahan ada Pasal III, tapi Satu Naskah Pasal II?"
        )
        self.assertEqual((result["status"], result["route"]), ("limited_answer", "source_anomaly_explanation"))
        self.assertFalse(result["citations"])
        self.assertEqual(
            result["source_conflict"]["source_conflict_id"],
            "uud_1945_amendment_4_aturan_tambahan_pasal_ii_iii_conflict",
        )

    def test_semantics_are_immutable_and_unknown_corpus_cannot_fallback(self) -> None:
        semantics = interpret_query(self.store, "uud", "Pasal berikutnya setelah Pasal 7")
        with self.assertRaises(FrozenInstanceError):
            semantics.requested_function = "other"  # type: ignore[misc]
        result = self.service.ask("unknown", "Pasal 7")
        self.assertEqual(result["status"], "unsupported_corpus")

    def test_relation_intent_is_not_collapsed_into_article_lookup(self) -> None:
        semantics = interpret_query(self.store, "uud", "perubahan keempat mencabut pasal 16?")
        self.assertEqual((semantics.requested_function, semantics.relation_intent), (
            "amendment_relation", "DELETE_OR_REMOVE_PROVISION"
        ))


if __name__ == "__main__":
    unittest.main()
