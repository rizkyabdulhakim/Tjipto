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

    def test_negated_proposition_is_contradicted_by_normative_text(self) -> None:
        result = self.service.ask("uud", "Pasal 7 tidak mengatur masa jabatan?")
        self.assertEqual((result["status"], result["reason_code"]), ("insufficient_evidence", "claim_support_contradicted"))
        self.assertEqual(result["claim_support"][0]["status"], "contradicted")
        self.assertFalse(result["citations"])

    def test_temporal_qualification_precedes_navigation(self) -> None:
        result = self.service.ask("uud", "Apa isi Pasal 7 setelah perubahan?")
        self.assertEqual((result["status"], result["route"]), ("answer_ready", "legal_reference"))
        self.assertEqual(result["citations"][0]["citation"], "Pasal 7")
        self.assertEqual(result["citations"][0]["source_role"], "current_consolidated")

    def test_scope_follows_retrieval_and_reports_missing_corpus(self) -> None:
        result = self.service.ask("uud", "Apa aturan tentang tanah di Jakarta?")
        self.assertEqual((result["status"], result["route"], result["reason_code"]), (
            "insufficient_evidence", "missing_corpus", "missing_corpus_support"
        ))
        self.assertTrue(result["retrieval_attempted"])
        self.assertEqual(result["available_corpora"], ("uud",))
        self.assertEqual(result["missing_corpora"], ("additional_legal_corpus",))

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
