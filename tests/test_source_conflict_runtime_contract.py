from __future__ import annotations

import json
from pathlib import Path
import unittest

from tjipto.core.manifest import read_jsonl
from tjipto.runtime.service import LegalRuntimeService


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "data/final/uud"


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
            self.assertEqual(result["source_conflict"]["source_anomaly_kind"], case["source_anomaly_kind"], query)
            if case.get("source_mapping_kind"):
                self.assertEqual(result["source_conflict"]["source_mapping_kind"], case["source_mapping_kind"], query)
            self.assertIn("provenance_bbox_status", result["source_conflict"], query)
            self.assertIn("provenance_highlight_scope", result["source_conflict"], query)
            for reason in case["expected_insufficient_reasons"]:
                self.assertIn(reason, result["insufficient_reasons"], query)
            for text in case["answer_contains"]:
                self.assertIn(text.casefold(), result["answer"].casefold(), query)
            for text in case.get("answer_not_contains", ()):
                self.assertNotIn(text.casefold(), result["answer"].casefold(), query)
            self.assertNotIn("catatan konflik sumber", result["answer"].casefold(), query)
            self.assertFalse(result["citations"], query)
            self.assertFalse(result["viewer_refs"], query)
            self.assertGreaterEqual(len(result.get("trace_support", ())), case["trace_support_count"], query)
            self.assertIn("source_conflict_not_final_legal_authority", result["warnings"], query)

    def test_source_anomaly_reuses_existing_exact_span_bbox_without_becoming_final(self) -> None:
        result = self.service.ask("uud", "Apa konflik sumber Aturan Tambahan Pasal III Perubahan Keempat?")
        self.assertEqual(result["status"], "limited_answer")
        self.assertFalse(result["citations"])
        self.assertFalse(result["viewer_refs"])
        self.assertTrue(result["trace_support"])
        citation = result["trace_support"][0]
        self.assertEqual(citation["authority_kind"], "source_anomaly")
        self.assertEqual(citation["support_kind"], "source_anomaly_provenance")
        self.assertEqual(citation["finality_policy"], "source_anomaly_provenance")
        self.assertEqual(citation["source_anomaly_kind"], "typed_source_discrepancy")
        self.assertEqual(citation["provenance_highlight_scope"], "all_relevant_spans")
        self.assertFalse(citation["citation_final"])
        self.assertEqual(citation["evidence_id"], "uud_1945_amendment_4_aturan_tambahan_pasal_ii_iii_conflict")
        self.assertEqual(result["source_conflict"]["provenance_bbox_status"], "exact_raw_provenance_bbox_available")

        self.assertEqual(result["source_conflict"]["provenance_highlight_scope"], "all_relevant_spans")
        self.assertEqual(result["source_conflict"]["source_anomaly_kind"], "typed_source_discrepancy")
        self.assertEqual(result["source_conflict"]["blocked_raw_provenance_text_span_count"], 0)
        conflict = next(
            row
            for row in read_jsonl(FINAL / "source_conflicts.jsonl")
            if row["source_conflict_id"] == "uud_1945_amendment_4_aturan_tambahan_pasal_ii_iii_conflict"
        )
        self.assertEqual(len(conflict["text_span_ids"]), 1)
        self.assertEqual(len(conflict["raw_provenance_text_span_ids"]), 1)
        self.assertEqual(conflict["blocked_raw_provenance_text_span_reasons"], {})
        viewer = self.service.viewer("uud", citation["evidence_id"])
        self.assertEqual(viewer["authority_kind"], "source_anomaly")
        self.assertFalse(viewer["citation_final"])
        self.assertEqual(viewer["status"], "viewer_payload_ready")
        self.assertTrue(viewer["viewer_highlightable"])
        self.assertEqual(viewer["bbox_count"], len(conflict["raw_provenance_bbox_ids"]))
        self.assertEqual(set(row["bbox_id"] for row in viewer["bbox_rectangles"]), set(conflict["raw_provenance_bbox_ids"]))
        self.assertEqual(result["trace_support"][0]["bbox_count"], len(conflict["raw_provenance_bbox_ids"]))
        self.assertEqual(result["trace_support"][0]["authority_kind"], "source_anomaly")
        self.assertEqual(result["trace_support"][0]["support_kind"], "source_anomaly_provenance")
        self.assertEqual(result["trace_support"][0]["source_anomaly_kind"], "typed_source_discrepancy")
        self.assertEqual(result["trace_support"][0]["provenance_highlight_scope"], "all_relevant_spans")

    def test_canonical_pasal_ii_does_not_use_printed_label_provenance(self) -> None:
        result = self.service.ask("uud", "Aturan Tambahan Pasal II")
        self.assertEqual(result["status"], "answer_ready")
        self.assertEqual(result["route"], "legal_reference")
        self.assertEqual(result["citations"][0]["source_role"], "current_consolidated")
        self.assertFalse(result.get("source_conflict"))

    def test_renumbering_provenance_is_not_a_substantive_conflict(self) -> None:
        result = self.service.ask("uud", "Pasal 25E menjadi Pasal 25A")
        relation = next(row for row in read_jsonl(FINAL / "article_amendment_relations.jsonl") if row.get("relation_type") == "RENUMBERED_TO")
        self.assertEqual(relation["relation_type"], "RENUMBERED_TO")
        self.assertFalse(relation["substantive_change"])
        self.assertFalse(relation["anomaly"])
        self.assertFalse(relation["source_conflict"])
        self.assertFalse(result.get("source_conflict"))

    def test_exact_source_conflict_provenance_can_resolve_existing_viewer_policy(self) -> None:
        result = self.service.ask("uud", "Pasal 25E menjadi Pasal 25A")
        self.assertEqual(result["status"], "answer_ready")
        self.assertTrue(result["historical_citations"])
        self.assertFalse(result.get("source_conflict"))
        self.assertIn("dinomori ulang", result["answer"].casefold())
        relation = next(row for row in result["article_amendment_relations"] if row.get("relation_type") == "RENUMBERED_TO")
        viewer = self.service.viewer("uud", relation["evidence_id"])
        self.assertEqual(viewer["status"], "viewer_payload_ready")
        self.assertTrue(viewer["viewer_highlightable"])

    def test_runtime_source_anomaly_presenter_does_not_hardcode_uud_specific_phrases(self) -> None:
        source = (ROOT / "src/tjipto/runtime/service.py").read_text(encoding="utf-8")
        for text in (
            "Pasal 25E",
            "Pasal 25A",
            "renumbering historis dari",
            "historical-to-canonical mapping untuk jejak audit",
        ):
            self.assertNotIn(text, source)

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

    def test_pasali_and_pasal_ii_source_routing_preserves_historical_provenance(self) -> None:
        pasal_i = self.service.ask("uud", "Aturan Tambahan Pasal I Perubahan Keempat")
        self.assertEqual(pasal_i["status"], "answer_ready")
        self.assertTrue(pasal_i["citations"])
        self.assertFalse(pasal_i["source_conflict"] if "source_conflict" in pasal_i else False)
        self.assertEqual(pasal_i["citations"][0]["citation"], "Pasal I")
        self.assertTrue(pasal_i["citations"][0]["citation_final"])

        pasal_ii = self.service.ask("uud", "Pasal II Perubahan Keempat")
        pasal_iii = self.service.ask("uud", "Pasal III Aturan Tambahan Perubahan Keempat")
        self.assertEqual(pasal_ii["route"], "legal_reference")
        self.assertEqual(pasal_iii["route"], "legal_reference")
        self.assertTrue(pasal_ii["citations"])
        self.assertTrue(pasal_iii["citations"])
        self.assertEqual(pasal_ii["citations"][0]["citation"], "Pasal II")
        self.assertEqual(pasal_iii["citations"][0]["citation"], "Pasal II")
        self.assertFalse(pasal_ii.get("source_conflict"))
        self.assertFalse(pasal_iii.get("source_conflict"))
        peralihan = self.service.ask("uud", "Pasal III Aturan Peralihan Perubahan Keempat")
        self.assertEqual(peralihan["route"], "legal_reference")
        self.assertIsNone(peralihan.get("source_conflict"))
        self.assertEqual(peralihan["citations"][0]["citation"], "Pasal III")

    def test_inserted_bab_heading_queries_publish_the_heading_as_the_answer(self) -> None:
        for label in ("BAB IXA", "BAB XA", "BAB VIIA", "BAB VIIB", "BAB VIIIA"):
            result = self.service.ask("uud", label)
            self.assertEqual(result["status"], "answer_ready", label)
            citation = result["citations"][0]
            self.assertEqual(citation["citation"], label, label)
            self.assertEqual(citation["authority_kind"], "legal_citation", label)
            self.assertTrue(citation["citation_final"], label)
            self.assertTrue(citation["viewer_ref"]["can_resolve"], label)


def _source_conflict_cases() -> tuple[dict, ...]:
    path = ROOT / "tests/fixtures/uud/source_conflict_cases.jsonl"
    return tuple(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _source_conflict_negative_cases() -> tuple[dict, ...]:
    path = ROOT / "tests/fixtures/uud/source_conflict_negative_cases.jsonl"
    return tuple(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


if __name__ == "__main__":
    unittest.main()
