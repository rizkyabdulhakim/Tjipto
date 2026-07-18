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
            self.assertEqual(bool(result["citations"]), case["has_citations"], query)
            self.assertEqual(bool(result["viewer_refs"]), case["has_viewer_refs"], query)
            self.assertEqual(len(result.get("trace_support", ())), case["trace_support_count"], query)
            self.assertIn("source_conflict_not_final_legal_authority", result["warnings"], query)

    def test_source_anomaly_reuses_existing_exact_span_bbox_without_becoming_final(self) -> None:
        result = self.service.ask("uud", "Apa konflik sumber Aturan Tambahan Pasal III Perubahan Keempat?")
        self.assertEqual(result["status"], "limited_answer")
        self.assertTrue(result["citations"])
        self.assertTrue(result["viewer_refs"])
        self.assertFalse(result["trace_support"])
        citation = result["citations"][0]
        self.assertEqual(citation["authority_kind"], "source_anomaly")
        self.assertEqual(citation["support_kind"], "source_anomaly_provenance")
        self.assertEqual(citation["finality_policy"], "source_anomaly_provenance")
        self.assertEqual(citation["source_anomaly_kind"], "source_marker_sequence_anomaly")
        self.assertEqual(citation["provenance_highlight_scope"], "all_relevant_spans")
        self.assertFalse(citation["citation_final"])
        self.assertEqual(citation["evidence_id"], "uud_1945_amendment_4_aturan_tambahan_pasal_ii_iii_conflict")
        self.assertEqual(result["source_conflict"]["provenance_bbox_status"], "exact_raw_provenance_bbox_available")
        self.assertEqual(result["source_conflict"]["provenance_highlight_scope"], "all_relevant_spans")
        self.assertEqual(result["source_conflict"]["source_anomaly_kind"], "source_marker_sequence_anomaly")
        self.assertEqual(result["source_conflict"]["blocked_raw_provenance_text_span_count"], 0)
        conflict = next(
            row
            for row in read_jsonl(FINAL / "source_conflicts.jsonl")
            if row["source_conflict_id"] == "uud_1945_amendment_4_aturan_tambahan_pasal_ii_iii_conflict"
        )
        self.assertEqual(len(conflict["text_span_ids"]), 3)
        self.assertEqual(len(conflict["raw_provenance_text_span_ids"]), 3)
        self.assertEqual(conflict["blocked_raw_provenance_text_span_reasons"], {})
        viewer = self.service.viewer("uud", citation["evidence_id"])
        self.assertEqual(viewer["authority_kind"], "source_anomaly")
        self.assertFalse(viewer["citation_final"])
        self.assertEqual(viewer["status"], "viewer_payload_ready")
        self.assertTrue(viewer["viewer_highlightable"])
        self.assertEqual(viewer["bbox_count"], len(conflict["raw_provenance_bbox_ids"]))
        self.assertEqual(set(row["bbox_id"] for row in viewer["bbox_rectangles"]), set(conflict["raw_provenance_bbox_ids"]))
        self.assertEqual(result["viewer_refs"][0]["bbox_count"], len(conflict["raw_provenance_bbox_ids"]))
        self.assertEqual(result["viewer_refs"][0]["authority_kind"], "source_anomaly")
        self.assertEqual(result["viewer_refs"][0]["support_kind"], "source_anomaly_provenance")
        self.assertEqual(result["viewer_refs"][0]["source_anomaly_kind"], "source_marker_sequence_anomaly")
        self.assertEqual(result["viewer_refs"][0]["provenance_highlight_scope"], "all_relevant_spans")

    def test_exact_source_conflict_provenance_can_resolve_existing_viewer_policy(self) -> None:
        result = self.service.ask("uud", "Apa konflik sumber Pasal 25E dan Pasal 25A Perubahan Kedua?")
        self.assertEqual(result["status"], "limited_answer")
        self.assertTrue(result["evidence"])
        self.assertTrue(result["citations"])
        self.assertTrue(result["viewer_refs"])
        self.assertFalse(result["trace_support"])
        self.assertEqual(result["source_conflict"]["provenance_bbox_status"], "exact_raw_provenance_bbox_available")
        self.assertEqual(result["source_conflict"]["provenance_highlight_scope"], "all_relevant_spans")
        self.assertEqual(result["source_conflict"]["source_anomaly_kind"], "renumbering_provenance")
        self.assertEqual(result["source_conflict"]["source_mapping_kind"], "historical_to_canonical_mapping")
        self.assertEqual(result["citations"][0]["authority_kind"], "source_conflict_provenance")
        self.assertEqual(result["citations"][0]["support_kind"], "source_anomaly_provenance")
        self.assertEqual(result["citations"][0]["finality_policy"], "source_anomaly_provenance")
        self.assertEqual(result["citations"][0]["source_anomaly_kind"], "renumbering_provenance")
        self.assertEqual(result["citations"][0]["source_mapping_kind"], "historical_to_canonical_mapping")
        self.assertFalse(result["citations"][0]["citation_final"])
        self.assertTrue(all(row["can_resolve"] for row in result["viewer_refs"]))
        self.assertEqual(result["viewer_refs"][0]["source_anomaly_kind"], "renumbering_provenance")
        self.assertEqual(result["viewer_refs"][0]["source_mapping_kind"], "historical_to_canonical_mapping")
        self.assertEqual(result["viewer_refs"][0]["provenance_highlight_scope"], "all_relevant_spans")
        viewer = self.service.viewer("uud", result["citations"][0]["evidence_id"])
        self.assertEqual(viewer["authority_kind"], "source_conflict_provenance")
        self.assertFalse(viewer["citation_final"])
        self.assertNotIn("Reviewer decision: .", result["answer"])
        self.assertNotIn("masih berlaku", result["answer"])
        self.assertNotIn("konflik norma", result["answer"].casefold())
        self.assertIn("renumbering provenance", result["answer"].casefold())
        self.assertIn("historical_to_canonical_mapping", result["answer"])

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
        self.assertEqual(pasal_i["citations"][0]["citation"], "Pasal I")
        self.assertEqual(pasal_i["citations"][0]["authority_kind"], "source_anomaly")
        self.assertFalse(pasal_i["citations"][0]["citation_final"])
        self.assertTrue(pasal_i["viewer_refs"][0]["can_resolve"])

        pasal_ii = self.service.ask("uud", "Pasal II Perubahan Keempat")
        pasal_iii = self.service.ask("uud", "Pasal III Perubahan Keempat")
        self.assertEqual(pasal_ii["source_conflict"]["source_conflict_id"], pasal_iii["source_conflict"]["source_conflict_id"])
        self.assertEqual(pasal_ii["route"], "source_anomaly_explanation")
        self.assertIn("Pasal III", pasal_ii["answer"])
        self.assertNotIn("PDF mencetak Pasal II.", pasal_ii["answer"])

    def test_inserted_bab_queries_use_structural_heading_owners(self) -> None:
        for label in ("BAB IXA", "BAB XA", "BAB VIIA", "BAB VIIB", "BAB VIIIA"):
            result = self.service.ask("uud", label)
            self.assertEqual(result["status"], "answer_ready", label)
            citation = result["citations"][0]
            self.assertEqual(citation["citation"], label, label)
            self.assertEqual(citation["authority_kind"], "structural_context", label)
            self.assertFalse(citation["citation_final"], label)
            self.assertTrue(result["viewer_refs"][0]["can_resolve"], label)


def _source_conflict_cases() -> tuple[dict, ...]:
    path = ROOT / "tests/fixtures/uud/source_conflict_cases.jsonl"
    return tuple(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _source_conflict_negative_cases() -> tuple[dict, ...]:
    path = ROOT / "tests/fixtures/uud/source_conflict_negative_cases.jsonl"
    return tuple(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


if __name__ == "__main__":
    unittest.main()
