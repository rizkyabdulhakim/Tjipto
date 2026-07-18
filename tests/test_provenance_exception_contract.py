from __future__ import annotations

import json
from pathlib import Path
import unittest

from tjipto.corpora.uud.provenance_exceptions import (
    ACCEPTED_FALSE_POSITIVE_SEGMENTATION_PUNCTUATION,
    ACCEPTED_NONCANONICAL_SOURCE_CONFLICT_TRACE_ONLY,
)


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "data/final/uud"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class ProvenanceExceptionContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.units = {row["legal_unit_id"]: row for row in read_jsonl(FINAL / "legal_units.jsonl")}
        self.chunks = {row["chunk_id"]: row for row in read_jsonl(FINAL / "chunks.jsonl")}
        self.report = read_json(FINAL / "validation_report.json")

    def test_needs_review_records_are_classified(self) -> None:
        for row_id in ("00623", "00634", "00645", "00647", "00648"):
            unit = self.units[f"uud_legal_unit_{row_id}"]
            chunk = self.chunks[f"uud_chunk_{row_id}"]
            self.assertIn("provenance_exception_category", unit)
            self.assertIn("provenance_exception_category", chunk)

    def test_no_runtime_loadable_needs_review(self) -> None:
        health = self.report["provenance_exception_health"]
        self.assertEqual(health["runtime_loadable_needs_review_count"], 0)
        self.assertEqual(health["unresolved_needs_review_count"], 0)

    def test_decision_clause_segmentation_false_positive_is_tracked(self) -> None:
        for row_id in ("00623", "00634", "00648"):
            self.assertEqual(
                self.units[f"uud_legal_unit_{row_id}"]["provenance_exception_category"],
                ACCEPTED_FALSE_POSITIVE_SEGMENTATION_PUNCTUATION,
            )
            self.assertFalse(self.units[f"uud_legal_unit_{row_id}"]["runtime_loadable"])

    def test_noncanonical_source_conflict_trace_is_not_canonical(self) -> None:
        for row_id in ("00645", "00646", "00647"):
            unit = self.units[f"uud_legal_unit_{row_id}"]
            chunk = self.chunks[f"uud_chunk_{row_id}"]
            self.assertFalse(unit["canonical_use_allowed"])
            self.assertFalse(chunk["canonical_use_allowed"])
        self.assertFalse(self.units["uud_legal_unit_00645"]["runtime_loadable"])
        for row_id in ("00646", "00647"):
            unit = self.units[f"uud_legal_unit_{row_id}"]
            chunk = self.chunks[f"uud_chunk_{row_id}"]
            self.assertTrue(unit["runtime_loadable"])
            self.assertTrue(chunk["runtime_loadable"])
            self.assertTrue(unit["bbox_ids"])
            self.assertTrue(unit["text_span_ids"])
            self.assertTrue(chunk["evidence_ids"])
            self.assertTrue(chunk["text_span_ids"])
        self.assertEqual(
            self.units["uud_legal_unit_00645"]["provenance_exception_category"],
            ACCEPTED_NONCANONICAL_SOURCE_CONFLICT_TRACE_ONLY,
        )

    def test_00646_does_not_have_misleading_aturan_tambahan_label(self) -> None:
        text = self.units["uud_legal_unit_00646"]["text"]
        unit = self.units["uud_legal_unit_00646"]
        evidence = next(row for row in read_jsonl(FINAL / "evidence_registry.jsonl") if row["legal_unit_id"] == unit["legal_unit_id"])
        self.assertNotIn("provenance_exception_category", unit)
        self.assertTrue(unit["runtime_loadable"])
        self.assertEqual(evidence["authority_kind"], "normative_legal_text")
        self.assertFalse(evidence["citation_final"])
        self.assertTrue(evidence["viewer_highlightable"])
        self.assertIn("Majelis Permusyawaratan Rakyat", text)
        self.assertNotIn("Segala peraturan perundangundangan", text)

    def test_00647_does_not_duplicate_heading(self) -> None:
        text = self.units["uud_legal_unit_00647"]["text"]
        self.assertEqual(
            self.units["uud_legal_unit_00647"]["provenance_exception_category"],
            ACCEPTED_NONCANONICAL_SOURCE_CONFLICT_TRACE_ONLY,
        )
        self.assertEqual([line.strip() for line in text.splitlines()].count("Pasal III"), 1)

    def test_previously_excluded_records_are_admitted_from_exact_grounding(self) -> None:
        evidence = read_jsonl(FINAL / "evidence_registry.jsonl")
        by_unit = {row["legal_unit_id"]: row for row in evidence}
        for row_id in ("00383", "00400", "00475", "00479", "00521", "00566"):
            unit = self.units[f"uud_legal_unit_{row_id}"]
            admitted = by_unit[unit["legal_unit_id"]]
            self.assertTrue(unit["runtime_loadable"])
            self.assertEqual(admitted["source_role"], unit["source_role"])
            self.assertEqual(admitted["bbox_precision"], "exact")
            self.assertTrue(admitted["viewer_highlightable"])

    def test_validation_report_has_provenance_exception_health(self) -> None:
        health = self.report["provenance_exception_health"]
        self.assertEqual(health["accepted_false_positive_segmentation_punctuation_count"], 12)
        self.assertGreaterEqual(health["accepted_noncanonical_source_conflict_trace_only_count"], 3)
        self.assertLessEqual(health["builder_slicing_label_issue_confirmed_count"], 2)
        self.assertLessEqual(health["duplicated_heading_artifact_issue_confirmed_count"], 2)

    def test_source_conflict_provenance_health_distinguishes_final_and_raw_bbox_status(self) -> None:
        health = self.report["source_conflict_provenance_health"]
        self.assertEqual(health["status"], "complete")
        self.assertEqual(health["source_conflict_count"], 2)
        self.assertEqual(health["renumbering_provenance_count"], 1)
        self.assertEqual(health["historical_to_canonical_mapping_count"], 1)
        self.assertEqual(health["source_marker_sequence_anomaly_count"], 1)
        self.assertEqual(health["missing_anchor_terms_count"], 0)
        self.assertEqual(health["missing_query_anchor_terms_count"], 0)
        self.assertEqual(health["missing_provenance_summary_count"], 0)
        self.assertEqual(health["missing_final_authority_policy_count"], 0)
        self.assertEqual(health["unknown_source_anomaly_kind_count"], 0)
        self.assertEqual(health["invalid_source_mapping_kind_count"], 0)
        self.assertEqual(health["invalid_provenance_exception_category_count"], 0)
        self.assertEqual(health["invalid_provenance_review_status_count"], 0)
        self.assertEqual(health["final_evidence_available_count"], 2)
        self.assertEqual(health["raw_provenance_exact_available_count"], 2)
        self.assertEqual(health["raw_provenance_partial_available_count"], 0)
        self.assertEqual(health["raw_provenance_unavailable_count"], 0)
        self.assertEqual(health["all_relevant_span_highlight_count"], 2)
        self.assertEqual(health["anchor_only_highlight_count"], 0)
        self.assertEqual(health["contradictory_failure_reason_count"], 0)

    def test_unresolved_needs_review_count_is_zero_or_explicitly_tracked(self) -> None:
        self.assertEqual(self.report["provenance_exception_health"]["unresolved_needs_review_count"], 0)
        unresolved = [
            row for row in read_jsonl(FINAL / "validation_exceptions.jsonl") if row.get("status") == "unresolved_manual_review_required"
        ]
        self.assertFalse(unresolved)

    def test_validation_exception_source_conflict_is_reviewed_nonruntime(self) -> None:
        row = next(
            row
            for row in read_jsonl(FINAL / "validation_exceptions.jsonl")
            if row["exception_id"] == "excluded_conflict_scope::uud_1945_amendment_4_aturan_tambahan_pasal_ii_iii_conflict_v1"
        )
        self.assertEqual(row["status"], ACCEPTED_NONCANONICAL_SOURCE_CONFLICT_TRACE_ONLY)
        self.assertFalse(row["runtime_loadable"])
        self.assertFalse(row["canonical_use_allowed"])
        self.assertNotIn("evidence_id", row)
        self.assertNotIn("bbox_id", row)
        self.assertNotIn("text_span_id", row)

    def test_validation_exception_chunk_references_are_current(self) -> None:
        exceptions = read_jsonl(FINAL / "validation_exceptions.jsonl")
        for row in exceptions:
            for field in ("chunk_id", "unresolved_chunk_reference"):
                chunk_id = row.get(field)
                if chunk_id:
                    self.assertIn(chunk_id, self.chunks, row["exception_id"])
            chunk_id = row.get("unresolved_chunk_reference")
            label = (row.get("evidence_summary") or {}).get("unit_label")
            if chunk_id and label:
                self.assertIn(label, self.chunks[chunk_id]["hierarchy"], row["exception_id"])

    def test_all_text_disposition_health_is_reported_without_fake_classification(self) -> None:
        spans = read_jsonl(FINAL / "page_text_spans.jsonl")
        referenced = {span_id for row in (*self.units.values(), *self.chunks.values()) for span_id in row.get("text_span_ids") or ()}
        health = self.report["all_text_disposition_health"]
        self.assertEqual(health["page_text_span_count"], len(spans))
        self.assertEqual(health["span_disposition_present_count"], len(spans))
        self.assertEqual(health["span_disposition_missing_count"], 0)
        self.assertEqual(health["known_unreferenced_span_count"], len({row["text_span_id"] for row in spans} - referenced))
        self.assertEqual(health["promotion_status_present_count"], len(spans))
        self.assertEqual(health["legal_force_present_count"], len(spans))
        self.assertEqual(health["needs_review_count"], 0)
        self.assertEqual(health["fake_grounding_id_count"], 0)
        self.assertEqual(health["status"], "complete")

    def test_artifact_rebuild_is_idempotent(self) -> None:
        self.assertEqual(read_json(FINAL / "manifest.json")["files"]["validation_report.json"]["origin"], "generated")


if __name__ == "__main__":
    unittest.main()
