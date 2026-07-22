from __future__ import annotations

import json
from pathlib import Path
import unittest

from tjipto.corpora.disposition import (
    LEGAL_FORCES,
    PROMOTION_STATUSES,
    REVIEW_STATUSES,
    SEMANTIC_CLASSIFICATIONS,
    SPAN_DISPOSITION_FIELDS,
    SPAN_ROLES,
)


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "data/final/uud"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class TextSpanDispositionContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.spans = read_jsonl(FINAL / "page_text_spans.jsonl")
        self.units = {row["legal_unit_id"]: row for row in read_jsonl(FINAL / "legal_units.jsonl")}
        self.chunks = {row["chunk_id"]: row for row in read_jsonl(FINAL / "chunks.jsonl")}
        self.metadata = {row["metadata_grounding_id"]: row for row in read_jsonl(FINAL / "metadata_grounding.jsonl")}
        self.source_conflicts = {row["source_conflict_id"]: row for row in read_jsonl(FINAL / "source_conflicts.jsonl")}
        self.refs_by_span: dict[str, list[dict]] = {}
        for unit in self.units.values():
            for span_id in unit.get("text_span_ids") or ():
                self.refs_by_span.setdefault(span_id, []).append(unit)

    def test_every_page_text_span_has_disposition(self) -> None:
        for span in self.spans:
            for field in SPAN_DISPOSITION_FIELDS:
                self.assertIn(field, span, span["text_span_id"])
            self.assertTrue(span["span_role"], span["text_span_id"])
            self.assertTrue(span["semantic_classification"], span["text_span_id"])
            self.assertTrue(span["promotion_status"], span["text_span_id"])
            self.assertTrue(span["legal_force"], span["text_span_id"])
            self.assertIn(span["span_role"], SPAN_ROLES, span["text_span_id"])
            self.assertIn(span["semantic_classification"], SEMANTIC_CLASSIFICATIONS, span["text_span_id"])
            self.assertIn(span["promotion_status"], PROMOTION_STATUSES, span["text_span_id"])
            self.assertIn(span["legal_force"], LEGAL_FORCES, span["text_span_id"])
            self.assertIn(span["review_status"], REVIEW_STATUSES, span["text_span_id"])

    def test_excluded_and_needs_review_spans_fail_closed(self) -> None:
        for span in self.spans:
            if span["promotion_status"].startswith("excluded") or span["promotion_status"] in {
                "nonruntime_instrument_text",
                "needs_review",
            }:
                self.assertTrue(span["exclusion_reason"], span["text_span_id"])
            if span["promotion_status"] == "needs_review":
                self.assertIsNot(span.get("runtime_loadable"), True, span["text_span_id"])
                self.assertIsNot(span.get("canonical_use_allowed"), True, span["text_span_id"])

    def test_promoted_span_targets_resolve(self) -> None:
        for span in self.spans:
            target_type = span["promotion_target_type"]
            target_id = span["promotion_target_id"]
            if span["promotion_status"] == "promoted_legal_unit":
                self.assertEqual(target_type, "legal_unit")
                self.assertIn(target_id, self.units)
            elif span["promotion_status"] == "promoted_metadata":
                self.assertIn(target_id, self.metadata)
            elif span["promotion_status"] == "promoted_source_conflict":
                if target_type == "source_conflict":
                    self.assertIn(target_id, self.source_conflicts)
                else:
                    self.assertIn(target_id, self.units)

    def test_source_conflict_trace_spans_are_not_runtime_or_canonical(self) -> None:
        for span in self.spans:
            if span["span_role"] == "source_conflict_trace":
                self.assertNotEqual(span["legal_force"], "canonical_normative")
                self.assertNotEqual(span["promotion_status"], "promoted_legal_unit")

    def test_normative_spans_are_not_stolen_by_parent_structural_units(self) -> None:
        for span in self.spans:
            unit_types = {row["unit_type"] for row in self.refs_by_span.get(span["text_span_id"], [])}
            if unit_types & {"pasal_record", "ayat_record", "pembukaan_record"}:
                self.assertNotEqual(span["span_role"], "structural_heading", span["text_span_id"])
                self.assertNotEqual(span["promotion_status"], "excluded_structural", span["text_span_id"])
            if unit_types & {"ayat_record"}:
                self.assertEqual(self.units[span["promotion_target_id"]]["unit_type"], "ayat_record", span["text_span_id"])
            elif unit_types & {"pasal_record"}:
                target_id = span["promotion_target_id"]
                target_type = self.units[target_id]["unit_type"]
                if target_type not in {"pasal_record", "ayat_record"}:
                    self.assertTrue(
                        any(
                            target_id in unit.get("ancestor_legal_unit_ids", ())
                            for unit in self.refs_by_span[span["text_span_id"]]
                            if unit["unit_type"] == "pasal_record"
                        ),
                        span["text_span_id"],
                    )

    def test_representative_semantic_precedence_examples(self) -> None:
        self.assert_span("Negara Indonesia ialah Negara Kesatuan", "normative_text", "promoted_legal_unit", "ayat_record")
        self.assert_span("Segala warga negara bersamaan kedudukannya", "normative_text", "promoted_legal_unit", "ayat_record")
        self.assert_span("Bahwa sesungguhnya kemerdekaan", "normative_text", "promoted_legal_unit", "pembukaan_record")
        self.assert_span("BAB IXA", "structural_heading", "excluded_structural", "bab_record")
        self.assert_span("ATURAN TAMBAHAN", "structural_heading", "excluded_structural", "aturan_tambahan_record")
        self.assert_span("Peralihan Pasal I, II, dan III", "instrument_scope", "nonruntime_instrument_text", None)

    def test_known_instrument_text_gaps_remain_nonruntime(self) -> None:
        for chunk_id in (
            "uud_chunk_00623",
            "uud_chunk_00624",
            "uud_chunk_00634",
            "uud_chunk_00635",
            "uud_chunk_00648",
        ):
            chunk = self.chunks[chunk_id]
            self.assertFalse(chunk["runtime_loadable"], chunk_id)
            self.assertIn(
                chunk["validation_status"],
                {
                    "accepted_false_positive_segmentation_punctuation",
                    "builder_slicing_label_issue_confirmed",
                    "duplicated_heading_artifact_issue_confirmed",
                },
            )

    def test_validation_report_matches_disposition_counts(self) -> None:
        health = read_json(FINAL / "validation_report.json")["all_text_disposition_health"]
        self.assertEqual(health["page_text_span_count"], len(self.spans))
        self.assertEqual(health["classified_span_count"], len(self.spans))
        self.assertEqual(health["span_disposition_missing_count"], 0)
        self.assertEqual(health["semantic_classification_present_count"], len(self.spans))
        self.assertEqual(health["promotion_status_present_count"], len(self.spans))
        self.assertEqual(health["legal_force_present_count"], len(self.spans))
        self.assertEqual(health["missing_source_ref_count"], 0)
        self.assertEqual(health["missing_page_ref_count"], 0)
        self.assertEqual(health["missing_bbox_coordinate_count"], 0)
        self.assertEqual(health["invalid_bbox_coordinate_count"], 0)
        self.assertEqual(health["invalid_span_role_count"], 0)
        self.assertEqual(health["invalid_semantic_classification_count"], 0)
        self.assertEqual(health["invalid_legal_force_count"], 0)
        self.assertEqual(health["invalid_promotion_status_count"], 0)
        self.assertEqual(health["invalid_review_status_count"], 0)
        self.assertEqual(health["ambiguous_disposition_count"], 0)
        self.assertEqual(health["needs_review_count"], 0)
        self.assertEqual(health["status"], "complete")

        semantic = read_json(FINAL / "validation_report.json")["semantic_precedence_health"]
        self.assertEqual(semantic["normative_spans_classified_structural_count"], 0)
        self.assertEqual(semantic["pasal_ayat_spans_classified_structural_count"], 0)
        self.assertEqual(semantic["parent_structural_override_count"], 0)
        self.assertEqual(semantic["structural_spans_with_normative_target_count"], 0)
        self.assertEqual(semantic["source_conflict_runtime_or_canonical_count"], 0)
        self.assertEqual(semantic["status"], "complete")

    def assert_span(
        self,
        text: str,
        role: str,
        status: str,
        target_unit_type: str | None,
    ) -> None:
        span = next(row for row in self.spans if text in row["text"])
        self.assertEqual(span["span_role"], role, span["text_span_id"])
        self.assertEqual(span["promotion_status"], status, span["text_span_id"])
        if target_unit_type:
            self.assertEqual(self.units[span["promotion_target_id"]]["unit_type"], target_unit_type, span["text_span_id"])


if __name__ == "__main__":
    unittest.main()
