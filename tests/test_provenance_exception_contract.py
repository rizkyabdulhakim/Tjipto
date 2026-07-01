from __future__ import annotations

import json
from pathlib import Path
import unittest

from tjipto.corpora.uud.provenance_exceptions import (
    ACCEPTED_FALSE_POSITIVE_SEGMENTATION_PUNCTUATION,
    ACCEPTED_NONCANONICAL_SOURCE_CONFLICT_TRACE_ONLY,
    BUILDER_SLICING_LABEL_ISSUE_CONFIRMED,
    DUPLICATED_HEADING_ARTIFACT_ISSUE_CONFIRMED,
    SOURCE_TEXT_ACCEPTED_NONRUNTIME_NO_EVIDENCE_BBOX,
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
        for row_id in ("00623", "00634", "00645", "00646", "00647", "00648"):
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

    def test_noncanonical_source_conflict_trace_is_not_runtime_or_canonical(self) -> None:
        for row_id in ("00645", "00646", "00647"):
            unit = self.units[f"uud_legal_unit_{row_id}"]
            chunk = self.chunks[f"uud_chunk_{row_id}"]
            self.assertFalse(unit["runtime_loadable"])
            self.assertFalse(unit["canonical_use_allowed"])
            self.assertFalse(chunk["runtime_loadable"])
            self.assertFalse(chunk["canonical_use_allowed"])
            self.assertFalse(unit["bbox_ids"])
            self.assertFalse(unit["text_span_ids"])
            self.assertFalse(chunk["bbox_ids"])
            self.assertFalse(chunk["evidence_ids"])
            self.assertFalse(chunk["text_span_ids"])
        self.assertEqual(
            self.units["uud_legal_unit_00645"]["provenance_exception_category"],
            ACCEPTED_NONCANONICAL_SOURCE_CONFLICT_TRACE_ONLY,
        )

    def test_00646_does_not_have_misleading_aturan_tambahan_label(self) -> None:
        text = self.units["uud_legal_unit_00646"]["text"]
        self.assertEqual(
            self.units["uud_legal_unit_00646"]["provenance_exception_category"],
            BUILDER_SLICING_LABEL_ISSUE_CONFIRMED,
        )
        self.assertIn("Majelis Permusyawaratan Rakyat", text)
        self.assertNotIn("Segala peraturan perundangundangan", text)

    def test_00647_does_not_duplicate_heading(self) -> None:
        text = self.units["uud_legal_unit_00647"]["text"]
        self.assertEqual(
            self.units["uud_legal_unit_00647"]["provenance_exception_category"],
            DUPLICATED_HEADING_ARTIFACT_ISSUE_CONFIRMED,
        )
        self.assertEqual([line.strip() for line in text.splitlines()].count("Pasal III"), 1)

    def test_legacy_review_records_are_source_text_accepted_nonruntime_no_evidence_bbox(self) -> None:
        for chunk_id in ("uud_chunk_00005", "uud_chunk_00022", "uud_chunk_00101", "uud_chunk_00188"):
            chunk = self.chunks[chunk_id]
            self.assertEqual(chunk["status"], SOURCE_TEXT_ACCEPTED_NONRUNTIME_NO_EVIDENCE_BBOX)
            self.assertEqual(chunk["provenance_exception_category"], SOURCE_TEXT_ACCEPTED_NONRUNTIME_NO_EVIDENCE_BBOX)
            self.assertFalse(chunk["runtime_loadable"])
            self.assertFalse(chunk["canonical_use_allowed"])
            self.assertTrue(chunk["text_span_ids"])
            self.assertFalse(chunk["evidence_ids"])
            self.assertFalse(chunk["bbox_ids"])

    def test_validation_report_has_provenance_exception_health(self) -> None:
        health = self.report["provenance_exception_health"]
        self.assertEqual(health["accepted_false_positive_segmentation_punctuation_count"], 16)
        self.assertEqual(health["accepted_noncanonical_source_conflict_trace_only_count"], 3)
        self.assertEqual(health["builder_slicing_label_issue_confirmed_count"], 2)
        self.assertEqual(health["duplicated_heading_artifact_issue_confirmed_count"], 2)
        self.assertEqual(health["source_text_accepted_nonruntime_no_evidence_bbox_count"], 8)

    def test_unresolved_needs_review_count_is_zero_or_explicitly_tracked(self) -> None:
        self.assertEqual(self.report["provenance_exception_health"]["unresolved_needs_review_count"], 0)

    def test_artifact_rebuild_is_idempotent(self) -> None:
        self.assertEqual(read_json(FINAL / "manifest.json")["files"]["validation_report.json"]["origin"], "generated")


if __name__ == "__main__":
    unittest.main()
