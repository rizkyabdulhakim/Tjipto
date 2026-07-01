from __future__ import annotations

import json
from pathlib import Path
import unittest

from tjipto.corpora.disposition import SPAN_DISPOSITION_FIELDS


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

    def test_every_page_text_span_has_disposition(self) -> None:
        for span in self.spans:
            for field in SPAN_DISPOSITION_FIELDS:
                self.assertIn(field, span, span["text_span_id"])
            self.assertTrue(span["span_role"], span["text_span_id"])
            self.assertTrue(span["semantic_classification"], span["text_span_id"])
            self.assertTrue(span["promotion_status"], span["text_span_id"])
            self.assertTrue(span["legal_force"], span["text_span_id"])

    def test_excluded_and_needs_review_spans_fail_closed(self) -> None:
        for span in self.spans:
            if span["promotion_status"].startswith("excluded") or span["promotion_status"] in {"nonruntime_instrument_text", "needs_review"}:
                self.assertTrue(span["exclusion_reason"], span["text_span_id"])
            if span["promotion_status"] == "needs_review":
                self.assertIsNot(span.get("runtime_loadable"), True, span["text_span_id"])
                self.assertIsNot(span.get("canonical_use_allowed"), True, span["text_span_id"])

    def test_promoted_span_targets_resolve(self) -> None:
        for span in self.spans:
            target_type = span["promotion_target_type"]
            target_id = span["promotion_target_id"]
            if span["promotion_status"] == "promoted_legal_unit":
                self.assertIn(target_type, {"chunk", "legal_unit"})
                self.assertIn(target_id, self.chunks if target_type == "chunk" else self.units)
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

    def test_known_instrument_text_gaps_remain_nonruntime(self) -> None:
        for chunk_id in (
            "uud_chunk_00632",
            "uud_chunk_00639",
            "uud_chunk_00623",
            "uud_chunk_00624",
            "uud_chunk_00634",
            "uud_chunk_00635",
            "uud_chunk_00648",
            "uud_chunk_00646",
            "uud_chunk_00647",
        ):
            chunk = self.chunks[chunk_id]
            self.assertFalse(chunk["runtime_loadable"], chunk_id)
            self.assertIn(chunk["validation_status"], {
                "accepted_false_positive_segmentation_punctuation",
                "builder_slicing_label_issue_confirmed",
                "duplicated_heading_artifact_issue_confirmed",
            })

    def test_validation_report_matches_disposition_counts(self) -> None:
        health = read_json(FINAL / "validation_report.json")["all_text_disposition_health"]
        self.assertEqual(health["page_text_span_count"], len(self.spans))
        self.assertEqual(health["span_disposition_missing_count"], 0)
        self.assertEqual(health["semantic_classification_present_count"], len(self.spans))
        self.assertEqual(health["promotion_status_present_count"], len(self.spans))
        self.assertEqual(health["legal_force_present_count"], len(self.spans))
        self.assertEqual(health["needs_review_count"], 0)
        self.assertEqual(health["status"], "complete")


if __name__ == "__main__":
    unittest.main()
