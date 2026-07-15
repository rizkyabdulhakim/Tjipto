from __future__ import annotations

from pathlib import Path
import unittest

from tests.test_source_conflict_runtime_contract import _source_conflict_cases
from tjipto.core.manifest import read_jsonl


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "data/final/uud"


class SourceConflictGroundingContractTest(unittest.TestCase):
    def test_source_conflicts_have_stable_grounding_ids(self) -> None:
        conflicts = {row["source_conflict_id"]: row for row in read_jsonl(FINAL / "source_conflicts.jsonl")}
        text_span_ids = {row["text_span_id"] for row in read_jsonl(FINAL / "page_text_spans.jsonl")}
        evidence_ids = {row["evidence_id"] for row in read_jsonl(FINAL / "evidence_registry.jsonl")}
        bbox_ids = {row["bbox_id"] for row in read_jsonl(FINAL / "bbox_registry.jsonl")} | {
            row["word_bbox_id"] for row in read_jsonl(FINAL / "word_bboxes.jsonl")
        }
        for case in _source_conflict_cases():
            row = conflicts[case["source_conflict_id"]]
            self.assertEqual(row["text_span_ids"], case["text_span_ids"])
            self.assertEqual(row["evidence_ids"], case["evidence_ids"])
            self.assertEqual(row["bbox_ids"], case["bbox_ids"])
            self.assertEqual(row["source_anomaly_kind"], case["source_anomaly_kind"])
            self.assertTrue(row["anchor_terms"])
            self.assertTrue(row["query_anchor_terms"])
            self.assertTrue(row["provenance_summary"])
            self.assertTrue(row["final_authority_policy"])
            policy = row["source_anomaly_policy"]
            self.assertEqual(policy["corpus_id"], "uud")
            self.assertEqual(policy["anomaly_kind"], row["source_anomaly_kind"])
            self.assertEqual(policy["finality_policy"], "source_anomaly_provenance")
            self.assertEqual(policy["provenance_highlight_scope"], row["provenance_highlight_scope"])
            self.assertEqual(policy["reviewer_status"], "reviewed")
            self.assertIn("{summary}", policy["public_wording_template"])
            self.assertEqual(row["grounding_status"], "text_span_exact")
            self.assertEqual(row["validation_status"], "accepted_source_conflict_record")
            self.assertTrue(set(row["text_span_ids"]) <= text_span_ids)
            self.assertTrue(set(row["evidence_ids"]) <= evidence_ids)
            self.assertTrue(set(row["bbox_ids"]) <= bbox_ids)
            self.assertIn(
                row["provenance_bbox_status"], {"exact_raw_provenance_bbox_available", "partial_exact_raw_provenance_bbox_available"}
            )
            self.assertIn(row["provenance_highlight_scope"], {"all_relevant_spans", "anchor_span_only"})
            self.assertTrue(set(row["raw_provenance_text_span_ids"]) <= text_span_ids)
            self.assertTrue(set(row["raw_provenance_bbox_ids"]) <= bbox_ids)
            if case.get("source_mapping_kind"):
                self.assertEqual(row["source_mapping_kind"], case["source_mapping_kind"])
            if not row["evidence_ids"] or not row["bbox_ids"]:
                self.assertEqual(row["failure_reason"], case["failure_reason"])


if __name__ == "__main__":
    unittest.main()
