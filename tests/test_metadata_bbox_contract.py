from __future__ import annotations

import json
from pathlib import Path
import unittest

from tjipto.core.manifest import read_jsonl


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "data/final/uud"


class MetadataBBoxContractTest(unittest.TestCase):
    def test_metadata_grounding_registry_has_stable_row_identity(self) -> None:
        rows = read_jsonl(FINAL / "metadata_grounding_registry.jsonl")
        ref_ids = [row.get("metadata_grounding_ref_id") for row in rows]
        bbox_ids = [row["bbox_id"] for row in rows]
        self.assertEqual(len(rows), 112)
        self.assertEqual(sum(1 for ref_id in ref_ids if ref_id), len(rows))
        self.assertEqual(len(set(ref_ids)), len(rows))
        self.assertGreater(len(bbox_ids) - len(set(bbox_ids)), 0)
        for bbox_id in {bbox_id for bbox_id in bbox_ids if bbox_ids.count(bbox_id) > 1}:
            duplicate_refs = [row["metadata_grounding_ref_id"] for row in rows if row["bbox_id"] == bbox_id]
            self.assertEqual(len(duplicate_refs), len(set(duplicate_refs)))

    def test_metadata_grounding_registry_exact_rows_resolve_to_bbox_registry(self) -> None:
        registry_rows = read_jsonl(FINAL / "metadata_grounding_registry.jsonl")
        bboxes = {row["bbox_id"]: row for row in read_jsonl(FINAL / "bbox_registry.jsonl")}
        exact_rows = [row for row in registry_rows if row["bbox_precision"] == "exact"]
        unresolved_rows = [row for row in registry_rows if row["bbox_id"] not in bboxes]
        self.assertTrue(exact_rows)
        self.assertFalse([row for row in exact_rows if row["bbox_id"] not in bboxes])
        for row in exact_rows:
            bbox = bboxes[row["bbox_id"]]
            for field in ("source_document_id", "page_number", "x0", "y0", "x1", "y1"):
                self.assertIn(field, bbox)
        for row in unresolved_rows:
            self.assertEqual(row["bbox_precision"], "page_grounded_only", row["metadata_grounding_ref_id"])
            self.assertEqual(row["failure_reason"], "metadata_bbox_reference_unresolved")

    def test_metadata_grounding_distinguishes_exact_from_page_grounded(self) -> None:
        registry_ids = {row["bbox_id"] for row in read_jsonl(FINAL / "metadata_grounding_registry.jsonl")}
        bbox_ids = {row["bbox_id"] for row in read_jsonl(FINAL / "bbox_registry.jsonl")}
        text_span_ids = {row["text_span_id"] for row in read_jsonl(FINAL / "page_text_spans.jsonl")}
        exact_rows = []
        for row in read_jsonl(FINAL / "metadata_grounding.jsonl"):
            self.assertFalse(row["viewer_highlightable"], row["metadata_grounding_id"])
            self.assertTrue(set(row["bbox_refs"]) <= registry_ids, row["metadata_grounding_id"])
            self.assertIn("grounding_status", row)
            if row["bbox_precision"] == "exact":
                exact_rows.append(row)
                self.assertEqual(row["grounding_status"], "text_bbox_exact", row["metadata_grounding_id"])
                self.assertTrue(set(row["bbox_ids"]) <= bbox_ids, row["metadata_grounding_id"])
                self.assertTrue(set(row["text_span_ids"]) <= text_span_ids, row["metadata_grounding_id"])
                self.assertTrue(row["bbox_ids"], row["metadata_grounding_id"])
                self.assertTrue(row["text_span_ids"], row["metadata_grounding_id"])
            else:
                self.assertEqual(row["bbox_precision"], "page_grounded_only", row["metadata_grounding_id"])
                self.assertIn("failure_reason", row, row["metadata_grounding_id"])
        self.assertTrue(exact_rows)

    def test_metadata_report_does_not_claim_highlight_ready(self) -> None:
        report = json.loads((FINAL / "validation_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["metadata_grounding_contract"]["status"], "field_grounded")
        self.assertEqual(report["metadata_bbox_registry_health"]["metadata_bbox_false_exact_claims"], 0)
        for filename in ("metadata_grounding.jsonl", "metadata_grounding_registry.jsonl"):
            for row in read_jsonl(FINAL / filename):
                self.assertFalse(row["viewer_highlightable"], row.get("metadata_grounding_id") or row["metadata_grounding_ref_id"])
                if row["bbox_precision"] == "page_grounded_only":
                    self.assertIn("failure_reason", row)

    def test_fixture_metadata_rows_keep_stable_exact_grounding_ids(self) -> None:
        rows = {row["metadata_grounding_id"]: row for row in read_jsonl(FINAL / "metadata_grounding.jsonl")}
        for case in _metadata_grounding_cases():
            row = rows[case["metadata_grounding_id"]]
            for field in (
                "metadata_field",
                "source_document_id",
                "grounding_status",
                "bbox_precision",
                "text_span_ids",
                "bbox_ids",
            ):
                self.assertEqual(row[field], case[field], case["metadata_grounding_id"])


def _metadata_grounding_cases() -> tuple[dict, ...]:
    path = ROOT / "tests/fixtures/uud/metadata_grounding_cases.jsonl"
    return tuple(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


if __name__ == "__main__":
    unittest.main()
