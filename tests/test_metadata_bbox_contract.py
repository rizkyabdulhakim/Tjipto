from __future__ import annotations

import json
from pathlib import Path
import unittest

from tjipto.core.manifest import read_jsonl


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "data/final/uud"


class MetadataBBoxContractTest(unittest.TestCase):
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

    def test_fixture_metadata_rows_keep_stable_exact_grounding_ids(self) -> None:
        rows = {
            row["metadata_grounding_id"]: row
            for row in read_jsonl(FINAL / "metadata_grounding.jsonl")
        }
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
