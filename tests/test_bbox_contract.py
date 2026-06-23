from __future__ import annotations

from pathlib import Path
import unittest

from tjipto.core.manifest import read_jsonl
from tjipto.evidence.bbox import bbox_is_accepted


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "data/final/uud"


class BBoxContractTest(unittest.TestCase):
    def test_bbox_rows_are_accepted_and_page_grounded(self) -> None:
        rows = read_jsonl(FINAL / "bbox_registry.jsonl")
        self.assertEqual(len(rows), 1542)
        for row in rows:
            self.assertTrue(bbox_is_accepted(row))
            self.assertGreaterEqual(row["x1"], row["x0"])
            self.assertGreaterEqual(row["y1"], row["y0"])
            self.assertTrue(row["text"])
            self.assertTrue((ROOT / row["source_pdf_path"]).exists())

    def test_metadata_grounding_is_non_normative_and_bbox_linked(self) -> None:
        legal_evidence_bbox_ids = {
            row["bbox_id"] for row in read_jsonl(FINAL / "bbox_registry.jsonl")
        }
        metadata_grounding_ids = {
            row["bbox_id"] for row in read_jsonl(FINAL / "metadata_grounding_registry.jsonl")
        }
        rows = read_jsonl(FINAL / "metadata_grounding.jsonl")
        self.assertEqual(len(rows), 5)
        self.assertEqual(len(metadata_grounding_ids), 5)
        for row in rows:
            self.assertEqual(row["status"], "accepted_metadata_grounding")
            self.assertFalse(row["viewer_highlightable"])
            self.assertTrue(row["quoted_text"])
            self.assertTrue(row["bbox_refs"])
            self.assertTrue(set(row["bbox_refs"]).isdisjoint(legal_evidence_bbox_ids))
            self.assertTrue(set(row["bbox_refs"]) <= metadata_grounding_ids)
            self.assertTrue((ROOT / row["source_pdf_path"]).exists())

    def test_inserted_bab_heading_bboxes_attach_to_matching_structure(self) -> None:
        evidence = {
            row["evidence_id"]: row
            for row in read_jsonl(FINAL / "evidence_registry.jsonl")
        }
        expected = {
            "BAB IXA": "BAB IXA",
            "BAB XA": "BAB XA",
            "BAB VIIA": "BAB VIIA",
            "BAB VIIB": "BAB VIIB",
            "BAB VIIIA": "BAB VIIIA",
        }
        heading_rows = [
            row for row in read_jsonl(FINAL / "bbox_registry.jsonl")
            if row["text"] in expected
        ]
        self.assertEqual(len(heading_rows), 5)
        for row in heading_rows:
            target = evidence[row["evidence_id"]]
            self.assertEqual(target["hierarchy"][0], expected[row["text"]], row["bbox_id"])
            self.assertIn(row["bbox_id"], target["bbox_refs"])
            self.assertIn("heading_bab_", row["bbox_id"])


if __name__ == "__main__":
    unittest.main()
