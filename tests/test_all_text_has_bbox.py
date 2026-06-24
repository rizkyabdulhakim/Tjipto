from __future__ import annotations

from pathlib import Path
import unittest

from tjipto.core.manifest import read_jsonl


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "data/final/uud"


class AllTextHasBBoxTest(unittest.TestCase):
    def test_every_text_span_has_bbox_coordinates_or_failure_status(self) -> None:
        for row in read_jsonl(FINAL / "page_text_spans.jsonl"):
            if row["status"] == "accepted_text_span":
                self.assertLess(row["x0"], row["x1"], row["text_span_id"])
                self.assertLess(row["y0"], row["y1"], row["text_span_id"])
                self.assertEqual(row["bbox_precision"], "exact", row["text_span_id"])
            else:
                self.assertTrue(row.get("failure_reason"), row["text_span_id"])


if __name__ == "__main__":
    unittest.main()
