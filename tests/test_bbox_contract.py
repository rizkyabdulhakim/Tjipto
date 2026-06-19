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
        self.assertEqual(len(rows), 1388)
        for row in rows:
            self.assertTrue(bbox_is_accepted(row))
            self.assertGreaterEqual(row["x1"], row["x0"])
            self.assertGreaterEqual(row["y1"], row["y0"])
            self.assertTrue(row["text"])
            self.assertTrue((ROOT / row["source_pdf_path"]).exists())


if __name__ == "__main__":
    unittest.main()
