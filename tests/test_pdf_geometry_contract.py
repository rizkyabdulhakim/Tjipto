from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re
import unicodedata
import unittest

from tjipto.core.manifest import read_jsonl


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "data/final/uud"


class PdfGeometryContractTest(unittest.TestCase):
    def test_all_exact_highlights_are_bounded_and_overlap_source_text(self) -> None:
        spans: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
        for row in read_jsonl(FINAL / "page_text_spans.jsonl"):
            spans[(row["source_document_id"], row["page_number"], _compact(row["text"]))].append(row)
        exact = [
            row
            for row in read_jsonl(FINAL / "bbox_registry.jsonl")
            if row.get("bbox_precision") == "exact" and row.get("viewer_highlightable") is True
        ]
        self.assertEqual(len(exact), 1566)
        for row in exact:
            with self.subTest(bbox_id=row["bbox_id"]):
                self.assertEqual(row["coordinate_space"], "pdf_user_space")
                self.assertEqual(row["coordinate_origin"], "top_left")
                self.assertEqual(row["page_rotation"], 0)
                self.assertEqual(row["page_box_basis"], "media_box")
                self.assertEqual(row["transform_version"], "pymupdf_top_left_v1")
                self.assertTrue(0 <= row["x0"] < row["x1"] <= row["page_width"])
                self.assertTrue(0 <= row["y0"] < row["y1"] <= row["page_height"])
                matches = spans[(row["source_document_id"], row["page_number"], _compact(row["text"]))]
                self.assertTrue(matches)
                self.assertTrue(any(_overlaps(row, span) for span in matches))
                converted = (row["x0"], row["page_height"] - row["y1"], row["x1"], row["page_height"] - row["y0"])
                round_trip = (converted[0], row["page_height"] - converted[3], converted[2], row["page_height"] - converted[1])
                self.assertLessEqual(
                    max(abs(left - right) for left, right in zip(round_trip, (row["x0"], row["y0"], row["x1"], row["y1"]))), 0.5
                )

    def test_page_only_and_nonhighlightable_rows_are_not_renderable(self) -> None:
        rows = read_jsonl(FINAL / "bbox_registry.jsonl")
        self.assertFalse(any(row.get("bbox_precision") == "page_grounded_only" and row.get("viewer_highlightable") is True for row in rows))


def _compact(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text or "").replace("\u00ad", "")).strip().casefold()


def _overlaps(left: dict, right: dict) -> bool:
    return min(left["x1"], right["x1"]) > max(left["x0"], right["x0"]) and min(left["y1"], right["y1"]) > max(left["y0"], right["y0"])


if __name__ == "__main__":
    unittest.main()
