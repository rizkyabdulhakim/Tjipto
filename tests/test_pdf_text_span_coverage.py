from __future__ import annotations

from pathlib import Path
import unittest

from tjipto.core.manifest import read_jsonl


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "data/final/uud"


class PdfTextSpanCoverageTest(unittest.TestCase):
    def test_page_text_spans_cover_all_source_documents(self) -> None:
        source_ids = {row["source_document_id"] for row in read_jsonl(FINAL / "source_documents.jsonl")}
        spans = read_jsonl(FINAL / "page_text_spans.jsonl")
        self.assertTrue(spans)
        self.assertEqual({row["source_document_id"] for row in spans}, source_ids)
        for row in spans:
            self.assertEqual(row["status"], "accepted_text_span")
            self.assertTrue(row["text"].strip())
            self.assertEqual(row["bbox_precision"], "exact")
            self.assertFalse(row["viewer_highlightable"])


if __name__ == "__main__":
    unittest.main()
