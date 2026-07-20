from __future__ import annotations

from pathlib import Path
import unittest

from tjipto.core.manifest import read_jsonl
from tjipto.ingestion.pdf.text_spans import build_pdf_text_spans


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
            self.assertIsInstance(row["viewer_highlightable"], bool)
            self.assertEqual(row["object_role"], "source_span")
            self.assertNotIn("highlightable", row)

    def test_text_span_builder_uses_source_metadata_not_id_format(self) -> None:
        rows = build_pdf_text_spans(
            source_documents={
                "synthetic-source": {
                    "filename": "synthetic.pdf",
                    "path": "data/sources/synthetic.pdf",
                    "source_role": "synthetic_role",
                    "temporal_context": "synthetic_time",
                    "sha256": "abc123",
                }
            },
            pdf_lines={"synthetic-source": {1: [{"text": "hello", "x0": 1, "x1": 2, "y0": 3, "y1": 4}]}},
            corpus_id="synthetic",
            text_span_id_prefix="span",
        )
        self.assertEqual(rows[0]["source_document_id"], "synthetic-source")
        self.assertEqual(rows[0]["source_role"], "synthetic_role")
        self.assertEqual(rows[0]["temporal_context"], "synthetic_time")
        self.assertEqual(rows[0]["text_span_id"], "span::synthetic_role::0001::0000")


if __name__ == "__main__":
    unittest.main()
