from __future__ import annotations

from pathlib import Path
from hashlib import sha256
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

    def test_selectors_reconstruct_the_canonical_page_stream(self) -> None:
        spans = read_jsonl(FINAL / "page_text_spans.jsonl")
        streams: dict[str, list[dict]] = {}
        for row in spans:
            streams.setdefault(row["stream_id"], []).append(row)
        for stream_id, rows in streams.items():
            rows.sort(key=lambda row: row["text_start"])
            stream = "\n".join(row["text"] for row in rows)
            self.assertEqual(sha256(stream.encode("utf-8")).hexdigest(), rows[0]["page_text_hash"], stream_id)
            for row in rows:
                self.assertEqual(stream[row["text_start"] : row["text_end"]], row["exact_quote"], row["text_span_id"])

    def test_source_markers_have_raw_disposition_and_never_enter_semantic_text(self) -> None:
        markers = {"*)", "**)", "***)", "****)", "***/****)"}
        raw = read_jsonl(FINAL / "raw_source_spans.jsonl")
        marker_rows = [row for row in raw if row["classification"] == "source_annotation_marker"]
        self.assertEqual({row["raw_text"] for row in marker_rows}, markers)
        self.assertTrue(all(row["disposition_reason"] == "source_annotation_marker" for row in marker_rows))
        self.assertTrue(all(row["legal_text"] is False for row in marker_rows))
        self.assertTrue(all(row["citation_eligible"] is False for row in marker_rows))
        self.assertTrue(all(row["relevant_quote_eligible"] is False for row in marker_rows))
        self.assertTrue(all(row["default_highlight_eligible"] is False for row in marker_rows))
        for name, field in (("page_text_spans", "text"), ("legal_units", "text"), ("evidence_registry", "quoted_text")):
            rows = read_jsonl(FINAL / f"{name}.jsonl")
            self.assertFalse(any(marker in str(row.get(field) or "") for row in rows for marker in markers), name)


if __name__ == "__main__":
    unittest.main()
