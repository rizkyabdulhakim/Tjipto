from __future__ import annotations

from pathlib import Path
import unittest

from tjipto.core.manifest import read_json, read_jsonl
from tjipto.ingestion.pdf.health import build_pdf_health_report


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "data/final/uud"


class PdfHealthContractTest(unittest.TestCase):
    def test_pdf_health_report_classifies_all_sources_and_pages_native_text_ok(self) -> None:
        report = read_json(FINAL / "pdf_health_report.json")
        self.assertEqual(report["status"], "native_text_ok")
        self.assertEqual(report["source_count"], 6)
        self.assertEqual(report["page_count"], 63)
        self.assertEqual(report["native_text_ok_source_count"], 6)
        self.assertEqual(report["native_text_ok_page_count"], 63)
        self.assertEqual(report["ocr_required_count"], 0)
        self.assertEqual(report["ocr_dependency_status"], "not_required")
        self.assertFalse(report["ocr_candidates"])
        for row in report["source_documents"]:
            self.assertTrue(row["path_exists"], row["source_document_id"])
            self.assertTrue(row["sha256_match"], row["source_document_id"])
            self.assertTrue(row["file_size_match"], row["source_document_id"])
            self.assertTrue(row["page_count_match"], row["source_document_id"])
            self.assertTrue(row["native_text_ok"], row["source_document_id"])
            self.assertFalse(row["ocr_required"], row["source_document_id"])
            self.assertEqual(row["health_decision"], "native_text_ok", row["source_document_id"])
        for row in report["pages"]:
            self.assertTrue(row["native_text_available"], row)
            self.assertTrue(row["page_text_matches_artifact"], row)
            self.assertGreater(row["text_span_count"], 0, row)
            self.assertFalse(row["ocr_required"], row)
            self.assertEqual(row["health_decision"], "native_text_ok", row)

    def test_pdf_health_report_rebuilds_from_sources_pages_and_spans(self) -> None:
        self.assertEqual(
            build_pdf_health_report(
                repo_root=ROOT,
                corpus_id="uud",
                source_documents={row["source_document_id"]: row for row in read_jsonl(FINAL / "source_documents.jsonl")},
                pages=read_jsonl(FINAL / "pages.jsonl"),
                page_text_spans=read_jsonl(FINAL / "page_text_spans.jsonl"),
            ),
            read_json(FINAL / "pdf_health_report.json"),
        )

    def test_validation_report_includes_pdf_health_gate(self) -> None:
        health = read_json(FINAL / "validation_report.json")["pdf_health"]
        self.assertEqual(health["status"], "native_text_ok")
        self.assertEqual(health["source_count"], 6)
        self.assertEqual(health["page_count"], 63)
        self.assertEqual(health["ocr_required_count"], 0)
        self.assertEqual(health["needs_review_count"], 0)
        self.assertEqual(health["repair_required_count"], 0)


if __name__ == "__main__":
    unittest.main()
