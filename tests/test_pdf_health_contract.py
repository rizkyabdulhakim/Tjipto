from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from tjipto.core.manifest import file_sha256, read_json, read_jsonl
from tjipto.ingestion.pdf.health import build_pdf_health_report


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "data/final/uud"


class _FakePage:
    def __init__(self, text: str) -> None:
        self._text = text

    def get_text(self, kind: str) -> str:
        return self._text


class _FakeDocument:
    def __init__(self, page_texts: list[str]) -> None:
        self._pages = [_FakePage(text) for text in page_texts]
        self.page_count = len(self._pages)
        self.closed = False

    def __getitem__(self, index: int) -> _FakePage:
        return self._pages[index]

    def close(self) -> None:
        self.closed = True


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

    def test_pdf_health_flags_missing_or_scanned_text_without_ocr_dependency(self) -> None:
        with self.subTest("missing native text"):
            report = self._build_fake_report(
                page_texts=[""],
                pages=[{"source_document_id": "doc", "page_number": 1, "text": ""}],
                spans=[],
                ocr_dependency_available=False,
            )
            self.assertEqual(report["status"], "needs_review")
            self.assertEqual(report["ocr_required_count"], 1)
            self.assertEqual(report["ocr_dependency_status"], "ocr_dependency_unavailable")
            self.assertEqual(report["pages"][0]["health_decision"], "ocr_required")

        with self.subTest("image only scanned page"):
            report = self._build_fake_report(
                page_texts=[""],
                pages=[{"source_document_id": "doc", "page_number": 1, "text": "artifact text"}],
                spans=[],
                ocr_dependency_available=False,
            )
            self.assertEqual(report["pages"][0]["health_decision"], "ocr_required")
            self.assertTrue(report["pages"][0]["ocr_required"])

    def test_pdf_health_flags_corrupt_and_repair_required_sources(self) -> None:
        with self.subTest("corrupt unreadable PDF"):
            report = self._build_fake_report(page_texts=None, pages=[], spans=[])
            self.assertEqual(report["source_documents"][0]["health_decision"], "source_unusable")
            self.assertEqual(report["page_count"], 0)

        with self.subTest("native text mismatch requires repair"):
            report = self._build_fake_report(
                page_texts=["native text"],
                pages=[{"source_document_id": "doc", "page_number": 1, "text": "different text"}],
                spans=[{"source_document_id": "doc", "page_number": 1}],
            )
            self.assertEqual(report["pages"][0]["health_decision"], "repair_required")
            self.assertFalse(report["pages"][0]["ocr_required"])

    def test_pdf_health_reports_available_ocr_dependency_without_running_ocr(self) -> None:
        report = self._build_fake_report(
            page_texts=[""],
            pages=[{"source_document_id": "doc", "page_number": 1, "text": ""}],
            spans=[],
            ocr_dependency_available=True,
        )
        self.assertEqual(report["ocr_dependency_status"], "available")
        self.assertEqual(report["ocr_required_count"], 1)

    def _build_fake_report(
        self,
        *,
        page_texts: list[str] | None,
        pages: list[dict],
        spans: list[dict],
        ocr_dependency_available: bool = False,
    ) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Path(tmp) / "pdf_health_probe.pdf"
            fixture.write_bytes(b"%PDF probe")
            source_documents = {
                "doc": {
                    "source_document_id": "doc",
                    "path": str(fixture),
                    "sha256": file_sha256(fixture),
                    "file_size": fixture.stat().st_size,
                    "page_count": len(page_texts or []),
                }
            }

            def open_fake(_path: Path) -> _FakeDocument:
                if page_texts is None:
                    raise RuntimeError("unreadable PDF")
                return _FakeDocument(page_texts)

            return build_pdf_health_report(
                repo_root=ROOT,
                corpus_id="test",
                source_documents=source_documents,
                pages=pages,
                page_text_spans=spans,
                pdf_opener=open_fake,
                ocr_dependency_available=ocr_dependency_available,
            )


if __name__ == "__main__":
    unittest.main()
