from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from importlib.util import find_spec
from pathlib import Path
from typing import Protocol

from tjipto.core.manifest import file_sha256


class PdfPage(Protocol):
    def get_text(self, kind: str) -> str: ...


class PdfDocument(Protocol):
    page_count: int

    def __getitem__(self, index: int) -> PdfPage: ...

    def close(self) -> None: ...


def build_pdf_health_report(
    *,
    repo_root: Path,
    corpus_id: str,
    source_documents: dict[str, dict],
    pages: list[dict],
    page_text_spans: list[dict],
    pdf_opener: Callable[[Path], PdfDocument] | None = None,
    ocr_dependency_available: bool | None = None,
) -> dict:
    if pdf_opener is None:
        try:
            import fitz
        except ImportError as error:  # pragma: no cover
            raise RuntimeError("PyMuPDF is required to inspect PDF health") from error

        pdf_opener = fitz.open
    if ocr_dependency_available is None:
        ocr_dependency_available = any(find_spec(name) is not None for name in ("pytesseract", "rapidocr_onnxruntime"))

    pages_by_key = {(row["source_document_id"], row["page_number"]): row for row in pages}
    span_counts: dict[tuple[str, int], int] = {}
    for row in page_text_spans:
        span_counts[(row["source_document_id"], row["page_number"])] = (
            span_counts.get((row["source_document_id"], row["page_number"]), 0) + 1
        )

    source_rows = []
    page_rows = []
    for source_id, source in sorted(source_documents.items()):
        path = repo_root / source["path"]
        path_exists = path.exists()
        sha256_match = path_exists and file_sha256(path) == source["sha256"]
        file_size_match = path_exists and path.stat().st_size == source["file_size"]
        native_pages_ok = True
        page_count_match = False
        if path_exists:
            try:
                doc = pdf_opener(path)
            except Exception:
                doc = None
            try:
                if doc is None:
                    native_pages_ok = False
                else:
                    page_count_match = doc.page_count == source["page_count"]
                    for page_number in range(1, doc.page_count + 1):
                        native_text = doc[page_number - 1].get_text("text")
                        page = pages_by_key.get((source_id, page_number), {})
                        native_text_ok = bool(_compact(native_text))
                        page_text_match = _compact(page.get("text")) == _compact(native_text)
                        has_spans = span_counts.get((source_id, page_number), 0) > 0
                        if native_text_ok and page_text_match and has_spans:
                            decision = "native_text_ok"
                        elif not native_text_ok:
                            decision = "ocr_required"
                        else:
                            decision = "repair_required"
                        native_pages_ok = native_pages_ok and decision == "native_text_ok"
                        page_rows.append(
                            {
                                "corpus_id": corpus_id,
                                "source_document_id": source_id,
                                "page_number": page_number,
                                "native_text_available": native_text_ok,
                                "page_text_matches_artifact": page_text_match,
                                "text_span_count": span_counts.get((source_id, page_number), 0),
                                "ocr_required": decision == "ocr_required",
                                "health_decision": decision,
                            }
                        )
            finally:
                if doc is not None:
                    doc.close()
        source_is_usable = path_exists and sha256_match and file_size_match and page_count_match
        if source_is_usable and native_pages_ok:
            source_decision = "native_text_ok"
        elif not source_is_usable:
            source_decision = "source_unusable"
        else:
            source_decision = "needs_review"
        source_rows.append(
            {
                "corpus_id": corpus_id,
                "source_document_id": source_id,
                "path": source["path"],
                "path_exists": path_exists,
                "sha256_match": sha256_match,
                "file_size_match": file_size_match,
                "page_count_match": page_count_match,
                "native_text_ok": source_decision == "native_text_ok",
                "ocr_required": any(row["ocr_required"] for row in page_rows if row["source_document_id"] == source_id),
                "health_decision": source_decision,
            }
        )

    ocr_candidates = [row for row in page_rows if row["ocr_required"]]
    return {
        "corpus_id": corpus_id,
        "status": "native_text_ok" if not ocr_candidates and all(row["native_text_ok"] for row in source_rows) else "needs_review",
        "source_count": len(source_rows),
        "page_count": len(page_rows),
        "native_text_ok_source_count": sum(1 for row in source_rows if row["native_text_ok"]),
        "native_text_ok_page_count": sum(1 for row in page_rows if row["health_decision"] == "native_text_ok"),
        "ocr_required_count": len(ocr_candidates),
        "ocr_dependency_status": (
            "not_required" if not ocr_candidates else "available" if ocr_dependency_available else "ocr_dependency_unavailable"
        ),
        "ocr_candidates": ocr_candidates,
        "source_documents": source_rows,
        "pages": page_rows,
    }


def _compact(text: object) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or "")).replace("\u00ad", "")
    return "".join(re.findall(r"\w+", normalized.casefold()))
