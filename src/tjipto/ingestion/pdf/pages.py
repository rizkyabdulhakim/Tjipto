from __future__ import annotations

from pathlib import Path
import re

from tjipto.ingestion.pdf.text_spans import normalize_semantic_text


def build_pdf_pages(
    *,
    repo_root: Path,
    source_documents: dict[str, dict],
    source_order: tuple[str, ...],
    page_id_prefixes: dict[str, str],
    corpus_id: str,
    status: str,
) -> list[dict]:
    try:
        import fitz
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("PyMuPDF is required to rebuild PDF pages") from error

    rows = []
    for source_id in source_order:
        source = source_documents[source_id]
        doc = fitz.open(repo_root / source["path"])
        for page_number in range(1, doc.page_count + 1):
            rows.append(
                {
                    "corpus_id": corpus_id,
                    "page_count": doc.page_count,
                    "page_id": f"{page_id_prefixes[source_id]}::{page_number:04d}",
                    "page_number": page_number,
                    "source_document_id": source_id,
                    "source_pdf_path": source["path"],
                    "source_sha256": source["sha256"],
                    "status": status,
                    "text": pdf_page_text(doc, page_number),
                }
            )
    return rows


def pdf_page_text(doc, page_number: int) -> str:
    text = doc[page_number - 1].get_text("text").replace("\xa0", " ").replace("\xad", "")
    return "\n".join(normalize_semantic_text(re.sub(r" {2,}", " ", line)) for line in text.split("\n"))
