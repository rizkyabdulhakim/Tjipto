from __future__ import annotations

from pathlib import Path
import re

from tjipto.corpora.uud.specs import PAGE_ID_PREFIXES, PAGE_SOURCE_ORDER


def build_pages(repo_root: Path, source_documents: dict[str, dict]) -> list[dict]:
    try:
        import fitz
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("PyMuPDF is required to rebuild UUD pages") from error

    rows = []
    for source_id in PAGE_SOURCE_ORDER:
        source = source_documents[source_id]
        doc = fitz.open(repo_root / source["path"])
        for page_number in range(1, doc.page_count + 1):
            rows.append({
                "corpus_id": "uud",
                "page_count": doc.page_count,
                "page_id": f"{PAGE_ID_PREFIXES[source_id]}::{page_number:04d}",
                "page_number": page_number,
                "source_document_id": source_id,
                "source_pdf_path": source["path"],
                "source_sha256": source["sha256"],
                "status": "finalized_text_boundary",
                "text": _page_text(doc, page_number),
            })
    return rows


def _page_text(doc, page_number: int) -> str:
    text = doc[page_number - 1].get_text("text").replace("\xa0", " ").replace("\xad", "")
    return "\n".join(re.sub(r" {2,}", " ", line) for line in text.split("\n"))
