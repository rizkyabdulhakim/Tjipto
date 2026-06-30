from __future__ import annotations

from pathlib import Path

from tjipto.ingestion.pdf.pages import build_pdf_pages
from tjipto.corpora.uud.specs import PAGE_ID_PREFIXES, PAGE_SOURCE_ORDER


def build_pages(repo_root: Path, source_documents: dict[str, dict]) -> list[dict]:
    return build_pdf_pages(
        repo_root=repo_root,
        source_documents=source_documents,
        source_order=PAGE_SOURCE_ORDER,
        page_id_prefixes=PAGE_ID_PREFIXES,
        corpus_id="uud",
        status="finalized_text_boundary",
    )
