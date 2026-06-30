from __future__ import annotations

from tjipto.ingestion.pdf.text_spans import build_pdf_text_spans


def build_page_text_spans(*, source_documents: dict[str, dict], pdf_lines: dict[str, dict[int, list[dict]]]) -> list[dict]:
    return build_pdf_text_spans(
        source_documents=source_documents,
        pdf_lines=pdf_lines,
        corpus_id="uud",
        text_span_id_prefix="uud_text_span",
    )
