from __future__ import annotations

from tjipto.ingestion.pdf.text_spans import build_pdf_text_spans
from tjipto.corpora.uud.source_policy import NORMALIZATION_CONTRACT, normalize_semantic_text, segment_source_line


def build_page_text_spans(*, source_documents: dict[str, dict], pdf_lines: dict[str, dict[int, list[dict]]], raw_source_spans: list[dict] | None = None, word_bboxes: list[dict] | None = None) -> list[dict]:
    return build_pdf_text_spans(
        source_documents=source_documents,
        pdf_lines=pdf_lines,
        corpus_id="uud",
        text_span_id_prefix="uud_text_span",
        raw_source_spans=raw_source_spans,
        word_bboxes=word_bboxes,
        semantic_normalizer=normalize_semantic_text,
        source_segmenter=segment_source_line,
        normalization_contract=NORMALIZATION_CONTRACT,
    )
