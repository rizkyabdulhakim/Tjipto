from __future__ import annotations


def build_page_text_spans(*, source_documents: dict[str, dict], pdf_lines: dict[str, dict[int, list[dict]]]) -> list[dict]:
    rows: list[dict] = []
    for source_id, pages in sorted(pdf_lines.items()):
        source = source_documents[source_id]
        for page_number, lines in sorted(pages.items()):
            for index, line in enumerate(lines):
                rows.append({
                    "bbox_precision": "exact",
                    "corpus_id": "uud",
                    "page_number": page_number,
                    "source_document_id": source_id,
                    "source_pdf": source["filename"],
                    "source_pdf_path": source["path"],
                    "source_role": source_id.split("::", 1)[1],
                    "source_sha256": source["sha256"],
                    "status": "accepted_text_span",
                    "text": line["text"],
                    "text_span_id": f"uud_text_span::{source_id.split('::', 1)[1]}::{page_number:04d}::{index:04d}",
                    "viewer_highlightable": False,
                    "x0": line["x0"],
                    "x1": line["x1"],
                    "y0": line["y0"],
                    "y1": line["y1"],
                })
    return rows
