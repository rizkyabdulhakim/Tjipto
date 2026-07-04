from __future__ import annotations


def build_pdf_text_spans(
    *,
    source_documents: dict[str, dict],
    pdf_lines: dict[str, dict[int, list[dict]]],
    corpus_id: str,
    text_span_id_prefix: str,
) -> list[dict]:
    rows: list[dict] = []
    for source_id, pages in sorted(pdf_lines.items()):
        source = source_documents[source_id]
        source_role = source["source_role"]
        temporal_context = source.get("temporal_context", source_role)
        for page_number, lines in sorted(pages.items()):
            for index, line in enumerate(lines):
                rows.append(
                    {
                        "bbox_precision": "exact",
                        "corpus_id": corpus_id,
                        "page_number": page_number,
                        "source_document_id": source_id,
                        "source_pdf": source["filename"],
                        "source_pdf_path": source["path"],
                        "source_role": source_role,
                        "source_sha256": source["sha256"],
                        "status": "accepted_text_span",
                        "temporal_context": temporal_context,
                        "text": line["text"],
                        "text_span_id": f"{text_span_id_prefix}::{source_role}::{page_number:04d}::{index:04d}",
                        "viewer_highlightable": False,
                        "x0": line["x0"],
                        "x1": line["x1"],
                        "y0": line["y0"],
                        "y1": line["y1"],
                    }
                )
    return rows
