from __future__ import annotations

from hashlib import sha256
import unicodedata


NORMALIZATION_CONTRACT = "NFKC;NBSP=SPACE;SOFT_HYPHEN=REMOVE;FOOTNOTE_MARKER=REMOVE;LINE_SEPARATOR=LF"


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
            normalized_lines = [_normalize(line["text"]) for line in lines]
            stream = "\n".join(normalized_lines)
            stream_hash = sha256(stream.encode("utf-8")).hexdigest()
            stream_id = f"{corpus_id}::page_text::{source_id}::{page_number:04d}"
            cursor = 0
            for index, line in enumerate(lines):
                text = normalized_lines[index]
                start = cursor
                end = start + len(text)
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
                        "text": text,
                        "exact_quote": text,
                        "stream_id": stream_id,
                        "page_text_hash": stream_hash,
                        "normalization_contract": NORMALIZATION_CONTRACT,
                        "unicode_offset_basis": "python_unicode_code_points",
                        "text_start": start,
                        "text_end": end,
                        "text_prefix": stream[max(0, start - 32):start],
                        "text_suffix": stream[end:end + 32],
                        "text_span_id": f"{text_span_id_prefix}::{source_role}::{page_number:04d}::{index:04d}",
                        "viewer_highlightable": False,
                        "x0": line["x0"],
                        "x1": line["x1"],
                        "y0": line["y0"],
                        "y1": line["y1"],
                    }
                )
                cursor = end + (1 if index < len(lines) - 1 else 0)
    return rows


def _normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "").replace("\xa0", " ").replace("\u00ad", "")
    return value if not value.startswith("Dihapus.") else value.split("****", 1)[0].strip()
