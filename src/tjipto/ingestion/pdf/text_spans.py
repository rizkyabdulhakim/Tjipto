from __future__ import annotations

from hashlib import sha256
import re
import unicodedata


NORMALIZATION_CONTRACT = "NFKC;NBSP=SPACE;SOFT_HYPHEN=REMOVE;FOOTNOTE_MARKER=REMOVE;LINE_SEPARATOR=LF"
FOOTNOTE_MARKER_RE = re.compile(r"\*{1,4}(?:/\*{1,4})?\)")


def build_pdf_text_spans(
    *,
    source_documents: dict[str, dict],
    pdf_lines: dict[str, dict[int, list[dict]]],
    corpus_id: str,
    text_span_id_prefix: str,
    raw_source_spans: list[dict] | None = None,
    word_bboxes: list[dict] | None = None,
) -> list[dict]:
    rows: list[dict] = []
    for source_id, pages in sorted(pdf_lines.items()):
        source = source_documents[source_id]
        source_role = source["source_role"]
        temporal_context = source.get("temporal_context", source_role)
        for page_number, lines in sorted(pages.items()):
            normalized_lines = [_normalize(line["text"]) for line in lines]
            if raw_source_spans is not None:
                _append_raw_source_spans(
                    raw_source_spans,
                    source_id=source_id,
                    source=source,
                    page_number=page_number,
                    lines=lines,
                    corpus_id=corpus_id,
                    word_bboxes=word_bboxes or [],
                )
            semantic_lines = [text for text in normalized_lines if text]
            stream = "\n".join(semantic_lines)
            stream_hash = sha256(stream.encode("utf-8")).hexdigest()
            stream_id = f"{corpus_id}::page_text::{source_id}::{page_number:04d}"
            cursor = 0
            semantic_index = 0
            for index, line in enumerate(lines):
                text = normalized_lines[index]
                if not text:
                    continue
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
                        "text_span_id": f"{text_span_id_prefix}::{source_role}::{page_number:04d}::{semantic_index:04d}",
                        "viewer_highlightable": False,
                        "x0": line["x0"],
                        "x1": line["x1"],
                        "y0": line["y0"],
                        "y1": line["y1"],
                    }
                )
                cursor = end + (1 if semantic_index < len(semantic_lines) - 1 else 0)
                semantic_index += 1
    return rows


def _normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "").replace("\xa0", " ").replace("\u00ad", "")
    value = FOOTNOTE_MARKER_RE.sub("", value)
    return value if not value.startswith("Dihapus.") else value.split("****", 1)[0].strip()


def normalize_semantic_text(value: str) -> str:
    return _normalize(value)


def _append_raw_source_spans(
    rows: list[dict],
    *,
    source_id: str,
    source: dict,
    page_number: int,
    lines: list[dict],
    corpus_id: str,
    word_bboxes: list[dict],
) -> None:
    raw_stream = "\n".join(str(line.get("text") or "") for line in lines)
    raw_hash = sha256(raw_stream.encode("utf-8")).hexdigest()
    cursor = 0
    extraction_order = 0
    page_words = [word for word in word_bboxes if word.get("source_document_id") == source_id and word.get("page_number") == page_number]
    for line_index, line in enumerate(lines):
        raw_text = str(line.get("text") or "")
        if raw_text:
            start = cursor
            end = start + len(raw_text)
            marker_matches = list(FOOTNOTE_MARKER_RE.finditer(raw_text))
            rows.append({
                "raw_source_span_id": f"{corpus_id}::raw_source::{source_id}::{page_number:04d}::{line_index:04d}",
                "source_document_id": source_id,
                "source_sha256": source["sha256"],
                "source_role": source["source_role"],
                "page_number": page_number,
                "extraction_order": extraction_order,
                "raw_text": raw_text,
                "raw_quote": raw_text,
                "raw_stream_id": f"{corpus_id}::raw_page_text::{source_id}::{page_number:04d}",
                "raw_stream_sha256": raw_hash,
                "raw_text_start": start,
                "raw_text_end": end,
                "x0": line["x0"], "y0": line["y0"], "x1": line["x1"], "y1": line["y1"],
                "classification": "source_text",
                "legal_text": not marker_matches,
                "citation_eligible": not marker_matches,
                "relevant_quote_eligible": not marker_matches,
                "default_highlight_eligible": not marker_matches,
                "normalization_actions": [
                    {"action": "remove_source_annotation_marker", "start": match.start(), "end": match.end(), "quote": match.group(0)}
                    for match in marker_matches
                ],
                "disposition_reason": "semantic_projection" if not marker_matches else "source_annotation_marker_removed",
            })
            extraction_order += 1
            for marker_index, match in enumerate(marker_matches):
                geometry = _marker_geometry(line, raw_text, match, page_words)
                rows.append({
                    "raw_source_span_id": f"{corpus_id}::raw_marker::{source_id}::{page_number:04d}::{line_index:04d}::{marker_index:02d}",
                    "source_document_id": source_id,
                    "source_sha256": source["sha256"],
                    "source_role": source["source_role"],
                    "page_number": page_number,
                    "extraction_order": extraction_order,
                    "raw_text": match.group(0),
                    "raw_quote": match.group(0),
                    "raw_stream_id": f"{corpus_id}::raw_page_text::{source_id}::{page_number:04d}",
                    "raw_stream_sha256": raw_hash,
                    "raw_text_start": cursor + match.start(),
                    "raw_text_end": cursor + match.end(),
                    "x0": geometry["x0"], "y0": geometry["y0"], "x1": geometry["x1"], "y1": geometry["y1"],
                    "raw_geometry_method": geometry["method"],
                    "classification": "source_annotation_marker",
                    "legal_text": False,
                    "citation_eligible": False,
                    "relevant_quote_eligible": False,
                    "default_highlight_eligible": False,
                    "normalization_actions": [{"action": "remove_source_annotation_marker", "start": match.start(), "end": match.end(), "quote": match.group(0)}],
                    "disposition_reason": "source_annotation_marker",
                })
                extraction_order += 1
        cursor += len(raw_text) + (1 if line_index < len(lines) - 1 else 0)


def _marker_geometry(line: dict, raw_text: str, match: re.Match[str], page_words: list[dict]) -> dict:
    candidates = []
    marker_start, marker_end = match.span()
    for word in page_words:
        word_text = str(word.get("text") or "")
        word_start = raw_text.find(word_text)
        if word_start < 0 or not (word.get("y1", 0) >= line.get("y0", 0) and word.get("y0", 0) <= line.get("y1", 0)):
            continue
        if word_start >= marker_end or word_start + len(word_text) <= marker_start:
            continue
        for char in word.get("characters") or []:
            char_start = word_start + int(char.get("char_start", 0))
            char_end = word_start + int(char.get("char_end", 0))
            if char_start < marker_end and char_end > marker_start:
                candidates.append(char)
    if candidates:
        return {"x0": min(char["x0"] for char in candidates), "y0": min(char["y0"] for char in candidates), "x1": max(char["x1"] for char in candidates), "y1": max(char["y1"] for char in candidates), "method": "pdf_character_bbox"}
    return {"x0": line["x0"], "y0": line["y0"], "x1": line["x1"], "y1": line["y1"], "method": "source_line_bbox"}
