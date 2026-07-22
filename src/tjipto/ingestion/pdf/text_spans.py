from __future__ import annotations

from hashlib import sha256
import unicodedata


NORMALIZATION_CONTRACT = "NFKC;NBSP=SPACE;SOFT_HYPHEN=REMOVE;LINE_SEPARATOR=LF"
class RawSegmentGeometryError(ValueError):
    code = "raw_segment_geometry_unavailable"


def build_pdf_text_spans(
    *,
    source_documents: dict[str, dict],
    pdf_lines: dict[str, dict[int, list[dict]]],
    corpus_id: str,
    text_span_id_prefix: str,
    raw_source_spans: list[dict] | None = None,
    word_bboxes: list[dict] | None = None,
    semantic_normalizer=None,
    source_segmenter=None,
    normalization_contract: str = NORMALIZATION_CONTRACT,
) -> list[dict]:
    rows: list[dict] = []
    for source_id, pages in sorted(pdf_lines.items()):
        source = source_documents[source_id]
        source_role = source["source_role"]
        temporal_context = source.get("temporal_context", source_role)
        for page_number, lines in sorted(pages.items()):
            normalized_lines = [(semantic_normalizer or _normalize)(line["text"]) for line in lines]
            if raw_source_spans is not None:
                _append_raw_source_spans(
                    raw_source_spans,
                    source_id=source_id,
                    source=source,
                    page_number=page_number,
                    lines=lines,
                    corpus_id=corpus_id,
                    word_bboxes=word_bboxes or [],
                    source_segmenter=source_segmenter or _default_source_segmenter,
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
                        "normalization_contract": normalization_contract,
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
    return value


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
    source_segmenter,
) -> None:
    raw_stream = "\n".join(str(line.get("text") or "") for line in lines)
    raw_hash = sha256(raw_stream.encode("utf-8")).hexdigest()
    cursor = 0
    extraction_order = 0
    page_words = [word for word in word_bboxes if word.get("source_document_id") == source_id and word.get("page_number") == page_number]
    for line_index, line in enumerate(lines):
        raw_text = str(line.get("text") or "")
        if raw_text:
            segments = source_segmenter(raw_text)
            for segment_index, segment in enumerate(segments):
                start, end = segment["start"], segment["end"]
                if start == end:
                    continue
                segment_text = raw_text[start:end]
                geometry = _segment_geometry(line, raw_text, start, end, page_words, source_id, page_number, line_index)
                rows.append({
                    "raw_source_span_id": f"{corpus_id}::raw::{source_id}::{page_number:04d}::{line_index:04d}::{segment_index:02d}",
                    "source_document_id": source_id,
                    "source_sha256": source["sha256"],
                    "source_role": source["source_role"],
                    "page_number": page_number,
                    "line_index": line_index,
                    "segment_order": segment_index,
                    "extraction_order": extraction_order,
                    "raw_text": segment_text,
                    "raw_quote": segment_text,
                    "raw_stream_id": f"{corpus_id}::raw_page_text::{source_id}::{page_number:04d}",
                    "raw_stream_sha256": raw_hash,
                    "raw_text_start": cursor + start,
                    "raw_text_end": cursor + end,
                    "x0": geometry["x0"], "y0": geometry["y0"], "x1": geometry["x1"], "y1": geometry["y1"],
                    "raw_geometry_method": geometry["method"],
                    "classification": segment["classification"],
                    "legal_text": segment["legal_text"],
                    "citation_eligible": segment["citation_eligible"],
                    "relevant_quote_eligible": segment["relevant_quote_eligible"],
                    "default_highlight_eligible": segment["default_highlight_eligible"],
                    "normalization_actions": segment["normalization_actions"],
                    "disposition_reason": segment["disposition_reason"],
                })
                extraction_order += 1
        cursor += len(raw_text) + (1 if line_index < len(lines) - 1 else 0)


def _segment_geometry(line: dict, raw_text: str, start: int, end: int, page_words: list[dict], source_id: str, page_number: int, line_index: int) -> dict:
    candidates = []
    word_cursor = 0
    line_words = [word for word in page_words if word.get("y1", 0) >= line.get("y0", 0) and word.get("y0", 0) <= line.get("y1", 0) and word.get("x1", 0) >= line.get("x0", 0) and word.get("x0", 0) <= line.get("x1", 0)]
    exact_words = [word for word in line_words if str(word.get("text") or "").strip() == raw_text[start:end]]
    if exact_words:
        word = min(exact_words, key=lambda item: (item.get("x0", 0), item.get("y0", 0)))
        chars = word.get("characters") or []
        if chars:
            return {"x0": min(char["x0"] for char in chars), "y0": min(char["y0"] for char in chars), "x1": max(char["x1"] for char in chars), "y1": max(char["y1"] for char in chars), "method": "pdf_character_bbox"}
        return {"x0": word["x0"], "y0": word["y0"], "x1": word["x1"], "y1": word["y1"], "method": "pdf_word_bbox"}
    for word in sorted(line_words, key=lambda item: (item.get("x0", 0), item.get("y0", 0))):
        word_text = str(word.get("text") or "")
        while word_cursor < len(raw_text) and raw_text[word_cursor].isspace():
            word_cursor += 1
        if not raw_text.startswith(word_text, word_cursor):
            continue
        word_start = word_cursor
        word_cursor += len(word_text)
        if word_start >= end or word_start + len(word_text) <= start:
            continue
        for char in word.get("characters") or []:
            char_start = word_start + int(char.get("char_start", 0))
            char_end = word_start + int(char.get("char_end", 0))
            if char_start < end and char_end > start:
                candidates.append(char)
        if not word.get("characters") and start <= word_start and word_start + len(word_text) <= end:
            return {"x0": word["x0"], "y0": word["y0"], "x1": word["x1"], "y1": word["y1"], "method": "pdf_word_bbox"}
    if candidates:
        return {"x0": min(char["x0"] for char in candidates), "y0": min(char["y0"] for char in candidates), "x1": max(char["x1"] for char in candidates), "y1": max(char["y1"] for char in candidates), "method": "pdf_character_bbox"}
    raise RawSegmentGeometryError(f"raw_segment:{source_id}:{page_number}:{line_index}:{start}:{end}:{raw_text!r}:{line!r}")


def _default_source_segmenter(raw_text: str) -> list[dict]:
    return [{
        "start": 0, "end": len(raw_text), "classification": "source_text", "legal_text": True,
        "citation_eligible": True, "relevant_quote_eligible": True, "default_highlight_eligible": True,
        "normalization_actions": [], "disposition_reason": "semantic_projection",
    }]
