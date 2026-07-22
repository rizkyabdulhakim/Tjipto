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
    for line_index, line in enumerate(lines):
        source_line_index = int(line.get("line_index", line_index))
        raw_text = str(line.get("text") or "")
        if raw_text:
            segments = source_segmenter(raw_text)
            for segment_index, segment in enumerate(segments):
                start, end = segment["start"], segment["end"]
                if start == end:
                    continue
                segment_text = raw_text[start:end]
                geometry = _segment_geometry(line, raw_text, start, end, source_id, page_number, line_index)
                rows.append({
                    "raw_source_span_id": f"{corpus_id}::raw::{source_id}::{page_number:04d}::{int(line.get('block_index', 0)):04d}::{source_line_index:04d}::{segment_index:02d}",
                    "source_document_id": source_id,
                    "source_sha256": source["sha256"],
                    "source_role": source["source_role"],
                    "page_number": page_number,
                    "line_index": source_line_index,
                    "block_index": geometry["block_index"],
                    "span_index": geometry["span_index"],
                    "character_start": geometry["character_start"],
                    "character_end": geometry["character_end"],
                    "character_ids": geometry["character_ids"],
                    "character_texts": geometry["character_texts"],
                    "character_bboxes": geometry["character_bboxes"],
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


def _segment_geometry(line: dict, raw_text: str, start: int, end: int, source_id: str, page_number: int, line_index: int) -> dict:
    """Resolve a segment only through its rawdict character lineage.

    A word rectangle is not an acceptable substitute: it can include adjacent
    punctuation, annotations, or whitespace.  Any missing character lineage
    is a construction error and therefore fails the staged artifact build.
    """
    characters = list(line.get("characters") or [])
    selected = [
        character
        for character in characters
        if int(character.get("char_start", -1)) < end and int(character.get("char_end", -1)) > start
    ]
    selected_text = "".join(str(character.get("text") or "") for character in selected)
    expected = raw_text[start:end]
    if not selected or selected_text != expected:
        raise RawSegmentGeometryError(
            f"raw_segment_character_mismatch:{source_id}:{page_number}:{line_index}:{start}:{end}:{expected!r}:{selected_text!r}"
        )
    if any(len(character.get("bbox") or ()) != 4 for character in selected):
        raise RawSegmentGeometryError(f"raw_segment_character_bbox_missing:{source_id}:{page_number}:{line_index}:{start}:{end}")
    boxes = [tuple(character["bbox"]) for character in selected]
    return {
        "x0": min(box[0] for box in boxes),
        "y0": min(box[1] for box in boxes),
        "x1": max(box[2] for box in boxes),
        "y1": max(box[3] for box in boxes),
        "method": "pdf_rawdict_character_bbox",
        "block_index": selected[0]["block_index"],
        "span_index": selected[0]["span_index"],
        "character_start": selected[0]["char_start"],
        "character_end": selected[-1]["char_end"],
        "character_ids": [character["character_id"] for character in selected],
        "character_texts": [character["text"] for character in selected],
        "character_bboxes": [
            {"character_id": character["character_id"], "x0": character["x0"], "y0": character["y0"], "x1": character["x1"], "y1": character["y1"]}
            for character in selected
        ],
    }


def _default_source_segmenter(raw_text: str) -> list[dict]:
    return [{
        "start": 0, "end": len(raw_text), "classification": "source_text", "legal_text": True,
        "citation_eligible": True, "relevant_quote_eligible": True, "default_highlight_eligible": True,
        "normalization_actions": [], "disposition_reason": "semantic_projection",
    }]
