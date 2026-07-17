from __future__ import annotations

import re
from typing import cast
import unicodedata

STABLE_EXTRACTOR_VERSION = "pymupdf_words"


def build_word_bbox_rows(
    *,
    doc,
    corpus_id: str,
    source_document_id: str,
    source_meta: dict,
    bbox_id_prefix: str,
    extractor: str = "pymupdf",
    bbox_source: str = "pymupdf_words",
    extractor_version: str | None = STABLE_EXTRACTOR_VERSION,
) -> list[dict]:
    rows: list[dict] = []
    for page_number in range(1, doc.page_count + 1):
        page = doc[page_number - 1]
        rect = page.rect
        page_words = page.get_text("words", sort=True)
        page_characters = _page_characters(page)
        character_cursor = 0
        for word_index, word in enumerate(page_words):
            x0, y0, x1, y1, text, block_index, line_index, word_no = word
            normalized_text = normalize_text(text)
            if not compact_text(normalized_text):
                continue
            character_rows, character_cursor = _match_word_characters(
                text, page_characters, character_cursor, source_document_id, page_number, word_index
            )
            rows.append(
                {
                    "word_bbox_id": f"{bbox_id_prefix}::{source_document_id}::{page_number:04d}::{word_index:05d}",
                    "corpus_id": corpus_id,
                    "source_document_id": source_document_id,
                    "source_pdf": source_meta["filename"],
                    "source_pdf_path": source_meta["path"],
                    "source_sha256": source_meta["sha256"],
                    "page_number": page_number,
                    "page_width": rect.width,
                    "page_height": rect.height,
                    "coordinate_space": "pdf_user_space",
                    "coordinate_origin": "top_left",
                    "page_rotation": int(page.rotation),
                    "page_box_basis": "media_box",
                    "transform_version": "pymupdf_top_left_v1",
                    "word_index": word_index,
                    "block_index": int(block_index),
                    "line_index": int(line_index),
                    "word_no": int(word_no),
                    "text": text,
                    "normalized_text": normalized_text,
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "bbox_source": bbox_source,
                    "extractor": extractor,
                    "extractor_version": extractor_version or STABLE_EXTRACTOR_VERSION,
                    "characters": character_rows,
                }
            )
    return rows


def _page_characters(page) -> list[dict]:
    """Return actual PDF character geometry; never approximates from word boxes."""
    rows: list[dict] = []
    raw = page.get_text("rawdict", sort=True)
    for block in raw.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                rows.extend(span.get("chars", []))
    return rows


def _match_word_characters(
    text: str,
    characters: list[dict],
    cursor: int,
    source_document_id: str,
    page_number: int,
    word_index: int,
) -> tuple[list[dict], int]:
    target = compact_text(text)
    if not target:
        return [], cursor
    compacted = ""
    matched: list[dict] = []
    index = cursor
    while index < len(characters) and len(compacted) < len(target):
        char = characters[index]
        value = str(char.get("c") or "")
        matched.append(char)
        compacted = compact_text(compacted + value)
        index += 1
    if compacted != target:
        return [], cursor
    result = []
    char_index = 0
    for char in matched:
        value = str(char.get("c") or "")
        if not compact_text(value):
            continue
        x0, y0, x1, y1 = char.get("bbox", (None, None, None, None))
        if not all(isinstance(item, (int, float)) for item in (x0, y0, x1, y1)):
            return [], cursor
        result.append(
            {
                "character_bbox_id": f"uud_character_bbox::{source_document_id}::{page_number:04d}::{word_index:05d}::{char_index:03d}",
                "text": value,
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
            }
        )
        char_index += 1
    return result, index


def word_rows_by_page(word_bboxes: list[dict]) -> dict[tuple[str, int], list[dict]]:
    rows_by_page: dict[tuple[str, int], list[dict]] = {}
    for row in word_bboxes:
        rows_by_page.setdefault((row["source_document_id"], row["page_number"]), []).append(row)
    return rows_by_page


def align_text_to_word_bboxes(
    *,
    text: str | None,
    source_document_id: str,
    page_numbers: list[int] | tuple[int, ...],
    words_by_page: dict[tuple[str, int], list[dict]],
    reference_bbox: dict | None = None,
) -> dict | None:
    target = compact_text(text)
    if not target:
        return None
    candidates: list[dict] = []
    for page_number in page_numbers:
        page_words = [row for row in words_by_page.get((source_document_id, page_number), []) if row.get("normalized_text")]
        for start in range(len(page_words)):
            joined = ""
            matched: list[dict] = []
            for row in page_words[start:]:
                matched.append(row)
                joined = compact_text(f"{joined} {row.get('text', '')}")
                if joined == target:
                    union_bbox = union_word_bbox(matched)
                    candidates.append(
                        {
                            "page_number": page_number,
                            "matched_word_bbox_ids": [item["word_bbox_id"] for item in matched],
                            "union_bbox": union_bbox,
                            "distance_to_existing_span_bbox": bbox_center_distance(union_bbox, reference_bbox),
                        }
                    )
                    break
                if len(joined) > len(target) + 24 or not target.startswith(joined):
                    break
    if not candidates:
        return None
    candidates.sort(
        key=lambda row: (
            row["distance_to_existing_span_bbox"],
            row["page_number"],
            row["matched_word_bbox_ids"][0],
        )
    )
    chosen = candidates[0]
    return {
        "text_span_id": None,
        "matched_word_bbox_ids": chosen["matched_word_bbox_ids"],
        "match_method": "contiguous_compact_word_sequence",
        "match_confidence": "exact_compact_sequence" if len(candidates) == 1 else "exact_compact_sequence_nearest_reference_bbox",
        "normalized_text_match": True,
        "union_bbox": chosen["union_bbox"],
        "candidate_count": len(candidates),
        "distance_to_existing_span_bbox": chosen["distance_to_existing_span_bbox"],
    }


def union_word_bbox(rows: list[dict]) -> dict:
    return {
        "source_document_id": rows[0]["source_document_id"],
        "source_pdf": rows[0]["source_pdf"],
        "source_pdf_path": rows[0]["source_pdf_path"],
        "source_sha256": rows[0]["source_sha256"],
        "page_number": rows[0]["page_number"],
        "page_width": rows[0].get("page_width"),
        "page_height": rows[0].get("page_height"),
        "text": " ".join(row["text"] for row in rows).strip(),
        "normalized_text": normalize_text(" ".join(row["text"] for row in rows)),
        "x0": min(row["x0"] for row in rows),
        "y0": min(row["y0"] for row in rows),
        "x1": max(row["x1"] for row in rows),
        "y1": max(row["y1"] for row in rows),
    }


def bbox_center_distance(bbox: dict, reference_bbox: dict | None) -> float:
    if not reference_bbox:
        return 0.0
    x0 = reference_bbox.get("x0")
    y0 = reference_bbox.get("y0")
    x1 = reference_bbox.get("x1")
    y1 = reference_bbox.get("y1")
    if not all(isinstance(value, (int, float)) for value in (x0, y0, x1, y1)):
        return 0.0
    ref_x0 = cast(float, x0)
    ref_y0 = cast(float, y0)
    ref_x1 = cast(float, x1)
    ref_y1 = cast(float, y1)
    return (
        ((bbox["x0"] + bbox["x1"]) / 2 - (ref_x0 + ref_x1) / 2) ** 2 + ((bbox["y0"] + bbox["y1"]) / 2 - (ref_y0 + ref_y1) / 2) ** 2
    ) ** 0.5


def normalize_text(text: str | None) -> str:
    text = unicodedata.normalize("NFKC", text or "").replace("\u00ad", "").replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def compact_text(text: str | None) -> str:
    normalized = normalize_text(text).casefold()
    return "".join(re.findall(r"\w+", normalized))
