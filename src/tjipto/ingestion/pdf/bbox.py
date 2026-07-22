from __future__ import annotations

import re
import unicodedata
from hashlib import sha256
import json
from typing import Any, cast

from tjipto.contracts.coordinates import coordinate_metadata


def build_text_bbox_rows(
    *,
    evidence_id: str,
    source_meta: dict,
    source_id: str,
    text: str,
    page_start: int,
    page_end: int,
    line_entries: dict[int, list[dict]],
    corpus_id: str,
    bbox_id_prefix: str,
) -> list[dict]:
    matched = _matching_sequence(
        [
            {"page_number": page_number, **candidate}
            for page_number in range(page_start, page_end + 1)
            for candidate in line_entries.get(page_number, [])
        ],
        text,
    )
    if not matched:
        expected = [_compact(line) for line in text.splitlines() if line.strip()]
        target_index = 0
        matched = []
        for page_number in range(page_start, page_end + 1):
            candidates = line_entries.get(page_number, [])
            for candidate in candidates:
                if target_index >= len(expected):
                    break
                if _compact(candidate["text"]) != expected[target_index]:
                    continue
                matched.append(
                    {
                        "page_number": page_number,
                        **candidate,
                    }
                )
                target_index += 1
            if target_index >= len(expected):
                break
        if target_index < len(expected):
            matched = []
    if not matched:
        return _fallback_bbox_rows(
            evidence_id=evidence_id,
            source_meta=source_meta,
            source_id=source_id,
            text=text,
            page_start=page_start,
            page_end=page_end,
            line_entries=line_entries,
            corpus_id=corpus_id,
            bbox_id_prefix=bbox_id_prefix,
        )
    return [
        {
            "bbox_id": _geometry_id(
                source_id=source_id,
                source_sha256=source_meta["sha256"],
                page_number=row["page_number"],
                text=row["text"],
                coordinates=row,
            ),
            "bbox_precision": "exact",
            "corpus_id": corpus_id,
            "page_number": row["page_number"],
            "source_document_id": source_id,
            "source_pdf": source_meta["filename"],
            "source_pdf_path": source_meta["path"],
            "source_sha256": source_meta["sha256"],
            "status": "accepted",
            "text": row["text"],
            "viewer_highlightable": True,
            "x0": row["x0"],
            "x1": row["x1"],
            "y0": row["y0"],
            "y1": row["y1"],
            **coordinate_metadata(row, highlightable=True),
        }
        for index, row in enumerate(matched)
    ]


def _matching_sequence(rows: list[dict], text: str) -> list[dict]:
    target = _compact(text)
    if not target:
        return []
    for start in range(len(rows)):
        selected = []
        joined = ""
        for row in rows[start:]:
            selected.append(row)
            joined = _compact(f"{joined} {row.get('text', '')}")
            if joined == target:
                return selected
            if len(joined) > len(target) + 80 or not target.startswith(joined):
                break
    return []


def pdf_lines(doc) -> dict[int, list[dict]]:
    pages: dict[int, list[dict]] = {}
    for page_number in range(1, doc.page_count + 1):
        page = doc[page_number - 1]
        entries = []
        # rawdict is the only extraction mode that preserves the character
        # lineage needed by source-segment provenance.  The dict/text modes
        # flatten spans and make an exact sub-line geometry impossible.
        # Preserve the PDF extraction order used by the corpus builders.  The
        # rawdict payload still supplies character lineage; sorting here would
        # silently reorder repeated source occurrences.
        raw_payload = cast(dict[str, Any], page.get_text("rawdict"))
        legacy_payload = cast(dict[str, Any], page.get_text("dict"))
        for block_index, block in enumerate(raw_payload.get("blocks", [])):
            if block.get("type") != 0:
                continue
            for line_index, line in enumerate(block.get("lines", [])):
                raw_characters: list[dict[str, Any]] = []
                raw_offset = 0
                for span_index, span in enumerate(line.get("spans", [])):
                    for character_index, character in enumerate(span.get("chars", [])):
                        value = str(character.get("c") or "")
                        if not value:
                            continue
                        raw_characters.append(
                            {
                                "character_id": f"pdf_character::{page_number:04d}::{block_index:04d}::{line_index:04d}::{span_index:04d}::{character_index:04d}",
                                "block_index": block_index,
                                "line_index": line_index,
                                "span_index": span_index,
                                "character_index": character_index,
                                "char_start": raw_offset,
                                "char_end": raw_offset + len(value),
                                "text": value,
                                "bbox": tuple(character.get("bbox") or ()),
                            }
                        )
                        raw_offset += len(value)
                raw_text = "".join(character["text"] for character in raw_characters)
                left_trim = len(raw_text) - len(raw_text.lstrip())
                right_trim = len(raw_text.rstrip())
                text = raw_text[left_trim:right_trim]
                if not text:
                    continue
                characters = []
                for character in raw_characters:
                    if character["char_end"] <= left_trim or character["char_start"] >= right_trim:
                        continue
                    if len(character["bbox"]) != 4:
                        raise ValueError(f"missing_character_bbox:{page_number}:{block_index}:{line_index}:{character['character_id']}")
                    characters.append(
                        {
                            **character,
                            "char_start": character["char_start"] - left_trim,
                            "char_end": character["char_end"] - left_trim,
                            "x0": character["bbox"][0],
                            "y0": character["bbox"][1],
                            "x1": character["bbox"][2],
                            "y1": character["bbox"][3],
                        }
                    )
                if not characters or "".join(character["text"] for character in characters) != text:
                    raise ValueError(f"raw_line_character_mismatch:{page_number}:{block_index}:{line_index}")
                legacy_line = (
                    legacy_payload.get("blocks", [])[block_index].get("lines", [])[line_index]
                    if block_index < len(legacy_payload.get("blocks", []))
                    and line_index < len(legacy_payload.get("blocks", [])[block_index].get("lines", []))
                    else {}
                )
                legacy_spans = legacy_line.get("spans", [])
                legacy_text = "".join(str(span.get("text") or "") for span in legacy_spans).strip()
                if legacy_text != text:
                    raise ValueError(f"legacy_raw_line_mismatch:{page_number}:{block_index}:{line_index}")
                legacy_boxes = [span.get("bbox") for span in legacy_spans if len(span.get("bbox") or ()) == 4]
                x0 = min(box[0] for box in legacy_boxes) if legacy_boxes else min(character["x0"] for character in characters)
                y0 = min(box[1] for box in legacy_boxes) if legacy_boxes else min(character["y0"] for character in characters)
                x1 = max(box[2] for box in legacy_boxes) if legacy_boxes else max(character["x1"] for character in characters)
                y1 = max(box[3] for box in legacy_boxes) if legacy_boxes else max(character["y1"] for character in characters)
                entries.append({"text": text, "x0": x0, "y0": y0, "x1": x1, "y1": y1, "width": page.rect.width, "height": page.rect.height, "block_index": block_index, "line_index": line_index, "characters": characters})
        pages[page_number] = entries
    return pages


def aggregate_bbox_precision(rows: list[dict]) -> str:
    precisions = {row.get("bbox_precision") for row in rows}
    if precisions == {"exact"}:
        return "exact"
    if "page_grounded_only" in precisions:
        return "page_grounded_only"
    return "coarse"


def bbox_precision_counts(bbox_rows: list[dict]) -> dict[str, int]:
    return {
        "exact": sum(1 for row in bbox_rows if row.get("bbox_precision") == "exact"),
        "coarse": sum(1 for row in bbox_rows if row.get("bbox_precision") == "coarse"),
        "page_grounded_only": sum(1 for row in bbox_rows if row.get("bbox_precision") == "page_grounded_only"),
    }


def _fallback_bbox_rows(
    *,
    evidence_id: str,
    source_meta: dict,
    source_id: str,
    text: str,
    page_start: int,
    page_end: int,
    line_entries: dict[int, list[dict]],
    corpus_id: str,
    bbox_id_prefix: str,
) -> list[dict]:
    rows = []
    for index, page_number in enumerate(range(page_start, page_end + 1)):
        candidates = line_entries.get(page_number, [])
        if not candidates:
            continue
        rows.append(
            {
                "bbox_id": _geometry_id(
                    source_id=source_id,
                    source_sha256=source_meta["sha256"],
                    page_number=page_number,
                    text=text.strip() if index == 0 else "",
                    coordinates={
                        "x0": min(row["x0"] for row in candidates),
                        "y0": min(row["y0"] for row in candidates),
                        "x1": max(row["x1"] for row in candidates),
                        "y1": max(row["y1"] for row in candidates),
                    },
                ),
                "bbox_precision": "page_grounded_only",
                "corpus_id": corpus_id,
                "page_number": page_number,
                "source_document_id": source_id,
                "source_pdf": source_meta["filename"],
                "source_pdf_path": source_meta["path"],
                "source_sha256": source_meta["sha256"],
                "status": "accepted",
                "text": text.strip() if index == 0 else "",
                "viewer_highlightable": False,
                "x0": min(row["x0"] for row in candidates),
                "x1": max(row["x1"] for row in candidates),
                "y0": min(row["y0"] for row in candidates),
                "y1": max(row["y1"] for row in candidates),
                **coordinate_metadata(candidates[0], highlightable=False),
            }
        )
    if not rows:
        raise ValueError(f"unable_to_build_bbox_rows:{source_id}:{text[:80]}")
    return rows


def _geometry_id(*, source_id: str, source_sha256: str, page_number: int, text: str, coordinates: dict) -> str:
    identity = {
        "source_document_id": source_id,
        "source_sha256": source_sha256,
        "page_number": page_number,
        "coordinate_space": coordinates.get("coordinate_space", "pdf_points"),
        "coordinate_origin": coordinates.get("coordinate_origin", "top_left"),
        "transform_version": coordinates.get("transform_version", "pdf_points_v1"),
        "text": text,
        "x0": coordinates.get("x0"),
        "y0": coordinates.get("y0"),
        "x1": coordinates.get("x1"),
        "y1": coordinates.get("y1"),
    }
    digest = sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return f"uud_geometry::{digest}"


def _compact(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").replace("\u00ad", "").replace("\u00c2", "")
    return re.sub(r"\s+", " ", text).strip().casefold()
