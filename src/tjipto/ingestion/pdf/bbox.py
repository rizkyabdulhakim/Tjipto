from __future__ import annotations

import re
import unicodedata
from hashlib import sha256
import json

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
        for block_index, block in enumerate(page.get_text("dict").get("blocks", [])):
            if block.get("type") != 0:
                continue
            for line_index, line in enumerate(block.get("lines", [])):
                text = "".join(span.get("text", "") for span in line.get("spans", [])).strip()
                if not text:
                    continue
                if not text:
                    continue
                x0 = min(span["bbox"][0] for span in line.get("spans", []))
                y0 = min(span["bbox"][1] for span in line.get("spans", []))
                x1 = max(span["bbox"][2] for span in line.get("spans", []))
                y1 = max(span["bbox"][3] for span in line.get("spans", []))
                entries.append({"text": text, "x0": x0, "y0": y0, "x1": x1, "y1": y1, "width": page.rect.width, "height": page.rect.height, "block_index": block_index, "line_index": line_index})
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
