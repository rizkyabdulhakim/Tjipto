from __future__ import annotations

from collections import defaultdict
import re
import unicodedata


INSERTED_BAB_HEADING_BBOX_MARKER = "::heading_bab_"


def build_bbox_rows(
    *,
    evidence_id: str,
    source_meta: dict,
    source_id: str,
    text: str,
    page_start: int,
    page_end: int,
    line_entries: dict[int, list[dict]],
) -> list[dict]:
    expected = [_compact(line) for line in text.splitlines() if line.strip()]
    matched = []
    target_index = 0
    for page_number in range(page_start, page_end + 1):
        candidates = line_entries.get(page_number, [])
        for candidate in candidates:
            if target_index >= len(expected):
                break
            if _compact(candidate["text"]) != expected[target_index]:
                continue
            matched.append({
                "page_number": page_number,
                **candidate,
            })
            target_index += 1
        if target_index >= len(expected):
            break
    if target_index < len(expected):
        return _fallback_bbox_rows(
            evidence_id=evidence_id,
            source_meta=source_meta,
            source_id=source_id,
            text=text,
            page_start=page_start,
            page_end=page_end,
            line_entries=line_entries,
        )
    rows = []
    for index, row in enumerate(matched):
        rows.append({
            "bbox_id": f"uud_unified_bbox::{evidence_id}::{index:04d}",
            "bbox_precision": "exact",
            "corpus_id": "uud",
            "evidence_id": evidence_id,
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
        })
    return rows


def pdf_lines(doc) -> dict[int, list[dict]]:
    pages: dict[int, list[dict]] = {}
    for page_number in range(1, doc.page_count + 1):
        page = doc[page_number - 1]
        entries = []
        for block in page.get_text("dict").get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                text = "".join(span.get("text", "") for span in line.get("spans", [])).strip()
                if not text:
                    continue
                x0 = min(span["bbox"][0] for span in line.get("spans", []))
                y0 = min(span["bbox"][1] for span in line.get("spans", []))
                x1 = max(span["bbox"][2] for span in line.get("spans", []))
                y1 = max(span["bbox"][3] for span in line.get("spans", []))
                entries.append({"text": text, "x0": x0, "y0": y0, "x1": x1, "y1": y1})
        pages[page_number] = entries
    return pages


def aggregate_bbox_precision(rows: list[dict]) -> str:
    precisions = {row.get("bbox_precision") for row in rows}
    if precisions == {"exact"}:
        return "exact"
    if "page_grounded_only" in precisions:
        return "page_grounded_only"
    return "coarse"


def apply_inserted_bab_heading_bbox_policy(bbox_rows: list[dict], evidence: list[dict]) -> None:
    evidence_by_id = {row["evidence_id"]: row for row in evidence}
    for row in bbox_rows:
        if INSERTED_BAB_HEADING_BBOX_MARKER not in row["bbox_id"]:
            continue
        evidence_row = evidence_by_id.get(row["evidence_id"])
        if evidence_row and evidence_row.get("citation") == row.get("text"):
            continue
        row["viewer_highlightable"] = False
    by_evidence: dict[str, list[dict]] = defaultdict(list)
    for row in bbox_rows:
        by_evidence[row["evidence_id"]].append(row)
    for row in evidence:
        bbox_records = by_evidence.get(row["evidence_id"], [])
        if not bbox_records:
            continue
        row["bbox_precision"] = aggregate_bbox_precision(bbox_records)
        row["viewer_highlightable"] = any(item.get("viewer_highlightable") is True for item in bbox_records)


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
) -> list[dict]:
    rows = []
    for index, page_number in enumerate(range(page_start, page_end + 1)):
        candidates = line_entries.get(page_number, [])
        if not candidates:
            continue
        rows.append({
            "bbox_id": f"uud_unified_bbox::{evidence_id}::{index:04d}",
            "bbox_precision": "page_grounded_only",
            "corpus_id": "uud",
            "evidence_id": evidence_id,
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
        })
    if not rows:
        raise ValueError(f"unable_to_build_bbox_rows:{source_id}:{text[:80]}")
    return rows


def _compact(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").replace("\u00ad", "").replace("Â", "")
    return re.sub(r"\s+", " ", text).strip().casefold()
