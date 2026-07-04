from __future__ import annotations

from collections import defaultdict

from tjipto.ingestion.pdf.bbox import (
    aggregate_bbox_precision,
    bbox_precision_counts,
    build_text_bbox_rows,
    pdf_lines,
)


INSERTED_BAB_HEADING_BBOX_MARKER = "::heading_bab_"
__all__ = [
    "aggregate_bbox_precision",
    "apply_inserted_bab_heading_bbox_policy",
    "bbox_precision_counts",
    "build_bbox_rows",
    "pdf_lines",
]


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
    return build_text_bbox_rows(
        evidence_id=evidence_id,
        source_meta=source_meta,
        source_id=source_id,
        text=text,
        page_start=page_start,
        page_end=page_end,
        line_entries=line_entries,
        corpus_id="uud",
        bbox_id_prefix="uud_unified_bbox",
    )


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
