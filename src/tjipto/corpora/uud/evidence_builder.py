from __future__ import annotations

from tjipto.corpora.uud.bbox_builder import aggregate_bbox_precision, build_bbox_rows


def rebuild_evidence(
    existing: dict,
    text: str,
    line_entries: dict[int, list[dict]],
    source_meta: dict,
    bbox_by_evidence: dict[str, list[dict]],
) -> None:
    bbox_records = build_bbox_rows(
        evidence_id=existing["evidence_id"],
        source_meta=source_meta,
        source_id=existing["source_document_id"],
        text=text,
        page_start=min(existing["page_numbers"]),
        page_end=max(existing["page_numbers"]),
        line_entries=line_entries,
    )
    existing["quoted_text"] = "\n".join(row["text"] for row in bbox_records)
    existing["page_numbers"] = sorted({row["page_number"] for row in bbox_records})
    existing["bbox_refs"] = [row["bbox_id"] for row in bbox_records]
    existing["bbox_precision"] = aggregate_bbox_precision(bbox_records)
    existing["viewer_highlightable"] = any(row["viewer_highlightable"] for row in bbox_records)
    bbox_by_evidence[existing["evidence_id"]] = bbox_records
