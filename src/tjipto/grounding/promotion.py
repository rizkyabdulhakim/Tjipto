from __future__ import annotations

import re
import unicodedata


def build_promotion_decisions(
    *,
    evidence: list[dict],
    metadata_grounding: list[dict],
    bbox_rows: list[dict],
    pages: list[dict],
) -> list[dict]:
    page_text = {(row["source_document_id"], row["page_number"]): row.get("text", "") for row in pages}
    decisions = [
        _record_decision(
            record=row,
            record_id=row["evidence_id"],
            record_type="evidence",
            page_text=page_text,
        )
        for row in evidence
        if _needs_decision(row)
    ]
    decisions.extend(
        _record_decision(
            record=row,
            record_id=row["metadata_grounding_id"],
            record_type="metadata_grounding",
            page_text=page_text,
        )
        for row in metadata_grounding
        if _needs_decision(row)
    )
    decisions.extend(
        _record_decision(
            record=row,
            record_id=row["bbox_id"],
            record_type="bbox",
            page_text=page_text,
        )
        for row in bbox_rows
        if _needs_decision(row)
    )
    return sorted(decisions, key=lambda row: (row["record_type"], row["record_id"], row["page_number"]))


def _needs_decision(row: dict) -> bool:
    return row.get("bbox_precision") != "exact" or row.get("viewer_highlightable") is not True


def _record_decision(*, record: dict, record_id: str, record_type: str, page_text: dict[tuple[str, int], str]) -> dict:
    page_numbers = list(record.get("page_numbers") or ([record["page_number"]] if record.get("page_number") is not None else []))
    source_document_id = record.get("source_document_id")
    exact_quote = _quote_available(record, page_text, page_numbers)
    exact_span = bool(record.get("text_span_ids"))
    exact_bbox = record.get("bbox_precision") == "exact" and bool(record.get("bbox_refs") or record.get("bbox_id"))
    highlightable = record.get("viewer_highlightable") is True
    reason = _reason(record)
    return {
        "candidate_status": "exact_candidate" if exact_quote and exact_span and exact_bbox else "blocked",
        "current_grounding_status": record.get("grounding_status") or record.get("bbox_precision"),
        "decision": "keep_non_exact",
        "decision_id": f"promotion_decision::{record_type}::{record_id}",
        "exact_bbox_available": exact_bbox,
        "exact_quote_available": exact_quote,
        "exact_span_available": exact_span,
        "failure_reason": reason,
        "highlightable": highlightable,
        "page_number": min(page_numbers) if page_numbers else None,
        "policy_reason": reason,
        "record_id": record_id,
        "record_type": record_type,
        "review_status": record.get("provenance_review_status") or "validated",
        "source_document_id": source_document_id,
    }


def _reason(record: dict) -> str:
    return (
        record.get("failure_reason")
        or record.get("rejection_reason")
        or record.get("grounding_status")
        or ("non_highlightable_exact_bbox" if record.get("bbox_precision") == "exact" else "non_exact_grounding")
    )


def _quote_available(record: dict, page_text: dict[tuple[str, int], str], page_numbers: list[int]) -> bool:
    quote = _normalize(record.get("quoted_text") or record.get("quote") or record.get("text"))
    source_document_id = record.get("source_document_id")
    if not quote or not source_document_id:
        return False
    haystack = " ".join(_normalize(page_text.get((source_document_id, page_number), "")) for page_number in page_numbers)
    return quote in haystack


def _normalize(text: str | None) -> str:
    normalized = unicodedata.normalize("NFKC", text or "").replace("\xad", "").replace("\xa0", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"\s+([,.;:])", r"\1", normalized)
    return normalized.strip().casefold()
