from __future__ import annotations

from copy import deepcopy

from tjipto.corpora.uud.provenance_exceptions import ACCEPTED_NONCANONICAL_SOURCE_CONFLICT_TRACE_ONLY
from tjipto.corpora.uud.specs import SOURCE_CONFLICT_SPECS


def build_source_conflicts() -> list[dict]:
    return deepcopy(list(SOURCE_CONFLICT_SPECS))


def apply_source_conflict_grounding(
    source_conflicts: list[dict],
    evidence: list[dict],
    bbox_rows: list[dict],
    page_text_spans: list[dict],
) -> None:
    evidence_by_id = {row["evidence_id"]: row for row in evidence}
    bbox_by_evidence: dict[str, list[str]] = {}
    for row in bbox_rows:
        bbox_by_evidence.setdefault(row["evidence_id"], []).append(row["bbox_id"])
    for row in source_conflicts:
        evidence_ids = [
            ref
            for ref in _candidate_evidence_refs(row)
            if ref in evidence_by_id and evidence_by_id[ref]["source_document_id"] == row["source_document_id"]
        ]
        row["page_numbers"] = list(row.get("affected_pages") or [])
        row["text_span_ids"] = _matching_text_spans(row, page_text_spans)
        row["evidence_ids"] = evidence_ids
        row["bbox_ids"] = [bbox_id for evidence_id in evidence_ids for bbox_id in bbox_by_evidence.get(evidence_id, [])]
        row["canonical_use_allowed"] = False
        row["grounding_status"] = "text_span_exact" if row["text_span_ids"] else "grounding_unavailable"
        row["validation_status"] = "accepted_source_conflict_record" if row["text_span_ids"] else "grounding_unavailable"
        if row.get("type") == "source_marker_sequence_conflict":
            row["provenance_exception_category"] = ACCEPTED_NONCANONICAL_SOURCE_CONFLICT_TRACE_ONLY
            row["provenance_review_status"] = "reviewed"
        if not row["evidence_ids"] or not row["bbox_ids"]:
            row["failure_reason"] = "source_conflict_evidence_or_bbox_unavailable"


def _candidate_evidence_refs(row: dict) -> tuple[str, ...]:
    decision = row.get("resolution_decision") or {}
    return tuple(
        value
        for key, value in decision.items()
        if key.endswith("_reference") or key.endswith("_evidence_id") or key == "historical_source_reference"
    )


def _matching_text_spans(row: dict, page_text_spans: list[dict]) -> list[str]:
    anchors = _anchors(row.get("type"))
    if not anchors:
        return []
    pages = set(row.get("affected_pages") or ())
    return [
        span["text_span_id"]
        for span in page_text_spans
        if span["source_document_id"] == row["source_document_id"]
        and span["page_number"] in pages
        and any(anchor in span.get("text", "") for anchor in anchors)
    ]


def _anchors(conflict_type: str | None) -> tuple[str, ...]:
    if conflict_type == "article_renumbering_conflict":
        return ("Pasal 25E", "Pasal 25A")
    if conflict_type == "source_marker_sequence_conflict":
        return ("ATURAN TAMBAHAN", "Pasal III")
    return ()
