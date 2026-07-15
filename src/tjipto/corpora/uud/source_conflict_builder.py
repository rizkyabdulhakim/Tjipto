from __future__ import annotations

import json

from tjipto.corpora.uud.specs import SOURCE_CONFLICT_SPECS
from tjipto.evidence.store import exact_bboxes_for_text_spans
from tjipto.ingestion.pdf.words import align_text_to_word_bboxes, word_rows_by_page


def build_source_conflicts() -> list[dict]:
    return json.loads(json.dumps(SOURCE_CONFLICT_SPECS))


def apply_source_conflict_grounding(
    source_conflicts: list[dict],
    evidence: list[dict],
    bbox_rows: list[dict],
    word_bboxes: list[dict],
    page_text_spans: list[dict],
) -> None:
    evidence_by_id = {row["evidence_id"]: row for row in evidence}
    bbox_by_evidence: dict[str, list[str]] = {}
    span_by_id = {row["text_span_id"]: row for row in page_text_spans if row.get("text_span_id")}
    words_by_page = word_rows_by_page(word_bboxes)
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
        raw_provenance_bboxes = exact_bboxes_for_text_spans(
            [span_by_id.get(text_span_id) for text_span_id in row["text_span_ids"]],
            bbox_rows,
        )
        raw_provenance_bbox_ids = [bbox["bbox_id"] for bbox in raw_provenance_bboxes if bbox.get("bbox_id")]
        raw_provenance_text_span_ids = []
        for text_span_id in row["text_span_ids"]:
            span = span_by_id.get(text_span_id)
            if _text_span_has_exact_bbox(span, raw_provenance_bboxes):
                raw_provenance_text_span_ids.append(text_span_id)
                continue
            match = align_text_to_word_bboxes(
                text=span.get("text") if span else None,
                source_document_id=row["source_document_id"],
                page_numbers=[span["page_number"]] if span else [],
                words_by_page=words_by_page,
                reference_bbox=span,
            )
            if not match:
                continue
            raw_provenance_text_span_ids.append(text_span_id)
            raw_provenance_bbox_ids.extend(match["matched_word_bbox_ids"])
        raw_provenance_bbox_ids = list(dict.fromkeys(raw_provenance_bbox_ids))
        blocked_text_span_ids = [
            text_span_id for text_span_id in row["text_span_ids"] if text_span_id not in set(raw_provenance_text_span_ids)
        ]
        row["canonical_use_allowed"] = False
        row["grounding_status"] = "text_span_exact" if row["text_span_ids"] else "grounding_unavailable"
        row["validation_status"] = "accepted_source_conflict_record" if row["text_span_ids"] else "grounding_unavailable"
        row["raw_provenance_bbox_ids"] = raw_provenance_bbox_ids
        row["raw_provenance_text_span_ids"] = raw_provenance_text_span_ids
        row["blocked_raw_provenance_text_span_ids"] = blocked_text_span_ids
        row["blocked_raw_provenance_reason"] = "source_anomaly_anchor_only_until_exact_span_available" if blocked_text_span_ids else None
        row["blocked_raw_provenance_text_span_reasons"] = {
            text_span_id: row["blocked_raw_provenance_reason"] for text_span_id in blocked_text_span_ids
        }
        row["provenance_bbox_status"] = _provenance_bbox_status(row["text_span_ids"], raw_provenance_text_span_ids)
        row["provenance_highlight_scope"] = _provenance_highlight_scope(row["text_span_ids"], raw_provenance_text_span_ids)
        row["final_evidence_available"] = bool(row["evidence_ids"] and row["bbox_ids"])
        if not row["evidence_ids"] or not row["bbox_ids"]:
            row["failure_reason"] = _source_conflict_failure_reason(
                text_span_ids=row["text_span_ids"],
                raw_provenance_text_span_ids=raw_provenance_text_span_ids,
            )


def _candidate_evidence_refs(row: dict) -> tuple[str, ...]:
    decision = row.get("resolution_decision") or {}
    return tuple(
        value
        for key, value in decision.items()
        if key.endswith("_reference") or key.endswith("_evidence_id") or key == "historical_source_reference"
    )


def _matching_text_spans(row: dict, page_text_spans: list[dict]) -> list[str]:
    anchors = _anchor_terms(row)
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


def _anchor_terms(row: dict) -> tuple[str, ...]:
    return tuple(str(value) for value in row.get("anchor_terms") or () if str(value).strip())


def _text_span_has_exact_bbox(span: dict | None, raw_provenance_bboxes: list[dict]) -> bool:
    if not span:
        return False
    return any(
        all(
            span.get(field) == bbox.get(field)
            for field in ("source_document_id", "source_sha256", "page_number", "text", "x0", "y0", "x1", "y1")
        )
        for bbox in raw_provenance_bboxes
    )


def _provenance_bbox_status(text_span_ids: list[str], raw_provenance_text_span_ids: list[str]) -> str:
    if not raw_provenance_text_span_ids:
        return "exact_raw_provenance_bbox_unavailable"
    if len(raw_provenance_text_span_ids) == len(text_span_ids):
        return "exact_raw_provenance_bbox_available"
    return "partial_exact_raw_provenance_bbox_available"


def _provenance_highlight_scope(text_span_ids: list[str], raw_provenance_text_span_ids: list[str]) -> str:
    if not raw_provenance_text_span_ids:
        return "unavailable"
    if len(raw_provenance_text_span_ids) == len(text_span_ids):
        return "all_relevant_spans"
    return "anchor_span_only"


def _source_conflict_failure_reason(*, text_span_ids: list[str], raw_provenance_text_span_ids: list[str]) -> str:
    if not raw_provenance_text_span_ids:
        return "final_evidence_unavailable_raw_provenance_bbox_unavailable"
    if len(raw_provenance_text_span_ids) == len(text_span_ids):
        return "final_evidence_unavailable_raw_provenance_bbox_available"
    return "final_evidence_unavailable_partial_raw_provenance_bbox_available"
