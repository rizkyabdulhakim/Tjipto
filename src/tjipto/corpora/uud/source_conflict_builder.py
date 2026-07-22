from __future__ import annotations

import json

from tjipto.corpora.uud.specs import SOURCE_CONFLICT_SPECS
from tjipto.evidence.store import exact_bboxes_for_text_spans
from tjipto.ingestion.pdf.words import align_text_to_word_bboxes, word_rows_by_page


def build_source_conflicts() -> list[dict]:
    return [
        row for row in json.loads(json.dumps(SOURCE_CONFLICT_SPECS))
        if row.get("source_anomaly_kind") != "renumbering_provenance"
    ]


def apply_source_conflict_grounding(
    source_conflicts: list[dict],
    evidence: list[dict],
    bbox_rows: list[dict],
    word_bboxes: list[dict],
    page_text_spans: list[dict],
) -> None:
    evidence_by_id = {row["evidence_id"]: row for row in evidence}
    bbox_by_evidence: dict[str, list[str]] = {
        row["evidence_id"]: list(row.get("bbox_refs") or ()) for row in evidence
    }
    span_by_id = {row["text_span_id"]: row for row in page_text_spans if row.get("text_span_id")}
    words_by_page = word_rows_by_page(word_bboxes)
    for row in source_conflicts:
        evidence_ids = [
            ref
            for ref in _candidate_evidence_refs(row)
            if ref in evidence_by_id and evidence_by_id[ref]["source_document_id"] == row["source_document_id"]
        ]
        evidence_ids = list(dict.fromkeys(evidence_ids))
        row["page_numbers"] = list(row.get("affected_pages") or [])
        row["text_span_ids"] = _matching_text_spans(row, page_text_spans)
        matched_spans = [span_by_id[span_id] for span_id in row["text_span_ids"] if span_id in span_by_id]
        row["source_sha256"] = matched_spans[0].get("source_sha256") if matched_spans else None
        row["source_quote"] = "\n".join(span.get("exact_quote") or span.get("text") or "" for span in matched_spans)
        comparison = [
            span
            for span in page_text_spans
            if span.get("source_document_id") == "uud::current_consolidated"
            and row.get("canonical_label")
            and row.get("canonical_label") in (span.get("text") or "")
        ]
        row["comparison_source_document_id"] = comparison[0].get("source_document_id") if comparison else None
        row["comparison_source_sha256"] = comparison[0].get("source_sha256") if comparison else None
        row["comparison_quote"] = "\n".join(span.get("exact_quote") or span.get("text") or "" for span in comparison)
        row["review_evidence"] = row.get("provenance_summary") or row.get("provenance_review_status") or "reviewed_source_comparison"
        row["evidence_ids"] = evidence_ids
        row["authoritative_evidence_id"] = evidence_ids[0] if evidence_ids else None
        row["authority_kind"] = "source_anomaly_provenance"
        row["citation_final"] = False
        row["object_role"] = "trace_support"
        row["is_citation_object"] = False
        row["target_precision"] = "source_provenance"
        row["recovery_capability"] = "exact_materialized" if evidence_ids else "technically_unrecoverable"
        row["recovery_status"] = "materialized" if evidence_ids else "failed"
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
            row["recovery_failure_code"] = row["failure_reason"]


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
