from __future__ import annotations

import re

from tjipto.ingestion.pdf.words import align_text_to_word_bboxes, word_rows_by_page
from tjipto.corpora.uud.span_disposition_policy import (
    classification_for_role,
    role_for_legal_unit,
    specificity_for_legal_unit,
    unreferenced_role,
)


def apply_page_text_span_dispositions(
    *,
    page_text_spans: list[dict],
    bbox_rows: list[dict],
    word_bboxes: list[dict],
    legal_units: list[dict],
    chunks: list[dict],
    metadata_grounding: list[dict],
    source_conflicts: list[dict],
) -> None:
    chunks_by_unit = {row["legal_unit_id"]: row for row in chunks}
    legal_refs = _legal_span_refs(legal_units, chunks_by_unit)
    metadata_refs = _row_refs(metadata_grounding, "metadata_grounding_id")
    conflict_refs = _row_refs(source_conflicts, "source_conflict_id")
    instrument_units = [row for row in legal_units if role_for_legal_unit(row) not in {"normative_text", "structural_heading"}]

    for span in page_text_spans:
        span_id = span["text_span_id"]
        if span_id in legal_refs:
            _apply(span, legal_refs[span_id])
        elif span_id in metadata_refs:
            _apply(span, _metadata_disposition(metadata_refs[span_id]))
        elif span_id in conflict_refs:
            _apply(span, _source_conflict_disposition(conflict_refs[span_id]))
        else:
            disposition = _instrument_text_disposition(span, instrument_units) or _fallback_disposition(span)
            _apply(span, disposition)
    _apply_bbox_registry_coverage(
        page_text_spans=page_text_spans,
        bbox_rows=bbox_rows,
        word_bboxes=word_bboxes,
        metadata_grounding=metadata_grounding,
    )


def _legal_span_refs(legal_units: list[dict], chunks_by_unit: dict[str, dict]) -> dict[str, dict]:
    refs: dict[str, dict] = {}
    for unit in legal_units:
        chunk = chunks_by_unit.get(unit["legal_unit_id"], {})
        role = role_for_legal_unit(unit)
        for span_id in unit.get("text_span_ids") or chunk.get("text_span_ids") or ():
            candidate = _legal_disposition(unit, chunk, role)
            if span_id not in refs or specificity_for_legal_unit(unit) > refs[span_id]["specificity"]:
                refs[span_id] = {"specificity": specificity_for_legal_unit(unit), **candidate}
    for ref in refs.values():
        ref.pop("specificity", None)
    return refs


def _legal_disposition(unit: dict, chunk: dict, role: str) -> dict:
    if role == "source_conflict_trace":
        return {
            "span_role": role,
            "semantic_classification": "source_conflict_trace",
            "legal_force": "source_conflict_trace_only",
            "promotion_status": "promoted_source_conflict",
            "promotion_target_type": "legal_unit",
            "promotion_target_id": unit["legal_unit_id"],
            "exclusion_reason": None,
            "validation_basis": "source_conflict_trace_text_span",
            "review_status": "reviewed",
        }
    if role == "structural_heading":
        return {
            "span_role": role,
            "semantic_classification": classification_for_role(role),
            "legal_force": "metadata_only",
            "promotion_status": "excluded_structural",
            "promotion_target_type": "legal_unit",
            "promotion_target_id": unit["legal_unit_id"],
            "exclusion_reason": "structural_heading",
            "validation_basis": "legal_unit_structural_text_span",
            "review_status": "accepted",
        }
    if role != "normative_text":
        return _instrument_disposition(unit, role, "legal_unit_text_span")
    return {
        "span_role": role,
        "semantic_classification": classification_for_role(role),
        "legal_force": _legal_force_for_source(unit.get("source_role")),
        "promotion_status": "promoted_legal_unit",
        "promotion_target_type": "legal_unit",
        "promotion_target_id": unit["legal_unit_id"],
        "exclusion_reason": None,
        "validation_basis": "legal_unit_text_span",
        "review_status": "accepted",
    }


def _legal_force_for_source(source_role: str | None) -> str:
    if source_role == "current_consolidated":
        return "canonical_normative"
    if source_role in {
        "original_historical",
        "amendment_1_historical",
        "amendment_2_historical",
        "amendment_3_historical",
        "amendment_4_historical",
    }:
        return "historical_normative"
    return "unknown_needs_review"


def _instrument_disposition(unit: dict, role: str, basis: str) -> dict:
    return {
        "span_role": role,
        "semantic_classification": classification_for_role(role),
        "legal_force": "source_conflict_trace_only" if role == "source_conflict_trace" else "amendment_instrument",
        "promotion_status": "promoted_source_conflict" if role == "source_conflict_trace" else "nonruntime_instrument_text",
        "promotion_target_type": "legal_unit",
        "promotion_target_id": unit["legal_unit_id"],
        "exclusion_reason": None if role == "source_conflict_trace" else "nonruntime_instrument_text",
        "validation_basis": basis,
        "review_status": "reviewed" if unit.get("provenance_exception_category") else "accepted",
    }


def _metadata_disposition(row: dict) -> dict:
    return {
        "span_role": "metadata_text",
        "semantic_classification": "session_institution_metadata",
        "legal_force": "metadata_only",
        "promotion_status": "promoted_metadata",
        "promotion_target_type": "metadata_grounding",
        "promotion_target_id": row["metadata_grounding_id"],
        "exclusion_reason": None,
        "validation_basis": "metadata_grounding_text_span",
        "review_status": "accepted",
    }


def _source_conflict_disposition(row: dict) -> dict:
    return {
        "span_role": "source_conflict_trace",
        "semantic_classification": "source_conflict_trace",
        "legal_force": "source_conflict_trace_only",
        "promotion_status": "promoted_source_conflict",
        "promotion_target_type": "source_conflict",
        "promotion_target_id": row["source_conflict_id"],
        "exclusion_reason": None,
        "validation_basis": "source_conflict_text_span",
        "review_status": "reviewed",
    }


def _instrument_text_disposition(span: dict, instrument_units: list[dict]) -> dict | None:
    span_text = _compact(span.get("text"))
    if not span_text:
        return None
    for unit in instrument_units:
        if (
            unit["source_document_id"] == span["source_document_id"]
            and unit["page_start"] <= span["page_number"] <= unit["page_end"]
            and span_text in _compact(unit["text"])
        ):
            return _instrument_disposition(unit, role_for_legal_unit(unit), "instrument_text_nonpromoting_match")
    return None


def _fallback_disposition(span: dict) -> dict:
    role = unreferenced_role(span)
    if role == "needs_review":
        return {
            "span_role": role,
            "semantic_classification": "needs_review",
            "legal_force": "unknown_needs_review",
            "promotion_status": "needs_review",
            "promotion_target_type": None,
            "promotion_target_id": None,
            "exclusion_reason": "unclassified_text_span",
            "validation_basis": "no_artifact_reference",
            "review_status": "needs_review",
        }
    if role in {"instrument_scope", "decision_clause", "effective_clause", "signatory_block", "metadata_text"}:
        return {
            "span_role": role,
            "semantic_classification": classification_for_role(role),
            "legal_force": "metadata_only" if role == "metadata_text" else "amendment_instrument",
            "promotion_status": "nonruntime_instrument_text",
            "promotion_target_type": None,
            "promotion_target_id": None,
            "exclusion_reason": "nonruntime_instrument_text",
            "validation_basis": "uud_span_policy",
            "review_status": "accepted",
        }
    return {
        "span_role": role,
        "semantic_classification": classification_for_role(role),
        "legal_force": "metadata_only" if role == "structural_heading" else "nonlegal",
        "promotion_status": "excluded_structural" if role == "structural_heading" else "excluded_nonlegal",
        "promotion_target_type": None,
        "promotion_target_id": None,
        "exclusion_reason": role,
        "validation_basis": "uud_span_policy",
        "review_status": "accepted",
    }


def _row_refs(rows: list[dict], id_key: str) -> dict[str, dict]:
    refs: dict[str, dict] = {}
    for row in rows:
        for span_id in row.get("text_span_ids") or ():
            refs.setdefault(span_id, row)
    return refs


def _rows_by_span(rows: list[dict]) -> dict[str, list[dict]]:
    refs: dict[str, list[dict]] = {}
    for row in rows:
        for span_id in row.get("text_span_ids") or ():
            refs.setdefault(span_id, []).append(row)
    return refs


def _apply_bbox_registry_coverage(
    *,
    page_text_spans: list[dict],
    bbox_rows: list[dict],
    word_bboxes: list[dict],
    metadata_grounding: list[dict],
) -> None:
    bbox_keys = {_bbox_registry_key(row) for row in bbox_rows}
    metadata_context = _rows_by_span(metadata_grounding)
    words_by_page = word_rows_by_page(word_bboxes)
    for span in page_text_spans:
        if _bbox_registry_key(span) in bbox_keys:
            span["bbox_registry_coverage_status"] = "bbox_key_present"
            if span.get("promotion_status") == "promoted_legal_unit":
                span["bbox_registry_coverage_reason"] = "blocked_by_no_word_level_bbox_artifact"
                _apply_word_bbox_promotion(span, words_by_page)
            continue
        bucket, reason = _missing_bbox_coverage(
            span=span,
            metadata_rows=metadata_context.get(span["text_span_id"], []),
        )
        span["bbox_registry_coverage_status"] = "bbox_key_absent"
        span["bbox_registry_coverage_bucket"] = bucket
        span["bbox_registry_coverage_reason"] = reason
        span["exposure_policy"] = _exposure_policy(bucket, reason)
        span["exposure_target_kind"] = span.get("promotion_target_type") or "raw_text_span"
        span["exposure_target_ref"] = span.get("promotion_target_id") or span["text_span_id"]
        span["exposure_clickable"] = False
        span["exposure_citation_final"] = False
        span["exposure_exactness_level"] = "position_only"
        span["field_bbox_feasibility"] = _field_bbox_feasibility(reason)
        _apply_word_bbox_promotion(span, words_by_page)


def _missing_bbox_coverage(*, span: dict, metadata_rows: list[dict]) -> tuple[str, str]:
    if span.get("promotion_status") == "excluded_nonlegal" or _is_marker_only_text(span.get("text")):
        return "nonlegal_excluded_provenance", "nonlegal_excluded_from_public_highlight"
    if span.get("promotion_status") == "promoted_metadata" or metadata_rows:
        reasons = {row.get("failure_reason") for row in metadata_rows}
        if "metadata_publication_block_requires_page_level_support" in reasons:
            return "metadata_provenance_candidate", "metadata_page_grounded_only_by_policy"
        if "metadata_decision_sentence_continues_beyond_field" in reasons:
            return "metadata_provenance_candidate", "blocked_by_layout"
        return "metadata_provenance_candidate", "blocked_by_missing_exact_bbox"
    if span.get("promotion_status") == "promoted_source_conflict":
        return "source_anomaly_provenance_candidate", "source_anomaly_anchor_only_until_exact_span_available"
    if span.get("promotion_status") == "excluded_structural":
        return "structural_provenance_only", "structural_provenance_only"
    if span.get("promotion_status") == "promoted_legal_unit":
        return "legal_citation_candidate", "blocked_by_no_word_level_bbox_artifact"
    return "raw_span_only_with_reason", "blocked_by_missing_exact_bbox"


def _exposure_policy(bucket: str, reason: str) -> str:
    if bucket == "structural_provenance_only":
        return "structural_provenance_position"
    if bucket == "nonlegal_excluded_provenance":
        return "nonlegal_excluded_position"
    if reason == "blocked_by_no_word_level_bbox_artifact":
        return "blocked_no_word_level_bbox"
    return "raw_provenance_position"


def _field_bbox_feasibility(reason: str) -> str:
    if reason == "exact_word_bbox_available":
        return "exact_safe"
    if reason == "blocked_by_no_word_level_bbox_artifact":
        return "requires_word_level_bbox"
    if reason == "blocked_by_layout":
        return "blocked_by_layout"
    if reason == "metadata_page_grounded_only_by_policy":
        return "page_level_only"
    if reason == "source_anomaly_anchor_only_until_exact_span_available":
        return "line_level_only"
    if reason == "blocked_by_text_mismatch":
        return "blocked_by_text_boundary"
    return "page_level_only"


def _bbox_registry_key(row: dict) -> tuple[object, ...]:
    return (
        row.get("source_document_id"),
        row.get("source_sha256"),
        row.get("page_number"),
        row.get("text"),
        row.get("x0"),
        row.get("y0"),
        row.get("x1"),
        row.get("y1"),
    )


def _apply_word_bbox_promotion(
    span: dict,
    words_by_page: dict[tuple[str, int], list[dict]],
) -> None:
    if span.get("promotion_status") != "promoted_legal_unit" and span.get("bbox_registry_coverage_reason") not in {
        "blocked_by_missing_exact_bbox",
        "blocked_by_layout",
        "blocked_by_no_word_level_bbox_artifact",
        "source_anomaly_anchor_only_until_exact_span_available",
    }:
        return
    match = align_text_to_word_bboxes(
        text=span.get("text"),
        source_document_id=span["source_document_id"],
        page_numbers=[span["page_number"]],
        words_by_page=words_by_page,
        reference_bbox=span,
    )
    if not match:
        return
    span["word_bbox_ids"] = match["matched_word_bbox_ids"]
    span["span_bbox_ids"] = [
        bbox_id
        for bbox_id in match["matched_word_bbox_ids"]
        if any(
            row.get("word_bbox_id") == bbox_id and _intersects(span, row)
            for row in words_by_page[(span["source_document_id"], span["page_number"])]
        )
    ]
    span["word_bbox_match_method"] = match["match_method"]
    span["word_bbox_match_confidence"] = match["match_confidence"]
    span["word_bbox_candidate_count"] = match["candidate_count"]
    span["word_bbox_distance_to_existing_span_bbox"] = match["distance_to_existing_span_bbox"]
    span["bbox_registry_coverage_reason"] = "exact_word_bbox_available"


def _intersects(left: dict, right: dict) -> bool:
    return min(left["x1"], right["x1"]) > max(left["x0"], right["x0"]) and min(left["y1"], right["y1"]) > max(left["y0"], right["y0"])


def _is_marker_only_text(text: str | None) -> bool:
    stripped = (text or "").strip()
    return bool(stripped) and re.fullmatch(r"[*_\-\s\u00ad]+\)?", stripped) is not None


def _apply(span: dict, disposition: dict) -> None:
    span.update(disposition)


def _compact(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("\u00a0", " ").replace("\u00ad", "")).strip().casefold()
