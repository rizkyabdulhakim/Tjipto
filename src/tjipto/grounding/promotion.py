from __future__ import annotations

import re
import unicodedata


def build_promotion_decisions(
    *,
    evidence: list[dict],
    metadata_grounding: list[dict],
    bbox_rows: list[dict],
    page_text_spans: list[dict],
    pages: list[dict],
) -> list[dict]:
    page_text = {(row["source_document_id"], row["page_number"]): row.get("text", "") for row in pages}
    spans_by_page: dict[tuple[str, int], list[dict]] = {}
    for row in page_text_spans:
        spans_by_page.setdefault((row["source_document_id"], row["page_number"]), []).append(row)
    bbox_by_id = {row["bbox_id"]: row for row in bbox_rows}
    decisions = [
        _record_decision(
            record=row,
            record_id=row["evidence_id"],
            record_type="evidence",
            page_text=page_text,
            spans_by_page=spans_by_page,
            bbox_by_id=bbox_by_id,
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
            spans_by_page=spans_by_page,
            bbox_by_id=bbox_by_id,
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
            spans_by_page=spans_by_page,
            bbox_by_id=bbox_by_id,
        )
        for row in bbox_rows
        if _needs_decision(row)
    )
    return sorted(decisions, key=lambda row: (row["record_type"], row["record_id"], row["page_number"]))


def _needs_decision(row: dict) -> bool:
    return row.get("bbox_precision") != "exact" or row.get("viewer_highlightable") is not True


def _record_decision(
    *,
    record: dict,
    record_id: str,
    record_type: str,
    page_text: dict[tuple[str, int], str],
    spans_by_page: dict[tuple[str, int], list[dict]],
    bbox_by_id: dict[str, dict],
) -> dict:
    page_numbers = list(record.get("page_numbers") or ([record["page_number"]] if record.get("page_number") is not None else []))
    source_document_id = record.get("source_document_id")
    text = record.get("quoted_text") or record.get("quote") or record.get("text")
    quote_match_status = _quote_match_status(record, page_text, page_numbers)
    exact_quote = quote_match_status == "exact_full_quote_match"
    span_match_status, matched_span_ids, matched_excerpt = _span_match(
        text=text,
        source_document_id=source_document_id,
        page_numbers=page_numbers,
        spans_by_page=spans_by_page,
        explicit_span_ids=record.get("text_span_ids") or (),
    )
    subspan_match_status = "matched" if span_match_status == "subspan_inside_larger_span" else "not_applicable"
    bbox_refs = list(record.get("bbox_refs") or ([record["bbox_id"]] if record.get("bbox_id") else []))
    exact_bbox = record.get("bbox_precision") == "exact" and _bbox_refs_valid(bbox_refs, bbox_by_id, exact_only=True)
    bbox_union_status = _bbox_union_status(bbox_refs, bbox_by_id, exact_bbox)
    highlightable = record.get("viewer_highlightable") is True
    reason = _reason(record)
    can_be_exact = exact_quote and span_match_status == "normalized_span_sequence_match" and exact_bbox and highlightable
    metadata_feasibility = _metadata_promotion_feasibility(
        record_type=record_type,
        reason=reason,
        exact_quote=exact_quote,
        span_match_status=span_match_status,
        exact_bbox=exact_bbox,
        matched_span_ids=matched_span_ids,
        can_be_exact=can_be_exact,
    )
    return {
        "bbox_union_status": bbox_union_status,
        "blocker_evidence": _blocker_evidence(
            reason=reason,
            quote_match_status=quote_match_status,
            span_match_status=span_match_status,
            bbox_union_status=bbox_union_status,
        ),
        "can_be_exact_citation": can_be_exact,
        "can_be_exact_highlight": can_be_exact,
        "candidate_status": "exact_candidate" if can_be_exact else "blocked",
        "current_grounding_status": record.get("grounding_status") or record.get("bbox_precision"),
        "decision": "keep_non_exact",
        "decision_id": f"promotion_decision::{record_type}::{record_id}",
        "exact_bbox_available": exact_bbox,
        "exact_quote_available": exact_quote,
        "exact_span_available": span_match_status == "normalized_span_sequence_match",
        "failure_reason": reason,
        "highlightable": highlightable,
        "matched_page_numbers": page_numbers,
        "matched_span_ids": matched_span_ids,
        "matched_text_excerpt": matched_excerpt,
        "metadata_exact_promotion_feasibility": metadata_feasibility,
        "page_number": min(page_numbers) if page_numbers else None,
        "policy_reason": reason,
        "promotion_attempt_method": "source_page_span_bbox_feasibility",
        "promotion_attempt_result": "blocked_after_feasibility_check",
        "promotion_attempted": True,
        "quote_match_status": quote_match_status,
        "record_id": record_id,
        "record_type": record_type,
        "review_status": record.get("provenance_review_status") or "validated",
        "source_document_id": source_document_id,
        "span_match_status": span_match_status,
        "subspan_match_status": subspan_match_status,
    }


def _reason(record: dict) -> str:
    return (
        record.get("failure_reason")
        or record.get("rejection_reason")
        or record.get("grounding_status")
        or ("non_highlightable_exact_bbox" if record.get("bbox_precision") == "exact" else "non_exact_grounding")
    )


def _quote_match_status(record: dict, page_text: dict[tuple[str, int], str], page_numbers: list[int]) -> str:
    quote = _normalize(record.get("quoted_text") or record.get("quote") or record.get("text"))
    source_document_id = record.get("source_document_id")
    if not quote or not source_document_id:
        return "missing_quote"
    haystack = " ".join(_normalize(page_text.get((source_document_id, page_number), "")) for page_number in page_numbers)
    return "exact_full_quote_match" if quote in haystack else "no_full_quote_match"


def _span_match(
    *,
    text: str | None,
    source_document_id: str | None,
    page_numbers: list[int],
    spans_by_page: dict[tuple[str, int], list[dict]],
    explicit_span_ids: list[str] | tuple[str, ...],
) -> tuple[str, list[str], str]:
    if explicit_span_ids:
        return "normalized_span_sequence_match", list(explicit_span_ids), _excerpt(text)
    target = _normalize(text)
    if not target or not source_document_id:
        return "no_span_match", [], ""
    spans = [span for page in page_numbers for span in spans_by_page.get((source_document_id, page), [])]
    for start in range(len(spans)):
        joined = ""
        matched: list[str] = []
        for span in spans[start:]:
            joined = _normalize(f"{joined} {span.get('text', '')}")
            matched.append(span["text_span_id"])
            if joined == target:
                return "normalized_span_sequence_match", matched, _excerpt(text)
            if len(joined) > len(target) + 80 or not target.startswith(joined):
                break
    for span in spans:
        span_text = _normalize(span.get("text"))
        if target and target in span_text:
            return "subspan_inside_larger_span", [span["text_span_id"]], _excerpt(span.get("text"))
    return "page_level_text_match_only", [], _excerpt(text)


def _bbox_refs_valid(refs: list[str], bbox_by_id: dict[str, dict], *, exact_only: bool) -> bool:
    if not refs:
        return False
    for ref in refs:
        bbox = bbox_by_id.get(ref)
        if not bbox:
            return False
        if exact_only and bbox.get("bbox_precision") != "exact":
            return False
        if not (
            all(bbox.get(key) is not None for key in ("x0", "y0", "x1", "y1"))
            and bbox.get("x1", 0) >= bbox.get("x0", 0)
            and bbox.get("y1", 0) >= bbox.get("y0", 0)
        ):
            return False
    return True


def _bbox_union_status(refs: list[str], bbox_by_id: dict[str, dict], exact_bbox: bool) -> str:
    if exact_bbox:
        return "exact_bbox_available"
    if _bbox_refs_valid(refs, bbox_by_id, exact_only=False):
        return "bbox_union_available"
    return "not_supported_by_current_bbox_artifact"


def _blocker_evidence(*, reason: str, quote_match_status: str, span_match_status: str, bbox_union_status: str) -> dict:
    return {
        "bbox_union_status": bbox_union_status,
        "policy_reason": reason,
        "quote_match_status": quote_match_status,
        "span_match_status": span_match_status,
    }


def _metadata_promotion_feasibility(
    *,
    record_type: str,
    reason: str,
    exact_quote: bool,
    span_match_status: str,
    exact_bbox: bool,
    matched_span_ids: list[str],
    can_be_exact: bool,
) -> str | None:
    if record_type != "metadata_grounding":
        return None
    if can_be_exact:
        return "promotable_exact"
    if reason == "metadata_publication_block_requires_page_level_support":
        return "page_level_only_by_policy"
    if not exact_quote or span_match_status == "page_level_text_match_only":
        return "blocked_by_text_boundary"
    if reason == "metadata_decision_sentence_continues_beyond_field":
        return "blocked_by_layout"
    if span_match_status == "normalized_span_sequence_match" and len(matched_span_ids) > 1 and not exact_bbox:
        return "multi_span_exact_possible"
    if span_match_status == "normalized_span_sequence_match" and not exact_bbox:
        return "exact_span_found_but_bbox_missing"
    if not exact_bbox:
        return "blocked_by_no_exact_bbox"
    return "blocked_by_layout"


def _excerpt(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip()[:240]


def _normalize(text: str | None) -> str:
    normalized = unicodedata.normalize("NFKC", text or "").replace("\xad", "").replace("\xa0", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"\s+([,.;:])", r"\1", normalized)
    return normalized.strip().casefold()
