"""Terminal disposition inventory for raw PDF objects."""

from __future__ import annotations


TERMINAL_DISPOSITIONS = frozenset(
    {
        "promoted_normative_evidence",
        "promoted_structural_evidence",
        "promoted_metadata",
        "promoted_source_anomaly",
        "resolved_mixed_dispositions",
        "source_annotation_object",
        "excluded_nonlegal_object",
        "excluded_duplicate",
        "unsupported_nontext_object",
        "extraction_failed",
        "needs_review",
    }
)


def build_source_object_inventory(
    *, source_objects: tuple[dict, ...], page_text_spans: list[dict], raw_source_spans: list[dict], source_documents: dict[str, dict]
) -> list[dict]:
    """Classify every raw object once, after its source spans are disposed."""
    spans_by_object: dict[str, list[dict]] = {}
    for span in page_text_spans:
        object_id = span.get("source_object_id")
        if object_id:
            spans_by_object.setdefault(str(object_id), []).append(span)
    raw_by_object: dict[tuple[str, int, int], list[dict]] = {}
    for raw in raw_source_spans:
        raw_by_object.setdefault(
            (str(raw["source_document_id"]), int(raw["page_number"]), int(raw["block_index"])), []
        ).append(raw)
    rows: list[dict] = []
    for source_object in source_objects:
        source = source_documents[str(source_object["source_document_id"])]
        spans = spans_by_object.get(str(source_object["source_object_id"]), [])
        raw_spans = raw_by_object.get(
            (str(source_object["source_document_id"]), int(source_object["page_number"]), int(source_object["block_index"])), []
        ) if source_object.get("block_index") is not None else []
        disposition, reason = _disposition(source_object, spans, raw_spans)
        rows.append(
            {
                **{key: value for key, value in source_object.items() if not key.startswith("_")},
                "source_sha256": source["sha256"],
                "source_pdf_path": source["path"],
                "object_role": "source_object",
                "text_span_ids": tuple(str(span["text_span_id"]) for span in spans),
                "target_refs": tuple(
                    sorted(
                        {
                            f"{span.get('promotion_target_type')}::{span.get('promotion_target_id')}"
                            for span in spans
                            if span.get("promotion_target_type") and span.get("promotion_target_id")
                        }
                    )
                ),
                "disposition": disposition,
                "reason": reason,
            }
        )
    return sorted(rows, key=lambda row: str(row["source_object_id"]))


def _disposition(source_object: dict, spans: list[dict], raw_spans: list[dict]) -> tuple[str, str]:
    if source_object.get("extraction_error"):
        return "extraction_failed", "raw_pdf_extraction_failed"
    if source_object.get("pdf_block_type") != 0:
        return "unsupported_nontext_object", "nontext_pdf_block"
    if not spans:
        raw_text = "".join(str(row.get("raw_text") or "") for row in raw_spans) or str(
            source_object.get("_raw_character_text") or ""
        )
        if raw_text and raw_text.isspace():
            return "excluded_nonlegal_object", "whitespace_only_text_block"
        if raw_text and not raw_text.replace("\u00ad", ""):
            return "excluded_nonlegal_object", "decorative_soft_hyphen_block"
        if raw_spans and all(row.get("classification") == "source_annotation_marker" for row in raw_spans):
            return "source_annotation_object", "source_annotation_marker_block"
        return "needs_review", "unclassified_meaningful_text_block"
    statuses = {str(span.get("promotion_status") or "") for span in spans}
    roles = {str(span.get("span_role") or "") for span in spans}
    if statuses == {"promoted_legal_unit"}:
        return (
            "promoted_normative_evidence" if roles == {"normative_text"} else "promoted_structural_evidence",
            "source_span_promotion",
        )
    if statuses == {"promoted_metadata"}:
        return "promoted_metadata", "source_span_promotion"
    if statuses == {"promoted_source_conflict"}:
        return "promoted_source_anomaly", "source_span_promotion"
    if all(span.get("exclusion_reason") for span in spans):
        return "excluded_nonlegal_object", "all_source_spans_excluded"
    if all(status and status != "needs_review" for status in statuses):
        return "resolved_mixed_dispositions", "all_source_spans_have_terminal_dispositions"
    return "needs_review", "mixed_or_unresolved_source_span_dispositions"
