from __future__ import annotations

SPAN_DISPOSITION_FIELDS = (
    "span_role",
    "semantic_classification",
    "legal_force",
    "promotion_status",
    "promotion_target_type",
    "promotion_target_id",
    "exclusion_reason",
    "validation_basis",
    "review_status",
)

SPAN_ROLES = {
    "decision_clause",
    "effective_clause",
    "footnote_marker",
    "header_footer",
    "instrument_scope",
    "metadata_text",
    "needs_review",
    "nonlegal_artifact",
    "normative_text",
    "separator",
    "signatory_block",
    "source_conflict_trace",
    "structural_heading",
}

SEMANTIC_CLASSIFICATIONS = {
    "amendment_instrument_text",
    "decision_clause",
    "effective_clause",
    "footnote_marker",
    "header_footer",
    "needs_review",
    "nonlegal_artifact",
    "normative_constitutional_text",
    "separator",
    "session_institution_metadata",
    "signatory_block",
    "source_conflict_trace",
    "structural_heading",
}

LEGAL_FORCES = {
    "amendment_instrument",
    "canonical_normative",
    "historical_normative",
    "metadata_only",
    "nonlegal",
    "source_conflict_trace_only",
    "unknown_needs_review",
}

PROMOTION_STATUSES = {
    "excluded_nonlegal",
    "excluded_structural",
    "needs_review",
    "nonruntime_instrument_text",
    "promoted_legal_unit",
    "promoted_metadata",
    "promoted_source_conflict",
}

REVIEW_STATUSES = {
    "accepted",
    "needs_review",
    "reviewed",
}

PROMOTED_STATUSES = {
    "promoted_legal_unit",
    "promoted_metadata",
    "promoted_source_conflict",
}

EXCLUDED_STATUSES = {
    "excluded_structural",
    "excluded_nonlegal",
    "nonruntime_instrument_text",
    "needs_review",
}
