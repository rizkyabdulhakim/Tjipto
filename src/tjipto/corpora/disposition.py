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
