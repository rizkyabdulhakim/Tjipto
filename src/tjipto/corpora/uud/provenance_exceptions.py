from __future__ import annotations

ACCEPTED_FALSE_POSITIVE_SEGMENTATION_PUNCTUATION = "accepted_false_positive_segmentation_punctuation"
ACCEPTED_NONCANONICAL_SOURCE_CONFLICT_TRACE_ONLY = "accepted_noncanonical_source_conflict_trace_only"
BUILDER_SLICING_LABEL_ISSUE_CONFIRMED = "builder_slicing_label_issue_confirmed"
DUPLICATED_HEADING_ARTIFACT_ISSUE_CONFIRMED = "duplicated_heading_artifact_issue_confirmed"
SOURCE_TEXT_ACCEPTED_NONRUNTIME_NO_EVIDENCE_BBOX = "source_text_accepted_nonruntime_no_evidence_bbox"
UNRESOLVED_NEEDS_REVIEW = "unresolved_needs_review"

NONCANONICAL_SOURCE_TYPO_REF = "source_typo_reference::uud_source_typo_reference_00001"
SEGMENTATION_BOUNDARY_LABELS = {
    "Perubahan Pertama Decision",
    "Perubahan Pertama Effective",
    "Perubahan Ketiga Scope",
    "Perubahan Ketiga Decision",
    "Perubahan Ketiga Effective",
    "Perubahan Keempat Scope",
    "Perubahan Keempat Decision",
    "Perubahan Keempat Effective",
}
LEGACY_SOURCE_TEXT_ACCEPTED_CHUNKS = {
    "uud_chunk_00005",
    "uud_chunk_00022",
    "uud_chunk_00101",
    "uud_chunk_00188",
}

PROVENANCE_REVIEW_CATEGORIES = (
    ACCEPTED_FALSE_POSITIVE_SEGMENTATION_PUNCTUATION,
    ACCEPTED_NONCANONICAL_SOURCE_CONFLICT_TRACE_ONLY,
    BUILDER_SLICING_LABEL_ISSUE_CONFIRMED,
    DUPLICATED_HEADING_ARTIFACT_ISSUE_CONFIRMED,
    SOURCE_TEXT_ACCEPTED_NONRUNTIME_NO_EVIDENCE_BBOX,
    UNRESOLVED_NEEDS_REVIEW,
)


def review_category(row: dict) -> str | None:
    if row.get("provenance_exception_category"):
        return row["provenance_exception_category"]
    label = row.get("unit_label") or _last_hierarchy_label(row)
    if label in SEGMENTATION_BOUNDARY_LABELS and row.get("failure_reason") == "text_span_exact_match_unavailable":
        return ACCEPTED_FALSE_POSITIVE_SEGMENTATION_PUNCTUATION
    if row.get("exclusion_ref") == NONCANONICAL_SOURCE_TYPO_REF:
        if label == "ATURAN TAMBAHAN source typo reference":
            return ACCEPTED_NONCANONICAL_SOURCE_CONFLICT_TRACE_ONLY
        if label == "Pasal I":
            return BUILDER_SLICING_LABEL_ISSUE_CONFIRMED
        if label == "Pasal III":
            return DUPLICATED_HEADING_ARTIFACT_ISSUE_CONFIRMED
    if row.get("chunk_id") in LEGACY_SOURCE_TEXT_ACCEPTED_CHUNKS or row.get("status") == SOURCE_TEXT_ACCEPTED_NONRUNTIME_NO_EVIDENCE_BBOX:
        return SOURCE_TEXT_ACCEPTED_NONRUNTIME_NO_EVIDENCE_BBOX
    if needs_review(row):
        return UNRESOLVED_NEEDS_REVIEW
    return None


def apply_review_category(row: dict) -> None:
    category = review_category(row)
    if not category or category == UNRESOLVED_NEEDS_REVIEW:
        return
    row["provenance_exception_category"] = category
    row["provenance_review_status"] = "reviewed"
    if category in {
        ACCEPTED_FALSE_POSITIVE_SEGMENTATION_PUNCTUATION,
        ACCEPTED_NONCANONICAL_SOURCE_CONFLICT_TRACE_ONLY,
        BUILDER_SLICING_LABEL_ISSUE_CONFIRMED,
        DUPLICATED_HEADING_ARTIFACT_ISSUE_CONFIRMED,
        SOURCE_TEXT_ACCEPTED_NONRUNTIME_NO_EVIDENCE_BBOX,
    }:
        row["validation_status"] = category
        row["failure_reason"] = category
    if category != ACCEPTED_FALSE_POSITIVE_SEGMENTATION_PUNCTUATION:
        row["canonical_use_allowed"] = False
    if category in {
        ACCEPTED_NONCANONICAL_SOURCE_CONFLICT_TRACE_ONLY,
        BUILDER_SLICING_LABEL_ISSUE_CONFIRMED,
        DUPLICATED_HEADING_ARTIFACT_ISSUE_CONFIRMED,
    }:
        row["text_span_ids"] = []
        row["grounding_status"] = "grounding_unavailable"
    row["runtime_loadable"] = False


def needs_review(row: dict) -> bool:
    values = {
        str(row.get("status") or ""),
        str(row.get("validation_status") or ""),
        str(row.get("failure_reason") or ""),
    }
    return any("needs_review" in value or value in {"grounding_unavailable", "text_span_exact_match_unavailable"} for value in values)


def _last_hierarchy_label(row: dict) -> str | None:
    hierarchy = row.get("hierarchy") or ()
    return hierarchy[-1] if hierarchy else None
