from __future__ import annotations

from tjipto.corpora.uud.artifact_policy import ALLOWED_ARTIFACT_ORIGINS
from tjipto.corpora.uud.provenance_exceptions import (
    ACCEPTED_FALSE_POSITIVE_SEGMENTATION_PUNCTUATION,
    ACCEPTED_NONCANONICAL_SOURCE_CONFLICT_TRACE_ONLY,
    BUILDER_SLICING_LABEL_ISSUE_CONFIRMED,
    DUPLICATED_HEADING_ARTIFACT_ISSUE_CONFIRMED,
    UNRESOLVED_NEEDS_REVIEW,
    needs_review,
    review_category,
)


def _provenance_exception_health(chunks: list[dict], legal_units: list[dict], source_conflicts: list[dict]) -> dict:
    rows = [*legal_units, *chunks]
    needs_review_rows = [row for row in rows if needs_review(row)]
    category_counts = {
        category: sum(1 for row in rows if review_category(row) == category)
        for category in (
            ACCEPTED_FALSE_POSITIVE_SEGMENTATION_PUNCTUATION,
            ACCEPTED_NONCANONICAL_SOURCE_CONFLICT_TRACE_ONLY,
            BUILDER_SLICING_LABEL_ISSUE_CONFIRMED,
            DUPLICATED_HEADING_ARTIFACT_ISSUE_CONFIRMED,
            UNRESOLVED_NEEDS_REVIEW,
        )
    }
    noncanonical_conflicts = [
        row for row in source_conflicts if row.get("provenance_exception_category") == ACCEPTED_NONCANONICAL_SOURCE_CONFLICT_TRACE_ONLY
    ]
    validate_text_reviewed = [
        row
        for row in rows
        if review_category(row) == ACCEPTED_FALSE_POSITIVE_SEGMENTATION_PUNCTUATION
        and not row.get("text_span_ids")
        and bool(row.get("bbox_ids") or row.get("evidence_ids"))
    ]
    return {
        "total_reviewed_exceptions": sum(count for category, count in category_counts.items() if category != UNRESOLVED_NEEDS_REVIEW)
        + len(noncanonical_conflicts),
        "validate_text_provenance_needs_review_count": len(validate_text_reviewed),
        "accepted_false_positive_segmentation_punctuation_count": category_counts[ACCEPTED_FALSE_POSITIVE_SEGMENTATION_PUNCTUATION],
        "accepted_noncanonical_source_conflict_trace_only_count": category_counts[ACCEPTED_NONCANONICAL_SOURCE_CONFLICT_TRACE_ONLY]
        + len(noncanonical_conflicts),
        "builder_slicing_label_issue_confirmed_count": category_counts[BUILDER_SLICING_LABEL_ISSUE_CONFIRMED],
        "duplicated_heading_artifact_issue_confirmed_count": category_counts[DUPLICATED_HEADING_ARTIFACT_ISSUE_CONFIRMED],
        "unresolved_needs_review_count": category_counts[UNRESOLVED_NEEDS_REVIEW],
        "runtime_loadable_needs_review_count": sum(1 for row in needs_review_rows if row.get("runtime_loadable") is True),
        "noncanonical_conflict_trace_runtime_loadable_count": sum(
            1 for row in noncanonical_conflicts if row.get("runtime_loadable") is True
        ),
        "noncanonical_conflict_trace_canonical_use_allowed_count": sum(
            1 for row in noncanonical_conflicts if row.get("canonical_use_allowed") is True
        ),
    }


def _source_conflict_provenance_health(source_conflicts: list[dict]) -> dict:
    counts = {
        "source_conflict_count": len(source_conflicts),
        "renumbering_provenance_count": sum(1 for row in source_conflicts if row.get("source_anomaly_kind") == "renumbering_provenance"),
        "historical_to_canonical_mapping_count": sum(
            1 for row in source_conflicts if row.get("source_mapping_kind") == "historical_to_canonical_mapping"
        ),
        "source_marker_sequence_anomaly_count": sum(
            1 for row in source_conflicts if row.get("source_anomaly_kind") == "source_marker_sequence_anomaly"
        ),
        "missing_anchor_terms_count": sum(1 for row in source_conflicts if not row.get("anchor_terms")),
        "missing_query_anchor_terms_count": sum(1 for row in source_conflicts if not row.get("query_anchor_terms")),
        "missing_provenance_summary_count": sum(1 for row in source_conflicts if not str(row.get("provenance_summary") or "").strip()),
        "missing_final_authority_policy_count": sum(
            1 for row in source_conflicts if not str(row.get("final_authority_policy") or "").strip()
        ),
        "unknown_source_anomaly_kind_count": sum(
            1
            for row in source_conflicts
            if row.get("source_anomaly_kind")
            not in {"renumbering_provenance", "source_marker_sequence_anomaly", "typed_source_discrepancy"}
        ),
        "invalid_source_mapping_kind_count": sum(
            1
            for row in source_conflicts
            if row.get("source_anomaly_kind") == "renumbering_provenance"
            and row.get("source_mapping_kind") != "historical_to_canonical_mapping"
        ),
        "invalid_provenance_exception_category_count": sum(
            1
            for row in source_conflicts
            if row.get("source_anomaly_kind") == "source_marker_sequence_anomaly"
            and row.get("provenance_exception_category") != ACCEPTED_NONCANONICAL_SOURCE_CONFLICT_TRACE_ONLY
        ),
        "invalid_provenance_review_status_count": sum(
            1
            for row in source_conflicts
            if row.get("source_anomaly_kind") == "source_marker_sequence_anomaly" and row.get("provenance_review_status") != "reviewed"
        ),
        "final_evidence_available_count": sum(1 for row in source_conflicts if row.get("final_evidence_available") is True),
        "raw_provenance_exact_available_count": sum(
            1 for row in source_conflicts if row.get("provenance_bbox_status") == "exact_raw_provenance_bbox_available"
        ),
        "raw_provenance_partial_available_count": sum(
            1 for row in source_conflicts if row.get("provenance_bbox_status") == "partial_exact_raw_provenance_bbox_available"
        ),
        "raw_provenance_unavailable_count": sum(
            1 for row in source_conflicts if row.get("provenance_bbox_status") == "exact_raw_provenance_bbox_unavailable"
        ),
        "all_relevant_span_highlight_count": sum(
            1 for row in source_conflicts if row.get("provenance_highlight_scope") == "all_relevant_spans"
        ),
        "anchor_only_highlight_count": sum(1 for row in source_conflicts if row.get("provenance_highlight_scope") == "anchor_span_only"),
        "unknown_highlight_scope_count": sum(
            1
            for row in source_conflicts
            if row.get("provenance_highlight_scope") not in {"all_relevant_spans", "anchor_span_only", "unavailable"}
        ),
        "contradictory_failure_reason_count": sum(
            1
            for row in source_conflicts
            if row.get("provenance_bbox_status") in {"exact_raw_provenance_bbox_available", "partial_exact_raw_provenance_bbox_available"}
            and row.get("failure_reason") == "source_conflict_evidence_or_bbox_unavailable"
        ),
    }
    error_keys = {
        "missing_anchor_terms_count",
        "missing_query_anchor_terms_count",
        "missing_provenance_summary_count",
        "missing_final_authority_policy_count",
        "unknown_source_anomaly_kind_count",
        "invalid_source_mapping_kind_count",
        "invalid_provenance_exception_category_count",
        "invalid_provenance_review_status_count",
        "unknown_highlight_scope_count",
        "contradictory_failure_reason_count",
    }
    return {**counts, "status": "complete" if not any(counts[key] for key in error_keys) else "incomplete"}


def _structural_authority_contract_health(
    legal_units: list[dict], chunks: list[dict], graph_nodes: list[dict], graph_edges: list[dict], retrieval_units: list[dict]
) -> dict:
    units = {row.get("legal_unit_id"): row for row in legal_units}
    nodes = {row.get("node_id"): row for row in graph_nodes}
    unit_fields = {
        "stable_unit_id",
        "source_document_id",
        "unit_type",
        "structural_role",
        "ancestor_legal_unit_ids",
        "structural_depth",
        "sibling_order",
        "canonical_label",
    }
    missing_unit_fields = sum(1 for row in legal_units if any(field not in row for field in unit_fields))
    bad_parent_count = 0
    bad_ancestor_count = 0
    for row in legal_units:
        parent = row.get("parent_legal_unit_id")
        parents = list(row.get("parent_legal_unit_ids") or ())
        if len(parents) > 1 or parents != ([parent] if parent else []) or (parent and parent not in units):
            bad_parent_count += 1
        expected: list[str] = []
        current = parent
        seen = {row.get("legal_unit_id")}
        while current and current not in seen and current in units:
            seen.add(current)
            expected.append(current)
            current = units[current].get("parent_legal_unit_id")
        if (
            current in seen
            or list(reversed(expected)) != row.get("ancestor_legal_unit_ids")
            or len(expected) != row.get("structural_depth")
        ):
            bad_ancestor_count += 1
    chunk_fields = {"chunk_id", "legal_unit_id", "evidence_ids", "text_span_ids"}
    missing_chunk_fields = sum(1 for row in chunks if any(field not in row for field in chunk_fields))
    parent_final_count = 0
    required_edge_fields = {
        "source_id",
        "target_id",
        "object_role",
        "support_relation_ids",
        "support_evidence_ids",
        "support_exception_ids",
        "support_kind",
    }
    bad_edge_count = sum(
        1
        for row in graph_edges
        if any(field not in row for field in required_edge_fields)
        or row.get("source_id") not in nodes
        or row.get("target_id") not in nodes
        or row.get("object_role") != "graph_projection"
        or not row.get("support_kind")
    )
    bad_retrieval_trace_count = sum(
        1
        for row in retrieval_units
        if row.get("object_role") != "retrieval_index_record"
        or row.get("artifact_status") not in {"published", "excluded"}
        or not isinstance(row.get("page_locator"), dict)
        or row.get("evidence_id") is None
    )
    counts = {
        "missing_unit_fields_count": missing_unit_fields,
        "bad_parent_count": bad_parent_count,
        "bad_ancestor_count": bad_ancestor_count,
        "missing_chunk_fields_count": missing_chunk_fields,
        "parent_context_final_count": parent_final_count,
        "bad_graph_edge_count": bad_edge_count,
        "bad_retrieval_trace_count": bad_retrieval_trace_count,
    }
    return counts | {"status": "complete" if not any(counts.values()) else "incomplete"}


def _artifact_origin_health(manifest_files: dict[str, dict]) -> dict:
    rows = list(manifest_files.values())
    generated = [row for row in rows if row.get("origin") == "generated"]
    non_generated = [row for row in rows if row.get("origin") in {"carried_forward", "manual_review_artifact", "deprecated"}]
    return {
        "manifest_file_rows": len(rows),
        "files_with_origin": sum(1 for row in rows if row.get("origin")),
        "files_missing_origin": sum(1 for row in rows if not row.get("origin")),
        "invalid_origin_values": sum(1 for row in rows if row.get("origin") not in ALLOWED_ARTIFACT_ORIGINS),
        "generated_count": len(generated),
        "carried_forward_count": sum(1 for row in rows if row.get("origin") == "carried_forward"),
        "manual_review_artifact_count": sum(1 for row in rows if row.get("origin") == "manual_review_artifact"),
        "deprecated_count": sum(1 for row in rows if row.get("origin") == "deprecated"),
        "generated_missing_producer_count": sum(1 for row in generated if not str(row.get("producer") or "").strip()),
        "generated_missing_build_stage_count": sum(1 for row in generated if not str(row.get("build_stage") or "").strip()),
        "non_generated_missing_origin_reason_count": sum(1 for row in non_generated if not str(row.get("origin_reason") or "").strip()),
    }


def _edge_type_counts(edges: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in edges:
        edge_type = str(row.get("edge_type") or "")
        if edge_type:
            counts[edge_type] = counts.get(edge_type, 0) + 1
    return dict(sorted(counts.items()))
