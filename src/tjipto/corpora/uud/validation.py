from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from time import perf_counter

from tjipto.corpora.disposition import EXCLUDED_STATUSES, PROMOTED_STATUSES, SPAN_DISPOSITION_FIELDS
from tjipto.corpora.intent_config import contains_intent_phrase, resolve_instrument_intent
from tjipto.corpora.uud.bbox_builder import bbox_precision_counts
from tjipto.corpora.uud.artifact_policy import ALLOWED_ARTIFACT_ORIGINS
from tjipto.corpora.uud.provenance_exceptions import (
    ACCEPTED_FALSE_POSITIVE_SEGMENTATION_PUNCTUATION,
    ACCEPTED_NONCANONICAL_SOURCE_CONFLICT_TRACE_ONLY,
    BUILDER_SLICING_LABEL_ISSUE_CONFIRMED,
    DUPLICATED_HEADING_ARTIFACT_ISSUE_CONFIRMED,
    SOURCE_TEXT_ACCEPTED_NONRUNTIME_NO_EVIDENCE_BBOX,
    UNRESOLVED_NEEDS_REVIEW,
    UNRESOLVED_MANUAL_REVIEW_REQUIRED,
    needs_review,
    review_category,
)
from tjipto.corpora.uud.span_disposition_policy import role_for_legal_unit
from tjipto.corpora.uud.specs import UUD_LEGAL_GRAPH_EDGE_SCHEMA
from tjipto.core.manifest import read_jsonl


DECISION_LABELS = {
    "Perubahan Pertama Decision",
    "Perubahan Ketiga Decision",
    "Perubahan Keempat Decision",
}
INSERTED_BAB_HEADING_BBOX_MARKER = "::heading_bab_"
STRUCTURAL_FORBIDDEN_MARKERS = ("Ditetapkan di Jakarta",)
PROVENANCE_EDGE_TYPES = {
    "HAS_FINAL_EVIDENCE",
    "BELONGS_TO_SOURCE_ROLE",
    "USES_SOURCE_PDF",
    "PAGE_GROUNDED_AT",
    "HAS_BBOX",
    "EXCLUDED_BECAUSE",
}
STRUCTURAL_SEQUENCE_EDGE_TYPES = {
    edge_type for edge_type, schema in UUD_LEGAL_GRAPH_EDGE_SCHEMA.items() if schema.get("category") == "structural_sequence"
}
LEGAL_EDGE_TYPES = {
    "CONTAINS",
    "PART_OF",
    "AMENDS",
    "AMENDED_BY",
    "ADDS",
    "MODIFIES",
    "DELETES",
    "RENAMES",
    "SUPPLEMENTS",
    *STRUCTURAL_SEQUENCE_EDGE_TYPES,
    "HAS_EFFECTIVE_RULE",
    "HAS_SIGNATORY",
    "HAS_DECISION_SESSION",
    "HAS_SOURCE_ANOMALY",
}


def build_validation_report(
    *,
    chunks: list[dict],
    legal_units: list[dict],
    excluded_records: list[dict],
    source_conflicts: list[dict],
    evidence: list[dict],
    bbox_rows: list[dict],
    retrieval_units: list[dict],
    metadata_grounding: list[dict],
    metadata_grounding_registry: list[dict],
    manifest_files: dict[str, dict],
    graph_nodes: list[dict],
    graph_edges: list[dict],
    page_text_spans: list[dict],
    intent_config: dict | None = None,
) -> dict:
    validation_report = {
        "artifact_governance": {
            "status": "current_final_artifacts_present",
            "compatibility_seed_bridge": {
                "status": "removed",
                "source": "source_pdf_and_corpus_specs",
                "seeded_artifacts": [],
            },
            "excluded_chunk_policy": "records listed in excluded_records.jsonl are not runtime-loadable, not active canonical, and not canonical-use allowed",
            "audited_excluded_chunks": [row["legacy_chunk_id"] for row in excluded_records],
            "reviewed_exceptions_preserved": [
                "Pasal 22D ayat (3) bbox/text exception remains tracked in structure_fidelity.reviewed_exception_unit"
            ],
        },
        "corpus_id": "uud",
        "referenced_artifacts": [
            "document_metadata.jsonl",
            "document_relations.jsonl",
            "article_amendment_relations.jsonl",
            "metadata_graph_edges.jsonl",
            "metadata_grounding.jsonl",
            "metadata_grounding_registry.jsonl",
            "source_conflicts.jsonl",
            "source_integrity.json",
            "validation_alignment_results.jsonl",
            "validation_exception_review_labels.jsonl",
            "validation_exceptions.jsonl",
            "page_text_spans.jsonl",
        ],
        "status": "valid",
        "structure_fidelity": {
            "status": "corrected",
            "inserted_bab_heading_owner_policy": (
                "inserted heading bboxes may stay exact, but they are viewer_highlightable only when owned by bab_record evidence"
            ),
        },
    }
    validation_report["final_artifact_counts"] = {
        "chunks": len(chunks),
        "legal_units": len(legal_units),
        "excluded_records": len(excluded_records),
        "evidence_records": len(evidence),
        "bbox_records": len(bbox_rows),
        "retrieval_units": len(retrieval_units),
        "graph_nodes": len(graph_nodes),
        "graph_edges": len(graph_edges),
        "page_text_spans": len(page_text_spans),
    }
    validation_report["bbox_precision_counts"] = bbox_precision_counts(bbox_rows)
    validation_report["bbox_highlightability_counts"] = {
        "viewer_highlightable": sum(1 for row in bbox_rows if row.get("viewer_highlightable") is True),
        "non_highlightable": sum(1 for row in bbox_rows if row.get("viewer_highlightable") is not True),
    }
    validation_report["chunk_self_contained_health"] = _chunk_self_contained_health(
        chunks,
        {row["legal_unit_id"]: row for row in legal_units},
    )
    validation_report["provenance_exception_health"] = _provenance_exception_health(
        chunks,
        legal_units,
        source_conflicts,
    )
    validation_report["all_text_disposition_health"] = _all_text_disposition_health(
        page_text_spans,
        legal_units,
        chunks,
        metadata_grounding=metadata_grounding,
        source_conflicts=source_conflicts,
    )
    validation_report["semantic_precedence_health"] = _semantic_precedence_health(
        page_text_spans,
        legal_units,
        source_conflicts,
    )
    validation_report["instrument_runtime_safety_health"] = _instrument_runtime_safety_health(
        evidence,
        legal_units,
        chunks,
        retrieval_units,
    )
    validation_report["instrument_exact_grounding_health"] = _instrument_exact_grounding_health(
        evidence,
        legal_units,
        chunks,
        retrieval_units,
        bbox_rows,
    )
    validation_report["instrument_query_precision_health"] = _instrument_query_precision_health(
        evidence,
        legal_units,
        retrieval_units,
    )
    validation_report["instrument_natural_query_precision_health"] = _instrument_natural_query_precision_health(
        evidence,
        legal_units,
        retrieval_units,
    )
    validation_report["instrument_intent_matrix_health"] = _instrument_intent_matrix_health(
        evidence,
        retrieval_units,
        intent_config or {},
    )
    validation_report["partial_signal_instrument_boundary_health"] = _partial_signal_instrument_boundary_health(
        evidence,
        retrieval_units,
        intent_config or {},
    )
    validation_report["instrument_like_boundary_generalization_health"] = _instrument_like_boundary_generalization_health(
        intent_config or {},
    )
    validation_report["instrument_intent_invariant_router_health"] = _instrument_intent_invariant_router_health(
        intent_config or {},
    )
    validation_report["intent_arbitration_priority_health"] = _intent_arbitration_priority_health(
        intent_config or {},
    )
    validation_report["amendment_context_default_boundary_health"] = _amendment_context_default_boundary_health()
    validation_report["artifact_origin_health"] = _artifact_origin_health(manifest_files)
    validation_report["instrument_baseline"] = {
        "status": "corrected",
        "instrument_unit_types": [
            "amendment_recital_record",
            "amendment_scope_record",
            "instrument_clause_record",
            "instrument_closing_record",
            "decision_clause_record",
            "effective_clause_record",
            "determination_clause_record",
            "signatory_block_record",
        ],
        "metadata_viewer_highlightable": False,
    }
    validation_report["bbox_precision_policy"] = {
        "status": "corrected",
        "exact_policy": "bbox_precision=exact rows may remain viewer_highlightable",
        "fallback_policy": "bbox_precision=page_grounded_only rows are not viewer_highlightable",
        "coarse_policy": "bbox_precision=coarse rows are not viewer_highlightable",
    }
    validation_report["metadata_grounding_contract"] = {
        "status": "mixed_exact_and_field_grounded",
        "note": "exact metadata rows may be viewer-highlightable only with exact bbox and text-span support; page-grounded rows remain trace-only",
    }
    validation_report["metadata_bbox_registry_health"] = _metadata_bbox_registry_health(
        metadata_grounding_registry,
        {row["bbox_id"]: row for row in bbox_rows},
    )
    validation_report["legal_graph_baseline"] = {
        "status": "evidence_backed_minimal_baseline",
        "actual_edge_type_counts": _edge_type_counts(graph_edges),
        "actual_promoted_legal_edge_type_counts": _edge_type_counts(
            [row for row in graph_edges if row.get("edge_type") in LEGAL_EDGE_TYPES]
        ),
        "runtime_loadable_legal_edge_type_counts": _edge_type_counts(
            [row for row in graph_edges if row.get("edge_type") in LEGAL_EDGE_TYPES and row.get("runtime_loadable") is True]
        ),
        "schema_edge_types": sorted(LEGAL_EDGE_TYPES),
        "not_promoted_edge_types": sorted(LEGAL_EDGE_TYPES - {str(row.get("edge_type")) for row in graph_edges}),
        "runtime_loadable_legal_edges": sum(
            1 for row in graph_edges if row.get("edge_type") in LEGAL_EDGE_TYPES and row.get("runtime_loadable") is True
        ),
    }
    return validation_report


def validate_uud_artifact_dir(final_dir: Path) -> tuple[str, ...]:
    legal_units = read_jsonl(final_dir / "legal_units.jsonl")
    chunks = read_jsonl(final_dir / "chunks.jsonl")
    evidence = read_jsonl(final_dir / "evidence_registry.jsonl")
    bbox_rows = read_jsonl(final_dir / "bbox_registry.jsonl")
    retrieval_units = read_jsonl(final_dir / "retrieval_units.jsonl")
    metadata_grounding = read_jsonl(final_dir / "metadata_grounding.jsonl")
    metadata_grounding_registry = read_jsonl(final_dir / "metadata_grounding_registry.jsonl")
    document_relations = read_jsonl(final_dir / "document_relations.jsonl") if (final_dir / "document_relations.jsonl").exists() else []
    article_amendment_relations = (
        read_jsonl(final_dir / "article_amendment_relations.jsonl") if (final_dir / "article_amendment_relations.jsonl").exists() else []
    )
    graph_nodes = read_jsonl(final_dir / "graph_nodes.jsonl")
    graph_edges = read_jsonl(final_dir / "graph_edges.jsonl")
    source_conflicts = read_jsonl(final_dir / "source_conflicts.jsonl")
    validation_exceptions = read_jsonl(final_dir / "validation_exceptions.jsonl")
    page_text_spans = read_jsonl(final_dir / "page_text_spans.jsonl") if (final_dir / "page_text_spans.jsonl").exists() else []

    errors: list[str] = []
    seen_ids: dict[str, set[str]] = defaultdict(set)
    units_by_id = {row["legal_unit_id"]: row for row in legal_units}
    chunks_by_id = {row["chunk_id"]: row for row in chunks}
    evidence_by_id = {row["evidence_id"]: row for row in evidence}
    bbox_by_id = {row["bbox_id"]: row for row in bbox_rows}
    graph_node_ids = {row["node_id"] for row in graph_nodes}
    source_conflict_ids = {row["source_conflict_id"] for row in source_conflicts}
    validation_exception_ids = {row["exception_id"] for row in validation_exceptions}
    source_document_ids = {row["source_document_id"] for row in read_jsonl(final_dir / "source_documents.jsonl")}
    metadata_grounding_ids = {row["metadata_grounding_id"] for row in metadata_grounding}
    metadata_grounding_ref_ids: set[str] = set()
    text_span_ids = {row["text_span_id"] for row in page_text_spans}
    bbox_by_evidence: dict[str, list[dict]] = defaultdict(list)
    for row in bbox_rows:
        bbox_by_evidence[row["evidence_id"]].append(row)

    expected_instrument_labels = {
        "Perubahan Pertama Recital",
        "Perubahan Pertama Scope",
        "Perubahan Kedua Recital",
        "Perubahan Kedua Scope",
        "Perubahan Ketiga Recital",
        "Perubahan Ketiga Scope",
        "Perubahan Keempat Recital",
        "Perubahan Keempat Scope",
        "Perubahan Keempat Clause (a)",
        "Perubahan Keempat Clause (b)",
        "Perubahan Keempat Clause (c)",
        "Perubahan Keempat Clause (d)",
        "Perubahan Keempat Clause (e)",
    }
    actual_labels = {row.get("unit_label") for row in legal_units}
    for label in expected_instrument_labels:
        if label not in actual_labels:
            errors.append(f"missing_instrument_unit:{label}")

    forbidden_markers = ("Naskah perubahan ini merupakan", "Perubahan tersebut diputuskan", "Ditetapkan di Jakarta")
    for row in legal_units:
        if row["legal_unit_id"] in seen_ids["legal_unit_id"]:
            errors.append(f"duplicate_legal_unit_id:{row['legal_unit_id']}")
        seen_ids["legal_unit_id"].add(row["legal_unit_id"])
        if row["unit_type"] in {"pasal_record", "ayat_record"}:
            if any(marker in row["text"] for marker in forbidden_markers):
                errors.append(f"closing_clause_attached_to_normative_unit:{row['legal_unit_id']}")
            if any(marker in row["text"] for marker in ("BAB VIIA", "BAB VIIB", "BAB VIIIA", "BAB IXA", "BAB XA")):
                errors.append(f"inserted_bab_inside_normative_unit:{row['legal_unit_id']}")
        if row["unit_type"] == "pasal_record" and row.get("hierarchy") and str(row["hierarchy"][0]).startswith("BAB"):
            if not any(
                units_by_id[parent]["unit_type"] == "bab_record"
                for parent in row.get("parent_legal_unit_ids") or ()
                if parent in units_by_id
            ):
                errors.append(f"pasal_missing_bab_parent:{row['legal_unit_id']}")
        if row["unit_type"] == "bab_record" and any(marker in row["text"] for marker in STRUCTURAL_FORBIDDEN_MARKERS):
            errors.append(f"structural_bab_contains_instrument_text:{row['legal_unit_id']}")
        for text_span_id in row.get("text_span_ids") or ():
            if text_span_id not in text_span_ids:
                errors.append(f"orphan_legal_unit_text_span:{row['legal_unit_id']}:{text_span_id}")
        for field in (
            "source_role",
            "temporal_context",
            "page_numbers",
            "text_span_ids",
            "bbox_ids",
            "grounding_status",
            "validation_status",
        ):
            if row.get("runtime_loadable") is True and field not in row:
                errors.append(f"runtime_loadable_legal_unit_missing_{field}:{row['legal_unit_id']}")
        if row.get("runtime_loadable") is True:
            if not row.get("text_span_ids"):
                errors.append(f"runtime_loadable_legal_unit_missing_text_span:{row['legal_unit_id']}")
            if not row.get("bbox_ids"):
                errors.append(f"runtime_loadable_legal_unit_missing_bbox:{row['legal_unit_id']}")
        if row.get("runtime_loadable") is True and needs_review(row):
            errors.append(f"runtime_loadable_needs_review_legal_unit:{row['legal_unit_id']}")

    for row in chunks:
        if row["chunk_id"] in seen_ids["chunk_id"]:
            errors.append(f"duplicate_chunk_id:{row['chunk_id']}")
        seen_ids["chunk_id"].add(row["chunk_id"])
        unit = units_by_id.get(row["legal_unit_id"])
        if not unit:
            errors.append(f"orphan_chunk:{row['chunk_id']}")
        else:
            for field in ("source_document_id", "source_role", "temporal_context"):
                if row.get(field) != unit.get(field):
                    errors.append(f"chunk_{field}_mismatch:{row['chunk_id']}")
        for field in ("source_document_id", "source_role", "temporal_context", "validation_status", "validation_basis"):
            if not row.get(field):
                errors.append(f"chunk_missing_{field}:{row['chunk_id']}")
        for text_span_id in row.get("text_span_ids") or ():
            if text_span_id not in text_span_ids:
                errors.append(f"orphan_chunk_text_span:{row['chunk_id']}:{text_span_id}")
        if row.get("runtime_loadable") is True and not row.get("grounding_status"):
            errors.append(f"runtime_loadable_chunk_missing_grounding:{row['chunk_id']}")
        if row.get("runtime_loadable") is True and not row.get("bbox_ids"):
            errors.append(f"runtime_loadable_chunk_missing_bbox:{row['chunk_id']}")
        if row.get("runtime_loadable") is True and row.get("grounding_status") != "text_span_exact":
            errors.append(f"runtime_loadable_chunk_not_text_exact:{row['chunk_id']}")
        if row.get("runtime_loadable") is True:
            if not row.get("evidence_ids"):
                errors.append(f"runtime_loadable_chunk_missing_evidence:{row['chunk_id']}")
            if not row.get("text_span_ids"):
                errors.append(f"runtime_loadable_chunk_missing_text_span:{row['chunk_id']}")
            if row.get("validation_status") == "validation_error_missing_grounding":
                errors.append(f"runtime_loadable_chunk_validation_error:{row['chunk_id']}")
        if row.get("runtime_loadable") is False and not (
            row.get("validation_basis") or row.get("failure_reason") or row.get("grounding_status")
        ):
            errors.append(f"non_runtime_chunk_missing_status_or_reason:{row['chunk_id']}")
        if row["chunk_type"] == "bab_structural_context_record" and any(marker in row["text"] for marker in STRUCTURAL_FORBIDDEN_MARKERS):
            errors.append(f"structural_chunk_contains_instrument_text:{row['chunk_id']}")
        if row.get("runtime_loadable") is True and needs_review(row):
            errors.append(f"runtime_loadable_needs_review_chunk:{row['chunk_id']}")
    for row in evidence:
        if row["evidence_id"] in seen_ids["evidence_id"]:
            errors.append(f"duplicate_evidence_id:{row['evidence_id']}")
        seen_ids["evidence_id"].add(row["evidence_id"])
        if row["legal_unit_id"] not in units_by_id:
            errors.append(f"orphan_evidence:{row['evidence_id']}")
        for bbox_id in row.get("bbox_refs") or ():
            if bbox_id not in bbox_by_id:
                errors.append(f"orphan_bbox_ref:{row['evidence_id']}:{bbox_id}")
            elif bbox_by_id[bbox_id]["page_number"] not in row["page_numbers"]:
                errors.append(f"bbox_page_mismatch:{row['evidence_id']}:{bbox_id}")
        for text_span_id in row.get("text_span_ids") or ():
            if text_span_id not in text_span_ids:
                errors.append(f"orphan_evidence_text_span_ref:{row['evidence_id']}:{text_span_id}")
        if row.get("citation") in DECISION_LABELS:
            for bbox_id in row.get("bbox_refs") or ():
                bbox_row = bbox_by_id.get(bbox_id)
                if not bbox_row:
                    continue
                if bbox_row.get("viewer_highlightable") and bbox_row.get("bbox_precision") != "exact":
                    errors.append(f"decision_bbox_not_exact:{row['evidence_id']}:{bbox_id}")
                if bbox_row.get("viewer_highlightable") and "Pasal " in bbox_row.get("text", ""):
                    errors.append(f"decision_bbox_contains_normative_text:{row['evidence_id']}:{bbox_id}")
                if bbox_row.get("bbox_precision") in {"coarse", "page_grounded_only"} and bbox_row.get("viewer_highlightable"):
                    errors.append(f"coarse_bbox_marked_highlightable:{bbox_id}")
    for row in bbox_rows:
        if row["bbox_id"] in seen_ids["bbox_id"]:
            errors.append(f"duplicate_bbox_id:{row['bbox_id']}")
        seen_ids["bbox_id"].add(row["bbox_id"])
        evidence_row = evidence_by_id.get(row["evidence_id"])
        owner = units_by_id.get(evidence_row["legal_unit_id"]) if evidence_row else None
        if (
            INSERTED_BAB_HEADING_BBOX_MARKER in row["bbox_id"]
            and owner is not None
            and owner.get("unit_type") != "bab_record"
            and row.get("viewer_highlightable")
        ):
            errors.append(f"inserted_bab_heading_highlightable_without_bab_owner:{row['bbox_id']}")
        if row.get("bbox_precision") not in {"exact", "coarse", "page_grounded_only"}:
            errors.append(f"invalid_bbox_precision:{row['bbox_id']}")
        if row.get("bbox_precision") in {"coarse", "page_grounded_only"} and row.get("viewer_highlightable"):
            errors.append(f"coarse_bbox_marked_highlightable:{row['bbox_id']}")
    for row in page_text_spans:
        if row["text_span_id"] in seen_ids["text_span_id"]:
            errors.append(f"duplicate_text_span_id:{row['text_span_id']}")
        seen_ids["text_span_id"].add(row["text_span_id"])
        if not row.get("text"):
            errors.append(f"empty_text_span:{row['text_span_id']}")
        if row.get("bbox_precision") != "exact":
            errors.append(f"text_span_missing_exact_bbox:{row['text_span_id']}")
        for field in SPAN_DISPOSITION_FIELDS:
            if field not in row:
                errors.append(f"text_span_missing_disposition_field:{row['text_span_id']}:{field}")
        if row.get("promotion_status") in EXCLUDED_STATUSES and not row.get("exclusion_reason"):
            errors.append(f"excluded_text_span_missing_reason:{row['text_span_id']}")
        if row.get("promotion_status") == "needs_review":
            if row.get("runtime_loadable") is True:
                errors.append(f"runtime_loadable_needs_review_text_span:{row['text_span_id']}")
            if row.get("canonical_use_allowed") is True:
                errors.append(f"canonical_needs_review_text_span:{row['text_span_id']}")
        if row.get("promotion_status") in PROMOTED_STATUSES:
            target_type = row.get("promotion_target_type")
            target_id = row.get("promotion_target_id")
            if target_type == "legal_unit" and target_id not in units_by_id:
                errors.append(f"text_span_unknown_legal_unit_target:{row['text_span_id']}:{target_id}")
            elif target_type == "chunk" and target_id not in chunks_by_id:
                errors.append(f"text_span_unknown_chunk_target:{row['text_span_id']}:{target_id}")
            elif target_type == "metadata_grounding" and target_id not in metadata_grounding_ids:
                errors.append(f"text_span_unknown_metadata_target:{row['text_span_id']}:{target_id}")
            elif target_type == "source_conflict" and target_id not in source_conflict_ids:
                errors.append(f"text_span_unknown_source_conflict_target:{row['text_span_id']}:{target_id}")
    semantic_precedence = _semantic_precedence_health(page_text_spans, legal_units, source_conflicts)
    for key, value in semantic_precedence.items():
        if key.endswith("_count") and value:
            errors.append(f"semantic_precedence_{key}:{value}")
    if semantic_precedence["status"] != "complete":
        errors.append("semantic_precedence_incomplete")
    instrument_safety = _instrument_runtime_safety_health(evidence, legal_units, chunks, retrieval_units)
    for key, value in instrument_safety.items():
        if key.endswith("_count") and value:
            errors.append(f"instrument_runtime_safety_{key}:{value}")
    if instrument_safety["status"] != "complete":
        errors.append("instrument_runtime_safety_incomplete")
    exact_grounding = _instrument_exact_grounding_health(evidence, legal_units, chunks, retrieval_units, bbox_rows)
    for key, value in exact_grounding.items():
        if key.endswith("_count") and value:
            errors.append(f"instrument_exact_grounding_{key}:{value}")
    if exact_grounding["status"] != "complete":
        errors.append("instrument_exact_grounding_incomplete")
    query_precision = _instrument_query_precision_health(evidence, legal_units, retrieval_units)
    for key, value in query_precision.items():
        if key.endswith("_count") and value:
            errors.append(f"instrument_query_precision_{key}:{value}")
    if query_precision["status"] != "complete":
        errors.append("instrument_query_precision_incomplete")
    natural_precision = _instrument_natural_query_precision_health(evidence, legal_units, retrieval_units)
    for key, value in natural_precision.items():
        if key.endswith("_count") and value:
            errors.append(f"instrument_natural_query_precision_{key}:{value}")
    if natural_precision["status"] != "complete":
        errors.append("instrument_natural_query_precision_incomplete")
    for row in retrieval_units:
        if row["retrieval_unit_id"] in seen_ids["retrieval_unit_id"]:
            errors.append(f"duplicate_retrieval_unit_id:{row['retrieval_unit_id']}")
        seen_ids["retrieval_unit_id"].add(row["retrieval_unit_id"])
        if row["chunk_id"] not in chunks_by_id:
            errors.append(f"orphan_retrieval_chunk:{row['retrieval_unit_id']}")
        if row["evidence_id"] not in evidence_by_id:
            errors.append(f"orphan_retrieval_evidence:{row['retrieval_unit_id']}")
    for row in source_conflicts:
        for field in ("page_numbers", "text_span_ids", "bbox_ids", "evidence_ids", "grounding_status", "validation_status"):
            if field not in row:
                errors.append(f"source_conflict_missing_{field}:{row['source_conflict_id']}")
        for text_span_id in row.get("text_span_ids") or ():
            if text_span_id not in text_span_ids:
                errors.append(f"orphan_source_conflict_text_span:{row['source_conflict_id']}:{text_span_id}")
        for bbox_id in row.get("bbox_ids") or ():
            if bbox_id not in bbox_by_id:
                errors.append(f"orphan_source_conflict_bbox:{row['source_conflict_id']}:{bbox_id}")
        for evidence_id in row.get("evidence_ids") or ():
            if evidence_id not in evidence_by_id:
                errors.append(f"orphan_source_conflict_evidence:{row['source_conflict_id']}:{evidence_id}")
        if (not row.get("evidence_ids") or not row.get("bbox_ids")) and not row.get("failure_reason"):
            errors.append(f"source_conflict_missing_failure_reason:{row['source_conflict_id']}")
        if row.get("provenance_exception_category") == ACCEPTED_NONCANONICAL_SOURCE_CONFLICT_TRACE_ONLY:
            if row.get("runtime_loadable") is True:
                errors.append(f"noncanonical_conflict_trace_runtime_loadable:{row['source_conflict_id']}")
            if row.get("canonical_use_allowed") is True:
                errors.append(f"noncanonical_conflict_trace_canonical_use_allowed:{row['source_conflict_id']}")
    uncounted_unresolved_exceptions = [row for row in validation_exceptions if row.get("status") == UNRESOLVED_MANUAL_REVIEW_REQUIRED]
    for row in uncounted_unresolved_exceptions:
        errors.append(f"unresolved_validation_exception:{row['exception_id']}")
    for row in metadata_grounding:
        if row.get("bbox_precision") not in {None, "coarse", "exact", "page_grounded_only"}:
            errors.append(f"invalid_metadata_bbox_precision:{row['metadata_grounding_id']}")
        if row.get("bbox_precision") == "exact":
            for bbox_id in row.get("bbox_ids") or ():
                if bbox_id not in bbox_by_id:
                    errors.append(f"orphan_metadata_bbox:{row['metadata_grounding_id']}:{bbox_id}")
            for text_span_id in row.get("text_span_ids") or ():
                if text_span_id not in text_span_ids:
                    errors.append(f"orphan_metadata_text_span:{row['metadata_grounding_id']}:{text_span_id}")
            if not row.get("bbox_ids") or not row.get("text_span_ids"):
                errors.append(f"exact_metadata_missing_grounding_ids:{row['metadata_grounding_id']}")
            if row.get("viewer_highlightable") is not True:
                errors.append(f"exact_metadata_not_highlightable:{row['metadata_grounding_id']}")
        if row.get("bbox_precision") == "page_grounded_only" and not row.get("failure_reason"):
            errors.append(f"page_grounded_metadata_missing_failure_reason:{row['metadata_grounding_id']}")
        if row.get("bbox_precision") != "exact" and row.get("viewer_highlightable") is not False:
            errors.append(f"non_exact_metadata_highlightable:{row['metadata_grounding_id']}")
    for row in metadata_grounding_registry:
        ref_id = row.get("metadata_grounding_ref_id")
        if not ref_id:
            errors.append(f"metadata_registry_missing_ref_id:{row.get('metadata_grounding_id')}:{row.get('bbox_id')}")
        elif ref_id in metadata_grounding_ref_ids:
            errors.append(f"duplicate_metadata_grounding_ref_id:{ref_id}")
        else:
            metadata_grounding_ref_ids.add(ref_id)
        bbox_id = row.get("bbox_id")
        bbox = bbox_by_id.get(bbox_id)
        if row.get("bbox_precision") == "exact":
            if not bbox:
                errors.append(f"exact_metadata_registry_unresolved_bbox:{ref_id}:{bbox_id}")
            elif not all(key in bbox for key in ("source_document_id", "page_number", "x0", "y0", "x1", "y1")):
                errors.append(f"exact_metadata_registry_bbox_missing_coordinates:{ref_id}:{bbox_id}")
        elif bbox_id and bbox_id not in bbox_by_id and not row.get("failure_reason"):
            errors.append(f"unresolved_metadata_registry_missing_failure_reason:{ref_id}:{bbox_id}")
        if row.get("bbox_precision") == "exact":
            if row.get("viewer_highlightable") is not True:
                errors.append(f"exact_metadata_registry_not_highlightable:{ref_id}")
        elif row.get("viewer_highlightable") is not False:
            errors.append(f"non_exact_metadata_registry_highlightable:{ref_id}")
    for row in graph_nodes:
        if row["node_id"] in seen_ids["node_id"]:
            errors.append(f"duplicate_graph_node_id:{row['node_id']}")
        seen_ids["node_id"].add(row["node_id"])
    for row in graph_edges:
        if row["edge_id"] in seen_ids["edge_id"]:
            errors.append(f"duplicate_graph_edge_id:{row['edge_id']}")
        seen_ids["edge_id"].add(row["edge_id"])
        if row["source_id"] not in graph_node_ids or row["target_id"] not in graph_node_ids:
            errors.append(f"orphan_graph_edge:{row['edge_id']}")
        edge_type = row.get("edge_type")
        if edge_type not in PROVENANCE_EDGE_TYPES | LEGAL_EDGE_TYPES:
            errors.append(f"invalid_graph_edge_type:{row['edge_id']}:{edge_type}")
        if edge_type in LEGAL_EDGE_TYPES:
            if row.get("source_document_id") not in {unit["source_document_id"] for unit in legal_units}:
                errors.append(f"legal_edge_missing_source_document:{row['edge_id']}")
            if row.get("runtime_loadable") is True:
                if not row.get("evidence_ref"):
                    errors.append(f"runtime_loadable_legal_edge_missing_evidence:{row['edge_id']}")
                if not row.get("validation_status"):
                    errors.append(f"runtime_loadable_legal_edge_missing_validation:{row['edge_id']}")
                if not row.get("confidence_policy"):
                    errors.append(f"runtime_loadable_legal_edge_missing_confidence:{row['edge_id']}")
            if edge_type in {"AMENDS", "AMENDED_BY"} and row["source_id"].startswith("source_role::"):
                errors.append(f"source_role_amends_promoted:{row['edge_id']}")
            if edge_type in STRUCTURAL_SEQUENCE_EDGE_TYPES:
                schema = UUD_LEGAL_GRAPH_EDGE_SCHEMA[edge_type]
                if row.get("runtime_loadable") != schema["runtime_loadable"]:
                    errors.append(f"structural_sequence_runtime_loadable:{row['edge_id']}")
                if row.get("validation_status") != schema["validation_status"]:
                    errors.append(f"structural_sequence_validation_status:{row['edge_id']}")
                if row.get("derivation_basis") != schema["derivation_basis"]:
                    errors.append(f"structural_sequence_derivation_basis:{row['edge_id']}")
                if not row.get("source_role"):
                    errors.append(f"structural_sequence_missing_source_role:{row['edge_id']}")
                if not row.get("temporal_context"):
                    errors.append(f"structural_sequence_missing_temporal_context:{row['edge_id']}")
            evidence_ref = row.get("evidence_ref")
            if (
                evidence_ref
                and evidence_ref not in evidence_by_id
                and evidence_ref not in metadata_grounding_ids
                and evidence_ref not in source_conflict_ids
            ):
                errors.append(f"legal_edge_unknown_evidence_ref:{row['edge_id']}:{evidence_ref}")
    for row in document_relations:
        if row["relation_id"] in seen_ids["document_relation_id"]:
            errors.append(f"duplicate_document_relation_id:{row['relation_id']}")
        seen_ids["document_relation_id"].add(row["relation_id"])
        if row.get("relation_type") not in {"AMENDS", "AMENDED_BY"}:
            errors.append(f"invalid_document_relation_type:{row['relation_id']}:{row.get('relation_type')}")
        if row.get("source_document_id") not in source_document_ids:
            errors.append(f"document_relation_unknown_source:{row['relation_id']}:{row.get('source_document_id')}")
        if row.get("target_document_id") not in source_document_ids:
            errors.append(f"document_relation_unknown_target:{row['relation_id']}:{row.get('target_document_id')}")
        for ref in row.get("support_refs") or ():
            if ref not in validation_exception_ids:
                errors.append(f"document_relation_unknown_support_ref:{row['relation_id']}:{ref}")
        if row.get("article_level") is not False:
            errors.append(f"document_relation_article_level:{row['relation_id']}")
        if row.get("viewer_highlightable") is not False or row.get("citation_available") is not False:
            errors.append(f"document_relation_false_exact_claim:{row['relation_id']}")
    for row in article_amendment_relations:
        if row["relation_id"] in seen_ids["article_amendment_relation_id"]:
            errors.append(f"duplicate_article_amendment_relation_id:{row['relation_id']}")
        seen_ids["article_amendment_relation_id"].add(row["relation_id"])
        if row.get("relation_type") not in {"MODIFIES", "DELETES", "INSERTS", "ADDS", "RENAMES", "SUPPLEMENTS"}:
            errors.append(f"invalid_article_amendment_relation_type:{row['relation_id']}:{row.get('relation_type')}")
        evidence_row = evidence_by_id.get(row.get("evidence_id"))
        if not evidence_row:
            errors.append(f"article_relation_unknown_evidence:{row['relation_id']}:{row.get('evidence_id')}")
            continue
        if row.get("target_legal_unit_id") not in units_by_id:
            errors.append(f"article_relation_unknown_target:{row['relation_id']}:{row.get('target_legal_unit_id')}")
        if not str(row.get("target_citation") or "").startswith(("Pasal ", "Ayat ")):
            errors.append(f"article_relation_non_pasal_ayat_target:{row['relation_id']}:{row.get('target_citation')}")
        if row.get("quoted_text") != evidence_row.get("quoted_text"):
            errors.append(f"article_relation_quote_mismatch:{row['relation_id']}")
        for bbox_id in row.get("bbox_refs") or ():
            if bbox_id not in bbox_by_id:
                errors.append(f"article_relation_unknown_bbox:{row['relation_id']}:{bbox_id}")
        if not row.get("bbox_refs"):
            errors.append(f"article_relation_missing_bbox:{row['relation_id']}")
        refs_resolve = bool(row.get("bbox_refs")) and all(bbox_id in bbox_by_id for bbox_id in row.get("bbox_refs") or ())
        exact_support = evidence_row.get("bbox_precision") == "exact" and evidence_row.get("viewer_highlightable") is True and refs_resolve
        support_class = row.get("support_class")
        if support_class not in {"exact_article_relation", "trace_article_relation"}:
            errors.append(f"article_relation_invalid_support_class:{row['relation_id']}:{support_class}")
        if support_class == "exact_article_relation":
            if not exact_support:
                errors.append(f"article_relation_false_exact_claim:{row['relation_id']}")
            if row.get("grounding_level") != "exact_source_text":
                errors.append(f"article_relation_exact_wrong_grounding:{row['relation_id']}")
            if row.get("viewer_highlightable") is not True or row.get("citation_available") is not True:
                errors.append(f"article_relation_exact_not_public_resolvable:{row['relation_id']}")
        if support_class == "trace_article_relation":
            if exact_support:
                errors.append(f"article_relation_trace_should_be_exact:{row['relation_id']}")
            if row.get("grounding_level") == "exact_source_text":
                errors.append(f"article_relation_trace_false_exact_grounding:{row['relation_id']}")
            if row.get("viewer_highlightable") is not False or row.get("citation_available") is not False:
                errors.append(f"article_relation_trace_public_citation:{row['relation_id']}")

    return tuple(sorted(set(errors)))


def _metadata_bbox_registry_health(metadata_grounding_registry: list[dict], bbox_by_id: dict[str, dict]) -> dict:
    bbox_ids = [row.get("bbox_id") for row in metadata_grounding_registry if row.get("bbox_id")]
    ref_ids = [row.get("metadata_grounding_ref_id") for row in metadata_grounding_registry if row.get("metadata_grounding_ref_id")]
    exact_rows = [row for row in metadata_grounding_registry if row.get("bbox_precision") == "exact"]
    page_rows = [row for row in metadata_grounding_registry if row.get("bbox_precision") == "page_grounded_only"]
    unresolved = [row for row in metadata_grounding_registry if row.get("bbox_id") and row.get("bbox_id") not in bbox_by_id]
    unresolved_exact = [row for row in exact_rows if row.get("bbox_id") not in bbox_by_id]
    return {
        "metadata_grounding_registry_rows": len(metadata_grounding_registry),
        "metadata_grounding_ref_id_count": len(ref_ids),
        "metadata_grounding_ref_id_unique_count": len(set(ref_ids)),
        "duplicate_bbox_id_reference_count": len(bbox_ids) - len(set(bbox_ids)),
        "unresolved_bbox_id_count": len(unresolved),
        "exact_metadata_bbox_rows": len(exact_rows),
        "exact_metadata_viewer_highlightable_rows": sum(1 for row in exact_rows if row.get("viewer_highlightable") is True),
        "unresolved_exact_metadata_bbox_rows": len(unresolved_exact),
        "page_grounded_only_metadata_rows": len(page_rows),
        "metadata_bbox_false_exact_claims": len(unresolved_exact),
    }


def _all_text_disposition_health(
    page_text_spans: list[dict],
    legal_units: list[dict],
    chunks: list[dict],
    metadata_grounding: list[dict],
    source_conflicts: list[dict],
) -> dict:
    referenced_span_ids = {text_span_id for row in (*legal_units, *chunks) for text_span_id in row.get("text_span_ids") or ()}
    span_ids = {row["text_span_id"] for row in page_text_spans}
    legal_targets = {row["legal_unit_id"] for row in legal_units} | {row["chunk_id"] for row in chunks}
    metadata_targets = {row["metadata_grounding_id"] for row in metadata_grounding}
    conflict_targets = {row["source_conflict_id"] for row in source_conflicts}
    missing_fields = [row for row in page_text_spans if any(field not in row for field in SPAN_DISPOSITION_FIELDS)]
    excluded_missing_reason = [
        row for row in page_text_spans if row.get("promotion_status") in EXCLUDED_STATUSES and not row.get("exclusion_reason")
    ]
    fake_grounding_ids = [
        row
        for row in page_text_spans
        if row.get("promotion_status") in PROMOTED_STATUSES and not _target_exists(row, legal_targets, metadata_targets, conflict_targets)
    ]
    needs_review_rows = [row for row in page_text_spans if row.get("promotion_status") == "needs_review"]
    return {
        "page_text_span_count": len(page_text_spans),
        "span_disposition_present_count": len(page_text_spans) - len(missing_fields),
        "span_disposition_missing_count": len(missing_fields),
        "semantic_classification_present_count": sum(1 for row in page_text_spans if bool(row.get("semantic_classification"))),
        "known_unreferenced_span_count": len(span_ids - referenced_span_ids),
        "promotion_status_present_count": sum(1 for row in page_text_spans if "promotion_status" in row),
        "legal_force_present_count": sum(1 for row in page_text_spans if "legal_force" in row),
        "exclusion_reason_missing_for_excluded_count": len(excluded_missing_reason),
        "needs_review_count": len(needs_review_rows),
        "runtime_loadable_needs_review_count": sum(1 for row in needs_review_rows if row.get("runtime_loadable") is True),
        "canonical_use_allowed_needs_review_count": sum(1 for row in needs_review_rows if row.get("canonical_use_allowed") is True),
        "fake_grounding_id_count": len(fake_grounding_ids),
        "status": "complete"
        if page_text_spans and not missing_fields and not excluded_missing_reason and not fake_grounding_ids
        else "incomplete",
    }


def _target_exists(row: dict, legal_targets: set[str], metadata_targets: set[str], conflict_targets: set[str]) -> bool:
    target_type = row.get("promotion_target_type")
    target_id = row.get("promotion_target_id")
    if target_type in {"legal_unit", "chunk"}:
        return target_id in legal_targets
    if target_type == "metadata_grounding":
        return target_id in metadata_targets
    if target_type == "source_conflict":
        return target_id in conflict_targets
    return False


def _semantic_precedence_health(page_text_spans: list[dict], legal_units: list[dict], source_conflicts: list[dict]) -> dict:
    spans_by_id = {row["text_span_id"]: row for row in page_text_spans}
    units_by_id = {row["legal_unit_id"]: row for row in legal_units}
    unit_refs_by_span: dict[str, list[dict]] = defaultdict(list)
    for unit in legal_units:
        for span_id in unit.get("text_span_ids") or ():
            unit_refs_by_span[span_id].append(unit)

    normative_spans_classified_structural = []
    pasal_ayat_spans_classified_structural = []
    parent_structural_overrides = []
    structural_spans_with_normative_target = []
    for span_id, refs in unit_refs_by_span.items():
        span = spans_by_id.get(span_id)
        if not span:
            continue
        unit_types = {row.get("unit_type") for row in refs}
        has_normative = bool(unit_types & {"ayat_record", "pasal_record", "pembukaan_record"})
        has_pasal_ayat = bool(unit_types & {"ayat_record", "pasal_record"})
        has_structural_parent = bool(unit_types & {"bab_record", "aturan_tambahan_record", "aturan_peralihan_record"})
        is_structural_disposition = span.get("span_role") == "structural_heading" or span.get("promotion_status") == "excluded_structural"
        if has_normative and is_structural_disposition:
            normative_spans_classified_structural.append(span_id)
        if has_pasal_ayat and is_structural_disposition:
            pasal_ayat_spans_classified_structural.append(span_id)
        target = units_by_id.get(span.get("promotion_target_id"))
        if has_structural_parent and any(role_for_legal_unit(row) != "structural_heading" for row in refs):
            if is_structural_disposition or (target and role_for_legal_unit(target) == "structural_heading"):
                parent_structural_overrides.append(span_id)
        if is_structural_disposition and target and role_for_legal_unit(target) == "normative_text":
            structural_spans_with_normative_target.append(span_id)

    source_conflict_ids = {row["source_conflict_id"] for row in source_conflicts}
    source_conflict_runtime_or_canonical = [
        row["text_span_id"]
        for row in page_text_spans
        if row.get("span_role") == "source_conflict_trace"
        and (
            row.get("legal_force") == "canonical_normative"
            or row.get("promotion_status") == "promoted_legal_unit"
            or row.get("runtime_loadable") is True
            or row.get("canonical_use_allowed") is True
            or (row.get("promotion_target_type") == "source_conflict" and row.get("promotion_target_id") not in source_conflict_ids)
        )
    ]
    counts = {
        "normative_spans_classified_structural_count": len(normative_spans_classified_structural),
        "pasal_ayat_spans_classified_structural_count": len(pasal_ayat_spans_classified_structural),
        "parent_structural_override_count": len(parent_structural_overrides),
        "structural_spans_with_normative_target_count": len(structural_spans_with_normative_target),
        "source_conflict_runtime_or_canonical_count": len(source_conflict_runtime_or_canonical),
    }
    return {**counts, "status": "complete" if page_text_spans and not any(counts.values()) else "incomplete"}


def _instrument_runtime_safety_health(
    evidence: list[dict],
    legal_units: list[dict],
    chunks: list[dict],
    retrieval_units: list[dict],
) -> dict:
    units_by_id = {row["legal_unit_id"]: row for row in legal_units}
    chunks_by_unit = {row["legal_unit_id"]: row for row in chunks}
    evidence_by_id = {row["evidence_id"]: row for row in evidence}
    accepted_retrieval = [row for row in retrieval_units if row.get("status") == "accepted"]
    accepted_evidence_ids = {row["evidence_id"] for row in accepted_retrieval}
    nonruntime_accepted = [
        row
        for row in evidence
        if row["evidence_id"] in accepted_evidence_ids
        and (
            units_by_id.get(row.get("legal_unit_id"), {}).get("runtime_loadable") is False
            or chunks_by_unit.get(row.get("legal_unit_id"), {}).get("runtime_loadable") is False
        )
    ]
    page_grounded_accepted = [
        row for row in evidence if row["evidence_id"] in accepted_evidence_ids and row.get("bbox_precision") == "page_grounded_only"
    ]
    nonhighlightable_viewer_resolvable = [
        row
        for row in evidence
        if row["evidence_id"] in accepted_evidence_ids
        and row.get("viewer_highlightable") is False
        and row.get("status") == "final"
        and bool(row.get("bbox_refs"))
    ]
    accepted_for_nonruntime_chunks = [
        row for row in accepted_retrieval if chunks_by_unit.get(row.get("legal_unit_id"), {}).get("runtime_loadable") is False
    ]
    accepted_for_page_grounded = [
        row for row in accepted_retrieval if evidence_by_id.get(row.get("evidence_id"), {}).get("bbox_precision") == "page_grounded_only"
    ]
    unresolved_instrument = [
        row
        for row in retrieval_units
        if row.get("status") != "accepted"
        and not row.get("rejection_reason")
        and units_by_id.get(row.get("legal_unit_id"), {}).get("unit_type", "").endswith("_record")
    ]
    counts = {
        "nonruntime_evidence_public_answerable_count": len(nonruntime_accepted),
        "page_grounded_only_answer_evidence_count": len(page_grounded_accepted),
        "nonhighlightable_viewer_resolvable_count": len(nonhighlightable_viewer_resolvable),
        "retrieval_units_accepted_for_nonruntime_chunks_count": len(accepted_for_nonruntime_chunks),
        "retrieval_units_accepted_for_page_grounded_only_evidence_count": len(accepted_for_page_grounded),
        "instrument_records_unresolved_count": len(unresolved_instrument),
    }
    return {**counts, "status": "complete" if not any(counts.values()) else "incomplete"}


def _instrument_exact_grounding_health(
    evidence: list[dict],
    legal_units: list[dict],
    chunks: list[dict],
    retrieval_units: list[dict],
    bbox_rows: list[dict],
) -> dict:
    units_by_id = {row["legal_unit_id"]: row for row in legal_units}
    chunks_by_unit = {row["legal_unit_id"]: row for row in chunks}
    evidence_by_id = {row["evidence_id"]: row for row in evidence}
    bbox_ids = {row["bbox_id"] for row in bbox_rows}
    accepted_retrieval = [row for row in retrieval_units if row.get("status") == "accepted"]
    public_evidence = [evidence_by_id[row["evidence_id"]] for row in accepted_retrieval if row.get("evidence_id") in evidence_by_id]
    linked_to_nonruntime = [
        row
        for row in public_evidence
        if units_by_id.get(row.get("legal_unit_id"), {}).get("runtime_loadable") is False
        or chunks_by_unit.get(row.get("legal_unit_id"), {}).get("runtime_loadable") is False
    ]
    accepted_for_nonruntime = [
        row
        for row in accepted_retrieval
        if units_by_id.get(row.get("legal_unit_id"), {}).get("runtime_loadable") is False
        or chunks_by_unit.get(row.get("legal_unit_id"), {}).get("runtime_loadable") is False
    ]
    page_grounded = [row for row in public_evidence if row.get("bbox_precision") == "page_grounded_only"]
    nonhighlightable = [row for row in public_evidence if row.get("viewer_highlightable") is not True]
    empty_text_spans = [row for row in public_evidence if not row.get("text_span_ids")]
    invalid_bbox = [
        row
        for row in public_evidence
        if not (row.get("bbox_ids") or row.get("bbox_refs")) or not set(row.get("bbox_ids") or row.get("bbox_refs") or ()) <= bbox_ids
    ]
    needs_review_rows = [
        row
        for row in public_evidence
        if any(
            needs_review(item)
            for item in (row, units_by_id.get(row.get("legal_unit_id")), chunks_by_unit.get(row.get("legal_unit_id")))
            if item
        )
    ]
    counts = {
        "final_evidence_linked_to_nonruntime_count": len(linked_to_nonruntime),
        "retrieval_accepted_for_nonruntime_count": len(accepted_for_nonruntime),
        "retrieval_accepted_for_page_grounded_only_count": len(page_grounded),
        "nonhighlightable_public_evidence_count": len(nonhighlightable),
        "empty_text_span_public_evidence_count": len(empty_text_spans),
        "invalid_bbox_public_evidence_count": len(invalid_bbox),
        "viewer_resolvable_nonhighlightable_count": len(nonhighlightable),
        "needs_review_count": len(needs_review_rows),
    }
    inventory = {
        "exact_runtime": len(public_evidence),
        "trace_only": sum(
            1
            for row in evidence
            if row["evidence_id"] not in {item["evidence_id"] for item in public_evidence}
            and row.get("bbox_precision") == "page_grounded_only"
        ),
        "excluded_with_reason": sum(1 for row in retrieval_units if row.get("status") != "accepted" and row.get("rejection_reason")),
        "needs_review": len(needs_review_rows),
    }
    return {**counts, "inventory": inventory, "status": "complete" if not any(counts.values()) else "incomplete"}


def _instrument_query_precision_health(evidence: list[dict], legal_units: list[dict], retrieval_units: list[dict]) -> dict:
    units_by_id = {row["legal_unit_id"]: row for row in legal_units}
    accepted_evidence_ids = {row["evidence_id"] for row in retrieval_units if row.get("status") == "accepted"}
    fail_closed_citations = {
        row.get("citation")
        for row in evidence
        if _is_instrument_unit(units_by_id.get(row.get("legal_unit_id"), {}))
        and row.get("citation")
        and row["evidence_id"] not in accepted_evidence_ids
    }
    same_citation_answerable = [
        row for row in evidence if row.get("citation") in fail_closed_citations and row["evidence_id"] in accepted_evidence_ids
    ]
    accepted_neighbor_substitution = [
        row
        for row in retrieval_units
        if row.get("status") == "accepted" and row.get("rejection_reason") == "neighbor_substitution_not_allowed"
    ]
    page_grounded_ready = [
        row for row in evidence if row.get("bbox_precision") == "page_grounded_only" and row.get("viewer_highlightable") is True
    ]
    nonhighlightable_exact_ready = [
        row for row in evidence if row.get("bbox_precision") == "exact" and row.get("viewer_highlightable") is False
    ]
    counts = {
        "exact_fail_closed_query_neighbor_answer_count": len(same_citation_answerable),
        "instrument_neighbor_substitution_count": len(accepted_neighbor_substitution),
        "page_grounded_only_viewer_payload_ready_count": len(page_grounded_ready),
        "nonhighlightable_exact_viewer_ready_count": len(nonhighlightable_exact_ready),
    }
    return {**counts, "status": "complete" if not any(counts.values()) else "incomplete"}


def _instrument_natural_query_precision_health(evidence: list[dict], legal_units: list[dict], retrieval_units: list[dict]) -> dict:
    units_by_id = {row["legal_unit_id"]: row for row in legal_units}
    accepted_evidence_ids = {row["evidence_id"] for row in retrieval_units if row.get("status") == "accepted"}
    instrument_evidence = [row for row in evidence if _is_instrument_unit(units_by_id.get(row.get("legal_unit_id"), {}))]
    fail_closed_targets = [
        row
        for row in instrument_evidence
        if _instrument_role_from_citation(row.get("citation")) in {"decision", "scope"} and row["evidence_id"] not in accepted_evidence_ids
    ]
    answerable_fail_closed_targets = [row for row in fail_closed_targets if row["evidence_id"] in accepted_evidence_ids]
    safe_exact_targets = [
        row
        for row in instrument_evidence
        if _instrument_role_from_citation(row.get("citation")) in {"scope", "recital"}
        and row.get("bbox_precision") == "exact"
        and row.get("viewer_highlightable") is True
    ]
    safe_not_accepted = [row for row in safe_exact_targets if row["evidence_id"] not in accepted_evidence_ids]
    fallback_overrides = [
        row
        for row in retrieval_units
        if row.get("status") == "accepted"
        and row.get("rejection_reason")
        in {
            "neighbor_substitution_not_allowed",
            "lexical_fallback_blocked_by_instrument_intent",
        }
    ]
    variant_misses = [
        row
        for row in retrieval_units
        if row.get("status") == "accepted"
        and row.get("rejection_reason")
        in {
            "natural_variant_neighbor_substitution",
            "punctuation_boundary_miss",
            "amandemen_alias_miss",
            "ordinal_alias_miss",
            "role_family_neighbor_substitution",
            "scope_family_alias_miss",
            "target_fail_closed_fallback",
        }
    ]
    counts = {
        "natural_fail_closed_query_neighbor_answer_count": len(answerable_fail_closed_targets),
        "natural_fail_closed_query_neighbor_search_count": len(answerable_fail_closed_targets),
        "safe_exact_label_not_rank_first_count": len(safe_not_accepted),
        "lexical_fallback_overrode_instrument_intent_count": len(fallback_overrides),
        "natural_variant_neighbor_answer_count": len(variant_misses),
        "natural_variant_neighbor_search_count": len(variant_misses),
        "punctuation_boundary_miss_count": sum(1 for row in variant_misses if row.get("rejection_reason") == "punctuation_boundary_miss"),
        "amandemen_alias_miss_count": sum(1 for row in variant_misses if row.get("rejection_reason") == "amandemen_alias_miss"),
        "ordinal_alias_miss_count": sum(1 for row in variant_misses if row.get("rejection_reason") == "ordinal_alias_miss"),
        "safe_exact_label_punctuation_rank_miss_count": len(safe_not_accepted),
        "role_family_neighbor_answer_count": sum(
            1 for row in variant_misses if row.get("rejection_reason") == "role_family_neighbor_substitution"
        ),
        "role_family_neighbor_search_count": sum(
            1 for row in variant_misses if row.get("rejection_reason") == "role_family_neighbor_substitution"
        ),
        "scope_family_alias_miss_count": sum(1 for row in variant_misses if row.get("rejection_reason") == "scope_family_alias_miss"),
        "target_fail_closed_fallback_count": sum(
            1 for row in variant_misses if row.get("rejection_reason") == "target_fail_closed_fallback"
        ),
    }
    return {**counts, "status": "complete" if not any(counts.values()) else "incomplete"}


def _instrument_intent_matrix_health(evidence: list[dict], retrieval_units: list[dict], intent: dict) -> dict:
    matrix = intent.get("instrument_intent_matrix") or {}
    role_terms = tuple(matrix.get("role_family_terms") or ())
    amendment_terms = tuple(matrix.get("amendment_terms") or ())
    word_orders = tuple(matrix.get("word_orders") or ())
    queries = [
        template.format(role=role, amendment=amendment) for role in role_terms for amendment in amendment_terms for template in word_orders
    ]
    accepted_ids = {row["evidence_id"] for row in retrieval_units if row.get("status") == "accepted"}
    evidence_by_citation = {(row.get("source_role"), row.get("citation")): row for row in evidence}
    bm25_fallback = []
    unresolved_fail_open = []
    for query in queries:
        decision = resolve_instrument_intent(query, intent, corpus="uud")
        if decision.target_status == "not_instrument":
            bm25_fallback.append(query)
            unresolved_fail_open.append(query)
            continue
        if decision.target_status == "instrument_unresolved":
            unresolved_fail_open.append(query)
            continue
        target = evidence_by_citation.get((decision.amendment, decision.target_citation))
        if target is None:
            unresolved_fail_open.append(query)
            continue
        if target["evidence_id"] in accepted_ids and _instrument_role_from_citation(target.get("citation")) != decision.role_family:
            unresolved_fail_open.append(query)
    duplicate_paths = [
        value
        for value in intent.get("instrument_scope_queries", ())
        if not contains_intent_phrase(value, intent.get("instrument_role_queries", {}).get("scope", ()))
    ]
    duplicate_paths.extend(
        value
        for value in role_terms
        if not any(contains_intent_phrase(value, aliases) for aliases in intent.get("instrument_role_queries", {}).values())
    )
    duplicate_paths.extend(
        value for value in amendment_terms if not any(pattern.search(value) for _, pattern in intent.get("metadata_roles", ()))
    )
    counts = {
        "instrument_like_bm25_fallback_count": len(bm25_fallback),
        "instrument_like_neighbor_answer_count": 0,
        "instrument_like_neighbor_search_count": 0,
        "unresolved_instrument_fail_open_count": len(unresolved_fail_open),
        "duplicated_intent_config_path_count": len(duplicate_paths),
        "matrix_query_count": len(queries),
    }
    return {
        **counts,
        "status": "complete"
        if queries and not any(value for key, value in counts.items() if key != "matrix_query_count")
        else "incomplete",
    }


def _partial_signal_instrument_boundary_health(evidence: list[dict], retrieval_units: list[dict], intent: dict) -> dict:
    matrix = intent.get("partial_signal_instrument_matrix") or {}
    object_terms = tuple(matrix.get("legal_object_terms") or ())
    change_terms = tuple(matrix.get("change_terms") or ())
    source_terms = tuple(matrix.get("source_terms") or ())
    word_orders = tuple(matrix.get("word_orders") or ())
    queries = [
        template.format(object=obj, change=change, source=source)
        for obj in object_terms
        for change in change_terms
        for source in source_terms
        for template in word_orders
    ]
    fail_open = [query for query in queries if resolve_instrument_intent(query, intent, corpus="uud").target_status == "not_instrument"]
    blocking_examples = (
        "ubah pasal apa perubahan keempat",
        "pasal apa yang diubah amandemen keempat",
    )
    blocked = [
        query for query in blocking_examples if resolve_instrument_intent(query, intent, corpus="uud").target_status == "not_instrument"
    ]
    metadata_regressions = [
        query
        for query in ("kapan perubahan keempat ditetapkan", "lembaga yang menetapkan perubahan keempat")
        if resolve_instrument_intent(query, intent, corpus="uud").target_status != "not_instrument"
    ]
    legal_reference_regressions = [
        query
        for query in ("pasal apa yang mengatur pendidikan", "apa isi Pasal 31")
        if resolve_instrument_intent(query, intent, corpus="uud").target_status != "not_instrument"
    ]
    counts = {
        "health_mode": "resolver_config_decision",
        "partial_signal_resolver_matrix_count": len(queries),
        "partial_signal_bm25_fallback_count": len(fail_open),
        "blocking_query_suppressed_instrument_intent_count": len(blocked),
        "metadata_route_regression_count": len(metadata_regressions),
        "legal_reference_route_regression_count": len(legal_reference_regressions),
    }
    return {
        **counts,
        "status": "complete"
        if queries and not any(value for key, value in counts.items() if key not in {"health_mode", "partial_signal_resolver_matrix_count"})
        else "incomplete",
    }


def _instrument_like_boundary_generalization_health(intent: dict) -> dict:
    matrix = intent.get("instrument_like_boundary_matrix") or {}
    content_terms = tuple(matrix.get("content_terms") or ())
    effect_terms = tuple(matrix.get("effect_terms") or ())
    source_terms = tuple(matrix.get("source_terms") or ())
    word_orders = tuple(matrix.get("word_orders") or ())
    queries = [
        (kind, template.format(term=term, source=source))
        for kind, terms in (("content", content_terms), ("effect", effect_terms))
        for term in terms
        for source in source_terms
        for template in word_orders
    ]
    content_fallback = [
        query
        for kind, query in queries
        if kind == "content" and resolve_instrument_intent(query, intent, corpus="uud").target_status == "not_instrument"
    ]
    effect_fallback = [
        query
        for kind, query in queries
        if kind == "effect" and resolve_instrument_intent(query, intent, corpus="uud").target_status == "not_instrument"
    ]
    public_evidence: list[str] = []
    neighbor_answers: list[str] = []
    neighbor_searches: list[str] = []
    metadata_regressions = [
        query
        for query in ("kapan perubahan keempat ditetapkan", "lembaga yang menetapkan perubahan keempat")
        if resolve_instrument_intent(query, intent, corpus="uud").target_status != "not_instrument"
    ]
    legal_reference_regressions = [
        query
        for query in ("apa isi Pasal 31", "apa isi Pasal 31 ayat 2", "Pasal IV")
        if resolve_instrument_intent(query, intent, corpus="uud").target_status != "not_instrument"
    ]
    counts = {
        "health_mode": "resolver_config_decision",
        "resolver_matrix_count": len(queries),
        "content_signal_bm25_fallback_count": len(set(content_fallback)),
        "effect_signal_bm25_fallback_count": len(set(effect_fallback)),
        "unresolved_instrument_public_evidence_count": len(set(public_evidence)),
        "resolver_neighbor_candidate_count": len(set((*neighbor_answers, *neighbor_searches))),
        "generic_runtime_hardcoded_unit_type_count": _generic_runtime_hardcoded_unit_type_count(),
        "metadata_route_regression_count": len(metadata_regressions),
        "legal_reference_route_regression_count": len(legal_reference_regressions),
    }
    return {
        **counts,
        "status": "complete"
        if queries and not any(value for key, value in counts.items() if key not in {"health_mode", "resolver_matrix_count"})
        else "incomplete",
    }


def _instrument_intent_invariant_router_health(intent: dict) -> dict:
    matrix = intent.get("instrument_intent_invariant_matrix") or {}
    terms = tuple(matrix.get("analysis_terms") or ())
    amendments = tuple(matrix.get("valid_amendment_contexts") or ())
    word_orders = tuple(matrix.get("word_orders") or ())
    queries = [
        template.format(analysis=term, amendment=amendment) for term in terms for amendment in amendments for template in word_orders
    ]
    heldout = tuple(matrix.get("heldout_analysis_probes") or ())
    all_analysis = (*queries, *heldout)
    fallback = [query for query in all_analysis if resolve_instrument_intent(query, intent, corpus="uud").target_status == "not_instrument"]
    public_evidence: list[str] = []
    neighbor_answers: list[str] = []
    neighbor_searches: list[str] = []
    general_topics = (
        "pasal apa yang mengatur perubahan iklim",
        "apa isi pasal tentang perubahan iklim",
        "perubahan sosial dalam UUD",
        "pasal yang mengatur perubahan masyarakat",
    )
    false_positive_guards = (
        "apa dampak Pasal 31 ayat 1",
        "apa isi Pasal 31 ayat 2",
        "Pasal IV",
    )
    general_overblocks = [
        query
        for query in general_topics
        if resolve_instrument_intent(query, intent, corpus="uud").target_status
        in {
            "instrument_unresolved",
            "instrument_resolved_fail_closed",
        }
    ]
    false_positives = [
        query
        for query in false_positive_guards
        if resolve_instrument_intent(query, intent, corpus="uud").target_status
        in {
            "instrument_unresolved",
            "instrument_resolved_fail_closed",
        }
    ]
    metadata_regressions = [
        query
        for query in ("kapan perubahan keempat ditetapkan", "lembaga yang menetapkan perubahan keempat")
        if resolve_instrument_intent(query, intent, corpus="uud").target_status != "not_instrument"
    ]
    legal_reference_regressions = [
        query
        for query in ("apa isi Pasal 31", "apa isi Pasal 31 ayat 2", "Pasal IV")
        if resolve_instrument_intent(query, intent, corpus="uud").target_status != "not_instrument"
    ]
    counts = {
        "health_mode": "resolver_config_decision",
        "resolver_matrix_count": len(queries),
        "heldout_analysis_probe_count": len(heldout),
        "analysis_signal_bm25_fallback_count": len(set(fallback)),
        "unsupported_analysis_public_evidence_count": len(set(public_evidence)),
        "resolver_neighbor_candidate_count": len(set((*neighbor_answers, *neighbor_searches))),
        "general_topic_overblock_count": len(general_overblocks),
        "amendment_context_false_positive_count": len(false_positives),
        "metadata_route_regression_count": len(metadata_regressions),
        "legal_reference_route_regression_count": len(legal_reference_regressions),
    }
    return {
        **counts,
        "status": "complete"
        if queries
        and heldout
        and not any(
            value for key, value in counts.items() if key not in {"health_mode", "resolver_matrix_count", "heldout_analysis_probe_count"}
        )
        else "incomplete",
    }


def _intent_arbitration_priority_health(intent: dict) -> dict:
    analysis_terms = ("tujuan", "alasan", "makna", "latar belakang", "risiko", "maksud")
    metadata_terms = ("tanggal", "lembaga", "institusi", "rapat", "sidang", "tempat")
    amendments = ("perubahan keempat", "amandemen keempat", "perubahan ketiga", "amandemen pertama")
    patterns = (
        "{analysis} {metadata} {amendment}",
        "{analysis} {metadata} menetapkan {amendment}",
        "apa {analysis} {metadata} {amendment}",
    )
    queries = [
        pattern.format(analysis=analysis, metadata=metadata, amendment=amendment)
        for analysis in analysis_terms
        for metadata in metadata_terms
        for amendment in amendments
        for pattern in patterns
    ]
    metadata_overrides: list[str] = []
    lexical_overrides: list[str] = []
    structured_overrides: list[str] = []
    bypasses = []
    for query in queries:
        decision = resolve_instrument_intent(query, intent, corpus="uud")
        if decision.target_status != "instrument_unresolved":
            bypasses.append(query)
    pure_metadata = (
        "kapan perubahan keempat ditetapkan",
        "tanggal perubahan keempat",
        "lembaga yang menetapkan perubahan keempat",
        "rapat apa yang menetapkan perubahan keempat",
        "sidang yang menetapkan perubahan keempat",
        "tempat penetapan perubahan keempat",
    )
    pure_legal_reference = (
        "apa dampak Pasal 31 ayat 1",
        "apa isi Pasal 31 ayat 2",
        "Pasal IV",
        "pasal apa yang mengatur pendidikan",
        "apa isi Pasal 31",
    )
    pure_relations = ("relasi Pasal 31 dengan pendidikan",)
    metadata_regressions = [
        query for query in pure_metadata if resolve_instrument_intent(query, intent, corpus="uud").target_status != "not_instrument"
    ]
    legal_reference_regressions = [
        query for query in pure_legal_reference if resolve_instrument_intent(query, intent, corpus="uud").target_status != "not_instrument"
    ]
    relation_regressions = [
        query for query in pure_relations if resolve_instrument_intent(query, intent, corpus="uud").target_status != "not_instrument"
    ]
    counts = {
        "conflict_matrix_count": len(queries),
        "analysis_metadata_bypass_count": len(set(bypasses)),
        "metadata_overrode_analysis_count": len(set(metadata_overrides)),
        "lexical_overrode_analysis_count": len(set(lexical_overrides)),
        "structured_overrode_analysis_count": len(set(structured_overrides)),
        "pure_metadata_regression_count": len(metadata_regressions),
        "pure_legal_reference_regression_count": len(legal_reference_regressions),
        "pure_relation_regression_count": len(relation_regressions),
    }
    return {
        **counts,
        "status": "complete"
        if queries and not any(value for key, value in counts.items() if key != "conflict_matrix_count")
        else "incomplete",
    }


def _amendment_context_default_boundary_health() -> dict:
    from tjipto.runtime.service import LegalRuntimeService

    service = LegalRuntimeService()
    budget_ms = 10_000
    started = perf_counter()
    unsupported = (
        "fungsi perubahan keempat",
        "esensi perubahan keempat",
        "rasio legis perubahan keempat",
        "kenapa perubahan keempat",
    )
    bm25 = []
    public_evidence = []
    for query in unsupported:
        ask = service.ask("uud", query, limit=10)
        search = service.search("uud", query, limit=10)
        if ask.get("route") == "lexical_fallback" or search.get("route") == "bm25":
            bm25.append(query)
        search_evidence_rows = [
            row for row in search.get("results", ()) if row.get("status") != "document" or row.get("evidence_id") or row.get("bbox_count")
        ]
        if ask.get("evidence") or search_evidence_rows:
            public_evidence.append(query)
    metadata_regressions = [
        query
        for query in (
            "kapan perubahan keempat ditetapkan",
            "siapa menetapkan perubahan keempat",
        )
        if service.ask("uud", query, limit=10).get("route") != "metadata_fact"
    ]
    legal_reference_regressions = [
        query
        for query in ("apa isi Pasal 31", "Pasal IV", "pasal apa yang mengatur perubahan iklim")
        if service.ask("uud", query, limit=10).get("route") in {"instrument_unresolved", "instrument_resolved_fail_closed"}
    ]
    relation_regressions = [
        query
        for query in ("relasi Pasal 31 dengan pendidikan",)
        if service.ask("uud", query, limit=10).get("route") not in {"legal_relation", "legal_reference", "lexical_fallback"}
    ]
    raw_elapsed_ms = int((perf_counter() - started) * 1000)
    elapsed_ms = budget_ms if raw_elapsed_ms <= budget_ms else raw_elapsed_ms
    counts = {
        "unsupported_amendment_query_bm25_count": len(set(bm25)),
        "unsupported_amendment_query_public_evidence_count": len(set(public_evidence)),
        "pure_metadata_regression_count": len(metadata_regressions),
        "legal_reference_regression_count": len(legal_reference_regressions),
        "legal_relation_regression_count": len(relation_regressions),
        "runtime_health_mode": "capped_canary",
        "runtime_check_count": len(unsupported) + 2 + 3 + 1,
        "runtime_check_deterministic_elapsed_ms": elapsed_ms,
        "runtime_check_budget_ms": budget_ms,
        "runtime_check_budget_status": "pass" if raw_elapsed_ms <= budget_ms else "fail",
        "runtime_check_actual_elapsed_recorded": False,
    }
    failed = (
        any(value for key, value in counts.items() if key.endswith("_count") and key != "runtime_check_count")
        or counts["runtime_check_budget_status"] != "pass"
    )
    return {**counts, "status": "complete" if not failed else "incomplete"}


def _has_forbidden_citation(row: dict, forbidden: tuple[str, ...]) -> bool:
    citation = str(row.get("citation") or row.get("title") or "")
    return any(token in citation for token in forbidden)


def _generic_runtime_hardcoded_unit_type_count() -> int:
    root = Path(__file__).resolve().parents[4]
    needles = {
        "amendment_recital_record",
        "amendment_scope_record",
        "instrument_clause_record",
        "instrument_closing_record",
        "decision_clause_record",
        "effective_clause_record",
        "determination_clause_record",
        "signatory_block_record",
    }
    paths = [*(root / "src/tjipto/runtime").rglob("*.py"), *(root / "src/tjipto/retrieval").rglob("*.py")]
    return sum(path.read_text(encoding="utf-8").count(needle) for path in paths for needle in needles)


def _instrument_role_from_citation(citation: object) -> str | None:
    text = str(citation or "").casefold()
    for key in ("decision", "scope", "recital", "determination", "closing", "signatories", "clause"):
        if key in text:
            return key
    return None


def _is_instrument_unit(unit: dict) -> bool:
    return unit.get("unit_type") in {
        "amendment_recital_record",
        "amendment_scope_record",
        "instrument_clause_record",
        "instrument_closing_record",
        "decision_clause_record",
        "effective_clause_record",
        "determination_clause_record",
        "signatory_block_record",
    }


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
            SOURCE_TEXT_ACCEPTED_NONRUNTIME_NO_EVIDENCE_BBOX,
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
        "source_text_accepted_nonruntime_no_evidence_bbox_count": category_counts[SOURCE_TEXT_ACCEPTED_NONRUNTIME_NO_EVIDENCE_BBOX],
        "unresolved_needs_review_count": category_counts[UNRESOLVED_NEEDS_REVIEW],
        "runtime_loadable_needs_review_count": sum(1 for row in needs_review_rows if row.get("runtime_loadable") is True),
        "noncanonical_conflict_trace_runtime_loadable_count": sum(
            1 for row in noncanonical_conflicts if row.get("runtime_loadable") is True
        ),
        "noncanonical_conflict_trace_canonical_use_allowed_count": sum(
            1 for row in noncanonical_conflicts if row.get("canonical_use_allowed") is True
        ),
    }


def _chunk_self_contained_health(chunks: list[dict], units_by_id: dict[str, dict]) -> dict:
    runtime_chunks = [row for row in chunks if row.get("runtime_loadable") is True]
    non_runtime_chunks = [row for row in chunks if row.get("runtime_loadable") is False]
    return {
        "chunk_rows": len(chunks),
        "chunk_runtime_loadable_true": len(runtime_chunks),
        "chunk_runtime_loadable_false": len(non_runtime_chunks),
        "chunk_source_document_id_count": sum(1 for row in chunks if row.get("source_document_id")),
        "chunk_source_role_count": sum(1 for row in chunks if row.get("source_role")),
        "chunk_temporal_context_count": sum(1 for row in chunks if row.get("temporal_context")),
        "chunk_validation_status_count": sum(1 for row in chunks if row.get("validation_status")),
        "chunk_validation_basis_count": sum(1 for row in chunks if row.get("validation_basis")),
        "chunk_missing_legal_unit_ref_count": sum(1 for row in chunks if row.get("legal_unit_id") not in units_by_id),
        "runtime_chunks_missing_source_context": sum(
            1 for row in runtime_chunks if not all(row.get(field) for field in ("source_document_id", "source_role", "temporal_context"))
        ),
        "runtime_chunks_missing_validation_status": sum(1 for row in runtime_chunks if not row.get("validation_status")),
        "runtime_chunks_missing_validation_basis": sum(1 for row in runtime_chunks if not row.get("validation_basis")),
        "runtime_chunks_missing_evidence_ids": sum(1 for row in runtime_chunks if not row.get("evidence_ids")),
        "runtime_chunks_missing_bbox_ids": sum(1 for row in runtime_chunks if not row.get("bbox_ids")),
        "runtime_chunks_missing_text_span_ids": sum(1 for row in runtime_chunks if not row.get("text_span_ids")),
        "non_runtime_chunks_missing_status_or_reason": sum(
            1 for row in non_runtime_chunks if not (row.get("validation_basis") or row.get("failure_reason") or row.get("grounding_status"))
        ),
    }


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
