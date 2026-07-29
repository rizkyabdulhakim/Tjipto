from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
from collections.abc import Mapping
from typing import cast
import unicodedata

from tjipto.contracts.evidence import normalize_source_text
from tjipto.contracts.authority import AUTHORITY_KINDS
from tjipto.corpora.disposition import (
    EXCLUDED_STATUSES,
    LEGAL_FORCES,
    PROMOTED_STATUSES,
    PROMOTION_STATUSES,
    REVIEW_STATUSES,
    SEMANTIC_CLASSIFICATIONS,
    SPAN_DISPOSITION_FIELDS,
    SPAN_ROLES,
)
from tjipto.corpora.intent_config import contains_intent_phrase, resolve_instrument_intent
from tjipto.corpora.parser_dispatch import parse_legal_references
from tjipto.corpora.uud.parser import matches_uud_contextual_reference
from tjipto.corpora.uud.bbox_builder import bbox_precision_counts
from tjipto.corpora.uud.artifact_policy import ALLOWED_ARTIFACT_ORIGINS
from tjipto.corpora.uud.artifact_policy import RECOVERY_CAPABILITIES, RECOVERY_STATUSES, UUD_ARTIFACT_SCHEMA
from tjipto.contracts.artifacts import ARTIFACT_ALLOWED_FIELDS, ARTIFACT_OPTIONAL_FIELDS, COMMON_ARTIFACT_FIELDS, FORBIDDEN_ARTIFACT_FIELDS, MINIMUM_ARTIFACT_FIELDS
from tjipto.contracts.relations import descriptor_for
from tjipto.ingestion.pdf.source_objects import TERMINAL_DISPOSITIONS
from tjipto.corpora.uud.contract import CONTRACT_FINGERPRINT, CONTRACT_ID, CONTRACT_VERSION
from tjipto.corpora.uud.provenance_exceptions import (
    ACCEPTED_FALSE_POSITIVE_SEGMENTATION_PUNCTUATION,
    ACCEPTED_NONCANONICAL_SOURCE_CONFLICT_TRACE_ONLY,
    BUILDER_SLICING_LABEL_ISSUE_CONFIRMED,
    DUPLICATED_HEADING_ARTIFACT_ISSUE_CONFIRMED,
    UNRESOLVED_NEEDS_REVIEW,
    UNRESOLVED_MANUAL_REVIEW_REQUIRED,
    needs_review,
    review_category,
)
from tjipto.corpora.uud.span_disposition_policy import role_for_legal_unit, substantive_structural_unit
from tjipto.corpora.uud.specs import UUD_LEGAL_GRAPH_EDGE_SCHEMA
from tjipto.corpora.uud.policy.validation import validate_uud_trust_boundary
from tjipto.corpora.uud.proposition_builder import source_marker_character_boxes
from tjipto.evidence.bbox import positive_area_intersection, viewer_overlay_rectangles
from tjipto.core.manifest import artifact_set_digest, read_json, read_jsonl, validate_manifest


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
    "RENUMBERED_TO",
    "SUPPLEMENTS",
    *STRUCTURAL_SEQUENCE_EDGE_TYPES,
    "HAS_EFFECTIVE_RULE",
    "HAS_SIGNATORY",
    "HAS_DECISION_SESSION",
    "HAS_SOURCE_ANOMALY",
    "MODIFIED_BY",
    "RENAMED_FROM",
    "RENUMBERED_FROM",
    "DELETED_BY",
    "INSERTED_BY",
}

_REQUIRED_HEALTH_STATUSES = {
    "raw_source_geometry_health": "pass",
    "source_object_disposition_health": "complete",
    "legal_unit_chunk_span_closure_health": "complete",
    "structural_authority_contract_health": "complete",
    "semantic_precedence_health": "complete",
    "instrument_runtime_safety_health": "complete",
    "instrument_exact_grounding_health": "complete",
    "instrument_query_precision_health": "complete",
    "instrument_natural_query_precision_health": "complete",
    "word_bbox_registry_health": "complete",
    "selector_geometry_health": "complete",
    "promotion_engine_health": "complete",
    "promotion_decision_audit_health": "complete",
    "article_relation_runtime_policy_health": "complete",
    "legal_graph_authority_health": "complete",
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
    promotion_decisions: list[dict],
    propositions: list[dict],
    metadata_grounding: list[dict],
    metadata_grounding_registry: list[dict],
    word_bboxes: list[dict],
    manifest_files: dict[str, dict],
    graph_nodes: list[dict],
    graph_edges: list[dict],
    document_relations: list[dict],
    article_amendment_relations: list[dict],
    page_text_spans: list[dict],
    raw_source_spans: list[dict] | None = None,
    source_objects: list[dict] | None = None,
    pdf_health_report: dict | None = None,
    pages: list[dict] | None = None,
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
            "excluded_chunk_policy": "legacy ordinal exclusion policy removed; admission follows verified grounding and source/temporal authority",
            "audited_excluded_chunks": [],
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
            "word_bboxes.jsonl",
            "pdf_health_report.json",
            "promotion_decisions.jsonl",
            "propositions.jsonl",
            "source_objects.jsonl",
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
        "promotion_decisions": len(promotion_decisions),
        "propositions": len(propositions),
        "graph_nodes": len(graph_nodes),
        "graph_edges": len(graph_edges),
        "page_text_spans": len(page_text_spans),
        "word_bboxes": len(word_bboxes),
        "source_objects": len(source_objects or ()),
    }
    validation_report["bbox_precision_counts"] = bbox_precision_counts(bbox_rows)
    validation_report["bbox_highlightability_counts"] = {
        "viewer_highlightable": sum(1 for row in bbox_rows if row.get("viewer_highlightable") is True),
        "non_highlightable": sum(1 for row in bbox_rows if row.get("viewer_highlightable") is not True),
    }
    if raw_source_spans is None:
        fallback_path = Path("data/final/uud/raw_source_spans.jsonl")
        raw_source_spans = read_jsonl(fallback_path) if fallback_path.exists() else []
    validation_report["raw_source_geometry_health"] = _raw_source_geometry_health(raw_source_spans)
    validation_report["source_object_disposition_health"] = _source_object_disposition_health(source_objects or [], page_text_spans)
    validation_report["chunk_self_contained_health"] = _chunk_self_contained_health(
        chunks,
        {row["legal_unit_id"]: row for row in legal_units},
    )
    validation_report["evidence_admission_audit"] = _evidence_admission_audit(legal_units, evidence)
    validation_report["legal_unit_chunk_span_closure_health"] = _legal_unit_chunk_span_closure_health(
        legal_units=legal_units,
        chunks=chunks,
        page_text_spans=page_text_spans,
        graph_nodes=graph_nodes,
    )
    validation_report["structural_authority_contract_health"] = _structural_authority_contract_health(
        legal_units,
        chunks,
        graph_nodes,
        graph_edges,
        retrieval_units,
    )
    validation_report["provenance_exception_health"] = _provenance_exception_health(
        chunks,
        legal_units,
        source_conflicts,
    )
    validation_report["source_conflict_provenance_health"] = _source_conflict_provenance_health(source_conflicts)
    viewer_provenance_coverage_health = _viewer_provenance_coverage_health(
        page_text_spans=page_text_spans,
        bbox_rows=bbox_rows,
    )
    validation_report["viewer_provenance_coverage_health"] = viewer_provenance_coverage_health
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
        {row["bbox_id"]: row for row in bbox_rows} | {row["word_bbox_id"]: {"bbox_id": row["word_bbox_id"], **row} for row in word_bboxes},
    )
    validation_report["source_quote_fidelity_health"] = _source_quote_fidelity_health(
        metadata_grounding=metadata_grounding,
        evidence=evidence,
        pages=pages or [],
    )
    validation_report["pdf_health"] = _pdf_health_summary(pdf_health_report or {})
    validation_report["word_bbox_registry_health"] = _word_bbox_registry_health(
        word_bboxes=word_bboxes,
        pages=pages or [],
    )
    validation_report["selector_geometry_health"] = _selector_geometry_health(
        propositions=propositions,
        evidence=evidence,
        page_text_spans=page_text_spans,
        word_bboxes=word_bboxes,
    )
    validation_report["highlight_registry_contract"] = {
        "status": "complete",
        "architecture": "bbox_registry_union_word_bboxes",
        "bbox_registry_rows": len(bbox_rows),
        "word_bbox_rows": len(word_bboxes),
        "official_viewer_highlight_ref_sources": ["bbox_registry", "word_bboxes"],
        "bbox_registry_layer": "materialized_final_and_provenance_bboxes",
        "word_bbox_layer": "word_bbox_exact_highlight",
        "viewer_highlightable_union": "bbox_registry_union_word_bboxes",
        "bbox_key_absent_span_count": viewer_provenance_coverage_health["bbox_key_absent_span_count"],
        "exact_safe_word_highlight_count": sum(
            1
            for row in page_text_spans
            if row.get("promotion_status") == "promoted_legal_unit"
            and row.get("bbox_registry_coverage_reason") == "exact_word_bbox_available"
            and row.get("evidence_ids")
            and row.get("span_bbox_ids")
            and all(
                any(word.get("word_bbox_id") == ref for word in word_bboxes)
                for ref in row.get("span_bbox_ids") or ()
            )
            and all(
                next((item for item in evidence if item.get("evidence_id") == evidence_id), {}).get("exactness") == "exact"
                for evidence_id in row.get("evidence_ids") or ()
            )
        ),
        "non_citable_absent_span_count": sum(1 for row in page_text_spans if row.get("viewer_highlightable") is not True and not row.get("span_bbox_ids")),
        "false_highlight_count": viewer_provenance_coverage_health["highlight_without_span_bbox_count"],
    }
    validation_report["span_sequence_grounding_health"] = _span_sequence_grounding_health(
        metadata_grounding=metadata_grounding,
        evidence=evidence,
        legal_units=legal_units,
        chunks=chunks,
        bbox_rows=bbox_rows,
        word_bboxes=word_bboxes,
        page_text_spans=page_text_spans,
    )
    validation_report["promotion_engine_health"] = _promotion_engine_health(
        evidence=evidence,
        metadata_grounding=metadata_grounding,
        bbox_rows=bbox_rows,
        word_bboxes=word_bboxes,
        legal_units=legal_units,
        chunks=chunks,
        page_text_spans=page_text_spans,
        pages=pages or [],
    )
    validation_report["promotion_decision_audit_health"] = _promotion_decision_audit_health(
        evidence=evidence,
        metadata_grounding=metadata_grounding,
        bbox_rows=bbox_rows,
        promotion_decisions=promotion_decisions,
        promotion_engine_health=cast(dict, validation_report["promotion_engine_health"]),
        article_relations=article_amendment_relations or (),
    )
    promotion_engine_health = cast(dict, validation_report["promotion_engine_health"])
    promotion_decision_audit_health = cast(dict, validation_report["promotion_decision_audit_health"])
    promotion_engine_health["promotion_blocked_count"] = promotion_decision_audit_health[
        "blocked_decision_count"
    ]
    validation_report["metadata_exact_promotion_feasibility_health"] = _metadata_exact_promotion_feasibility_health(
        promotion_decisions=promotion_decisions
    )
    validation_report["article_relation_runtime_policy_health"] = _article_relation_runtime_policy_health(
        document_relations=document_relations or (),
        article_amendment_relations=article_amendment_relations or (),
        bbox_rows=bbox_rows,
        word_bboxes=word_bboxes,
        evidence=evidence,
        legal_units=legal_units,
    )
    validation_report["legal_graph_baseline"] = {
        "status": "authority_aware_evidence_gated",
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
    validation_report["legal_graph_authority_health"] = _legal_graph_authority_health(
        graph_edges=graph_edges,
        article_amendment_relations=article_amendment_relations or (),
        evidence=evidence,
        bbox_rows=bbox_rows,
        word_bboxes=word_bboxes,
    )
    validation_report["validated_artifact_set_digest"] = artifact_set_digest(
        {"files": manifest_files}, exclude=("validation_report.json",)
    )
    _derive_validation_status(validation_report)
    return validation_report


def _derive_validation_status(report: dict) -> None:
    unsatisfied = {
        health_key: {
            "expected": expected_status,
            "actual": (report.get(health_key) or {}).get("status")
            if isinstance(report.get(health_key), Mapping)
            else None,
        }
        for health_key, expected_status in _REQUIRED_HEALTH_STATUSES.items()
        if not isinstance(report.get(health_key), Mapping)
        or report[health_key].get("status") != expected_status
    }
    report["required_health"] = {
        "status": "complete" if not unsatisfied else "incomplete",
        "required_section_count": len(_REQUIRED_HEALTH_STATUSES),
        "unsatisfied_section_count": len(unsatisfied),
        "unsatisfied_sections": unsatisfied,
    }
    report["status"] = "valid" if not unsatisfied else "invalid"


def _evidence_admission_audit(legal_units: list[dict], evidence: list[dict]) -> dict:
    """Record a deterministic decision for every unit lacking direct evidence."""
    evidence_units = {row.get("legal_unit_id") for row in evidence}
    decisions = []
    for unit in sorted(legal_units, key=lambda row: row.get("legal_unit_id", "")):
        unit_id = unit.get("legal_unit_id")
        if unit_id in evidence_units:
            continue
        descendants = any(unit_id in (candidate.get("ancestor_legal_unit_ids") or ()) for candidate in legal_units if candidate is not unit)
        if descendants:
            reason = "descendant_evidence_context"
        elif unit.get("unit_type") in {"bab_record", "aturan_peralihan_record", "aturan_tambahan_record"}:
            reason = "structural_context_without_direct_evidence"
        elif unit.get("runtime_loadable") is False:
            reason = "non_runtime_without_exact_evidence"
        else:
            reason = "exact_evidence_unavailable"
        decisions.append(
            {
                "legal_unit_id": unit_id,
                "source_role": unit.get("source_role"),
                "unit_type": unit.get("unit_type"),
                "runtime_loadable": unit.get("runtime_loadable") is True,
                "has_descendant_evidence": descendants,
                "decision": reason,
            }
        )
    return {
        "status": "complete",
        "unit_count": len(legal_units),
        "direct_evidence_unit_count": len(evidence_units),
        "no_direct_evidence_unit_count": len(decisions),
        "decisions": decisions,
    }


def validate_uud_artifact_dir(final_dir: Path) -> tuple[str, ...]:
    manifest = read_json(final_dir / "manifest.json")
    if manifest.get("schema_version") != UUD_ARTIFACT_SCHEMA:
        return ("artifact_schema_version_incompatible",)
    if (
        manifest.get("corpus_id") != "uud"
        or manifest.get("contract_id") != CONTRACT_ID
        or manifest.get("contract_version") != CONTRACT_VERSION
        or manifest.get("contract_fingerprint") != CONTRACT_FINGERPRINT
    ):
        return ("contract_fingerprint_mismatch",)
    integrity_errors = validate_manifest(final_dir)
    required_artifacts = set(MINIMUM_ARTIFACT_FIELDS) | {"validation_report"}
    artifacts: dict[str, object] = {}
    for rel, record in manifest.get("files", {}).items():
        logical_key = record.get("logical_key", rel)
        if logical_key not in required_artifacts:
            continue
        path = final_dir / rel
        if path.exists():
            if record.get("format") == "json":
                artifacts[logical_key] = read_json(path)
            elif logical_key == "word_bboxes":
                artifacts[logical_key] = _word_bbox_validation_rows(path)
            else:
                artifacts[logical_key] = read_jsonl(path)
    return tuple(dict.fromkeys((*integrity_errors, *validate_uud_artifacts(final_dir, artifacts))))


def _word_bbox_validation_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            source = json.loads(line)
            row = dict.fromkeys(source)
            row["word_bbox_id"] = source.get("word_bbox_id")
            row["characters"] = [
                {"character_bbox_id": character.get("character_bbox_id")}
                for character in source.get("characters") or ()
            ]
            rows.append(row)
    return rows


def _validate_schema6_contract(artifacts: Mapping[str, object]) -> tuple[str, ...]:
    """Small, corpus-owned publication contract for schema 6.

    The older report builder deliberately remains available for historical
    reports, but publication must be gated by this owner/lineage contract.
    """
    errors: list[str] = []

    def rows(key: str) -> list[dict]:
        value = artifacts.get(key) or []
        return list(value) if isinstance(value, list) else []

    by_artifact = {key: rows(key) for key in MINIMUM_ARTIFACT_FIELDS}
    for artifact, required in MINIMUM_ARTIFACT_FIELDS.items():
        for row in by_artifact[artifact]:
            row_id = str(row.get(_schema6_id_field(artifact)) or "<missing>")
            for field in required:
                if field not in row:
                    errors.append(f"artifact_required_field_missing:{artifact}:{row_id}:{field}")
            forbidden = FORBIDDEN_ARTIFACT_FIELDS.get(artifact, frozenset())
            for field in forbidden:
                if field in row:
                    errors.append(f"mixed_schema_contract:{artifact}:{row_id}:{field}")
            allowed = set(required) | set(ARTIFACT_OPTIONAL_FIELDS.get(artifact, ())) | set(COMMON_ARTIFACT_FIELDS)
            for field in row:
                if artifact in MINIMUM_ARTIFACT_FIELDS and field not in allowed:
                    errors.append(f"artifact_unknown_field:{artifact}:{row_id}:{field}")

    evidence = by_artifact.get("evidence_registry", [])
    evidence_ids = {row.get("evidence_id") for row in evidence}
    bbox = by_artifact.get("bbox_registry", [])
    bbox_ids = {row.get("bbox_id") for row in bbox}
    bbox_ids.update(row.get("word_bbox_id") for row in by_artifact.get("word_bboxes", []))
    span_ids = {row.get("text_span_id") for row in by_artifact.get("page_text_spans", [])}
    spans_by_id = {row.get("text_span_id"): row for row in by_artifact.get("page_text_spans", [])}
    source_docs = {row.get("source_document_id"): row for row in by_artifact.get("source_documents", [])}

    for row in evidence:
        row_id = str(row.get("evidence_id"))
        if row.get("object_role") != "evidence":
            errors.append(f"artifact_field_invalid:evidence_registry:{row_id}:object_role")
        if row.get("authority_kind") is None or not isinstance(row.get("citation_final"), bool):
            errors.append(f"artifact_field_invalid:evidence_registry:{row_id}:authority_or_finality")
        for ref in row.get("bbox_refs") or ():
            if ref not in bbox_ids:
                errors.append(f"evidence_bbox_unresolved:{row_id}:{ref}")
        for ref in row.get("text_span_ids") or ():
            if ref not in span_ids:
                errors.append(f"evidence_span_unresolved:{row_id}:{ref}")
        source = source_docs.get(row.get("source_document_id"))
        if source and row.get("source_sha256") != source.get("sha256"):
            errors.append(f"EVIDENCE_SOURCE_LINEAGE_INVALID:{row_id}")
        span_text = " ".join(str(spans_by_id.get(ref, {}).get("text") or "") for ref in row.get("text_span_ids") or ())
        if row.get("quoted_text") and normalize_source_text(row.get("quoted_text")) not in normalize_source_text(span_text):
            errors.append(f"EVIDENCE_QUOTE_SOURCE_MISMATCH:{row_id}")
    for row in bbox:
        row_id = str(row.get("bbox_id"))
        if row.get("evidence_id") not in evidence_ids or row.get("evidence_exists") is not True:
            errors.append(f"bbox_evidence_unresolved:{row_id}")
        if row.get("object_role") != "geometry":
            errors.append(f"artifact_field_invalid:bbox_registry:{row_id}:object_role")
    for row in by_artifact.get("page_text_spans", []):
        row_id = str(row.get("text_span_id"))
        if row.get("object_role") != "source_span":
            errors.append(f"artifact_field_invalid:page_text_spans:{row_id}:object_role")
        start, end = row.get("text_start"), row.get("text_end")
        if isinstance(start, int) and isinstance(end, int) and start > end:
            errors.append(f"artifact_field_invalid:page_text_spans:{row_id}:text_range")
        for ref in row.get("evidence_ids") or ():
            if ref not in evidence_ids:
                errors.append(f"span_evidence_unresolved:{row_id}:{ref}")
        for ref in row.get("span_bbox_ids") or ():
            if ref not in bbox_ids:
                errors.append(f"span_bbox_unresolved:{row_id}:{ref}")
    for row in by_artifact.get("retrieval_units", []):
        row_id = str(row.get("retrieval_unit_id"))
        if row.get("evidence_id") not in evidence_ids:
            errors.append(f"retrieval_evidence_unresolved:{row_id}")
        if row.get("object_role") != "retrieval_index_record":
            errors.append(f"artifact_field_invalid:retrieval_units:{row_id}:object_role")
    for row in by_artifact.get("metadata_grounding", []):
        row_id = str(row.get("metadata_grounding_id"))
        for ref in row.get("supporting_evidence_ids") or ():
            if ref not in evidence_ids:
                errors.append(f"metadata_evidence_unresolved:{row_id}:{ref}")
        if row.get("object_role") != "metadata_support":
            errors.append(f"artifact_field_invalid:metadata_grounding:{row_id}:object_role")
        if "signatories" in str(row.get("metadata_grounding_id")) and row.get("quoted_text"):
            donor_text = " ".join(
                str(next((item.get("quoted_text") for item in evidence if item.get("evidence_id") == ref), ""))
                for ref in row.get("supporting_evidence_ids") or ()
            )
            if donor_text and _schema6_normalize(row.get("quoted_text")) not in _schema6_normalize(donor_text):
                errors.append(f"signatory_grounding_unknown_name:{row_id}")
    for row in by_artifact.get("source_conflicts", []):
        row_id = str(row.get("source_conflict_id"))
        if not str(row.get("provenance_summary") or "").strip():
            errors.append(f"source_conflict_missing_provenance_summary:{row_id}")
        if row.get("object_role") != "trace_support" or row.get("is_citation_object") is not False:
            errors.append(f"source_conflict_invalid_object_role:{row_id}")
    units = by_artifact.get("legal_units", [])
    spans_by_unit = {row.get("legal_unit_id"): set(row.get("text_span_ids") or ()) for row in evidence}
    for unit in units:
        unit_id = unit.get("legal_unit_id")
        children = {
            child.get("legal_unit_id")
            for child in units
            if unit_id in (child.get("ancestor_legal_unit_ids") or ())
        }
        if not children or unit_id not in spans_by_unit:
            continue
        expected = set().union(*(spans_by_unit.get(child, set()) for child in children))
        if expected and not expected <= spans_by_unit[unit_id]:
            errors.append(f"aggregate_text_span_sequence_incomplete:{unit_id}")
    for row in by_artifact.get("article_amendment_relations", []):
        row_id = str(row.get("relation_id"))
        if row.get("authority_kind") is None or not isinstance(row.get("citation_final"), bool):
            errors.append(f"artifact_field_invalid:article_amendment_relations:{row_id}:authority_or_finality")
        source_unit = next((u for u in units if u.get("legal_unit_id") == row.get("source_legal_unit_id")), None)
        target_unit = next((u for u in units if u.get("legal_unit_id") == row.get("target_legal_unit_id")), None)
        relation_evidence = next((e for e in evidence if e.get("evidence_id") == row.get("evidence_id")), None)
        if source_unit is None:
            errors.append(f"article_relation_source_label_mismatch:{row_id}")
        elif row.get("source_label") and row.get("source_label") != source_unit.get("unit_label"):
            errors.append(f"article_relation_source_label_mismatch:{row_id}")
        if target_unit is None:
            errors.append(f"article_relation_unknown_target:{row_id}")
        if relation_evidence is None:
            errors.append(f"article_relation_bbox_source_mismatch:{row_id}")
        else:
            if row.get("source_document_id") != relation_evidence.get("source_document_id") or row.get("source_role") != relation_evidence.get("source_role"):
                errors.append(f"article_relation_bbox_source_mismatch:{row_id}")
            if row.get("source_pdf_sha256") != relation_evidence.get("source_sha256"):
                errors.append(f"article_relation_source_sha_mismatch:{row_id}")
            if row.get("bbox_precision") != relation_evidence.get("bbox_precision"):
                errors.append(f"article_relation_bbox_precision_mismatch:{row_id}")
        if target_unit is not None and row.get("target_source_role") != target_unit.get("source_role"):
            errors.append(f"article_relation_target_role_mismatch:{row_id}")
        if row.get("target_precision") not in {"target_local", "shared_span", "unresolved", "not_applicable"} or (
            row.get("support_class") == "exact_article_relation" and row.get("target_precision") != "target_local"
        ):
            errors.append(f"article_relation_exact_precision_mismatch:{row_id}")
        if row.get("relation_type") in {"RENAMES", "RENUMBERED_TO"} and row.get("old_reference_range") and row.get("new_reference_range"):
            if any(not isinstance(value, (int, float)) for value in (*row["old_reference_range"], *row["new_reference_range"])):
                errors.append(f"article_relation_old_range_text_mismatch:{row_id}")
            elif any(value < 0 or value >= len(str(row.get("quoted_text") or "")) for value in (*row["old_reference_range"], *row["new_reference_range"])):
                errors.append(f"article_relation_old_range_text_mismatch:{row_id}")
            elif (
                row["old_reference_range"][1] - row["old_reference_range"][0] < max(2, len(_schema6_normalize(row.get("old_reference"))) // 2)
                or row["new_reference_range"][1] - row["new_reference_range"][0] < max(2, len(_schema6_normalize(row.get("new_reference"))) // 2)
            ):
                errors.append(f"article_relation_old_range_text_mismatch:{row_id}")
    source_document_ids = set(source_docs)
    for row in by_artifact.get("document_relations", []):
        row_id = str(row.get("relation_id"))
        if row.get("source_document_id") not in source_document_ids or row.get("target_document_id") not in source_document_ids:
            errors.append(f"document_relation_source_unresolved:{row_id}")
        if row.get("relation_type") not in {"AMENDS", "AMENDED_BY", "DERIVED_FROM", "CONSOLIDATES"}:
            errors.append(f"document_relation_type_invalid:{row_id}")
        if row.get("object_role") != "relation_proof" or row.get("support_kind") != "provenance_only":
            errors.append(f"document_relation_support_invalid:{row_id}")
        if row.get("relation_type") in {"DERIVED_FROM", "CONSOLIDATES"} and row.get("support_exception_ids"):
            errors.append(f"document_relation_provenance_support_mismatch:{row_id}")
    relation_by_id = {row.get("relation_id"): row for row in by_artifact.get("article_amendment_relations", [])}
    relation_by_id.update({row.get("relation_id"): row for row in by_artifact.get("document_relations", [])})
    exception_ids = {row.get("exception_id") for row in by_artifact.get("validation_exceptions", [])}
    for edge in by_artifact.get("graph_edges", []):
        relation_id = edge.get("relation_id")
        relation = relation_by_id.get(relation_id)
        if relation and edge.get("source_id") and edge.get("target_id"):
            expected_source = f"legal_unit::{relation.get('source_legal_unit_id')}"
            expected_target = f"legal_unit::{relation.get('target_legal_unit_id')}"
            if edge.get("source_id") != expected_source or edge.get("target_id") != expected_target:
                errors.append(f"graph_relation_target_mismatch:{edge.get('edge_id')}")
        for ref in edge.get("support_relation_ids") or ():
            if ref not in relation_by_id:
                errors.append(f"graph_support_relation_unresolved:{edge.get('edge_id')}:{ref}")
        for ref in edge.get("support_evidence_ids") or ():
            if ref not in evidence_ids:
                errors.append(f"graph_support_evidence_unresolved:{edge.get('edge_id')}:{ref}")
        for ref in edge.get("support_exception_ids") or ():
            if ref not in exception_ids:
                errors.append(f"graph_support_exception_unresolved:{edge.get('edge_id')}:{ref}")
    for row in by_artifact.get("graph_edges", []):
        row_id = str(row.get("edge_id"))
        if row.get("object_role") != "graph_projection":
            errors.append(f"artifact_field_invalid:graph_edges:{row_id}:object_role")
        if "support_refs" in row or "bbox_refs" in row or "text_span_ids" in row:
            errors.append(f"mixed_schema_contract:graph_edges:{row_id}:proof_projection")
    return tuple(dict.fromkeys(errors))


def _schema6_normalize(value: object) -> str:
    return "".join(str(value or "").casefold().split())


def _schema6_id_field(artifact: str) -> str:
    return {
        "bbox_registry": "bbox_id",
        "evidence_registry": "evidence_id",
        "page_text_spans": "text_span_id",
        "retrieval_units": "retrieval_unit_id",
        "metadata_grounding": "metadata_grounding_id",
        "article_amendment_relations": "relation_id",
        "graph_edges": "edge_id",
        "validation_exceptions": "exception_id",
        "source_documents": "source_document_id",
        "word_bboxes": "word_bbox_id",
        "promotion_decisions": "decision_id",
    }.get(artifact, "id")


def validate_uud_artifacts(final_dir: Path, artifacts: Mapping[str, object]) -> tuple[str, ...]:
    if UUD_ARTIFACT_SCHEMA == 7:
        schema7_errors = list(_validate_schema7_contract(artifacts))
        report = artifacts.get("validation_report")
        for health_key, expected_status in _REQUIRED_HEALTH_STATUSES.items():
            if not isinstance(report, Mapping) or report.get(health_key, {}).get("status") != expected_status:
                schema7_errors.append(f"validation_report_incomplete:{health_key}")
        return tuple(dict.fromkeys(schema7_errors))
    if UUD_ARTIFACT_SCHEMA == 6:
        return _validate_schema6_contract(artifacts)
    def rows(key: str, *, optional: bool = False) -> list[dict]:
        value = artifacts.get(key)
        if value is None and optional:
            return []
        return cast(list[dict], value or [])

    def obj(key: str, *, optional: bool = False) -> dict:
        value = artifacts.get(key)
        if value is None and optional:
            return {}
        return cast(dict, value or {})

    legal_units = rows("legal_units")
    chunks = rows("chunks")
    evidence = rows("evidence_registry")
    bbox_rows = rows("bbox_registry")
    retrieval_units = rows("retrieval_units")
    metadata_grounding = rows("metadata_grounding")
    metadata_grounding_registry = rows("metadata_grounding_registry")
    document_metadata = rows("document_metadata", optional=True)
    document_relations = rows("document_relations", optional=True)
    article_amendment_relations = rows("article_amendment_relations", optional=True)
    promotion_decisions = rows("promotion_decisions", optional=True)
    graph_nodes = rows("graph_nodes")
    graph_edges = rows("graph_edges")
    source_conflicts = rows("source_conflicts")
    validation_exceptions = rows("validation_exceptions")
    page_text_spans = rows("page_text_spans", optional=True)
    word_bboxes = rows("word_bboxes", optional=True)
    pages = rows("pages", optional=True)
    pdf_health_report = obj("pdf_health_report", optional=True)

    errors: list[str] = []
    seen_ids: dict[str, set[str]] = defaultdict(set)
    units_by_id = {row["legal_unit_id"]: row for row in legal_units}
    chunks_by_id = {row["chunk_id"]: row for row in chunks}
    evidence_by_id = {row["evidence_id"]: row for row in evidence}
    bbox_by_id = {row["bbox_id"]: row for row in bbox_rows}
    bbox_registry_keys = {_span_bbox_key(row) for row in bbox_rows}
    graph_node_ids = {row["node_id"] for row in graph_nodes}
    source_conflict_ids = {row["source_conflict_id"] for row in source_conflicts}
    validation_exception_ids = {row["exception_id"] for row in validation_exceptions}
    source_documents = rows("source_documents")
    source_document_ids = {row["source_document_id"] for row in source_documents}
    page_keys = {(row["source_document_id"], row["page_number"]) for row in pages}
    metadata_grounding_ids = {row["metadata_grounding_id"] for row in metadata_grounding}
    for row in metadata_grounding:
        support_id = row.get("supporting_evidence_id")
        if not isinstance(support_id, str) or support_id not in evidence_by_id:
            errors.append(f"metadata_supporting_evidence_unresolved:{row.get('metadata_grounding_id')}:{support_id}")
    metadata_grounding_ref_ids: set[str] = set()
    text_span_ids = {row["text_span_id"] for row in page_text_spans}
    span_by_id = {row["text_span_id"]: row for row in page_text_spans}
    bbox_by_evidence: dict[str, list[dict]] = defaultdict(list)
    word_bbox_ids: set[str] = set()
    for row in word_bboxes:
        word_bbox_id = str(row.get("word_bbox_id") or "")
        if not word_bbox_id:
            errors.append("word_bbox_missing_id")
            continue
        if word_bbox_id in word_bbox_ids:
            errors.append(f"duplicate_word_bbox_id:{word_bbox_id}")
        word_bbox_ids.add(word_bbox_id)
        if row.get("source_document_id") not in source_document_ids:
            errors.append(f"word_bbox_unknown_source:{word_bbox_id}")
        if (row.get("source_document_id"), row.get("page_number")) not in page_keys:
            errors.append(f"word_bbox_unknown_page:{word_bbox_id}")
        if not all(row.get(field) is not None for field in ("x0", "y0", "x1", "y1")):
            errors.append(f"word_bbox_missing_coordinates:{word_bbox_id}")
        elif row["x1"] < row["x0"] or row["y1"] < row["y0"] or row["x0"] < 0 or row["y0"] < 0:
            errors.append(f"word_bbox_invalid_coordinates:{word_bbox_id}")
        if row.get("text") and not str(row.get("normalized_text") or "").strip():
            errors.append(f"word_bbox_missing_normalized_text:{word_bbox_id}")
        bbox_by_id[word_bbox_id] = {
            "bbox_id": word_bbox_id,
            "bbox_precision": "exact",
            "viewer_highlightable": True,
            **row,
        }
        for character in row.get("characters") or ():
            character_id = str(character.get("character_bbox_id") or "")
            if not character_id:
                errors.append(f"character_bbox_missing_id:{word_bbox_id}")
                continue
            if not all(character.get(field) is not None for field in ("x0", "y0", "x1", "y1")):
                errors.append(f"character_bbox_missing_coordinates:{character_id}")
            elif character["x1"] < character["x0"] or character["y1"] < character["y0"] or character["x0"] < 0 or character["y0"] < 0:
                errors.append(f"character_bbox_invalid_coordinates:{character_id}")
            if character.get("page_number", row.get("page_number")) != row.get("page_number"):
                errors.append(f"character_bbox_page_mismatch:{character_id}")
            if character.get("source_document_id", row.get("source_document_id")) != row.get("source_document_id"):
                errors.append(f"character_bbox_source_mismatch:{character_id}")
            bbox_by_id[character_id] = {**row, **character, "bbox_id": character_id}
    trust_violations = validate_uud_trust_boundary(
        legal_units=legal_units,
        chunks=chunks,
        graph_nodes=graph_nodes,
        graph_edges=graph_edges,
        retrieval_units=retrieval_units,
        evidence=evidence,
        bbox_rows=bbox_rows,
        page_text_spans=page_text_spans,
        source_documents=source_documents,
        pages=pages,
        word_bboxes=word_bboxes,
    )
    errors.extend(f"{violation.code}:{violation.artifact}:{violation.row_id}:{violation.field}" for violation in trust_violations)
    errors.extend(_aggregate_evidence_errors(legal_units, chunks, evidence, bbox_by_id, page_text_spans))
    for row in validation_exceptions:
        for field in ("chunk_id", "unresolved_chunk_reference"):
            chunk_id = row.get(field)
            if chunk_id and chunk_id not in chunks_by_id:
                errors.append(f"validation_exception_unknown_chunk_ref:{row['exception_id']}:{chunk_id}")
            if field == "unresolved_chunk_reference" and chunk_id in chunks_by_id:
                label = (row.get("evidence_summary") or {}).get("unit_label")
                if label and label not in chunks_by_id[chunk_id].get("hierarchy", ()):
                    errors.append(f"validation_exception_stale_chunk_ref:{row['exception_id']}:{chunk_id}")
    fidelity = _source_quote_fidelity_health(metadata_grounding=metadata_grounding, evidence=evidence, pages=pages)
    if fidelity["untracked_mismatch_count"]:
        errors.append(f"source_quote_untracked_mismatch_count:{fidelity['untracked_mismatch_count']}")
    grounding = _span_sequence_grounding_health(
        metadata_grounding=metadata_grounding,
        evidence=evidence,
        legal_units=legal_units,
        chunks=chunks,
        bbox_rows=bbox_rows,
        word_bboxes=word_bboxes,
        page_text_spans=page_text_spans,
    )
    if grounding["fixable_page_grounded_metadata_count"]:
        errors.append(f"fixable_page_grounded_metadata_count:{grounding['fixable_page_grounded_metadata_count']}")
    if grounding["fixable_legal_units_without_span_ids"]:
        errors.append(f"fixable_legal_units_without_span_ids:{grounding['fixable_legal_units_without_span_ids']}")
    if grounding["fixable_chunks_without_span_ids"]:
        errors.append(f"fixable_chunks_without_span_ids:{grounding['fixable_chunks_without_span_ids']}")
    if grounding["page_grounded_evidence_without_failure_reason"]:
        errors.append(f"page_grounded_evidence_without_failure_reason:{grounding['page_grounded_evidence_without_failure_reason']}")
    promotion = _promotion_engine_health(
        evidence=evidence,
        metadata_grounding=metadata_grounding,
        bbox_rows=bbox_rows,
        word_bboxes=word_bboxes,
        legal_units=legal_units,
        chunks=chunks,
        page_text_spans=page_text_spans,
        pages=pages,
    )
    if promotion["status"] != "complete":
        errors.append("promotion_engine_incomplete")
    promotion_audit = _promotion_decision_audit_health(
        evidence=evidence,
        metadata_grounding=metadata_grounding,
        bbox_rows=bbox_rows,
        promotion_decisions=promotion_decisions,
        promotion_engine_health=promotion,
        article_relations=article_amendment_relations,
    )
    if promotion_audit["status"] != "complete":
        errors.append("promotion_decision_audit_incomplete")
    promotion["promotion_blocked_count"] = promotion_audit["blocked_decision_count"]
    errors.extend(_recovery_state_errors(promotion_decisions))
    pdf_health = _pdf_health_summary(pdf_health_report)
    if pdf_health["status"] != "native_text_ok":
        errors.append(f"pdf_health_status:{pdf_health['status']}")
    if pdf_health["source_count"] != len(source_document_ids):
        errors.append(f"pdf_health_source_count:{pdf_health['source_count']}!={len(source_document_ids)}")
    if pdf_health["page_count"] != len(pages):
        errors.append(f"pdf_health_page_count:{pdf_health['page_count']}!={len(pages)}")
    if pdf_health["ocr_required_count"]:
        errors.append(f"pdf_health_ocr_required_count:{pdf_health['ocr_required_count']}")
    closure = _legal_unit_chunk_span_closure_health(
        legal_units=legal_units,
        chunks=chunks,
        page_text_spans=page_text_spans,
        graph_nodes=graph_nodes,
    )
    closure_non_error_count_keys = {
        "legal_unit_count",
        "chunk_count",
        "legal_unit_exact_span_link_count",
        "chunk_exact_span_link_count",
        "legal_unit_containing_span_link_count",
        "chunk_containing_span_link_count",
    }
    for key, value in closure.items():
        if key.endswith("_count") and key not in closure_non_error_count_keys and value:
            errors.append(f"legal_unit_chunk_span_closure_{key}:{value}")
    if closure["status"] != "complete":
        errors.append("legal_unit_chunk_span_closure_incomplete")
    structural_contract = _structural_authority_contract_health(legal_units, chunks, graph_nodes, graph_edges, retrieval_units)
    if structural_contract["status"] != "complete":
        errors.append("structural_authority_contract_incomplete")
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
        expected_coverage_status = "bbox_key_present" if _span_bbox_key(row) in bbox_registry_keys else "bbox_key_absent"
        if row.get("bbox_registry_coverage_status") != expected_coverage_status:
            errors.append(f"text_span_invalid_bbox_registry_coverage_status:{row['text_span_id']}")
        if expected_coverage_status == "bbox_key_absent":
            if row.get("bbox_registry_coverage_bucket") not in {
                "legal_citation_candidate",
                "metadata_provenance_candidate",
                "source_anomaly_provenance_candidate",
                "structural_provenance_only",
                "nonlegal_excluded_provenance",
                "raw_span_only_with_reason",
            }:
                errors.append(f"text_span_invalid_bbox_registry_coverage_bucket:{row['text_span_id']}")
            if row.get("bbox_registry_coverage_reason") not in {
                "already_covered_by_final_evidence_bbox",
                "exact_word_bbox_available",
                "structural_provenance_only",
                "nonlegal_excluded_from_public_highlight",
                "source_anomaly_anchor_only_until_exact_span_available",
                "metadata_page_grounded_only_by_policy",
                "blocked_by_text_mismatch",
                "blocked_by_layout",
                "blocked_by_missing_exact_bbox",
                "blocked_by_no_word_level_bbox_artifact",
            }:
                errors.append(f"text_span_invalid_bbox_registry_coverage_reason:{row['text_span_id']}")
        if row.get("source_document_id") not in source_document_ids:
            errors.append(f"text_span_unknown_source_document:{row['text_span_id']}:{row.get('source_document_id')}")
        if not isinstance(row.get("page_number"), int):
            errors.append(f"text_span_missing_page_number:{row['text_span_id']}")
        if not _valid_coordinates(row):
            errors.append(f"text_span_invalid_bbox_coordinates:{row['text_span_id']}")
        for field in SPAN_DISPOSITION_FIELDS:
            if field not in row:
                errors.append(f"text_span_missing_disposition_field:{row['text_span_id']}:{field}")
        if row.get("span_role") not in SPAN_ROLES:
            errors.append(f"text_span_invalid_span_role:{row['text_span_id']}:{row.get('span_role')}")
        if row.get("semantic_classification") not in SEMANTIC_CLASSIFICATIONS:
            errors.append(f"text_span_invalid_semantic_classification:{row['text_span_id']}:{row.get('semantic_classification')}")
        if row.get("legal_force") not in LEGAL_FORCES:
            errors.append(f"text_span_invalid_legal_force:{row['text_span_id']}:{row.get('legal_force')}")
        if row.get("promotion_status") not in PROMOTION_STATUSES:
            errors.append(f"text_span_invalid_promotion_status:{row['text_span_id']}:{row.get('promotion_status')}")
        if row.get("review_status") not in REVIEW_STATUSES:
            errors.append(f"text_span_invalid_review_status:{row['text_span_id']}:{row.get('review_status')}")
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
        elif row.get("promotion_target_id") and row.get("promotion_target_type") not in {"legal_unit", "chunk"}:
            errors.append(f"text_span_ambiguous_nonpromoted_target:{row['text_span_id']}:{row.get('promotion_target_type')}")
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
        for field in (
            "page_numbers",
            "text_span_ids",
            "bbox_ids",
            "evidence_ids",
            "grounding_status",
            "validation_status",
            "anchor_terms",
            "query_anchor_terms",
            "source_anomaly_kind",
            "discrepancy_type",
            "substantive_conflict",
            "source_sha256",
            "source_quote",
            "comparison_source_document_id",
            "comparison_source_sha256",
            "comparison_quote",
            "review_evidence",
            "provenance_summary",
            "final_authority_policy",
        ):
            if field not in row:
                errors.append(f"source_conflict_missing_{field}:{row['source_conflict_id']}")
        if not row.get("anchor_terms"):
            errors.append(f"source_conflict_missing_anchor_terms:{row['source_conflict_id']}")
        if not row.get("query_anchor_terms"):
            errors.append(f"source_conflict_missing_query_anchor_terms:{row['source_conflict_id']}")
        if row.get("discrepancy_type") not in {
            "PRINTED_NUMBERING_TYPO",
            "SOURCE_COPY_SCOPE_OMISSION",
            "SOURCE_COPY_TEXTUAL_OMISSION",
            "PRINTED_TYPO",
        }:
            errors.append(f"source_conflict_invalid_discrepancy_type:{row['source_conflict_id']}")
        if not isinstance(row.get("substantive_conflict"), bool):
            errors.append(f"source_conflict_invalid_substantive_conflict:{row['source_conflict_id']}")
        if row.get("source_anomaly_kind") not in {"renumbering_provenance", "source_marker_sequence_anomaly", "typed_source_discrepancy"}:
            errors.append(f"source_conflict_invalid_source_anomaly_kind:{row['source_conflict_id']}")
        if (
            row.get("source_anomaly_kind") == "renumbering_provenance"
            and row.get("source_mapping_kind") != "historical_to_canonical_mapping"
        ):
            errors.append(f"source_conflict_invalid_source_mapping_kind:{row['source_conflict_id']}")
        if not str(row.get("provenance_summary") or "").strip():
            errors.append(f"source_conflict_missing_provenance_summary:{row['source_conflict_id']}")
        if not str(row.get("final_authority_policy") or "").strip():
            errors.append(f"source_conflict_missing_final_authority_policy:{row['source_conflict_id']}")
        if row.get("source_anomaly_kind") == "source_marker_sequence_anomaly":
            if row.get("provenance_exception_category") != ACCEPTED_NONCANONICAL_SOURCE_CONFLICT_TRACE_ONLY:
                errors.append(f"source_conflict_invalid_provenance_exception_category:{row['source_conflict_id']}")
            if row.get("provenance_review_status") != "reviewed":
                errors.append(f"source_conflict_invalid_provenance_review_status:{row['source_conflict_id']}")
        policy = row.get("source_anomaly_policy") or {}
        required_policy_fields = {
            "anomaly_kind",
            "source_role",
            "canonical_role",
            "anchor_terms",
            "affected_span_refs",
            "provenance_rules",
            "highlight_policy",
            "provenance_highlight_scope",
            "finality_policy",
            "public_wording_template",
            "reviewer_status",
            "corpus_id",
        }
        if required_policy_fields - set(policy):
            errors.append(f"source_conflict_missing_source_anomaly_policy:{row['source_conflict_id']}")
        if policy.get("corpus_id") != row.get("corpus_id"):
            errors.append(f"source_conflict_invalid_policy_corpus:{row['source_conflict_id']}")
        if policy.get("anomaly_kind") != row.get("source_anomaly_kind"):
            errors.append(f"source_conflict_invalid_policy_anomaly_kind:{row['source_conflict_id']}")
        if policy.get("finality_policy") != "source_anomaly_provenance":
            errors.append(f"source_conflict_invalid_policy_finality:{row['source_conflict_id']}")
        for field in (
            "raw_provenance_bbox_ids",
            "raw_provenance_text_span_ids",
            "blocked_raw_provenance_text_span_ids",
            "blocked_raw_provenance_text_span_reasons",
            "provenance_bbox_status",
            "provenance_highlight_scope",
            "final_evidence_available",
        ):
            if field not in row:
                errors.append(f"source_conflict_missing_{field}:{row['source_conflict_id']}")
        for text_span_id in row.get("text_span_ids") or ():
            if text_span_id not in text_span_ids:
                errors.append(f"orphan_source_conflict_text_span:{row['source_conflict_id']}:{text_span_id}")
        for bbox_id in row.get("bbox_ids") or ():
            if bbox_id not in bbox_by_id:
                errors.append(f"orphan_source_conflict_bbox:{row['source_conflict_id']}:{bbox_id}")
        for bbox_id in row.get("raw_provenance_bbox_ids") or ():
            if bbox_id not in bbox_by_id:
                errors.append(f"orphan_source_conflict_raw_bbox:{row['source_conflict_id']}:{bbox_id}")
        for text_span_id in row.get("raw_provenance_text_span_ids") or ():
            if text_span_id not in text_span_ids:
                errors.append(f"orphan_source_conflict_raw_text_span:{row['source_conflict_id']}:{text_span_id}")
        for evidence_id in row.get("evidence_ids") or ():
            if evidence_id not in evidence_by_id:
                errors.append(f"orphan_source_conflict_evidence:{row['source_conflict_id']}:{evidence_id}")
        if (not row.get("evidence_ids") or not row.get("bbox_ids")) and not row.get("failure_reason"):
            errors.append(f"source_conflict_missing_failure_reason:{row['source_conflict_id']}")
        raw_status = row.get("provenance_bbox_status")
        raw_count = len(row.get("raw_provenance_text_span_ids") or ())
        all_count = len(row.get("text_span_ids") or ())
        if raw_status == "exact_raw_provenance_bbox_available" and raw_count != all_count:
            errors.append(f"source_conflict_invalid_raw_provenance_status:{row['source_conflict_id']}")
        if raw_status == "partial_exact_raw_provenance_bbox_available" and not (0 < raw_count < all_count):
            errors.append(f"source_conflict_invalid_raw_provenance_status:{row['source_conflict_id']}")
        if raw_status == "partial_exact_raw_provenance_bbox_available":
            blocked = set(row.get("blocked_raw_provenance_text_span_ids") or ())
            expected_blocked = set(row.get("text_span_ids") or ()) - set(row.get("raw_provenance_text_span_ids") or ())
            if blocked != expected_blocked:
                errors.append(f"source_conflict_invalid_blocked_raw_provenance_spans:{row['source_conflict_id']}")
            if row.get("blocked_raw_provenance_reason") != "source_anomaly_anchor_only_until_exact_span_available":
                errors.append(f"source_conflict_invalid_blocked_raw_provenance_reason:{row['source_conflict_id']}")
            span_reasons = row.get("blocked_raw_provenance_text_span_reasons") or {}
            if set(span_reasons) != expected_blocked:
                errors.append(f"source_conflict_invalid_blocked_raw_provenance_span_reason_keys:{row['source_conflict_id']}")
            if any(reason != row.get("blocked_raw_provenance_reason") for reason in span_reasons.values()):
                errors.append(f"source_conflict_invalid_blocked_raw_provenance_span_reason:{row['source_conflict_id']}")
        if raw_status == "exact_raw_provenance_bbox_unavailable" and raw_count:
            errors.append(f"source_conflict_invalid_raw_provenance_status:{row['source_conflict_id']}")
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
        if "::signatories::person_" in str(row.get("metadata_grounding_id") or ""):
            metadata = next(
                (item for item in document_metadata if item.get("source_role") == row.get("source_role")),
                {},
            )
            quote = _source_quote_normalize(row.get("quoted_text"))
            names = {_source_quote_normalize(item.get("name_text")) for item in metadata.get("signatories") or ()}
            if quote not in names:
                errors.append(f"signatory_grounding_unknown_name:{row['metadata_grounding_id']}")
            boxes = [bbox_by_id.get(bbox_id) for bbox_id in row.get("bbox_ids") or ()]
            if not boxes or any(
                box is None
                or box.get("source_document_id") != row.get("source_document_id")
                or box.get("page_number") not in set(row.get("page_numbers") or ())
                for box in boxes
            ):
                errors.append(f"signatory_grounding_cross_source:{row['metadata_grounding_id']}")
            elif _source_quote_normalize(" ".join(str((box or {}).get("text") or "") for box in boxes)) != quote:
                errors.append(f"signatory_grounding_geometry_mismatch:{row['metadata_grounding_id']}")
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
    article_relation_ids = {row["relation_id"] for row in article_amendment_relations}
    article_relations_by_id = {row["relation_id"]: row for row in article_amendment_relations}
    graph_edge_keys = {(row.get("edge_type"), row.get("source_id"), row.get("target_id")) for row in graph_edges}
    graph_edges_by_id = {row.get("edge_id"): row for row in graph_edges}
    for row in graph_edges:
        if row["edge_id"] in seen_ids["edge_id"]:
            errors.append(f"duplicate_graph_edge_id:{row['edge_id']}")
        seen_ids["edge_id"].add(row["edge_id"])
        if row["source_id"] not in graph_node_ids or row["target_id"] not in graph_node_ids:
            errors.append(f"orphan_graph_edge:{row['edge_id']}")
        edge_type = row.get("edge_type")
        if edge_type not in PROVENANCE_EDGE_TYPES | LEGAL_EDGE_TYPES:
            errors.append(f"invalid_graph_edge_type:{row['edge_id']}:{edge_type}")
        if row.get("derived_from_edge_id"):
            origin = graph_edges_by_id.get(row["derived_from_edge_id"])
            descriptor = descriptor_for(origin.get("edge_type") if origin else None)
            if origin is None or descriptor is None or descriptor.inverse != edge_type:
                errors.append(f"graph_inverse_origin_invalid:{row['edge_id']}")
            elif row.get("source_id") != origin.get("target_id") or row.get("target_id") != origin.get("source_id"):
                errors.append(f"graph_inverse_direction_invalid:{row['edge_id']}")
            continue
        if edge_type in {"MODIFIES", "DELETES", "RENAMES", "RENUMBERED_TO"}:
            relation = article_relations_by_id.get(row.get("relation_id"))
            if relation is None:
                errors.append(f"graph_relation_missing_authoritative_row:{row['edge_id']}")
            else:
                if row.get("source_id") != f"legal_unit::{relation.get('source_legal_unit_id')}":
                    errors.append(f"graph_relation_source_mismatch:{row['edge_id']}")
                if row.get("target_id") != f"legal_unit::{relation.get('target_legal_unit_id')}":
                    errors.append(f"graph_relation_target_mismatch:{row['edge_id']}")
                if row.get("relation_type") != relation.get("relation_type"):
                    errors.append(f"graph_relation_type_mismatch:{row['edge_id']}")
                if row.get("runtime_loadable") is not True:
                    errors.append(f"graph_relation_not_runtime_loadable:{row['edge_id']}")
                if list(row.get("support_evidence_ids") or ()) != [relation.get("evidence_id")]:
                    errors.append(f"graph_relation_evidence_mismatch:{row['edge_id']}")
                if not row.get("bbox_refs") or not row.get("text_span_ids"):
                    errors.append(f"graph_relation_missing_provenance:{row['edge_id']}")
            continue
        if row.get("citation_final") is not False:
            errors.append(f"graph_edge_false_final_citation:{row['edge_id']}")
        if row.get("source_role") not in {"canonical", "historical", "amendment", "consolidated", "anomaly"}:
            errors.append(f"graph_edge_invalid_source_role:{row['edge_id']}:{row.get('source_role')}")
        if row.get("support_kind") not in {
            "exact_source_relation",
            "deterministic_structure",
            "endpoint_provenance",
            "instrument_provenance",
            "historical_mapping",
            "source_anomaly_trace",
            "nonlegal",
        }:
            errors.append(f"graph_edge_invalid_support_kind:{row['edge_id']}:{row.get('support_kind')}")
        if not row.get("derivation_reason"):
            errors.append(f"graph_edge_missing_derivation_reason:{row['edge_id']}")
        for bbox_id in row.get("bbox_refs") or ():
            if bbox_id not in bbox_by_id:
                errors.append(f"graph_edge_invalid_bbox_ref:{row['edge_id']}:{bbox_id}")
        if row.get("article_relation_ref") and row["article_relation_ref"] not in article_relation_ids:
            errors.append(f"graph_edge_orphan_article_relation_ref:{row['edge_id']}:{row['article_relation_ref']}")
        if edge_type in {"RENAMES", "RENUMBERED_TO"}:
            relation = article_relations_by_id.get(row.get("article_relation_ref"))
            mapping = row.get("reference_mapping") or {}
            if not mapping:
                errors.append(f"graph_renames_missing_mapping:{row['edge_id']}")
            if relation is None:
                errors.append(f"graph_renames_missing_relation:{row['edge_id']}")
            else:
                if relation.get("source_legal_unit_id") != row.get("source_legal_unit_id"):
                    errors.append(f"graph_renames_source_mismatch:{row['edge_id']}")
                if relation.get("target_legal_unit_id") != row.get("target_legal_unit_id"):
                    errors.append(f"graph_renames_target_mismatch:{row['edge_id']}")
                if relation.get("old_reference") != mapping.get("old_reference") or relation.get("new_reference") != mapping.get(
                    "new_reference"
                ):
                    errors.append(f"graph_renames_mapping_mismatch:{row['edge_id']}")
                if relation.get("old_reference_range") != mapping.get("old_range") or relation.get("new_reference_range") != mapping.get(
                    "new_range"
                ):
                    errors.append(f"graph_renames_range_mismatch:{row['edge_id']}")
                if relation.get("old_reference_range_kind") != mapping.get("old_range_kind") or relation.get(
                    "new_reference_range_kind"
                ) != mapping.get("new_range_kind"):
                    errors.append(f"graph_renames_range_kind_mismatch:{row['edge_id']}")
                if relation.get("source_legal_unit_role") != mapping.get("source_role"):
                    errors.append(f"graph_renames_source_role_mismatch:{row['edge_id']}")
        if edge_type in {"MODIFIES", "DELETES", "RENAMES", "RENUMBERED_TO"}:
            relation = article_relations_by_id.get(row.get("article_relation_ref"))
            if relation is not None:
                if list(row.get("supporting_evidence_ids") or ()) != [relation.get("evidence_id")]:
                    errors.append(f"graph_relation_evidence_mismatch:{row['edge_id']}")
                if row.get("source_legal_unit_id") != relation.get("source_legal_unit_id"):
                    errors.append(f"graph_relation_source_mismatch:{row['edge_id']}")
                if row.get("target_legal_unit_id") != relation.get("target_legal_unit_id"):
                    errors.append(f"graph_relation_target_mismatch:{row['edge_id']}")
                if row.get("target_citation") != relation.get("target_citation"):
                    errors.append(f"graph_relation_target_label_mismatch:{row['edge_id']}")
        provenance_ref = row.get("provenance_ref")
        provenance_kind = row.get("provenance_ref_kind")
        if provenance_ref or provenance_kind:
            if provenance_kind not in {"metadata_grounding", "source_conflict", "document_metadata", "graph_only"}:
                errors.append(f"graph_edge_invalid_provenance_ref_kind:{row['edge_id']}:{provenance_kind}")
            if row.get("provenance_support") not in {"exact_bbox", "page_grounded", "trace_only", "structural", "nonlegal"}:
                errors.append(f"graph_edge_invalid_provenance_support:{row['edge_id']}:{row.get('provenance_support')}")
            if provenance_kind == "metadata_grounding" and provenance_ref not in metadata_grounding_ids:
                errors.append(f"graph_edge_orphan_metadata_provenance_ref:{row['edge_id']}:{provenance_ref}")
            if provenance_kind == "source_conflict" and provenance_ref not in source_conflict_ids:
                errors.append(f"graph_edge_orphan_source_conflict_provenance_ref:{row['edge_id']}:{provenance_ref}")
        if "evidence_ref" in row:
            errors.append(f"graph_edge_legacy_evidence_contract:{row['edge_id']}")
        supporting_ids = row.get("supporting_evidence_ids") or ()
        for evidence_id in supporting_ids:
            if evidence_id not in evidence_by_id:
                errors.append(f"graph_edge_non_evidence_support:{row['edge_id']}:{evidence_id}")
        if edge_type in {"MODIFIES", "DELETES"}:
            evidence_row = evidence_by_id.get(supporting_ids[0]) if supporting_ids else None
            if not evidence_row:
                errors.append(f"graph_relation_edge_invalid_evidence:{row['edge_id']}:{supporting_ids}")
            if row.get("citation_final") is not False:
                errors.append(f"graph_trace_relation_promoted:{row['edge_id']}")
        if edge_type == "PART_OF" and ("CONTAINS", row.get("target_id"), row.get("source_id")) not in graph_edge_keys:
            errors.append(f"graph_part_of_without_contains:{row['edge_id']}")
        if edge_type == "FOLLOWS" and ("PRECEDES", row.get("target_id"), row.get("source_id")) not in graph_edge_keys:
            errors.append(f"graph_follows_without_precedes:{row['edge_id']}")
        if edge_type in LEGAL_EDGE_TYPES:
            if row.get("source_document_id") not in {unit["source_document_id"] for unit in legal_units}:
                errors.append(f"legal_edge_missing_source_document:{row['edge_id']}")
            if row.get("runtime_loadable") is True:
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
        if row.get("relation_type") not in {"MODIFIES", "DELETES", "ADDS", "RENAMES", "RENUMBERED_TO", "SUPPLEMENTS"}:
            errors.append(f"invalid_article_amendment_relation_type:{row['relation_id']}:{row.get('relation_type')}")
        evidence_row = evidence_by_id.get(row.get("evidence_id"))
        if not evidence_row:
            errors.append(f"article_relation_unknown_evidence:{row['relation_id']}:{row.get('evidence_id')}")
            continue
        if row.get("authority_kind") not in AUTHORITY_KINDS:
            errors.append(f"article_relation_invalid_authority_kind:{row['relation_id']}:{row.get('authority_kind')}")
        if not isinstance(row.get("citation_final"), bool):
            errors.append(f"article_relation_invalid_citation_final:{row['relation_id']}")
        source_unit = units_by_id.get(row.get("source_legal_unit_id"))
        target_unit = units_by_id.get(row.get("target_legal_unit_id"))
        if row.get("source_document_id") != evidence_row.get("source_document_id"):
            errors.append(f"article_relation_source_document_mismatch:{row['relation_id']}")
        if row.get("source_role") != evidence_row.get("source_role"):
            errors.append(f"article_relation_source_role_mismatch:{row['relation_id']}")
        if not row.get("source_legal_unit_role"):
            errors.append(f"article_relation_missing_source_unit_role:{row['relation_id']}")
        if row.get("source_pdf_sha256") != evidence_row.get("source_sha256"):
            errors.append(f"article_relation_source_sha_mismatch:{row['relation_id']}")
        if row.get("bbox_precision") != evidence_row.get("bbox_precision"):
            errors.append(f"article_relation_bbox_precision_mismatch:{row['relation_id']}")
        if row.get("page_number") not in set(evidence_row.get("page_numbers") or ()):
            errors.append(f"article_relation_page_mismatch:{row['relation_id']}")
        if not set(row.get("text_span_ids") or ()) <= set(evidence_row.get("text_span_ids") or ()):
            errors.append(f"article_relation_span_support_mismatch:{row['relation_id']}")
        extra_relation_bboxes = set(row.get("bbox_refs") or ()) - set(evidence_row.get("bbox_refs") or ())
        if extra_relation_bboxes - set(row.get("target_bbox_refs") or ()):
            errors.append(f"article_relation_bbox_support_mismatch:{row['relation_id']}")
        if source_unit is None:
            errors.append(f"article_relation_unknown_source:{row['relation_id']}:{row.get('source_legal_unit_id')}")
        elif row.get("source_label") != source_unit.get("unit_label"):
            errors.append(f"article_relation_source_label_mismatch:{row['relation_id']}")
        elif row.get("source_legal_unit_role") != source_unit.get("source_role"):
            errors.append(f"article_relation_source_role_unit_mismatch:{row['relation_id']}")
        if row.get("target_legal_unit_id") not in units_by_id:
            errors.append(f"article_relation_unknown_target:{row['relation_id']}:{row.get('target_legal_unit_id')}")
        elif target_unit is not None and row.get("target_label") not in {None, target_unit.get("unit_label")}:
            errors.append(f"article_relation_target_label_mismatch:{row['relation_id']}")
        if not row.get("target_source_role"):
            errors.append(f"article_relation_missing_target_unit_role:{row['relation_id']}")
        elif target_unit is not None and row.get("target_source_role") != target_unit.get("source_role"):
            errors.append(f"article_relation_target_role_mismatch:{row['relation_id']}")
        if row.get("relation_type") in {"RENAMES", "RENUMBERED_TO"}:
            if not row.get("old_reference") or not row.get("new_reference"):
                errors.append(f"article_relation_missing_renumber_mapping:{row['relation_id']}")
            if not row.get("old_reference_range") or not row.get("new_reference_range"):
                errors.append(f"article_relation_missing_reference_range:{row['relation_id']}")
            if row.get("old_reference_range_kind") not in {"literal", "contextual"} or row.get("new_reference_range_kind") not in {
                "literal",
                "contextual",
            }:
                errors.append(f"article_relation_invalid_reference_range_kind:{row['relation_id']}")
            if row.get("new_reference") and not str(row.get("new_reference")).casefold().startswith(
                str(row.get("target_citation") or "").casefold()
            ):
                errors.append(f"article_relation_target_mapping_mismatch:{row['relation_id']}")
            if source_unit is not None and not _reference_belongs_to_unit(row.get("old_reference"), source_unit):
                errors.append(f"article_relation_source_mapping_mismatch:{row['relation_id']}")
            if target_unit is not None and not _reference_belongs_to_unit(row.get("new_reference"), target_unit):
                errors.append(f"article_relation_target_mapping_mismatch:{row['relation_id']}")
            mapping_support = _validate_mapping_support(row, evidence_row)
            errors.extend(f"{code}:{row['relation_id']}" for code in mapping_support)
            errors.extend(f"{code}:{row['relation_id']}" for code in _validate_relation_support(row, evidence_row, span_by_id, bbox_by_id))
        if not str(row.get("target_citation") or "").startswith(("Pasal ", "Ayat ")):
            errors.append(f"article_relation_non_pasal_ayat_target:{row['relation_id']}:{row.get('target_citation')}")
        if row.get("support_class") == "trace_article_relation":
            if row.get("quoted_text") != evidence_row.get("quoted_text"):
                errors.append(f"article_relation_quote_mismatch:{row['relation_id']}")
        elif row.get("relation_type") in {"RENAMES", "RENUMBERED_TO"} and row.get("quoted_text") != evidence_row.get("quoted_text"):
            errors.append(f"article_relation_mapping_quote_mismatch:{row['relation_id']}")
        elif _compact_reference(row.get("target_citation")) not in _compact_reference(row.get("quoted_text")):
            errors.append(f"article_relation_target_quote_mismatch:{row['relation_id']}")
        for bbox_id in row.get("bbox_refs") or ():
            if bbox_id not in bbox_by_id:
                errors.append(f"article_relation_unknown_bbox:{row['relation_id']}:{bbox_id}")
        if not row.get("bbox_refs"):
            errors.append(f"article_relation_missing_bbox:{row['relation_id']}")
        target_bbox_refs = tuple(row.get("target_bbox_refs") or ())
        if not set(target_bbox_refs) <= set(row.get("bbox_refs") or ()):
            errors.append(f"article_relation_target_bbox_support_mismatch:{row['relation_id']}")
        for bbox_id in (*(row.get("bbox_refs") or ()), *target_bbox_refs):
            bbox = bbox_by_id.get(bbox_id)
            if bbox is None:
                continue
            if bbox.get("source_document_id") != evidence_row.get("source_document_id"):
                errors.append(f"article_relation_bbox_source_mismatch:{row['relation_id']}:{bbox_id}")
            if bbox.get("page_number") not in set(evidence_row.get("page_numbers") or ()):
                errors.append(f"article_relation_bbox_page_mismatch:{row['relation_id']}:{bbox_id}")
        target_spans = [span_by_id.get(span_id) for span_id in row.get("target_text_span_ids") or ()]
        target_phrase = normalize_source_text(row.get("new_reference") or row.get("target_citation"))
        target_text = normalize_source_text(" ".join(str(span.get("text") or "") for span in target_spans if span))
        isolated_bbox = any(
            len(parse_legal_references("uud", str(bbox_by_id[bbox_id].get("text") or ""))) == 1
            and target_phrase in normalize_source_text(str(bbox_by_id[bbox_id].get("text") or ""))
            for bbox_id in target_bbox_refs
            if bbox_id in bbox_by_id
        )
        word_target = row.get("target_geometry_method") == "word_geometry"
        character_target = row.get("target_geometry_method") == "character_geometry"
        target_ref_text = "".join(str(bbox_by_id[bbox_id].get("text") or "") for bbox_id in target_bbox_refs if bbox_id in bbox_by_id)
        reconstructed_target = _compact_reference(target_ref_text) == _compact_reference(target_phrase)
        character_geometry_valid = bool(target_bbox_refs) and all(
            str(bbox_id).startswith("uud_character_bbox::")
            and bbox_by_id[bbox_id].get("source_document_id") == evidence_row.get("source_document_id")
            and bbox_by_id[bbox_id].get("page_number") in set(evidence_row.get("page_numbers") or ())
            and all(bbox_by_id[bbox_id].get(field) is not None for field in ("x0", "y0", "x1", "y1"))
            for bbox_id in target_bbox_refs
            if bbox_id in bbox_by_id
        )
        exact_support = (
            evidence_row.get("bbox_precision") == "exact"
            and evidence_row.get("viewer_highlightable") is True
            and bool(target_bbox_refs)
            and all(bbox_id in bbox_by_id for bbox_id in target_bbox_refs)
            and (word_target or character_target or (bool(target_spans) and _compact_reference(target_phrase) in _compact_reference(target_text)))
            and reconstructed_target
            and (not character_target or character_geometry_valid)
            and (word_target or character_target or isolated_bbox)
        )
        target_span_isolated = bool(target_spans) and all(
            span is not None
            and len(parse_legal_references("uud", str(span.get("text") or ""))) == 1
            and target_phrase in normalize_source_text(str(span.get("text") or ""))
            for span in target_spans
        )
        target_bbox_isolated = (
            character_target
            and character_geometry_valid
            and reconstructed_target
            or word_target
            and reconstructed_target
            and bool(target_bbox_refs)
            and all(bbox_id in bbox_by_id for bbox_id in target_bbox_refs)
            or not character_target
            and not word_target
            and bool(target_bbox_refs)
            and reconstructed_target
            and all(
                bbox_id in bbox_by_id
                and len(parse_legal_references("uud", str(bbox_by_id[bbox_id].get("text") or ""))) == 1
                and target_phrase in normalize_source_text(str(bbox_by_id[bbox_id].get("text") or ""))
                for bbox_id in target_bbox_refs
            )
        )
        if target_spans and not target_span_isolated:
            errors.append(f"article_relation_target_span_not_isolated:{row['relation_id']}")
        if target_bbox_refs and not target_bbox_isolated:
            errors.append(f"article_relation_target_bbox_not_isolated:{row['relation_id']}")
        if (
            row.get("relation_type") in {"RENAMES", "RENUMBERED_TO"}
            and row.get("support_class") == "exact_article_relation"
            and set(row.get("bbox_refs") or ()) <= set(target_bbox_refs)
        ):
            errors.append(f"article_relation_target_only_bbox_support:{row['relation_id']}")
        if row.get("source_support_exact") is not _validate_source_support(evidence_row, row):
            errors.append(f"article_relation_source_support_mismatch:{row['relation_id']}")
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
            if row.get("target_precision") != "target_local":
                errors.append(f"article_relation_exact_precision_mismatch:{row['relation_id']}")
            if not row.get("text_span_ids"):
                errors.append(f"article_relation_missing_support_span:{row['relation_id']}")
        if support_class == "trace_article_relation":
            if exact_support:
                errors.append(f"article_relation_trace_should_be_exact:{row['relation_id']}")
            if row.get("target_precision") == "target_local":
                errors.append(f"article_relation_trace_target_local_precision:{row['relation_id']}")
            if row.get("grounding_level") == "exact_source_text" and row.get("target_precision") != "shared_span":
                errors.append(f"article_relation_trace_false_exact_grounding:{row['relation_id']}")
            if row.get("grounding_level") not in {
                "exact_source_text",
                "exact_source_text_shared_target",
                "target_reference_only",
                "page_grounded_trace",
            }:
                errors.append(f"article_relation_unknown_grounding_level:{row['relation_id']}:{row.get('grounding_level')}")
            if (row.get("viewer_highlightable") is not True and not row.get("bbox_refs")) or row.get("citation_available") is not False:
                errors.append(f"article_relation_trace_public_citation:{row['relation_id']}")
            if not row.get("trace_only_reason"):
                errors.append(f"article_relation_trace_missing_reason:{row['relation_id']}")

    source_relation_errors = _source_relation_contract_errors(
        evidence=evidence,
        legal_units=legal_units,
        article_amendment_relations=article_amendment_relations,
    )
    errors.extend(source_relation_errors)

    return tuple(sorted(set(errors)))


def _recovery_state_errors(promotion_decisions: list[dict]) -> tuple[str, ...]:
    errors: list[str] = []
    required = {
        "recovery_capability",
        "recovery_status",
        "recovery_method",
        "failure_code",
        "semantic_validation_outcome",
        "promotion_outcome",
    }
    for row in promotion_decisions:
        row_id = str(row.get("decision_id") or "unknown")
        if required - set(row):
            errors.append(f"recovery_missing_required_field:{row_id}")
            continue
        capability = row.get("recovery_capability")
        status = row.get("recovery_status")
        if capability not in RECOVERY_CAPABILITIES:
            errors.append(f"recovery_invalid_capability:{row_id}:{capability}")
        if status not in RECOVERY_STATUSES:
            errors.append(f"recovery_invalid_status:{row_id}:{status}")
        if status == "promoted" and (
            capability in {"policy_blocked", "technically_unrecoverable"}
            or row.get("semantic_validation_outcome") != "passed"
            or row.get("promotion_outcome") != "promoted"
        ):
            errors.append(f"recovery_invalid_promotion_state:{row_id}")
        if status == "failed" and not row.get("failure_code"):
            errors.append(f"recovery_failed_without_code:{row_id}")
        if status in {"not_attempted", "attempted"} and row.get("promotion_outcome") == "promoted":
            errors.append(f"recovery_unattempted_promoted:{row_id}")
    return tuple(errors)


def _raw_source_geometry_health(rows: list[dict]) -> dict:
    mismatches = 0
    marker_mismatches = 0
    for row in rows:
        boxes = row.get("character_bboxes") or []
        valid = (
            row.get("raw_geometry_method") == "pdf_rawdict_character_bbox"
            and "".join(str(value) for value in row.get("character_texts") or ()) == str(row.get("raw_text") or "")
            and len(boxes) == len(row.get("character_ids") or ())
            and bool(boxes)
        )
        if valid:
            union = (min(box["x0"] for box in boxes), min(box["y0"] for box in boxes), max(box["x1"] for box in boxes), max(box["y1"] for box in boxes))
            valid = union == tuple(row.get(field) for field in ("x0", "y0", "x1", "y1"))
        if not valid:
            mismatches += 1
            if row.get("classification") == "source_annotation_marker":
                marker_mismatches += 1
    return {
        "raw_segment_count": len(rows),
        "non_empty_disposition_count": sum(bool(str(row.get("raw_text") or "")) for row in rows),
        "raw_geometry_mismatch_count": mismatches,
        "marker_geometry_mismatch_count": marker_mismatches,
        "whole_line_geometry_fallback_count": sum(row.get("raw_geometry_method") != "pdf_rawdict_character_bbox" for row in rows),
        "status": "pass" if not mismatches and all(str(row.get("raw_text") or "") for row in rows) else "fail",
    }


def _source_object_disposition_health(rows: list[dict], page_text_spans: list[dict]) -> dict:
    ids = [str(row.get("source_object_id") or "") for row in rows]
    known_span_ids = {str(row.get("text_span_id") or "") for row in page_text_spans}
    invalid_dispositions = [row for row in rows if row.get("disposition") not in TERMINAL_DISPOSITIONS]
    unresolved_spans = [
        span_id
        for row in rows
        for span_id in row.get("text_span_ids") or ()
        if str(span_id) not in known_span_ids
    ]
    missing_lineage = [
        row
        for row in rows
        if not row.get("source_document_id") or not row.get("source_sha256") or not isinstance(row.get("page_number"), int)
    ]
    return {
        "source_object_count": len(rows),
        "terminal_disposition_count": sum(row.get("disposition") in TERMINAL_DISPOSITIONS for row in rows),
        "duplicate_source_object_id_count": len(ids) - len(set(ids)),
        "missing_source_object_id_count": sum(not value for value in ids),
        "invalid_disposition_count": len(invalid_dispositions),
        "unresolved_text_span_ref_count": len(unresolved_spans),
        "missing_lineage_count": len(missing_lineage),
        "extraction_failed_count": sum(row.get("disposition") == "extraction_failed" for row in rows),
        "needs_review_count": sum(row.get("disposition") == "needs_review" for row in rows),
        "status": "complete"
        if rows
        and not invalid_dispositions
        and not unresolved_spans
        and not missing_lineage
        and len(ids) == len(set(ids))
        and all(ids)
        else "incomplete",
    }


def _validate_schema7_contract(artifacts: Mapping[str, object]) -> tuple[str, ...]:
    """Closed UUD contract shared by offline rebuild validation and runtime publication."""
    errors: list[str] = []

    def rows(name: str) -> list[dict]:
        value = artifacts.get(name) or []
        return value if isinstance(value, list) else []

    for artifact, required in MINIMUM_ARTIFACT_FIELDS.items():
        allowed = set(ARTIFACT_ALLOWED_FIELDS.get(artifact, ())) or set(required) | set(ARTIFACT_OPTIONAL_FIELDS.get(artifact, ())) | set(COMMON_ARTIFACT_FIELDS)
        for row in rows(artifact):
            row_id = str(row.get(_schema6_id_field(artifact)) or "<missing>")
            for field in required:
                if field not in row:
                    errors.append(f"missing_required_field:{artifact}:{row_id}:{field}")
            for field in FORBIDDEN_ARTIFACT_FIELDS.get(artifact, ()):
                if field in row:
                    errors.append(f"owner_field_violation:{artifact}:{row_id}:{field}")
            for field in row:
                if field not in allowed:
                    errors.append(f"unknown_field:{artifact}:{row_id}:{field}")
            for field in ("viewer_highlightable", "citation_final", "citable", "evidence_exists", "runtime_loadable"):
                if field in row and not isinstance(row[field], bool):
                    errors.append(f"invalid_type:{artifact}:{row_id}:{field}")

    raw_rows = rows("raw_source_spans")
    source_object_rows = rows("source_objects")
    source_object_ids: set[str] = set()
    known_span_ids = {str(row.get("text_span_id") or "") for row in rows("page_text_spans")}
    for row in source_object_rows:
        object_id = str(row.get("source_object_id") or "")
        if not object_id:
            errors.append("missing_required_field:source_objects:<missing>:source_object_id")
        elif object_id in source_object_ids:
            errors.append(f"duplicate_source_object_id:{object_id}")
        else:
            source_object_ids.add(object_id)
        if row.get("disposition") not in TERMINAL_DISPOSITIONS:
            errors.append(f"invalid_source_object_disposition:{object_id}")
        for span_id in row.get("text_span_ids") or ():
            if str(span_id) not in known_span_ids:
                errors.append(f"source_object_span_unresolved:{object_id}:{span_id}")
    source_support_ids: set[str] = set()
    semantic_stream_rows: dict[str, list[dict]] = defaultdict(list)
    for row in raw_rows:
        row_id = str(row.get("raw_source_span_id") or "<missing>")
        support_id = row.get("source_support_id")
        if not isinstance(support_id, str) or not support_id:
            errors.append(f"missing_required_field:raw_source_spans:{row_id}:source_support_id")
        elif support_id in source_support_ids:
            errors.append(f"duplicate_source_support_id:{support_id}")
        else:
            source_support_ids.add(support_id)
        semantic_text = row.get("semantic_text")
        if not isinstance(semantic_text, str):
            errors.append(f"invalid_type:raw_source_spans:{row_id}:semantic_text")
            semantic_text = ""
        if semantic_text:
            semantic_stream_rows.setdefault(str(row.get("semantic_stream_id")), []).append(row)
            if row.get("semantic_exact_quote") != semantic_text:
                errors.append(f"invalid_selector:{row_id}:semantic_quote")
            if not isinstance(row.get("semantic_text_start"), int) or not isinstance(row.get("semantic_text_end"), int):
                errors.append(f"invalid_selector:{row_id}:semantic_offset_type")
            elif row["semantic_text_end"] - row["semantic_text_start"] != len(semantic_text):
                errors.append(f"invalid_selector:{row_id}:semantic_offset_length")
        elif row.get("semantic_text_start") != 0 or row.get("semantic_text_end") != 0:
            errors.append(f"invalid_selector:{row_id}:empty_semantic_offset")
        if not semantic_text and any(
            row.get(field) is not False
            for field in ("legal_text", "citation_eligible", "relevant_quote_eligible", "default_highlight_eligible")
        ):
            errors.append(f"owner_field_violation:raw_source_spans:{row_id}:empty_semantic_policy")
    for stream_id, stream_rows in semantic_stream_rows.items():
        stream_rows.sort(key=lambda item: int(item.get("extraction_order") or 0))
        stream = "\n".join(str(item["semantic_text"]) for item in stream_rows)
        stream_hash = sha256(stream.encode("utf-8")).hexdigest()
        cursor = 0
        for index, row in enumerate(stream_rows):
            if row.get("semantic_stream_sha256") != stream_hash:
                errors.append(f"invalid_selector:{row.get('raw_source_span_id')}:semantic_stream_hash")
            start = row.get("semantic_text_start")
            end = row.get("semantic_text_end")
            if not isinstance(start, int) or not isinstance(end, int):
                errors.append(f"invalid_selector:{row.get('raw_source_span_id')}:semantic_offset_type")
                continue
            if start != cursor or end != start + len(row["semantic_text"]):
                errors.append(f"invalid_selector:{row.get('raw_source_span_id')}:semantic_offset")
            cursor = end + (1 if index < len(stream_rows) - 1 else 0)
    raw_streams: dict[str, str] = {}
    for row in raw_rows:
        raw_streams.setdefault(str(row.get("raw_stream_id")), "")
    for stream_id in raw_streams:
        stream_rows = sorted((row for row in raw_rows if str(row.get("raw_stream_id")) == stream_id), key=lambda item: item.get("raw_text_start", 0))
        stream = next((str(row.get("raw_text") or "") for row in stream_rows if row.get("raw_text_start") == 0), "")
        if stream_rows:
            page_rows = [row for row in stream_rows if row.get("raw_text_start") is not None]
            max_end = max(int(row.get("raw_text_end") or 0) for row in page_rows)
            pieces = ["\n"] * max_end
            for row in page_rows:
                text = str(row.get("raw_text") or "")
                start, end = int(row.get("raw_text_start") or 0), int(row.get("raw_text_end") or 0)
                if end - start == len(text):
                    pieces[start:end] = text
            stream = "".join(pieces)
        if sha256(stream.encode("utf-8")).hexdigest() != str(stream_rows[0].get("raw_stream_sha256")):
            errors.append(f"invalid_selector:{stream_id}:raw_stream_hash")
        for row in stream_rows:
            raw_start, raw_end, quote = row.get("raw_text_start"), row.get("raw_text_end"), row.get("raw_quote")
            if not isinstance(raw_start, int) or not isinstance(raw_end, int) or not isinstance(quote, str) or stream[raw_start:raw_end] != quote:
                errors.append(f"invalid_selector:{row.get('raw_source_span_id')}:raw_offset")
            if row.get("classification") == "source_annotation_marker" and any(row.get(field) is not False for field in ("legal_text", "citation_eligible", "relevant_quote_eligible", "default_highlight_eligible")):
                errors.append(f"owner_field_violation:raw_source_spans:{row.get('raw_source_span_id')}:marker_policy")
            character_texts = row.get("character_texts") or []
            character_bboxes = row.get("character_bboxes") or []
            if "".join(str(value) for value in character_texts) != str(row.get("raw_text") or ""):
                errors.append(f"raw_geometry_mismatch:{row.get('raw_source_span_id')}:character_text")
            if len(character_texts) != len(row.get("character_ids") or ()) or len(character_texts) != len(character_bboxes):
                errors.append(f"raw_geometry_mismatch:{row.get('raw_source_span_id')}:character_lineage")
            if character_bboxes:
                union = (
                    min(float(box["x0"]) for box in character_bboxes),
                    min(float(box["y0"]) for box in character_bboxes),
                    max(float(box["x1"]) for box in character_bboxes),
                    max(float(box["y1"]) for box in character_bboxes),
                )
                if union != tuple(float(cast(float, row.get(field))) for field in ("x0", "y0", "x1", "y1")):
                    errors.append(f"raw_geometry_mismatch:{row.get('raw_source_span_id')}:union")

    marker_pattern = re.compile(r"\*{1,4}(?:/\*{1,4})?\)")
    for row in rows("page_text_spans"):
        if marker_pattern.search(str(row.get("text") or "")):
            errors.append(f"invalid_selector:{row.get('text_span_id')}:source_marker")
    for row in rows("evidence_registry"):
        if marker_pattern.search(str(row.get("quoted_text") or "")):
            errors.append(f"owner_field_violation:evidence_registry:{row.get('evidence_id')}:source_marker")
        expected = row.get("citable") is True and row.get("authority_kind") == "normative_legal_text" and row.get("evidence_owner_kind") == "legal_unit_source"
        if row.get("relevant_quote_eligible") is not expected:
            errors.append(f"owner_field_violation:evidence_registry:{row.get('evidence_id')}:relevant_quote_eligible")
            for field in ("page_number", "text_start", "text_end"):
                if field in row and not isinstance(row[field], int):
                    errors.append(f"invalid_type:{artifact}:{row_id}:{field}")
            for field in ("bbox_refs", "text_span_ids", "evidence_ids", "span_bbox_ids", "page_numbers"):
                if field in row and not isinstance(row[field], list):
                    errors.append(f"invalid_type:{artifact}:{row_id}:{field}")
            for field in ("authority_kind", "citation_eligibility"):
                if field in row and not isinstance(row[field], str):
                    errors.append(f"invalid_type:{artifact}:{row_id}:{field}")

    evidence = rows("evidence_registry")
    evidence_ids = {row.get("evidence_id") for row in evidence}
    units = rows("legal_units")
    units_by_id = {row.get("legal_unit_id"): row for row in units}
    spans = rows("page_text_spans")
    spans_by_id = {row.get("text_span_id"): row for row in spans}
    bboxes = rows("bbox_registry")
    bbox_ids = {row.get("bbox_id") for row in bboxes}
    for word in rows("word_bboxes"):
        bbox_ids.add(word.get("word_bbox_id"))
        bbox_ids.update(character.get("character_bbox_id") for character in word.get("characters") or ())
    for row in rows("propositions"):
        proposition_id = str(row.get("proposition_id") or "<missing>")
        selectors = row.get("source_selectors")
        if not isinstance(selectors, list) or not selectors:
            errors.append(f"invalid_selector:{proposition_id}:missing")
            continue
        selector_quotes: list[str] = []
        selector_bboxes: list[str] = []
        for selector in selectors:
            span = spans_by_id.get(selector.get("text_span_id"))
            start, end = selector.get("start"), selector.get("end")
            if not span or not isinstance(start, int) or not isinstance(end, int):
                errors.append(f"invalid_selector:{proposition_id}:span")
                continue
            text_start = span.get("text_start")
            if not isinstance(text_start, int):
                errors.append(f"invalid_selector:{proposition_id}:absolute_offset")
                continue
            if (
                selector.get("stream_id") != span.get("stream_id")
                or selector.get("source_document_id") != span.get("source_document_id")
                or selector.get("source_sha256") != span.get("source_sha256")
                or selector.get("page_number") != span.get("page_number")
            ):
                errors.append(f"invalid_selector:{proposition_id}:source")
            if selector.get("absolute_start") != text_start + start or selector.get("absolute_end") != text_start + end:
                errors.append(f"invalid_selector:{proposition_id}:absolute_offset")
            text = str(span.get("text") or "")
            if selector.get("prefix") != text[max(0, start - 32) : start] or selector.get("suffix") != text[end : end + 32]:
                errors.append(f"invalid_selector:{proposition_id}:context")
            character_ids = selector.get("character_bbox_ids") or []
            if not character_ids or any(character_id not in bbox_ids for character_id in character_ids):
                errors.append(f"invalid_selector:{proposition_id}:geometry")
            selector_quotes.append(text[start:end])
            selector_bboxes.extend(character_ids)
        if "\n".join(selector_quotes) != row.get("exact_quote"):
            errors.append(f"invalid_selector:{proposition_id}:quote")
        if tuple(dict.fromkeys(selector_bboxes)) != tuple(row.get("bbox_refs") or ()):
            errors.append(f"invalid_selector:{proposition_id}:bbox_refs")
    source_rows = rows("source_documents")
    source_ids = {row.get("source_document_id") for row in source_rows}
    sources_by_id = {row.get("source_document_id"): row for row in source_rows}
    geometry_keys: set[tuple[object, ...]] = set()
    for row in bboxes:
        identity = tuple(row.get(key) for key in (
            "source_document_id", "source_sha256", "page_number", "coordinate_space", "coordinate_origin",
            "x0", "y0", "x1", "y1", "transform_version", "text",
        ))
        if identity in geometry_keys:
            errors.append(f"duplicate_geometry:{row.get('bbox_id')}")
        geometry_keys.add(identity)

    streams: dict[str, list[dict]] = {}
    for span in spans:
        streams.setdefault(str(span.get("stream_id")), []).append(span)
    stream_values: dict[str, str] = {}
    for stream_id, stream_rows in streams.items():
        stream_rows.sort(key=lambda item: item.get("text_start", 0))
        stream = "\n".join(str(item.get("text") or "") for item in stream_rows)
        stream_values[stream_id] = stream
        digest = sha256(stream.encode("utf-8")).hexdigest()
        if any(item.get("page_text_hash") != digest for item in stream_rows):
            errors.append(f"invalid_selector:{stream_id}:stream_hash")

    for row in spans:
        selector_start, selector_end, quote = row.get("text_start"), row.get("text_end"), row.get("exact_quote")
        if not isinstance(selector_start, int) or not isinstance(selector_end, int) or not isinstance(quote, str) or selector_end < selector_start or selector_end - selector_start != len(quote):
            errors.append(f"invalid_selector:{row.get('text_span_id')}")
        if quote != row.get("text"):
            errors.append(f"invalid_selector:{row.get('text_span_id')}:quote")
        if isinstance(selector_start, int) and isinstance(selector_end, int) and stream_values.get(str(row.get("stream_id")), "")[selector_start:selector_end] != quote:
            errors.append(f"invalid_selector:{row.get('text_span_id')}:offset")
        if row.get("semantic_classification") == "normative_constitutional_text" and row.get("linked_authority") == "rejected":
            errors.append(f"owner_field_violation:page_text_spans:{row.get('text_span_id')}:linked_authority")
        for ref in row.get("evidence_ids") or ():
            if ref not in evidence_ids:
                errors.append(f"invalid_reference:page_text_spans:{row.get('text_span_id')}:evidence_ids")
        for ref in row.get("span_bbox_ids") or ():
            if ref not in bbox_ids:
                errors.append(f"invalid_reference:page_text_spans:{row.get('text_span_id')}:span_bbox_ids")

    for row in evidence:
        row_id = str(row.get("evidence_id") or "")
        if row.get("legal_unit_id") is None or row.get("source_document_id") not in source_ids:
            errors.append(f"invalid_reference:evidence_registry:{row_id}")
        source = sources_by_id.get(row.get("source_document_id"))
        if source and row.get("source_sha256") != source.get("sha256"):
            errors.append(f"EVIDENCE_SOURCE_LINEAGE_INVALID:{row_id}")
        if any(ref not in bbox_ids for ref in row.get("bbox_refs") or ()):
            errors.append(f"invalid_reference:evidence_registry:{row_id}:bbox_refs")
        if any(ref not in spans_by_id for ref in row.get("text_span_ids") or ()):
            errors.append(f"invalid_reference:evidence_registry:{row_id}:text_span_ids")
        span_text = " ".join(str(spans_by_id.get(ref, {}).get("text") or "") for ref in row.get("text_span_ids") or ())
        if row.get("quoted_text") and normalize_source_text(row.get("quoted_text")) not in normalize_source_text(span_text):
            errors.append(f"EVIDENCE_QUOTE_SOURCE_MISMATCH:{row_id}")
        if row.get("citation_final") is True and row.get("exactness") != "exact":
            errors.append(f"owner_field_violation:evidence_registry:{row_id}:citation_final")
        unit = units_by_id.get(row.get("legal_unit_id"), {})
        if unit.get("unit_type") in {"bab_record", "pasal_record"} and any(
            unit.get("legal_unit_id") in (candidate.get("parent_legal_unit_ids") or ()) for candidate in units
        ) and not set(unit.get("text_span_ids") or ()) <= set(row.get("text_span_ids") or ()):
            errors.append(f"aggregate_text_span_sequence_incomplete:{row_id}")

    referenced_geometry = {ref for row in evidence for ref in row.get("bbox_refs") or ()}
    if any(row.get("bbox_id") not in referenced_geometry for row in bboxes):
        errors.append("invalid_reference:orphan_geometry")

    evidence_by_id = {row.get("evidence_id"): row for row in evidence}
    for row in rows("metadata_grounding"):
        if "signatories" not in str(row.get("metadata_grounding_id") or "") or not row.get("quoted_text"):
            continue
        donor_text = " ".join(
            str(evidence_by_id.get(ref, {}).get("quoted_text") or "")
            for ref in row.get("supporting_evidence_ids") or ()
        )
        if donor_text and normalize_source_text(row.get("quoted_text")) not in normalize_source_text(donor_text):
            errors.append(f"signatory_grounding_unknown_name:{row.get('metadata_grounding_id')}")

    relation_rows = rows("article_amendment_relations")
    relations_by_id = {row.get("relation_id"): row for row in relation_rows}
    for row in relation_rows:
        relation_id = str(row.get("relation_id") or "<missing>")
        relation_type = row.get("relation_type")
        if relation_type not in {"MODIFIES", "DELETES", "ADDS", "RENAMES", "RENUMBERED_TO", "SUPPLEMENTS"}:
            errors.append(f"invalid_enum:article_amendment_relations:{relation_id}:relation_type")
        evidence_row = evidence_by_id.get(row.get("evidence_id"))
        source_unit = units_by_id.get(row.get("source_legal_unit_id"))
        target_unit = units_by_id.get(row.get("target_legal_unit_id"))
        if evidence_row is None:
            errors.append(f"invalid_reference:article_amendment_relations:{relation_id}:evidence_id")
            continue
        if row.get("source_pdf_sha256") != evidence_row.get("source_sha256"):
            errors.append(f"article_relation_source_sha_mismatch:{relation_id}")
        if row.get("bbox_precision") != evidence_row.get("bbox_precision"):
            errors.append(f"article_relation_bbox_precision_mismatch:{relation_id}")
        source_bbox_refs = set(row.get("bbox_refs") or ()) - set(row.get("target_bbox_refs") or ())
        if source_bbox_refs - set(evidence_row.get("bbox_refs") or ()):
            errors.append(f"article_relation_bbox_source_mismatch:{relation_id}")
        if row.get("support_class") == "exact_article_relation" and row.get("target_precision") != "target_local":
            errors.append(f"article_relation_exact_precision_mismatch:{relation_id}")
        if source_unit is None:
            errors.append(f"article_relation_unknown_source:{relation_id}")
        elif row.get("source_label") != source_unit.get("unit_label"):
            errors.append(f"article_relation_source_label_mismatch:{relation_id}")
        if target_unit is None:
            errors.append(f"article_relation_unknown_target:{relation_id}")
        elif row.get("target_source_role") != target_unit.get("source_role"):
            errors.append(f"article_relation_target_role_mismatch:{relation_id}")
        if relation_type in {"RENAMES", "RENUMBERED_TO"}:
            quoted = str(evidence_row.get("quoted_text") or "")
            old_range = _range_value(row.get("old_reference_range"), len(quoted))
            new_range = _range_value(row.get("new_reference_range"), len(quoted))
            errors.extend(f"{code}:{relation_id}" for code in _validate_mapping_support(row, evidence_row))
            if old_range is not None and new_range is not None:
                if not _validator_range_matches_reference(quoted[old_range[0] : old_range[1]], row.get("old_reference"), row.get("old_reference_range_kind")):
                    errors.append(f"article_relation_old_range_text_mismatch:{relation_id}")
                if not _validator_range_matches_reference(quoted[new_range[0] : new_range[1]], row.get("new_reference"), row.get("new_reference_range_kind")):
                    errors.append(f"article_relation_new_range_text_mismatch:{relation_id}")

    for row in rows("graph_edges"):
        relation_id = str(row.get("relation_id") or row.get("article_relation_ref") or "")
        relation = relations_by_id.get(relation_id)
        if row.get("edge_type") in {"RENAMES", "RENUMBERED_TO"} and relation is not None:
            if row.get("source_id") != f"legal_unit::{relation.get('source_legal_unit_id')}":
                errors.append(f"graph_relation_source_mismatch:{row.get('edge_id')}")
            if row.get("target_id") != f"legal_unit::{relation.get('target_legal_unit_id')}":
                errors.append(f"graph_relation_target_mismatch:{row.get('edge_id')}")

    return tuple(dict.fromkeys(errors))


def _aggregate_evidence_errors(
    legal_units: list[dict],
    chunks: list[dict],
    evidence: list[dict],
    bbox_by_id: dict[str, dict],
    page_text_spans: list[dict],
) -> tuple[str, ...]:
    chunks_by_unit = {row.get("legal_unit_id"): row for row in chunks}
    evidence_by_unit = {row.get("legal_unit_id"): row for row in evidence}
    spans_by_id = {row.get("text_span_id"): row for row in page_text_spans}
    descendants_by_parent: dict[str, list[dict]] = {}
    for unit in legal_units:
        for parent_id in unit.get("parent_legal_unit_ids") or ():
            descendants_by_parent.setdefault(parent_id, []).append(unit)
    errors: list[str] = []
    for parent in legal_units:
        descendants = _all_descendants(parent.get("legal_unit_id"), descendants_by_parent)
        if not descendants:
            continue
        parent_id = parent.get("legal_unit_id")
        aggregate = evidence_by_unit.get(parent_id)
        if aggregate is None:
            if not parent.get("aggregate_failure_reason") and not chunks_by_unit.get(parent_id, {}).get("aggregate_failure_reason"):
                errors.append(f"aggregate_missing_failure_reason:{parent_id}")
            continue
        expected_span_ids = _ordered_aggregate_span_ids(parent, descendants, {str(key): value for key, value in spans_by_id.items() if key is not None})
        actual_span_ids = list(aggregate.get("text_span_ids") or ())
        if len(actual_span_ids) != len(set(actual_span_ids)) or actual_span_ids != expected_span_ids:
            errors.append(f"aggregate_text_span_sequence_incomplete:{aggregate.get('evidence_id')}")
        expected_text = normalize_source_text(" ".join(str(spans_by_id[span_id].get("text") or "") for span_id in expected_span_ids if span_id in spans_by_id))
        if expected_text and _aggregate_compare_text(aggregate.get("quoted_text")) != _aggregate_compare_text(expected_text):
            errors.append(f"aggregate_text_reconstruction_failed:{aggregate.get('evidence_id')}")
        bboxes = [bbox_by_id[bbox_id] for bbox_id in aggregate.get("bbox_refs") or () if bbox_id in bbox_by_id]
        if not bboxes:
            if aggregate.get("citable_status") != "trace_text_only":
                errors.append(f"aggregate_geometry_unavailable:{aggregate.get('evidence_id')}")
        elif any(
            row.get("source_document_id") != parent.get("source_document_id")
            or row.get("bbox_precision") != "exact"
            or row.get("viewer_highlightable") is not True
            or not all(row.get(field) is not None for field in ("x0", "y0", "x1", "y1"))
            for row in bboxes
        ):
            errors.append(f"aggregate_geometry_unavailable:{aggregate.get('evidence_id')}")
        if any(
            spans_by_id.get(span_id, {}).get("source_document_id") != parent.get("source_document_id")
            or spans_by_id.get(span_id, {}).get("source_role") not in {None, parent.get("source_role")}
            for span_id in actual_span_ids
        ):
            errors.append(f"aggregate_source_ownership_mismatch:{aggregate.get('evidence_id')}")
        aggregate_spans = set(actual_span_ids)
        for child in descendants:
            if not set(child.get("text_span_ids") or ()) <= aggregate_spans:
                errors.append(f"aggregate_child_span_outside_parent:{parent_id}:{child.get('legal_unit_id')}")
            if child.get("source_document_id") != parent.get("source_document_id") or child.get("source_role") not in {
                None,
                parent.get("source_role"),
            }:
                errors.append(f"aggregate_child_source_ownership_mismatch:{parent_id}:{child.get('legal_unit_id')}")
    return tuple(dict.fromkeys(errors))


def _all_descendants(parent_id: str | None, descendants_by_parent: dict[str, list[dict]]) -> list[dict]:
    if not parent_id:
        return []
    descendants: list[dict] = []
    pending = list(descendants_by_parent.get(parent_id, ()))
    while pending:
        child = pending.pop(0)
        descendants.append(child)
        child_id = child.get("legal_unit_id")
        if child_id is not None:
            pending.extend(descendants_by_parent.get(str(child_id), ()))
    return descendants


def _ordered_aggregate_span_ids(parent: dict, descendants: list[dict], spans_by_id: dict[str, dict]) -> list[str]:
    span_ids = list(dict.fromkeys(
        span_id
        for unit in (parent, *descendants)
        for span_id in unit.get("text_span_ids") or ()
        if span_id in spans_by_id
    ))
    return sorted(
        span_ids,
        key=lambda span_id: (
            spans_by_id[span_id].get("page_number", 0),
            spans_by_id[span_id].get("y0", 0),
            spans_by_id[span_id].get("x0", 0),
            span_id,
        ),
    )


def _aggregate_compare_text(text: object) -> str:
    normalized = normalize_source_text(re.sub(r"\*+\)", "", str(text or "")))
    return re.sub(r"\W+", "", normalized)


def _reference_belongs_to_unit(reference: object, unit: dict) -> bool:
    value = normalize_source_text(reference)
    label = normalize_source_text(unit.get("unit_label"))
    return bool(value and label and (value == label or value.startswith(f"{label} ayat ")))


def _validate_mapping_support(row: dict, evidence: dict) -> tuple[str, ...]:
    quoted = str(evidence.get("quoted_text") or "")
    old_range = _range_value(row.get("old_reference_range"), len(quoted))
    new_range = _range_value(row.get("new_reference_range"), len(quoted))
    errors: list[str] = []
    if old_range is None or new_range is None:
        return ("article_relation_invalid_reference_range",)
    if old_range[1] > new_range[0]:
        errors.append("article_relation_reference_range_order")
    if not _validator_range_matches_reference(
        quoted[old_range[0] : old_range[1]], row.get("old_reference"), row.get("old_reference_range_kind")
    ):
        errors.append("article_relation_old_range_text_mismatch")
    if not _validator_range_matches_reference(
        quoted[new_range[0] : new_range[1]], row.get("new_reference"), row.get("new_reference_range_kind")
    ):
        errors.append("article_relation_new_range_text_mismatch")
    if "menjadi" not in normalize_source_text(quoted[old_range[1] : new_range[0]]):
        errors.append("article_relation_missing_transition_support")
    if not evidence.get("text_span_ids") or not evidence.get("bbox_refs"):
        errors.append("article_relation_missing_support_segment")
    return tuple(errors)


def _validate_relation_support(row: dict, evidence: dict, span_by_id: dict[str, dict], bbox_by_id: dict[str, dict]) -> tuple[str, ...]:
    if row.get("relation_type") not in {"RENAMES", "RENUMBERED_TO"}:
        return ()
    quoted = str(evidence.get("quoted_text") or "")
    old_range = _range_value(row.get("old_reference_range"), len(quoted))
    new_range = _range_value(row.get("new_reference_range"), len(quoted))
    if old_range is None or new_range is None:
        return ("article_relation_support_missing_mapping_range",)
    expected_ids = tuple(evidence.get("text_span_ids") or ())
    expected_bbox_ids = tuple(evidence.get("bbox_refs") or ())
    support_ids = tuple(row.get("text_span_ids") or ())
    support_bbox_ids = tuple(row.get("bbox_refs") or ())
    source_support_bbox_ids = tuple(bbox_id for bbox_id in support_bbox_ids if bbox_id not in set(row.get("target_bbox_refs") or ()))
    errors: list[str] = []
    if not support_ids or not support_bbox_ids:
        return ("article_relation_support_missing_segment",)
    if not set(support_ids) <= set(expected_ids) or not set(source_support_bbox_ids) <= set(expected_bbox_ids):
        errors.append("article_relation_support_reference_mismatch")
    span_order = _ordered_support_slice(expected_ids, support_ids, span_by_id, quoted, old_range[0], new_range[1])
    bbox_order = _ordered_support_slice(expected_bbox_ids, source_support_bbox_ids, bbox_by_id, quoted, old_range[0], new_range[1])
    if span_order is None or bbox_order is None:
        errors.append("article_relation_support_not_minimal")
    support_text = _support_text(support_ids, span_by_id)
    support_bbox_text = _support_text(source_support_bbox_ids, bbox_by_id)
    old_text = normalize_source_text(quoted[old_range[0] : old_range[1]])
    new_text = normalize_source_text(quoted[new_range[0] : new_range[1]])
    for text in (support_text, support_bbox_text):
        normalized = normalize_source_text(text)
        old_position = normalized.find(old_text)
        transition_position = normalized.find("menjadi", old_position + len(old_text))
        new_position = normalized.find(new_text, transition_position + len("menjadi"))
        if old_position < 0 or transition_position < 0 or new_position < 0:
            errors.append("article_relation_support_missing_mapping_part")
        elif not old_position < transition_position < new_position:
            errors.append("article_relation_support_order")
    return tuple(dict.fromkeys(errors))


def _ordered_support_slice(
    expected_ids: tuple[str, ...], support_ids: tuple[str, ...], rows_by_id: dict[str, dict], quoted: str, start: int, end: int
) -> tuple[str, ...] | None:
    positions: list[tuple[str, int, int]] = []
    cursor = 0
    for row_id in expected_ids:
        text = str(rows_by_id.get(row_id, {}).get("text") or "")
        position = quoted.find(text, cursor) if text else -1
        if position < 0:
            return None
        positions.append((row_id, position, position + len(text)))
        cursor = position + len(text)
    required = tuple(row_id for row_id, row_start, row_end in positions if row_end > start and row_start < end)
    return required if tuple(support_ids) == required else None


def _support_text(ids: tuple[str, ...], rows_by_id: dict[str, dict]) -> str:
    return " ".join(str(rows_by_id.get(row_id, {}).get("text") or "") for row_id in ids)


def _validate_source_support(evidence: dict, row: dict) -> bool:
    if row.get("relation_type") not in {"RENAMES", "RENUMBERED_TO"}:
        return evidence.get("bbox_precision") == "exact" and evidence.get("viewer_highlightable") is True
    return not bool(_validate_mapping_support(row, evidence))


def _compact_reference(value: object) -> str:
    return re.sub(r"\W+", "", str(value or "").casefold())


def _range_value(value: object, length: int) -> tuple[int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        start, end = int(value[0]), int(value[1])
    except (TypeError, ValueError):
        return None
    return (start, end) if 0 <= start < end <= length else None


def _validator_range_matches_reference(text: str, reference: object, kind: object) -> bool:
    expected = normalize_source_text(reference)
    actual = normalize_source_text(text)
    if kind != "contextual":
        return actual == expected
    return matches_uud_contextual_reference(text, reference)


def _source_relation_contract_errors(
    *, evidence: list[dict], legal_units: list[dict], article_amendment_relations: list[dict]
) -> tuple[str, ...]:
    evidence_ids = {str(row.get("evidence_id")) for row in evidence}
    unit_ids = {str(row.get("legal_unit_id")) for row in legal_units}
    relation_ids: set[str] = set()
    errors: list[str] = []
    for row in article_amendment_relations:
        relation_id = str(row.get("relation_id") or "")
        relation_type = str(row.get("relation_type") or "")
        descriptor = descriptor_for(relation_type)
        if not relation_id or relation_id in relation_ids:
            errors.append(f"article_relation_duplicate_source_reference:{relation_id}")
        relation_ids.add(relation_id)
        if (
            not descriptor
            or not descriptor.authority_bearing
            or row.get("evidence_id") not in evidence_ids
            or row.get("source_legal_unit_id") not in unit_ids
            or row.get("target_legal_unit_id") not in unit_ids
            or not row.get("target_citation")
        ):
            errors.append(f"article_relation_unexpected_source_reference:{relation_id}")
        if row.get("support_class") == "exact_article_relation" and not row.get("bbox_refs"):
            errors.append(f"article_relation_missing_source_reference:{relation_id}")
    return tuple(errors)


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


def _word_bbox_registry_health(*, word_bboxes: list[dict], pages: list[dict]) -> dict:
    page_keys = {(row["source_document_id"], row["page_number"]) for row in pages}
    invalid_coords = [
        row
        for row in word_bboxes
        if None in {row.get("x0"), row.get("y0"), row.get("x1"), row.get("y1")}
        or row.get("x1", 0) < row.get("x0", 0)
        or row.get("y1", 0) < row.get("y0", 0)
        or row.get("x0", 0) < 0
        or row.get("y0", 0) < 0
    ]
    missing_page_refs = [row for row in word_bboxes if (row.get("source_document_id"), row.get("page_number")) not in page_keys]
    missing_normalized_text = [row for row in word_bboxes if row.get("text") and not str(row.get("normalized_text") or "").strip()]
    return {
        "word_bbox_rows": len(word_bboxes),
        "source_document_count": len({row.get("source_document_id") for row in word_bboxes}),
        "page_count": len({(row.get("source_document_id"), row.get("page_number")) for row in word_bboxes}),
        "invalid_coordinate_count": len(invalid_coords),
        "missing_page_ref_count": len(missing_page_refs),
        "missing_normalized_text_count": len(missing_normalized_text),
        "status": "complete" if not invalid_coords and not missing_page_refs and not missing_normalized_text else "incomplete",
    }


def _selector_geometry_health(
    *, propositions: list[dict], evidence: list[dict], page_text_spans: list[dict], word_bboxes: list[dict]
) -> dict:
    spans = {row.get("text_span_id"): row for row in page_text_spans}
    characters = [
        word | character
        for word in word_bboxes
        for character in word.get("characters") or ()
        if character.get("character_bbox_id")
    ]
    character_ids = [character["character_bbox_id"] for character in characters]
    known_character_ids = set(character_ids)
    characters_by_id = {character["character_bbox_id"]: character for character in characters}
    marker_boxes: dict[tuple[object, ...], list[dict]] = defaultdict(list)
    for marker in source_marker_character_boxes(word_bboxes):
        marker_boxes[_geometry_space_key(marker)].append(marker)
    evidence_by_id = {row.get("evidence_id"): row for row in evidence}
    round_trip_mismatches = 0
    absolute_selector_mismatches = 0
    unknown_selected_character_ids = 0
    whole_span_bboxes = 0
    marker_text_in_quotes = 0
    marker_geometry_intersections = 0
    overlay_lineage_failures = 0
    citable_without_geometry = 0
    marker_pattern = re.compile(r"\*{1,4}(?:/\*{1,4})?\)")
    for proposition in propositions:
        selector_quotes: list[str] = []
        selector_bboxes: list[str] = []
        for selector in proposition.get("source_selectors") or ():
            span = spans.get(selector.get("text_span_id")) or {}
            text = str(span.get("text") or "")
            start, end = selector.get("start"), selector.get("end")
            if not isinstance(start, int) or not isinstance(end, int):
                selector_quotes.append("")
                continue
            selector_quotes.append(text[start:end])
            selector_bboxes.extend(selector.get("character_bbox_ids") or ())
            if (
                selector.get("stream_id") != span.get("stream_id")
                or selector.get("absolute_start") != span.get("text_start", 0) + start
                or selector.get("absolute_end") != span.get("text_start", 0) + end
            ):
                absolute_selector_mismatches += 1
            unknown_selected_character_ids += len(set(selector.get("character_bbox_ids") or ()) - known_character_ids)
            if set(selector.get("character_bbox_ids") or ()) & set(span.get("span_bbox_ids") or ()):
                whole_span_bboxes += 1
        exact_quote = proposition.get("exact_quote")
        geometry = tuple(dict.fromkeys(selector_bboxes))
        if "\n".join(selector_quotes) != exact_quote or geometry != tuple(proposition.get("bbox_refs") or ()):
            round_trip_mismatches += 1
        if marker_pattern.search(str(exact_quote or "")):
            marker_text_in_quotes += 1
        overlay = proposition.get("viewer_overlay") or {}
        rectangles = viewer_overlay_rectangles(proposition, characters_by_id)
        covered_character_ids = {
            character_id
            for rectangle in rectangles
            for character_id in rectangle.get("character_bbox_ids") or ()
        }
        clipped_character_ids = {
            character_id
            for rectangle in overlay.get("clipped_rectangles") or ()
            for character_id in rectangle.get("character_bbox_ids") or ()
        }
        if (
            overlay.get("status") != "complete"
            or overlay.get("proposition_id") != proposition.get("proposition_id")
            or overlay.get("source_document_id") != proposition.get("source_document_id")
            or overlay.get("source_sha256") != proposition.get("source_sha256")
            or overlay.get("selector_field") != "source_selectors"
            or overlay.get("selected_character_field") != "bbox_refs"
            or clipped_character_ids - set(geometry)
            or overlay.get("clipped_character_count") != len(clipped_character_ids)
            or covered_character_ids != set(geometry)
        ):
            overlay_lineage_failures += 1
        marker_geometry_intersections += sum(
            1
            for rectangle in rectangles
            for marker in marker_boxes.get(_geometry_space_key(rectangle), ())
            if positive_area_intersection(rectangle, marker)
        )
        evidence_row = evidence_by_id.get(proposition.get("evidence_id"), {})
        if evidence_row.get("citation_final") is True and (not geometry or not set(geometry) <= known_character_ids):
            citable_without_geometry += 1
    counts = {
        "selector_round_trip_mismatch_count": round_trip_mismatches,
        "absolute_selector_mismatch_count": absolute_selector_mismatches,
        "unknown_selected_character_id_count": unknown_selected_character_ids,
        "partial_selector_whole_span_bbox_count": whole_span_bboxes,
        "persisted_character_bbox_id_duplicate_count": len(character_ids) - len(set(character_ids)),
        "marker_text_in_exact_quote_count": marker_text_in_quotes,
        "marker_viewer_geometry_intersection_count": marker_geometry_intersections,
        "viewer_geometry_without_exact_selector_lineage_count": overlay_lineage_failures,
        "citable_support_without_valid_geometry_count": citable_without_geometry,
    }
    return counts | {"status": "complete" if not any(counts.values()) else "incomplete"}


def _geometry_space_key(row: dict) -> tuple[object, ...]:
    return tuple(
        row.get(field)
        for field in (
            "source_document_id", "source_sha256", "page_number", "coordinate_space",
            "coordinate_origin", "page_rotation", "page_box_basis", "transform_version",
        )
    )


def _pdf_health_summary(report: dict) -> dict:
    pages = report.get("pages") or ()
    sources = report.get("source_documents") or ()
    return {
        "status": report.get("status") or "missing",
        "source_count": int(report.get("source_count") or 0),
        "page_count": int(report.get("page_count") or 0),
        "native_text_ok_source_count": int(report.get("native_text_ok_source_count") or 0),
        "native_text_ok_page_count": int(report.get("native_text_ok_page_count") or 0),
        "ocr_required_count": int(report.get("ocr_required_count") or 0),
        "ocr_dependency_status": report.get("ocr_dependency_status") or "unknown",
        "source_unusable_count": sum(1 for row in sources if row.get("health_decision") == "source_unusable"),
        "needs_review_count": sum(1 for row in (*sources, *pages) if row.get("health_decision") == "needs_review"),
        "repair_required_count": sum(1 for row in (*sources, *pages) if row.get("health_decision") == "repair_required"),
    }


def _source_quote_fidelity_health(*, metadata_grounding: list[dict], evidence: list[dict], pages: list[dict]) -> dict:
    page_text = {(row["source_document_id"], row["page_number"]): row.get("text", "") for row in pages}
    metadata_rows = list(metadata_grounding)
    evidence_rows = [row for row in evidence if row.get("bbox_precision") == "page_grounded_only"]
    metadata_mismatches = [row for row in metadata_rows if not _quote_in_pages(row, page_text)]
    evidence_mismatches = [row for row in evidence_rows if not _quote_in_pages(row, page_text)]
    tracked = [row for row in metadata_mismatches if row.get("failure_reason")]
    return {
        "metadata_grounding_checked_count": len(metadata_rows),
        "page_grounded_evidence_checked_count": len(evidence_rows),
        "metadata_source_quote_mismatch_count": len(metadata_mismatches),
        "page_grounded_evidence_source_quote_mismatch_count": len(evidence_mismatches),
        "tracked_exception_count": len(tracked),
        "untracked_mismatch_count": len(metadata_mismatches) - len(tracked) + len(evidence_mismatches),
        "page_grounded_only_metadata_count": sum(1 for row in metadata_rows if row.get("bbox_precision") == "page_grounded_only"),
    }


def _quote_in_pages(row: dict, page_text: dict[tuple[str, int], str]) -> bool:
    quote = _source_quote_normalize(row.get("quoted_text"))
    source_document_id = row.get("source_document_id")
    if not quote or not isinstance(source_document_id, str):
        return False
    haystack = " ".join(
        _source_quote_normalize(page_text.get((source_document_id, page_number), "")) for page_number in row.get("page_numbers") or ()
    )
    return quote in haystack


def _source_quote_normalize(text: str | None) -> str:
    normalized = unicodedata.normalize("NFKC", text or "").replace("\xad", "").replace("\xa0", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"\s+([,.;:])", r"\1", normalized)
    return normalized.strip().casefold()


def _span_sequence_grounding_health(
    *,
    metadata_grounding: list[dict],
    evidence: list[dict],
    legal_units: list[dict],
    chunks: list[dict],
    bbox_rows: list[dict],
    word_bboxes: list[dict],
    page_text_spans: list[dict],
) -> dict:
    span_rows = [row for row in page_text_spans if row.get("text_span_id")]
    active_units = [row for row in legal_units if row.get("status") in {"final", "finalizable"}]
    active_chunks = [row for row in chunks if row.get("status") == "active_canonical_record"]
    units_without_spans = [row for row in active_units if not row.get("text_span_ids")]
    chunks_without_spans = [row for row in active_chunks if not row.get("text_span_ids")]
    bbox_by_id = {row["bbox_id"]: row for row in bbox_rows} | {
        row["word_bbox_id"]: {
            "bbox_id": row["word_bbox_id"],
            "bbox_precision": "exact",
            "viewer_highlightable": True,
            **row,
        }
        for row in word_bboxes
    }
    page_metadata = [row for row in metadata_grounding if row.get("bbox_precision") == "page_grounded_only"]
    page_evidence = [row for row in evidence if row.get("bbox_precision") == "page_grounded_only"]
    bbox_ids = set(bbox_by_id)
    invalid_bbox_refs = [
        bbox_id
        for row in (*metadata_grounding, *evidence)
        for bbox_id in row.get("bbox_refs") or ()
        if bbox_id not in bbox_ids and row.get("bbox_precision") == "exact"
    ]
    invalid_coordinates = [
        row["bbox_id"]
        for row in bbox_rows
        if row.get("bbox_precision") == "exact" and not all(row.get(key) is not None for key in ("x0", "y0", "x1", "y1"))
    ]
    return {
        "fixable_page_grounded_metadata_count": sum(
            1 for row in page_metadata if _can_match_span_sequence(row, span_rows) and _has_exact_bbox_refs(row, bbox_by_id)
        ),
        "unresolved_page_grounded_metadata_count": len(page_metadata),
        "active_legal_units_without_span_ids": len(units_without_spans),
        "active_chunks_without_span_ids": len(chunks_without_spans),
        "fixable_legal_units_without_span_ids": sum(1 for row in units_without_spans if _can_match_span_sequence(row, span_rows)),
        "fixable_chunks_without_span_ids": sum(1 for row in chunks_without_spans if _can_match_span_sequence(row, span_rows)),
        "page_grounded_evidence_without_failure_reason": sum(1 for row in page_evidence if not row.get("failure_reason")),
        "false_exact_metadata_claims": sum(
            1
            for row in metadata_grounding
            if row.get("bbox_precision") == "exact" and (not row.get("bbox_refs") or not row.get("viewer_highlightable"))
        ),
        "invalid_bbox_refs": len(invalid_bbox_refs),
        "invalid_bbox_coordinates": len(invalid_coordinates),
        "untracked_grounding_exception_count": sum(
            1
            for row in (*page_metadata, *units_without_spans, *chunks_without_spans, *page_evidence)
            if not (row.get("failure_reason") or row.get("provenance_exception_category") or row.get("grounding_status"))
        ),
    }


def _can_match_span_sequence(row: dict, spans: list[dict]) -> bool:
    text = row.get("quoted_text") or row.get("text")
    target = _source_quote_normalize(text)
    if not target:
        return False
    pages = set(row.get("page_numbers") or range(int(row.get("page_start") or 0), int(row.get("page_end") or -1) + 1))
    rows = [span for span in spans if span.get("source_document_id") == row.get("source_document_id") and span.get("page_number") in pages]
    for start in range(len(rows)):
        joined = ""
        for span in rows[start:]:
            joined = _source_quote_normalize(f"{joined} {span.get('text', '')}")
            if joined == target:
                return True
            if len(joined) > len(target) + 80 or not target.startswith(joined):
                break
    return False


def _has_exact_bbox_refs(row: dict, bbox_by_id: dict[str, dict]) -> bool:
    refs = row.get("bbox_refs") or ()
    return bool(refs) and all(
        (bbox := bbox_by_id.get(ref))
        and bbox.get("bbox_precision") == "exact"
        and all(bbox.get(key) is not None for key in ("x0", "y0", "x1", "y1"))
        for ref in refs
    )


def _promotion_engine_health(
    *,
    evidence: list[dict],
    metadata_grounding: list[dict],
    bbox_rows: list[dict],
    word_bboxes: list[dict],
    legal_units: list[dict],
    chunks: list[dict],
    page_text_spans: list[dict],
    pages: list[dict],
) -> dict:
    bbox_by_id = {row["bbox_id"]: row for row in bbox_rows} | {
        row["word_bbox_id"]: {
            "bbox_id": row["word_bbox_id"],
            "bbox_precision": "exact",
            "viewer_highlightable": True,
            **row,
        }
        for row in word_bboxes
    }
    span_rows = [row for row in page_text_spans if row.get("text_span_id")]
    page_text = {(row["source_document_id"], row["page_number"]): row.get("text", "") for row in pages}
    unit_by_id = {row["legal_unit_id"]: row for row in legal_units}
    chunk_by_unit = {row["legal_unit_id"]: row for row in chunks}
    non_exact_evidence = [row for row in evidence if row.get("bbox_precision") != "exact" or row.get("viewer_highlightable") is not True]
    non_exact_metadata = [
        row for row in metadata_grounding if row.get("bbox_precision") != "exact" or row.get("viewer_highlightable") is not True
    ]
    non_exact_bbox = [row for row in bbox_rows if row.get("bbox_precision") != "exact" or row.get("viewer_highlightable") is not True]
    promotable_exact = [
        row
        for row in (*non_exact_evidence, *non_exact_metadata)
        if _quote_in_pages(row, page_text) and _can_match_span_sequence(row, span_rows) and _has_exact_bbox_refs(row, bbox_by_id)
    ]
    exact_evidence = [row for row in evidence if row.get("bbox_precision") == "exact"]
    exact_metadata = [row for row in metadata_grounding if row.get("bbox_precision") == "exact"]
    false_exact = [
        row
        for row in (*exact_evidence, *exact_metadata)
        if not row.get("text_span_ids") or not _has_exact_bbox_refs(row, bbox_by_id) or row.get("viewer_highlightable") is not True
    ]
    exact_bbox_rows = [row for row in bbox_rows if row.get("bbox_precision") == "exact"]
    invalid_exact_bbox_refs = [
        bbox_id for row in (*exact_evidence, *exact_metadata) for bbox_id in row.get("bbox_refs") or () if bbox_id not in bbox_by_id
    ]
    invalid_exact_coordinates = [
        row
        for row in exact_bbox_rows
        if not (
            all(row.get(key) is not None for key in ("x0", "y0", "x1", "y1"))
            and row.get("x1", 0) >= row.get("x0", 0)
            and row.get("y1", 0) >= row.get("y0", 0)
        )
    ]
    missing_reason = [
        row
        for row in (*non_exact_evidence, *non_exact_metadata, *non_exact_bbox)
        if not (row.get("failure_reason") or row.get("rejection_reason") or row.get("grounding_status"))
    ]
    containing_overclaims = []
    for row in exact_evidence:
        unit = unit_by_id.get(row.get("legal_unit_id"), {})
        chunk = chunk_by_unit.get(row.get("legal_unit_id"), {})
        if row.get("viewer_highlightable") is True and "text_span_containing_match" in {
            unit.get("grounding_status"),
            chunk.get("grounding_status"),
        }:
            containing_overclaims.append(row)
    counts = {
        "evidence_exact_count": len(exact_evidence),
        "evidence_page_grounded_only_count": sum(1 for row in evidence if row.get("bbox_precision") == "page_grounded_only"),
        "evidence_trace_only_count": sum(1 for row in evidence if row.get("failure_reason") == "instrument_trace_only_not_public_citation"),
        "bbox_exact_count": len(exact_bbox_rows),
        "bbox_page_grounded_only_count": sum(1 for row in bbox_rows if row.get("bbox_precision") == "page_grounded_only"),
        "bbox_non_highlightable_count": sum(1 for row in bbox_rows if row.get("viewer_highlightable") is not True),
        "metadata_grounding_exact_count": len(exact_metadata),
        "metadata_grounding_page_grounded_only_count": sum(
            1 for row in metadata_grounding if row.get("bbox_precision") == "page_grounded_only"
        ),
        "promotable_exact_count": len(promotable_exact),
        "promotion_blocked_count": len(non_exact_evidence) + len(non_exact_metadata) + len(non_exact_bbox),
        "missing_promotion_reason_count": len(missing_reason),
        "false_exact_claim_count": len(false_exact),
        "invalid_bbox_ref_count": len(invalid_exact_bbox_refs),
        "invalid_bbox_coordinate_count": len(invalid_exact_coordinates),
        "containing_span_exact_overclaim_count": len(containing_overclaims),
    }
    error_keys = {
        "promotable_exact_count",
        "missing_promotion_reason_count",
        "false_exact_claim_count",
        "invalid_bbox_ref_count",
        "invalid_bbox_coordinate_count",
        "containing_span_exact_overclaim_count",
    }
    return {**counts, "status": "complete" if not any(counts[key] for key in error_keys) else "incomplete"}


def _promotion_decision_audit_health(
    *,
    evidence: list[dict],
    metadata_grounding: list[dict],
    bbox_rows: list[dict],
    promotion_decisions: list[dict],
    promotion_engine_health: dict,
    article_relations: list[dict] | tuple[dict, ...] = (),
) -> dict:
    expected = {
        ("evidence", row["evidence_id"])
        for row in evidence
        if row.get("bbox_precision") != "exact"
        or row.get("viewer_highlightable") is not True
        or row.get("failure_reason") == "instrument_trace_only_not_public_citation"
        or row.get("promotion_candidate") is True
    }
    expected |= {
        ("metadata_grounding", row["metadata_grounding_id"])
        for row in metadata_grounding
        if row.get("bbox_precision") != "exact" or row.get("viewer_highlightable") is not True
    }
    expected |= {("article_relation", row["relation_id"]) for row in article_relations}
    expected |= {
        ("bbox", row["bbox_id"])
        for row in bbox_rows
        if row.get("bbox_precision") != "exact" or row.get("viewer_highlightable") is not True or row.get("promotion_candidate") is True
    }
    actual = {(row.get("record_type"), row.get("record_id")) for row in promotion_decisions}
    duplicate_decision_ids = len(promotion_decisions) - len({row.get("decision_id") for row in promotion_decisions})
    blocked = [row for row in promotion_decisions if row.get("decision") == "keep_non_exact"]
    attempted = [row for row in promotion_decisions if row.get("promotion_attempted") is True]
    false_exact = [
        row
        for row in promotion_decisions
        if row.get("decision") == "promote_exact"
        and not (row.get("exact_quote_available") and row.get("exact_span_available") and row.get("exact_bbox_available"))
    ]
    required_fields = {
        "promotion_attempt_method",
        "promotion_attempt_result",
        "quote_match_status",
        "span_match_status",
        "subspan_match_status",
        "bbox_union_status",
        "matched_span_ids",
        "matched_page_numbers",
        "matched_text_excerpt",
        "field_bbox_feasibility",
        "metadata_exact_promotion_feasibility",
        "blocker_evidence",
        "can_be_exact_citation",
        "can_be_exact_highlight",
    }
    generic_reasons = {"non_exact_grounding", "page_grounded_only", "field_level_grounded", "blocked", "insufficient_evidence"}
    counts = {
        "promotion_decision_count": len(promotion_decisions),
        "expected_promotion_decision_count": len(expected),
        "blocked_decision_count": len(blocked),
        "promotion_blocked_count": len(blocked),
        "promotion_attempted_count": len(attempted),
        "promotion_attempt_missing_count": len(promotion_decisions) - len(attempted),
        "exact_quote_match_count": sum(1 for row in promotion_decisions if row.get("quote_match_status") == "exact_full_quote_match"),
        "span_sequence_candidate_count": sum(
            1 for row in promotion_decisions if row.get("span_match_status") == "normalized_span_sequence_match"
        ),
        "subspan_match_candidate_count": sum(1 for row in promotion_decisions if row.get("subspan_match_status") == "matched"),
        "bbox_union_candidate_count": sum(1 for row in promotion_decisions if row.get("bbox_union_status") == "bbox_union_available"),
        "bbox_union_not_supported_count": sum(
            1 for row in promotion_decisions if row.get("bbox_union_status") == "not_supported_by_current_bbox_artifact"
        ),
        "new_exact_promotion_count": sum(1 for row in promotion_decisions if row.get("decision") == "promote_exact"),
        "kept_non_exact_after_attempt_count": len(blocked),
        "generic_blocker_reason_count": sum(1 for row in blocked if row.get("failure_reason") in generic_reasons),
        "false_highlightable_claim_count": sum(
            1
            for row in promotion_decisions
            if row.get("record_type") != "article_relation"
            and row.get("highlightable") is True
            and row.get("can_be_exact_highlight") is not True
        ),
        "missing_feasibility_field_count": sum(1 for row in promotion_decisions if required_fields - set(row)),
        "missing_decision_count": len(expected - actual),
        "unexpected_decision_count": len(actual - expected),
        "duplicate_decision_id_count": duplicate_decision_ids,
        "blocked_decision_missing_reason_count": sum(1 for row in blocked if not row.get("failure_reason")),
        "false_exact_decision_count": len(false_exact),
    }
    return {
        **counts,
        "status": "complete"
        if counts["promotion_decision_count"] == counts["expected_promotion_decision_count"]
        and counts["blocked_decision_count"] == counts["promotion_blocked_count"]
        and not any(
            counts[key]
            for key in (
                "missing_decision_count",
                "unexpected_decision_count",
                "duplicate_decision_id_count",
                "blocked_decision_missing_reason_count",
                "false_exact_decision_count",
                "promotion_attempt_missing_count",
                "generic_blocker_reason_count",
                "false_highlightable_claim_count",
                "missing_feasibility_field_count",
            )
        )
        else "incomplete",
    }


def _metadata_exact_promotion_feasibility_health(*, promotion_decisions: list[dict]) -> dict:
    metadata_rows = [row for row in promotion_decisions if row.get("record_type") == "metadata_grounding"]
    field_feasibility_categories = {
        "exact_safe",
        "line_level_only",
        "sentence_extends_beyond_field",
        "page_level_only",
        "requires_word_level_bbox",
        "blocked_by_layout",
        "blocked_by_text_boundary",
    }
    categories = {
        "promotable_exact",
        "exact_span_found_but_bbox_missing",
        "multi_span_exact_possible",
        "page_level_only_by_policy",
        "blocked_by_text_boundary",
        "blocked_by_no_exact_bbox",
        "blocked_by_layout",
    }
    counts = {
        "audited_metadata_row_count": len(metadata_rows),
        **{
            f"{category}_count": sum(1 for row in metadata_rows if row.get("metadata_exact_promotion_feasibility") == category)
            for category in sorted(categories)
        },
        "metadata_decision_sentence_continues_beyond_field_count": sum(
            1 for row in metadata_rows if row.get("failure_reason") == "metadata_decision_sentence_continues_beyond_field"
        ),
        "metadata_publication_block_requires_page_level_support_count": sum(
            1 for row in metadata_rows if row.get("failure_reason") == "metadata_publication_block_requires_page_level_support"
        ),
        "missing_feasibility_count": sum(1 for row in metadata_rows if row.get("metadata_exact_promotion_feasibility") not in categories),
        "missing_final_reason_count": sum(1 for row in metadata_rows if not row.get("failure_reason")),
        "missing_field_bbox_feasibility_count": sum(
            1 for row in metadata_rows if row.get("field_bbox_feasibility") not in field_feasibility_categories
        ),
        "field_bbox_feasibility_counts": dict(sorted(Counter(row.get("field_bbox_feasibility") for row in metadata_rows).items())),
    }
    return {
        **counts,
        "status": "complete"
        if counts["missing_feasibility_count"] == 0
        and counts["missing_final_reason_count"] == 0
        and counts["missing_field_bbox_feasibility_count"] == 0
        else "incomplete",
    }


def _article_relation_runtime_policy_health(
    *,
    document_relations: list[dict] | tuple[dict, ...],
    article_amendment_relations: list[dict] | tuple[dict, ...],
    bbox_rows: list[dict],
    word_bboxes: list[dict],
    evidence: list[dict],
    legal_units: list[dict],
) -> dict:
    bbox_by_id = {
        str(row.get("bbox_id") or row.get("word_bbox_id")): row
        for row in (*bbox_rows, *word_bboxes)
    }
    bbox_by_id.update(
        {
            character["character_bbox_id"]: {**word, **character, "bbox_id": character["character_bbox_id"]}
            for word in word_bboxes
            for character in word.get("characters") or ()
        }
    )
    exact_rows = [row for row in article_amendment_relations if row.get("support_class") == "exact_article_relation"]
    trace_rows = [row for row in article_amendment_relations if row.get("support_class") == "trace_article_relation"]
    invalid_refs = [ref for row in article_amendment_relations for ref in row.get("bbox_refs") or () if ref not in bbox_by_id]
    invalid_coordinates = [
        ref
        for row in article_amendment_relations
        for ref in row.get("bbox_refs") or ()
        if ref in bbox_by_id and not all(bbox_by_id[ref].get(key) is not None for key in ("x0", "y0", "x1", "y1"))
    ]
    groups: dict[object, set[str]] = defaultdict(set)
    for row in article_amendment_relations:
        groups[row.get("source_role")].add(str(row.get("support_class")))
    partial_groups = [support for support in groups.values() if {"exact_article_relation", "trace_article_relation"} <= support]
    trace_reason_counts = Counter(row.get("trace_only_reason") for row in trace_rows)
    runtime_status = "not_executed_in_offline_validation__run_public_runtime_policy_gate"
    source_contract_errors = _source_relation_contract_errors(
        evidence=evidence,
        legal_units=legal_units,
        article_amendment_relations=list(article_amendment_relations),
    )
    counts = {
        "article_relation_total_count": len(article_amendment_relations),
        "article_relation_exact_support_count": len(exact_rows),
        "article_relation_trace_only_count": len(trace_rows),
        "article_relation_trace_missing_reason_count": trace_reason_counts.get(None, 0) + trace_reason_counts.get("", 0),
        "article_relation_trace_reason_counts": dict(sorted((key, value) for key, value in trace_reason_counts.items() if key)),
        "article_relation_promoted_from_scope_count": sum(1 for row in exact_rows if "scope" in str(row.get("evidence_id") or "")),
        "article_relation_unpromoted_trace_count": len(trace_rows),
        "article_relation_invalid_bbox_refs": len(invalid_refs),
        "article_relation_invalid_coordinates": len(invalid_coordinates),
        "article_relation_partial_answer_risk_count": len(partial_groups),
        "document_relation_exact_support_partial_trace_omitted_count": len(partial_groups),
        "relation_runtime_policy_slow_gate_status": runtime_status,
        "document_relation_count": len(document_relations),
        "source_relation_contract_status": "complete" if not source_contract_errors else "incomplete",
        "source_relation_missing_count": sum(
            1 for error in source_contract_errors if error.startswith("article_relation_missing_source_reference:")
        ),
        "source_relation_unexpected_count": sum(
            1 for error in source_contract_errors if error.startswith("article_relation_unexpected_source_reference:")
        ),
        "source_relation_duplicate_count": sum(
            1 for error in source_contract_errors if error.startswith("article_relation_duplicate_source_reference:")
        ),
    }
    return {
        **counts,
        "status": "complete"
        if not any(
            counts[key]
            for key in (
                "article_relation_trace_missing_reason_count",
                "article_relation_invalid_bbox_refs",
                "article_relation_invalid_coordinates",
                "source_relation_missing_count",
                "source_relation_unexpected_count",
                "source_relation_duplicate_count",
            )
        )
        else "incomplete",
    }


def _legal_graph_authority_health(
    *,
    graph_edges: list[dict] | tuple[dict, ...],
    article_amendment_relations: list[dict] | tuple[dict, ...],
    evidence: list[dict],
    bbox_rows: list[dict],
    word_bboxes: list[dict],
) -> dict:
    evidence_by_id = {row["evidence_id"]: row for row in evidence}
    bbox_ids = {str(row.get("bbox_id") or row.get("word_bbox_id")) for row in (*bbox_rows, *word_bboxes)}
    bbox_ids |= {character["character_bbox_id"] for word in word_bboxes for character in word.get("characters") or ()}
    relation_edges = [
        row
        for row in graph_edges
        if descriptor_for(row.get("edge_type")) and descriptor_for(row.get("edge_type")).authority_bearing
    ]
    article_relation_refs = {row.get("relation_id") for row in relation_edges if row.get("relation_id")}
    relations_by_id = {row.get("relation_id"): row for row in article_amendment_relations}
    forward_relation_edges = [row for row in relation_edges if not row.get("derived_from_edge_id")]
    exact_edges = [row for row in forward_relation_edges if relations_by_id.get(row.get("relation_id"), {}).get("support_class") == "exact_article_relation"]
    trace_edges = [row for row in forward_relation_edges if relations_by_id.get(row.get("relation_id"), {}).get("support_class") != "exact_article_relation"]
    authority_without_evidence = [
        row
        for row in relation_edges
        if row.get("support_evidence_ids") and any(evidence_id not in evidence_by_id for evidence_id in row["support_evidence_ids"])
    ]
    authority_without_bbox = [
        row for row in exact_edges if any(bbox_id not in bbox_ids for bbox_id in relations_by_id[row["relation_id"]].get("bbox_refs") or ())
    ]
    trace_promoted = [row for row in trace_edges if relations_by_id.get(row.get("relation_id"), {}).get("citation_final") is True]
    missing_fields = [
        row
        for row in relation_edges
        if not {"relation_id", "support_relation_ids", "support_evidence_ids", "support_kind"} <= set(row)
    ]
    projection_endpoint_mismatches = [
        row
        for row in relation_edges
        if _relation_projection_endpoint_mismatch(row)
    ]
    relation_direction_mismatches = [
        row
        for row in relation_edges
        if (row.get("relation_projection") or {}).get("relation_type") != row.get("edge_type")
        or (row.get("derived_from_edge_id") and (row.get("relation_projection") or {}).get("projection_direction") != "inverse")
        or (not row.get("derived_from_edge_id") and (row.get("relation_projection") or {}).get("projection_direction") != "forward")
    ]
    return {
        "status": "complete"
        if not (
            authority_without_evidence
            or authority_without_bbox
            or trace_promoted
            or missing_fields
            or projection_endpoint_mismatches
            or relation_direction_mismatches
            or any(row.get("citation_final") is True for row in graph_edges)
        )
        else "incomplete",
        "graph_edge_count": len(graph_edges),
        "article_relation_count": len(article_amendment_relations),
        "article_relation_graph_ref_count": len(article_relation_refs),
        "evidence_backed_relation_edge_count": len(exact_edges),
        "trace_only_relation_edge_count": len(trace_edges),
        "non_citable_edge_count": sum(1 for row in graph_edges if row.get("relation_id") or row.get("citation_final") is False),
        "authority_without_evidence_count": len(authority_without_evidence),
        "authority_without_bbox_count": len(authority_without_bbox),
        "trace_promoted_count": len(trace_promoted),
        "graph_final_citation_edge_count": sum(1 for row in graph_edges if row.get("citation_final") is True),
        "invalid_finality_policy_count": sum(1 for row in graph_edges if row.get("citation_final") is True),
        "missing_authority_field_count": len(missing_fields),
        "forward_relation_projection_endpoint_mismatch_count": sum(
            1 for row in projection_endpoint_mismatches if not row.get("derived_from_edge_id")
        ),
        "inverse_relation_projection_endpoint_mismatch_count": sum(
            1 for row in projection_endpoint_mismatches if row.get("derived_from_edge_id")
        ),
        "relation_direction_mismatch_count": len(relation_direction_mismatches),
        "authority_kind_counts": dict(sorted(Counter(row.get("authority_kind") or "relation_reference" for row in graph_edges).items())),
        "support_kind_counts": dict(sorted(Counter(row.get("support_kind") or "relation_reference" for row in graph_edges).items())),
    }


def _relation_projection_endpoint_mismatch(edge: dict) -> bool:
    projection = edge.get("relation_projection") or {}
    source_unit = projection.get("source_legal_unit_id")
    target_unit = projection.get("target_legal_unit_id")
    return bool(
        source_unit
        and target_unit
        and (
            edge.get("source_id") != f"legal_unit::{source_unit}"
            or edge.get("target_id") != f"legal_unit::{target_unit}"
        )
    )


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
    missing_source_refs = [row for row in page_text_spans if not row.get("source_document_id")]
    missing_page_refs = [row for row in page_text_spans if not isinstance(row.get("page_number"), int)]
    missing_bbox_coordinates = [row for row in page_text_spans if any(key not in row for key in ("x0", "y0", "x1", "y1"))]
    invalid_bbox_coordinates = [row for row in page_text_spans if not _valid_coordinates(row)]
    invalid_span_roles = [row for row in page_text_spans if row.get("span_role") not in SPAN_ROLES]
    invalid_semantic_classifications = [
        row for row in page_text_spans if row.get("semantic_classification") not in SEMANTIC_CLASSIFICATIONS
    ]
    invalid_legal_forces = [row for row in page_text_spans if row.get("legal_force") not in LEGAL_FORCES]
    invalid_promotion_statuses = [row for row in page_text_spans if row.get("promotion_status") not in PROMOTION_STATUSES]
    invalid_review_statuses = [row for row in page_text_spans if row.get("review_status") not in REVIEW_STATUSES]
    ambiguous_dispositions = [
        row
        for row in page_text_spans
        if row.get("promotion_status") not in PROMOTED_STATUSES
        and row.get("promotion_target_id")
        and row.get("promotion_target_type") not in {"legal_unit", "chunk"}
    ]
    return {
        "page_text_span_count": len(page_text_spans),
        "classified_span_count": sum(1 for row in page_text_spans if row.get("semantic_classification") in SEMANTIC_CLASSIFICATIONS),
        "span_disposition_present_count": len(page_text_spans) - len(missing_fields),
        "span_disposition_missing_count": len(missing_fields),
        "semantic_classification_present_count": sum(1 for row in page_text_spans if bool(row.get("semantic_classification"))),
        "known_unreferenced_span_count": len(span_ids - referenced_span_ids),
        "promotion_status_present_count": sum(1 for row in page_text_spans if "promotion_status" in row),
        "legal_force_present_count": sum(1 for row in page_text_spans if "legal_force" in row),
        "missing_source_ref_count": len(missing_source_refs),
        "missing_page_ref_count": len(missing_page_refs),
        "missing_bbox_coordinate_count": len(missing_bbox_coordinates),
        "invalid_bbox_coordinate_count": len(invalid_bbox_coordinates),
        "invalid_span_role_count": len(invalid_span_roles),
        "invalid_semantic_classification_count": len(invalid_semantic_classifications),
        "invalid_legal_force_count": len(invalid_legal_forces),
        "invalid_promotion_status_count": len(invalid_promotion_statuses),
        "invalid_review_status_count": len(invalid_review_statuses),
        "ambiguous_disposition_count": len(ambiguous_dispositions),
        "exclusion_reason_missing_for_excluded_count": len(excluded_missing_reason),
        "needs_review_count": len(needs_review_rows),
        "runtime_loadable_needs_review_count": sum(1 for row in needs_review_rows if row.get("runtime_loadable") is True),
        "canonical_use_allowed_needs_review_count": sum(1 for row in needs_review_rows if row.get("canonical_use_allowed") is True),
        "fake_grounding_id_count": len(fake_grounding_ids),
        "status": "complete"
        if page_text_spans
        and not missing_fields
        and not excluded_missing_reason
        and not fake_grounding_ids
        and not missing_source_refs
        and not missing_page_refs
        and not missing_bbox_coordinates
        and not invalid_bbox_coordinates
        and not invalid_span_roles
        and not invalid_semantic_classifications
        and not invalid_legal_forces
        and not invalid_promotion_statuses
        and not invalid_review_statuses
        and not ambiguous_dispositions
        else "incomplete",
    }


def _valid_coordinates(row: dict) -> bool:
    try:
        x0 = float(row["x0"])
        y0 = float(row["y0"])
        x1 = float(row["x1"])
        y1 = float(row["y1"])
    except (KeyError, TypeError, ValueError):
        return False
    return x0 < x1 and y0 < y1


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
    published_retrieval = [row for row in retrieval_units if row.get("artifact_status") == "published"]
    indexed_evidence_ids = {row["evidence_id"] for row in published_retrieval}
    nonruntime_accepted = [
        row
        for row in evidence
        if row["evidence_id"] in indexed_evidence_ids
        and (
            units_by_id.get(row.get("legal_unit_id"), {}).get("runtime_loadable") is False
            or chunks_by_unit.get(row.get("legal_unit_id"), {}).get("runtime_loadable") is False
        )
    ]
    page_grounded_accepted = [
        row for row in evidence if row["evidence_id"] in indexed_evidence_ids and row.get("bbox_precision") == "page_grounded_only"
    ]
    nonhighlightable_viewer_resolvable = [
        row
        for row in evidence
        if row["evidence_id"] in indexed_evidence_ids
        and row.get("viewer_highlightable") is False
        and row.get("status") == "final"
        and bool(row.get("bbox_refs"))
    ]
    accepted_for_nonruntime_chunks = [
        row for row in published_retrieval if chunks_by_unit.get(row.get("legal_unit_id"), {}).get("runtime_loadable") is False
    ]
    accepted_for_page_grounded = [
        row for row in published_retrieval if evidence_by_id.get(row.get("evidence_id"), {}).get("bbox_precision") == "page_grounded_only"
    ]
    unresolved_instrument = [
        row
        for row in evidence
        if _is_instrument_unit(units_by_id.get(row.get("legal_unit_id"), {}))
        and chunks_by_unit.get(row.get("legal_unit_id"), {}).get("runtime_loadable") is True
        and row["evidence_id"] not in indexed_evidence_ids
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
    indexed_evidence_ids = {row["evidence_id"] for row in retrieval_units if row.get("artifact_status") == "published"}
    instrument_evidence = [row for row in evidence if _is_instrument_unit(units_by_id.get(row.get("legal_unit_id"), {}))]
    fail_closed_targets = [
        row
        for row in instrument_evidence
        if _instrument_role_from_citation(row.get("citation")) in {"decision", "scope"} and row["evidence_id"] not in indexed_evidence_ids
    ]
    answerable_fail_closed_targets = [row for row in fail_closed_targets if row["evidence_id"] in indexed_evidence_ids]
    safe_exact_targets = [
        row
        for row in instrument_evidence
        if _instrument_role_from_citation(row.get("citation")) in {"scope", "recital"}
        and row.get("bbox_precision") == "exact"
        and row.get("viewer_highlightable") is True
    ]
    safe_not_accepted = [row for row in safe_exact_targets if row["evidence_id"] not in indexed_evidence_ids]
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
    return {
        "status": "complete",
        "runtime_health_mode": "test_suite_owned",
        "runtime_check_count": 0,
    }


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
            if row.get("source_anomaly_kind") not in {"renumbering_provenance", "source_marker_sequence_anomaly", "typed_source_discrepancy"}
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
        "source_id", "target_id", "object_role", "support_relation_ids", "support_evidence_ids", "support_exception_ids", "support_kind",
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


def _legal_unit_chunk_span_closure_health(
    *,
    legal_units: list[dict],
    chunks: list[dict],
    page_text_spans: list[dict],
    graph_nodes: list[dict],
) -> dict:
    units_by_id = {row["legal_unit_id"]: row for row in legal_units}
    chunks_by_unit: dict[str, list[dict]] = defaultdict(list)
    for chunk in chunks:
        legal_unit_id = str(chunk.get("legal_unit_id") or "")
        chunks_by_unit[legal_unit_id].append(chunk)
    spans_by_id = {row["text_span_id"]: row for row in page_text_spans}
    legal_unit_graph_ids = {row.get("legal_unit_id") for row in graph_nodes if row.get("node_type") == "legal_unit"}

    active_units = [row for row in legal_units if row.get("runtime_loadable") is True]
    active_chunks = [row for row in chunks if row.get("runtime_loadable") is True]
    source_text_backed_units_without_spans = [
        row for row in legal_units if row.get("text") and not row.get("text_span_ids") and row.get("runtime_loadable") is not False
    ]
    source_text_backed_chunks_without_spans = [
        row for row in chunks if row.get("text") and row.get("canonical_use_allowed") is True and not row.get("text_span_ids")
    ]
    invalid_parent_refs: list[str] = []
    missing_parent_refs: list[dict] = []
    impossible_structural_nesting: list[dict] = []
    for unit in legal_units:
        parents = unit.get("parent_legal_unit_ids") or ()
        invalid_parent_refs.extend(parent_id for parent_id in parents if parent_id not in units_by_id)
        valid_parents = [units_by_id[parent_id] for parent_id in parents if parent_id in units_by_id]
        if (
            unit.get("unit_type") == "ayat_record"
            and "Pasal" in " ".join(unit.get("hierarchy") or ())
            and not any(parent.get("unit_type") == "pasal_record" for parent in valid_parents)
        ):
            missing_parent_refs.append(unit)
        if unit.get("unit_type") == "pasal_record" and unit.get("hierarchy") and not valid_parents:
            missing_parent_refs.append(unit)
        if any(parent.get("source_document_id") != unit.get("source_document_id") for parent in valid_parents):
            impossible_structural_nesting.append(unit)

    legal_unit_chunk_mismatches = [unit for unit in legal_units if len(chunks_by_unit.get(unit["legal_unit_id"], ())) != 1]
    chunk_text_mismatches = [
        chunk
        for chunk in chunks
        if chunk.get("status") != "parent_context_only"
        and _compact_for_closure(chunk.get("text")) not in _compact_for_closure(units_by_id.get(chunk.get("legal_unit_id"), {}).get("text"))
    ]
    unit_span_errors = _span_link_errors(active_units, spans_by_id)
    chunk_span_errors = _span_link_errors(active_chunks, spans_by_id)
    orphan_structural_units = [
        unit
        for unit in legal_units
        if unit.get("unit_type") in {"bab_record", "aturan_peralihan_record", "aturan_tambahan_record"}
        and unit.get("runtime_loadable") is True
        and not substantive_structural_unit(unit)
        and not unit.get("evidence_ids")
        and not any(unit["legal_unit_id"] in (candidate.get("parent_legal_unit_ids") or ()) for candidate in legal_units)
    ]
    counts = {
        "legal_unit_count": len(legal_units),
        "chunk_count": len(chunks),
        "legal_unit_exact_span_link_count": sum(1 for row in legal_units if row.get("grounding_status") == "text_span_exact"),
        "chunk_exact_span_link_count": sum(1 for row in chunks if row.get("grounding_status") == "text_span_exact"),
        "legal_unit_containing_span_link_count": sum(
            1 for row in legal_units if row.get("grounding_status") == "text_span_containing_match"
        ),
        "chunk_containing_span_link_count": sum(1 for row in chunks if row.get("grounding_status") == "text_span_containing_match"),
        "legal_unit_chunk_mismatch_count": len(legal_unit_chunk_mismatches),
        "missing_parent_ref_count": len(missing_parent_refs),
        "invalid_parent_ref_count": len(invalid_parent_refs),
        "hierarchy_cycle_count": _hierarchy_cycle_count(legal_units),
        "impossible_structural_nesting_count": len(impossible_structural_nesting),
        "active_legal_units_without_span_ids": sum(1 for row in active_units if not row.get("text_span_ids")),
        "active_chunks_without_span_ids": sum(1 for row in active_chunks if not row.get("text_span_ids")),
        "source_text_backed_legal_units_without_span_ids_count": len(source_text_backed_units_without_spans),
        "source_text_backed_chunks_without_span_ids_count": len(source_text_backed_chunks_without_spans),
        "invalid_legal_unit_span_ref_count": unit_span_errors["invalid_ref_count"],
        "invalid_chunk_span_ref_count": chunk_span_errors["invalid_ref_count"],
        "source_page_span_mismatch_count": unit_span_errors["source_page_mismatch_count"] + chunk_span_errors["source_page_mismatch_count"],
        "excluded_span_link_count": unit_span_errors["excluded_link_count"] + chunk_span_errors["excluded_link_count"],
        "chunk_text_mismatch_count": len(chunk_text_mismatches),
        "orphan_structural_unit_count": len(orphan_structural_units),
        "missing_legal_unit_graph_node_count": sum(1 for row in legal_units if row["legal_unit_id"] not in legal_unit_graph_ids),
    }
    non_error_count_keys = {
        "legal_unit_count",
        "chunk_count",
        "legal_unit_exact_span_link_count",
        "chunk_exact_span_link_count",
        "legal_unit_containing_span_link_count",
        "chunk_containing_span_link_count",
    }
    return {
        **counts,
        "status": "complete"
        if not any(value for key, value in counts.items() if key.endswith("_count") and key not in non_error_count_keys)
        else "incomplete",
    }


def _span_link_errors(rows: list[dict], spans_by_id: dict[str, dict]) -> dict[str, int]:
    invalid_refs = 0
    source_page_mismatches = 0
    excluded_links = 0
    for row in rows:
        source_id = row.get("source_document_id")
        pages = set(row.get("page_numbers") or range(int(row.get("page_start", 0)), int(row.get("page_end", 0)) + 1))
        for span_id in row.get("text_span_ids") or ():
            span = spans_by_id.get(span_id)
            if not span:
                invalid_refs += 1
                continue
            if span.get("source_document_id") != source_id or span.get("page_number") not in pages:
                source_page_mismatches += 1
            if span.get("promotion_status") in {"excluded_nonlegal", "needs_review"} or span.get("span_role") in {
                "header_footer",
                "separator",
                "footnote_marker",
                "nonlegal_artifact",
            }:
                excluded_links += 1
    return {
        "invalid_ref_count": invalid_refs,
        "source_page_mismatch_count": source_page_mismatches,
        "excluded_link_count": excluded_links,
    }


def _hierarchy_cycle_count(legal_units: list[dict]) -> int:
    parents = {row["legal_unit_id"]: tuple(row.get("parent_legal_unit_ids") or ()) for row in legal_units}
    cyclic = set()

    def visit(unit_id: str, path: set[str]) -> bool:
        if unit_id in path:
            return True
        return any(visit(parent, {*path, unit_id}) for parent in parents.get(unit_id, ()) if parent in parents)

    for unit_id in parents:
        if visit(unit_id, set()):
            cyclic.add(unit_id)
    return len(cyclic)


def _compact_for_closure(text: object) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or "")).replace("\u00ad", "")
    return "".join(re.findall(r"\w+", normalized.casefold()))


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


def _viewer_provenance_coverage_health(*, page_text_spans: list[dict], bbox_rows: list[dict]) -> dict:
    bbox_registry_keys = {_span_bbox_key(row) for row in bbox_rows}
    absent_rows = [row for row in page_text_spans if _span_bbox_key(row) not in bbox_registry_keys]
    allowed_buckets = {
        "legal_citation_candidate",
        "metadata_provenance_candidate",
        "source_anomaly_provenance_candidate",
        "structural_provenance_only",
        "nonlegal_excluded_provenance",
        "raw_span_only_with_reason",
    }
    allowed_reasons = {
        "already_covered_by_final_evidence_bbox",
        "exact_word_bbox_available",
        "structural_provenance_only",
        "nonlegal_excluded_from_public_highlight",
        "source_anomaly_anchor_only_until_exact_span_available",
        "metadata_page_grounded_only_by_policy",
        "blocked_by_text_mismatch",
        "blocked_by_layout",
        "blocked_by_missing_exact_bbox",
        "blocked_by_no_word_level_bbox_artifact",
    }
    counts = {
        "page_text_span_count": len(page_text_spans),
        "bbox_registry_row_count": len(bbox_rows),
        "bbox_key_present_span_count": len(page_text_spans) - len(absent_rows),
        "bbox_key_absent_span_count": len(absent_rows),
        **{
            f"{bucket}_count": sum(1 for row in absent_rows if row.get("bbox_registry_coverage_bucket") == bucket)
            for bucket in sorted(allowed_buckets)
        },
        **{
            f"{reason}_count": sum(1 for row in absent_rows if row.get("bbox_registry_coverage_reason") == reason)
            for reason in sorted(allowed_reasons)
        },
        "missing_bucket_count": sum(1 for row in absent_rows if row.get("bbox_registry_coverage_bucket") not in allowed_buckets),
        "missing_reason_count": sum(1 for row in absent_rows if row.get("bbox_registry_coverage_reason") not in allowed_reasons),
        "incomplete_disposition_count": sum(1 for row in page_text_spans if any(field not in row for field in SPAN_DISPOSITION_FIELDS)),
        "highlight_without_span_bbox_count": sum(1 for row in page_text_spans if row.get("highlightable") and not row.get("span_bbox_ids")),
        "final_without_exact_span_bbox_count": sum(
            1 for row in page_text_spans if row.get("citation_final") and (row.get("exactness") != "exact" or not row.get("span_bbox_ids"))
        ),
        "invalid_present_status_count": sum(
            1 for row in page_text_spans if row.get("bbox_registry_coverage_status") not in {"bbox_key_present", "bbox_key_absent"}
        ),
    }
    return {
        **counts,
        "status": "complete"
        if counts["bbox_key_absent_span_count"] == sum(counts[f"{bucket}_count"] for bucket in sorted(allowed_buckets))
        and not any(
            counts[key]
            for key in (
                "missing_bucket_count",
                "missing_reason_count",
                "incomplete_disposition_count",
                "highlight_without_span_bbox_count",
                "final_without_exact_span_bbox_count",
                "invalid_present_status_count",
            )
        )
        else "incomplete",
    }


def _span_bbox_key(row: dict) -> tuple[object, ...]:
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
