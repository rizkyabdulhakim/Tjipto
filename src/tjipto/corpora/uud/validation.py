from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Mapping
from typing import cast

from tjipto.corpora.uud.bbox_builder import bbox_precision_counts
from tjipto.corpora.uud.artifact_policy import UUD_ARTIFACT_SCHEMA
from tjipto.contracts.artifacts import MINIMUM_ARTIFACT_FIELDS
from tjipto.corpora.uud.contract import CONTRACT_FINGERPRINT, CONTRACT_ID, CONTRACT_VERSION
from tjipto.corpora.uud.specs import UUD_LEGAL_GRAPH_EDGE_SCHEMA
from tjipto.corpora.uud.policy.validation import (
    viewer_provenance_coverage_health as _viewer_provenance_coverage_health,
)
from tjipto.corpora.uud.policy.structure import (
    chunk_self_contained_health as _chunk_self_contained_health,
    legal_unit_chunk_span_closure_health as _legal_unit_chunk_span_closure_health,
)
from tjipto.corpora.uud.schema_validation import validate_schema_contract
from tjipto.corpora.uud.validation_provenance import (
    _artifact_origin_health,
    _edge_type_counts,
    _provenance_exception_health,
    _source_conflict_provenance_health,
    _structural_authority_contract_health,
)
from tjipto.corpora.uud.validation_intent import (
    _amendment_context_default_boundary_health,
    _instrument_exact_grounding_health,
    _instrument_intent_invariant_router_health,
    _instrument_intent_matrix_health,
    _instrument_like_boundary_generalization_health,
    _instrument_natural_query_precision_health,
    _instrument_query_precision_health,
    _instrument_runtime_safety_health,
    _intent_arbitration_priority_health,
    _partial_signal_instrument_boundary_health,
    _semantic_precedence_health,
)
from tjipto.corpora.uud.validation_grounding import (
    _all_text_disposition_health,
    _metadata_bbox_registry_health,
    _metadata_exact_promotion_feasibility_health,
    _pdf_health_summary,
    _promotion_decision_audit_health,
    _promotion_engine_health,
    _raw_source_geometry_health,
    _selector_geometry_health,
    _source_object_disposition_health,
    _source_quote_fidelity_health,
    _span_sequence_grounding_health,
    _word_bbox_registry_health,
)
from tjipto.corpora.uud.policy.relations import (
    article_relation_runtime_policy_health as _article_relation_runtime_policy_health,
    legal_graph_authority_health as _legal_graph_authority_health,
)
from tjipto.core.manifest import artifact_set_digest, read_json, read_jsonl, validate_manifest


STRUCTURAL_SEQUENCE_EDGE_TYPES = {
    edge_type for edge_type, schema in UUD_LEGAL_GRAPH_EDGE_SCHEMA.items() if schema.get("category") == "structural_sequence"
}
LEGAL_EDGE_TYPES = {
    "CONTAINS",
    "PART_OF",
    "AMENDS",
    "AMENDED_BY",
    "ADDS",
    "AMBIGUOUS_OPERATION",
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
        {row["bbox_id"] for row in bbox_rows} | {row["word_bbox_id"] for row in word_bboxes},
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
            and all(any(word.get("word_bbox_id") == ref for word in word_bboxes) for ref in row.get("span_bbox_ids") or ())
            and all(
                next((item for item in evidence if item.get("evidence_id") == evidence_id), {}).get("exactness") == "exact"
                for evidence_id in row.get("evidence_ids") or ()
            )
        ),
        "non_citable_absent_span_count": sum(
            1 for row in page_text_spans if row.get("viewer_highlightable") is not True and not row.get("span_bbox_ids")
        ),
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
    promotion_engine_health["promotion_blocked_count"] = promotion_decision_audit_health["blocked_decision_count"]
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
    validation_report["validated_artifact_set_digest"] = artifact_set_digest({"files": manifest_files}, exclude=("validation_report.json",))
    _derive_validation_status(validation_report)
    return validation_report


def _derive_validation_status(report: dict) -> None:
    unsatisfied = {
        health_key: {
            "expected": expected_status,
            "actual": (report.get(health_key) or {}).get("status") if isinstance(report.get(health_key), Mapping) else None,
        }
        for health_key, expected_status in _REQUIRED_HEALTH_STATUSES.items()
        if not isinstance(report.get(health_key), Mapping) or report[health_key].get("status") != expected_status
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
            row["characters"] = [{"character_bbox_id": character.get("character_bbox_id")} for character in source.get("characters") or ()]
            rows.append(row)
    return rows


def validate_uud_artifacts(final_dir: Path, artifacts: Mapping[str, object]) -> tuple[str, ...]:
    schema_errors = list(validate_schema_contract(artifacts))
    report = artifacts.get("validation_report")
    for health_key, expected_status in _REQUIRED_HEALTH_STATUSES.items():
        if not isinstance(report, Mapping) or report.get(health_key, {}).get("status") != expected_status:
            schema_errors.append(f"validation_report_incomplete:{health_key}")
    return tuple(dict.fromkeys(schema_errors))
