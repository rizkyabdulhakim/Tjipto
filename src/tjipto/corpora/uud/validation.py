from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from tjipto.corpora.uud.bbox_builder import bbox_precision_counts
from tjipto.core.manifest import read_jsonl


DECISION_LABELS = {
    "Perubahan Pertama Decision",
    "Perubahan Ketiga Decision",
    "Perubahan Keempat Decision",
}
INSERTED_BAB_HEADING_BBOX_MARKER = "::heading_bab_"
STRUCTURAL_FORBIDDEN_MARKERS = (
    "Ditetapkan di Jakarta",
)
PROVENANCE_EDGE_TYPES = {
    "HAS_FINAL_EVIDENCE",
    "BELONGS_TO_SOURCE_ROLE",
    "USES_SOURCE_PDF",
    "PAGE_GROUNDED_AT",
    "HAS_BBOX",
    "EXCLUDED_BECAUSE",
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
    "HAS_EFFECTIVE_RULE",
    "HAS_SIGNATORY",
    "HAS_DECISION_SESSION",
    "HAS_SOURCE_ANOMALY",
}


def update_validation_report(
    validation_report: dict,
    *,
    chunks: list[dict],
    legal_units: list[dict],
    excluded_records: list[dict],
    evidence: list[dict],
    bbox_rows: list[dict],
    retrieval_units: list[dict],
    graph_nodes: list[dict],
    graph_edges: list[dict],
    page_text_spans: list[dict],
) -> None:
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
    validation_report.setdefault("instrument_baseline", {})
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
        "status": "field_grounded",
        "note": "field-level metadata grounding preserves block-level rows and keeps metadata viewer highlights fail-closed unless exact accepted support exists",
    }
    if "referenced_artifacts" in validation_report and "page_text_spans.jsonl" not in validation_report["referenced_artifacts"]:
        validation_report["referenced_artifacts"].append("page_text_spans.jsonl")
    validation_report["legal_graph_baseline"] = {
        "status": "evidence_backed_minimal_baseline",
        "legal_edge_types": sorted(LEGAL_EDGE_TYPES),
        "runtime_loadable_legal_edges": sum(
            1
            for row in graph_edges
            if row.get("edge_type") in LEGAL_EDGE_TYPES and row.get("runtime_loadable") is True
        ),
    }
    validation_report.setdefault("structure_fidelity", {})
    validation_report["structure_fidelity"]["inserted_bab_heading_owner_policy"] = (
        "inserted heading bboxes may stay exact, but they are viewer_highlightable only when owned by bab_record evidence"
    )


def validate_uud_artifact_dir(final_dir: Path) -> tuple[str, ...]:
    legal_units = read_jsonl(final_dir / "legal_units.jsonl")
    chunks = read_jsonl(final_dir / "chunks.jsonl")
    evidence = read_jsonl(final_dir / "evidence_registry.jsonl")
    bbox_rows = read_jsonl(final_dir / "bbox_registry.jsonl")
    retrieval_units = read_jsonl(final_dir / "retrieval_units.jsonl")
    metadata_grounding = read_jsonl(final_dir / "metadata_grounding.jsonl")
    graph_nodes = read_jsonl(final_dir / "graph_nodes.jsonl")
    graph_edges = read_jsonl(final_dir / "graph_edges.jsonl")
    source_conflicts = read_jsonl(final_dir / "source_conflicts.jsonl")
    page_text_spans = read_jsonl(final_dir / "page_text_spans.jsonl") if (final_dir / "page_text_spans.jsonl").exists() else []

    errors: list[str] = []
    seen_ids: dict[str, set[str]] = defaultdict(set)
    units_by_id = {row["legal_unit_id"]: row for row in legal_units}
    chunks_by_id = {row["chunk_id"]: row for row in chunks}
    evidence_by_id = {row["evidence_id"]: row for row in evidence}
    bbox_by_id = {row["bbox_id"]: row for row in bbox_rows}
    graph_node_ids = {row["node_id"] for row in graph_nodes}
    source_conflict_ids = {row["source_conflict_id"] for row in source_conflicts}
    metadata_grounding_ids = {row["metadata_grounding_id"] for row in metadata_grounding}
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
            if not any(units_by_id[parent]["unit_type"] == "bab_record" for parent in row.get("parent_legal_unit_ids") or () if parent in units_by_id):
                errors.append(f"pasal_missing_bab_parent:{row['legal_unit_id']}")
        if row["unit_type"] == "bab_record" and any(marker in row["text"] for marker in STRUCTURAL_FORBIDDEN_MARKERS):
            errors.append(f"structural_bab_contains_instrument_text:{row['legal_unit_id']}")
        for text_span_id in row.get("text_span_ids") or ():
            if text_span_id not in text_span_ids:
                errors.append(f"orphan_legal_unit_text_span:{row['legal_unit_id']}:{text_span_id}")
        for field in ("source_role", "temporal_context", "page_numbers", "text_span_ids", "bbox_ids", "grounding_status", "validation_status"):
            if row.get("runtime_loadable") is True and field not in row:
                errors.append(f"runtime_loadable_legal_unit_missing_{field}:{row['legal_unit_id']}")
        if row.get("runtime_loadable") is True:
            if not row.get("text_span_ids"):
                errors.append(f"runtime_loadable_legal_unit_missing_text_span:{row['legal_unit_id']}")
            if not row.get("bbox_ids"):
                errors.append(f"runtime_loadable_legal_unit_missing_bbox:{row['legal_unit_id']}")

    for row in chunks:
        if row["chunk_id"] in seen_ids["chunk_id"]:
            errors.append(f"duplicate_chunk_id:{row['chunk_id']}")
        seen_ids["chunk_id"].add(row["chunk_id"])
        if row["legal_unit_id"] not in units_by_id:
            errors.append(f"orphan_chunk:{row['chunk_id']}")
        for text_span_id in row.get("text_span_ids") or ():
            if text_span_id not in text_span_ids:
                errors.append(f"orphan_chunk_text_span:{row['chunk_id']}:{text_span_id}")
        if row.get("runtime_loadable") is True and not row.get("grounding_status"):
            errors.append(f"runtime_loadable_chunk_missing_grounding:{row['chunk_id']}")
        if row.get("runtime_loadable") is True and not row.get("bbox_ids"):
            errors.append(f"runtime_loadable_chunk_missing_bbox:{row['chunk_id']}")
        if row.get("runtime_loadable") is True and row.get("grounding_status") != "text_span_exact":
            errors.append(f"runtime_loadable_chunk_not_text_exact:{row['chunk_id']}")
        if row["chunk_type"] == "bab_structural_context_record" and any(marker in row["text"] for marker in STRUCTURAL_FORBIDDEN_MARKERS):
            errors.append(f"structural_chunk_contains_instrument_text:{row['chunk_id']}")
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
    for row in metadata_grounding:
        if row.get("viewer_highlightable") is not False:
            errors.append(f"metadata_grounding_highlightable_not_clarified:{row['metadata_grounding_id']}")
        if row.get("bbox_precision") not in {None, "coarse", "exact", "page_grounded_only"}:
            errors.append(f"invalid_metadata_bbox_precision:{row['metadata_grounding_id']}")
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
            evidence_ref = row.get("evidence_ref")
            if evidence_ref and evidence_ref not in evidence_by_id and evidence_ref not in metadata_grounding_ids and evidence_ref not in source_conflict_ids:
                errors.append(f"legal_edge_unknown_evidence_ref:{row['edge_id']}:{evidence_ref}")

    return tuple(sorted(set(errors)))
