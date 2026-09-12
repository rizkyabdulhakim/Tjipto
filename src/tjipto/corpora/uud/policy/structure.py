from __future__ import annotations

from collections import defaultdict

from tjipto.contracts.evidence import compact_source_words
from tjipto.corpora.uud.span_disposition_policy import substantive_structural_unit
from tjipto.corpora.uud.specs import UUD_INSERTED_BAB_PREDECESSORS


def apply_uud_parent_policy(units: list[dict]) -> None:
    by_id = {row["legal_unit_id"]: row for row in units}
    for row in units:
        predecessors = set(UUD_INSERTED_BAB_PREDECESSORS.get(row.get("unit_label"), ()))
        if not predecessors:
            continue
        row["parent_legal_unit_ids"] = [
            parent_id
            for parent_id in row.get("parent_legal_unit_ids") or ()
            if by_id.get(parent_id, {}).get("unit_label") not in predecessors
        ]


def legal_unit_chunk_span_closure_health(
    *,
    legal_units: list[dict],
    chunks: list[dict],
    page_text_spans: list[dict],
    graph_nodes: list[dict],
) -> dict:
    units_by_id = {row["legal_unit_id"]: row for row in legal_units}
    chunks_by_unit: dict[str, list[dict]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_unit[str(chunk.get("legal_unit_id") or "")].append(chunk)
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
        and compact_source_words(chunk.get("text")) not in compact_source_words(units_by_id.get(chunk.get("legal_unit_id"), {}).get("text"))
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
        "legal_unit_containing_span_link_count": sum(1 for row in legal_units if row.get("grounding_status") == "text_span_containing_match"),
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


def chunk_self_contained_health(chunks: list[dict], units_by_id: dict[str, dict]) -> dict:
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
