from __future__ import annotations


def apply_graph_authority_contract(edges: list[dict], nodes: list[dict], legal_units: list[dict], evidence: list[dict]) -> None:
    """Make graph provenance explicit without promoting graph traversal to legal citation."""
    nodes_by_id = {row["node_id"]: row for row in nodes}
    units_by_id = {row["legal_unit_id"]: row for row in legal_units}
    evidence_by_unit: dict[str, list[dict]] = {}
    for row in evidence:
        evidence_by_unit.setdefault(row["legal_unit_id"], []).append(row)
    for edge in edges:
        source = nodes_by_id.get(edge.get("source_id"), {})
        target = nodes_by_id.get(edge.get("target_id"), {})
        unit_ids = _unit_ids(source, target, units_by_id)
        support = [row for unit_id in unit_ids for row in evidence_by_unit.get(unit_id, ())]
        edge["source_node_type"] = source.get("node_type")
        edge["target_node_type"] = target.get("node_type")
        edge["supporting_evidence_ids"] = sorted({row["evidence_id"] for row in support})
        edge["source_document_ids"] = sorted(
            {str(row.get("source_document_id")) for row in support if row.get("source_document_id")}
            or {
                str(units_by_id[unit_id].get("source_document_id"))
                for unit_id in unit_ids
                if units_by_id[unit_id].get("source_document_id")
            }
        )
        edge["page_numbers"] = sorted({page for row in support for page in row.get("page_numbers") or ()})
        edge["text_span_ids"] = sorted({span for row in support for span in row.get("text_span_ids") or ()})
        edge["bbox_refs"] = sorted({bbox for row in support for bbox in row.get("bbox_refs") or ()})
        edge["derivation_method"] = _derivation_method(edge)
        edge["authority_kind"] = edge.get("edge_authority_level") or "trace"
        edge["citation_final"] = False
        edge["citation_finality_reason"] = _finality_reason(edge)


def _unit_ids(source: dict, target: dict, units_by_id: dict[str, dict]) -> list[str]:
    values = []
    for node in (source, target):
        unit_id = node.get("legal_unit_id")
        if unit_id in units_by_id:
            values.append(unit_id)
    return values


def _derivation_method(edge: dict) -> str:
    edge_type = str(edge.get("edge_type") or edge.get("relation_type") or "")
    if edge_type in {"CONTAINS", "PRECEDES", "PART_OF"}:
        return "deterministic_structural_rule"
    if edge_type in {"AMENDS", "DELETES", "RENAMES", "RENAMES_OR_RENUMBERS", "HISTORICAL_TO_CANONICAL", "SOURCE_CONFLICT"}:
        return "reviewed_corpus_spec"
    return "explicit_source_text"


def _finality_reason(edge: dict) -> str:
    edge_type = str(edge.get("edge_type") or edge.get("relation_type") or "")
    if edge_type in {"HISTORICAL_TO_CANONICAL", "SOURCE_CONFLICT", "SOURCE_MARKER_ANOMALY"}:
        return "historical_or_anomaly_trace_not_final"
    if edge.get("edge_authority_level") == "provenance":
        return "instrument_provenance_not_final"
    return "graph_relation_requires_exact_evidence_citation"
