from __future__ import annotations

STRUCTURAL_EDGES = {"CONTAINS", "PART_OF", "PRECEDES", "FOLLOWS", "INSERTED_AFTER"}
ENDPOINT_EDGES = {
    "HAS_BBOX",
    "PAGE_GROUNDED_AT",
    "USES_SOURCE_PDF",
    "BELONGS_TO_SOURCE_ROLE",
    "HAS_FINAL_EVIDENCE",
}
INSTRUMENT_EDGES = {"MODIFIES", "DELETES", "HAS_SIGNATORY", "HAS_DECISION_SESSION", "HAS_EFFECTIVE_RULE"}


def apply_graph_relation_policy(*, edges: list[dict], nodes: list[dict], evidence: list[dict]) -> None:
    nodes_by_id = {row["node_id"]: row for row in nodes}
    evidence_by_id = {row["evidence_id"]: row for row in evidence}
    for edge in edges:
        source = nodes_by_id[edge["source_id"]]
        target = nodes_by_id[edge["target_id"]]
        edge_type = edge["edge_type"]
        evidence_ids = _direct_evidence_ids(edge, source, target, evidence_by_id)
        supports = [evidence_by_id[evidence_id] for evidence_id in evidence_ids]
        if edge_type in STRUCTURAL_EDGES:
            support_kind = "deterministic_structure"
            derivation_method = "deterministic_structural_rule"
            evidence_ids = []
            supports = []
        elif edge_type in ENDPOINT_EDGES:
            support_kind = "endpoint_provenance"
            derivation_method = "endpoint_metadata"
        elif edge_type in INSTRUMENT_EDGES:
            support_kind = "instrument_provenance"
            derivation_method = "reviewed_corpus_spec" if edge_type == "DELETES" else "explicit_source_text"
        elif edge_type == "HAS_SOURCE_ANOMALY":
            support_kind = "source_anomaly_trace"
            derivation_method = "reviewed_corpus_spec"
        elif edge_type == "EXCLUDED_BECAUSE":
            support_kind = "nonlegal"
            derivation_method = "reviewed_corpus_spec"
        else:
            raise ValueError(f"unknown_uud_graph_edge:{edge_type}")
        edge.update(
            {
                "relation_type": edge_type,
                "source_node_type": source["node_type"],
                "target_node_type": target["node_type"],
                "support_kind": support_kind,
                "supporting_evidence_ids": evidence_ids,
                "source_document_ids": _ordered_unique(
                    [
                        source.get("source_document_id"),
                        target.get("source_document_id"),
                        *(row.get("source_document_id") for row in supports),
                    ]
                ),
                "page_numbers": sorted(
                    {
                        page
                        for row in (source, target, *supports)
                        for page in ([row.get("page_number")] if row.get("page_number") else row.get("page_numbers") or [])
                        if isinstance(page, int)
                    }
                ),
                "text_span_ids": _ordered_unique(span for row in supports for span in row.get("text_span_ids") or ()),
                "bbox_refs": _edge_bbox_refs(edge_type, target, supports),
                "derivation_method": derivation_method,
                "derivation_reason": f"uud_{support_kind}",
                "citation_final": False,
            }
        )


def _direct_evidence_ids(edge: dict, source: dict, target: dict, evidence_by_id: dict[str, dict]) -> list[str]:
    candidates = [edge.get("evidence_ref"), source.get("final_evidence_id"), target.get("final_evidence_id")]
    for node in (source, target):
        if node.get("node_type") == "final_evidence":
            candidates.append(node.get("evidence_id") or str(node.get("node_id", "")).removeprefix("final_evidence::"))
    return _ordered_unique(value for value in candidates if value in evidence_by_id)


def _edge_bbox_refs(edge_type: str, target: dict, supports: list[dict]) -> list[str]:
    if edge_type == "HAS_BBOX" and target.get("bbox_id"):
        return [target["bbox_id"]]
    return _ordered_unique(ref for row in supports for ref in row.get("bbox_refs") or ())


def _ordered_unique(values) -> list:
    return list(dict.fromkeys(value for value in values if value is not None))
