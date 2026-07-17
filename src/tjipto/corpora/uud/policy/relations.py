from __future__ import annotations

STRUCTURAL_EDGES = {"CONTAINS", "PART_OF", "PRECEDES", "FOLLOWS", "INSERTED_AFTER"}
ENDPOINT_EDGES = {
    "HAS_BBOX",
    "PAGE_GROUNDED_AT",
    "USES_SOURCE_PDF",
    "BELONGS_TO_SOURCE_ROLE",
    "HAS_FINAL_EVIDENCE",
}
INSTRUMENT_EDGES = {"MODIFIES", "DELETES", "RENAMES", "HAS_SIGNATORY", "HAS_DECISION_SESSION", "HAS_EFFECTIVE_RULE"}


def is_renumbering_provision(row: dict) -> bool:
    """Identify the reviewed UUD provision by structure, not display citation."""
    hierarchy = {str(value).strip().casefold() for value in row.get("hierarchy") or ()}
    text = str(row.get("quoted_text") or "").casefold()
    return str(row.get("source_role") or "").startswith("amendment_") and "(c)" in hierarchy and "menjadi" in text


def is_scope_provision(row: dict) -> bool:
    hierarchy = tuple(str(value).strip().casefold() for value in row.get("hierarchy") or ())
    return bool(hierarchy) and hierarchy[-1].endswith(" scope")


def is_deletion_provision(row: dict) -> bool:
    hierarchy = {str(value).strip().casefold() for value in row.get("hierarchy") or ()}
    text = str(row.get("quoted_text") or "").casefold()
    return "(d)" in hierarchy and any(term in text for term in ("dihapus", "menghapus", "penghapusan"))


def apply_graph_relation_policy(*, edges: list[dict], nodes: list[dict], evidence: list[dict], article_relations: list[dict]) -> None:
    nodes_by_id = {row["node_id"]: row for row in nodes}
    evidence_by_id = {row["evidence_id"]: row for row in evidence}
    relations_by_id = {row["relation_id"]: row for row in article_relations}
    for edge in edges:
        for field in (
            "edge_authority_level",
            "evidence_requirement",
            "relation_support",
            "graph_finality_policy",
            "viewer_highlightable",
            "reason",
        ):
            edge.pop(field, None)
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
        elif edge_type in {"MODIFIES", "DELETES", "RENAMES"}:
            relation = relations_by_id.get(str(edge.get("article_relation_ref") or ""), {})
            exact_relation = relation.get("support_class") == "exact_article_relation"
            support_kind = "exact_source_relation" if exact_relation else "instrument_provenance"
            derivation_method = "explicit_source_text" if exact_relation else "reviewed_corpus_spec"
        elif edge_type in INSTRUMENT_EDGES:
            support_kind = "instrument_provenance"
            derivation_method = "explicit_source_text" if supports else "reviewed_corpus_spec"
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
                "source_role": _source_role_class(source.get("source_role") or target.get("source_role") or edge.get("source_role")),
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
    candidates = [*(edge.get("supporting_evidence_ids") or ()), source.get("final_evidence_id"), target.get("final_evidence_id")]
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


def _source_role_class(source_role: object) -> str:
    value = str(source_role or "")
    if value == "current_consolidated":
        return "consolidated"
    if value == "original_historical":
        return "historical"
    if value.startswith("amendment_"):
        return "amendment"
    if "anomaly" in value:
        return "anomaly"
    return "canonical"
