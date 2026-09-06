from __future__ import annotations

from collections import Counter, defaultdict

from tjipto.contracts.relations import descriptor_for


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
        edge_type = edge["edge_type"]
        descriptor = descriptor_for(edge_type)
        if descriptor is None:
            raise ValueError(f"unknown_uud_graph_edge:{edge_type}")
        if descriptor.authority_bearing:
            relation_id = str(edge.get("article_relation_ref") or "")
            relation = relations_by_id.get(relation_id)
            if relation is None:
                raise ValueError(f"missing_uud_article_relation:{edge['edge_id']}")
            edge_id = edge["edge_id"]
            edge.clear()
            edge.update(
                {
                    "edge_id": edge_id,
                    "source_id": f"legal_unit::{relation['source_legal_unit_id']}",
                    "target_id": f"legal_unit::{relation['target_legal_unit_id']}",
                    "edge_type": edge_type,
                    "relation_type": relation["relation_type"],
                    "relation_id": relation_id,
                    "runtime_loadable": True,
                    "support_kind": "exact_source_relation",
                    "support_relation_ids": [relation_id],
                    "support_evidence_ids": list(relation.get("supporting_evidence_ids") or [relation.get("evidence_id")]),
                    "support_exception_ids": [],
                    "text_span_ids": list(relation.get("text_span_ids") or ()),
                    "bbox_refs": list(relation.get("bbox_refs") or ()),
                    "source_role": relation.get("source_role"),
                    "temporal_context": relation.get("source_role"),
                    "citation_final": False,
                }
            )
            continue
        for field in (
            "edge_authority_level",
            "evidence_requirement",
            "relation_support",
            "graph_finality_policy",
            "viewer_highlightable",
            "reason",
        ):
            edge.pop(field, None)
        if edge_type == "AMBIGUOUS_OPERATION":
            relation_id = str(edge.get("article_relation_ref") or "")
            relation = relations_by_id.get(relation_id)
            if relation is None:
                raise ValueError(f"missing_uud_article_relation:{edge['edge_id']}")
            edge["relation_id"] = relation_id
            edge["support_relation_ids"] = [relation_id]
        source = nodes_by_id[edge["source_id"]]
        target = nodes_by_id[edge["target_id"]]
        evidence_ids = _direct_evidence_ids(edge, source, target, evidence_by_id)
        supports = [evidence_by_id[evidence_id] for evidence_id in evidence_ids]
        if descriptor.relevance_eligible and not descriptor.authority_bearing:
            support_kind = "deterministic_structure"
            derivation_method = "deterministic_structural_rule"
            evidence_ids = []
            supports = []
        elif not descriptor.relevance_eligible and edge_type not in {"HAS_SOURCE_ANOMALY", "EXCLUDED_BECAUSE"}:
            support_kind = "endpoint_provenance"
            derivation_method = "endpoint_metadata"
        elif edge_type == "HAS_SOURCE_ANOMALY":
            support_kind = "source_anomaly_trace"
            derivation_method = "reviewed_corpus_spec"
        elif edge_type == "EXCLUDED_BECAUSE":
            support_kind = "nonlegal"
            derivation_method = "reviewed_corpus_spec"
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
                "source_role": (
                    source.get("source_role")
                    if descriptor.query_eligible
                    else _source_role_class(source.get("source_role") or target.get("source_role") or edge.get("source_role"))
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


def article_relation_runtime_policy_health(
    *,
    document_relations: list[dict] | tuple[dict, ...],
    article_amendment_relations: list[dict] | tuple[dict, ...],
    bbox_rows: list[dict],
    word_bboxes: list[dict],
    evidence: list[dict],
    legal_units: list[dict],
) -> dict:
    referenced_bbox_ids = {
        str(bbox_id)
        for row in article_amendment_relations
        for bbox_id in row.get("bbox_refs") or ()
    }
    bbox_by_id = {
        str(row.get("bbox_id")): row
        for row in bbox_rows
        if str(row.get("bbox_id")) in referenced_bbox_ids
    }
    for word in word_bboxes:
        word_id = str(word.get("word_bbox_id"))
        if word_id in referenced_bbox_ids:
            bbox_by_id[word_id] = word
        for character in word.get("characters") or ():
            character_id = str(character.get("character_bbox_id"))
            if character_id in referenced_bbox_ids:
                bbox_by_id[character_id] = character
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
        "relation_runtime_policy_slow_gate_status": "not_executed_in_offline_validation__run_public_runtime_policy_gate",
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
    invalid = (
        "article_relation_trace_missing_reason_count",
        "article_relation_invalid_bbox_refs",
        "article_relation_invalid_coordinates",
        "source_relation_missing_count",
        "source_relation_unexpected_count",
        "source_relation_duplicate_count",
    )
    return {**counts, "status": "complete" if not any(counts[key] for key in invalid) else "incomplete"}


def legal_graph_authority_health(
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
    exact_edges = [
        row
        for row in forward_relation_edges
        if relations_by_id.get(row.get("relation_id"), {}).get("support_class") == "exact_article_relation"
    ]
    trace_edges = [
        row
        for row in forward_relation_edges
        if relations_by_id.get(row.get("relation_id"), {}).get("support_class") != "exact_article_relation"
    ]
    authority_without_evidence = [
        row
        for row in relation_edges
        if row.get("support_evidence_ids") and any(evidence_id not in evidence_by_id for evidence_id in row["support_evidence_ids"])
    ]
    authority_without_bbox = [
        row
        for row in exact_edges
        if any(bbox_id not in bbox_ids for bbox_id in relations_by_id[row["relation_id"]].get("bbox_refs") or ())
    ]
    trace_promoted = [row for row in trace_edges if relations_by_id.get(row.get("relation_id"), {}).get("citation_final") is True]
    missing_fields = [
        row
        for row in relation_edges
        if not {"relation_id", "support_relation_ids", "support_evidence_ids", "support_kind"} <= set(row)
    ]
    endpoint_mismatches = [row for row in relation_edges if _relation_projection_endpoint_mismatch(row)]
    direction_mismatches = [
        row
        for row in relation_edges
        if (row.get("relation_projection") or {}).get("relation_type") != row.get("edge_type")
        or (row.get("derived_from_edge_id") and (row.get("relation_projection") or {}).get("projection_direction") != "inverse")
        or (not row.get("derived_from_edge_id") and (row.get("relation_projection") or {}).get("projection_direction") != "forward")
    ]
    invalid = (
        authority_without_evidence
        or authority_without_bbox
        or trace_promoted
        or missing_fields
        or endpoint_mismatches
        or direction_mismatches
        or any(row.get("citation_final") is True for row in graph_edges)
    )
    return {
        "status": "incomplete" if invalid else "complete",
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
            1 for row in endpoint_mismatches if not row.get("derived_from_edge_id")
        ),
        "inverse_relation_projection_endpoint_mismatch_count": sum(
            1 for row in endpoint_mismatches if row.get("derived_from_edge_id")
        ),
        "relation_direction_mismatch_count": len(direction_mismatches),
        "authority_kind_counts": dict(sorted(Counter(row.get("authority_kind") or "relation_reference" for row in graph_edges).items())),
        "support_kind_counts": dict(sorted(Counter(row.get("support_kind") or "relation_reference" for row in graph_edges).items())),
    }


def _source_relation_contract_errors(
    *, evidence: list[dict], legal_units: list[dict], article_amendment_relations: list[dict]
) -> tuple[str, ...]:
    evidence_ids = {str(row.get("evidence_id")) for row in evidence}
    unit_ids = {str(row.get("legal_unit_id")) for row in legal_units}
    relation_ids: set[str] = set()
    errors: list[str] = []
    for row in article_amendment_relations:
        relation_id = str(row.get("relation_id") or "")
        descriptor = descriptor_for(str(row.get("relation_type") or ""))
        if not relation_id or relation_id in relation_ids:
            errors.append(f"article_relation_duplicate_source_reference:{relation_id}")
        relation_ids.add(relation_id)
        if (
            not descriptor
            or not descriptor.provenance_required
            or row.get("evidence_id") not in evidence_ids
            or row.get("source_legal_unit_id") not in unit_ids
            or row.get("target_legal_unit_id") not in unit_ids
            or not row.get("target_citation")
        ):
            errors.append(f"article_relation_unexpected_source_reference:{relation_id}")
        if row.get("support_class") == "exact_article_relation" and not row.get("bbox_refs"):
            errors.append(f"article_relation_missing_source_reference:{relation_id}")
    return tuple(errors)


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
