from __future__ import annotations

from collections import defaultdict

from tjipto.contracts.authority import authority_state_error
from tjipto.contracts.coordinates import COORDINATE_SPACE, TRANSFORM_VERSION
from tjipto.contracts.evidence import exact_quote_support_reason, source_lineage_reason
from tjipto.contracts.violations import Violation


AUTHORITY_FIELDS = (
    "authority_kind",
    "citable_status",
    "citable",
    "citation_final",
    "exactness",
    "evidence_exists",
    "reason_code",
    "citation_finality_reason",
)
COORDINATE_FIELDS = (
    "coordinate_space",
    "coordinate_origin",
    "page_width",
    "page_height",
    "page_rotation",
    "page_box_basis",
    "transform_version",
)
DERIVATION_METHODS = {
    "endpoint_metadata",
    "deterministic_structural_rule",
    "explicit_source_text",
    "reviewed_corpus_spec",
}


def validate_uud_trust_boundary(
    *,
    legal_units: list[dict],
    chunks: list[dict],
    graph_nodes: list[dict],
    graph_edges: list[dict],
    retrieval_units: list[dict],
    evidence: list[dict],
    bbox_rows: list[dict],
    page_text_spans: list[dict],
    source_documents: list[dict],
    pages: list[dict],
    word_bboxes: list[dict] | tuple[dict, ...] = (),
) -> list[Violation]:
    if retrieval_units and retrieval_units[0].get("object_role") == "retrieval_index_record":
        return _validate_schema6_trust_boundary(
            legal_units=legal_units,
            chunks=chunks,
            graph_nodes=graph_nodes,
            graph_edges=graph_edges,
            retrieval_units=retrieval_units,
            evidence=evidence,
            bbox_rows=bbox_rows,
            page_text_spans=page_text_spans,
            source_documents=source_documents,
        )
    violations: list[Violation] = []
    # Authority is owned by evidence/support records.  Index, geometry, span,
    # graph and legal-unit rows are projections and must not repeat the
    # decision fields merely to satisfy the legacy trust boundary.
    owner_collections = {
        "evidence_registry": (evidence, "evidence_id"),
    }
    for artifact, (rows, id_field) in owner_collections.items():
        for row in rows:
            row_id = str(row.get(id_field) or "<missing>")
            missing = [field for field in AUTHORITY_FIELDS if row.get(field) is None]
            if missing:
                violations.append(
                    _violation("AUTHORITY_MISSING", artifact, row_id, missing[0], "non-null", None, "authority decision incomplete")
                )
                continue
            expected_status = "citable_exact" if row["citable"] else "not_citable"
            if row["citable_status"] != expected_status:
                violations.append(
                    _violation(
                        "CITABLE_STATUS_CONFLICT",
                        artifact,
                        row_id,
                        "citable_status",
                        expected_status,
                        row["citable_status"],
                        "citable status contradicts boolean decision",
                    )
                )
            if row["citation_finality_reason"] != row["reason_code"]:
                violations.append(
                    _violation(
                        "FINALITY_REASON_CONFLICT",
                        artifact,
                        row_id,
                        "citation_finality_reason",
                        row["reason_code"],
                        row["citation_finality_reason"],
                        "finality reason contradicts authority reason",
                    )
                )
            error = authority_state_error(
                authority_kind=row["authority_kind"],
                citable=row["citable"],
                citation_final=row["citation_final"],
                exactness=row["exactness"],
                evidence_exists=row["evidence_exists"],
                reason_code=row["reason_code"],
            )
            if error:
                violations.append(_violation("AUTHORITY_STATE_CONTRADICTION", artifact, row_id, "authority", "allowed state", error, error))
    _validate_runtime_evidence_links(retrieval_units, chunks, evidence, page_text_spans, violations)
    _validate_coordinates(bbox_rows, violations)
    _validate_hierarchy(legal_units, chunks, graph_edges, violations)
    _validate_graph(graph_nodes, graph_edges, legal_units, evidence, bbox_rows, source_documents, pages, page_text_spans, violations)
    _validate_evidence_closure(evidence, bbox_rows, page_text_spans, source_documents, pages, violations, list(word_bboxes))
    return violations


def _validate_schema6_trust_boundary(
    *,
    legal_units: list[dict],
    chunks: list[dict],
    graph_nodes: list[dict],
    graph_edges: list[dict],
    retrieval_units: list[dict],
    evidence: list[dict],
    bbox_rows: list[dict],
    page_text_spans: list[dict],
    source_documents: list[dict],
) -> list[Violation]:
    violations: list[Violation] = []
    evidence_ids = {row.get("evidence_id") for row in evidence}
    source_ids = {row.get("source_document_id") for row in source_documents}
    source_pages = {row.get("source_document_id"): int(row.get("page_count") or 0) for row in source_documents}
    span_ids = {row.get("text_span_id") for row in page_text_spans}
    bbox_ids = {row.get("bbox_id") for row in bbox_rows}
    node_ids = {row.get("node_id") for row in graph_nodes}
    for row in evidence:
        row_id = str(row.get("evidence_id"))
        for field in AUTHORITY_FIELDS:
            if row.get(field) is None:
                violations.append(_violation("AUTHORITY_MISSING", "evidence_registry", row_id, field, "non-null", None, "authority decision incomplete"))
        if row.get("source_document_id") not in source_ids:
            violations.append(_violation("REFERENCE_UNRESOLVED_SOURCE", "evidence_registry", row_id, "source_document_id", "existing source", row.get("source_document_id"), "source missing"))
        if not isinstance(row.get("citation_final"), bool) or row.get("authority_kind") not in {"normative_legal_text", "instrument_provenance", "source_anomaly_trace", "structural_context"}:
            violations.append(_violation("AUTHORITY_STATE_CONTRADICTION", "evidence_registry", row_id, "authority", "valid authority/finality", row, "authority state invalid"))
        if any(page < 1 or page > source_pages.get(row.get("source_document_id"), 0) for page in row.get("page_numbers") or ()):
            violations.append(_violation("REFERENCE_UNRESOLVED_PAGE", "evidence_registry", row_id, "page_numbers", "existing page", row.get("page_numbers"), "page missing"))
        if any(ref not in span_ids for ref in row.get("text_span_ids") or ()):
            violations.append(_violation("REFERENCE_UNRESOLVED_SPAN", "evidence_registry", row_id, "text_span_ids", "existing span", row.get("text_span_ids"), "span missing"))
        if any(ref not in bbox_ids for ref in row.get("bbox_refs") or ()):
            violations.append(_violation("REFERENCE_UNRESOLVED_BBOX", "evidence_registry", row_id, "bbox_refs", "existing bbox", row.get("bbox_refs"), "bbox missing"))
    for row in bbox_rows:
        row_id = str(row.get("bbox_id"))
        if row.get("evidence_id") not in evidence_ids or row.get("evidence_exists") is not True:
            violations.append(_violation("REFERENCE_UNRESOLVED_EVIDENCE", "bbox_registry", row_id, "evidence_id", "existing evidence", row.get("evidence_id"), "bbox owner missing"))
        if row.get("viewer_highlightable") is True and any(row.get(field) is None for field in COORDINATE_FIELDS):
            violations.append(_violation("COORDINATE_METADATA_MISSING", "bbox_registry", row_id, "coordinates", "complete", None, "highlightable bbox requires coordinates"))
    for row in retrieval_units:
        if row.get("evidence_id") not in evidence_ids:
            violations.append(_violation("RETRIEVAL_EVIDENCE_UNRESOLVED", "retrieval_units", str(row.get("retrieval_unit_id")), "evidence_id", "existing evidence", row.get("evidence_id"), "retrieval evidence missing"))
    for row in chunks:
        for ref in row.get("evidence_ids") or ():
            if ref not in evidence_ids:
                violations.append(_violation("CHUNK_EVIDENCE_UNRESOLVED", "chunks", str(row.get("chunk_id")), "evidence_ids", "existing evidence", ref, "chunk evidence missing"))
    for row in graph_edges:
        if row.get("source_id") not in node_ids or row.get("target_id") not in node_ids:
            violations.append(_violation("GRAPH_EDGE_ENDPOINT_UNRESOLVED", "graph_edges", str(row.get("edge_id")), "endpoint", "existing graph node", row, "graph endpoint missing"))
        if row.get("object_role") != "graph_projection" or not all(field in row for field in ("support_relation_ids", "support_evidence_ids", "support_exception_ids", "support_kind")):
            violations.append(_violation("RELATION_SUPPORT_MISMATCH", "graph_edges", str(row.get("edge_id")), "support", "typed support", row, "typed graph support incomplete"))
    return violations


def _validate_runtime_evidence_links(
    retrieval_units: list[dict],
    chunks: list[dict],
    evidence: list[dict],
    spans: list[dict],
    violations: list[Violation],
) -> None:
    evidence_ids = {row["evidence_id"] for row in evidence}
    for row in retrieval_units:
        if row.get("evidence_id") not in evidence_ids:
            violations.append(
                _violation(
                    "RETRIEVAL_EVIDENCE_UNRESOLVED",
                    "retrieval_units",
                    row["retrieval_unit_id"],
                    "evidence_id",
                    "existing evidence",
                    row.get("evidence_id"),
                    "retrieval evidence missing",
                )
            )
    for row in chunks:
        for evidence_id in row.get("evidence_ids") or ():
            if evidence_id not in evidence_ids:
                violations.append(
                    _violation(
                        "CHUNK_EVIDENCE_UNRESOLVED",
                        "chunks",
                        row["chunk_id"],
                        "evidence_ids",
                        "existing evidence",
                        evidence_id,
                        "chunk evidence missing",
                    )
                )
    # Page spans and retrieval units are projections in schema 6.  Their
    # authority/finality is dereferenced from the evidence owner rather than
    # copied into the projection, so no legacy trace decision is validated here.


def _validate_coordinates(rows: list[dict], violations: list[Violation]) -> None:
    for row in rows:
        if row.get("viewer_highlightable") is not True:
            continue
        row_id = row["bbox_id"]
        missing = [field for field in COORDINATE_FIELDS if row.get(field) is None]
        if missing:
            violations.append(
                _violation(
                    "COORDINATE_METADATA_MISSING",
                    "bbox_registry",
                    row_id,
                    missing[0],
                    "non-null",
                    None,
                    "highlightable bbox requires coordinates",
                )
            )
            continue
        valid = (
            row["coordinate_space"] == COORDINATE_SPACE
            and row["coordinate_origin"] == "top_left"
            and row["page_rotation"] == 0
            and row["page_box_basis"] == "media_box"
            and row["transform_version"] == TRANSFORM_VERSION
            and all(isinstance(row[field], (int, float)) for field in ("x0", "y0", "x1", "y1", "page_width", "page_height"))
            and 0 <= row["x0"] < row["x1"] <= row["page_width"]
            and 0 <= row["y0"] < row["y1"] <= row["page_height"]
        )
        if not valid:
            violations.append(
                _violation(
                    "COORDINATE_METADATA_INVALID",
                    "bbox_registry",
                    row_id,
                    "coordinates",
                    "bounded rotation-0 media box",
                    {field: row.get(field) for field in COORDINATE_FIELDS},
                    "coordinate contract invalid",
                )
            )


def _validate_hierarchy(units: list[dict], chunks: list[dict], edges: list[dict], violations: list[Violation]) -> None:
    by_id = {row["legal_unit_id"]: row for row in units}
    groups: dict[tuple[str, str | None], list[dict]] = defaultdict(list)
    graph_relations = {(row["edge_type"], row["source_id"], row["target_id"]) for row in edges}
    for row in units:
        row_id = row["legal_unit_id"]
        if row.get("stable_unit_id") != row_id:
            violations.append(
                _violation(
                    "STABLE_UNIT_ID_MISMATCH",
                    "legal_units",
                    row_id,
                    "stable_unit_id",
                    row_id,
                    row.get("stable_unit_id"),
                    "stable identity changed",
                )
            )
        parent = row.get("parent_legal_unit_id")
        if parent and parent not in by_id:
            violations.append(
                _violation("PARENT_UNRESOLVED", "legal_units", row_id, "parent_legal_unit_id", "existing unit", parent, "parent missing")
            )
        expected = _ancestor_path(row_id, by_id)
        if expected is not None and row.get("ancestor_legal_unit_ids") != expected:
            violations.append(
                _violation(
                    "ANCESTOR_PATH_INCORRECT",
                    "legal_units",
                    row_id,
                    "ancestor_legal_unit_ids",
                    expected,
                    row.get("ancestor_legal_unit_ids"),
                    "ancestor path mismatch",
                )
            )
        if parent in by_id:
            parent_node, child_node = f"legal_unit::{parent}", f"legal_unit::{row_id}"
            if ("CONTAINS", parent_node, child_node) not in graph_relations or ("PART_OF", child_node, parent_node) not in graph_relations:
                violations.append(
                    _violation(
                        "HIERARCHY_GRAPH_MISMATCH",
                        "legal_units",
                        row_id,
                        "parent_legal_unit_id",
                        "reciprocal graph relations",
                        parent,
                        "parent and graph disagree",
                    )
                )
        groups[(row["source_document_id"], parent)].append(row)
    for (source_id, parent), siblings in groups.items():
        raw_orders = [row.get("sibling_order") for row in siblings]
        orders = sorted(value for value in raw_orders if isinstance(value, int))
        group_id = f"{source_id}:{parent or '<root>'}"
        if len(set(orders)) != len(orders):
            violations.append(
                _violation("SIBLING_ORDER_DUPLICATE", "legal_units", group_id, "sibling_order", "unique", orders, "duplicate sibling order")
            )
        if len(orders) != len(raw_orders) or orders != list(range(1, len(orders) + 1)):
            violations.append(
                _violation(
                    "SIBLING_ORDER_NONCONTIGUOUS",
                    "legal_units",
                    group_id,
                    "sibling_order",
                    list(range(1, len(orders) + 1)),
                    raw_orders,
                    "sibling order gap",
                )
            )
    for row in chunks:
        children = row.get("contributing_child_legal_unit_ids") or []
        if not children or any(child not in by_id for child in children):
            violations.append(
                _violation(
                    "CONTRIBUTING_CHILD_MISSING",
                    "chunks",
                    row["chunk_id"],
                    "contributing_child_legal_unit_ids",
                    "existing units",
                    children,
                    "chunk child closure incomplete",
                )
            )


def _ancestor_path(unit_id: str, by_id: dict[str, dict]) -> list[str] | None:
    path: list[str] = []
    seen = {unit_id}
    current = by_id[unit_id].get("parent_legal_unit_id")
    while current:
        if current in seen or current not in by_id:
            return None
        seen.add(current)
        path.append(current)
        current = by_id[current].get("parent_legal_unit_id")
    return list(reversed(path))


def _validate_graph(
    nodes: list[dict],
    edges: list[dict],
    units: list[dict],
    evidence: list[dict],
    bboxes: list[dict],
    sources: list[dict],
    pages: list[dict],
    spans: list[dict],
    violations: list[Violation],
) -> None:
    node_ids = {row["node_id"] for row in nodes}
    unit_ids = {row["legal_unit_id"] for row in units}
    evidence_ids = {row["evidence_id"] for row in evidence}
    bbox_ids = {row["bbox_id"] for row in bboxes}
    source_ids = {row["source_document_id"] for row in sources}
    page_keys = {(row["source_document_id"], row["page_number"]) for row in pages}
    span_ids = {row["text_span_id"] for row in spans}
    degree: set[str] = set()
    for node in nodes:
        if node.get("node_type") == "legal_unit" and node.get("legal_unit_id") not in unit_ids:
            violations.append(
                _violation(
                    "GRAPH_NODE_LEGAL_UNIT_UNRESOLVED",
                    "graph_nodes",
                    node["node_id"],
                    "legal_unit_id",
                    "existing unit",
                    node.get("legal_unit_id"),
                    "node unit missing",
                )
            )
    for edge in edges:
        row_id = edge["edge_id"]
        for field in ("source_id", "target_id"):
            endpoint = edge.get(field)
            if endpoint not in node_ids:
                violations.append(
                    _violation(
                        "GRAPH_EDGE_ENDPOINT_UNRESOLVED", "graph_edges", row_id, field, "existing node", endpoint, "edge endpoint missing"
                    )
                )
            elif isinstance(endpoint, str):
                degree.add(endpoint)
        if edge.get("relation_id"):
            continue
        if edge.get("derivation_method") not in DERIVATION_METHODS:
            violations.append(
                _violation(
                    "DERIVATION_METHOD_UNKNOWN",
                    "graph_edges",
                    row_id,
                    "derivation_method",
                    sorted(DERIVATION_METHODS),
                    edge.get("derivation_method"),
                    "derivation fails closed",
                )
            )
        _validate_edge_refs(edge, evidence_ids, bbox_ids, source_ids, page_keys, span_ids, violations)
        _validate_relation_support(edge, violations)
    for node in nodes:
        if node["node_id"] not in degree and node.get("runtime_loadable") is not False:
            violations.append(
                _violation(
                    "GRAPH_NODE_ORPHAN",
                    "graph_nodes",
                    node["node_id"],
                    "runtime_loadable",
                    False,
                    node.get("runtime_loadable"),
                    "runtime graph node has no edge",
                )
            )


def _validate_edge_refs(
    edge: dict,
    evidence_ids: set[str],
    bbox_ids: set[str],
    source_ids: set[str],
    page_keys: set[tuple[str, int]],
    span_ids: set[str],
    violations: list[Violation],
) -> None:
    row_id = edge["edge_id"]
    checks = (
        ("supporting_evidence_ids", evidence_ids, "REFERENCE_UNRESOLVED_EVIDENCE"),
        ("bbox_refs", bbox_ids, "REFERENCE_UNRESOLVED_BBOX"),
        ("source_document_ids", source_ids, "REFERENCE_UNRESOLVED_SOURCE"),
        ("text_span_ids", span_ids, "REFERENCE_UNRESOLVED_SPAN"),
    )
    for field, known, code in checks:
        for value in edge.get(field) or ():
            if value not in known:
                violations.append(_violation(code, "graph_edges", row_id, field, "resolvable", value, "graph provenance reference missing"))
    if edge.get("page_numbers") and not any(
        (source, page) in page_keys for source in edge.get("source_document_ids") or () for page in edge["page_numbers"]
    ):
        violations.append(
            _violation(
                "REFERENCE_UNRESOLVED_PAGE",
                "graph_edges",
                row_id,
                "page_numbers",
                "source/page pair",
                edge["page_numbers"],
                "graph page missing",
            )
        )


def _validate_relation_support(edge: dict, violations: list[Violation]) -> None:
    kind, method, evidence_ids = edge.get("support_kind"), edge.get("derivation_method"), edge.get("supporting_evidence_ids") or []
    valid = True
    if kind == "exact_source_relation":
        valid = method == "explicit_source_text" and bool(evidence_ids) and bool(edge.get("text_span_ids")) and bool(edge.get("bbox_refs"))
    elif kind == "deterministic_structure":
        valid = method == "deterministic_structural_rule" and not evidence_ids
    elif kind == "endpoint_provenance":
        valid = method == "endpoint_metadata" and bool(evidence_ids)
    elif kind == "instrument_provenance":
        valid = method in {"explicit_source_text", "reviewed_corpus_spec"}
    elif kind == "source_anomaly_trace":
        valid = method == "reviewed_corpus_spec"
    elif kind == "nonlegal":
        valid = method == "reviewed_corpus_spec"
    else:
        valid = False
    if not valid:
        violations.append(
            _violation(
                "RELATION_SUPPORT_MISMATCH",
                "graph_edges",
                edge["edge_id"],
                "support_kind",
                "semantically matched derivation",
                {"support_kind": kind, "derivation_method": method, "evidence_ids": evidence_ids},
                "relation support is broad or contradictory",
            )
        )


def _validate_evidence_closure(
    evidence: list[dict],
    bboxes: list[dict],
    spans: list[dict],
    sources: list[dict],
    pages: list[dict],
    violations: list[Violation],
    word_bboxes: list[dict],
) -> None:
    evidence_by_id = {row["evidence_id"]: row for row in evidence}
    bbox_by_id = {row["bbox_id"]: row for row in bboxes} | {row["word_bbox_id"]: row for row in word_bboxes}
    bbox_by_id.update(
        {
            character["character_bbox_id"]: {**word, **character, "bbox_id": character["character_bbox_id"]}
            for word in word_bboxes
            for character in word.get("characters") or ()
        }
    )
    bbox_ids = set(bbox_by_id)
    span_ids = {row["text_span_id"] for row in spans}
    source_ids = {row["source_document_id"] for row in sources}
    sources_by_id = {row["source_document_id"]: row for row in sources}
    page_keys = {(row["source_document_id"], row["page_number"]) for row in pages}
    for row in evidence:
        row_id = row["evidence_id"]
        for field, known, code in (
            ("bbox_refs", bbox_ids, "REFERENCE_UNRESOLVED_BBOX"),
            ("text_span_ids", span_ids, "REFERENCE_UNRESOLVED_SPAN"),
        ):
            for value in row.get(field) or ():
                if value not in known:
                    violations.append(
                        _violation(code, "evidence_registry", row_id, field, "resolvable", value, "evidence reference missing")
                    )
        if row.get("source_document_id") not in source_ids:
            violations.append(
                _violation(
                    "REFERENCE_UNRESOLVED_SOURCE",
                    "evidence_registry",
                    row_id,
                    "source_document_id",
                    "existing source",
                    row.get("source_document_id"),
                    "evidence source missing",
                )
            )
        for page in row.get("page_numbers") or ():
            if (row.get("source_document_id"), page) not in page_keys:
                violations.append(
                    _violation(
                        "REFERENCE_UNRESOLVED_PAGE",
                        "evidence_registry",
                        row_id,
                        "page_numbers",
                        "existing page",
                        page,
                        "evidence page missing",
                    )
                )
        if (
            row.get("exactness") == "exact"
            or row.get("citable") is True
            or row.get("citation_final") is True
            or row.get("viewer_highlightable") is True
        ):
            quote_error = exact_quote_support_reason(
                quoted_text=row.get("quoted_text"),
                source_document_id=row.get("source_document_id"),
                page_numbers=row.get("page_numbers") or (),
                text_span_ids=row.get("text_span_ids") or (),
                bbox_refs=row.get("bbox_refs") or (),
                spans_by_id={item["text_span_id"]: item for item in spans},
                bboxes_by_id=bbox_by_id,
            )
            if quote_error:
                violations.append(
                    _violation(
                        "EVIDENCE_QUOTE_SOURCE_MISMATCH",
                        "evidence_registry",
                        row_id,
                        "quoted_text",
                        "accepted source span and exact BBox text",
                        quote_error,
                        "exact evidence quote is not source-faithful",
                    )
                )
            lineage_error = source_lineage_reason(
                evidence=row,
                source_documents_by_id=sources_by_id,
                spans_by_id={item["text_span_id"]: item for item in spans},
                bboxes_by_id=bbox_by_id,
            )
            if lineage_error:
                violations.append(
                    _violation(
                        "EVIDENCE_SOURCE_LINEAGE_INVALID",
                        "evidence_registry",
                        row_id,
                        "source_lineage",
                        "source-faithful evidence",
                        lineage_error,
                        "evidence lineage does not resolve to its source document",
                    )
                )
    for span in spans:
        row_id = span["text_span_id"]
        for evidence_id in span.get("evidence_ids") or ():
            target = evidence_by_id.get(evidence_id)
            if target is None:
                violations.append(
                    _violation(
                        "REFERENCE_UNRESOLVED_EVIDENCE",
                        "page_text_spans",
                        row_id,
                        "evidence_ids",
                        "existing evidence",
                        evidence_id,
                        "span evidence missing",
                    )
                )
            elif row_id not in (target.get("text_span_ids") or ()):
                violations.append(
                    _violation(
                        "SPAN_EVIDENCE_REVERSE_CLOSURE",
                        "page_text_spans",
                        row_id,
                        "evidence_ids",
                        "evidence references span",
                        evidence_id,
                        "reverse span closure missing",
                    )
                )
        for bbox_id in span.get("span_bbox_ids") or ():
            bbox = bbox_by_id.get(bbox_id)
            if bbox is None:
                violations.append(
                    _violation(
                        "REFERENCE_UNRESOLVED_BBOX",
                        "page_text_spans",
                        row_id,
                        "span_bbox_ids",
                        "existing bbox",
                        bbox_id,
                        "span bbox missing",
                    )
                )
        for bbox_id in span.get("context_bbox_ids") or ():
            bbox = bbox_by_id.get(bbox_id)
            if bbox is None:
                violations.append(
                    _violation(
                        "REFERENCE_UNRESOLVED_BBOX",
                        "page_text_spans",
                        row_id,
                        "context_bbox_ids",
                        "existing bbox",
                        bbox_id,
                        "context bbox missing",
                    )
                )
            elif bbox_id in (span.get("span_bbox_ids") or ()) or bbox_id in (span.get("evidence_bbox_ids") or ()):
                violations.append(
                    _violation(
                        "CONTEXT_BBOX_OVERLAP",
                        "page_text_spans",
                        row_id,
                        "context_bbox_ids",
                        "context-only bbox references",
                        bbox_id,
                        "context bbox cannot be quote geometry",
                    )
                )
            elif bbox.get("source_document_id") != span.get("source_document_id") or bbox.get("page_number") != span.get("page_number"):
                violations.append(
                    _violation(
                        "CONTEXT_BBOX_PROVENANCE_MISMATCH",
                        "page_text_spans",
                        row_id,
                        "context_bbox_ids",
                        "same source and page",
                        bbox_id,
                        "context bbox provenance mismatch",
                    )
                )
            elif (not _intersects(span, bbox) and not str(bbox_id).startswith("uud_unified_bbox::")) or any(
                span.get(field) != bbox.get(field) for field in ("source_document_id", "page_number")
            ):
                violations.append(
                    _violation(
                        "SPAN_BBOX_NONOVERLAP",
                        "page_text_spans",
                        row_id,
                        "span_bbox_ids",
                        "intersecting source/page bbox",
                        bbox_id,
                        "span-local bbox is not quote geometry",
                    )
                )


def _intersects(left: dict, right: dict) -> bool:
    return min(left["x1"], right["x1"]) > max(left["x0"], right["x0"]) and min(left["y1"], right["y1"]) > max(left["y0"], right["y0"])


def _violation(code: str, artifact: str, row_id: str, field: str, expected: object, actual: object, reason: str) -> Violation:
    return Violation(code, "error", artifact, row_id, field, expected, actual, reason)
