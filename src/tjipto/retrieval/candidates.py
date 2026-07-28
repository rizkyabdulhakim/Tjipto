from __future__ import annotations

ROUTE_WEIGHT = {
    "exact": 1000.0,
    "metadata": 900.0,
    "relation": 850.0,
    "structured": 800.0,
    "bm25": 600.0,
    "graph": 120.0,
}
CANDIDATE_TYPE = {
    "exact": "legal_unit_candidate",
    "structured": "legal_unit_candidate",
    "metadata": "metadata_candidate",
    "relation": "relation_candidate",
    "bm25": "lexical_candidate",
    "graph": "graph_candidate",
}


def merge_ranked(
    store,
    route_rows: dict[str, tuple[dict, ...]],
    filters: dict,
    *,
    expand_graph: bool = True,
    semantic: bool = False,
) -> tuple[tuple[dict, ...], tuple[dict, ...]]:
    rows_by_id: dict[str, dict] = {}
    for route, rows in route_rows.items():
        for order, row in enumerate(rows):
            _add(rows_by_id, store, row, route, order, None)
    trace = graph_expand(store, tuple(rows_by_id.values()), filters, semantic=semantic) if expand_graph else ()
    for order, item in enumerate(trace):
        row = store.get(item["evidence_id"])
        if row is not None:
            _add(rows_by_id, store, row, "graph", order, item)
    ranked = sorted(rows_by_id.values(), key=lambda row: (-row["route_score"], row["evidence_id"]))
    return tuple(ranked), trace


def graph_expand(store, seeds: tuple[dict, ...], filters: dict, per_seed: int = 2, *, semantic: bool = False) -> tuple[dict, ...]:
    if not seeds:
        return ()
    evidence_by_id = {row["evidence_id"]: row for row in store.evidence}
    neighbors = _neighbors(store, semantic=semantic)
    out = []
    seen = {row["evidence_id"] for row in seeds}
    for seed in seeds[:5]:
        if "bm25" in set(seed.get("route_sources") or ()) and seed.get("lexical_relevance_ok") is False:
            continue
        seed_node = f"final_evidence::{seed['evidence_id']}"
        added = 0
        queue: list[tuple[str, tuple[dict, ...], tuple[str, ...]]] = [(seed_node, (), (seed_node,))]
        while queue and added < per_seed:
            node, path, nodes = queue.pop(0)
            if len(path) >= 3:
                continue
            for edge in neighbors.get(node, ()):
                target = edge["target_id"]
                if target in nodes:
                    continue
                next_path = (*path, edge)
                if target.startswith("final_evidence::") and target != seed_node:
                    evidence_id = target.split("::", 1)[1]
                    row = evidence_by_id.get(evidence_id)
                    crosses_source = any(item["edge_type"] in _AUTHORITY_RELATION_EDGE_TYPES for item in next_path)
                    if evidence_id in seen or not _usable(store, row, filters):
                        continue
                    if crosses_source and row is not None and row.get("source_role") != seed.get("source_role"):
                        continue
                    seen.add(evidence_id)
                    added += 1
                    out.append(
                        {
                            "evidence_id": evidence_id,
                            "from_evidence_id": seed["evidence_id"],
                            "via": next_path[-1]["edge_type"],
                            "edge_types": tuple(item["edge_type"] for item in next_path),
                            "edge_ids": tuple(item["edge_id"] for item in next_path if item.get("edge_id")),
                            "relation_ids": tuple(item["relation_id"] for item in next_path if item.get("relation_id")),
                            "reason": "validated_semantic_graph_relation" if crosses_source else "shared_validated_graph_relation",
                        }
                    )
                    continue
                queue.append((target, next_path, (*nodes, target)))
    return tuple(out)


_SEMANTIC_EDGE_TYPES = {
    "CONTAINS", "PART_OF", "MODIFIES", "RENAMES", "RENUMBERED_TO", "DELETES", "INSERTS",
    "HAS_FINAL_EVIDENCE", "PAGE_GROUNDED_AT",
}
_AUTHORITY_RELATION_EDGE_TYPES = _SEMANTIC_EDGE_TYPES - {"HAS_FINAL_EVIDENCE", "PAGE_GROUNDED_AT"}


def _neighbors(store, *, semantic: bool) -> dict[str, tuple[dict, ...]]:
    graph: dict[str, list[dict]] = {}
    for edge in store.graph_edges:
        edge_type = edge.get("edge_type")
        if edge_type not in {"HAS_FINAL_EVIDENCE", "PAGE_GROUNDED_AT"} and not (
            semantic and edge_type in _SEMANTIC_EDGE_TYPES and edge.get("runtime_loadable") is True
        ):
            continue
        _connect(graph, edge)
    if semantic:
        for relation in getattr(store, "article_amendment_relations", ()):
            if not _valid_relation(store, relation):
                continue
            _connect(
                graph,
                {
                    "edge_id": f"relation::{relation['relation_id']}",
                    "source_id": f"legal_unit::{relation['source_legal_unit_id']}",
                    "target_id": f"legal_unit::{relation['target_legal_unit_id']}",
                    "edge_type": relation["relation_type"],
                    "relation_id": relation["relation_id"],
                },
            )
    return {node: tuple(sorted(items, key=lambda item: (item["target_id"], item["edge_type"], item.get("edge_id", "")))) for node, items in graph.items()}


def _connect(graph: dict[str, list[dict]], edge: dict) -> None:
    source, target = edge["source_id"], edge["target_id"]
    graph.setdefault(source, []).append(edge | {"target_id": target})
    graph.setdefault(target, []).append(edge | {"source_id": target, "target_id": source})


def _valid_relation(store, relation: dict) -> bool:
    evidence_id = relation.get("evidence_id")
    evidence = store.get(evidence_id) if evidence_id else None
    return (
        relation.get("runtime_loadable") is True
        and relation.get("validator_status") == "valid"
        and relation.get("relation_type") in _SEMANTIC_EDGE_TYPES
        and bool(relation.get("bbox_refs"))
        and evidence is not None
        and evidence.get("status") == "final"
        and bool(store.bboxes_for(evidence_id))
    )


def _usable(store, row: dict | None, filters: dict) -> bool:
    if not row or row.get("status") != "final" or not store.bboxes_for(row["evidence_id"]):
        return False
    if "source_role" in filters and row.get("source_role") != filters["source_role"]:
        return False
    return "temporal_context" not in filters or row.get("temporal_context") == filters["temporal_context"]


def _add(rows_by_id: dict[str, dict], store, row: dict, route: str, order: int, trace: dict | None) -> None:
    evidence_id = row["evidence_id"]
    existing = rows_by_id.get(evidence_id)
    if existing is None:
        existing = dict(row)
        existing["bbox_count"] = row.get("bbox_count", len(store.bboxes_for(evidence_id)))
        existing["provenance_status"] = "pass" if row.get("status") == "final" and existing["bbox_count"] else "needs_review"
        existing["route_sources"] = ()
        existing["route_scores"] = {}
        existing["rank_reasons"] = ()
        existing["expansion_trace"] = ()
        rows_by_id[evidence_id] = existing
    if route not in existing["route_sources"]:
        existing["route_sources"] = (*existing["route_sources"], route)
    if "candidate_type" not in existing or route == "graph":
        existing["candidate_type"] = row.get("candidate_type") or CANDIDATE_TYPE.get(
            route, existing.get("candidate_type", "legal_unit_candidate")
        )
    score = 0.0 if route == "bm25" and row.get("lexical_relevance_ok") is False else ROUTE_WEIGHT[route] - order
    existing["route_scores"][route] = max(existing["route_scores"].get(route, 0.0), score)
    if trace:
        existing["expansion_trace"] = (*existing["expansion_trace"], trace)
    existing["route_score"] = sum(existing["route_scores"].values())
    if existing.get("source_role") == getattr(getattr(store, "config", None), "preferred_source_role", None):
        existing["route_score"] += 5.0
    existing["rank_reasons"] = tuple(sorted(existing["route_sources"])) + (existing["provenance_status"],)
