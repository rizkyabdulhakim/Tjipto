from __future__ import annotations

ROUTE_WEIGHT = {"exact": 1000.0, "metadata": 900.0, "structured": 800.0, "bm25": 600.0, "graph": 120.0}


def merge_ranked(store, route_rows: dict[str, tuple[dict, ...]], filters: dict) -> tuple[tuple[dict, ...], tuple[dict, ...]]:
    rows_by_id: dict[str, dict] = {}
    for route, rows in route_rows.items():
        for order, row in enumerate(rows):
            _add(rows_by_id, store, row, route, order, None)
    trace = graph_expand(store, tuple(rows_by_id.values()), filters)
    for order, item in enumerate(trace):
        row = store.get(item["evidence_id"])
        if row is not None:
            _add(rows_by_id, store, row, "graph", order, item)
    ranked = sorted(rows_by_id.values(), key=lambda row: (-row["route_score"], row["evidence_id"]))
    return tuple(ranked), trace


def graph_expand(store, seeds: tuple[dict, ...], filters: dict, per_seed: int = 2) -> tuple[dict, ...]:
    if not seeds:
        return ()
    evidence_by_id = {row["evidence_id"]: row for row in store.evidence}
    neighbors = _neighbors(store.graph_edges)
    out = []
    seen = {row["evidence_id"] for row in seeds}
    for seed in seeds[:5]:
        if (
            "bm25" in set(seed.get("route_sources") or ())
            and seed.get("lexical_relevance_ok") is False
        ):
            continue
        seed_node = f"final_evidence::{seed['evidence_id']}"
        added = 0
        for node in sorted(neighbors.get(seed_node, ())):
            if not (node.startswith("legal_unit::") or node.startswith("page::")):
                continue
            for candidate_node in sorted(neighbors.get(node, ())):
                if not candidate_node.startswith("final_evidence::"):
                    continue
                evidence_id = candidate_node.split("::", 1)[1]
                row = evidence_by_id.get(evidence_id)
                if evidence_id in seen or not _usable(store, row, filters):
                    continue
                seen.add(evidence_id)
                added += 1
                out.append({
                    "evidence_id": evidence_id,
                    "from_evidence_id": seed["evidence_id"],
                    "via": node.split("::", 1)[0],
                    "reason": "shared_validated_graph_relation",
                })
                if added >= per_seed:
                    break
            if added >= per_seed:
                break
    return tuple(out)


def _neighbors(edges: list[dict]) -> dict[str, set[str]]:
    accepted = {"HAS_FINAL_EVIDENCE", "PAGE_GROUNDED_AT"}
    graph: dict[str, set[str]] = {}
    for edge in edges:
        if edge.get("edge_type") not in accepted:
            continue
        source, target = edge["source_id"], edge["target_id"]
        graph.setdefault(source, set()).add(target)
        graph.setdefault(target, set()).add(source)
    return graph


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
    score = (
        0.0
        if route == "bm25" and row.get("lexical_relevance_ok") is False
        else ROUTE_WEIGHT[route] - order
    )
    existing["route_scores"][route] = max(existing["route_scores"].get(route, 0.0), score)
    if trace:
        existing["expansion_trace"] = (*existing["expansion_trace"], trace)
    existing["route_score"] = sum(existing["route_scores"].values())
    if existing.get("source_role") == getattr(getattr(store, "config", None), "preferred_source_role", None):
        existing["route_score"] += 5.0
    existing["rank_reasons"] = tuple(sorted(existing["route_sources"])) + (existing["provenance_status"],)
