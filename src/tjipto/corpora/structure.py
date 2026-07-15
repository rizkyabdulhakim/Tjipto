from __future__ import annotations


def apply_structural_contract(units: list[dict], *, role_by_unit_type: dict[str, str]) -> None:
    """Attach a deterministic, single-parent structural view to source-derived units."""
    by_id = {row["legal_unit_id"]: row for row in units}
    for row in units:
        row["stable_unit_id"] = row["legal_unit_id"]
        row["structural_role"] = role_by_unit_type.get(row.get("unit_type"), "metadata")
        row["canonical_label"] = row.get("unit_label")
        row["historical_label"] = row.get("unit_label") if row.get("source_role") != "current_consolidated" else None
    for row in units:
        parents = [parent for parent in row.get("parent_legal_unit_ids") or () if parent in by_id]
        parent_id = _canonical_parent(row, parents, by_id)
        row["parent_legal_unit_ids"] = [parent_id] if parent_id else []
        row["parent_legal_unit_id"] = parent_id

    children: dict[str | None, list[dict]] = {}
    for row in units:
        children.setdefault(row.get("parent_legal_unit_id"), []).append(row)
    for siblings in children.values():
        siblings.sort(key=lambda row: (row["source_document_id"], _unit_number(row["legal_unit_id"])))
        for order, row in enumerate(siblings, start=1):
            row["sibling_order"] = order

    for row in units:
        ancestors: list[str] = []
        current = row.get("parent_legal_unit_id")
        seen = {row["legal_unit_id"]}
        while current:
            if current in seen:
                raise ValueError(f"structural_cycle:{row['legal_unit_id']}")
            seen.add(current)
            ancestors.append(current)
            current = by_id[current].get("parent_legal_unit_id")
        row["ancestor_legal_unit_ids"] = list(reversed(ancestors))
        row["structural_depth"] = len(ancestors)


def apply_chunk_structural_contract(chunks: list[dict], legal_units: list[dict]) -> None:
    units = {row["legal_unit_id"]: row for row in legal_units}
    children: dict[str, list[str]] = {}
    for row in legal_units:
        parent = row.get("parent_legal_unit_id")
        if parent:
            children.setdefault(parent, []).append(row["legal_unit_id"])
    for chunk in chunks:
        unit = units[chunk["legal_unit_id"]]
        for field in (
            "stable_unit_id",
            "parent_legal_unit_id",
            "ancestor_legal_unit_ids",
            "structural_depth",
            "sibling_order",
            "structural_role",
            "canonical_label",
            "historical_label",
        ):
            chunk[field] = unit.get(field)
        chunk["canonical_unit_ref"] = unit["legal_unit_id"]
        chunk["contributing_child_legal_unit_ids"] = _descendant_leaves(unit["legal_unit_id"], children)
        chunk["authority_kind"] = "structural_context" if chunk.get("status") == "parent_context_only" else "legal_text"
        chunk["citable_status"] = "context_only" if chunk["authority_kind"] == "structural_context" else "exact_evidence_required"
        chunk["citation_final"] = False
        chunk["citation_finality_reason"] = (
            "parent_context_requires_child_evidence" if chunk["authority_kind"] == "structural_context" else "resolve_exact_evidence"
        )
    for unit in legal_units:
        unit["source_text_span_ids"] = list(unit.get("text_span_ids") or ())
        unit["source_bbox_refs"] = list(unit.get("bbox_ids") or ())
        unit["authority_kind"] = "legal_text" if unit.get("runtime_loadable") else "structural_or_trace"
        unit["citable_status"] = "exact_evidence_required" if unit.get("runtime_loadable") else "not_citable"
        unit["citation_final"] = False
        unit["citation_finality_reason"] = "resolve_exact_evidence"


def _canonical_parent(row: dict, parents: list[str], by_id: dict[str, dict]) -> str | None:
    preferred = {"subprovision": "provision", "item": "subprovision"}
    wanted = preferred.get(row.get("structural_role"))
    matched = next((parent for parent in parents if by_id[parent].get("structural_role") == wanted), None)
    if matched:
        return matched
    if wanted:
        labels = row.get("hierarchy") or ()
        parent_label = labels[-1] if labels else None
        inferred = [
            candidate["legal_unit_id"]
            for candidate in by_id.values()
            if candidate.get("source_document_id") == row.get("source_document_id")
            and candidate.get("structural_role") == wanted
            and candidate.get("unit_label") == parent_label
        ]
        if len(inferred) == 1:
            return inferred[0]
    return parents[-1] if parents else None


def _descendant_leaves(unit_id: str, children: dict[str, list[str]]) -> list[str]:
    direct = children.get(unit_id, [])
    if not direct:
        return [unit_id]
    leaves: list[str] = []
    for child in direct:
        leaves.extend(_descendant_leaves(child, children))
    return leaves


def _unit_number(value: str) -> int:
    try:
        return int(value.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return 0
