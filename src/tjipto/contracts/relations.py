"""Code-owned semantic relation descriptors used by build and runtime."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256


@dataclass(frozen=True)
class RelationDescriptor:
    inverse: str | None
    source_types: frozenset[str]
    target_types: frozenset[str]
    relevance_eligible: bool
    authority_bearing: bool
    provenance_required: bool
    query_eligible: bool = False


_LEGAL_UNIT = frozenset({"legal_unit"})
_SOURCE_ROLE = frozenset({"source_role"})

RELATIONS: dict[str, RelationDescriptor] = {
    "CONTAINS": RelationDescriptor("PART_OF", _LEGAL_UNIT, _LEGAL_UNIT, True, False, False),
    "PART_OF": RelationDescriptor("CONTAINS", _LEGAL_UNIT, _LEGAL_UNIT, True, False, False),
    "PRECEDES": RelationDescriptor("FOLLOWS", _LEGAL_UNIT, _LEGAL_UNIT, True, False, False),
    "FOLLOWS": RelationDescriptor("PRECEDES", _LEGAL_UNIT, _LEGAL_UNIT, True, False, False),
    "INSERTED_AFTER": RelationDescriptor(None, _LEGAL_UNIT, _LEGAL_UNIT, True, False, False),
    "MODIFIES": RelationDescriptor("MODIFIED_BY", _LEGAL_UNIT, _LEGAL_UNIT, True, True, True, True),
    "MODIFIED_BY": RelationDescriptor("MODIFIES", _LEGAL_UNIT, _LEGAL_UNIT, True, True, True, True),
    "RENAMES": RelationDescriptor("RENAMED_FROM", _LEGAL_UNIT, _LEGAL_UNIT, True, True, True, True),
    "RENAMED_FROM": RelationDescriptor("RENAMES", _LEGAL_UNIT, _LEGAL_UNIT, True, True, True, True),
    "RENUMBERED_TO": RelationDescriptor("RENUMBERED_FROM", _LEGAL_UNIT, _LEGAL_UNIT, True, True, True, True),
    "RENUMBERED_FROM": RelationDescriptor("RENUMBERED_TO", _LEGAL_UNIT, _LEGAL_UNIT, True, True, True, True),
    "DELETES": RelationDescriptor("DELETED_BY", _LEGAL_UNIT, _LEGAL_UNIT, True, True, True, True),
    "DELETED_BY": RelationDescriptor("DELETES", _LEGAL_UNIT, _LEGAL_UNIT, True, True, True, True),
    "ADDS": RelationDescriptor("INSERTED_BY", _LEGAL_UNIT, _LEGAL_UNIT, True, True, True, True),
    "INSERTED_BY": RelationDescriptor("ADDS", _LEGAL_UNIT, _LEGAL_UNIT, True, True, True, True),
    "AMENDS": RelationDescriptor("AMENDED_BY", _SOURCE_ROLE, _SOURCE_ROLE, False, False, True, True),
    "AMENDED_BY": RelationDescriptor("AMENDS", _SOURCE_ROLE, _SOURCE_ROLE, False, False, True, True),
    "DERIVED_FROM": RelationDescriptor("DERIVES", _SOURCE_ROLE, _SOURCE_ROLE, False, False, True, True),
    "DERIVES": RelationDescriptor("DERIVED_FROM", _SOURCE_ROLE, _SOURCE_ROLE, False, False, True, True),
    "CONSOLIDATES": RelationDescriptor("CONSOLIDATED_BY", _SOURCE_ROLE, _SOURCE_ROLE, False, False, True, True),
    "CONSOLIDATED_BY": RelationDescriptor("CONSOLIDATES", _SOURCE_ROLE, _SOURCE_ROLE, False, False, True, True),
    "HAS_SOURCE_ANOMALY": RelationDescriptor(None, frozenset({"source_role"}), frozenset({"source_conflict"}), False, False, True),
    "HAS_SIGNATORY": RelationDescriptor(None, frozenset({"source_role"}), _LEGAL_UNIT, False, False, True),
    "HAS_DECISION_SESSION": RelationDescriptor(None, frozenset({"source_role"}), _LEGAL_UNIT, False, False, True),
    "HAS_EFFECTIVE_RULE": RelationDescriptor(None, frozenset({"source_role"}), _LEGAL_UNIT, False, False, True),
    "EXCLUDED_BECAUSE": RelationDescriptor(None, frozenset({"excluded_record"}), frozenset({"source_role"}), False, False, False),
    "BELONGS_TO_SOURCE_ROLE": RelationDescriptor(None, frozenset({"final_evidence"}), frozenset({"source_role"}), False, False, True),
    "HAS_FINAL_EVIDENCE": RelationDescriptor(None, _LEGAL_UNIT, frozenset({"final_evidence"}), False, False, True),
    "PAGE_GROUNDED_AT": RelationDescriptor(None, frozenset({"final_evidence"}), frozenset({"page"}), False, False, True),
    "HAS_BBOX": RelationDescriptor(None, frozenset({"final_evidence"}), frozenset({"bbox"}), False, False, True),
    "USES_SOURCE_PDF": RelationDescriptor(None, frozenset({"final_evidence"}), frozenset({"source_pdf"}), False, False, True),
}


def descriptor_for(edge_type: object) -> RelationDescriptor | None:
    return RELATIONS.get(str(edge_type))


def is_relevance_relation(edge_type: object) -> bool:
    descriptor = descriptor_for(edge_type)
    return bool(descriptor and descriptor.relevance_eligible)


def is_authority_relation(edge_type: object) -> bool:
    descriptor = descriptor_for(edge_type)
    return bool(descriptor and descriptor.authority_bearing)


def is_query_relation(edge_type: object) -> bool:
    descriptor = descriptor_for(edge_type)
    return bool(descriptor and descriptor.query_eligible)


def materialize_inverse_edges(edges: list[dict]) -> None:
    """Persist declared inverse relations; traversal never invents an edge."""
    existing = {str(row.get("edge_id")) for row in edges}
    existing_directions = {(row.get("edge_type"), row.get("source_id"), row.get("target_id")) for row in edges}
    additions: list[dict] = []
    for row in tuple(edges):
        descriptor = descriptor_for(row.get("edge_type"))
        if not descriptor or not descriptor.inverse or row.get("derived_from_edge_id"):
            continue
        if (descriptor.inverse, row.get("target_id"), row.get("source_id")) in existing_directions:
            continue
        inverse_id = f"edge::{sha256(f'{row.get("edge_id")}|{descriptor.inverse}'.encode('utf-8')).hexdigest()}"
        if inverse_id in existing:
            continue
        additions.append(
            row
            | {
                "edge_id": inverse_id,
                "edge_type": descriptor.inverse,
                "relation_type": descriptor.inverse,
                "source_id": row["target_id"],
                "target_id": row["source_id"],
                "source_node_type": row.get("target_node_type"),
                "target_node_type": row.get("source_node_type"),
                "derived_from_edge_id": row["edge_id"],
            }
        )
        existing.add(inverse_id)
        existing_directions.add((descriptor.inverse, row.get("target_id"), row.get("source_id")))
    edges.extend(additions)
