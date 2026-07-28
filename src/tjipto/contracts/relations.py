"""Code-owned semantic relation descriptors used by build and runtime."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RelationDescriptor:
    inverse: str | None
    source_types: frozenset[str]
    target_types: frozenset[str]
    relevance_eligible: bool
    authority_bearing: bool
    provenance_required: bool


_LEGAL_UNIT = frozenset({"legal_unit"})

RELATIONS: dict[str, RelationDescriptor] = {
    "CONTAINS": RelationDescriptor("PART_OF", _LEGAL_UNIT, _LEGAL_UNIT, True, False, False),
    "PART_OF": RelationDescriptor("CONTAINS", _LEGAL_UNIT, _LEGAL_UNIT, True, False, False),
    "PRECEDES": RelationDescriptor("FOLLOWS", _LEGAL_UNIT, _LEGAL_UNIT, True, False, False),
    "FOLLOWS": RelationDescriptor("PRECEDES", _LEGAL_UNIT, _LEGAL_UNIT, True, False, False),
    "INSERTED_AFTER": RelationDescriptor(None, _LEGAL_UNIT, _LEGAL_UNIT, True, False, False),
    "MODIFIES": RelationDescriptor("MODIFIED_BY", _LEGAL_UNIT, _LEGAL_UNIT, True, True, True),
    "MODIFIED_BY": RelationDescriptor("MODIFIES", _LEGAL_UNIT, _LEGAL_UNIT, True, True, True),
    "RENAMES": RelationDescriptor("RENAMED_FROM", _LEGAL_UNIT, _LEGAL_UNIT, True, True, True),
    "RENAMED_FROM": RelationDescriptor("RENAMES", _LEGAL_UNIT, _LEGAL_UNIT, True, True, True),
    "RENUMBERED_TO": RelationDescriptor("RENUMBERED_FROM", _LEGAL_UNIT, _LEGAL_UNIT, True, True, True),
    "RENUMBERED_FROM": RelationDescriptor("RENUMBERED_TO", _LEGAL_UNIT, _LEGAL_UNIT, True, True, True),
    "DELETES": RelationDescriptor("DELETED_BY", _LEGAL_UNIT, _LEGAL_UNIT, True, True, True),
    "DELETED_BY": RelationDescriptor("DELETES", _LEGAL_UNIT, _LEGAL_UNIT, True, True, True),
    "INSERTS": RelationDescriptor("INSERTED_BY", _LEGAL_UNIT, _LEGAL_UNIT, True, True, True),
    "INSERTED_BY": RelationDescriptor("INSERTS", _LEGAL_UNIT, _LEGAL_UNIT, True, True, True),
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
