"""Request-scoped verified evidence grouping and sufficiency assessment."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
import re
from typing import Iterable

from tjipto.retrieval.answer import validate_answer_candidate


@dataclass(frozen=True)
class EvidenceRequirement:
    requirement_id: str
    description: str = ""
    retrieval_query: str | None = None
    required_entities: tuple[str, ...] = ()
    relation_endpoints: tuple[str, ...] = ()
    contrast_entities: tuple[str, ...] = ()
    explicit_references: tuple[str, ...] = ()
    semantic_terms: tuple[str, ...] = ()
    support_terms: tuple[str, ...] = ()
    entity_must_lead: bool = False
    legal_target: str | None = None
    relation_family: str | None = None
    required_operation_terms: tuple[str, ...] = ()
    concept_facet: str | None = None
    source_role: str | None = None
    temporal_context: str | None = None
    authority_kinds: tuple[str, ...] = ()
    hierarchy_depth: int | None = None
    evidence_ids: tuple[str, ...] = ()
    legal_unit_ids: tuple[str, ...] = ()
    min_supports: int = 1
    allow_partial: bool = False
    allow_shared: bool = False
    shareable_with: tuple[str, ...] = ()
    metadata_field: str | None = None
    requires_final_citation: bool = False

    @property
    def typed(self) -> bool:
        return bool(
            self.required_entities
            or self.relation_endpoints
            or self.explicit_references
            or self.semantic_terms
            or self.support_terms
            or self.legal_target
            or self.relation_family
            or self.required_operation_terms
            or self.concept_facet
            or self.source_role
            or self.temporal_context
            or self.authority_kinds
            or self.hierarchy_depth is not None
            or self.evidence_ids
            or self.legal_unit_ids
            or self.metadata_field
            or self.requires_final_citation
        )

    def accepts(self, row: dict) -> bool:
        if not self.typed:
            return False
        if self.source_role is not None and row.get("source_role") != self.source_role:
            return False
        if self.temporal_context is not None and row.get("temporal_context") != self.temporal_context:
            return False
        if self.authority_kinds and row.get("authority_kind") not in set(self.authority_kinds):
            return False
        if self.hierarchy_depth is not None and len(row.get("hierarchy") or ()) != self.hierarchy_depth:
            return False
        if self.evidence_ids and str(row.get("evidence_id")) not in set(self.evidence_ids):
            return False
        if self.legal_unit_ids and str(row.get("legal_unit_id")) not in set(self.legal_unit_ids):
            return False
        if self.metadata_field is not None and row.get("metadata_field") != self.metadata_field:
            return False
        if self.requires_final_citation and row.get("citation_final") is not True:
            return False
        text = _semantic_text(row)
        if self.required_entities:
            positions = tuple(_phrase_position(text, entity) for entity in self.required_entities)
            if any(position < 0 for position in positions):
                return False
            if self.entity_must_lead and len(positions) == 1:
                competing = tuple(
                    position
                    for entity in self.contrast_entities
                    if (position := _phrase_position(text, entity)) >= 0
                )
                if competing and positions[0] > min(competing):
                    return False
        if self.relation_endpoints:
            if any(_phrase_position(text, endpoint) < 0 for endpoint in self.relation_endpoints):
                return False
        if self.explicit_references:
            haystack = " ".join(
                str(row.get(key) or "")
                for key in ("citation", "label", "legal_unit_id", "hierarchy", "quoted_text")
            ).casefold()
            if not all(str(item).casefold() in haystack for item in self.explicit_references):
                return False
        if self.legal_target:
            haystack = " ".join(
                str(row.get(key) or "")
                for key in ("citation", "label", "legal_unit_id", "hierarchy", "quoted_text")
            ).casefold()
            if str(self.legal_target).casefold() not in haystack:
                return False
        if self.relation_family and row.get("relation_family", row.get("relation_type")) != self.relation_family:
            return False
        if self.concept_facet and row.get("concept_facet") != self.concept_facet:
            return False
        tokens = set(_tokens(text))
        if self.required_operation_terms and not set(self.required_operation_terms) <= tokens:
            return False
        if self.semantic_terms and not set(self.semantic_terms) <= tokens:
            return False
        if self.support_terms and not set(self.support_terms) & tokens:
            return False
        return True

    def match_score(self, row: dict) -> tuple[int, int, int, int, str]:
        """Prefer rows carrying the strongest requirement-specific semantics."""
        text = _semantic_text(row)
        tokens = set(_tokens(text))
        return (
            len(tokens & set(self.support_terms)),
            len(tokens & set(self.required_operation_terms)),
            len(tokens & set(self.semantic_terms)),
            -len(text),
            str(row.get("evidence_id") or ""),
        )

    def discovered_for(self, row: dict) -> bool:
        """Require requirement-scoped discovery for query-bound requirements."""
        if not self.typed:
            return False
        scoped: set[str] = set()
        for key in ("_requirement_ids", "research_requirement_ids", "requirement_ids"):
            values = row.get(key) or ()
            values = (values,) if isinstance(values, str) else values if isinstance(values, Sequence) else ()
            scoped.update(str(value) for value in values)
        if self.retrieval_query or self.required_entities or self.explicit_references or self.relation_family:
            return self.requirement_id in scoped or bool(set(scoped) & set(self.shareable_with))
        return True


@dataclass(frozen=True)
class EvidenceSet:
    supports: tuple[dict, ...]
    assignments: tuple[tuple[str, tuple[str, ...]], ...]
    missing_requirement_ids: tuple[str, ...]
    missing_reasons: tuple[tuple[str, str], ...] = ()

    @property
    def complete(self) -> bool:
        return not self.missing_requirement_ids

    def assignment_map(self) -> dict[str, tuple[str, ...]]:
        return dict(self.assignments)


@dataclass(frozen=True)
class SufficiencyAssessment:
    status: str
    fulfilled_requirement_ids: tuple[str, ...]
    missing_requirement_ids: tuple[str, ...]
    missing_reasons: tuple[tuple[str, str], ...]
    retry_allowed: bool


def collect_evidence_set(store, matches: Iterable[dict], requirements: Iterable[EvidenceRequirement]) -> EvidenceSet:
    """Assign each verified row to at most one requirement by default."""
    verified: list[dict] = []
    seen: set[str] = set()
    for row in matches:
        evidence_id = str(row.get("evidence_id") or "")
        if not evidence_id or evidence_id in seen:
            continue
        try:
            accepted, _ = validate_answer_candidate(store, row)
        except (AttributeError, KeyError, TypeError, ValueError):
            accepted = False
        if accepted:
            verified.append(row)
            seen.add(evidence_id)
    available = {str(row.get("evidence_id")): row for row in verified}
    used: set[str] = set()
    assignments: list[tuple[str, tuple[str, ...]]] = []
    supports: list[dict] = []
    support_ids: set[str] = set()
    missing: list[str] = []
    reasons: list[tuple[str, str]] = []
    for requirement in requirements:
        selected: list[dict] = []
        candidates = sorted(
            available.items(),
            key=lambda item: tuple(-value if isinstance(value, int) else value for value in requirement.match_score(item[1])),
        )
        for evidence_id, row in candidates:
            if evidence_id in used and not requirement.allow_shared:
                continue
            if requirement.discovered_for(row) and requirement.accepts(row):
                selected.append(row)
                if len(selected) >= max(1, requirement.min_supports):
                    break
        if len(selected) < max(1, requirement.min_supports):
            missing.append(requirement.requirement_id)
            reasons.append((requirement.requirement_id, "verified_support_missing"))
            continue
        selected_ids = tuple(str(row["evidence_id"]) for row in selected)
        assignments.append((requirement.requirement_id, selected_ids))
        for row in selected:
            if str(row["evidence_id"]) not in support_ids:
                supports.append(row)
                support_ids.add(str(row["evidence_id"]))
        if not requirement.allow_shared:
            used.update(selected_ids)
    return EvidenceSet(tuple(supports), tuple(assignments), tuple(missing), tuple(reasons))


def _semantic_text(row: dict) -> str:
    return " ".join(
        str(value or "")
        for value in (
            row.get("entity_identity"),
            row.get("printed_name"),
            row.get("institution"),
            row.get("citation"),
            " ".join(row.get("hierarchy") or ()),
            row.get("quoted_text"),
            row.get("display_text"),
        )
    ).casefold()


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+", value.casefold()))


def _phrase_position(text: str, phrase: str) -> int:
    normalized = " ".join(_tokens(phrase))
    return " ".join(_tokens(text)).find(normalized) if normalized else -1


def assess_sufficiency(
    evidence_set: EvidenceSet,
    requirements: Iterable[EvidenceRequirement],
    *,
    retry_budget: int = 0,
) -> SufficiencyAssessment:
    requirements = tuple(requirements)
    fulfilled = tuple(requirement.requirement_id for requirement in requirements if requirement.requirement_id not in evidence_set.missing_requirement_ids)
    missing = evidence_set.missing_requirement_ids
    if not missing:
        status = "complete"
    elif fulfilled and all(requirement.allow_partial for requirement in requirements if requirement.requirement_id in missing):
        status = "partial"
    else:
        status = "insufficient"
    return SufficiencyAssessment(
        status=status,
        fulfilled_requirement_ids=fulfilled,
        missing_requirement_ids=missing,
        missing_reasons=evidence_set.missing_reasons,
        retry_allowed=bool(missing and retry_budget > 0),
    )


def retry_allowed(attempt: int, max_attempts: int) -> bool:
    return max_attempts > 0 and attempt >= 0 and attempt < max_attempts


__all__ = [
    "EvidenceRequirement",
    "EvidenceSet",
    "SufficiencyAssessment",
    "assess_sufficiency",
    "collect_evidence_set",
    "retry_allowed",
]
