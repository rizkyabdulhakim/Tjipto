"""Request-scoped verified evidence grouping and sufficiency assessment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

from tjipto.retrieval.answer import validate_answer_candidate


@dataclass(frozen=True)
class EvidenceRequirement:
    requirement_id: str
    description: str = ""
    source_role: str | None = None
    temporal_context: str | None = None
    evidence_ids: tuple[str, ...] = ()
    legal_unit_ids: tuple[str, ...] = ()
    min_supports: int = 1
    allow_partial: bool = False
    allow_shared: bool = False
    predicate: Callable[[dict], bool] | None = field(default=None, compare=False, repr=False)

    def accepts(self, row: dict) -> bool:
        if self.source_role is not None and row.get("source_role") != self.source_role:
            return False
        if self.temporal_context is not None and row.get("temporal_context") != self.temporal_context:
            return False
        if self.evidence_ids and str(row.get("evidence_id")) not in set(self.evidence_ids):
            return False
        if self.legal_unit_ids and str(row.get("legal_unit_id")) not in set(self.legal_unit_ids):
            return False
        return self.predicate(row) if self.predicate is not None else True


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
    missing: list[str] = []
    reasons: list[tuple[str, str]] = []
    for requirement in requirements:
        selected: list[dict] = []
        for evidence_id, row in available.items():
            if evidence_id in used and not requirement.allow_shared:
                continue
            if requirement.accepts(row):
                selected.append(row)
                if len(selected) >= max(1, requirement.min_supports):
                    break
        if len(selected) < max(1, requirement.min_supports):
            missing.append(requirement.requirement_id)
            reasons.append((requirement.requirement_id, "verified_support_missing"))
            continue
        selected_ids = tuple(str(row["evidence_id"]) for row in selected)
        assignments.append((requirement.requirement_id, selected_ids))
        supports.extend(selected)
        if not requirement.allow_shared:
            used.update(selected_ids)
    return EvidenceSet(tuple(supports), tuple(assignments), tuple(missing), tuple(reasons))


def assess_sufficiency(
    evidence_set: EvidenceSet,
    requirements: Iterable[EvidenceRequirement],
    *,
    partial_allowed: bool = False,
    retry_budget: int = 0,
) -> SufficiencyAssessment:
    requirements = tuple(requirements)
    fulfilled = tuple(requirement.requirement_id for requirement in requirements if requirement.requirement_id not in evidence_set.missing_requirement_ids)
    missing = evidence_set.missing_requirement_ids
    if not missing:
        status = "complete"
    elif partial_allowed and fulfilled and all(requirement.allow_partial for requirement in requirements if requirement.requirement_id in missing):
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
