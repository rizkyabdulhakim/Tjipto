"""Small, provider-neutral planning values for bounded legal research."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence


@dataclass(frozen=True)
class ResearchIntent:
    semantic_retrieval: bool = True
    multiple_supports: bool = False
    comparison: bool = False
    decomposition: bool = False
    relation_traversal: bool = False
    max_variants: int = 4
    max_rounds: int = 2

    @property
    def complex(self) -> bool:
        return self.multiple_supports or self.comparison or self.decomposition or self.relation_traversal


@dataclass(frozen=True)
class QueryVariant:
    query: str
    origin: str = "original"
    required_entities: tuple[str, ...] = ()
    explicit_references: tuple[str, ...] = ()
    source_role: str | None = None
    temporal_scope: str | None = None


@dataclass(frozen=True)
class ResearchPlan:
    original_query: str
    intent: ResearchIntent
    variants: tuple[QueryVariant, ...]
    provider_status: str = "deterministic"
    rejection_reasons: tuple[str, ...] = ()
    retrieval_lanes: tuple[str, ...] = ("sparse",)


class ResearchPlanningProvider(Protocol):
    def propose(self, request: Mapping[str, object]) -> object: ...


def plan_research(
    query: str,
    intent: ResearchIntent | None = None,
    *,
    provider: ResearchPlanningProvider | None = None,
    required_entities: Sequence[str] = (),
    explicit_references: Sequence[str] = (),
    source_role: str | None = None,
    temporal_scope: str | None = None,
) -> ResearchPlan:
    """Create a validated plan.  The original query is always variant zero."""
    intent = intent or ResearchIntent()
    original = QueryVariant(
        query=query,
        required_entities=tuple(required_entities),
        explicit_references=tuple(explicit_references),
        source_role=source_role,
        temporal_scope=temporal_scope,
    )
    if provider is None or not intent.complex:
        return ResearchPlan(query, intent, (original,), "deterministic")
    try:
        proposal = provider.propose(
            {
                "query": query,
                "intent": intent,
                "required_entities": tuple(required_entities),
                "explicit_references": tuple(explicit_references),
                "source_role": source_role,
                "temporal_scope": temporal_scope,
            }
        )
    except Exception:  # provider is optional and untrusted
        return ResearchPlan(query, intent, (original,), "unavailable", ("provider_failure",))
    variants, rejected = _validated_variants(
        proposal,
        original,
        intent,
    )
    lanes = _validated_lanes(proposal)
    if lanes is None:
        rejected = (*rejected, "retrieval_lane_invalid")
        lanes = ("sparse",)
    return ResearchPlan(
        query,
        intent,
        variants,
        "accepted" if not rejected else "degraded",
        rejected,
        lanes,
    )


def _validated_variants(
    proposal: object,
    original: QueryVariant,
    intent: ResearchIntent,
) -> tuple[tuple[QueryVariant, ...], tuple[str, ...]]:
    if (
        not isinstance(proposal, Mapping)
        or not isinstance(proposal.get("variants"), Sequence)
        or isinstance(proposal.get("variants"), (str, bytes))
    ):
        return (original,), ("malformed_schema",)
    variants: list[QueryVariant] = [original]
    rejected: list[str] = []
    seen = {original.query.strip()}
    for item in proposal["variants"]:
        if len(variants) >= max(1, intent.max_variants):
            rejected.append("variant_budget_exceeded")
            break
        if not isinstance(item, Mapping) or not isinstance(item.get("query"), str):
            rejected.append("variant_malformed")
            continue
        value = QueryVariant(
            query=str(item["query"]),
            origin=str(item.get("origin") or "provider"),
            required_entities=tuple(str(value) for value in item.get("required_entities", original.required_entities) or ()),
            explicit_references=tuple(str(value) for value in item.get("explicit_references", original.explicit_references) or ()),
            source_role=item.get("source_role", original.source_role),
            temporal_scope=item.get("temporal_scope", original.temporal_scope),
        )
        if not value.query.strip() or value.query.strip() in seen:
            rejected.append("duplicate_or_empty_variant")
            continue
        if not _preserves_scope(value, original):
            rejected.append("scope_invariant_violation")
            continue
        seen.add(value.query.strip())
        variants.append(value)
    return tuple(variants), tuple(rejected)


def _preserves_scope(value: QueryVariant, original: QueryVariant) -> bool:
    return (
        set(value.required_entities) == set(original.required_entities)
        and set(value.explicit_references) == set(original.explicit_references)
        and value.source_role == original.source_role
        and value.temporal_scope == original.temporal_scope
    )


def _validated_lanes(proposal: object) -> tuple[str, ...] | None:
    if not isinstance(proposal, Mapping):
        return ("sparse",)
    raw = proposal.get("retrieval_lanes", ("sparse",))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return None
    lanes = tuple(str(item) for item in raw)
    allowed = {"sparse", "dense", "hybrid"}
    return lanes if lanes and set(lanes) <= allowed else None


def execute_research(
    query: str,
    retrieve,
    *,
    intent: ResearchIntent | None = None,
    provider: ResearchPlanningProvider | None = None,
    max_rounds: int | None = None,
) -> tuple[ResearchPlan, tuple[dict, ...]]:
    """Run bounded variants, retaining the original query and deduplicating IDs."""
    plan = plan_research(query, intent, provider=provider)
    rounds = min(max_rounds if max_rounds is not None else plan.intent.max_rounds, plan.intent.max_rounds)
    rows: dict[str, dict] = {}
    for variant in plan.variants[: max(1, rounds)]:
        for row in retrieve(variant.query, variant):
            evidence_id = str(row.get("evidence_id") or "")
            if evidence_id and evidence_id not in rows:
                rows[evidence_id] = dict(row)
    return plan, tuple(rows.values())


__all__ = [
    "QueryVariant",
    "ResearchIntent",
    "ResearchPlan",
    "ResearchPlanningProvider",
    "execute_research",
    "plan_research",
]
