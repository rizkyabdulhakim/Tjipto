"""Provider-neutral planning and bounded requirement-scoped research."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import os
from typing import Mapping, Protocol, Sequence
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from tjipto.retrieval.sufficiency import (
    EvidenceRequirement,
    EvidenceSet,
    SufficiencyAssessment,
    assess_sufficiency,
    collect_evidence_set,
)


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
    polarity: str | None = None
    modality: str | None = None
    requirement_id: str | None = None
    retrieval_lane: str = "auto"


@dataclass(frozen=True)
class ResearchPlan:
    original_query: str
    intent: ResearchIntent
    variants: tuple[QueryVariant, ...]
    provider_status: str = "deterministic"
    rejection_reasons: tuple[str, ...] = ()
    retrieval_lanes: tuple[str, ...] = ("sparse",)
    requirements: tuple[EvidenceRequirement, ...] = ()
    task_kind: str = "retrieval"


class ResearchPlanningProvider(Protocol):
    def propose(self, request: Mapping[str, object]) -> object: ...


class OpenAICompatibleResearchPlanningProvider:
    """Optional, untrusted structured planner adapter."""

    def __init__(self, api_key: str, *, model: str, endpoint: str, timeout: float = 12.0):
        self._api_key = api_key
        self._model = model
        self._endpoint = endpoint
        self._timeout = timeout

    def propose(self, request: Mapping[str, object]) -> object:
        payload = {
            "model": self._model,
            "temperature": 0,
            "messages": [{
                "role": "user",
                "content": (
                    "Return JSON only. Propose retrieval planning data, never legal truth, authority, "
                    "source role, temporal status, citations, or evidence validity.\n"
                    + json.dumps(dict(request), ensure_ascii=False, sort_keys=True)
                ),
            }],
            "response_format": {"type": "json_object"},
        }
        req = Request(
            self._endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self._api_key}"},
            method="POST",
        )
        with urlopen(req, timeout=self._timeout) as response:  # nosec B310 - factory restricts HTTPS.
            body = json.load(response)
        return json.loads(str(body["choices"][0]["message"]["content"]))


def research_planning_provider_from_environment() -> ResearchPlanningProvider | None:
    """Create the optional planner only after explicit opt-in and valid config."""
    if os.environ.get("TJIPTO_EXTERNAL_RESEARCH_PLANNING", "").strip().casefold() != "enabled":
        return None
    if os.environ.get("TJIPTO_RESEARCH_PLANNING_PROVIDER", "").strip().casefold() != "openai_compatible":
        return None
    api_key = os.environ.get("TJIPTO_RESEARCH_PLANNING_API_KEY", "").strip()
    model = os.environ.get("TJIPTO_RESEARCH_PLANNING_MODEL", "").strip()
    base_url = os.environ.get("TJIPTO_RESEARCH_PLANNING_BASE_URL", "").strip().rstrip("/")
    if not api_key or not model or not base_url:
        return None
    try:
        timeout = max(1.0, float(os.environ.get("TJIPTO_RESEARCH_PLANNING_TIMEOUT_SECONDS", "12")))
    except ValueError:
        return None
    parsed = urlparse(base_url + "/chat/completions")
    if parsed.scheme != "https" or not parsed.netloc:
        return None
    return OpenAICompatibleResearchPlanningProvider(
        api_key,
        model=model,
        endpoint=base_url + "/chat/completions",
        timeout=timeout,
    )


def plan_research(
    query: str,
    intent: ResearchIntent | None = None,
    *,
    provider: ResearchPlanningProvider | None = None,
    required_entities: Sequence[str] = (),
    explicit_references: Sequence[str] = (),
    source_role: str | None = None,
    temporal_scope: str | None = None,
    polarity: str | None = None,
    modality: str | None = None,
    requirements: Sequence[EvidenceRequirement] = (),
) -> ResearchPlan:
    """Create a validated plan. The original query is always variant zero."""
    intent = intent or ResearchIntent()
    requirement_rows = tuple(requirements)
    if not required_entities:
        required_entities = tuple(
            dict.fromkeys(
                value
                for requirement in requirement_rows
                for value in (*requirement.required_entities, *requirement.relation_endpoints)
            )
        )
    if not explicit_references:
        explicit_references = tuple(
            dict.fromkeys(value for requirement in requirement_rows for value in requirement.explicit_references)
        )
    requirement_roles = tuple(dict.fromkeys(requirement.source_role for requirement in requirement_rows if requirement.source_role))
    requirement_temporal = tuple(
        dict.fromkeys(requirement.temporal_context for requirement in requirement_rows if requirement.temporal_context)
    )
    if source_role is None and len(requirement_roles) == 1:
        source_role = requirement_roles[0]
    if temporal_scope is None and len(requirement_temporal) == 1:
        temporal_scope = requirement_temporal[0]
    original = QueryVariant(
        query=query,
        required_entities=tuple(required_entities),
        explicit_references=tuple(explicit_references),
        source_role=source_role,
        temporal_scope=temporal_scope,
        polarity=polarity,
        modality=modality,
    )
    base_requirements = requirement_rows
    if provider is None or not intent.complex:
        return ResearchPlan(query, intent, (original,), "deterministic", requirements=base_requirements)
    try:
        proposal = provider.propose(
            _planner_request(
                query,
                intent,
                required_entities=required_entities,
                explicit_references=explicit_references,
                source_role=source_role,
                temporal_scope=temporal_scope,
                polarity=polarity,
                modality=modality,
            )
        )
    except Exception:  # provider is optional and untrusted
        return ResearchPlan(query, intent, (original,), "unavailable", ("provider_failure",), requirements=base_requirements)
    variants, rejected = _validated_variants(proposal, original, intent)
    lanes = _validated_lanes(proposal)
    if lanes is None:
        rejected = (*rejected, "retrieval_lane_invalid")
        lanes = ("sparse",)
    _proposed_requirements, requirement_rejections = _validated_requirements(
        proposal,
        required_entities=required_entities,
        explicit_references=explicit_references,
        source_role=source_role,
        temporal_scope=temporal_scope,
    )
    rejected = (*rejected, *requirement_rejections)
    if isinstance(proposal, Mapping) and proposal.get("requirements"):
        rejected = (*rejected, "provider_requirements_forbidden")
    task_kind = _validated_task_kind(proposal)
    if task_kind is None:
        rejected = (*rejected, "task_kind_invalid")
        task_kind = "retrieval"
    return ResearchPlan(
        query,
        intent,
        variants,
        "accepted" if not rejected else "degraded",
        rejected,
        lanes,
        tuple(base_requirements),
        task_kind,
    )


def _planner_request(
    query: str,
    intent: ResearchIntent,
    *,
    required_entities: Sequence[str],
    explicit_references: Sequence[str],
    source_role: str | None,
    temporal_scope: str | None,
    polarity: str | None,
    modality: str | None,
) -> dict[str, object]:
    """Return the JSON contract sent to an untrusted planning provider."""
    return {
        "query": query,
        "intent": {
            "semantic_retrieval": intent.semantic_retrieval,
            "multiple_supports": intent.multiple_supports,
            "comparison": intent.comparison,
            "decomposition": intent.decomposition,
            "relation_traversal": intent.relation_traversal,
            "max_variants": intent.max_variants,
            "max_rounds": intent.max_rounds,
        },
        "constraints": {
            "required_entities": tuple(str(value) for value in required_entities),
            "explicit_references": tuple(str(value) for value in explicit_references),
            "source_role": source_role,
            "temporal_scope": temporal_scope,
            "polarity": polarity,
            "modality": modality,
        },
        "allowed_proposals": ("variants", "task_kind", "retrieval_lanes"),
    }


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
            polarity=item.get("polarity", original.polarity),
            modality=item.get("modality", original.modality),
        )
        if value.source_role is not None and not isinstance(value.source_role, str):
            rejected.append("variant_scope_type_invalid")
            continue
        if value.temporal_scope is not None and not isinstance(value.temporal_scope, str):
            rejected.append("variant_scope_type_invalid")
            continue
        if value.polarity is not None and not isinstance(value.polarity, str):
            rejected.append("variant_scope_type_invalid")
            continue
        if value.modality is not None and not isinstance(value.modality, str):
            rejected.append("variant_scope_type_invalid")
            continue
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
        and value.polarity == original.polarity
        and value.modality == original.modality
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


def _validated_task_kind(proposal: object) -> str | None:
    if not isinstance(proposal, Mapping) or "task_kind" not in proposal:
        return "retrieval"
    value = proposal.get("task_kind")
    allowed = {"retrieval", "multiple_supports", "comparison", "decomposition", "relation_traversal"}
    return value if isinstance(value, str) and value in allowed else None


def _validated_requirements(
    proposal: object,
    *,
    required_entities: Sequence[str] = (),
    explicit_references: Sequence[str] = (),
    source_role: str | None = None,
    temporal_scope: str | None = None,
) -> tuple[tuple[EvidenceRequirement, ...], tuple[str, ...]]:
    if not isinstance(proposal, Mapping):
        return (), ()
    raw = proposal.get("requirements", ())
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return (), ("requirements_malformed",) if raw else ()
    allowed = {
        "requirement_id", "description", "retrieval_query", "required_entities", "relation_endpoints", "explicit_references",
        "legal_target", "relation_family", "concept_facet", "source_role", "temporal_context",
        "evidence_ids", "legal_unit_ids", "min_supports", "allow_partial", "allow_shared", "shareable_with",
    }
    result: list[EvidenceRequirement] = []
    rejected: list[str] = []
    seen: set[str] = set()
    known_ids = {
        item.get("requirement_id")
        for item in raw
        if isinstance(item, Mapping) and isinstance(item.get("requirement_id"), str)
    }
    text_fields = {"description", "retrieval_query", "legal_target", "relation_family", "concept_facet", "source_role", "temporal_context"}
    sequence_fields = {"required_entities", "relation_endpoints", "explicit_references", "evidence_ids", "legal_unit_ids", "shareable_with"}
    bool_fields = {"allow_partial", "allow_shared"}
    for item in raw:
        if not isinstance(item, Mapping) or not isinstance(item.get("requirement_id"), str):
            rejected.append("requirement_malformed")
            continue
        if set(item) - allowed:
            rejected.append("requirement_unknown_field")
            continue
        identifier = str(item["requirement_id"])
        if not identifier or identifier in seen:
            rejected.append("requirement_duplicate_or_empty")
            continue
        if any(item.get(field) is not None and not isinstance(item.get(field), str) for field in text_fields):
            rejected.append("requirement_text_field_invalid")
            continue
        invalid_sequences = any(field in item and not _valid_string_sequence(item.get(field)) for field in sequence_fields)
        if invalid_sequences or any(field in item and not isinstance(item.get(field), bool) for field in bool_fields):
            rejected.append("requirement_field_type_invalid")
            continue
        try:
            minimum = int(item.get("min_supports", 1))
        except (TypeError, ValueError):
            rejected.append("requirement_min_supports_invalid")
            continue
        if isinstance(item.get("min_supports", 1), bool) or minimum < 1:
            rejected.append("requirement_min_supports_invalid")
            continue
        entities = tuple(str(value) for value in item.get("required_entities") or ())
        relation_endpoints = tuple(str(value) for value in item.get("relation_endpoints") or ())
        references = tuple(str(value) for value in item.get("explicit_references") or ())
        if (
            set(entities) != set(required_entities)
            or not set(relation_endpoints) <= set(required_entities)
            or set(references) != set(explicit_references)
            or item.get("source_role", source_role) != source_role
            or item.get("temporal_context", temporal_scope) != temporal_scope
        ):
            rejected.append("requirement_scope_invariant_violation")
            continue
        requirement = EvidenceRequirement(
            requirement_id=identifier,
            description=str(item.get("description") or ""),
            retrieval_query=item.get("retrieval_query"),
            required_entities=entities,
            relation_endpoints=relation_endpoints,
            explicit_references=references,
            legal_target=item.get("legal_target"),
            relation_family=item.get("relation_family"),
            concept_facet=item.get("concept_facet"),
            source_role=item.get("source_role"),
            temporal_context=item.get("temporal_context"),
            evidence_ids=tuple(str(value) for value in item.get("evidence_ids") or ()),
            legal_unit_ids=tuple(str(value) for value in item.get("legal_unit_ids") or ()),
            min_supports=minimum,
            allow_partial=bool(item.get("allow_partial", False)),
            allow_shared=bool(item.get("allow_shared", False)),
            shareable_with=tuple(str(value) for value in item.get("shareable_with") or ()),
        )
        if not requirement.typed:
            rejected.append("requirement_unbound")
            continue
        if any(value not in known_ids or value == identifier for value in requirement.shareable_with):
            rejected.append("requirement_share_target_invalid")
            continue
        result.append(requirement)
        seen.add(identifier)
    return tuple(result), tuple(rejected)


def _valid_string_sequence(value: object) -> bool:
    if value is None:
        return True
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and all(isinstance(item, str) for item in value)


def execute_research_rounds(
    query: str,
    retrieve,
    *,
    store=None,
    intent: ResearchIntent | None = None,
    provider: ResearchPlanningProvider | None = None,
    requirements: Sequence[EvidenceRequirement] = (),
    max_rounds: int | None = None,
    required_entities: Sequence[str] = (),
    explicit_references: Sequence[str] = (),
    source_role: str | None = None,
    temporal_scope: str | None = None,
    polarity: str | None = None,
    modality: str | None = None,
) -> dict:
    """Run retrieval rounds, retrying only requirements still missing."""
    plan = plan_research(
        query,
        intent,
        provider=provider,
        requirements=requirements,
        required_entities=required_entities,
        explicit_references=explicit_references,
        source_role=source_role,
        temporal_scope=temporal_scope,
        polarity=polarity,
        modality=modality,
    )
    requirements = tuple(plan.requirements)
    rounds_limit = min(
        max(1, int(max_rounds if max_rounds is not None else plan.intent.max_rounds)),
        max(1, plan.intent.max_rounds),
    )
    rows: dict[str, dict] = {}
    routes: list[dict] = []
    lane = (
        "hybrid"
        if "hybrid" in plan.retrieval_lanes or (plan.intent.complex and plan.provider_status == "deterministic")
        else "dense"
        if "dense" in plan.retrieval_lanes
        else "auto"
    )
    plan = replace(plan, retrieval_lanes=(lane,))
    current_variants = tuple(
        replace(variant, retrieval_lane=lane)
        for variant in plan.variants[: max(1, plan.intent.max_variants)]
    )
    evidence_set: EvidenceSet | None = None
    assessment: SufficiencyAssessment | None = None
    stop_reason = "max_rounds"
    for round_number in range(rounds_limit):
        before = len(rows)
        for variant in current_variants:
            result = retrieve(variant.query, variant)
            route = result if isinstance(result, Mapping) and "matches" in result else {"matches": tuple(result or ())}
            routes.append(dict(route) | {"research_round": round_number + 1, "query_variant": variant.origin})
            for source_row in route.get("matches", ()):
                evidence_id = str(source_row.get("evidence_id") or "")
                if not evidence_id:
                    continue
                row = dict(source_row)
                if variant.requirement_id:
                    row["_requirement_ids"] = (*tuple(row.get("_requirement_ids") or ()), variant.requirement_id)
                current = rows.get(evidence_id)
                if current is None:
                    rows[evidence_id] = row
                elif variant.requirement_id:
                    # Requirement-scoped rediscovery may carry fresher
                    # lexical coverage and route provenance for the same
                    # verified support.  Keep the canonical row state while
                    # merging the requirement marker and lane trace.
                    markers = tuple(current.get("_requirement_ids") or ())
                    current_routes = tuple(current.get("route_sources") or ())
                    discovered_routes = tuple(row.get("route_sources") or ())
                    current.update(row)
                    current["_requirement_ids"] = tuple(dict.fromkeys((*markers, variant.requirement_id)))
                    if discovered_routes:
                        current["route_sources"] = tuple(dict.fromkeys((*current_routes, *discovered_routes)))
        matches = tuple(rows.values())
        if requirements:
            evidence_set = collect_evidence_set(store, matches, requirements)
            assessment = assess_sufficiency(evidence_set, requirements, retry_budget=rounds_limit - round_number - 1)
            if assessment.status == "complete":
                stop_reason = "complete"
                break
            unresolved = tuple(
                requirement
                for requirement in requirements
                if requirement.requirement_id in assessment.missing_requirement_ids
            )
            if round_number + 1 >= rounds_limit:
                stop_reason = "max_rounds"
                break
            current_variants = tuple(
                QueryVariant(
                    query=requirement.retrieval_query or query,
                    origin=f"requirement:{requirement.requirement_id}",
                    required_entities=tuple(dict.fromkeys((*requirement.required_entities, *requirement.relation_endpoints))),
                    explicit_references=requirement.explicit_references,
                    source_role=requirement.source_role,
                    temporal_scope=requirement.temporal_context,
                    requirement_id=requirement.requirement_id,
                    retrieval_lane=lane,
                )
                for requirement in unresolved
            )
            if not current_variants:
                stop_reason = "no_progress"
                break
        else:
            stop_reason = "complete" if len(rows) > before else "no_progress"
            break
        if len(rows) == before and round_number > 0 and current_variants:
            stop_reason = "no_progress"
            break
    matches = tuple(rows.values())
    return {
        "plan": plan,
        "routes": tuple(routes),
        "matches": matches,
        "evidence_set": evidence_set,
        "sufficiency": assessment,
        "stop_reason": stop_reason,
        "rounds": len({route.get("research_round") for route in routes}),
    }


def execute_research(
    query: str,
    retrieve,
    *,
    intent: ResearchIntent | None = None,
    provider: ResearchPlanningProvider | None = None,
    max_rounds: int | None = None,
) -> tuple[ResearchPlan, tuple[dict, ...]]:
    """Compatibility projection for callers that only need plan and rows."""
    result = execute_research_rounds(query, retrieve, intent=intent, provider=provider, max_rounds=max_rounds)
    return result["plan"], result["matches"]


__all__ = [
    "QueryVariant",
    "ResearchIntent",
    "ResearchPlan",
    "ResearchPlanningProvider",
    "OpenAICompatibleResearchPlanningProvider",
    "research_planning_provider_from_environment",
    "execute_research",
    "execute_research_rounds",
    "plan_research",
]
