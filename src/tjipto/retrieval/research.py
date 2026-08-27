"""Provider-neutral planning and bounded requirement-scoped research."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from typing import Mapping, Protocol, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from tjipto.core.external_llm import (
    ExternalLLMConfig,
    FallbackProposalProvider,
    OPENAI_COMPATIBLE_USER_AGENT,
    external_llm_config,
    fallback_external_llm_config,
    openai_compatible_latency_options,
)
from tjipto.retrieval.candidates import graph_expand
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
    # The runtime, not a lexical complexity heuristic, decides when a normal
    # semantic request is eligible for the untrusted planner.
    orchestrate: bool = False
    max_variants: int = 4
    max_rounds: int = 2

    @property
    def complex(self) -> bool:
        return self.multiple_supports or self.comparison or self.decomposition or self.relation_traversal


def expand_research_candidates(store, result: dict, *, decomposition: bool, limit: int) -> dict:
    """Add bounded graph candidates without granting them authority."""
    if not decomposition or not result.get("matches"):
        return result
    seeds = tuple(
        dict(row) | {"route_sources": tuple(dict.fromkeys(("structured", *(row.get("route_sources") or ()))))}
        for row in result.get("matches", ())
        if row.get("evidence_id")
    )
    trace = graph_expand(store, seeds, {}, per_seed=max(1, limit), semantic=True)
    expanded = tuple(
        dict(row) | {"route_sources": ("graph",)}
        for item in trace
        if (row := store.get(str(item.get("evidence_id") or ""))) is not None
    )
    return dict(result) | {"matches": tuple(result.get("matches", ())) + expanded} if expanded else result


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
class InformationNeed:
    """Untrusted semantic topic proposal; it carries no evidence authority."""

    description: str
    query: str | None = None
    concepts: tuple[str, ...] = ()
    kind: str = "concept"
    relation_traversal: bool = False


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
    information_needs: tuple[InformationNeed, ...] = ()
    clarification_question: str | None = None
    missing_dimensions: tuple[str, ...] = ()


class ResearchPlanningProvider(Protocol):
    def propose(self, request: Mapping[str, object]) -> object: ...


class OpenAICompatibleResearchPlanningProvider:
    """Untrusted structured planner adapter; the server validates every proposal."""

    def __init__(self, api_key: str, *, model: str, endpoint: str, timeout: float = 12.0):
        self._api_key = api_key
        self._model = model
        self._endpoint = endpoint
        self._timeout = timeout

    def propose(self, request: Mapping[str, object]) -> object:
        request_intent = request.get("intent")
        configured_max = request_intent.get("max_variants") if isinstance(request_intent, Mapping) else None
        provider_variant_limit = max(0, configured_max - 1) if isinstance(configured_max, int) else 3
        payload = {
            "model": self._model,
            "temperature": 0,
            **openai_compatible_latency_options(self._model, self._endpoint),
            "messages": [{
                "role": "user",
                "content": (
                    "Return exactly one JSON object and no markdown or prose. "
                    "Use only these top-level keys and exact shapes: "
                    '"variants": an array of objects with a non-empty string "query" (never strings); '
                    '"retrieval_lanes": a non-empty array containing only "sparse", "dense", or "hybrid"; '
                    '"task_kind": exactly one of "retrieval", "multiple_supports", "comparison", '
                    '"decomposition", or "relation_traversal"; '
                    '"information_needs": an array of objects whose allowed fields are '
                    '"description" (non-empty string), "query" (string or null), "concepts" '
                    '(array of strings), "kind" ("concept", "comparison", "procedure", or "relation"), '
                    'and "relation_traversal" (boolean). '
                    'If and only if a required user choice is missing, set "status" to "clarification_required", '
                    'name the missing dimensions, and write one neutral Indonesian clarification_question. '
                    'When constraints.comparison_target_required is true, the comparison has only one side: '
                    'status must be "clarification_required" and missing_dimensions must include "comparison_target". '
                    'A legal-analysis or legal-opinion query that already names its legal issue is ready for '
                    'semantic retrieval; it does not require an exact article, source period, or comparison target. '
                    'Otherwise set "status" to "ready" with empty missing_dimensions and null clarification_question. '
                    f"Return at most {provider_variant_limit} provider variants; the server already retains the original query. "
                    "Return no other fields in those objects. Never return requirements, evidence, citations, "
                    "authority, source role, temporal status, or evidence validity; those remain server-owned.\n"
                    + json.dumps(dict(request), ensure_ascii=False, sort_keys=True)
                ),
            }],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "tjipto_research_plan",
                    "strict": True,
                    "schema": _planner_schema(provider_variant_limit),
                },
            },
        }
        req = Request(
            self._endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
                "User-Agent": OPENAI_COMPATIBLE_USER_AGENT,
            },
            method="POST",
        )
        for attempt in range(2):
            try:
                with urlopen(req, timeout=self._timeout) as response:  # nosec B310 - factory restricts HTTPS.
                    body = json.load(response)
                return json.loads(str(body["choices"][0]["message"]["content"]))
            except HTTPError as error:
                if attempt or error.code not in {500, 502, 503, 504}:
                    raise
            except (TimeoutError, URLError):
                if attempt:
                    raise
        raise RuntimeError("planner request retry exhausted")  # pragma: no cover


def _planner_schema(provider_variant_limit: int) -> dict[str, object]:
    need = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "description": {"type": "string", "minLength": 1},
            "query": {"type": ["string", "null"]},
            "concepts": {"type": "array", "items": {"type": "string"}},
            "kind": {"type": "string", "enum": ["concept", "comparison", "procedure", "relation"]},
            "relation_traversal": {"type": "boolean"},
        },
        "required": ["description", "query", "concepts", "kind", "relation_traversal"],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "variants": {
                "type": "array",
                "maxItems": provider_variant_limit,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {"query": {"type": "string", "minLength": 1}},
                    "required": ["query"],
                },
            },
            "retrieval_lanes": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "string", "enum": ["sparse", "dense", "hybrid"]},
            },
            "task_kind": {
                "type": "string",
                "enum": ["retrieval", "multiple_supports", "comparison", "decomposition", "relation_traversal"],
            },
            "information_needs": {"type": "array", "maxItems": 3, "items": need},
            "status": {"type": "string", "enum": ["ready", "clarification_required"]},
            "missing_dimensions": {
                "type": "array",
                "maxItems": 2,
                "items": {"type": "string", "enum": ["legal_target", "source_scope", "temporal_scope", "comparison_target"]},
            },
            "clarification_question": {"type": ["string", "null"], "maxLength": 240},
        },
        "required": ["variants", "retrieval_lanes", "task_kind", "information_needs", "status", "missing_dimensions", "clarification_question"],
    }


def research_planning_provider_from_environment() -> ResearchPlanningProvider | None:
    """Create the configured planner; capability flags do not disable it."""
    primary = _research_planning_provider(external_llm_config("RESEARCH_PLANNING"))
    fallback = _research_planning_provider(fallback_external_llm_config())
    if primary is not None and fallback is not None:
        return FallbackProposalProvider(primary, fallback)
    return primary or fallback


def _research_planning_provider(config: ExternalLLMConfig | None) -> ResearchPlanningProvider | None:
    if config is None or config.provider != "openai_compatible" or not config.base_url:
        return None
    parsed = urlparse(config.base_url + "/chat/completions")
    if parsed.scheme != "https" or not parsed.netloc:
        return None
    return OpenAICompatibleResearchPlanningProvider(
        config.api_key,
        model=config.model,
        endpoint=config.base_url + "/chat/completions",
        timeout=config.timeout,
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
    if provider is None or not (intent.complex or intent.orchestrate):
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
    except Exception:
        return ResearchPlan(query, intent, (original,), "unavailable", ("provider_failure",), requirements=base_requirements)
    variants, rejected = _validated_variants(proposal, original, intent)
    lanes = _validated_lanes(proposal)
    if lanes is None:
        rejected = (*rejected, "retrieval_lane_invalid")
        lanes = ("sparse",)
    _, requirement_rejections = _validated_requirements(
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
    information_needs, need_rejections = _validated_information_needs(proposal, intent)
    rejected = (*rejected, *need_rejections)
    clarification_question, missing_dimensions, clarification_rejections = _validated_clarification(proposal)
    rejected = (*rejected, *clarification_rejections)
    return ResearchPlan(
        query,
        intent,
        variants,
        "accepted" if not rejected else "degraded",
        rejected,
        lanes,
        tuple(base_requirements),
        task_kind,
        information_needs,
        clarification_question,
        missing_dimensions,
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
            "comparison_target_required": bool(
                intent.comparison and source_role and len(tuple(explicit_references)) < 2 and len(tuple(required_entities)) < 2
            ),
        },
        "allowed_proposals": (
            "task_kind", "variants", "information_needs", "retrieval_lanes",
            "status", "missing_dimensions", "clarification_question",
        ),
    }


def _validated_clarification(proposal: object) -> tuple[str | None, tuple[str, ...], tuple[str, ...]]:
    if not isinstance(proposal, Mapping):
        return None, (), ()
    status = proposal.get("status", "ready")
    if status == "ready":
        return None, (), ()
    dimensions = proposal.get("missing_dimensions")
    question = proposal.get("clarification_question")
    allowed = {"legal_target", "source_scope", "temporal_scope", "comparison_target"}
    if (
        status != "clarification_required"
        or not isinstance(dimensions, Sequence)
        or isinstance(dimensions, (str, bytes))
        or not dimensions
        or len(dimensions) > 2
        or not all(isinstance(value, str) and value in allowed for value in dimensions)
        or len(set(dimensions)) != len(dimensions)
        or not isinstance(question, str)
        or not 1 <= len(question.strip()) <= 240
    ):
        return None, (), ("clarification_invalid",)
    return question.strip(), tuple(dimensions), ()


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


def _validated_information_needs(
    proposal: object,
    intent: ResearchIntent,
) -> tuple[tuple[InformationNeed, ...], tuple[str, ...]]:
    """Accept bounded topic proposals only; evidence requirements stay server-owned."""
    if not isinstance(proposal, Mapping):
        return (), ()
    raw = proposal.get("information_needs", ())
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return (), ("information_needs_malformed",) if raw else ()
    allowed = {"description", "query", "concepts", "kind", "relation_traversal"}
    kinds = {"concept", "comparison", "procedure", "relation"}
    needs: list[InformationNeed] = []
    rejected: list[str] = []
    seen: set[tuple[str, str | None]] = set()
    for item in raw:
        if len(needs) >= max(1, intent.max_variants - 1):
            rejected.append("information_need_budget_exceeded")
            break
        if not isinstance(item, Mapping) or set(item) - allowed:
            rejected.append("information_need_invalid")
            continue
        description = item.get("description")
        query = item.get("query")
        concepts = item.get("concepts", ())
        kind = item.get("kind", "concept")
        traversal = item.get("relation_traversal", False)
        if (
            not isinstance(description, str)
            or not description.strip()
            or query is not None and (not isinstance(query, str) or not query.strip())
            or not _valid_string_sequence(concepts)
            or not isinstance(kind, str)
            or kind not in kinds
            or not isinstance(traversal, bool)
        ):
            rejected.append("information_need_invalid")
            continue
        key = (description.strip(), query.strip() if isinstance(query, str) else None)
        if key in seen:
            rejected.append("information_need_duplicate")
            continue
        seen.add(key)
        needs.append(
            InformationNeed(
                description=description.strip(),
                query=query.strip() if isinstance(query, str) else None,
                concepts=tuple(str(value).strip() for value in concepts if str(value).strip()),
                kind=kind,
                relation_traversal=traversal,
            )
        )
    return tuple(needs), tuple(rejected)


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
    plan: ResearchPlan | None = None,
) -> dict:
    """Run retrieval rounds, retrying only requirements still missing."""
    plan = plan or plan_research(
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
    if requirements and plan.requirements != tuple(requirements):
        plan = replace(plan, requirements=tuple(requirements))
    requirements = tuple(plan.requirements)
    rounds_limit = min(
        max(1, int(max_rounds if max_rounds is not None else plan.intent.max_rounds)),
        max(1, plan.intent.max_rounds),
    )
    rows: dict[str, dict] = {}
    routes: list[dict] = []
    requested_lanes = set(plan.retrieval_lanes)
    lane = (
        "hybrid"
        if "hybrid" in requested_lanes
        or {"sparse", "dense"} <= requested_lanes
        or (plan.intent.complex and plan.provider_status == "deterministic")
        else "dense"
        if "dense" in requested_lanes
        else "sparse"
    )
    plan = replace(plan, retrieval_lanes=(lane,))
    initial_lane = "sparse" if lane != "sparse" and rounds_limit > 1 else lane
    source_scoped = bool(requirements) and all(
        requirement.source_role and requirement.requirement_id.startswith("source_occurrence_")
        for requirement in requirements
    )
    if source_scoped:
        current_variants = _bounded_variant_lanes(
            tuple(
                QueryVariant(
                    query=requirement.retrieval_query or query,
                    origin=f"requirement:{requirement.requirement_id}",
                    required_entities=tuple(dict.fromkeys((*requirement.required_entities, *requirement.relation_endpoints))),
                    explicit_references=requirement.explicit_references,
                    source_role=requirement.source_role,
                    temporal_scope=requirement.temporal_context,
                    requirement_id=requirement.requirement_id,
                    retrieval_lane=initial_lane,
                )
                for requirement in requirements
            ),
            initial_lane,
        )
    else:
        current_variants = _bounded_variant_lanes(
            plan.variants[: max(1, plan.intent.max_variants)],
            initial_lane,
        )
    expensive_lane_used = initial_lane != "sparse"
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
            current_variants = _bounded_variant_lanes(
                tuple(
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
                ),
                lane,
                allow_expensive=not expensive_lane_used,
            )
            expensive_lane_used = expensive_lane_used or any(
                variant.retrieval_lane != "sparse" for variant in current_variants
            )
            if not current_variants:
                stop_reason = "no_progress"
                break
        else:
            if len(rows) > before:
                stop_reason = "complete"
                break
            if round_number + 1 < rounds_limit and lane != "sparse" and not expensive_lane_used:
                current_variants = _bounded_variant_lanes(plan.variants[: max(1, plan.intent.max_variants)], lane)
                expensive_lane_used = True
                continue
            stop_reason = "no_progress"
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


def _bounded_variant_lanes(
    variants: Sequence[QueryVariant],
    primary_lane: str,
    *,
    allow_expensive: bool = True,
) -> tuple[QueryVariant, ...]:
    """Pay the short-lived dense model startup cost at most once per request."""
    return tuple(
        replace(variant, retrieval_lane=primary_lane if index == 0 and allow_expensive else "sparse")
        for index, variant in enumerate(variants)
    )


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
    "InformationNeed",
    "ResearchIntent",
    "ResearchPlan",
    "ResearchPlanningProvider",
    "OpenAICompatibleResearchPlanningProvider",
    "research_planning_provider_from_environment",
    "execute_research",
    "execute_research_rounds",
    "plan_research",
]
