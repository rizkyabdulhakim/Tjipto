from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from tjipto.corpora.intent_config import contains_intent_phrase, normalize_intent_text
from tjipto.corpora.source_arbitration import (
    source_anomaly_clarification,
    source_anomaly_comparison_query,
    source_anomaly_response,
    source_reference_mappings_for_query,
)
from tjipto.evidence.store import EvidenceStore
from tjipto.retrieval.requirements import (
    _research_candidate_limit,
    authoritative_retrieval_route,
    research_entities,
    research_intent_for_ask,
    research_requirements_for_ask,
    semantic_orchestration_required,
    semantic_scope_covered,
)
from tjipto.retrieval.relations import amendment_relation_target
from tjipto.retrieval.research import ResearchPlan, plan_research
from tjipto.retrieval.structured import source_occurrence_route, structural_count
from tjipto.retrieval.sufficiency import EvidenceRequirement, assess_sufficiency, collect_evidence_set
from tjipto.runtime.answer_arbitration import (
    _answer_templates,
    _document_title,
    compound_query_parts,
    document_summary_query,
    instrument_intent_context,
    source_document_response,
)
from tjipto.runtime.evidence_answer import answer_text, evidence_answer
from tjipto.runtime.public_document import (
    consolidated_definition_response,
    historical_pre_change_response,
)
from tjipto.runtime.query_semantics import QuerySemantics, interpret_query
from tjipto.runtime.response import (
    _compound_response,
    _empty_query_response,
    _integrity_failure,
    instrument_response,
    structural_response,
)
from tjipto.runtime.scope_guard import scope_failure_response, scope_guard_context
from tjipto.runtime.source_text import source_text_response
from tjipto.runtime.viewer import _scope_has_verified_support

if TYPE_CHECKING:
    from tjipto.runtime.service import LegalRuntimeService


_MAPPING_RELATION_TYPES = frozenset({"RENAMES", "RENUMBERED_TO"})
_DIRECT_INSTRUMENT_RELATION_TYPES = frozenset({"DELETES", "SUPPLEMENTS", *_MAPPING_RELATION_TYPES})


@dataclass(frozen=True)
class _AskContext:
    service: "LegalRuntimeService"
    corpus_id: str
    query: str
    limit: int
    filters: dict | None
    evidence_requirements: tuple[EvidenceRequirement, ...]
    clarification_id: str | None
    clarification_answer: str | None
    summary_mode: bool
    clarification_round: int
    store: EvidenceStore
    semantics: QuerySemantics


@dataclass(frozen=True)
class _RoutePolicy:
    metadata_query: bool
    amendment_target: dict
    instrument_candidate: tuple | None
    source_occurrence_exact: bool
    source_occurrence_routed: dict | None


@dataclass(frozen=True)
class _ResearchState:
    active_requirements: tuple[EvidenceRequirement, ...]
    semantic_plan: ResearchPlan | None
    semantic_scope_loss: bool
    research_routed: dict | None


def execute_ask(
    service: "LegalRuntimeService",
    corpus_id: str,
    query: str,
    limit: int = 3,
    filters: dict | None = None,
    evidence_requirements: tuple[EvidenceRequirement, ...] = (),
    clarification_id: str | None = None,
    clarification_answer: str | None = None,
    summary_mode: bool = False,
) -> dict:
    resumed = service._resume_clarification(corpus_id, clarification_id, clarification_answer)
    if isinstance(resumed, dict):
        return resumed
    clarification_round = 0
    if isinstance(resumed, tuple):
        query, clarification_round = resumed
    if not query.strip():
        return _empty_query_response(corpus_id)
    store = service._store(corpus_id)
    if store is None:
        return _integrity_failure(corpus_id, query, service._integrity_error)
    early = _pre_semantic_response(service, store, corpus_id, query, clarification_round)
    if early is not None:
        return early
    context = _AskContext(
        service=service,
        corpus_id=corpus_id,
        query=query,
        limit=limit,
        filters=filters,
        evidence_requirements=evidence_requirements,
        clarification_id=clarification_id,
        clarification_answer=clarification_answer,
        summary_mode=summary_mode,
        clarification_round=clarification_round,
        store=store,
        semantics=interpret_query(store, corpus_id, query, available_corpora=service.registry.corpus_ids()),
    )
    direct = _direct_response(context)
    if direct is not None:
        return direct
    policy = _route_policy(context)
    instrument = _instrument_answer(context, policy)
    if instrument is not None:
        return instrument
    research = _research_state(context, policy)
    if isinstance(research, dict):
        return research
    return _grounded_answer(context, policy, research)


def _pre_semantic_response(
    service: "LegalRuntimeService",
    store: EvidenceStore,
    corpus_id: str,
    query: str,
    clarification_round: int,
) -> dict | None:
    anomaly_plan = source_anomaly_clarification(store, query)
    if anomaly_plan is not None and clarification_round == 0:
        return service._clarification_response(corpus_id, query, anomaly_plan, clarification_round)
    if source_anomaly_comparison_query(store, query):
        anomaly = source_anomaly_response(store, corpus_id, query)
        if anomaly is not None:
            return anomaly
    return source_text_response(store, corpus_id, query)


def _direct_response(context: _AskContext) -> dict | None:
    definition = consolidated_definition_response(
        context.corpus_id,
        context.query,
        context.store,
        context.semantics,
    )
    if definition is not None:
        return definition
    compound = _compound_answer(context)
    if compound is not None:
        return compound
    aggregate = structural_count(
        context.store,
        context.query,
        strategy=getattr(context.store.config, "query_strategy", "generic"),
    )
    if aggregate is not None:
        return structural_response(context.corpus_id, context.query, context.semantics, aggregate)
    summary = _summary_answer(context)
    if summary is not None:
        return summary
    if context.semantics.temporal_scope == "historical_pre_change":
        return historical_pre_change_response(
            context.corpus_id,
            context.query,
            context.store,
            context.semantics,
            route_retrieval=context.service._route_retrieval,
        )
    if context.clarification_round or context.summary_mode:
        return None
    return source_document_response(
        context.store,
        context.corpus_id,
        context.query,
        has_resolved_target=bool(context.semantics.legal_references),
        document_title=_document_title,
        insufficient_answer=_answer_templates(context.store)["insufficient"],
        semantics=context.semantics,
    )


def _compound_answer(context: _AskContext) -> dict | None:
    if context.evidence_requirements or context.clarification_id is not None or context.clarification_answer is not None:
        return None
    parts = compound_query_parts(
        context.query,
        semantics=context.semantics,
        config=context.store.config,
    )
    if not parts:
        return None
    responses = tuple(context.service._ask(context.corpus_id, part, context.limit, context.filters) for part in parts)
    return _compound_response(
        context.corpus_id,
        context.query,
        context.semantics,
        parts,
        responses,
    )


def _summary_answer(context: _AskContext) -> dict | None:
    normalized = document_summary_query(
        context.query,
        strategy=getattr(context.store.config, "query_strategy", "generic"),
        config=context.store.config,
        semantics=context.semantics,
    )
    intent_config = context.store.config.setting("intent_config", {}) or {}
    summary_roles = set(intent_config.get("source_role_labels", ()) if isinstance(intent_config, dict) else ())
    instrument_summary = context.semantics.operation == "summarize" and bool(
        summary_roles.intersection(context.semantics.source_scopes)
    )
    if not normalized or normalized == context.query:
        return None
    if context.semantics.operation == "summarize" and not instrument_summary:
        return None
    result = context.service._ask(
        context.corpus_id,
        normalized,
        context.limit,
        context.filters,
        context.evidence_requirements,
        summary_mode=True,
    )
    result = _promote_historical_summary(context, result)
    return result | {
        "original_query": context.query,
        "normalized_query": normalized,
        "operation": context.semantics.operation,
        "source_scopes": context.semantics.source_scopes,
        "temporal_scope": context.semantics.temporal_scope,
    }


def _promote_historical_summary(context: _AskContext, result: dict) -> dict:
    if context.semantics.operation != "summarize" or result.get("route") != "instrument_resolved_answerable":
        return result
    historical_citations = tuple(result.get("trace_support") or ())
    if not historical_citations:
        return result
    context_pack = result.get("context_pack")
    if isinstance(context_pack, dict):
        context_pack = context_pack | {"historical_citations": tuple(context_pack.get("trace_support") or ())}
    return result | {
        "status": "answer_ready",
        "historical_citations": historical_citations,
        "context_pack": context_pack,
        "answer_scope": "source_backed_summary",
    }


def _route_policy(context: _AskContext) -> _RoutePolicy:
    intent_config = context.store.config.setting("intent_config", {}) or {}
    candidate_target = amendment_relation_target(context.store, context.query)
    occurrence_query, occurrence_routed = _source_occurrence(context, intent_config)
    relation_config = intent_config.get("document_relation", {}) or {}
    explicit_change = contains_intent_phrase(
        context.query,
        tuple(intent_config.get("direct_relation_words", ()) or ()) + tuple(relation_config.get("add_terms", ()) or ()),
    )
    metadata_fields = intent_config.get("metadata_fields", {}) or {}
    metadata_query = (
        not occurrence_query
        and not explicit_change
        and not context.semantics.legal_references
        and context.semantics.operation != "compare"
        and any(target in metadata_fields for target in getattr(context.semantics, "targets", ()) or ())
    )
    amendment_target = {"mode": None} if metadata_query else candidate_target
    instrument_candidate = _instrument_candidate(context)
    return _RoutePolicy(
        metadata_query=metadata_query,
        amendment_target=amendment_target,
        instrument_candidate=instrument_candidate,
        source_occurrence_exact=occurrence_query and bool(context.semantics.legal_references),
        source_occurrence_routed=occurrence_routed,
    )


def _source_occurrence(context: _AskContext, intent_config: dict) -> tuple[bool, dict | None]:
    terms = tuple(intent_config.get("all_source_scope_terms", ()) or ()) + tuple(
        intent_config.get("source_occurrence_separators", ()) or ()
    )
    occurrence_query = (
        context.semantics.operation == "search"
        and len(context.semantics.source_scopes) > 1
        and contains_intent_phrase(context.query, terms)
    )
    if not occurrence_query:
        return False, None
    routed = source_occurrence_route(
        context.store,
        context.corpus_id,
        context.query,
        context.semantics.legal_references,
        context.semantics.source_scopes,
    )
    return occurrence_query, (routed if context.semantics.legal_references or routed.get("matches") else None)


def _instrument_candidate(context: _AskContext) -> tuple | None:
    if (
        context.semantics.requested_function == "temporal_quotation"
        or context.semantics.legal_references
        or context.semantics.operation == "compare"
    ):
        return None
    return instrument_intent_context(context.store, context.query)


def _instrument_answer(context: _AskContext, policy: _RoutePolicy) -> dict | None:
    candidate = policy.instrument_candidate
    if not candidate:
        return None
    relation_types = set(policy.amendment_target.get("relation_types") or ())
    partial = (
        candidate[0] is None
        and candidate[1] == "instrument_unresolved"
        and policy.amendment_target.get("mode") == "article"
        and not _DIRECT_INSTRUMENT_RELATION_TYPES.intersection(relation_types)
    )
    if policy.amendment_target.get("mode") not in {None, "unsupported"} and not partial:
        return None
    return instrument_response(
        context.store,
        context.corpus_id,
        context.query,
        context.semantics,
        candidate,
        lambda status, evidence, templates: answer_text(
            context.store,
            status,
            evidence,
            templates,
        ),
    )


def _research_state(context: _AskContext, policy: _RoutePolicy) -> _ResearchState | dict:
    planner_request = context.clarification_round > 0 or semantic_orchestration_required(context.store, context.query, context.semantics)
    research_request = _research_required(context, policy, planner_request)
    semantic_plan = _semantic_plan(context, planner_request) if research_request else None
    requirements = tuple(context.evidence_requirements) or research_requirements_for_ask(
        context.store,
        context.semantics,
        context.query,
    )
    clarification = _planner_clarification(
        context,
        semantic_plan,
        requirements,
    )
    if clarification is not None:
        return clarification
    if not requirements:
        requirements = research_requirements_for_ask(
            context.store,
            context.semantics,
            context.query,
            information_needs=semantic_plan.information_needs if semantic_plan else (),
        )
    scope_loss = not semantic_scope_covered(
        context.store,
        context.semantics,
        context.query,
        requirements,
    )
    if context.clarification_round > 0 and context.semantics.source_role:
        scope_loss = False
    research_intent = replace(
        research_intent_for_ask(
            context.store,
            context.semantics,
            context.query,
            requirements,
        ),
        orchestrate=planner_request,
    )
    if semantic_plan is not None:
        semantic_plan = replace(
            semantic_plan,
            intent=research_intent,
            requirements=requirements,
        )
    routed = (
        _execute_research(context, planner_request, research_intent, requirements, semantic_plan, scope_loss) if research_request else None
    )
    return _ResearchState(requirements, semantic_plan, scope_loss, routed)


def _research_required(
    context: _AskContext,
    policy: _RoutePolicy,
    planner_request: bool,
) -> bool:
    return (
        not policy.metadata_query
        and not policy.source_occurrence_exact
        and (
            planner_request
            or context.semantics.operation in {"analyze", "compare", "trace"}
            or (len(context.semantics.source_scopes) > 1 and policy.amendment_target.get("mode") is None)
        )
    )


def _semantic_plan(context: _AskContext, planner_request: bool) -> ResearchPlan:
    planning_intent = replace(
        research_intent_for_ask(context.store, context.semantics, context.query, ()),
        orchestrate=planner_request,
        comparison=context.semantics.operation == "compare",
    )
    return plan_research(
        context.query,
        planning_intent,
        provider=context.service._planning_provider if planner_request else None,
        required_entities=research_entities(
            context.store.config.setting("research", {}) or {},
            normalize_intent_text(context.query),
        ),
        explicit_references=tuple(context.semantics.legal_references or ()),
        source_role=context.semantics.source_role,
        temporal_scope=context.semantics.temporal_context,
        polarity=(context.semantics.requested_proposition.polarity if context.semantics.requested_proposition else None),
        modality=(context.semantics.requested_proposition.modality if context.semantics.requested_proposition else None),
    )


def _planner_clarification(
    context: _AskContext,
    semantic_plan: ResearchPlan | None,
    requirements: tuple[EvidenceRequirement, ...],
) -> dict | None:
    if semantic_plan is None or not semantic_plan.clarification_question:
        return None
    useful_analysis = context.semantics.operation == "analyze" and (
        bool(semantic_plan.information_needs) or len(semantic_plan.variants) > 1
    )
    if requirements or useful_analysis:
        return None
    return context.service._clarification_response(
        context.corpus_id,
        context.query,
        semantic_plan,
        context.clarification_round,
    )


def _execute_research(
    context: _AskContext,
    planner_request: bool,
    research_intent,
    requirements: tuple[EvidenceRequirement, ...],
    semantic_plan: ResearchPlan | None,
    scope_loss: bool,
) -> dict | None:
    result = context.service.research(
        context.corpus_id,
        context.query,
        intent=research_intent,
        requirements=requirements,
        planning_provider=context.service._planning_provider if planner_request else None,
        limit=_research_candidate_limit(context.store, context.query, context.limit),
        required_entities=tuple(
            dict.fromkeys(
                value for requirement in requirements for value in (*requirement.required_entities, *requirement.relation_endpoints)
            )
        ),
        explicit_references=tuple(context.semantics.legal_references or ()),
        source_role=context.semantics.source_role,
        temporal_scope=context.semantics.temporal_context,
        polarity=(context.semantics.requested_proposition.polarity if context.semantics.requested_proposition else None),
        modality=(context.semantics.requested_proposition.modality if context.semantics.requested_proposition else None),
        plan=semantic_plan,
    )
    if not result.get("routes"):
        return None
    routed = dict(result["routes"][0])
    routed["matches"] = result.get("matches", ())
    routed["status"] = "found" if routed["matches"] else "no_results"
    if context.semantics.operation == "analyze":
        routed["route"] = "research"
    routed["research_plan"] = result.get("plan")
    routed["research_stop_reason"] = result.get("stop_reason")
    routed["semantic_scope_loss"] = scope_loss
    return routed


def _grounded_answer(
    context: _AskContext,
    policy: _RoutePolicy,
    research: _ResearchState,
) -> dict:
    candidate_limit = (
        _research_candidate_limit(context.store, context.query, context.limit)
        if context.semantics.operation == "search" and not context.semantics.legal_references
        else context.limit
    )
    scoped, failure = _scope_route(context, policy, candidate_limit)
    if failure is not None:
        return failure
    typed = _typed_route(context, policy, research.research_routed, scoped, candidate_limit)
    routed = _preferred_route(context, typed, research.research_routed)
    _annotate_route(context, routed, research.semantic_scope_loss)
    evidence_set = (
        collect_evidence_set(context.store, routed.get("matches", ()), research.active_requirements)
        if research.active_requirements
        else None
    )
    assessment = assess_sufficiency(evidence_set, research.active_requirements) if evidence_set is not None else None
    _attach_sufficiency(routed, evidence_set, assessment)
    routed["matches"] = tuple(
        {key: value for key, value in row.items() if not str(key).startswith("_")} for row in routed.get("matches", ())
    )
    return evidence_answer(
        context.store,
        context.query,
        context.semantics,
        routed,
        research.active_requirements,
        evidence_set,
        assessment,
        research.semantic_plan,
    )


def _scope_route(
    context: _AskContext,
    policy: _RoutePolicy,
    candidate_limit: int,
) -> tuple[dict | None, dict | None]:
    scope = scope_guard_context(
        context.store,
        context.query,
        capability=context.semantics.capability_decision,
    )
    if not scope:
        return None, None
    routed = context.service._route_retrieval(
        context.corpus_id,
        context.query,
        context.store,
        limit=candidate_limit,
        metadata_filters=context.filters,
        allow_navigation=context.semantics.requested_function != "temporal_quotation",
        allow_relation=(context.semantics.requested_function != "temporal_quotation" and not policy.metadata_query),
    )
    routed["original_query"] = context.query
    unsupported = (
        scope["route"] == "current_fact_unsupported"
        or context.semantics.capability_decision.missing_capabilities
        or not _scope_has_verified_support(context.store, routed)
    )
    if not unsupported:
        return routed, None
    return routed, scope_failure_response(
        scope,
        corpus_id=context.corpus_id,
        query=context.query,
        semantics=context.semantics,
        routed=routed,
        answer=_answer_templates(context.store)["insufficient"],
    )


def _typed_route(
    context: _AskContext,
    policy: _RoutePolicy,
    research_routed: dict | None,
    scoped_routed: dict | None,
    candidate_limit: int,
) -> dict:
    relation_routed = _relation_route(context, policy, candidate_limit)
    typed = scoped_routed or relation_routed or policy.source_occurrence_routed
    if typed is None:
        deterministic = context.service._route_retrieval(
            context.corpus_id,
            context.query,
            context.store,
            limit=candidate_limit,
            metadata_filters=_semantic_filters(context),
            allow_navigation=context.semantics.requested_function != "temporal_quotation",
            allow_relation=(context.semantics.requested_function != "temporal_quotation" and not policy.metadata_query),
        )
        typed = deterministic if research_routed is None or authoritative_retrieval_route(deterministic) else research_routed
    if _comparison_needs_research(context, research_routed):
        if research_routed is not None:
            return research_routed
    return typed


def _semantic_filters(context: _AskContext) -> dict:
    filters = dict(context.filters or {})
    if context.semantics.source_role and "source_role" not in filters:
        filters["source_role"] = context.semantics.source_role
    mapped_roles = tuple(
        str(mapping.get("source_role"))
        for mapping in source_reference_mappings_for_query(
            context.query,
            context.store.config,
        )
        if mapping.get("source_role")
    )
    if mapped_roles and "source_role" not in filters:
        filters["source_role"] = mapped_roles[0]
    return filters


def _relation_route(
    context: _AskContext,
    policy: _RoutePolicy,
    candidate_limit: int,
) -> dict | None:
    candidate = policy.instrument_candidate
    unresolved = (
        candidate
        and candidate[1] == "instrument_unresolved"
        and candidate[2] == "legal_object_unresolved"
        and not _MAPPING_RELATION_TYPES.intersection(set(policy.amendment_target.get("relation_types") or ()))
        and "atau" not in normalize_intent_text(context.query).split()
    )
    if policy.amendment_target.get("mode") is None or policy.metadata_query or unresolved:
        return None
    return context.service._route_retrieval(
        context.corpus_id,
        context.query,
        context.store,
        limit=candidate_limit,
        metadata_filters=context.filters,
        allow_navigation=False,
        allow_relation=True,
    )


def _comparison_needs_research(
    context: _AskContext,
    research_routed: dict | None,
) -> bool:
    if not research_routed or not research_routed.get("matches"):
        return False
    if context.semantics.operation != "compare":
        return False
    same_version = len(context.semantics.legal_references) > 1 and len(context.semantics.source_scopes) == 1
    cross_version = bool(context.semantics.legal_references) and len(context.semantics.source_scopes) > 1
    return same_version or cross_version


def _preferred_route(
    context: _AskContext,
    typed_routed: dict,
    research_routed: dict | None,
) -> dict:
    if research_routed is not None and (context.semantics.operation == "analyze" or not authoritative_retrieval_route(typed_routed)):
        return research_routed
    return typed_routed


def _annotate_route(
    context: _AskContext,
    routed: dict,
    semantic_scope_loss: bool,
) -> None:
    routed["operation"] = context.semantics.operation
    routed["source_scopes"] = context.semantics.source_scopes
    routed["legal_references"] = context.semantics.legal_references
    routed["temporal_scope"] = context.semantics.temporal_scope
    if semantic_scope_loss:
        routed["semantic_scope_loss"] = True


def _attach_sufficiency(routed: dict, evidence_set, assessment) -> None:
    if evidence_set is None or assessment is None:
        return
    routed["evidence_set"] = {
        "support_ids": tuple(str(row.get("evidence_id")) for row in evidence_set.supports),
        "assignments": evidence_set.assignments,
        "missing_requirement_ids": evidence_set.missing_requirement_ids,
        "missing_reasons": evidence_set.missing_reasons,
    }
    routed["sufficiency"] = {
        "status": assessment.status,
        "fulfilled_requirement_ids": assessment.fulfilled_requirement_ids,
        "missing_requirement_ids": assessment.missing_requirement_ids,
        "missing_reasons": assessment.missing_reasons,
        "retry_allowed": assessment.retry_allowed,
    }
