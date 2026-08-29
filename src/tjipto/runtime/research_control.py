"""Deterministic ownership for legal-research intent and evidence requirements.

This module turns the immutable query contract and corpus policy into bounded
research intent.  It never retrieves, verifies, or publishes evidence.
"""

from __future__ import annotations

from collections.abc import Iterable
import re

from tjipto.corpora.intent_config import contains_intent_phrase, intent_config_for, normalize_intent_text
from tjipto.evidence.store import EvidenceStore
from tjipto.retrieval.bm25 import lexical_aliases, meaningful_tokens
from tjipto.retrieval.metadata import metadata_lookup
from tjipto.retrieval.relations import amendment_relation_lookup, amendment_relation_target, has_relation_target
from tjipto.retrieval.research import ResearchIntent
from tjipto.retrieval.sufficiency import EvidenceRequirement
from tjipto.retrieval.structured import structured_occurrence_references
from tjipto.runtime.query_semantics import QuerySemantics
from tjipto.runtime.research_support import (
    research_entities,
    research_entity_labels,
    research_focus_query,
    research_relation_terms,
    research_semantic_terms,
    research_semantic_term_groups,
    semantic_support_context_terms,  # noqa: F401 - compatibility export
    semantic_supports_text,  # noqa: F401 - compatibility export
    semantic_support_score,  # noqa: F401 - compatibility export
    single_support_covers_query,
)


_AUTHORITATIVE_RETRIEVAL_ROUTES = frozenset(
    {
        "document_relation",
        "relation",
        "structured",
        "exact",
        "metadata",
        "structural_navigation",
        "structure_list",
    }
)


def semantic_orchestration_required(store: EvidenceStore, query: str, semantics: QuerySemantics) -> bool:
    """Keep authoritative resolvers ahead of the bounded semantic planner."""
    if semantics.requested_function != "retrieval" and semantics.operation != "analyze":
        return False
    # Document summaries are assembled from the verified unit hierarchy. A
    # planner round can only return an incidental structural hit and may
    # reclassify the request as navigation, so keep summaries on their owner.
    if semantics.operation == "summarize":
        return False
    if semantics.operation == "analyze":
        return not semantics.legal_references
    if semantics.operation == "compare" and len(semantics.source_scopes) > 1:
        return semantics.capability_decision.reason_code == "current_fact_unsupported"
    if not any((semantics.requires_multiple_supports, semantics.requires_comparison, semantics.requires_decomposition, semantics.requires_graph)):
        research = _research_policy(store)
        hints = research.get("semantic_hints", {}) if isinstance(research, dict) else {}
        has_policy_signal = any(
            contains_intent_phrase(query, tuple(str(value) for value in values if isinstance(value, str)))
            for values in hints.values()
            if isinstance(values, (tuple, list))
        ) or bool(research_entities(research, query))
        if not has_policy_signal and single_support_covers_query(store, query):
            return False
    if metadata_lookup(store, query, 1):
        return False
    if amendment_relation_target(store, query).get("mode") is not None:
        return False
    return not has_relation_target(
        query,
        strategy=getattr(store.config, "query_strategy", "generic"),
        config=store.config,
    )


def authoritative_retrieval_route(routed: dict | None) -> bool:
    return bool(routed and routed.get("route") in _AUTHORITATIVE_RETRIEVAL_ROUTES)


def research_intent_for_ask(
    store: EvidenceStore,
    semantics: QuerySemantics,
    query: str,
    requirements: tuple[EvidenceRequirement, ...],
) -> ResearchIntent:
    """Derive complexity from server-owned requirements and corpus hints."""
    if semantics.requested_function not in {"retrieval", "source_discrepancy"} and semantics.operation not in {
        "analyze",
        "compare",
    }:
        return ResearchIntent()
    research = _research_policy(store)
    normalized_query = normalize_intent_text(query)

    def configured(name: str) -> bool:
        hints = research.get("semantic_hints", {})
        values = hints.get(name, ()) if isinstance(hints, dict) else ()
        return contains_intent_phrase(normalized_query, tuple(str(value) for value in values if isinstance(value, str)))

    entity_dimensions = tuple(requirement for requirement in requirements if requirement.required_entities)
    instrument_scopes = instrument_scope_roles(store, semantics.source_scopes)
    comparison = (len(entity_dimensions) > 1 or len(instrument_scopes) > 1) and (
        configured("comparison")
        or configured("authority")
        or configured("relation")
        or len(instrument_scopes) > 1
    )
    multiple = len(requirements) > 1 or any(requirement.min_supports > 1 for requirement in requirements)
    return ResearchIntent(
        multiple_supports=multiple,
        comparison=comparison,
        decomposition=multiple and (configured("procedure") or comparison),
        relation_traversal=bool(requirements) and configured("relation"),
        max_variants=_positive_int(research.get("max_variants"), 4),
        max_rounds=_positive_int(research.get("max_rounds"), 2),
    )


def semantic_scope_covered(
    store: EvidenceStore,
    semantics: QuerySemantics,
    query: str,
    requirements: tuple[EvidenceRequirement, ...],
) -> bool:
    """Ensure a research plan retains explicit corpus-backed dimensions."""
    if semantics.operation == "compare" and metadata_lookup(store, query, 1):
        return True
    if semantics.requested_function in {"exact_citation", "temporal_quotation"} or (
        semantics.requested_function == "proposition_verification" and len(semantics.legal_references) == 1
    ):
        return True
    research = _research_policy(store)
    normalized = normalize_intent_text(query)
    hints = research.get("semantic_hints", {})

    def hinted(name: str) -> bool:
        values = hints.get(name, ()) if isinstance(hints, dict) else ()
        return contains_intent_phrase(normalized, tuple(str(value) for value in values if isinstance(value, str)))

    entity_labels = set(research_entities(research, normalized))
    planned_entities = {entity for requirement in requirements for entity in requirement.required_entities}
    if len(entity_labels) > 1 and not entity_labels <= planned_entities:
        return False
    instrument_roles = set(instrument_scope_roles(store, semantics.source_scopes))
    planned_roles = {str(requirement.source_role) for requirement in requirements if requirement.source_role}
    occurrence_roles = {
        str(requirement.source_role)
        for requirement in requirements
        if requirement.requirement_id.startswith("source_occurrence_") and requirement.source_role
    }
    if occurrence_roles:
        if not occurrence_roles <= set(semantics.source_scopes):
            return False
    elif len(instrument_roles) > 1 and not instrument_roles <= planned_roles:
        return False
    if semantics.source_role and semantics.source_role not in planned_roles and requirements:
        return False
    complex_signal = hinted("comparison") or hinted("procedure") or hinted("relation")
    shared_entity_requirement = any(entity_labels <= set(requirement.required_entities) for requirement in requirements)
    if (
        len(entity_labels) > 1
        and (complex_signal or len(requirements) > 1)
        and len(requirements) < 2
        and not shared_entity_requirement
    ):
        return False
    if complex_signal and not requirements and not single_support_covers_query(store, query):
        return False
    return not (
        semantics.relation_intent
        and len(entity_labels) > 1
        and not entity_labels <= planned_entities
    )


def research_requirements_for_ask(
    store: EvidenceStore,
    semantics: QuerySemantics,
    query: str,
    *,
    information_needs: Iterable[object] = (),
) -> tuple[EvidenceRequirement, ...]:
    """Derive typed requirements from operations and corpus-backed policy."""
    historical = semantics.temporal_scope == "historical_pre_change"
    if semantics.requested_function != "retrieval" and semantics.operation not in {"compare", "analyze"} and not historical:
        return ()
    research = _research_policy(store)
    instrument_roles = instrument_scope_roles(store, semantics.source_scopes)
    if _source_occurrence_requested(store, semantics, query):
        return _source_occurrence_requirements(store, semantics, query)
    if semantics.operation == "compare" and "signatory_metadata" in semantics.targets:
        return _metadata_comparison_requirements(instrument_roles)
    if historical:
        return _historical_requirements(store, semantics, query)
    if semantics.operation == "analyze":
        return _analysis_requirements(store, semantics, query, research)
    broad_comparison = research.get("broad_version_comparison", {})
    evidence_roles = _string_tuple(broad_comparison.get("evidence_roles")) if isinstance(broad_comparison, dict) else ()
    comparison_roles = _string_tuple(broad_comparison.get("source_roles")) if isinstance(broad_comparison, dict) else ()
    if semantics.operation == "compare" and evidence_roles and set(semantics.source_scopes) == set(comparison_roles):
        return _instrument_comparison_requirements(store, semantics, query, research, evidence_roles)
    return _configured_requirements(store, semantics, query, tuple(information_needs), research, instrument_roles)


def _source_occurrence_requested(store: EvidenceStore, semantics: QuerySemantics, query: str) -> bool:
    intent = intent_config_for(getattr(store.config, "query_strategy", "generic"), store.config)
    return (
        semantics.operation == "search"
        and len(semantics.source_scopes) > 1
        and contains_intent_phrase(query, tuple(intent.get("all_source_scope_terms", ()) or ()))
    )


def _source_occurrence_requirements(
    store: EvidenceStore,
    semantics: QuerySemantics,
    query: str,
) -> tuple[EvidenceRequirement, ...]:
    groups = research_semantic_term_groups(store, query)
    structural_references = structured_occurrence_references(store, query)
    structured_authority_kinds = ("normative_legal_text", "structural_context")
    if structural_references:
        legal_units_by_role = {
            role: tuple(dict.fromkeys(
                str(row.get("legal_unit_id"))
                for row in store.evidence
                if row.get("source_role") == role
                and row.get("status") == "final"
                and row.get("authority_kind") in structured_authority_kinds
                and row.get("legal_unit_id")
                and any(
                    _normalise_reference(value) == _normalise_reference(row.get("citation"))
                    or _normalise_reference(value) == _normalise_reference(" ".join(row.get("hierarchy") or ()))
                    for value in structural_references
                )
            ))
            for role in semantics.source_scopes
        }
        legal_units_by_role = {role: ids for role, ids in legal_units_by_role.items() if ids}
        roles = tuple(legal_units_by_role)
        constrained = bool(legal_units_by_role)
        allow_partial = False
    else:
        legal_units_by_role = _source_occurrence_legal_units(store, query, semantics.legal_references)
        constrained = bool(legal_units_by_role)
        roles = (
            tuple(legal_units_by_role)
            if semantics.legal_references and constrained
            else tuple(semantics.source_scopes)
        )
        allow_partial = not bool(semantics.legal_references)
    catalog = store.config.setting("document_catalog", {}) or {}
    titles = catalog.get("titles", {}) if isinstance(catalog, dict) else {}
    return tuple(
        EvidenceRequirement(
            f"source_occurrence_{role}",
            description=str(titles.get(role) or role),
            retrieval_query=query,
            source_role=role,
            temporal_context=role,
            legal_unit_ids=legal_units_by_role.get(role, ("__source_occurrence_unmatched__",)) if constrained else (),
            semantic_term_groups=groups,
            authority_kinds=structured_authority_kinds if structural_references else ("normative_legal_text",),
            max_supports=8,
            allow_partial=allow_partial,
        )
        for role in roles
    )


def _source_occurrence_legal_units(
    store: EvidenceStore,
    query: str,
    references: tuple[str, ...] = (),
) -> dict[str, tuple[str, ...]]:
    """Constrain issue-occurrence retrieval to corpus-declared reference units."""
    if references:
        exact = _reference_units_by_role(store, references)
        if exact:
            return exact
    research = _research_policy(store)
    policy = research.get("operation_requirements", {})
    analysis = policy.get("analyze", {}) if isinstance(policy, dict) else {}
    if not isinstance(analysis, dict):
        return {}
    aliases = lexical_aliases(store.config)
    query_terms = meaningful_tokens(query, aliases=aliases)
    issue_terms = set(
        token
        for value in _string_tuple(analysis.get("issue_reference_terms"))
        for token in meaningful_tokens(value, aliases=aliases)
    )
    if not query_terms.intersection(issue_terms):
        groups = research_semantic_term_groups(store, query)
        if not groups:
            return {}
        return {
            str(role): tuple(
                dict.fromkeys(
                    str(row.get("legal_unit_id"))
                    for row in store.evidence
                    if row.get("source_role") == role
                    and row.get("authority_kind") == "normative_legal_text"
                    and row.get("legal_unit_id")
                    and any(str(value).casefold().startswith("pasal ") for value in row.get("hierarchy") or ())
                    and any(
                        set(group)
                        <= meaningful_tokens(
                            " ".join(
                                str(value or "")
                                for value in (row.get("citation"), row.get("hierarchy"), row.get("quoted_text"))
                            ),
                            aliases=aliases,
                        )
                        for group in groups
                    )
                )
            )
            for role in getattr(store.config, "source_roles", ())
            if any(
                row.get("source_role") == role
                and row.get("authority_kind") == "normative_legal_text"
                and row.get("legal_unit_id")
                and any(str(value).casefold().startswith("pasal ") for value in row.get("hierarchy") or ())
                and any(
                    set(group)
                    <= meaningful_tokens(
                        " ".join(
                            str(value or "")
                            for value in (row.get("citation"), row.get("hierarchy"), row.get("quoted_text"))
                        ),
                        aliases=aliases,
                    )
                    for group in groups
                )
                for row in store.evidence
            )
        }
    articles = tuple(
        str(value).split(" ayat ", 1)[0]
        for value in _string_tuple(analysis.get("issue_references"))
    )
    if not articles:
        return {}
    result: dict[str, tuple[str, ...]] = {}
    for role in getattr(store.config, "source_roles", ()):
        ids = tuple(
            dict.fromkeys(
                str(row.get("legal_unit_id"))
                for row in store.evidence
                if row.get("source_role") == role
                and row.get("legal_unit_id")
                and any(
                    article.casefold() in " ".join(str(value) for value in row.get("hierarchy") or ()).casefold()
                    for article in articles
                )
            )
        )
        if ids:
            result[str(role)] = ids
    intent = intent_config_for(getattr(store.config, "query_strategy", "generic"), store.config)
    wrappers = tuple(intent.get("source_occurrence_query_wrappers", ()) or ())
    if not contains_intent_phrase(query, wrappers):
        normalized = normalize_intent_text(query)
        separators = tuple(intent.get("source_occurrence_separators", ()) or ())
        prefix = next(
            (normalized.split(normalize_intent_text(separator), 1)[0] for separator in separators if normalize_intent_text(separator) in normalized),
            normalized,
        )
        prefix_tokens = tuple(prefix.split())
        anchor_phrase = " ".join(prefix_tokens[:5]) if len(prefix_tokens) > 5 else ""
        if anchor_phrase:
            for role, ids in tuple(result.items()):
                result[role] = tuple(
                    legal_unit_id
                    for legal_unit_id in ids
                    if any(
                        row.get("legal_unit_id") == legal_unit_id
                        and normalize_intent_text(anchor_phrase)
                        in normalize_intent_text(str(row.get("quoted_text") or ""))
                        for row in store.evidence
                    )
                )
            result = {role: ids for role, ids in result.items() if ids}
    return result


def _reference_units_by_role(store: EvidenceStore, references: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    requested = {_normalise_reference(value) for value in references if value}
    if not requested:
        return {}
    result: dict[str, tuple[str, ...]] = {}
    for role in getattr(store.config, "source_roles", ()):
        ids = tuple(
            dict.fromkeys(str(row.get("legal_unit_id")) for row in store.evidence if (
                row.get("source_role") == role
                and row.get("authority_kind") == "normative_legal_text"
                and row.get("citation_final") is True
                and _normalised_reference_labels(row).intersection(requested)
            )
        )
        )
        if ids:
            result[str(role)] = ids
    return result


def _normalised_reference_labels(row: dict) -> set[str]:
    hierarchy = tuple(str(value).strip() for value in row.get("hierarchy") or () if str(value).strip())
    labels = {_normalise_reference(row.get("citation")), _normalise_reference(" ".join(hierarchy))}
    if len(hierarchy) >= 2 and re.fullmatch(r"\(\d+\)", hierarchy[-1]):
        labels.add(_normalise_reference(f"{hierarchy[-2]} ayat {hierarchy[-1]}"))
    return {label for label in labels if label}


def _normalise_reference(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").casefold().strip())


def _metadata_comparison_requirements(instrument_roles: tuple[str, ...]) -> tuple[EvidenceRequirement, ...]:
    return tuple(
        EvidenceRequirement(
            f"signatory_{role}",
            description=f"{role} signatories",
            source_role=role,
            temporal_context=role,
            metadata_field="signatories",
        )
        for role in instrument_roles
    )


def _historical_requirements(
    store: EvidenceStore,
    semantics: QuerySemantics,
    query: str,
) -> tuple[EvidenceRequirement, ...]:
    target = next(iter(semantics.legal_references), "")
    _, relation_edges = amendment_relation_lookup(store, query)
    projection = next((edge.get("relation_projection") or {} for edge in relation_edges if edge.get("relation_projection")), {})
    role = str(projection.get("source_role") or "") or None
    normative_role = str(projection.get("target_source_role") or "") or role
    source_label = str(projection.get("source_label") or query)
    intent = intent_config_for(getattr(store.config, "query_strategy", "generic"), store.config)
    relation_families = intent.get("document_relation", {}).get("relation_families", {})
    deletion_policy = relation_families.get("DELETE_OR_REMOVE_PROVISION", {}) if isinstance(relation_families, dict) else {}
    deletion_terms = tuple(
        str(value).casefold()
        for value in deletion_policy.get("terms", ())
        if isinstance(value, str)
    )
    return (
        EvidenceRequirement(
            "historical_normative_text",
            description="historical normative text",
            retrieval_query=target or query,
            explicit_references=(target,) if target else (),
            legal_target=target or None,
            source_role=normative_role,
            temporal_context=normative_role,
            authority_kinds=("normative_legal_text",),
        ),
        EvidenceRequirement(
            "deletion_provenance",
            description="deletion/amendment provenance",
            retrieval_query=source_label,
            legal_target=target or None,
            source_role=role,
            temporal_context=role,
            authority_kinds=("instrument_provenance", "normative_legal_text"),
            support_terms=deletion_terms,
            requires_final_citation=True,
        ),
    )


def _analysis_requirements(
    store: EvidenceStore,
    semantics: QuerySemantics,
    query: str,
    research: dict,
) -> tuple[EvidenceRequirement, ...]:
    operation_policy = research.get("operation_requirements", {})
    analysis_policy = operation_policy.get("analyze", {}) if isinstance(operation_policy, dict) else {}
    if not isinstance(analysis_policy, dict):
        analysis_policy = {}
    source_role = str(analysis_policy.get("source_role") or "") or None
    issue_description = str(analysis_policy.get("issue_description") or "issue-relevant legal provisions")
    limitation_description = str(analysis_policy.get("limitation_description") or "applicable limitations/exceptions")
    references = tuple(str(value) for value in semantics.legal_references if value)
    if not references:
        issue_terms = _string_tuple(analysis_policy.get("issue_reference_terms"))
        aliases = lexical_aliases(store.config)
        matched_terms = meaningful_tokens(query, aliases=aliases) if issue_terms else set()
        policy_terms = {
            token
            for value in issue_terms
            for token in meaningful_tokens(value, aliases=aliases)
        }
        if len(matched_terms.intersection(policy_terms)) >= 2:
            references = _string_tuple(analysis_policy.get("issue_references"))
    requirements = [
        EvidenceRequirement(
            "analysis_issue_provisions" if index == 1 else f"analysis_issue_provision_{index}",
            description=issue_description,
            retrieval_query=reference,
            explicit_references=_reference_parts(reference),
            legal_target=None,
            source_role=source_role,
            temporal_context=source_role,
            authority_kinds=("normative_legal_text",),
        )
        for index, reference in enumerate(references, 1)
    ]
    limitation_map = analysis_policy.get("limitations_by_target", {})
    normalized_references = {normalize_intent_text(reference) for reference in references}
    if isinstance(limitation_map, dict) and references:
        limitations = tuple(
            str(value)
            for target, values in limitation_map.items()
            if normalize_intent_text(str(target)) in normalized_references
            for value in (values if isinstance(values, (tuple, list)) else ())
            if isinstance(value, str) and value.strip()
        )
    else:
        limitations = tuple(
            str(value)
            for value in analysis_policy.get("limitations_references", ())
            if isinstance(value, str) and value.strip()
        )
    requirements.extend(
        EvidenceRequirement(
            "analysis_limitations_exceptions" if index == 1 else f"analysis_limitation_exception_{index}",
            description=limitation_description,
            retrieval_query=reference,
            explicit_references=_reference_parts(reference),
            source_role=source_role,
            temporal_context=source_role,
            authority_kinds=("normative_legal_text",),
        )
        for index, reference in enumerate(limitations, 1)
    )
    relation_signals = tuple(
        str(value) for value in analysis_policy.get("relation_signals", ()) if isinstance(value, str) and value.strip()
    )
    if semantics.relation_intent or contains_intent_phrase(query, relation_signals):
        requirements.append(
            EvidenceRequirement(
                "analysis_historical_relation",
                description="historical/relation support",
                retrieval_query=query,
                source_role=source_role,
                temporal_context=source_role,
                authority_kinds=("normative_legal_text", "instrument_provenance"),
            )
        )
    return tuple(requirements)


def _reference_parts(reference: str) -> tuple[str, ...]:
    """Match article and optional paragraph without requiring one raw phrase."""
    match = re.fullmatch(
        r"\s*(Pasal\s+[0-9]+[A-Z]?)(?:\s+ayat\s*\(\s*([0-9]+)\s*\))?\s*",
        reference,
        flags=re.IGNORECASE,
    )
    if not match:
        return (reference,)
    article = re.sub(r"\s+", " ", match.group(1)).strip()
    paragraph = match.group(2)
    return (article, f"({paragraph})") if paragraph else (article,)


def _configured_requirements(
    store: EvidenceStore,
    semantics: QuerySemantics,
    query: str,
    information_needs: tuple[object, ...],
    research: dict,
    instrument_roles: tuple[str, ...],
) -> tuple[EvidenceRequirement, ...]:
    planning_terms = tuple(
        value
        for need in information_needs
        for value in (
            getattr(need, "query", None),
            getattr(need, "description", None),
            *tuple(getattr(need, "concepts", ()) or ()),
        )
        if isinstance(value, str) and value.strip()
    )
    normalized = normalize_intent_text(" ".join((query, *planning_terms)))
    generation = research.get("requirement_generation", {})
    if not isinstance(generation, dict):
        return ()
    segments = _query_segments(normalized, generation.get("conjunction_delimiters", ()))
    minimums = generation.get("minimum_supports", {})
    if not isinstance(minimums, dict):
        minimums = {}
    hints = research.get("semantic_hints", {})
    entities = research_entities(research, normalized)
    all_entities = research_entity_labels(research)

    def hinted(name: str) -> bool:
        values = hints.get(name, ()) if isinstance(hints, dict) else ()
        return contains_intent_phrase(normalized, tuple(str(value) for value in values if isinstance(value, str)))

    support_terms = research.get("support_terms", {})
    if not isinstance(support_terms, dict):
        support_terms = {}
    authority_terms = _string_tuple(support_terms.get("authority"))
    procedure_terms = _string_tuple(support_terms.get("procedure"))
    if (
        semantics.operation == "compare"
        and semantics.requested_function == "temporal_quotation"
        and len(semantics.source_scopes) > 1
        and semantics.legal_references
    ):
        return tuple(
            EvidenceRequirement(
                f"reference_{role}",
                description=reference,
                retrieval_query=reference,
                explicit_references=_reference_parts(reference),
                source_role=role,
                temporal_context=role,
                authority_kinds=("normative_legal_text",),
            )
            for role in semantics.source_scopes
            for reference in semantics.legal_references
        )
    if len(instrument_roles) > 1:
        return _instrument_comparison_requirements(store, semantics, query, research, instrument_roles)
    if len(semantics.legal_references) > 1:
        reference_role = semantics.source_role or (
            semantics.source_scopes[0] if len(semantics.source_scopes) == 1 else None
        )
        return tuple(
            EvidenceRequirement(
                f"reference_{index}",
                description=reference,
                retrieval_query=reference,
                explicit_references=(reference,),
                source_role=reference_role,
                temporal_context=reference_role,
            )
            for index, reference in enumerate(semantics.legal_references, 1)
        )
    if hinted("procedure") and (family := _procedure_family(research, normalized, entities)):
        return _procedure_requirements(store, query, research, family, entities, procedure_terms, minimums)
    if len(entities) > 1 and (hinted("comparison") or hinted("authority")):
        return tuple(
            EvidenceRequirement(
                f"entity_{index}",
                description=entity,
                retrieval_query=f"{entity} {' '.join(authority_terms)}".strip(),
                required_entities=(entity,),
                contrast_entities=tuple(value for value in all_entities if value != entity),
                support_terms=authority_terms,
                entity_must_lead=True,
                authority_kinds=("normative_legal_text",),
            )
            for index, entity in enumerate(entities, 1)
        )
    planner_requests_relation = any(
        bool(getattr(need, "relation_traversal", False))
        or str(getattr(need, "kind", "")).casefold() in {"relation", "multi_hop", "multi-hop"}
        for need in information_needs
    )
    if len(entities) > 1 and (hinted("relation") or planner_requests_relation):
        relation_terms = research_relation_terms(
            store,
            research,
            normalized if planner_requests_relation else query,
            entities,
        )
        return tuple(
            EvidenceRequirement(
                f"relation_{index}",
                description=entity,
                retrieval_query=f"{entity} {research_focus_query(store, research, query)}",
                required_entities=(entity,),
                relation_endpoints=entities,
                support_terms=relation_terms,
                authority_kinds=("normative_legal_text",),
                hierarchy_depth=None if planner_requests_relation else 3,
                allow_shared=len(entities) > 2,
            )
            for index, entity in enumerate(entities, 1)
        )
    if len(segments) > 1 and not single_support_covers_query(store, query):
        return tuple(
            EvidenceRequirement(
                f"dimension_{index}",
                description=segment,
                retrieval_query=research_focus_query(store, research, segment),
                semantic_terms=research_semantic_terms(store, research, segment, ()),
            )
            for index, segment in enumerate(segments, 1)
        )
    if entities and hinted("authority"):
        entity = entities[0]
        return (
            EvidenceRequirement(
                "authority",
                description=entity,
                retrieval_query=research_focus_query(store, research, query),
                required_entities=(entity,),
                contrast_entities=tuple(value for value in all_entities if value != entity),
                support_terms=authority_terms,
                entity_must_lead=True,
                authority_kinds=("normative_legal_text",),
                min_supports=_positive_int(minimums.get("authority"), 2),
                allow_partial=True,
            ),
        )
    return _planner_proposed_requirements(query, information_needs)


def _instrument_comparison_requirements(
    store: EvidenceStore,
    semantics: QuerySemantics,
    query: str,
    research: dict,
    instrument_roles: tuple[str, ...],
) -> tuple[EvidenceRequirement, ...]:
    role_labels = intent_config_for(
        getattr(store.config, "structured_strategy", "generic"),
        store.config,
    ).get("source_role_labels", {})
    summary = store.config.setting("document_summary", {}) or {}
    role_queries = summary.get("source_role_queries", {}) if isinstance(summary, dict) else {}
    requirements = tuple(
        EvidenceRequirement(
            f"instrument_{role}",
            description=role,
            retrieval_query=str(role_queries.get(role) or role_labels.get(role) or query),
            source_role=role,
            temporal_context=role,
            semantic_terms=tuple(meaningful_tokens(str(role_queries.get(role) or role_labels.get(role) or query))),
            authority_kinds=("instrument_provenance", "normative_legal_text"),
        )
        for role in instrument_roles
    )
    operation_policy = research.get("operation_requirements", {})
    change_terms = _string_tuple(
        operation_policy.get("change_relation_signals") if isinstance(operation_policy, dict) else ()
    )
    relation_required = bool(
        semantics.requires_graph
        or semantics.relation_intent
        or contains_intent_phrase(query, change_terms)
    )
    if not relation_required:
        return requirements
    return requirements + (
        EvidenceRequirement(
            "change_relations",
            description="change relations",
            retrieval_query=query,
            support_terms=change_terms,
            required_operation_terms=change_terms[:1],
            authority_kinds=("instrument_provenance", "normative_legal_text"),
            allow_shared=True,
        ),
    )


def _procedure_requirements(
    store: EvidenceStore,
    query: str,
    research: dict,
    family: str,
    entities: tuple[str, ...],
    procedure_terms: tuple[str, ...],
    minimums: dict,
) -> tuple[EvidenceRequirement, ...]:
    family_requirements = research.get("procedure_requirements_by_family", {})
    configured = family_requirements.get(family, ()) if isinstance(family_requirements, dict) else ()
    stages = configured or research.get("procedure_requirements", ())
    if isinstance(stages, (tuple, list)) and stages:
        return tuple(
            EvidenceRequirement(
                str(stage["requirement_id"]),
                description=str(stage.get("description") or stage["requirement_id"]),
                retrieval_query=" ".join(
                    (
                        *_string_tuple(stage.get("required_entities")),
                        *_string_tuple(stage.get("support_terms")),
                    )
                ),
                required_entities=_string_tuple(stage.get("required_entities")),
                support_terms=_string_tuple(stage.get("support_terms")),
                authority_kinds=("normative_legal_text",),
                hierarchy_depth=int(stage["hierarchy_depth"]) if stage.get("hierarchy_depth") else None,
            )
            for stage in stages
            if isinstance(stage, dict) and stage.get("requirement_id")
        )
    return (
        EvidenceRequirement(
            "procedure",
            description="procedure",
            retrieval_query=research_focus_query(store, research, query),
            required_entities=entities,
            semantic_terms=research_semantic_terms(store, research, query, entities),
            support_terms=procedure_terms,
            authority_kinds=("normative_legal_text",),
            min_supports=_positive_int(minimums.get("procedure"), 3),
        ),
    )


def _planner_proposed_requirements(
    query: str,
    information_needs: tuple[object, ...],
) -> tuple[EvidenceRequirement, ...]:
    proposed = []
    for index, need in enumerate(information_needs, 1):
        concepts = tuple(
            dict.fromkeys(
                token
                for concept in tuple(getattr(need, "concepts", ()) or ())
                if isinstance(concept, str)
                for token in meaningful_tokens(concept)
            )
        )
        if concepts:
            proposed.append(
                EvidenceRequirement(
                    f"information_{index}",
                    description=str(getattr(need, "description", "")),
                    retrieval_query=getattr(need, "query", None) or query,
                    support_terms=concepts,
                )
            )
    return tuple(proposed)


def _research_policy(store: EvidenceStore) -> dict:
    value = store.config.setting("research", {}) or {}
    return value if isinstance(value, dict) else {}


def _positive_int(value: object, default: int) -> int:
    if not isinstance(value, (str, int, float)):
        return default
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default


def _string_tuple(value: object) -> tuple[str, ...]:
    return tuple(str(item) for item in value or () if isinstance(item, str) and item.strip()) if isinstance(value, (tuple, list)) else ()


def _query_segments(normalized: str, delimiters: object) -> tuple[str, ...]:
    if not isinstance(delimiters, (tuple, list)):
        return ()
    for delimiter in delimiters:
        if not isinstance(delimiter, str):
            continue
        parts = tuple(part.strip() for part in normalized.split(f" {normalize_intent_text(delimiter)} ") if part.strip())
        if len(parts) > 1:
            return parts
    return ()


def _procedure_family(research: dict, query: str, entities: tuple[str, ...]) -> str | None:
    families = research.get("procedure_applicability", {})
    if not isinstance(families, dict):
        return None
    for name, family in families.items():
        if not isinstance(family, dict):
            continue
        required_entities = _string_tuple(family.get("required_entities"))
        signals = _string_tuple(family.get("signals"))
        if required_entities and not set(required_entities).intersection(entities):
            continue
        if signals and not contains_intent_phrase(query, signals):
            continue
        return str(name)
    return None


def instrument_scope_roles(store: EvidenceStore, source_scopes: tuple[str, ...]) -> tuple[str, ...]:
    """Keep only corpus-configured instrument roles from the control contract."""
    intent = intent_config_for(getattr(store.config, "query_strategy", "generic"), store.config)
    instrument_roles = set(intent.get("source_role_labels", {}))
    return tuple(role for role in source_scopes if role in instrument_roles)


def ambiguity_reason(semantics, routed: dict) -> str | None:
    """Fail closed on unresolved alternatives without inventing prompt copy."""
    if semantics.requires_comparison or not routed.get("matches"):
        return None
    route = str(routed.get("route") or "")
    query = str(routed.get("original_query") or "")
    alternatives = re.sub(r"\bdan\s+(?:/\s*)?atau\b", "", normalize_intent_text(query))
    if len(semantics.legal_references) > 1 and re.search(r"\batau\b", alternatives):
        return "ambiguous_legal_target"
    roles = {
        str(role)
        for role in (
            *(routed.get("metadata_source_roles") or ()),
            *(row.get("source_role") for row in routed.get("matches", ())),
        )
        if role
    }
    entity_occurrences = bool(routed.get("matches")) and all(
        row.get("fact_kind") == "person_role" and row.get("entity_identity")
        for row in routed.get("matches", ())
    ) and _query_names_person(query=str(routed.get("original_query") or ""), rows=routed.get("matches", ()))
    if (
        route in {"metadata", "metadata_fact", "metadata_scope_unresolved"}
        and not semantics.source_scopes
        and len(roles) > 1
        and not entity_occurrences
    ):
        return "ambiguous_source_scope"
    relation_types = tuple((routed.get("relation_target") or {}).get("relation_types") or ())
    relation_target = routed.get("relation_target") or {}
    document_scope_relation = relation_target.get("mode") == "document" and len(tuple(relation_target.get("related_roles") or ())) > 1
    if route in {"document_relation", "legal_relation"} and not (semantics.relation_intent or relation_types or document_scope_relation):
        return "ambiguous_relation_operation"
    if re.search(r"\batau\b", alternatives):
        return "ambiguous_target"
    if route in {"bm25", "lexical_fallback"}:
        if len(meaningful_tokens(query)) == 1 and len(routed.get("matches", ())) > 1:
            return "ambiguous_concept"
    return None


def _query_names_person(*, query: str, rows: tuple[dict, ...]) -> bool:
    """Allow an unscoped person lookup only when a person is actually named."""
    normalized_query = normalize_intent_text(query)
    for row in rows:
        name_tokens = normalize_intent_text(row.get("printed_name") or row.get("entity_identity")).split()
        if any(" ".join(name_tokens[index:]) in normalized_query for index in range(max(len(name_tokens) - 1, 0))):
            return True
    return False
