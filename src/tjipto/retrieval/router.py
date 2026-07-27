from __future__ import annotations

from tjipto.evidence.store import EvidenceStore
from tjipto.retrieval.candidates import merge_ranked
from tjipto.retrieval.dense import dense_search
from tjipto.retrieval.metadata import (
    filter_evidence,
    has_metadata_target,
    metadata_lookup,
    normalize_filters,
    public_filters,
    resolve_source_scope,
)
from tjipto.retrieval.query import classify_intent, normalize_query
from tjipto.retrieval.relations import has_relation_target, relation_lookup
from tjipto.retrieval.service import RetrievalService
from tjipto.retrieval.structured import has_instrument_target, has_structured_target, structured_failure_reason, structured_lookup


def route_retrieval(
    corpus_id: str,
    query: str,
    store: EvidenceStore | None,
    *,
    limit: int = 10,
    allow_bm25_after_citation_miss: bool = False,
    allow_navigation: bool = True,
    route: str = "auto",
    metadata_filters: dict | None = None,
) -> dict:
    config = getattr(store, "config", None)
    query_strategy = getattr(config, "query_strategy", "generic")
    structured_strategy = getattr(config, "structured_strategy", "generic")
    normalized = normalize_query(query, strategy=query_strategy, config=config)
    scope = resolve_source_scope(normalized["normalized_query"], strategy=query_strategy, config=config)
    filters = normalize_filters(metadata_filters, config=config)
    if "source_role" not in filters:
        if scope.explicit:
            filters = dict(filters) | {"source_role": scope.role}
    applied_filters = public_filters(filters)
    corpus_supported = store is not None
    intent = classify_intent(
        corpus_id,
        normalized["normalized_query"],
        corpus_supported=corpus_supported,
        strategy=query_strategy,
        config=config,
    )
    if (
        "source_role" not in filters
        and not scope.unresolved
        and scope.role
        and (intent["intent"] == "exact_citation" or has_structured_target(
            normalized["normalized_query"], strategy=structured_strategy, config=config
        ))
        and not has_instrument_target(normalized["normalized_query"], strategy=structured_strategy, config=config)
    ):
        filters = dict(filters) | {"source_role": scope.role}
    applied_filters = public_filters(filters)
    envelope = {
        "status": "no_results",
        "route": "no_results",
        "intent": intent["intent"],
        "corpus_id": corpus_id,
        "original_query": normalized["original_query"],
        "normalized_query": normalized["normalized_query"],
        "matches": (),
        "reason": None,
        "required_corpus": None,
        "metadata_filters": applied_filters,
        "applied_filters": applied_filters,
    }
    if store is None:
        return envelope | {
            "status": "unsupported_corpus",
            "route": "unsupported_corpus",
            "reason": "unsupported_corpus",
        }
    if filters.get("_error"):
        return envelope | {
            "status": "invalid_filter",
            "route": "no_results",
            "reason": filters["_error"],
            "invalid_filters": filters.get("_invalid_filters", ()),
        }
    if scope.unresolved and "source_role" not in filters and not has_relation_target(
        normalized["normalized_query"], strategy=structured_strategy, config=config
    ):
        if has_metadata_target(normalized["normalized_query"], strategy=query_strategy, config=config, store=store):
            source_roles = tuple(
                sorted({row.get("source_role") for row in store.document_metadata if row.get("source_role")})
            )
            return envelope | {
                "status": "no_results",
                "route": "metadata_scope_unresolved",
                "intent": "metadata_lookup",
                "reason": "unresolved_source_scope",
                "metadata_source_roles": source_roles,
            }
        return envelope | {
            "status": "no_results",
            "route": "scope_unresolved",
            "reason": "unresolved_source_scope",
        }
    if route == "dense":
        return envelope | dense_search(store, normalized["normalized_query"], limit)

    service = RetrievalService(store)
    relation_all = tuple(relation_lookup(store, normalized["normalized_query"], len(store.evidence)))
    relation = filter_evidence(relation_all, filters)
    if relation:
        ranked, trace = merge_ranked(store, {"relation": relation}, filters)
        return envelope | {
            "status": "found",
            "route": "relation",
            "intent": "legal_relation_lookup",
            "matches": ranked if all(row.get("metadata_field") == "signatories" for row in ranked) else ranked[:limit],
            "expansion_trace": trace,
        }
    if relation_all:
        return envelope | {
            "status": "no_results",
            "route": "no_results",
            "intent": "legal_relation_lookup",
            "reason": "filters_removed_all",
        }
    # Some corpus adapters expose an exact structural/instrument support for a
    # relation-shaped query (for example a deleted chapter).  Only fail the
    # generic legal-relation route after that configured structured operation
    # has had a chance to resolve it.
    if has_relation_target(normalized["normalized_query"], strategy=query_strategy, config=config) and not has_instrument_target(
        normalized["normalized_query"], strategy=structured_strategy, config=config
    ):
        return envelope | {
            "status": "no_results",
            "route": "relation_not_found",
            "intent": "legal_relation_lookup",
            "reason": "relation_not_found",
        }

    structured_all = tuple(
        structured_lookup(
            store,
            normalized["normalized_query"],
            len(store.evidence),
            strategy=structured_strategy,
            source_role=filters.get("source_role"),
            allow_navigation=allow_navigation,
        )
    )
    navigation_all = tuple(row for row in structured_all if row.get("candidate_type") == "structural_navigation_candidate")
    if navigation_all:
        navigation = filter_evidence(navigation_all, filters)
        if navigation:
            ranked, trace = merge_ranked(store, {"structured": navigation}, filters)
            return envelope | {
                "status": "found",
                "route": "structural_navigation",
                "intent": "structural_navigation",
                "matches": ranked[:limit],
                "expansion_trace": trace,
            }
        return envelope | {
            "status": "no_results",
            "route": "no_results",
            "intent": "structural_navigation",
            "reason": "filters_removed_all",
        }

    if intent["intent"] == "exact_citation" and not structured_all and not has_structured_target(
        normalized["normalized_query"], strategy=structured_strategy, config=config
    ):
        matches = tuple(service.citation(normalized["normalized_query"]))
        filtered = filter_evidence(matches, filters)
        if filtered:
            ranked, trace = merge_ranked(store, {"exact": filtered}, filters, expand_graph=False)
            return envelope | {
                "status": "found",
                "route": "exact",
                "matches": ranked[:limit],
                "expansion_trace": trace,
            }
        if matches:
            return envelope | {
                "status": "no_results",
                "route": "no_results",
                "reason": "filters_removed_all",
            }
        if not allow_bm25_after_citation_miss:
            return envelope | {
                "status": "citation_not_found",
                "route": "citation_not_found",
                "reason": "citation_not_found",
            }

    metadata_all = tuple(metadata_lookup(store, normalized["normalized_query"], len(store.metadata_grounding)))
    metadata = filter_evidence(metadata_all, filters)
    if metadata:
        if all(row.get("metadata_field") == "signatories" for row in metadata):
            return envelope | {
                "status": "found",
                "route": "metadata",
                "intent": "metadata_lookup",
                "reason": None,
                "matches": metadata,
                "metadata_source_roles": tuple(sorted({row.get("source_role") for row in metadata if row.get("source_role")})),
            }
        ranked, trace = merge_ranked(store, {"metadata": metadata}, filters)
        return envelope | {
            "status": "found",
            "route": "metadata",
            "intent": "metadata_lookup",
            "matches": ranked if all(row.get("metadata_field") == "signatories" for row in ranked) else ranked[:limit],
            "expansion_trace": trace,
            "metadata_source_roles": tuple(sorted({row.get("source_role") for row in metadata if row.get("source_role")})),
        }
    if metadata_all:
        return envelope | {
            "status": "no_results",
            "route": "no_results",
            "intent": "metadata_lookup",
            "reason": "filters_removed_all",
        }
    if has_metadata_target(normalized["normalized_query"], strategy=query_strategy, config=config, store=store):
        return envelope | {
            "status": "no_results",
            "route": "metadata_not_found",
            "intent": "metadata_lookup",
            "reason": "metadata_not_found",
        }

    structured = filter_evidence(structured_all, filters)
    if structured:
        complete_set = any(row.get("candidate_type") == "structural_complete_set" for row in structured)
        if complete_set:
            return envelope | {
                "status": "found",
                "route": "structured",
                "intent": "structured_lookup",
                "matches": structured,
                "expansion_trace": (),
            }
        # A dedicated structural heading is already the authoritative owner.
        # Do not expand it through page/graph neighbors and replace the heading
        # with a child provision in a structured lookup response.
        heading = tuple(
            row
            for row in structured
            if row.get("authority_kind") == "structural_context"
            and str(row.get("citation") or "").casefold() == normalized["normalized_query"].casefold()
        )
        if heading and len(heading) == len(structured):
            heading = tuple(
                row
                | {
                    "route_sources": ("structured",),
                    "candidate_type": row.get("candidate_type") or "legal_unit_candidate",
                }
                for row in heading
            )
            return envelope | {
                "status": "found",
                "route": "structured",
                "intent": "structured_lookup",
                "matches": heading[:limit],
                "expansion_trace": (),
            }
        ranked, trace = merge_ranked(store, {"structured": structured}, filters, expand_graph=False)
        structural_navigation = any(row.get("candidate_type") == "structural_navigation_candidate" for row in structured)
        structure_list = any(row.get("candidate_type") == "structural_list_candidate" for row in structured)
        structured_route = "exact" if intent["intent"] == "exact_citation" else "structured"
        return envelope | {
            "status": "found",
            "route": "structure_list" if structure_list else "structural_navigation" if structural_navigation else structured_route,
            "intent": "structural_navigation"
            if structural_navigation
            else "exact_citation"
            if intent["intent"] == "exact_citation"
            else "structured_lookup",
            "matches": ranked if structure_list else ranked[:limit],
            "expansion_trace": trace,
        }
    if structured_all:
        return envelope | {
            "status": "no_results",
            "route": "no_results",
            "intent": "no_results",
            "reason": "filters_removed_all",
        }
    if has_structured_target(normalized["normalized_query"], strategy=structured_strategy, config=config):
        return envelope | {
            "status": "no_results",
            "route": "structured_not_found",
            "intent": "structured_lookup",
            "reason": structured_failure_reason(store, normalized["normalized_query"], strategy=structured_strategy)
            or "structured_not_found",
        }

    matches = tuple(service.search(normalized["normalized_query"], len(store.evidence)))
    filtered = filter_evidence(matches, filters)
    if "source_role" not in filters:
        preferred = tuple(row for row in filtered if row.get("source_role") == scope.role)
        if preferred:
            filtered = preferred
    if filtered:
        # Lexical rows are candidates only; graph proximity cannot turn a
        # neighbouring provision into answer support.
        ranked, trace = merge_ranked(store, {"bm25": filtered}, filters, expand_graph=False)
        return envelope | {
            "status": "found",
            "route": "bm25",
            "intent": "natural_language",
            "matches": ranked[:limit],
            "expansion_trace": trace,
        }
    if matches:
        return envelope | {
            "status": "no_results",
            "route": "no_results",
            "intent": "no_results",
            "reason": "filters_removed_all",
        }
    return envelope | {"status": "no_results", "route": "no_results", "intent": "no_results", "reason": "no_results"}
