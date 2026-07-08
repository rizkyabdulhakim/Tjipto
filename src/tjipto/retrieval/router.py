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
)
from tjipto.retrieval.query import classify_intent, normalize_query
from tjipto.retrieval.relations import has_relation_target, relation_lookup
from tjipto.retrieval.service import RetrievalService
from tjipto.retrieval.structured import has_structured_target, structured_lookup


def route_retrieval(
    corpus_id: str,
    query: str,
    store: EvidenceStore | None,
    *,
    limit: int = 10,
    allow_bm25_after_citation_miss: bool = False,
    route: str = "auto",
    metadata_filters: dict | None = None,
) -> dict:
    config = getattr(store, "config", None)
    query_strategy = getattr(config, "query_strategy", "generic")
    structured_strategy = getattr(config, "structured_strategy", "generic")
    normalized = normalize_query(query, strategy=query_strategy, config=config)
    filters = normalize_filters(metadata_filters, config=config)
    applied_filters = public_filters(filters)
    corpus_supported = store is not None
    intent = classify_intent(
        corpus_id,
        normalized["normalized_query"],
        corpus_supported=corpus_supported,
        strategy=query_strategy,
        config=config,
    )
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
            "matches": ranked[:limit],
            "expansion_trace": trace,
        }
    if relation_all:
        return envelope | {
            "status": "no_results",
            "route": "no_results",
            "intent": "legal_relation_lookup",
            "reason": "filters_removed_all",
        }
    if has_relation_target(normalized["normalized_query"], strategy=query_strategy, config=config):
        return envelope | {
            "status": "no_results",
            "route": "relation_not_found",
            "intent": "legal_relation_lookup",
            "reason": "relation_not_found",
        }

    if intent["intent"] == "exact_citation":
        matches = tuple(service.citation(normalized["normalized_query"]))
        filtered = filter_evidence(matches, filters)
        if filtered:
            ranked, trace = merge_ranked(store, {"exact": filtered}, filters)
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

    metadata_all = tuple(metadata_lookup(store, normalized["normalized_query"], len(store.document_metadata)))
    metadata = filter_evidence(metadata_all, filters)
    if metadata:
        ranked, trace = merge_ranked(store, {"metadata": metadata}, filters)
        return envelope | {
            "status": "found",
            "route": "metadata",
            "intent": "metadata_lookup",
            "matches": ranked[:limit],
            "expansion_trace": trace,
        }
    if metadata_all:
        return envelope | {
            "status": "no_results",
            "route": "no_results",
            "intent": "metadata_lookup",
            "reason": "filters_removed_all",
        }
    if has_metadata_target(normalized["normalized_query"], strategy=query_strategy, config=config):
        return envelope | {
            "status": "no_results",
            "route": "metadata_not_found",
            "intent": "metadata_lookup",
            "reason": "metadata_not_found",
        }

    structured_all = tuple(
        structured_lookup(
            store,
            normalized["normalized_query"],
            len(store.evidence),
            strategy=structured_strategy,
        )
    )
    structured = filter_evidence(structured_all, filters)
    if structured:
        ranked, trace = merge_ranked(store, {"structured": structured}, filters)
        return envelope | {
            "status": "found",
            "route": "structured",
            "intent": "structured_lookup",
            "matches": ranked[:limit],
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
            "reason": "structured_not_found",
        }

    matches = tuple(service.search(normalized["normalized_query"], len(store.evidence)))
    filtered = filter_evidence(matches, filters)
    if filtered:
        ranked, trace = merge_ranked(store, {"bm25": filtered}, filters)
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
