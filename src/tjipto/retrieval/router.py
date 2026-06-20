from __future__ import annotations

from tjipto.evidence.store import EvidenceStore
from tjipto.retrieval.candidates import merge_ranked
from tjipto.retrieval.dense import dense_search
from tjipto.retrieval.metadata import filter_evidence, normalize_filters, public_filters
from tjipto.retrieval.query import classify_intent, normalize_query
from tjipto.retrieval.service import RetrievalService
from tjipto.retrieval.structured import structured_lookup


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
    normalized = normalize_query(query)
    filters = normalize_filters(metadata_filters)
    applied_filters = public_filters(filters)
    corpus_supported = store is not None
    intent = classify_intent(
        corpus_id,
        normalized["normalized_query"],
        corpus_supported=corpus_supported,
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
        "required_corpus": intent["required_corpus"],
        "metadata_filters": applied_filters,
        "applied_filters": applied_filters,
    }
    if store is None:
        return envelope | {
            "status": "unsupported_corpus",
            "route": "unsupported_corpus",
            "reason": "unsupported_corpus",
        }
    if intent["intent"] == "out_of_corpus":
        return envelope | {
            "status": "insufficient_corpus",
            "route": "insufficient_corpus",
            "reason": "out_of_corpus",
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

    structured_all = tuple(structured_lookup(store, normalized["normalized_query"], len(store.evidence)))
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
