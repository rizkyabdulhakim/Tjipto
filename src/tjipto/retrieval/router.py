from __future__ import annotations

from tjipto.evidence.store import EvidenceStore
from tjipto.retrieval.dense import dense_search
from tjipto.retrieval.query import classify_intent, normalize_query
from tjipto.retrieval.service import RetrievalService


def route_retrieval(
    corpus_id: str,
    query: str,
    store: EvidenceStore | None,
    *,
    limit: int = 10,
    allow_bm25_after_citation_miss: bool = False,
    route: str = "auto",
) -> dict:
    normalized = normalize_query(query)
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
    if route == "dense":
        return envelope | dense_search(store, normalized["normalized_query"], limit)

    service = RetrievalService(store)
    if intent["intent"] == "exact_citation":
        matches = tuple(service.citation(normalized["normalized_query"]))[:limit]
        if matches:
            return envelope | {"status": "found", "route": "exact", "matches": matches}
        if not allow_bm25_after_citation_miss:
            return envelope | {
                "status": "citation_not_found",
                "route": "citation_not_found",
                "reason": "citation_not_found",
            }

    matches = tuple(service.search(normalized["normalized_query"], limit))
    if matches:
        return envelope | {"status": "found", "route": "bm25", "intent": "natural_language", "matches": matches}
    return envelope | {"status": "no_results", "route": "no_results", "intent": "no_results", "reason": "no_results"}
