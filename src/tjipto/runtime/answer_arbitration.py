from __future__ import annotations

from collections.abc import Callable
from typing import Any

from tjipto.corpora.intent_config import contains_intent_phrase, intent_config_for, resolve_instrument_intent
from tjipto.retrieval.answer import validate_answer_candidate
from tjipto.retrieval.metadata import metadata_lookup
from tjipto.retrieval.relations import has_relation_target
from tjipto.corpora.source_arbitration import resolve_source_scope


def document_open_requested(query: str, *, config: object) -> bool:
    """Recognize only an adapter-declared document navigation operation."""
    terms = getattr(config, "setting", lambda *_: ())("document_open_terms", ()) or ()
    return contains_intent_phrase(query, terms)


def document_summary_query(query: str, *, strategy: str, config: object, semantics=None) -> str | None:
    """Normalize a summary operation into one adapter-owned retrieval query."""
    if semantics is not None:
        return getattr(semantics, "operation_query", None) if getattr(semantics, "operation", None) == "summarize" else None
    policy: dict = getattr(config, "setting", lambda *_: {})("document_summary", {}) or {}
    if not contains_intent_phrase(query, policy.get("query_terms", ())):
        return None
    scope = resolve_source_scope(query, strategy=strategy, config=config)
    role_queries = policy.get("source_role_queries", {}) or {}
    if scope.explicit:
        normalized = role_queries.get(scope.role)
        return str(normalized) if normalized else None
    if not contains_intent_phrase(query, policy.get("document_terms", ())):
        return None
    normalized = policy.get("default_query")
    return str(normalized) if normalized else None


def instrument_intent_context(store: Any, query: str) -> tuple[dict | None, str, str] | None:
    """Resolve an adapter-owned instrument lookup before general retrieval."""
    config = getattr(store, "config", None)
    intent = intent_config_for(getattr(config, "structured_strategy", "generic"), config)
    decision = resolve_instrument_intent(query, intent, corpus=getattr(config, "corpus_id", ""))
    if metadata_lookup(store, query, 1) and decision.reason not in {
        "analysis_metadata_conflict",
        "unsupported_analysis_intent",
    }:
        return None
    if decision.target_status == "not_instrument":
        return None
    if decision.target_status == "instrument_unresolved":
        return None, "instrument_unresolved", decision.reason
    row = next(
        (
            item
            for item in store.evidence
            if item.get("source_role") == decision.amendment and item.get("citation") == decision.target_citation
        ),
        None,
    )
    if row is None:
        return None, "instrument_unresolved", decision.reason
    row = row | {
        "route_sources": ("structured",),
        "candidate_type": f"instrument_{decision.role_family}_candidate",
    }
    if validate_answer_candidate(store, row)[0]:
        return row, "instrument_resolved_answerable", "answer_evidence"
    return (
        row | {"forced_rejection_reason": "instrument_resolved_fail_closed"},
        "instrument_resolved_fail_closed",
        "instrument_resolved_fail_closed",
    )


def source_document_response(
    store: Any,
    corpus_id: str,
    query: str,
    *,
    has_resolved_target: bool,
    document_title: Callable[[object, dict], str],
    insufficient_answer: str,
    semantics=None,
) -> dict | None:
    """Resolve only an explicit document-open operation to a verified source."""
    config = getattr(store, "config", None)
    strategy = getattr(config, "query_strategy", "generic")
    intent = intent_config_for(strategy, config)
    if semantics is None:
        if not document_open_requested(query, config=config):
            return None
        scope = resolve_source_scope(query, strategy=strategy, config=config)
        scope_explicit = scope.explicit
        scope_role = scope.role
    else:
        if getattr(semantics, "operation", None) != "open_document":
            return None
        source_scopes = tuple(getattr(semantics, "source_scopes", ()) or ())
        scope_explicit = bool(source_scopes)
        scope_role = getattr(semantics, "source_role", None) or (source_scopes[0] if source_scopes else None)
    if has_resolved_target:
        return None
    if has_relation_target(query, strategy=strategy, config=config):
        return None
    if contains_intent_phrase(query, intent.get("instrument_analysis_signals", ())) or contains_intent_phrase(
        query, intent.get("instrument_effect_signals", ())
    ):
        return None
    if not scope_explicit:
        documents = tuple(
            {
                "source_document_id": source.get("source_document_id"),
                "source_role": source.get("source_role"),
                "temporal_context": source.get("temporal_context"),
                "document_title": document_title(store, source),
                "intent": "document_delivery",
                "viewer_target": {"action": "open_document", "source_document_id": source.get("source_document_id")},
            }
            for source in store.source_documents
        )
        return {
            "status": "answer_ready",
            "route": "source_document_collection",
            "intent": "document_delivery",
            "corpus_id": corpus_id,
            "original_query": query,
            "normalized_query": query.strip(),
            "reason": None,
            "answer_type": "source_document_collection",
            "answer": "Naskah sumber terverifikasi tersedia.",
            "document_source": None,
            "document_sources": documents,
            "citations": (),
            "final_citations": (),
            "historical_citations": (),
            "metadata_support": (),
            "structural_support": (),
            "trace_support": (),
            "viewer_refs": (),
            "metadata_facts": (),
            "evidence": (),
            "warnings": ("document_sources_have_no_legal_citation",),
            "insufficient_reasons": (),
        }
    source = next((row for row in store.source_documents if row.get("source_role") == scope_role), None)
    if source is None:
        reason = "source_document_not_found"
        return {
            "status": "insufficient_evidence",
            "route": "source_document",
            "intent": "document_delivery",
            "corpus_id": corpus_id,
            "original_query": query,
            "normalized_query": query.strip(),
            "reason": reason,
            "answer_type": "none",
            "answer": insufficient_answer,
            "document_source": None,
            "citations": (),
            "final_citations": (),
            "historical_citations": (),
            "metadata_support": (),
            "structural_support": (),
            "trace_support": (),
            "viewer_refs": (),
            "metadata_facts": (),
            "evidence": (),
            "warnings": (),
            "insufficient_reasons": (reason,),
        }
    title = document_title(store, source)
    document_source = {
        "source_document_id": source.get("source_document_id"),
        "source_role": source.get("source_role"),
        "temporal_context": source.get("temporal_context"),
        "document_title": title,
        "intent": "document_delivery",
        "viewer_target": {
            "action": "open_document",
            "source_document_id": source.get("source_document_id"),
        },
    }
    return {
        "status": "answer_ready",
        "route": "source_document",
        "intent": "document_delivery",
        "corpus_id": corpus_id,
        "original_query": query,
        "normalized_query": query.strip(),
        "reason": None,
        "answer_type": "source_document",
        "answer": f"Naskah sumber terverifikasi: {title}.",
        "document_source": document_source,
        "citations": (),
        "final_citations": (),
        "historical_citations": (),
        "metadata_support": (),
        "structural_support": (),
        "trace_support": (),
        "viewer_refs": (),
        "metadata_facts": (),
        "evidence": (),
        "warnings": ("document_source_has_no_legal_citation",),
        "insufficient_reasons": (),
    }
