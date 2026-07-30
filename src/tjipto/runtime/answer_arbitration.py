from __future__ import annotations

from collections.abc import Callable
from typing import Any

from tjipto.corpora.intent_config import contains_intent_phrase
from tjipto.corpora.intent_config import intent_config_for
from tjipto.retrieval.metadata import has_metadata_target, metadata_lookup
from tjipto.retrieval.relations import has_relation_target
from tjipto.corpora.source_arbitration import resolve_source_scope


def document_open_requested(query: str, *, config: object) -> bool:
    """Recognize only an adapter-declared document navigation operation."""
    terms = getattr(config, "setting", lambda *_: ())("document_open_terms", ()) or ()
    return contains_intent_phrase(query, terms)


def document_summary_query(query: str, *, strategy: str, config: object) -> str | None:
    """Normalize a summary operation into one adapter-owned retrieval query."""
    policy: dict = getattr(config, "setting", lambda *_: {})("document_summary", {}) or {}
    if not contains_intent_phrase(query, policy.get("query_terms", ())):
        return None
    if not contains_intent_phrase(query, policy.get("document_terms", ())):
        return None
    scope = resolve_source_scope(query, strategy=strategy, config=config)
    role_queries = policy.get("source_role_queries", {}) or {}
    if scope.explicit:
        normalized = role_queries.get(scope.role)
        return str(normalized) if normalized else None
    normalized = policy.get("default_query")
    return str(normalized) if normalized else None


def source_document_response(
    store: Any,
    corpus_id: str,
    query: str,
    *,
    has_resolved_target: bool,
    document_title: Callable[[object, dict], str],
    insufficient_answer: str,
) -> dict | None:
    """Resolve only an explicit document-open operation to a verified source."""
    config = getattr(store, "config", None)
    strategy = getattr(config, "query_strategy", "generic")
    intent = intent_config_for(strategy, config)
    if not document_open_requested(query, config=config):
        return None
    scope = resolve_source_scope(query, strategy=strategy, config=config)
    if not scope.explicit or has_resolved_target:
        return None
    if has_relation_target(query, strategy=strategy, config=config):
        return None
    relation_config = intent.get("document_relation", {})
    if not contains_intent_phrase(query, relation_config.get("target_document_terms", ())):
        return None
    if contains_intent_phrase(query, intent.get("instrument_analysis_signals", ())) or contains_intent_phrase(
        query, intent.get("instrument_effect_signals", ())
    ):
        return None
    if has_metadata_target(query, strategy=strategy, config=config, store=store) and _has_metadata_field_target(query, intent):
        return None
    metadata_rows = metadata_lookup(store, query, 1)
    if metadata_rows and metadata_rows[0].get("metadata_field") != "official_title":
        return None
    source = next((row for row in store.source_documents if row.get("source_role") == scope.role), None)
    if source is None:
        reason = "source_document_not_found"
        return {
            "status": "insufficient_evidence",
            "route": "source_document",
            "intent": "source_document_lookup",
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
        "viewer_target": {
            "action": "open_document",
            "source_document_id": source.get("source_document_id"),
        },
    }
    return {
        "status": "answer_ready",
        "route": "source_document",
        "intent": "source_document_lookup",
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


def _has_metadata_field_target(query: str, intent: dict) -> bool:
    return any(
        field != "official_title" and contains_intent_phrase(query, aliases)
        for field, aliases in intent.get("metadata_fields", {}).items()
    )
