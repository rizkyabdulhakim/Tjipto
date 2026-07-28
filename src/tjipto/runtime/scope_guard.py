from __future__ import annotations

from tjipto.retrieval.metadata import metadata_lookup
from tjipto.corpora.intent_config import contains_intent_phrase
from tjipto.corpora.parser_dispatch import resolve_navigation
from tjipto.runtime.intent import classify_legal_intent


def scope_guard_context(store, query: str) -> dict | None:
    if store is None:
        return None
    config = getattr(store, "config", None)
    if config is None:
        return None
    intent = classify_legal_intent(store, query)
    try:
        navigation = resolve_navigation(getattr(config, "corpus_id", ""), query)
    except ValueError:
        navigation = None
    guard = config.setting("scope_guard", {}) or {}
    identity_question = contains_intent_phrase(query, tuple(guard.get("identity_question_terms") or ()))
    current_subject = contains_intent_phrase(query, tuple(guard.get("current_fact_subjects") or ()))
    if navigation and identity_question and current_subject:
        return {
            "route": "current_fact_unsupported",
            "intent": "current_fact_query",
            "requested_function": "current_fact",
            "target_reference": None,
            "legal_domain": None,
            "reason": "current_fact_unsupported",
        }
    if navigation and not identity_question:
        return None
    if metadata_lookup(store, query, 1):
        return None
    if intent.answerability == "unsupported":
        return {
            "route": intent.route,
            "intent": intent.intent,
            "requested_function": intent.requested_function,
            "target_reference": intent.target_reference,
            "legal_domain": intent.legal_domain,
            "required_capabilities": intent.required_capabilities,
            "reason": intent.rejection_reason,
        }
    return None
