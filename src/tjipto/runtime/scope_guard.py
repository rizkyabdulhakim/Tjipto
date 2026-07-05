from __future__ import annotations

from tjipto.corpora.intent_config import contains_intent_phrase


def scope_guard_context(store, query: str) -> dict | None:
    if store is None:
        return None
    guard = store.config.setting("scope_guard", {}) or {}
    current = contains_intent_phrase(query, tuple(guard.get("current_fact_terms") or ()))
    subject = contains_intent_phrase(query, tuple(guard.get("current_fact_subjects") or ()))
    identity = contains_intent_phrase(query, tuple(guard.get("identity_question_terms") or ()))
    legal_scope = contains_intent_phrase(query, tuple(guard.get("legal_scope_terms") or ()))
    out_of_corpus = contains_intent_phrase(query, tuple(guard.get("out_of_corpus_terms") or ()))
    if subject and (current or (identity and not legal_scope)):
        return {"route": "current_fact_unsupported", "intent": "current_fact_query", "reason": "current_fact_unsupported"}
    if out_of_corpus:
        return {"route": "unsupported_scope", "intent": "out_of_corpus", "reason": "unsupported_scope"}
    return None
