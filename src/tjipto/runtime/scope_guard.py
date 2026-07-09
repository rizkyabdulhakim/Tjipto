from __future__ import annotations

from tjipto.runtime.intent import classify_legal_intent


def scope_guard_context(store, query: str) -> dict | None:
    if store is None:
        return None
    intent = classify_legal_intent(store, query)
    if intent.answerability == "unsupported":
        return {
            "route": intent.route,
            "intent": intent.intent,
            "requested_function": intent.requested_function,
            "target_reference": intent.target_reference,
            "legal_domain": intent.legal_domain,
            "reason": intent.rejection_reason,
        }
    return None
