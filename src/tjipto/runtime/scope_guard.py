from __future__ import annotations

from tjipto.corpora.capabilities import resolve_capability


def scope_guard_context(store, query: str, *, capability=None) -> dict | None:
    if store is None:
        return None
    config = getattr(store, "config", None)
    if config is None:
        return None
    capability = capability or resolve_capability(config, query, "exact_quotation", (config.corpus_id,))
    if capability.reason_code == "current_fact_unsupported":
        return {
            "route": "current_fact_unsupported",
            "intent": "current_fact_query",
            "requested_function": "current_fact",
            "target_reference": None,
            "legal_domain": None,
            "reason": "current_fact_unsupported",
        }
    if capability.reason_code:
        return {
            "route": "unsupported_scope",
            "intent": "out_of_corpus",
            "requested_function": capability.required_capabilities[0] if capability.required_capabilities else capability.requested_operation,
            "target_reference": None,
            "legal_domain": capability.legal_domain,
            "required_capabilities": capability.required_capabilities,
            "missing_capabilities": capability.missing_capabilities,
            "missing_corpora": capability.missing_corpora,
            "capability_decision": capability.public(),
            "reason": capability.reason_code,
        }
    return None
