from __future__ import annotations

from tjipto.corpora.capabilities import resolve_capability
from tjipto.retrieval.answer import empty_context_pack


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


def scope_failure_response(
    scope: dict,
    *,
    corpus_id: str,
    query: str,
    semantics,
    routed: dict,
    answer: str,
) -> dict:
    """Project a capability failure without exposing unsupported evidence."""
    capability = semantics.capability_decision
    missing_corpora = capability.missing_corpora
    missing_domain = bool(missing_corpora)
    reason = "missing_corpus_support" if missing_domain else scope["reason"]
    context_pack = empty_context_pack(reason)
    return scope | {
        "status": "insufficient_evidence",
        "route": "missing_corpus" if missing_domain else scope["route"],
        "reason": reason,
        "reason_code": reason,
        "corpus_id": corpus_id,
        "original_query": query,
        "normalized_query": query.strip(),
        "required_corpus": None,
        "matches": (),
        "answer_type": "none",
        "answer": answer,
        "context_pack": context_pack,
        "evidence": (),
        "citations": (),
        "final_citations": (),
        "historical_citations": context_pack.get("historical_citations", ()),
        "metadata_support": context_pack.get("metadata_support", ()),
        "structural_support": context_pack.get("structural_support", ()),
        "trace_support": context_pack.get("trace_support", ()),
        "viewer_refs": (),
        "metadata_facts": (),
        "legal_relations": (),
        "answer_scope": "insufficient_evidence",
        "warnings": (),
        "insufficient_reasons": (reason,),
        "capability_decision": capability.public(),
        "available_corpora": semantics.available_corpora,
        "needed_corpora": missing_corpora,
        "missing_corpora": missing_corpora,
        "required_capabilities": capability.required_capabilities,
        "missing_capabilities": capability.missing_capabilities,
        "retrieval_attempted": True,
        "retrieval_route": routed["route"],
        "retrieval_candidate_count": len(routed["matches"]),
    }
