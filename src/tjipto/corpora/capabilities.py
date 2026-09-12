"""Corpus-configured capability decisions with no invented corpus IDs."""

from __future__ import annotations

from dataclasses import dataclass
import re

from tjipto.corpora.intent_config import contains_intent_phrase


@dataclass(frozen=True)
class CapabilityDecision:
    requested_operation: str
    legal_domain: str | None
    required_capabilities: tuple[str, ...]
    eligible_corpora: tuple[str, ...]
    missing_capabilities: tuple[str, ...]
    missing_corpora: tuple[str, ...]
    confidence: str
    reason_code: str | None

    def public(self) -> dict:
        return {
            "requested_operation": self.requested_operation,
            "legal_domain": self.legal_domain,
            "required_capabilities": self.required_capabilities,
            "eligible_corpora": self.eligible_corpora,
            "missing_capabilities": self.missing_capabilities,
            "missing_corpora": self.missing_corpora,
            "confidence": self.confidence,
            "reason_code": self.reason_code,
        }


def resolve_capability(
    config,
    query: str,
    requested_operation: str,
    available_corpora: tuple[str, ...],
) -> CapabilityDecision:
    """Return only requirements declared by this corpus's trusted policy."""
    guard = config.setting("scope_guard", {}) or {}
    if requested_operation != "structural_navigation" and contains_intent_phrase(query, tuple(guard.get("current_fact_subjects") or ())) and (
        contains_intent_phrase(query, tuple(guard.get("current_fact_terms") or ()))
        or contains_intent_phrase(query, tuple(guard.get("identity_question_terms") or ()))
    ):
        return CapabilityDecision(
            requested_operation="current_nonlegal_fact",
            legal_domain=None,
            required_capabilities=("current_fact",),
            eligible_corpora=(),
            missing_capabilities=("current_fact",),
            missing_corpora=(),
            confidence="high",
            reason_code="current_fact_unsupported",
        )
    # A quantitative question without any corpus/legal anchor is an external
    # fact request.  Reject it before BM25 can return a merely adjacent legal
    # definition (for example, the constitutional definition of penduduk).
    metadata_fields = config.setting("intent_config", {}).get("metadata_fields", {}) or {}
    metadata_terms = tuple(
        value
        for values in metadata_fields.values()
        if isinstance(values, (tuple, list))
        for value in values
        if isinstance(value, str)
    )
    intent_policy = config.setting("intent_config", {}) or {}
    instrument_roles = intent_policy.get("instrument_role_queries", {}) if isinstance(intent_policy, dict) else {}
    metadata_terms += tuple(
        value
        for value in (instrument_roles.get("signatories", ()) if isinstance(instrument_roles, dict) else ())
        if isinstance(value, str)
    )
    lexical = config.setting("lexical_normalization", {}) or {}
    related_terms = lexical.get("related_terms", {}) if isinstance(lexical, dict) else {}
    configured_context_terms = tuple(
        term
        for key, values in (related_terms.items() if isinstance(related_terms, dict) else ())
        for term in (key, *(values if isinstance(values, (tuple, list)) else ()))
        if isinstance(term, str)
    )
    corpus_context_terms = (
        tuple(guard.get("legal_scope_terms") or ())
        + tuple(guard.get("in_corpus_discovery_terms") or ())
        + tuple(config.setting("document_target_words", ()) or ())
        + configured_context_terms
    )
    if re.search(r"\b(?:berapa|jumlah|total)\b", query.casefold()) and not contains_intent_phrase(
        query, corpus_context_terms
    ) and not contains_intent_phrase(query, metadata_terms):
        return CapabilityDecision(
            requested_operation="current_nonlegal_fact",
            legal_domain=None,
            required_capabilities=("current_fact",),
            eligible_corpora=(),
            missing_capabilities=("current_fact",),
            missing_corpora=(),
            confidence="high",
            reason_code="current_fact_unsupported",
        )
    policy = guard.get("legal_intent_policy", {}) or {}
    for rule in policy.get("unsupported_functions", ()):
        topic = contains_intent_phrase(query, tuple(rule.get("topic_terms") or ()))
        unsupported = contains_intent_phrase(query, tuple(rule.get("unsupported_function_terms") or ()))
        ambiguous = contains_intent_phrase(query, tuple(rule.get("ambiguous_criminal_terms") or ()))
        supported = contains_intent_phrase(query, tuple(rule.get("supported_function_terms") or ()))
        target = contains_intent_phrase(query, tuple(rule.get("target_reference_terms") or ()))
        if (unsupported and (topic or target)) or (topic and ambiguous and not supported):
            capability = str(rule.get("requested_function") or "external_legal_research")
            return CapabilityDecision(
                requested_operation="external_legal_domain_research",
                legal_domain=str(rule.get("legal_domain") or "") or None,
                required_capabilities=(capability,),
                eligible_corpora=(),
                missing_capabilities=(capability,),
                missing_corpora=(),
                confidence="high",
                reason_code=str(rule.get("rejection_reason") or "unsupported_legal_function"),
            )
    return CapabilityDecision(
        requested_operation=requested_operation,
        legal_domain=None,
        required_capabilities=(),
        eligible_corpora=tuple(corpus_id for corpus_id in available_corpora if corpus_id == config.corpus_id),
        missing_capabilities=(),
        missing_corpora=(),
        confidence="high",
        reason_code=None,
    )
