from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class InstrumentIntentDecision:
    corpus: str
    normalized_query: str
    role_family: str | None
    amendment: str | None
    target_status: str
    fallback_permission: bool
    reason: str
    target_citation: str | None = None


_GENERIC = {
    "document_target_words": (),
    "metadata_fields": {},
    "metadata_rules": {},
    "metadata_roles": (),
    "relation_words": (),
    "direct_relation_words": (),
    "pasal_parent_words": (),
    "relation_child_words": (),
    "relation_routes": {},
    "unsupported_relation_context_words": (),
    "instrument_scope_queries": (),
    "instrument_deletion_words": (),
    "instrument_deletion_evidence_words": (),
    "instrument_change_context_words": (),
    "instrument_citation_templates": {},
    "instrument_role_queries": {},
    "metadata_candidate_signals": (),
    "document_relation": {},
    "instrument_intent_matrix": {},
    "partial_signal_instrument_matrix": {},
    "instrument_like_boundary_matrix": {},
    "instrument_intent_invariant_matrix": {},
    "instrument_source_signals": (),
    "unresolved_source_scope_patterns": (),
    "instrument_content_signals": (),
    "instrument_effect_signals": (),
    "instrument_analysis_signals": (),
    "instrument_legal_object_signals": (),
    "instrument_change_signals": (),
    "source_role_labels": {},
    "temporal_current_terms": (),
    "structured_sections": (),
    "structural_navigation": {},
    "structured_lookup_enabled": False,
    "structure_list_terms": (),
    "structure_unit_type": "",
    "structure_detail_terms": (),
    "structure_request_terms": {},
    "clarification": {},
}


def intent_config_for(strategy: str | None, config=None) -> dict:
    raw = config.setting("intent_config") if config is not None else None
    if not raw:
        return _GENERIC
    return {
        "document_target_words": tuple(raw.get("document_target_words") or ()),
        "metadata_fields": {key: tuple(value) for key, value in (raw.get("metadata_fields") or {}).items()},
        "metadata_rules": {key: tuple(value) for key, value in (raw.get("metadata_rules") or {}).items()},
        "metadata_roles": tuple((row["role"], re.compile(row["pattern"], re.IGNORECASE)) for row in raw.get("metadata_roles", ())),
        "relation_words": tuple(raw.get("relation_words") or ()),
        "direct_relation_words": tuple(raw.get("direct_relation_words") or ()),
        "pasal_parent_words": tuple(raw.get("pasal_parent_words") or ()),
        "relation_child_words": tuple(raw.get("relation_child_words") or ()),
        "relation_routes": dict(raw.get("relation_routes") or {}),
        "unsupported_relation_context_words": tuple(raw.get("unsupported_relation_context_words") or ()),
        "instrument_scope_queries": tuple(raw.get("instrument_scope_queries") or ()),
        "instrument_deletion_words": tuple(raw.get("instrument_deletion_words") or ()),
        "instrument_deletion_evidence_words": tuple(raw.get("instrument_deletion_evidence_words") or ()),
        "instrument_change_context_words": tuple(raw.get("instrument_change_context_words") or ()),
        "instrument_citation_templates": dict(raw.get("instrument_citation_templates") or {}),
        "instrument_role_queries": {key: tuple(value) for key, value in (raw.get("instrument_role_queries") or {}).items()},
        "metadata_candidate_signals": tuple(raw.get("metadata_candidate_signals") or ()),
        "document_relation": dict(raw.get("document_relation") or {}),
        "instrument_intent_matrix": dict(raw.get("instrument_intent_matrix") or {}),
        "partial_signal_instrument_matrix": dict(raw.get("partial_signal_instrument_matrix") or {}),
        "instrument_like_boundary_matrix": dict(raw.get("instrument_like_boundary_matrix") or {}),
        "instrument_intent_invariant_matrix": dict(raw.get("instrument_intent_invariant_matrix") or {}),
        "instrument_source_signals": tuple(raw.get("instrument_source_signals") or ()),
        "unresolved_source_scope_patterns": tuple(
            re.compile(pattern, re.IGNORECASE) for pattern in raw.get("unresolved_source_scope_patterns", ())
        ),
        "instrument_content_signals": tuple(raw.get("instrument_content_signals") or ()),
        "instrument_effect_signals": tuple(raw.get("instrument_effect_signals") or ()),
        "instrument_analysis_signals": tuple(raw.get("instrument_analysis_signals") or ()),
        "instrument_legal_object_signals": tuple(raw.get("instrument_legal_object_signals") or ()),
        "instrument_change_signals": tuple(raw.get("instrument_change_signals") or ()),
        "source_role_labels": dict(raw.get("source_role_labels") or {}),
        "temporal_current_terms": tuple(raw.get("temporal_current_terms") or ()),
        "structured_sections": tuple(raw.get("structured_sections") or ()),
        "structural_navigation": {key: tuple(value) for key, value in (raw.get("structural_navigation") or {}).items()},
        "structured_lookup_enabled": bool(raw.get("structured_lookup_enabled")),
        "structure_list_terms": tuple(raw.get("structure_list_terms") or ()),
        "structure_unit_type": str(raw.get("structure_unit_type") or ""),
        "structure_detail_terms": tuple(raw.get("structure_detail_terms") or ()),
        "structure_request_terms": {
            key: tuple(value) for key, value in (raw.get("structure_request_terms") or {}).items()
        },
        "clarification": dict(raw.get("clarification") or {}),
    }


def normalize_intent_text(value: object) -> str:
    text = str(value or "").casefold()
    text = re.sub(r"(?<=\bke)-(?=\d+\b)", " ", text)
    text = re.sub(r"[^0-9a-z]+", " ", text)
    return " ".join(text.split())


def contains_intent_phrase(text: str, aliases: tuple[str, ...] | list[str]) -> bool:
    haystack = f" {normalize_intent_text(text)} "
    return any(f" {normalize_intent_text(alias)} " in haystack for alias in aliases)


def resolve_instrument_intent(query: str, intent: dict, *, corpus: str = "") -> InstrumentIntentDecision:
    normalized = normalize_intent_text(query)
    if not normalized:
        return InstrumentIntentDecision(corpus, normalized, None, None, "not_instrument", True, "not_instrument")
    role = next(
        (key for key, aliases in intent.get("instrument_role_queries", {}).items() if contains_intent_phrase(query, aliases)),
        None,
    )
    amendment = next((source_role for source_role, pattern in intent.get("metadata_roles", ()) if pattern.search(query or "")), None)
    valid_amendment_context = amendment is not None
    source_signal = valid_amendment_context
    analysis_signal = contains_intent_phrase(query, intent.get("instrument_analysis_signals", ()))
    metadata_signal = contains_intent_phrase(query, intent.get("metadata_candidate_signals", ()))
    if valid_amendment_context and analysis_signal:
        reason = "analysis_metadata_conflict" if metadata_signal else "unsupported_analysis_intent"
        return InstrumentIntentDecision(corpus, normalized, role, amendment, "instrument_unresolved", False, reason)
    if metadata_signal:
        return InstrumentIntentDecision(corpus, normalized, None, amendment, "not_instrument", True, "pure_metadata_intent")
    if contains_intent_phrase(query, intent.get("relation_words", ())):
        return InstrumentIntentDecision(corpus, normalized, None, None, "not_instrument", True, "not_instrument")
    content_signal = contains_intent_phrase(query, intent.get("instrument_content_signals", ()))
    effect_signal = contains_intent_phrase(query, intent.get("instrument_effect_signals", ()))
    object_signal = contains_intent_phrase(query, intent.get("instrument_legal_object_signals", ()))
    change_signal = contains_intent_phrase(query, intent.get("instrument_change_signals", ()))
    if role is None or amendment is None:
        if source_signal and effect_signal:
            return InstrumentIntentDecision(
                corpus, normalized, role, amendment, "instrument_unresolved", False, "effect_signal_unsupported"
            )
        if source_signal and content_signal:
            return InstrumentIntentDecision(
                corpus, normalized, role, amendment, "instrument_unresolved", False, "content_signal_unresolved"
            )
        if source_signal and (role is not None or object_signal or change_signal):
            return InstrumentIntentDecision(corpus, normalized, role, amendment, "instrument_unresolved", False, "instrument_unresolved")
        if source_signal:
            return InstrumentIntentDecision(corpus, normalized, role, amendment, "instrument_unresolved", False, "instrument_unresolved")
        return InstrumentIntentDecision(corpus, normalized, role, amendment, "not_instrument", True, "not_instrument")
    citation = _instrument_citation(intent, amendment, role, query)
    if not citation:
        return InstrumentIntentDecision(corpus, normalized, role, amendment, "instrument_unresolved", False, "instrument_unresolved")
    return InstrumentIntentDecision(
        corpus, normalized, role, amendment, "instrument_resolved_fail_closed", False, "instrument_resolved_fail_closed", citation
    )


def _instrument_citation(intent: dict, role: str, key: str, query: str) -> str:
    template = intent.get("instrument_citation_templates", {}).get(key, "")
    if not template:
        return ""
    values = {"ordinal": intent.get("source_role_labels", {}).get(role, "")}
    if "{clause}" in template:
        match = re.search(r"\b(?:clause|klausul|huruf|butir)\s*\(?([a-e])\)?|\(([a-e])\)", query or "", re.IGNORECASE)
        if not match:
            return ""
        values["clause"] = (match.group(1) or match.group(2)).lower()
    return template.format(**values)
