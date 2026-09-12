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
    "change_terms": (),
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
    "source_role_labels": {},
    "source_role_aliases": {},
    "source_role_connectors": (),
    "source_role_predecessors": {},
    "source_role_separator_pattern": "",
    "all_source_scope_terms": (),
    "source_occurrence_separators": (),
    "source_occurrence_query_wrappers": (),
    "temporal_current_terms": (),
    "structured_sections": (),
    "structural_navigation": {},
    "structured_lookup_enabled": False,
    "structure_list_terms": (),
    "structure_count_terms": (),
    "structure_count_unit_type": "",
    "structure_count_units": {},
    "structure_count_all_source_terms": (),
    "structure_unit_type": "",
    "structure_detail_terms": (),
    "structure_request_terms": {},
}


def _unique_terms(values) -> tuple[str, ...]:
    """Keep runtime phrase sets ordered and duplicate-free."""
    return tuple(dict.fromkeys(
        value.strip()
        for value in values or ()
        if isinstance(value, str) and value.strip()
    ))


def intent_config_for(strategy: str | None, config=None) -> dict:
    raw = config.setting("intent_config") if config is not None else None
    if not raw:
        return _GENERIC
    direct_relation_words = _unique_terms(raw.get("direct_relation_words"))
    document_relation = dict(raw.get("document_relation") or {})
    change_terms = _canonical_change_terms(direct_relation_words, document_relation)
    relation_families = {
        key: dict(value)
        for key, value in (document_relation.get("relation_families") or {}).items()
        if isinstance(value, dict)
    }
    modify_family = dict(relation_families.get("MODIFY_PROVISION") or {})
    relation_change_terms = change_terms
    modify_family["terms"] = relation_change_terms
    relation_families["MODIFY_PROVISION"] = modify_family
    document_relation["relation_families"] = relation_families
    document_relation["change_terms"] = relation_change_terms
    metadata_fields = {key: tuple(value) for key, value in (raw.get("metadata_fields") or {}).items()}
    metadata_rules = dict(metadata_fields)
    metadata_rules.update({key: tuple(value) for key, value in (raw.get("metadata_rules") or {}).items()})
    structure_count_units = {
        key: dict(value) for key, value in (raw.get("structure_count_units") or {}).items()
    }
    default_count_type = str(raw.get("structure_count_unit_type") or "")
    default_count_terms = next(
        (
            tuple(unit.get("terms") or ())
            for unit in structure_count_units.values()
            if unit.get("unit_type") == default_count_type
        ),
        tuple(raw.get("structure_count_terms") or ()),
    )
    return {
        "document_target_words": tuple(raw.get("document_target_words") or ()),
        "metadata_fields": metadata_fields,
        "metadata_rules": metadata_rules,
        "metadata_roles": tuple((row["role"], re.compile(row["pattern"], re.IGNORECASE)) for row in raw.get("metadata_roles", ())),
        "relation_words": _unique_terms(raw.get("relation_words")),
        "direct_relation_words": direct_relation_words,
        "change_terms": change_terms,
        "pasal_parent_words": _unique_terms(raw.get("pasal_parent_words")),
        "relation_child_words": _unique_terms(raw.get("relation_child_words")),
        "relation_routes": dict(raw.get("relation_routes") or {}),
        "unsupported_relation_context_words": tuple(raw.get("unsupported_relation_context_words") or ()),
        "instrument_deletion_words": tuple(raw.get("instrument_deletion_words") or ()),
        "instrument_deletion_evidence_words": tuple(raw.get("instrument_deletion_evidence_words") or ()),
        "instrument_change_context_words": tuple(raw.get("instrument_change_context_words") or ()),
        "instrument_citation_templates": dict(raw.get("instrument_citation_templates") or {}),
        "instrument_role_queries": {
            key: _unique_terms(value)
            for key, value in (raw.get("instrument_role_queries") or {}).items()
        },
        "metadata_candidate_signals": tuple(raw.get("metadata_candidate_signals") or ()),
        "document_relation": document_relation,
        "instrument_source_signals": _unique_terms(raw.get("instrument_source_signals")),
        "unresolved_source_scope_patterns": tuple(
            re.compile(pattern, re.IGNORECASE) for pattern in raw.get("unresolved_source_scope_patterns", ())
        ),
        "instrument_content_signals": _unique_terms(raw.get("instrument_content_signals")),
        "instrument_effect_signals": _unique_terms(raw.get("instrument_effect_signals")),
        "instrument_analysis_signals": _unique_terms(raw.get("instrument_analysis_signals")),
        "instrument_legal_object_signals": _unique_terms(raw.get("instrument_legal_object_signals")),
        "source_role_labels": dict(raw.get("source_role_labels") or {}),
        "source_role_aliases": {key: tuple(value) for key, value in (raw.get("source_role_aliases") or {}).items()},
        "source_role_connectors": tuple(raw.get("source_role_connectors") or ()),
        "source_role_predecessors": dict(raw.get("source_role_predecessors") or {}),
        "source_role_separator_pattern": str(raw.get("source_role_separator_pattern") or ""),
        "all_source_scope_terms": tuple(raw.get("all_source_scope_terms") or ()),
        "source_occurrence_separators": tuple(raw.get("source_occurrence_separators") or ()),
        "source_occurrence_query_wrappers": tuple(raw.get("source_occurrence_query_wrappers") or ()),
        "temporal_current_terms": tuple(raw.get("temporal_current_terms") or ()),
        "structured_sections": tuple(raw.get("structured_sections") or ()),
        "structural_navigation": {key: tuple(value) for key, value in (raw.get("structural_navigation") or {}).items()},
        "structured_lookup_enabled": bool(raw.get("structured_lookup_enabled")),
        "structure_list_terms": tuple(raw.get("structure_list_terms") or ()),
        "structure_count_terms": default_count_terms,
        "structure_count_unit_type": default_count_type,
        "structure_count_units": structure_count_units,
        "structure_count_all_source_terms": tuple(raw.get("structure_count_all_source_terms") or ()),
        "structure_unit_type": str(raw.get("structure_unit_type") or ""),
        "structure_detail_terms": tuple(raw.get("structure_detail_terms") or ()),
        "structure_request_terms": {
            key: tuple(value) for key, value in (raw.get("structure_request_terms") or {}).items()
        },
    }


def _canonical_change_terms(direct_relation_words: tuple[str, ...], document_relation: dict) -> tuple[str, ...]:
    """Derive one change vocabulary from direct relation terms and add/source exclusions."""
    add_terms = _unique_terms(document_relation.get("add_terms"))
    source_terms = _unique_terms(document_relation.get("source_terms"))
    excluded = {term.casefold() for term in (*add_terms, *source_terms)}
    return _unique_terms(term for term in direct_relation_words if term.casefold() not in excluded)


_VALIDATION_INTENT_KEYS = (
    "instrument_scope_queries",
    "instrument_intent_matrix",
    "partial_signal_instrument_matrix",
    "instrument_like_boundary_matrix",
    "instrument_intent_invariant_matrix",
)


def validation_intent_config_for(strategy: str | None, config=None) -> dict:
    """Project validation matrices only to the corpus validator owner."""
    intent = dict(intent_config_for(strategy, config))
    raw = config.setting("intent_config") if config is not None else None
    if not isinstance(raw, dict):
        return intent
    source_terms = _source_role_phrases(intent)
    role_terms = tuple(intent.get("instrument_role_queries", {}).get("scope", ()))
    change_terms = _unique_terms(intent.get("change_terms"))
    legal_object_terms = _unique_terms(
        (
            *tuple(intent.get("instrument_legal_object_signals", ()) or ()),
            *tuple(intent.get("instrument_content_signals", ()) or ()),
        )
    )
    word_orders = {
        key: dict(raw.get(key) or {}).get("word_orders", ())
        for key in _VALIDATION_INTENT_KEYS
        if isinstance(raw.get(key), dict)
    }
    intent["instrument_scope_queries"] = role_terms
    intent["instrument_intent_matrix"] = {
        "role_family_terms": role_terms,
        "amendment_terms": source_terms,
        "word_orders": tuple(word_orders.get("instrument_intent_matrix", ())),
    }
    intent["partial_signal_instrument_matrix"] = {
        "legal_object_terms": legal_object_terms,
        "change_terms": change_terms,
        "source_terms": source_terms,
        "word_orders": tuple(word_orders.get("partial_signal_instrument_matrix", ())),
    }
    intent["instrument_like_boundary_matrix"] = {
        "content_terms": tuple(intent.get("instrument_content_signals", ()) or ()),
        "effect_terms": tuple(intent.get("instrument_effect_signals", ()) or ()),
        "source_terms": source_terms,
        "word_orders": tuple(word_orders.get("instrument_like_boundary_matrix", ())),
    }
    intent["instrument_intent_invariant_matrix"] = {
        "analysis_terms": tuple(intent.get("instrument_analysis_signals", ()) or ()),
        "valid_amendment_contexts": source_terms,
        "word_orders": tuple(word_orders.get("instrument_intent_invariant_matrix", ())),
    }
    return intent


def instrument_policy_for(strategy: str | None, config=None) -> dict:
    """Project instrument routing policy for the research-control owner."""
    intent = intent_config_for(strategy, config)
    relation = intent.get("document_relation", {}) or {}
    summary = config.setting("document_summary", {}) if config is not None else {}
    summary = summary if isinstance(summary, dict) else {}
    return {
        "role_labels": dict(intent.get("source_role_labels", {}) or {}),
        "role_queries": dict(summary.get("source_role_queries", {}) or {}),
        "role_prefix": str(relation.get("source_role_label_prefix", "")),
        "change_terms": tuple(intent.get("change_terms", ()) or ()),
    }


def _source_role_phrases(intent: dict) -> tuple[str, ...]:
    labels = intent.get("source_role_labels", {}) or {}
    aliases = intent.get("source_role_aliases", {}) or {}
    phrases: list[str] = []
    for role, label in labels.items():
        numeric_alias = next(
            (value for value in tuple(aliases.get(role, ()) or ()) if isinstance(value, str) and value.isdigit()),
            None,
        )
        values = _unique_terms((label, f"ke-{numeric_alias}" if numeric_alias else ""))
        for value in values:
            phrases.extend((f"perubahan {value}", f"amandemen {value}"))
    return _unique_terms(phrases)


def wording_scope_terms_for(config=None) -> dict[str, tuple[str, ...]]:
    """Project scope markers for the wording trust boundary from corpus config."""
    intent = intent_config_for(getattr(config, "query_strategy", "generic"), config)
    current = list(intent.get("temporal_current_terms", ()) or ())
    current.extend(config.setting("explicit_current_source_terms", ()) if config is not None else ())
    historical = list(_source_role_phrases(intent))
    preferred_role = getattr(config, "preferred_source_role", None)
    if preferred_role:
        current.append(str(preferred_role))
    historical.extend(
        str(role)
        for role in getattr(config, "source_roles", ()) or ()
        if role and role != preferred_role
    )
    labels = config.setting("viewer_source_status_labels", {}) if config is not None else {}
    for role, label in (labels or {}).items():
        raw_label = str(label or "")
        parentheticals = re.findall(r"\(([^)]*)\)", raw_label)
        normalized_label = normalize_intent_text(raw_label)
        if re.search(r"\bhistor\w*\b", normalized_label):
            historical.extend(parentheticals)
            historical.extend(re.findall(r"\bhistor\w*\b", normalized_label))
        if role == preferred_role:
            current.extend(parentheticals)
    return {
        "historical": _unique_terms(historical),
        "current": _unique_terms(current),
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
    instrument_roles = set(intent.get("source_role_labels", {}) or {})
    amendment = next(
        (
            source_role
            for source_role, pattern in intent.get("metadata_roles", ())
            if source_role in instrument_roles and pattern.search(query or "")
        ),
        None,
    )
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
    change_signal = contains_intent_phrase(query, intent.get("change_terms", ()))
    scope_terms = tuple(intent.get("instrument_role_queries", {}).get("scope", ()) or ())
    scope_signal = contains_intent_phrase(
        query,
        tuple(
            term
            for term in scope_terms
            if not contains_intent_phrase(term, intent.get("change_terms", ()))
        ),
    )
    if valid_amendment_context and role in {None, "scope"} and (content_signal or effect_signal) and not scope_signal:
        reason = "effect_signal_unsupported" if effect_signal else "content_signal_unresolved"
        return InstrumentIntentDecision(corpus, normalized, role, amendment, "instrument_unresolved", False, reason)
    if role is None or amendment is None:
        if source_signal and effect_signal:
            return InstrumentIntentDecision(
                corpus, normalized, role, amendment, "instrument_unresolved", False, "effect_signal_unsupported"
            )
        if source_signal and content_signal:
            return InstrumentIntentDecision(
                corpus, normalized, role, amendment, "instrument_unresolved", False, "content_signal_unresolved"
            )
        if source_signal and object_signal:
            return InstrumentIntentDecision(
                corpus, normalized, role, amendment, "instrument_unresolved", False, "legal_object_unresolved"
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
