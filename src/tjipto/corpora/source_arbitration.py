"""Single deterministic owner for source and temporal scope selection."""

from __future__ import annotations

from dataclasses import dataclass
import re

from tjipto.corpora.intent_config import contains_intent_phrase, intent_config_for, normalize_intent_text


@dataclass(frozen=True)
class SourceScopeDecision:
    role: str | None
    state: str
    roles: tuple[str, ...] = ()

    @property
    def explicit(self) -> bool:
        return self.state in {"explicit_resolved", "explicit_current"}

    @property
    def temporal(self) -> bool:
        return self.state in {"explicit_resolved", "explicit_current", "generic_post_amendment"}

    @property
    def unresolved(self) -> bool:
        return self.state == "unresolved"


def source_roles_for_query(query: str, *, strategy: str = "generic", config=None) -> tuple[str, ...]:
    intent = intent_config_for(strategy, config)
    roles = [role for role, pattern in intent["metadata_roles"] if pattern.search(query or "")]
    preferred = getattr(config, "preferred_source_role", None)
    explicit_current_terms = getattr(config, "setting", lambda *_: ())("explicit_current_source_terms", ()) if config is not None else ()
    if (
        preferred in roles
        and any(role != preferred for role in roles)
        and not contains_intent_phrase(query, explicit_current_terms)
        and not any(
            re.search(rf"\b{re.escape(normalize_intent_text(connector))}\b", normalize_intent_text(query))
            for connector in intent.get("source_role_connectors", ())
            if normalize_intent_text(connector)
        )
        and not re.search(r"\bke\b", normalize_intent_text(query))
    ):
        # A word such as "berlaku" can describe the field being asked about,
        # not a request for the consolidated source.  A named historical
        # document takes precedence unless the current source is named too.
        roles = [role for role in roles if role != preferred]
    if len(roles) == 1:
        roles.extend(_coordinated_source_roles(query, intent, excluded=set(roles)))
    return tuple(dict.fromkeys(roles))


def _coordinated_source_roles(query: str, intent: dict, *, excluded: set[str]) -> tuple[str, ...]:
    aliases = intent.get("source_role_aliases", {})
    connectors = tuple(normalize_intent_text(value) for value in intent.get("source_role_connectors", ()) if value)
    source_terms = tuple(
        normalize_intent_text(value)
        for value in intent.get("document_relation", {}).get("source_terms", ())
        if value
    )
    if not isinstance(aliases, dict) or not connectors:
        return ()
    normalized = normalize_intent_text(query)
    raw_query = str(query or "").casefold()
    connector_pattern = "|".join(re.escape(value) for value in connectors)
    source_pattern = "|".join(re.escape(value) for value in source_terms)
    separator_pattern = str(intent.get("source_role_separator_pattern") or "")
    found = []
    for role, values in aliases.items():
        if role in excluded:
            continue
        alias_pattern = "|".join(
            re.escape(normalize_intent_text(value))
            for value in values
            if isinstance(value, str) and normalize_intent_text(value)
        )
        if not alias_pattern:
            continue
        optional_source = rf"(?:(?:{source_pattern})\s+)?" if source_pattern else ""
        after_connector = rf"\b(?:{connector_pattern})\s+{optional_source}(?:ke\s+)?(?:{alias_pattern})\b"
        before_connector = rf"\b(?:{alias_pattern})\s+(?:{connector_pattern})\b"
        after_separator = rf"(?:{separator_pattern})\s*{optional_source}(?:ke[-\s]*)?(?:{alias_pattern})\b"
        before_separator = rf"\b(?:{alias_pattern})\s*(?:{separator_pattern})"
        word_match = re.search(after_connector, normalized) or re.search(before_connector, normalized)
        separator_match = separator_pattern and (
            re.search(after_separator, raw_query) or re.search(before_separator, raw_query)
        )
        if word_match or separator_match:
            found.append(str(role))
    return tuple(found)


def resolve_source_scope(query: str, *, strategy: str = "generic", config=None) -> SourceScopeDecision:
    roles = source_roles_for_query(query, strategy=strategy, config=config)
    if roles:
        return SourceScopeDecision(roles[0], "explicit_resolved", roles)
    intent = intent_config_for(strategy, config)
    if contains_intent_phrase(query, intent.get("temporal_current_terms", ())):
        return SourceScopeDecision(getattr(config, "preferred_source_role", None), "generic_post_amendment", roles)
    # Treat a quantified post-amendment phrase as the configured current source
    # without adding another corpus-specific alias (for example, "setelah
    # semua amandemen").  The source and temporal markers remain config-owned.
    normalized = normalize_intent_text(query)
    temporal_markers = {
        token
        for phrase in intent.get("temporal_current_terms", ())
        for token in normalize_intent_text(phrase).split()
        if token in {"setelah", "sesudah", "pasca"}
    }
    source_markers = tuple(intent.get("document_relation", {}).get("source_terms", ()) or ())
    document_targets = tuple(intent.get("document_target_words", ()) or ())
    if (
        temporal_markers
        and any(re.search(rf"\b{re.escape(marker)}\b", normalized) for marker in temporal_markers)
        and contains_intent_phrase(normalized, source_markers)
        and contains_intent_phrase(normalized, document_targets)
    ):
        return SourceScopeDecision(getattr(config, "preferred_source_role", None), "generic_post_amendment", roles)
    if any(pattern.search(query or "") for pattern in intent["unresolved_source_scope_patterns"]):
        return SourceScopeDecision(None, "unresolved", roles)
    if contains_intent_phrase(query, intent.get("instrument_source_signals", ())):
        return SourceScopeDecision(None, "unresolved", roles)
    if _near_source_scope_match(query, intent):
        return SourceScopeDecision(None, "unresolved", roles)
    return SourceScopeDecision(getattr(config, "preferred_source_role", None), "unscoped", roles)


def source_role_for_query(query: str, *, strategy: str = "generic", config=None) -> str | None:
    decision = resolve_source_scope(query, strategy=strategy, config=config)
    return decision.role if decision.explicit else None


def initial_source_role(config) -> str | None:
    """Return the root role of the configured source-version chain."""
    intent = intent_config_for(getattr(config, "query_strategy", "generic"), config)
    predecessors = intent.get("source_role_predecessors", {}) or {}
    if not isinstance(predecessors, dict):
        return None
    roles = tuple(str(role) for role in getattr(config, "source_roles", ()) or () if role)
    referenced = {str(role) for role in predecessors.values() if role}
    candidates = tuple(role for role in roles if role in referenced and role not in predecessors)
    return candidates[0] if len(candidates) == 1 else None


def source_reference_mappings_for_query(query: str, config=None) -> tuple[dict, ...]:
    """Return configured printed-to-canonical mappings explicitly in *query*.

    Mapping policy stays corpus-owned.  A mapping is usable only when its
    printed reference and every configured context term are present, so a
    bare reference can never silently acquire the anomaly's source meaning.
    """
    normalized = normalize_intent_text(query)
    mappings = getattr(config, "setting", lambda *_: ())("source_reference_mappings", ()) if config is not None else ()

    def present(value: object) -> bool:
        alternatives = (
            normalize_intent_text(option)
            for option in str(value or "").split("|")
        )
        return any(
            phrase and re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", normalized)
            for phrase in alternatives
        )

    return tuple(
        dict(mapping)
        for mapping in mappings or ()
        if isinstance(mapping, dict)
        and present(mapping.get("raw_reference"))
        and all(present(term) for term in mapping.get("context_terms") or ())
    )


def _source_conflict_adapter(store):
    strategy = getattr(getattr(store, "config", None), "strategy", None)
    return getattr(strategy, "source_conflict_adapter", None)


def source_document_by_id(store, source_document_id: object) -> dict | None:
    return next(
        (
            row
            for row in getattr(store, "source_documents", ())
            if row.get("source_document_id") == source_document_id
        ),
        None,
    )


def source_conflict_viewer_evidence(store, evidence_id: str | None) -> tuple[dict | None, list[dict] | None]:
    adapter = _source_conflict_adapter(store)
    if adapter is None:
        return None, None
    return adapter._source_conflict_viewer_evidence(store, evidence_id)


def source_anomaly_clarification(store, query: str):
    adapter = _source_conflict_adapter(store)
    return adapter._source_anomaly_clarification(store, query) if adapter is not None else None


def source_anomaly_comparison_query(store, query: str) -> bool:
    adapter = _source_conflict_adapter(store)
    return bool(adapter and adapter._source_anomaly_comparison_query(store, query))


def source_anomaly_response(store, corpus_id: str, query: str) -> dict | None:
    adapter = _source_conflict_adapter(store)
    return adapter._source_anomaly_response(store, corpus_id, query) if adapter is not None else None


def attach_source_reference_provenance(store, query: str, response: dict) -> dict:
    adapter = _source_conflict_adapter(store)
    return adapter._attach_source_reference_provenance(store, query, response) if adapter is not None else response


def _near_source_scope_match(query: str, intent: dict) -> bool:
    tokens = normalize_intent_text(query).split()
    for index, token in enumerate(tokens):
        for candidate in _one_edit_candidates(token):
            mutated = tokens[:]
            mutated[index] = candidate
            candidate_query = " ".join(mutated)
            if any(pattern.search(candidate_query) for _, pattern in intent.get("metadata_roles", ())):
                return True
    return False


def _one_edit_candidates(token: str) -> tuple[str, ...]:
    if len(token) < 3:
        return ()
    deletion = tuple(token[:index] + token[index + 1 :] for index in range(len(token)))
    transposition = tuple(
        token[:index] + token[index + 1] + token[index] + token[index + 2 :]
        for index in range(len(token) - 1)
        if token[index] != token[index + 1]
    )
    return tuple(dict.fromkeys((*deletion, *transposition)))
