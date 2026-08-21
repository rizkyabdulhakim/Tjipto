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


def source_reference_mappings_for_query(query: str, config=None) -> tuple[dict, ...]:
    """Return configured printed-to-canonical mappings explicitly in *query*.

    Mapping policy stays corpus-owned.  A mapping is usable only when its
    printed reference and every configured context term are present, so a
    bare reference can never silently acquire the anomaly's source meaning.
    """
    normalized = normalize_intent_text(query)
    mappings = getattr(config, "setting", lambda *_: ())("source_reference_mappings", ()) if config is not None else ()

    def present(value: object) -> bool:
        phrase = normalize_intent_text(str(value or ""))
        return bool(phrase and re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", normalized))

    return tuple(
        dict(mapping)
        for mapping in mappings or ()
        if isinstance(mapping, dict)
        and present(mapping.get("raw_reference"))
        and all(present(term) for term in mapping.get("context_terms") or ())
    )


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
