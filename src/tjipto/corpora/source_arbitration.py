"""Single deterministic owner for source and temporal scope selection."""

from __future__ import annotations

from dataclasses import dataclass
import re

from tjipto.corpora.intent_config import contains_intent_phrase, intent_config_for, normalize_intent_text


@dataclass(frozen=True)
class SourceScopeDecision:
    role: str | None
    state: str

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
    if len(roles) < 2:
        labels = intent.get("source_role_labels", {})
        explicit_instrument = re.search(
            r"\b(?:perubahan|amandemen)\s*(?:ke[-\s]*)?(?:pertama|kedua|ketiga|keempat|\d+|i{1,3}|iv)\b",
            query or "",
            re.IGNORECASE,
        )
        if explicit_instrument:
            for role, label in labels.items():
                if role in roles:
                    continue
                ordinal = str(label or "")
                if not ordinal:
                    continue
                forms = _ordinal_forms(ordinal)
                forms_pattern = "|".join(forms)
                if re.search(
                    rf"\b(?:dan|atau|serta|maupun|,|/)\s*(?:perubahan|amandemen)?\s*(?:ke[-\s]*)?(?:{forms_pattern})\b",
                    query or "",
                    re.IGNORECASE,
                ) or re.search(
                    rf"(?:^|[,\s])(?:{forms_pattern})\s*(?:dan|atau|serta|maupun|,|/)\b",
                    query or "",
                    re.IGNORECASE,
                ):
                    roles.append(role)
    return tuple(dict.fromkeys(roles))


def _ordinal_forms(label: str) -> tuple[str, ...]:
    """Return generic ordinal spellings used by configured source labels."""
    forms = {label.casefold()}
    values = {
        "pertama": ("1", "i"),
        "kedua": ("2", "ii", "dua"),
        "ketiga": ("3", "iii", "tiga"),
        "keempat": ("4", "iv", "empat"),
    }
    forms.update(values.get(label.casefold(), ()))
    return tuple(sorted(forms, key=len, reverse=True))


def resolve_source_scope(query: str, *, strategy: str = "generic", config=None) -> SourceScopeDecision:
    roles = source_roles_for_query(query, strategy=strategy, config=config)
    if roles:
        return SourceScopeDecision(roles[0], "explicit_resolved")
    intent = intent_config_for(strategy, config)
    if contains_intent_phrase(query, intent.get("temporal_current_terms", ())):
        return SourceScopeDecision(getattr(config, "preferred_source_role", None), "generic_post_amendment")
    if any(pattern.search(query or "") for pattern in intent["unresolved_source_scope_patterns"]):
        return SourceScopeDecision(None, "unresolved")
    if contains_intent_phrase(query, intent.get("instrument_source_signals", ())):
        return SourceScopeDecision(None, "unresolved")
    if _near_source_scope_match(query, intent):
        return SourceScopeDecision(None, "unresolved")
    return SourceScopeDecision(getattr(config, "preferred_source_role", None), "unscoped")


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
