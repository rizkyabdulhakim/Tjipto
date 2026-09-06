"""Corpus-policy projections used by retrieval planning and requirements."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from itertools import product

from tjipto.corpora.intent_config import contains_intent_phrase, intent_config_for, normalize_intent_text
from tjipto.evidence.store import EvidenceStore
from tjipto.retrieval.answer import validate_answer_candidate
from tjipto.retrieval.bm25 import corpus_query_expansion, lexical_aliases, meaningful_tokens, sparse_index_for_store, tokens


def string_tuple(value: object) -> tuple[str, ...]:
    return tuple(str(item) for item in value or () if isinstance(item, str) and item.strip()) if isinstance(value, (tuple, list)) else ()


def research_focus_query(store: EvidenceStore, research: dict, query: str) -> str:
    signals = research.get("semantic_hints", {})
    excluded: set[str] = set()
    if isinstance(signals, dict):
        for name, values in signals.items():
            if name not in {"comparison", "procedure", "relation"} or not isinstance(values, (tuple, list)):
                continue
            for value in values:
                if isinstance(value, str):
                    excluded.update(normalize_intent_text(value).split())
    summary = store.config.setting("document_summary", {}) or {}
    if isinstance(summary, dict):
        for value in summary.get("document_terms", ()) or ():
            if isinstance(value, str):
                excluded.update(normalize_intent_text(value).split())
    aliases = lexical_aliases(store.config)
    words = [word for word in meaningful_tokens(query, aliases=aliases) if word not in excluded]
    return " ".join(sorted(words)) or query


def research_entity_labels(research: dict) -> tuple[str, ...]:
    aliases = research.get("entity_aliases", {})
    return tuple(str(label) for label in aliases) if isinstance(aliases, dict) else ()


def research_entities(research: dict, query: str) -> tuple[str, ...]:
    aliases = research.get("entity_aliases", {})
    if not isinstance(aliases, dict):
        return ()
    found = []
    for label, values in aliases.items():
        terms = (str(label), *string_tuple(values))
        if contains_intent_phrase(query, terms):
            found.append(str(label))
    return tuple(found)


def research_semantic_terms(
    store: EvidenceStore,
    research: dict,
    query: str,
    entities: tuple[str, ...],
) -> tuple[str, ...]:
    aliases = lexical_aliases(store.config)
    excluded = {
        token
        for values in (research.get("semantic_hints", {}) or {}).values()
        if isinstance(values, (tuple, list))
        for value in values
        if isinstance(value, str)
        for token in meaningful_tokens(value, aliases=aliases)
    }
    excluded.update(token for entity in entities for token in meaningful_tokens(entity, aliases=aliases))
    entity_aliases = research.get("entity_aliases", {})
    if isinstance(entity_aliases, dict):
        excluded.update(
            token
            for entity in entities
            for value in (entity, *string_tuple(entity_aliases.get(entity)))
            for token in meaningful_tokens(value, aliases=aliases)
        )
    return tuple(sorted(meaningful_tokens(query, aliases=aliases) - excluded))


def research_semantic_term_groups(store: EvidenceStore, query: str) -> tuple[tuple[str, ...], ...]:
    """Project corpus-related wording into bounded alternative term groups."""
    aliases = lexical_aliases(store.config)
    related = (store.config.setting("lexical_normalization", {}) or {}).get("related_terms", {})
    requested = _semantic_query_terms(store, corpus_query_expansion(query, store.evidence), aliases)
    requested.difference_update(semantic_support_excluded_terms(store, aliases))
    requested.difference_update(semantic_support_context_terms(store, aliases))
    if not requested or not isinstance(related, dict):
        return ()
    related_tokens = {
        token
        for key, values in related.items()
        if isinstance(key, str)
        for token in (key, *string_tuple(values))
        for token in meaningful_tokens(token, aliases=aliases)
    }
    requested.intersection_update(related_tokens)
    requested = {
        term
        for term in requested
        if not any(
            other != term
            and other in related
            and any(term in meaningful_tokens(value, aliases=aliases) for value in string_tuple(related.get(other)))
            for other in requested
        )
    }
    if not requested:
        return ()
    reverse_choices: dict[str, set[tuple[str, ...]]] = {}
    for key, values in related.items():
        if not isinstance(key, str):
            continue
        key_tokens = tuple(sorted(meaningful_tokens(key, aliases=aliases)))
        for value in string_tuple(values):
            value_tokens = tuple(sorted(meaningful_tokens(value, aliases=aliases)))
            for token in (*key_tokens, *value_tokens):
                reverse_choices.setdefault(token, set()).add(value_tokens or key_tokens)
    alternatives: list[tuple[tuple[str, ...], ...]] = []
    for term in sorted(requested):
        choices: set[tuple[str, ...]] = {(term,)}
        key_values = related.get(term, ())
        for value in string_tuple(key_values):
            tokens_for_value = tuple(sorted(meaningful_tokens(value, aliases=aliases)))
            if tokens_for_value:
                choices.add(tokens_for_value)
        choices.update(reverse_choices.get(term, ()))
        alternatives.append(tuple(sorted(choices)))
    groups: list[tuple[str, ...]] = []
    for choice in product(*alternatives):
        group = tuple(sorted(set(token for part in choice for token in part)))
        if group and group not in groups:
            groups.append(group)
        if len(groups) >= 32:
            break
    groups.extend(
        group for alternatives_for_term in alternatives for group in alternatives_for_term if len(group) > 1 and group not in groups
    )
    if len(requested) <= 2:
        groups.extend(group for alternatives_for_term in alternatives for group in alternatives_for_term if group not in groups)
    return tuple(groups)


def research_relation_terms(
    store: EvidenceStore,
    research: dict,
    query: str,
    entities: tuple[str, ...],
) -> tuple[str, ...]:
    aliases = lexical_aliases(store.config)
    ignored = {
        token
        for values in (research.get("semantic_hints", {}) or {}).values()
        if isinstance(values, (tuple, list))
        for value in values
        if isinstance(value, str)
        for token in tokens(value, aliases=aliases)
    }
    ignored.update(token for entity in entities for token in tokens(entity, aliases=aliases))
    ignored.update(
        token for value in research.get("relation_frame_terms", ()) if isinstance(value, str) for token in tokens(value, aliases=aliases)
    )
    operation_terms = research.get("relation_operation_terms", {})
    if isinstance(operation_terms, dict):
        for phrase, values in operation_terms.items():
            if isinstance(phrase, str) and isinstance(values, (tuple, list)) and contains_intent_phrase(query, (phrase,)):
                return tuple(token for value in values if isinstance(value, str) for token in tokens(value, aliases=aliases) if token)
    return tuple(sorted({token for token in tokens(query, aliases=aliases) if len(token) > 2 and token not in ignored}))


def semantic_support_excluded_terms(store: EvidenceStore, aliases: dict[str, str]) -> set[str]:
    policy = store.config.setting("lexical_normalization", {}) or {}
    return {
        token
        for phrase in policy.get("semantic_support_excluded_terms", ())
        if isinstance(phrase, str)
        for token in tokens(phrase, aliases=aliases)
    }


def semantic_support_context_terms(store: EvidenceStore, aliases: dict[str, str]) -> set[str]:
    policy = store.config.setting("lexical_normalization", {}) or {}
    return {
        token for phrase in policy.get("semantic_context_terms", ()) if isinstance(phrase, str) for token in tokens(phrase, aliases=aliases)
    }


def matches_policy_term(store: EvidenceStore, query: str, term: str) -> bool:
    aliases = lexical_aliases(store.config)
    expected = meaningful_tokens(term, aliases=aliases)
    return bool(expected and expected <= meaningful_tokens(query, aliases=aliases))


def research_issue_term_groups(store: EvidenceStore, query: str) -> tuple[tuple[str, ...], ...]:
    """Choose the most selective corpus-declared semantic expansion for an issue."""
    aliases = lexical_aliases(store.config)
    query_terms = meaningful_tokens(query, aliases=aliases)
    related = (store.config.setting("lexical_normalization", {}) or {}).get("related_terms", {})
    if not isinstance(related, dict):
        return ()
    candidates: list[tuple[int, tuple[tuple[str, ...], ...]]] = []
    for key, values in related.items():
        if not isinstance(key, str) or not meaningful_tokens(key, aliases=aliases) <= query_terms:
            continue
        groups = tuple(group for value in string_tuple(values) if (group := tuple(sorted(meaningful_tokens(value, aliases=aliases)))))
        if not groups:
            continue
        frequency = sum(
            1
            for row in store.evidence
            if row.get("authority_kind") == "normative_legal_text"
            and any(
                set(group)
                <= meaningful_tokens(
                    " ".join(str(row.get(field) or "") for field in ("citation", "hierarchy", "quoted_text")),
                    aliases=aliases,
                )
                for group in groups
            )
        )
        candidates.append((frequency, groups))
    if not candidates:
        return ()
    selected = set(min(candidates, key=lambda item: (item[0], item[1]))[1])
    changed = True
    while changed:
        changed = False
        for values in related.values():
            connected_groups = {
                tuple(sorted(meaningful_tokens(value, aliases=aliases)))
                for value in string_tuple(values)
                if meaningful_tokens(value, aliases=aliases)
            }
            if selected.intersection(connected_groups) and not connected_groups <= selected:
                selected.update(connected_groups)
                changed = True
    return tuple(sorted(selected))


def semantic_supports_text(store: EvidenceStore, query: str, source: str) -> bool:
    """Return whether one source span covers the query's retained legal terms."""
    aliases = lexical_aliases(store.config)
    requested = _semantic_query_terms(store, corpus_query_expansion(query, store.evidence), aliases)
    requested.difference_update(semantic_support_excluded_terms(store, aliases))
    requested.difference_update(semantic_support_context_terms(store, aliases))
    intent = intent_config_for(getattr(store.config, "query_strategy", "generic"), store.config)
    if contains_intent_phrase(query, tuple(intent.get("instrument_analysis_signals", ()) or ())) and not research_issue_term_groups(
        store, query
    ):
        return False
    if re.search(r"\bboleh(?:kah)?\b", normalize_intent_text(query)):
        requested.difference_update({"boleh", "bolehkah"})
    supported = meaningful_tokens(source, aliases=aliases)
    normalized_query = normalize_intent_text(query)
    numeric_words = {
        key: value for key, value in aliases.items() if str(value).isdigit() and re.search(rf"\b{re.escape(key)}\b", normalized_query)
    }
    numeric_alias_values = tuple(
        str(value) for key, value in aliases.items() if key in numeric_words or re.search(rf"\b{re.escape(key)}\b", normalized_query)
    )
    if numeric_words and any(numeric_alias_values.count(value) > 1 for value in set(numeric_alias_values)):
        return False
    support_score, missing = _semantic_support_score(store, requested, supported, normalized_query, aliases)
    if not requested:
        return False
    if support_score >= len(requested):
        return True
    # Request wording often contains generic legal framing terms.  They are
    # not allowed to disqualify a source when the corpus itself shows that
    # they occur broadly; issue-bearing terms still need direct or corpus-
    # derived alternative support.
    corpus_frequency = _corpus_term_frequency(store, aliases)
    framing_missing = {term for term in missing if corpus_frequency.get(term, 0) > max(10, len(store.evidence) // 20)}
    effective_missing = missing - framing_missing
    effective_required = len(requested) - len(framing_missing)
    if effective_required > 1 and support_score < 2:
        return False
    if support_score >= 2:
        return True
    return bool(support_score) and not effective_missing.intersection(corpus_frequency)


def semantic_supports_all_terms(store: EvidenceStore, query: str, source: str) -> bool:
    """Require every retained source-occurrence term, including corpus aliases."""
    aliases = lexical_aliases(store.config)
    requested = _semantic_query_terms(store, query, aliases)
    requested.difference_update(semantic_support_excluded_terms(store, aliases))
    requested.difference_update(semantic_support_context_terms(store, aliases))
    score, _ = _semantic_support_score(
        store,
        requested,
        meaningful_tokens(source, aliases=aliases),
        normalize_intent_text(query),
        aliases,
    )
    return bool(requested) and score >= len(requested)


def semantic_support_score(store: EvidenceStore, query: str, source: str) -> int:
    """Score corpus-backed query terms for deterministic candidate ordering."""
    aliases = lexical_aliases(store.config)
    requested = _semantic_query_terms(store, corpus_query_expansion(query, store.evidence), aliases)
    requested.difference_update(semantic_support_excluded_terms(store, aliases))
    requested.difference_update(semantic_support_context_terms(store, aliases))
    return _semantic_support_score(
        store,
        requested,
        meaningful_tokens(source, aliases=aliases),
        normalize_intent_text(query),
        aliases,
    )[0]


def _semantic_support_score(
    store: EvidenceStore,
    requested: set[str],
    supported: set[str],
    normalized_query: str,
    aliases: dict[str, str],
) -> tuple[int, set[str]]:
    related = (store.config.setting("lexical_normalization", {}) or {}).get("related_terms", {})
    alternatives: dict[str, set[str]] = {}
    for term in requested:
        alternatives[term] = {
            token for value in related.get(term, ()) if isinstance(value, str) for token in meaningful_tokens(value, aliases=aliases)
        }
    matched: set[str] = set()
    for term in requested:
        if term in supported or alternatives.get(term, set()) & supported:
            matched.add(term)
            continue
        if len(term) < 6:
            continue
        if any(len(candidate) >= 6 and SequenceMatcher(None, term, candidate).ratio() >= 0.88 for candidate in supported):
            matched.add(term)
    return len(matched), requested - matched


def _corpus_term_frequency(store: EvidenceStore, aliases: dict[str, str]) -> dict[str, int]:
    frequency: dict[str, int] = {}
    for row in store.evidence:
        row_terms = {
            token
            for value in (row.get("citation"), row.get("quoted_text"), " ".join(row.get("hierarchy") or ()))
            for token in meaningful_tokens(str(value or ""), aliases=aliases)
        }
        for token in row_terms:
            frequency[token] = frequency.get(token, 0) + 1
    return frequency


def _semantic_query_terms(store: EvidenceStore, query: str, aliases: dict[str, str]) -> set[str]:
    """Drop configured routing markers before evaluating lexical support."""
    requested = meaningful_tokens(query, aliases=aliases)
    marker_phrases: list[str] = []
    intent = intent_config_for(getattr(store.config, "query_strategy", "generic"), store.config)
    for key in (
        "document_open_terms",
        "all_source_scope_terms",
        "source_occurrence_separators",
        "source_occurrence_query_wrappers",
        "structure_list_terms",
        "structure_count_terms",
    ):
        values = intent.get(key, ()) or ()
        if not values:
            values = store.config.setting(key, ()) or ()
        marker_phrases.extend(value for value in values if isinstance(value, str))
    for phrase in marker_phrases:
        requested.difference_update(meaningful_tokens(phrase, aliases=aliases))
    return requested


def single_support_covers_query(store: EvidenceStore, query: str) -> bool:
    return any(
        row.get("authority_kind") not in {"structural_context", "structural_support"}
        and validate_answer_candidate(store, row | {"route_sources": ("bm25",)})[0]
        and _covers_all_query_terms(store, query, row)
        for row in sparse_index_for_store(store).search(query, limit=10)
    )


def _covers_all_query_terms(store: EvidenceStore, query: str, row: dict) -> bool:
    aliases = lexical_aliases(store.config)
    requested = _semantic_query_terms(store, query, aliases)
    if not requested:
        return False
    source = " ".join(str(row.get(key) or "") for key in ("citation", "hierarchy", "quoted_text"))
    return semantic_support_score(store, query, source) >= len(requested)
