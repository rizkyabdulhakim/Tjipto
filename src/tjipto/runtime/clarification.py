"""Corpus-backed ambiguous-query decisions, kept outside retrieval and publication."""

from __future__ import annotations

from dataclasses import dataclass
import json

from tjipto.corpora.intent_config import contains_intent_phrase, intent_config_for, normalize_intent_text
from tjipto.retrieval.bm25 import meaningful_tokens, tokens


@dataclass(frozen=True)
class ClarificationOption:
    label: str
    resolution: dict[str, str]


@dataclass(frozen=True)
class ClarificationDecision:
    kind: str
    question: str
    options: tuple[ClarificationOption, ...]


def clarification_decision(store, semantics, routed: dict) -> ClarificationDecision | None:
    """Offer choices only when distinct, already-published interpretations exist."""
    config = intent_config_for(getattr(store.config, "query_strategy", "generic"), store.config)
    policy = config.get("clarification", {})
    if semantics.legal_references and len(semantics.legal_references) > 1 and routed.get("route") != "document_relation":
        options = tuple(
            ClarificationOption(reference, {"legal_target": reference})
            for reference in semantics.legal_references
            if _has_final_legal_target(store, reference)
        )
        if len(options) > 1:
            return _decision(policy, "legal_target", options)
    entities = _entity_options(routed)
    if len(entities) > 1:
        return _decision(policy, "entity", entities)
    metadata = _metadata_options(store, routed)
    if metadata:
        return _decision(policy, "source_scope", metadata)
    operations = _operation_options(store, semantics, routed, config)
    if len(operations) > 1:
        return _decision(policy, "relation_operation", operations)
    lexical = _lexical_options(store, semantics, routed, policy, config)
    if len(lexical) > 1:
        kind = "legal_target" if contains_intent_phrase(routed.get("original_query") or "", config.get("relation_words", ())) else "concept_facet"
        return _decision(policy, kind, lexical)
    return None


def _decision(policy: dict, kind: str, options: tuple[ClarificationOption, ...]) -> ClarificationDecision:
    questions = policy.get("questions", {})
    question = str(questions.get(kind) or questions.get("default") or "Clarification required.")
    return ClarificationDecision(kind, question, options)


def _has_final_legal_target(store, reference: str) -> bool:
    unit_ids = {
        str(unit.get("legal_unit_id"))
        for unit in store.legal_units
        if str(unit.get("unit_label") or "").casefold() == reference.casefold()
    }
    return bool(unit_ids and any(row.get("legal_unit_id") in unit_ids and row.get("status") == "final" for row in store.evidence))


def _metadata_options(store, routed: dict) -> tuple[ClarificationOption, ...]:
    if routed.get("route") not in {"metadata", "metadata_scope_unresolved"} or routed.get("metadata_filters", {}).get("source_role"):
        return ()
    roles = tuple(routed.get("metadata_source_roles") or ())
    if not roles:
        roles = tuple(sorted({row.get("source_role") for row in routed.get("matches", ()) if row.get("source_role")}))
    identities = {row.get("entity_identity") for row in routed.get("matches", ()) if row.get("entity_identity")}
    if len(identities) == 1:
        return ()
    if len(roles) < 2:
        return ()
    intent = intent_config_for(getattr(store.config, "query_strategy", "generic"), store.config)
    labels = intent.get("source_role_labels", {})
    titles = (store.config.setting("document_catalog", {}) or {}).get("titles", {})
    return tuple(
        ClarificationOption(str(titles.get(role) or labels.get(role) or role), {"source_role": str(role)})
        for role in roles
    )


def _entity_options(routed: dict) -> tuple[ClarificationOption, ...]:
    if len(tuple(routed.get("metadata_source_roles") or ())) == 1:
        return ()
    rows = {}
    for row in routed.get("matches", ()):
        identity = row.get("entity_identity")
        label = row.get("printed_name") or row.get("metadata_answer")
        if identity and label:
            rows[str(identity)] = str(label)
    return tuple(ClarificationOption(label, {"entity": identity}) for identity, label in sorted(rows.items()))


def _operation_options(store, semantics, routed: dict, config: dict) -> tuple[ClarificationOption, ...]:
    if routed.get("route") not in {"document_relation", "relation_not_found"} or not semantics.legal_references:
        return ()
    families = config.get("document_relation", {}).get("relation_families", {})
    explicit_families = {
        name for name, family in families.items()
        if contains_intent_phrase(routed["original_query"], tuple(family.get("terms") or ()))
    }
    # The relation router has already classified an explicit operation (for
    # example a rename). Use that typed classification instead of offering
    # every family that happens to match the target citation.
    relation_target = routed.get("relation_target") or {}
    selected_types = set(relation_target.get("relation_types") or ())
    if selected_types and not explicit_families:
        explicit_families = {
            name for name, family in families.items()
            if selected_types.intersection(set(family.get("relation_types") or ()))
        }
    options = []
    for name, family in families.items():
        types = set(family.get("relation_types") or ())
        supported = any(
            edge.get("edge_type") in types
            and edge.get("relation_id")
            and str((edge.get("relation_projection") or {}).get("target_citation") or "").casefold()
            in {reference.casefold() for reference in semantics.legal_references}
            for edge in store.graph_edges
        )
        if name in explicit_families or (not explicit_families and supported):
            label = next((str(term) for term in family.get("terms") or () if term), str(name).replace("_", " ").title())
            options.append(ClarificationOption(label, {"relation_family": str(name)}))
    return tuple(options)


def _lexical_options(store, semantics, routed: dict, policy: dict, config: dict) -> tuple[ClarificationOption, ...]:
    if routed.get("route") != "bm25" or semantics.legal_references or semantics.source_role:
        return ()
    query = str(routed.get("original_query") or "")
    if contains_intent_phrase(query, tuple(policy.get("direct_retrieval_terms") or ())):
        return ()
    clauses = _split_ambiguity(query, tuple(policy.get("choice_terms") or ()))
    if len(clauses) > 1:
        clause_options: list[ClarificationOption] = []
        for clause in clauses:
            subset = tuple(row for row in routed.get("matches", ()) if _matches_clause(store, row, _concept_query(policy, clause)))
            if not subset:
                return ()
            clause_options.append(ClarificationOption(clause, {"concept_facet": _subset_key(subset)}))
        return tuple(clause_options)
    if _uses_configured_normalization(query, store.config):
        return ()
    units = {str(unit.get("legal_unit_id")): unit for unit in store.legal_units}
    target_types = set(policy.get("legal_target_unit_types") or ())
    matches = routed.get("matches", ())
    if _top_quote_contains_query(query, matches, store.config):
        return ()
    options: list[ClarificationOption] = []
    seen: set[str] = set()
    for row in matches:
        if not row.get("lexical_complete_coverage"):
            continue
        unit = _target_unit(units, str(row.get("legal_unit_id") or ""), target_types)
        label = str(unit.get("unit_label") or "") if unit else ""
        if not label or label in seen or not _has_final_legal_target(store, label):
            continue
        seen.add(label)
        options.append(ClarificationOption(label, {"legal_target": label}))
        if len(options) == int(policy.get("maximum_options") or 3):
            break
    return tuple(options)


def _uses_configured_normalization(query: str, config) -> bool:
    aliases = {
        normalize_intent_text(key): normalize_intent_text(value)
        for key, value in (config.setting("lexical_normalization", {}) or {}).get("aliases", {}).items()
    }
    return any(aliases.get(token) and aliases[token] != token for token in normalize_intent_text(query).split())


def _top_quote_contains_query(query: str, matches: tuple[dict, ...], config) -> bool:
    aliases = {
        str(key).casefold(): str(value).casefold()
        for key, value in (config.setting("lexical_normalization", {}) or {}).get("aliases", {}).items()
    }
    query_tokens = tokens(query, aliases=aliases)
    meaningful = meaningful_tokens(query, aliases=aliases)
    requested = tuple(token for token in query_tokens if token in meaningful)
    requested_pairs = tuple(zip(requested, requested[1:]))
    if not requested_pairs or not matches:
        return False
    for line in str(matches[0].get("quoted_text") or "").splitlines():
        quoted = tokens(line, aliases=aliases)
        cursor = 0
        for term in requested:
            try:
                cursor = quoted.index(term, cursor) + 1
            except ValueError:
                break
        else:
            return True
    return False


def _split_ambiguity(query: str, terms: tuple[str, ...]) -> tuple[str, ...]:
    normalized = normalize_intent_text(query)
    for term in terms:
        token = normalize_intent_text(term)
        parts = tuple(part.strip() for part in normalized.split(f" {token} ") if part.strip())
        if len(parts) > 1:
            return parts
    return ()


def _concept_query(policy: dict, clause: str) -> str:
    aliases = {normalize_intent_text(key): str(value) for key, value in (policy.get("concept_aliases") or {}).items()}
    return aliases.get(normalize_intent_text(clause), clause)


def _matches_clause(store, row: dict, clause: str) -> bool:
    aliases = {
        str(key).casefold(): str(value).casefold()
        for key, value in (store.config.setting("lexical_normalization", {}) or {}).get("aliases", {}).items()
    }
    terms = meaningful_tokens(clause, aliases=aliases)
    return bool(terms and terms <= set(row.get("lexical_supported_terms") or ()))


def _subset_key(rows: tuple[dict, ...]) -> str:
    keys = sorted(
        {
            str(row.get("evidence_id") or row.get("legal_unit_id"))
            for row in rows
            if row.get("evidence_id") or row.get("legal_unit_id")
        }
    )
    return json.dumps(keys, ensure_ascii=True, separators=(",", ":"))


def _target_unit(units: dict[str, dict], unit_id: str, target_types: set[str]) -> dict | None:
    unit = units.get(unit_id)
    while unit is not None and target_types and unit.get("unit_type") not in target_types:
        unit = units.get(str(unit.get("parent_legal_unit_id") or ""))
    return unit
