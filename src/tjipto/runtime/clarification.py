"""Corpus-backed ambiguous-query decisions, kept outside retrieval and publication."""

from __future__ import annotations

from dataclasses import dataclass

from tjipto.corpora.intent_config import contains_intent_phrase, intent_config_for, normalize_intent_text


@dataclass(frozen=True)
class ClarificationOption:
    label: str
    resolution: dict[str, str]


@dataclass(frozen=True)
class ClarificationDecision:
    kind: str
    options: tuple[ClarificationOption, ...]


def clarification_decision(store, semantics, routed: dict, *, entity_query: bool) -> ClarificationDecision | None:
    """Offer choices only when distinct, already-published interpretations exist."""
    if semantics.legal_references and len(semantics.legal_references) > 1:
        options = tuple(
            ClarificationOption(reference, {"query": reference})
            for reference in semantics.legal_references
            if _has_final_legal_target(store, reference)
        )
        if len(options) > 1:
            return ClarificationDecision("legal_target", options)
    metadata = _metadata_options(store, routed, entity_query=entity_query)
    if metadata:
        return ClarificationDecision("source_scope", metadata)
    operations = _operation_options(store, semantics, routed)
    if len(operations) > 1:
        return ClarificationDecision("relation_operation", operations)
    lexical = _lexical_options(store, semantics, routed)
    if len(lexical) > 1:
        return ClarificationDecision("lexical_target", lexical)
    return None


def _has_final_legal_target(store, reference: str) -> bool:
    unit_ids = {
        str(unit.get("legal_unit_id"))
        for unit in store.legal_units
        if str(unit.get("unit_label") or "").casefold() == reference.casefold()
    }
    return bool(unit_ids and any(row.get("legal_unit_id") in unit_ids and row.get("status") == "final" for row in store.evidence))


def _metadata_options(store, routed: dict, *, entity_query: bool) -> tuple[ClarificationOption, ...]:
    if routed.get("route") not in {"metadata", "metadata_scope_unresolved"} or routed.get("metadata_filters", {}).get("source_role"):
        return ()
    roles = tuple(routed.get("metadata_source_roles") or ())
    if not roles:
        roles = tuple(sorted({row.get("source_role") for row in routed.get("matches", ()) if row.get("source_role")}))
    if len(roles) < 2 or (entity_query and all(row.get("metadata_field") == "signatories" for row in routed.get("matches", ()))):
        return ()
    intent = intent_config_for(getattr(store.config, "query_strategy", "generic"), store.config)
    labels = intent.get("source_role_labels", {})
    titles = (store.config.setting("document_catalog", {}) or {}).get("titles", {})
    return tuple(
        ClarificationOption(str(titles.get(role) or labels.get(role) or role), {"source_role": str(role)})
        for role in roles
    )


def _operation_options(store, semantics, routed: dict) -> tuple[ClarificationOption, ...]:
    if routed.get("route") != "document_relation" or not semantics.legal_references:
        return ()
    config = intent_config_for(getattr(store.config, "query_strategy", "generic"), store.config)
    families = config.get("document_relation", {}).get("relation_families", {})
    options = []
    for name, family in families.items():
        if not contains_intent_phrase(routed["original_query"], tuple(family.get("terms") or ())):
            continue
        types = set(family.get("relation_types") or ())
        if any(edge.get("edge_type") in types and edge.get("relation_id") for edge in store.graph_edges):
            label = next((str(term) for term in family.get("terms") or () if term), str(name).replace("_", " ").title())
            options.append(ClarificationOption(label, {"relation_family": str(name)}))
    return tuple(options)


def _lexical_options(store, semantics, routed: dict) -> tuple[ClarificationOption, ...]:
    if (
        routed.get("route") != "bm25"
        or semantics.legal_references
        or semantics.source_role
        or "atau" not in normalize_intent_text(routed.get("original_query") or "").split()
    ):
        return ()
    units = {str(unit.get("legal_unit_id")): unit for unit in store.legal_units}
    options: list[ClarificationOption] = []
    seen: set[str] = set()
    for row in routed.get("matches", ()):
        unit = units.get(str(row.get("legal_unit_id")))
        label = str(unit.get("unit_label") or "") if unit else ""
        if not label.startswith(("Pasal ", "BAB ")) or label in seen or not _has_final_legal_target(store, label):
            continue
        seen.add(label)
        options.append(ClarificationOption(label, {"query": label}))
    return tuple(options)
