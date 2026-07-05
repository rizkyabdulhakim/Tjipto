from __future__ import annotations

from tjipto.corpora.intent_config import intent_config_for
from tjipto.corpora.parser_dispatch import (
    DEFAULT_CORPUS_ID,
    parse_bab_reference,
    parse_pasal_reference,
)


def relation_lookup(store, query: str, limit: int = 10) -> tuple[dict, ...]:
    config = getattr(store, "config", None)
    strategy = getattr(config, "query_strategy", "generic")
    if not has_relation_target(query, strategy=strategy, config=config):
        return ()
    relation = _relation(query, strategy=strategy, config=config)
    if relation is None:
        return ()
    intent = intent_config_for(strategy, config)
    route: dict = intent["relation_routes"].get(relation, {})
    descendants_by_parent, evidence_by_unit = _relation_indexes(store)
    if route.get("mode") == "parent":
        source = _unit(store, query, route["source_unit_type"])
        target = _hierarchy_parent(store, source, route["target_unit_type"]) if source else None
        if source is None or target is None:
            return ()
        row = _evidence_for_unit(store, source, evidence_by_unit, descendants_by_parent)
        if row is None:
            return ()
        return (
            row
            | {
                "legal_relation": {
                    "relation_type": relation,
                    "source_legal_unit_id": source["legal_unit_id"],
                    "source_label": source.get("unit_label"),
                    "target_legal_unit_id": target.get("legal_unit_id"),
                    "target_label": target.get("unit_label"),
                },
            },
        )
    parent = _parent_unit(store, query, route)
    if parent is None:
        return ()
    child_type = route.get("child_unit_type")
    children = [
        row
        for row in store.legal_units
        if row.get("unit_type") == child_type and parent["legal_unit_id"] in (row.get("parent_legal_unit_ids") or ())
    ]
    rows = []
    for child in children:
        row = _evidence_for_unit(store, child, evidence_by_unit, descendants_by_parent)
        if row is None:
            continue
        rows.append(
            row
            | {
                "legal_relation": {
                    "relation_type": relation,
                    "source_legal_unit_id": parent["legal_unit_id"],
                    "source_label": parent.get("unit_label"),
                    "target_legal_unit_id": child.get("legal_unit_id"),
                    "target_label": child.get("unit_label"),
                },
            }
        )
    return tuple(rows[:limit])


def has_relation_target(query: str, *, strategy: str = "generic", config=None) -> bool:
    folded = (query or "").casefold()
    intent = intent_config_for(strategy, config)
    if not any(intent[key] for key in ("relation_words", "direct_relation_words", "pasal_parent_words", "relation_child_words")):
        return False
    if _unsupported_relation_requested(query, folded, strategy=strategy, config=config):
        return True
    if _pasal_parent_requested(query, folded, strategy=strategy, config=config):
        return True
    corpus_id = _corpus_id(config)
    if _has_pasal(query, corpus_id) and _route_terms_match(intent, "pasal_children", folded):
        return _child_relation_requested(folded, intent)
    if _has_bab(query, corpus_id) and _route_terms_match(intent, "bab_children", folded):
        return _child_relation_requested(folded, intent)
    return False


def _relation(query: str, *, strategy: str, config=None) -> str | None:
    folded = (query or "").casefold()
    intent = intent_config_for(strategy, config)
    if _pasal_parent_requested(query, folded, strategy=strategy, config=config):
        return _route_name(intent, "parent")
    corpus_id = _corpus_id(config)
    if _has_pasal(query, corpus_id) and _route_terms_match(intent, "pasal_children", folded):
        return _route_name(intent, "pasal_children")
    if _has_bab(query, corpus_id) and _route_terms_match(intent, "bab_children", folded):
        return _route_name(intent, "bab_children")
    if _unsupported_relation_requested(query, folded, strategy=strategy, config=config):
        return _route_name(intent, "unsupported")
    return None


def _route_name(intent: dict, requested: str) -> str | None:
    for name, route in intent["relation_routes"].items():
        if requested == route.get("mode"):
            return name
    return None


def _route_terms_match(intent: dict, mode: str, folded: str) -> bool:
    route: dict = next((row for row in intent["relation_routes"].values() if row.get("mode") == mode), {})
    terms = route.get("required_terms") or ()
    return any(term in folded for term in terms)


def _unsupported_relation_requested(query: str, folded: str, *, strategy: str, config=None) -> bool:
    intent = intent_config_for(strategy, config)
    corpus_id = _corpus_id(config)
    has_pasal = _has_pasal(query, corpus_id)
    has_bab = _has_bab(query, corpus_id)
    direct_relation = any(pattern in folded for pattern in intent["direct_relation_words"])
    relation_words = any(pattern in folded for pattern in intent["relation_words"])
    unsupported_context = any(pattern in folded for pattern in intent["unsupported_relation_context_words"])
    return (direct_relation and (has_pasal or has_bab or relation_words)) or (
        relation_words and (has_pasal or has_bab or unsupported_context)
    )


def _child_relation_requested(folded: str, intent: dict) -> bool:
    return any(word in folded for word in intent["relation_child_words"])


def _pasal_parent_requested(query: str, folded: str, *, strategy: str, config=None) -> bool:
    return _has_pasal(query, _corpus_id(config)) and any(
        pattern in folded for pattern in intent_config_for(strategy, config)["pasal_parent_words"]
    )


def _parent_unit(store, query: str, route: dict) -> dict | None:
    if route.get("mode") == "unsupported":
        return None
    parent_type = route.get("parent_unit_type")
    return _unit(store, query, parent_type) if parent_type else None


def _unit(store, query: str, unit_type: str) -> dict | None:
    corpus_id = _corpus_id(getattr(store, "config", None))
    if unit_type == "pasal_record":
        label = parse_pasal_reference(corpus_id, query)
    else:
        label = parse_bab_reference(corpus_id, query)
    if label is None:
        return None
    preferred = getattr(store.config, "preferred_source_role", None)
    matches = [row for row in store.legal_units if row.get("unit_label") == label and row.get("unit_type") == unit_type]
    return next((row for row in matches if _source_role(row) == preferred), None) or (matches[0] if matches else None)


def _hierarchy_parent(store, unit: dict, unit_type: str) -> dict | None:
    parent_ids = set(unit.get("parent_legal_unit_ids") or ())
    if not parent_ids:
        return None
    preferred = getattr(store.config, "preferred_source_role", None)
    matches = [row for row in store.legal_units if row.get("legal_unit_id") in parent_ids and row.get("unit_type") == unit_type]
    return next((row for row in matches if _source_role(row) == preferred), None) or (matches[0] if matches else None)


def _relation_indexes(store) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    descendants_by_parent: dict[str, list[dict]] = {}
    for row in store.legal_units:
        for parent_id in row.get("parent_legal_unit_ids") or ():
            descendants_by_parent.setdefault(parent_id, []).append(row)
    evidence_by_unit: dict[str, list[dict]] = {}
    for row in store.evidence:
        evidence_by_unit.setdefault(row.get("legal_unit_id"), []).append(row)
    return descendants_by_parent, evidence_by_unit


def _source_role(row: dict) -> str | None:
    return row.get("source_role")


def _has_pasal(query: str, corpus_id: str) -> bool:
    return parse_pasal_reference(corpus_id, query) is not None


def _has_bab(query: str, corpus_id: str) -> bool:
    return parse_bab_reference(corpus_id, query) is not None


def _corpus_id(config) -> str:
    return getattr(config, "corpus_id", DEFAULT_CORPUS_ID)


def _evidence_for_unit(
    store, unit: dict, evidence_by_unit: dict[str, list[dict]], descendants_by_parent: dict[str, list[dict]]
) -> dict | None:
    for candidate in evidence_by_unit.get(unit["legal_unit_id"], ()):
        if candidate.get("status") == "final" and store.bboxes_for(candidate["evidence_id"]):
            return candidate
    for child in descendants_by_parent.get(unit["legal_unit_id"], ()):
        child_candidate = _evidence_for_unit(store, child, evidence_by_unit, descendants_by_parent)
        if child_candidate is not None:
            return child_candidate
    return None
