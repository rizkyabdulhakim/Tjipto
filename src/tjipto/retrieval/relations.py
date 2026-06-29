from __future__ import annotations

import re

from tjipto.corpora.intent_config import intent_config_for


BAB_RE = re.compile(r"\bbab\s+([ivxlcdm]+)\s*([a-z]?)\b", re.IGNORECASE)
PASAL_RE = re.compile(r"\bpasal\s+([0-9]+[a-z]?)\b", re.IGNORECASE)


def relation_lookup(store, query: str, limit: int = 10) -> tuple[dict, ...]:
    config = getattr(store, "config", None)
    strategy = getattr(config, "query_strategy", "generic")
    if not has_relation_target(query, strategy=strategy, config=config):
        return ()
    relation = _relation(query, strategy=strategy, config=config)
    if relation is None:
        return ()
    descendants_by_parent, evidence_by_unit = _relation_indexes(store)
    if relation == "pasal_parent_bab":
        source = _unit(store, query, "pasal_record")
        target = _hierarchy_parent(store, source, "bab_record") if source else None
        if source is None or target is None:
            return ()
        row = _evidence_for_unit(store, source, evidence_by_unit, descendants_by_parent)
        if row is None:
            return ()
        return (row | {
            "legal_relation": {
                "relation_type": relation,
                "source_legal_unit_id": source["legal_unit_id"],
                "source_label": source.get("unit_label"),
                "target_legal_unit_id": target.get("legal_unit_id"),
                "target_label": target.get("unit_label"),
            },
        },)
    parent = _parent_unit(store, query, relation)
    if parent is None:
        return ()
    child_type = "ayat_record" if relation == "pasal_ayat_children" else "pasal_record"
    children = [
        row
        for row in store.legal_units
        if row.get("unit_type") == child_type
        and parent["legal_unit_id"] in (row.get("parent_legal_unit_ids") or ())
    ]
    rows = []
    for child in children:
        row = _evidence_for_unit(store, child, evidence_by_unit, descendants_by_parent)
        if row is None:
            continue
        rows.append(row | {
            "legal_relation": {
                "relation_type": relation,
                "source_legal_unit_id": parent["legal_unit_id"],
                "source_label": parent.get("unit_label"),
                "target_legal_unit_id": child.get("legal_unit_id"),
                "target_label": child.get("unit_label"),
            },
        })
    return tuple(rows[:limit])


def has_relation_target(query: str, *, strategy: str = "generic", config=None) -> bool:
    folded = (query or "").casefold()
    intent = intent_config_for(strategy, config)
    if not intent["relation_words"] and not intent["direct_relation_words"] and not intent["pasal_parent_words"]:
        return False
    if _unsupported_relation_requested(query, folded, strategy=strategy, config=config):
        return True
    if _pasal_parent_requested(query, folded, strategy=strategy, config=config):
        return True
    if PASAL_RE.search(query or "") and "ayat" in folded:
        return any(word in folded for word in ("apa saja", "daftar", "dalam", "anak", "child"))
    if BAB_RE.search(query or "") and "pasal" in folded:
        return any(word in folded for word in ("apa saja", "daftar", "dalam", "anak", "child"))
    return False


def _relation(query: str, *, strategy: str, config=None) -> str | None:
    folded = (query or "").casefold()
    if _pasal_parent_requested(query, folded, strategy=strategy, config=config):
        return "pasal_parent_bab"
    if PASAL_RE.search(query or "") and "ayat" in folded:
        return "pasal_ayat_children"
    if BAB_RE.search(query or "") and "pasal" in folded:
        return "bab_pasal_children"
    if _unsupported_relation_requested(query, folded, strategy=strategy, config=config):
        return "unsupported_amendment_relation"
    return None


def _unsupported_relation_requested(query: str, folded: str, *, strategy: str, config=None) -> bool:
    intent = intent_config_for(strategy, config)
    has_pasal = PASAL_RE.search(query or "") is not None
    has_bab = BAB_RE.search(query or "") is not None
    direct_relation = any(pattern in folded for pattern in intent["direct_relation_words"])
    relation_words = any(pattern in folded for pattern in intent["relation_words"])
    return (
        direct_relation and (has_pasal or has_bab or relation_words)
    ) or (
        relation_words and (has_pasal or has_bab or "perubahan" in folded)
    )


def _pasal_parent_requested(query: str, folded: str, *, strategy: str, config=None) -> bool:
    return PASAL_RE.search(query or "") is not None and any(
        pattern in folded
        for pattern in intent_config_for(strategy, config)["pasal_parent_words"]
    )


def _parent_unit(store, query: str, relation: str) -> dict | None:
    if relation == "unsupported_amendment_relation":
        return None
    if relation == "pasal_ayat_children":
        return _unit(store, query, "pasal_record")
    else:
        return _unit(store, query, "bab_record")


def _unit(store, query: str, unit_type: str) -> dict | None:
    if unit_type == "pasal_record":
        match = PASAL_RE.search(query or "")
        label = f"Pasal {match.group(1).upper()}" if match else None
    else:
        match = BAB_RE.search(query or "")
        label = f"BAB {match.group(1).upper()}{match.group(2).upper()}" if match else None
    if label is None:
        return None
    preferred = getattr(store.config, "preferred_source_role", None)
    matches = [
        row for row in store.legal_units
        if row.get("unit_label") == label and row.get("unit_type") == unit_type
    ]
    return next((row for row in matches if _source_role(row) == preferred), None) or (matches[0] if matches else None)


def _hierarchy_parent(store, unit: dict, unit_type: str) -> dict | None:
    parent_ids = set(unit.get("parent_legal_unit_ids") or ())
    if not parent_ids:
        return None
    preferred = getattr(store.config, "preferred_source_role", None)
    matches = [
        row for row in store.legal_units
        if row.get("legal_unit_id") in parent_ids and row.get("unit_type") == unit_type
    ]
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
    source = str(row.get("source_document_id") or "")
    return source.split("::", 1)[1] if "::" in source else None


def _evidence_for_unit(store, unit: dict, evidence_by_unit: dict[str, list[dict]], descendants_by_parent: dict[str, list[dict]]) -> dict | None:
    for candidate in evidence_by_unit.get(unit["legal_unit_id"], ()):
        if candidate.get("status") == "final" and store.bboxes_for(candidate["evidence_id"]):
            return candidate
    for child in descendants_by_parent.get(unit["legal_unit_id"], ()):
        candidate = _evidence_for_unit(store, child, evidence_by_unit, descendants_by_parent)
        if candidate is not None:
            return candidate
    return None
