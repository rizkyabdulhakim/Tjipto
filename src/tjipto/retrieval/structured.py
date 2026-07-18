from __future__ import annotations

from tjipto.corpora.intent_config import intent_config_for, resolve_instrument_intent
from tjipto.corpora.parser_dispatch import (
    DEFAULT_CORPUS_ID,
    label_keys,
    parse_ayat_reference,
    parse_bab_reference,
    parse_pasal_reference,
    resolve_navigation,
)
from tjipto.evidence.store import EvidenceStore
from tjipto.retrieval.metadata import source_role_for_query


def structured_lookup(store: EvidenceStore, query: str, limit: int = 10, *, strategy: str = "uud_1945") -> tuple[dict, ...]:
    config = getattr(store, "config", None)
    intent = intent_config_for(strategy, config)
    if not intent["structured_lookup_enabled"]:
        return ()
    instrument = _instrument_rows(store, query, limit, strategy=strategy, config=config)
    if instrument:
        return instrument
    navigation = _navigation_rows(store, query, limit, _corpus_id(config))
    if navigation:
        return navigation
    targets = _targets(query, intent, _corpus_id(config))
    if not targets:
        return ()
    legal_unit_ids = {
        row["legal_unit_id"]
        for row in (*getattr(store, "legal_units", ()), *getattr(store, "chunks", ()))
        if row.get("legal_unit_id") and _matches_unit(row, targets)
    }
    requested_role = source_role_for_query(query, strategy=strategy, config=config)
    bab = parse_bab_reference(_corpus_id(config), query)
    if bab:
        dedicated_unit_ids = {
            unit.get("legal_unit_id")
            for unit in store.legal_units
            if unit.get("unit_type") == "bab_record" and unit.get("unit_label", "").casefold() == bab.casefold()
        }
        dedicated = [
            row
            for row in store.evidence
            if row.get("status") == "final"
            and store.bboxes_for(row["evidence_id"])
            and (requested_role is None or row.get("source_role") == requested_role)
            and row.get("citation", "").casefold() == bab.casefold()
            and row.get("legal_unit_id") in dedicated_unit_ids
        ]
        if not dedicated:
            dedicated = [
                row
                for row in store.evidence
                if row.get("status") == "final"
                and store.bboxes_for(row["evidence_id"])
                and row.get("citation", "").casefold() == bab.casefold()
                and row.get("legal_unit_id") in dedicated_unit_ids
            ]
        if dedicated:
            return tuple(dedicated[:limit])
    rows = [
        row
        for row in store.evidence
        if row.get("status") == "final"
        and store.bboxes_for(row["evidence_id"])
        and (requested_role is None or row.get("source_role") == requested_role)
        and (row.get("legal_unit_id") in legal_unit_ids or _matches(row, targets))
    ]
    return tuple(rows[:limit])


def has_structured_target(query: str, *, strategy: str = "uud_1945", config=None) -> bool:
    intent = intent_config_for(strategy, config)
    if not intent["structured_lookup_enabled"]:
        return False
    if _instrument_target(query, strategy=strategy, config=config):
        return True
    return bool(_targets(query, intent, _corpus_id(config)))


def _instrument_target(query: str, *, strategy: str, config=None) -> bool:
    return bool(_instrument_rows(None, query, 1, strategy=strategy, config=config, probe_only=True))


def _instrument_rows(
    store: EvidenceStore | None,
    query: str,
    limit: int,
    *,
    strategy: str,
    config=None,
    probe_only: bool = False,
) -> tuple[dict, ...]:
    folded = (query or "").casefold()
    intent = intent_config_for(strategy, config)
    corpus_id = _corpus_id(config)
    bab = parse_bab_reference(corpus_id, query)
    if (
        bab
        and any(pattern in folded for pattern in intent["instrument_deletion_words"])
        and any(pattern in folded for pattern in intent["instrument_change_context_words"])
    ):
        if probe_only:
            return ({"probe": True},)
        matches: list[dict] = []
        prefix = intent["instrument_citation_templates"].get("prefix", "")
        clause_marker = intent["instrument_citation_templates"].get("clause_marker", "")
        for row in getattr(store, "evidence", ()):
            citation = str(row.get("citation") or "")
            if not (prefix and clause_marker and citation.startswith(prefix) and clause_marker in citation):
                continue
            text = str(row.get("quoted_text") or "")
            if bab.casefold() in text.casefold() and any(word in text.casefold() for word in intent["instrument_deletion_evidence_words"]):
                candidate = _candidate(row, "instrument_clause_candidate")
                if candidate is not None:
                    matches.append(candidate)
        return tuple(matches[:limit])
    decision = resolve_instrument_intent(query, intent, corpus=corpus_id)
    if decision.target_status == "instrument_unresolved":
        return ({"probe": True},) if probe_only else ()
    if decision.target_status.startswith("instrument_resolved") and decision.target_citation:
        if probe_only:
            return ({"probe": True},)
        row = _instrument_evidence(store, decision.amendment or "", decision.target_citation)
        candidate = _candidate(row, f"instrument_{decision.role_family}_candidate")
        return (candidate,) if candidate is not None else ()
    return ()


def _targets(query: str, intent: dict, corpus_id: str) -> tuple[str, ...]:
    text = query or ""
    folded = text.casefold()
    for section in intent["structured_sections"]:
        if any(alias in folded for alias in section.get("aliases", ())):
            target = section["target"]
            return _with_pasal(target, text, corpus_id) if section.get("with_pasal") else (target,)
    bab = parse_bab_reference(corpus_id, text)
    if bab:
        return (bab.casefold(),)
    pasal = parse_pasal_reference(corpus_id, text, allow_roman=True)
    if pasal:
        targets = [pasal.casefold()]
        ayat = parse_ayat_reference(corpus_id, text)
        if ayat:
            targets.append(ayat)
        return tuple(targets)
    return ()


def _navigation_rows(
    store: EvidenceStore,
    query: str,
    limit: int,
    corpus_id: str,
) -> tuple[dict, ...]:
    navigation = resolve_navigation(corpus_id, query)
    if navigation is None:
        return ()
    label, direction = navigation
    preferred_role = getattr(store.config, "preferred_source_role", None)
    source = next(
        (
            row
            for row in store.legal_units
            if row.get("unit_label", "").casefold() == label.casefold()
            and row.get("structural_role") == "provision"
            and (preferred_role is None or row.get("source_role") == preferred_role)
        ),
        None,
    )
    if source is None:
        return ()
    offset = 1 if direction == "next" else -1
    target_order = source.get("sibling_order", -2) + offset
    target = next(
        (
            row
            for row in store.legal_units
            if row.get("source_document_id") == source.get("source_document_id")
            and row.get("parent_legal_unit_id") == source.get("parent_legal_unit_id")
            and row.get("sibling_order") == target_order
            and row.get("structural_role") == "provision"
        ),
        None,
    )
    if target is None:
        return ()
    rows = [
        row | {"candidate_type": "structural_navigation_candidate", "navigation_direction": direction}
        for row in store.evidence
        if row.get("legal_unit_id") == target.get("legal_unit_id") and row.get("status") == "final" and store.bboxes_for(row["evidence_id"])
    ]
    return tuple(rows[:limit])


def _with_pasal(section: str, text: str, corpus_id: str) -> tuple[str, ...]:
    pasal = parse_pasal_reference(corpus_id, text, allow_roman=True)
    return (section, pasal.casefold()) if pasal else (section,)


def _matches(row: dict, targets: tuple[str, ...]) -> bool:
    values = [row.get("citation", ""), *(row.get("hierarchy") or ())]
    haystack = {key for value in values for key in _label_keys(value)}
    return all(target in haystack for target in targets)


def _matches_unit(row: dict, targets: tuple[str, ...]) -> bool:
    values = [row.get("unit_label", ""), *(row.get("hierarchy") or ())]
    haystack = {key for value in values for key in _label_keys(value)}
    return all(target in haystack for target in targets)


def _label_keys(value: object) -> set[str]:
    return label_keys(DEFAULT_CORPUS_ID, value)


def _corpus_id(config) -> str:
    return getattr(config, "corpus_id", DEFAULT_CORPUS_ID)


def _instrument_evidence(store: EvidenceStore | None, source_role: str, citation: str) -> dict | None:
    if store is None:
        return None
    return next(
        (
            row
            for row in store.evidence
            if row.get("source_role") == source_role and row.get("citation") == citation and row.get("status") == "final"
        ),
        None,
    )


def _candidate(row: dict | None, candidate_type: str) -> dict | None:
    return row | {"candidate_type": candidate_type} if row else None
