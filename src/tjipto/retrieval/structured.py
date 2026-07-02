from __future__ import annotations

from tjipto.corpora.intent_config import intent_config_for, resolve_instrument_intent
from tjipto.corpora.parser_dispatch import (
    DEFAULT_CORPUS_ID,
    label_keys,
    parse_ayat_reference,
    parse_bab_reference,
    parse_pasal_reference,
)
from tjipto.evidence.store import EvidenceStore


def structured_lookup(store: EvidenceStore, query: str, limit: int = 10, *, strategy: str = "uud_1945") -> tuple[dict, ...]:
    config = getattr(store, "config", None)
    intent = intent_config_for(strategy, config)
    if not intent["structured_lookup_enabled"]:
        return ()
    instrument = _instrument_rows(store, query, limit, strategy=strategy, config=config)
    if instrument:
        return instrument
    targets = _targets(query, intent, _corpus_id(config))
    if not targets:
        return ()
    legal_unit_ids = {
        row["legal_unit_id"]
        for row in (*getattr(store, "legal_units", ()), *getattr(store, "chunks", ()))
        if row.get("legal_unit_id") and _matches_unit(row, targets)
    }
    rows = [
        row for row in store.evidence
        if row.get("status") == "final"
        and store.bboxes_for(row["evidence_id"])
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
        matches = []
        prefix = intent["instrument_citation_templates"].get("prefix", "")
        clause_marker = intent["instrument_citation_templates"].get("clause_marker", "")
        for row in getattr(store, "evidence", ()):
            citation = str(row.get("citation") or "")
            if not (prefix and clause_marker and citation.startswith(prefix) and clause_marker in citation):
                continue
            text = str(row.get("quoted_text") or "")
            if bab.casefold() in text.casefold() and any(word in text.casefold() for word in intent["instrument_deletion_evidence_words"]):
                matches.append(_candidate(row, "instrument_clause_candidate"))
        return tuple(matches[:limit])
    decision = resolve_instrument_intent(query, intent, corpus=corpus_id)
    if decision.target_status == "instrument_unresolved":
        return ({"probe": True},) if probe_only else ()
    if decision.target_status.startswith("instrument_resolved") and decision.target_citation:
        if probe_only:
            return ({"probe": True},)
        row = _instrument_evidence(store, decision.amendment or "", decision.target_citation)
        return (_candidate(row, f"instrument_{decision.role_family}_candidate"),) if row else ()
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
            row for row in store.evidence
            if row.get("source_role") == source_role
            and row.get("citation") == citation
            and row.get("status") == "final"
        ),
        None,
    )


def _candidate(row: dict | None, candidate_type: str) -> dict | None:
    return row | {"candidate_type": candidate_type} if row else None
