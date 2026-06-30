from __future__ import annotations

import re

from tjipto.corpora.intent_config import intent_config_for
from tjipto.evidence.store import EvidenceStore


AYAT_RE = re.compile(r"\bayat\s*\(?\s*([0-9]+)\s*\)?", re.IGNORECASE)
BAB_RE = re.compile(r"\bbab\s+([ivxlcdm]+)\s*([a-z]?)\b", re.IGNORECASE)
PASAL_RE = re.compile(r"\bpasal\s+([0-9]+[a-z]?|[ivxlcdm]+)\b", re.IGNORECASE)


def structured_lookup(store: EvidenceStore, query: str, limit: int = 10, *, strategy: str = "uud_1945") -> tuple[dict, ...]:
    config = getattr(store, "config", None)
    intent = intent_config_for(strategy, config)
    if not intent["structured_lookup_enabled"]:
        return ()
    instrument = _instrument_rows(store, query, limit, strategy=strategy, config=config)
    if instrument:
        return instrument
    targets = _targets(query, intent)
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
    return bool(_targets(query, intent))


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
    role = _source_role(query, strategy=strategy, config=config)
    bab = BAB_RE.search(query or "")
    if (
        bab
        and any(pattern in folded for pattern in intent["instrument_deletion_words"])
        and any(pattern in folded for pattern in intent["instrument_change_context_words"])
    ):
        if probe_only:
            return ({"probe": True},)
        target = f"BAB {bab.group(1).upper()}{bab.group(2).upper()}".strip()
        matches = []
        for row in getattr(store, "evidence", ()):
            citation = str(row.get("citation") or "")
            if not (citation.startswith("Perubahan ") and "Clause" in citation):
                continue
            text = str(row.get("quoted_text") or "")
            if target.casefold() in text.casefold() and any(word in text.casefold() for word in intent["instrument_deletion_evidence_words"]):
                matches.append(_candidate(row, "instrument_clause_candidate"))
        return tuple(matches[:limit])
    if role is None:
        return ()
    if _scope_query(folded, intent):
        if probe_only:
            return ({"probe": True},)
        row = _instrument_evidence(store, role, f"Perubahan {_ordinal_label(role, intent)} Scope")
        return (_candidate(row, "instrument_scope_candidate"),) if row else ()
    clause = _clause_letter(query)
    if clause:
        if probe_only:
            return ({"probe": True},)
        row = _instrument_evidence(store, role, f"Perubahan {_ordinal_label(role, intent)} Clause ({clause})")
        return (_candidate(row, "instrument_clause_candidate"),) if row else ()
    return ()


def _targets(query: str, intent: dict) -> tuple[str, ...]:
    text = query or ""
    folded = text.casefold()
    for section in intent["structured_sections"]:
        if any(alias in folded for alias in section.get("aliases", ())):
            target = section["target"]
            return _with_pasal(target, text) if section.get("with_pasal") else (target,)
    bab = BAB_RE.search(text)
    if bab:
        return (f"bab {bab.group(1)}{bab.group(2)}".casefold(),)
    pasal = PASAL_RE.search(text)
    if pasal:
        targets = [f"pasal {pasal.group(1).upper()}".casefold()]
        ayat = AYAT_RE.search(text)
        if ayat:
            targets.append(f"({ayat.group(1)})")
        return tuple(targets)
    return ()


def _with_pasal(section: str, text: str) -> tuple[str, ...]:
    pasal = PASAL_RE.search(text)
    return (section, f"pasal {pasal.group(1).upper()}".casefold()) if pasal else (section,)


def _matches(row: dict, targets: tuple[str, ...]) -> bool:
    values = [row.get("citation", ""), *(row.get("hierarchy") or ())]
    haystack = {key for value in values for key in _label_keys(value)}
    return all(target in haystack for target in targets)


def _matches_unit(row: dict, targets: tuple[str, ...]) -> bool:
    values = [row.get("unit_label", ""), *(row.get("hierarchy") or ())]
    haystack = {key for value in values for key in _label_keys(value)}
    return all(target in haystack for target in targets)


def _label_keys(value: object) -> set[str]:
    label = str(value).casefold()
    compact_bab = re.sub(r"\bbab\s+([ivxlcdm]+)\s+([a-z])\b", r"bab \1\2", label)
    return {label, compact_bab}


def _scope_query(folded: str, intent: dict) -> bool:
    return any(pattern in folded for pattern in intent["instrument_scope_queries"])


def _clause_letter(query: str) -> str | None:
    match = re.search(r"\b(?:butir|clause)\s*\(?([a-e])\)?", query or "", re.IGNORECASE)
    if not match:
        match = re.search(r"\(([a-e])\)", query or "", re.IGNORECASE)
    return match.group(1).lower() if match else None


def _source_role(query: str, *, strategy: str, config=None) -> str | None:
    for role, pattern in intent_config_for(strategy, config)["metadata_roles"]:
        if pattern.search(query or ""):
            return role
    return None


def _ordinal_label(role: str, intent: dict) -> str:
    return intent["source_role_labels"][role]


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
