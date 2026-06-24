from __future__ import annotations

import re

from tjipto.corpora.intent_config import intent_config_for
from tjipto.evidence.store import EvidenceStore


AYAT_RE = re.compile(r"\bayat\s*\(?\s*([0-9]+)\s*\)?", re.IGNORECASE)
BAB_RE = re.compile(r"\bbab\s+([ivxlcdm]+)\s*([a-z]?)\b", re.IGNORECASE)
PASAL_RE = re.compile(r"\bpasal\s+([0-9]+[a-z]?|[ivxlcdm]+)\b", re.IGNORECASE)


def structured_lookup(store: EvidenceStore, query: str, limit: int = 10, *, strategy: str = "uud_1945") -> tuple[dict, ...]:
    if strategy != "uud_1945":
        return ()
    instrument = _instrument_rows(store, query, limit, strategy=strategy)
    if instrument:
        return instrument
    targets = _targets(query)
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


def has_structured_target(query: str, *, strategy: str = "uud_1945") -> bool:
    if strategy != "uud_1945":
        return False
    if _instrument_target(query, strategy=strategy):
        return True
    return bool(_targets(query))


def _instrument_target(query: str, *, strategy: str) -> bool:
    return bool(_instrument_rows(None, query, 1, strategy=strategy, probe_only=True))


def _instrument_rows(
    store: EvidenceStore | None,
    query: str,
    limit: int,
    *,
    strategy: str,
    probe_only: bool = False,
) -> tuple[dict, ...]:
    folded = (query or "").casefold()
    role = _source_role(query, strategy=strategy)
    bab = BAB_RE.search(query or "")
    if bab and any(pattern in folded for pattern in ("dihapus", "penghapusan")) and "perubahan" in folded:
        if probe_only:
            return ({"probe": True},)
        target = f"BAB {bab.group(1).upper()}{bab.group(2).upper()}".strip()
        matches = []
        for row in getattr(store, "evidence", ()):
            citation = str(row.get("citation") or "")
            if not (citation.startswith("Perubahan ") and "Clause" in citation):
                continue
            text = str(row.get("quoted_text") or "")
            if target.casefold() in text.casefold() and ("hapus" in text.casefold() or "penghapusan" in text.casefold()):
                matches.append(_candidate(row, "instrument_clause_candidate"))
        return tuple(matches[:limit])
    if role is None:
        return ()
    if _scope_query(folded):
        if probe_only:
            return ({"probe": True},)
        row = _instrument_evidence(store, role, f"Perubahan {_ordinal_label(role)} Scope")
        return (_candidate(row, "instrument_scope_candidate"),) if row else ()
    clause = _clause_letter(query)
    if clause:
        if probe_only:
            return ({"probe": True},)
        row = _instrument_evidence(store, role, f"Perubahan {_ordinal_label(role)} Clause ({clause})")
        return (_candidate(row, "instrument_clause_candidate"),) if row else ()
    return ()


def _targets(query: str) -> tuple[str, ...]:
    text = query or ""
    folded = text.casefold()
    if "pembukaan" in folded:
        return ("pembukaan/preambule",)
    if "aturan peralihan" in folded:
        return _with_pasal("aturan peralihan", text)
    if "aturan tambahan" in folded:
        return _with_pasal("aturan tambahan", text)
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


def _scope_query(folded: str) -> bool:
    return any(pattern in folded for pattern in ("pasal apa saja", "mengubah pasal apa", "mengubah pasal apa saja", "menambah pasal apa"))


def _clause_letter(query: str) -> str | None:
    match = re.search(r"\b(?:butir|clause)\s*\(?([a-e])\)?", query or "", re.IGNORECASE)
    if not match:
        match = re.search(r"\(([a-e])\)", query or "", re.IGNORECASE)
    return match.group(1).lower() if match else None


def _source_role(query: str, *, strategy: str) -> str | None:
    for role, pattern in intent_config_for(strategy)["metadata_roles"]:
        if pattern.search(query or ""):
            return role
    return None


def _ordinal_label(role: str) -> str:
    return {
        "amendment_1_historical": "Pertama",
        "amendment_2_historical": "Kedua",
        "amendment_3_historical": "Ketiga",
        "amendment_4_historical": "Keempat",
    }[role]


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
