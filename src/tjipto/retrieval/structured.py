from __future__ import annotations

import re

from tjipto.evidence.store import EvidenceStore


AYAT_RE = re.compile(r"\bayat\s*\(?\s*([0-9]+)\s*\)?", re.IGNORECASE)
BAB_RE = re.compile(r"\bbab\s+([ivxlcdm]+)\b", re.IGNORECASE)
PASAL_RE = re.compile(r"\bpasal\s+([0-9]+[a-z]?|[ivxlcdm]+)\b", re.IGNORECASE)


def structured_lookup(store: EvidenceStore, query: str, limit: int = 10) -> tuple[dict, ...]:
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
        return (f"bab {bab.group(1)}".casefold(),)
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
    haystack = {str(value).casefold() for value in values}
    return all(target in haystack for target in targets)


def _matches_unit(row: dict, targets: tuple[str, ...]) -> bool:
    values = [row.get("unit_label", ""), *(row.get("hierarchy") or ())]
    haystack = {str(value).casefold() for value in values}
    return all(target in haystack for target in targets)
