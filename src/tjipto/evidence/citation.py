from __future__ import annotations

import re


PASAL_RE = re.compile(r"pasal\s+([0-9]+[A-Z]?)", re.IGNORECASE)
AYAT_RE = re.compile(r"ayat\s*\(?([0-9]+)\)?", re.IGNORECASE)


def parse_citation(text: str) -> tuple[str | None, str | None]:
    pasal = PASAL_RE.search(text or "")
    ayat = AYAT_RE.search(text or "")
    return (
        f"Pasal {pasal.group(1).upper()}" if pasal else None,
        f"({ayat.group(1)})" if ayat else None,
    )


def evidence_matches_citation(row: dict, pasal: str | None, ayat: str | None) -> bool:
    hierarchy = tuple(row.get("hierarchy") or ())
    if pasal and pasal not in hierarchy and row.get("citation") != pasal:
        return False
    if ayat and ayat not in hierarchy and row.get("citation") != ayat:
        return False
    return bool(pasal)
