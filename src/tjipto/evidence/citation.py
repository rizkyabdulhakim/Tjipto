from __future__ import annotations

from tjipto.corpora.uud.parser import parse_uud_ayat_reference, parse_uud_pasal_reference


def parse_citation(text: str) -> tuple[str | None, str | None]:
    return parse_uud_pasal_reference(text), parse_uud_ayat_reference(text)


def evidence_matches_citation(row: dict, pasal: str | None, ayat: str | None) -> bool:
    hierarchy = tuple(row.get("hierarchy") or ())
    if pasal and pasal not in hierarchy and row.get("citation") != pasal:
        return False
    if ayat and ayat not in hierarchy and row.get("citation") != ayat:
        return False
    return bool(pasal)
