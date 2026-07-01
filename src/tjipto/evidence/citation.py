from __future__ import annotations

from tjipto.corpora.parser_dispatch import DEFAULT_CORPUS_ID, parse_ayat_reference, parse_pasal_reference


def parse_citation(text: str) -> tuple[str | None, str | None]:
    return parse_pasal_reference(DEFAULT_CORPUS_ID, text), parse_ayat_reference(DEFAULT_CORPUS_ID, text)


def evidence_matches_citation(row: dict, pasal: str | None, ayat: str | None) -> bool:
    hierarchy = tuple(row.get("hierarchy") or ())
    if pasal and pasal not in hierarchy and row.get("citation") != pasal:
        return False
    if ayat and ayat not in hierarchy and row.get("citation") != ayat:
        return False
    return bool(pasal)
