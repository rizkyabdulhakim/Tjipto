from __future__ import annotations

from tjipto.corpora.parser_dispatch import parse_ayat_reference, parse_pasal_reference


def parse_citation(corpus_id: str, text: str) -> tuple[str | None, str | None]:
    return parse_pasal_reference(corpus_id, text), parse_ayat_reference(corpus_id, text)


def evidence_matches_citation(row: dict, pasal: str | None, ayat: str | None) -> bool:
    hierarchy = tuple(row.get("hierarchy") or ())
    if pasal and pasal not in hierarchy and row.get("citation") != pasal:
        return False
    if ayat and ayat not in hierarchy and row.get("citation") != ayat:
        return False
    return bool(pasal)
