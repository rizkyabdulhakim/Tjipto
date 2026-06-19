from __future__ import annotations

from tjipto.evidence.citation import evidence_matches_citation, parse_citation


def exact_citation(evidence: list[dict], query: str, source_role: str | None = None) -> list[dict]:
    pasal, ayat = parse_citation(query)
    matches = [
        row for row in evidence
        if row.get("status") == "final"
        and (source_role is None or row.get("source_role") == source_role)
        and evidence_matches_citation(row, pasal, ayat)
    ]
    return sorted(matches, key=lambda row: (row.get("source_role") != "current_consolidated", row["evidence_id"]))
