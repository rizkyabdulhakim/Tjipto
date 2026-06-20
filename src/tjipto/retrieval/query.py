from __future__ import annotations

import re

from tjipto.corpora.coverage import classify_coverage
from tjipto.evidence.citation import parse_citation


PASAL_LETTER_RE = re.compile(r"\bpasal\s+([0-9]+)\s+([a-z])\b", re.IGNORECASE)
PASAL_RE = re.compile(r"\bpasal\s+([0-9]+[a-z]?)\b", re.IGNORECASE)
AYAT_RE = re.compile(r"\bayat\s*\(?\s*([0-9]+)\s*\)?", re.IGNORECASE)
UUD_45_RE = re.compile(r"\bu\s*u\s*d\s+45\b", re.IGNORECASE)


def normalize_query(query: str) -> dict:
    original = query or ""
    normalized = original.strip()
    normalized = UUD_45_RE.sub("UUD 1945", normalized)
    normalized = PASAL_LETTER_RE.sub(
        lambda match: f"Pasal {match.group(1)}{match.group(2).upper()}",
        normalized,
    )
    normalized = PASAL_RE.sub(lambda match: f"Pasal {match.group(1).upper()}", normalized)
    normalized = AYAT_RE.sub(lambda match: f"ayat ({match.group(1)})", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return {"original_query": original, "normalized_query": normalized}


def classify_intent(corpus_id: str, query: str, *, corpus_supported: bool = True) -> dict:
    if not corpus_supported:
        return {"intent": "unsupported_corpus", "required_corpus": None}
    coverage = classify_coverage(corpus_id, query)
    missing = coverage["required_corpus"] if not coverage["coverage_warning"] else None
    if missing:
        return {"intent": "out_of_corpus", "required_corpus": missing, "coverage": coverage}
    pasal, _ = parse_citation(query)
    return {
        "intent": "exact_citation" if pasal else "natural_language",
        "required_corpus": None,
        "coverage": coverage,
    }
