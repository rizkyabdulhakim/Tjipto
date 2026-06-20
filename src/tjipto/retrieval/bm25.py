from __future__ import annotations

import math
import re
from collections import Counter


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokens(text: str) -> list[str]:
    return [token.casefold() for token in TOKEN_RE.findall(text or "")]


def _document_text(row: dict) -> str:
    return " ".join([
        row.get("quoted_text", ""),
        row.get("citation", ""),
        " ".join(row.get("hierarchy") or []),
    ])


def lexical_search(
    evidence: list[dict],
    query: str,
    limit: int = 10,
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[dict]:
    query_terms = tokens(query)
    if not query_terms:
        return []

    docs = [(row, tokens(_document_text(row))) for row in evidence]
    if not docs:
        return []
    avgdl = sum(len(doc_terms) for _, doc_terms in docs) / len(docs) or 1.0

    document_frequency: Counter[str] = Counter()
    for _, doc_terms in docs:
        document_frequency.update(set(doc_terms))

    scored: list[tuple[float, str, dict]] = []
    total_docs = len(docs)
    for row, doc_terms in docs:
        if not doc_terms:
            continue
        frequencies = Counter(doc_terms)
        doc_len = len(doc_terms)
        score = 0.0
        for term in query_terms:
            tf = frequencies.get(term, 0)
            if not tf:
                continue
            idf = math.log(1 + (total_docs - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5))
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avgdl))
        if score > 0:
            scored.append((score, row["evidence_id"], row))
    return [row for _, _, row in sorted(scored, key=lambda item: (-item[0], item[1]))[:limit]]
