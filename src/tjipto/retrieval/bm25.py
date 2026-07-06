from __future__ import annotations

import math
import re
from collections import Counter


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
NUMBER_WORDS = {
    "satu": "1",
    "dua": "2",
    "tiga": "3",
    "empat": "4",
    "lima": "5",
    "enam": "6",
    "tujuh": "7",
    "delapan": "8",
    "sembilan": "9",
    "sepuluh": "10",
}
TERM_ALIASES = {
    "berhak": "hak",  # nosec B105
    "bekerja": "kerja",
    "menjabat": "jabatan",
    "pekerjaan": "kerja",
}
# Shared Indonesian lexical baseline; keep corpus config for legal structure/policy.
STOPWORDS = {
    "adalah",
    "atau",
    "aturan",
    "berapa",
    "boleh",
    "dalam",
    "dan",
    "dari",
    "dengan",
    "di",
    "diatur",
    "ke",
    "ketentuan",
    "lama",
    "pada",
    "tentang",
    "undang",
    "undang-undang",
    "untuk",
    "yang",
}


def tokens(text: str) -> list[str]:
    return [_normalize_token(token.casefold()) for token in TOKEN_RE.findall(text or "")]


def meaningful_tokens(text: str) -> set[str]:
    return {_normalize_token(token) for token in tokens(text) if token not in STOPWORDS and len(token) > 2}


def _normalize_token(token: str) -> str:
    return NUMBER_WORDS.get(token, TERM_ALIASES.get(token, token))


def _document_text(row: dict) -> str:
    return " ".join(
        [
            row.get("quoted_text", ""),
            row.get("citation", ""),
            " ".join(row.get("hierarchy") or []),
        ]
    )


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
            scored.append((score, row["evidence_id"], _with_relevance(row, query)))
    return [row for _, _, row in sorted(scored, key=lambda item: (-item[0], item[1]))[:limit]]


def _with_relevance(row: dict, query: str) -> dict:
    query_terms = meaningful_tokens(query)
    doc_terms = meaningful_tokens(_document_text(row))
    supported = query_terms & doc_terms
    required = len(query_terms) if len(query_terms) <= 2 else max(2, (len(query_terms) + 1) // 2)
    ok = bool(query_terms) and len(supported) >= required
    return dict(
        row,
        lexical_query_terms=tuple(sorted(query_terms)),
        lexical_supported_terms=tuple(sorted(supported)),
        lexical_relevance_ok=ok,
        lexical_relevance_reason="answer_evidence" if ok else "insufficient_query_support",
    )
