from __future__ import annotations

import math
import re
from collections import Counter


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
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


def tokens(text: str, *, aliases: dict[str, str] | None = None) -> list[str]:
    return [_normalize_token(token.casefold(), aliases or {}) for token in TOKEN_RE.findall(text or "")]


def meaningful_tokens(text: str, *, aliases: dict[str, str] | None = None) -> set[str]:
    return {_normalize_token(token, aliases or {}) for token in tokens(text, aliases=aliases) if token not in STOPWORDS and len(token) > 2}


def _normalize_token(token: str, aliases: dict[str, str]) -> str:
    return aliases.get(token, token)


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
    config=None,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[dict]:
    aliases = _lexical_aliases(config)
    query_terms = tokens(query, aliases=aliases)
    if not query_terms:
        return []

    docs = [(row, tokens(_document_text(row), aliases=aliases)) for row in evidence]
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
            scored.append((score, row["evidence_id"], _with_relevance(row, query, aliases)))
    return [row for _, _, row in sorted(scored, key=lambda item: (-item[0], item[1]))[:limit]]


def _with_relevance(row: dict, query: str, aliases: dict[str, str]) -> dict:
    query_terms = meaningful_tokens(query, aliases=aliases)
    doc_terms = meaningful_tokens(_document_text(row), aliases=aliases)
    supported = query_terms & doc_terms
    # A lexical hit is answerable only when every meaningful query term is
    # present in the same evidence row. Partial overlap is a candidate signal,
    # not proof for the answer.
    required = len(query_terms)
    ok = bool(query_terms) and len(supported) >= required
    return dict(
        row,
        lexical_query_terms=tuple(sorted(query_terms)),
        lexical_supported_terms=tuple(sorted(supported)),
        lexical_relevance_ok=ok,
        lexical_relevance_reason="answer_evidence" if ok else "insufficient_query_support",
    )


def _lexical_aliases(config) -> dict[str, str]:
    settings: dict = getattr(config, "setting", lambda *_: {})("lexical_normalization", {}) or {}
    return {str(key).casefold(): str(value).casefold() for key, value in dict(settings.get("aliases") or {}).items()}
