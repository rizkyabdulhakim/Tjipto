from __future__ import annotations

import re


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokens(text: str) -> set[str]:
    return {token.casefold() for token in TOKEN_RE.findall(text or "")}


def lexical_search(evidence: list[dict], query: str, limit: int = 10) -> list[dict]:
    query_tokens = tokens(query)
    scored = []
    for row in evidence:
        haystack = " ".join([
            row.get("quoted_text", ""),
            row.get("citation", ""),
            " ".join(row.get("hierarchy") or []),
        ])
        score = len(query_tokens & tokens(haystack))
        if score:
            scored.append((score, row["evidence_id"], row))
    return [row for _, _, row in sorted(scored, key=lambda item: (-item[0], item[1]))[:limit]]
