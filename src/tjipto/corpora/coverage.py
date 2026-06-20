from __future__ import annotations

NON_UUD_LEGAL_TERMS = (
    "kuhp",
    "kuhap",
    "uu ite",
    "uu pers",
    "ketenagakerjaan",
    "uu pdp",
    "perseroan",
    "pemilu",
)


def required_missing_corpus(corpus_id: str, query: str) -> str | None:
    if corpus_id != "uud":
        return None
    lowered = query.casefold()
    return "non_uud" if any(term in lowered for term in NON_UUD_LEGAL_TERMS) else None
