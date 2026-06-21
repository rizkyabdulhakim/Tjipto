from __future__ import annotations

import re


BAB_XA_PASAL_RE = re.compile(r"\bpasal\s+28[a-j]\b", re.IGNORECASE)
UUD_CORPUS_ID = "".join(("u", "ud"))


def public_hierarchy(row: dict) -> tuple:
    hierarchy = tuple(item for item in (row.get("hierarchy") or ()) if item)
    if row.get("corpus_id") == UUD_CORPUS_ID and hierarchy[:1] == ("BAB X",) and _is_bab_xa_article(row):
        return ("BAB XA", *hierarchy[1:])
    return hierarchy


def matches_bab_xa_target(row: dict, targets: tuple[str, ...]) -> bool:
    return targets == ("bab xa",) and _is_bab_xa_article(row)


def _is_bab_xa_article(row: dict) -> bool:
    if row.get("corpus_id") != UUD_CORPUS_ID:
        return False
    values = [row.get("citation", ""), *(row.get("hierarchy") or ())]
    return any(BAB_XA_PASAL_RE.search(str(value)) for value in values)
