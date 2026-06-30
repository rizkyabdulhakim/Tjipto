from __future__ import annotations

import re

from tjipto.evidence.citation import parse_citation


PASAL_LETTER_RE = re.compile(r"\bpasal\s+([0-9]+)\s+([a-z])\b", re.IGNORECASE)
PASAL_SHORTHAND_AYAT_RE = re.compile(r"\bpasal\s+([0-9]+[a-z]?)\s*\(\s*([0-9]+)\s*\)", re.IGNORECASE)
PASAL_RE = re.compile(r"\bpasal\s+([0-9]+[a-z]?)\b", re.IGNORECASE)
AYAT_RE = re.compile(r"\bayat\s*\(?\s*([0-9]+)\s*\)?", re.IGNORECASE)


def normalize_query(query: str, *, strategy: str = "generic", config=None) -> dict:
    original = query or ""
    normalized = original.strip()
    if not _setting_enabled(config, "query_normalization_enabled"):
        return {
            "original_query": original,
            "normalized_query": re.sub(r"\s+", " ", normalized).strip(),
        }
    normalized = _apply_alias_rules(normalized, config)
    normalized = PASAL_LETTER_RE.sub(
        lambda match: f"Pasal {match.group(1)}{match.group(2).upper()}",
        normalized,
    )
    normalized = PASAL_SHORTHAND_AYAT_RE.sub(
        lambda match: f"Pasal {match.group(1).upper()} ayat ({match.group(2)})",
        normalized,
    )
    normalized = PASAL_RE.sub(lambda match: f"Pasal {match.group(1).upper()}", normalized)
    normalized = AYAT_RE.sub(lambda match: f"ayat ({match.group(1)})", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return {"original_query": original, "normalized_query": normalized}


def _apply_alias_rules(text: str, config=None) -> str:
    if config is None:
        return text
    for rule in config.setting("normalization_aliases", ()):
        text = re.sub(rule["pattern"], rule["replacement"], text, flags=re.IGNORECASE)
    return text


def classify_intent(
    corpus_id: str,
    query: str,
    *,
    corpus_supported: bool = True,
    strategy: str = "generic",
    config=None,
) -> dict:
    if not corpus_supported:
        return {"intent": "unsupported_corpus"}
    if not _setting_enabled(config, "exact_citation_intent_enabled"):
        return {"intent": "natural_language"}
    pasal, _ = parse_citation(query)
    return {"intent": "exact_citation" if pasal else "natural_language"}


def _setting_enabled(config, key: str) -> bool:
    return bool(getattr(config, "setting", lambda *_: False)(key, False))
