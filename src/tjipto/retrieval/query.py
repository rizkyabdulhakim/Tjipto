from __future__ import annotations

import re

from tjipto.corpora.parser_dispatch import normalize_query_reference
from tjipto.evidence.citation import parse_citation


def normalize_query(query: str, *, strategy: str = "generic", config=None) -> dict:
    original = query or ""
    normalized = original.strip()
    if not _setting_enabled(config, "query_normalization_enabled"):
        return {
            "original_query": original,
            "normalized_query": re.sub(r"\s+", " ", normalized).strip(),
        }
    normalized = _apply_alias_rules(normalized, config)
    corpus_id = str(getattr(config, "corpus_id", "") or "")
    if corpus_id:
        normalized = normalize_query_reference(corpus_id, normalized, config=config)
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
    corpus_id = str(getattr(config, "corpus_id", "") or "")
    if not corpus_id:
        return {"intent": "natural_language"}
    pasal, _ = parse_citation(corpus_id, query)
    return {"intent": "exact_citation" if pasal else "natural_language"}


def _setting_enabled(config, key: str) -> bool:
    return bool(getattr(config, "setting", lambda *_: False)(key, False))
