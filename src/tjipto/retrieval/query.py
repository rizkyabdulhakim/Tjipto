from __future__ import annotations

import re
from dataclasses import dataclass

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
    normalized = _apply_source_reference_mappings(normalized, config)
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


@dataclass(frozen=True)
class LegalReferenceMapping:
    raw_reference: str
    canonical_target: str
    source_role: str
    mapping_kind: str
    provenance: str
    context_terms: tuple[str, ...]


def _apply_source_reference_mappings(text: str, config=None) -> str:
    """Apply only corpus-declared source-reference discrepancies."""
    for raw in config.setting("source_reference_mappings", ()) if config is not None else ():
        try:
            mapping = LegalReferenceMapping(
                raw_reference=str(raw["raw_reference"]),
                canonical_target=str(raw["canonical_target"]),
                source_role=str(raw["source_role"]),
                mapping_kind=str(raw["mapping_kind"]),
                provenance=str(raw["provenance"]),
                context_terms=tuple(str(value) for value in raw["context_terms"]),
            )
        except (KeyError, TypeError):
            continue
        if not mapping.context_terms or not all(
            re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text, re.IGNORECASE)
            for term in mapping.context_terms
        ):
            continue
        text = re.sub(
            rf"(?<!\w){re.escape(mapping.raw_reference)}(?!\w)",
            mapping.canonical_target,
            text,
            flags=re.IGNORECASE,
        )
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
