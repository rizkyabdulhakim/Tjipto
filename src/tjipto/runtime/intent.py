from __future__ import annotations

from dataclasses import dataclass

from tjipto.corpora.intent_config import contains_intent_phrase, intent_config_for
from tjipto.corpora.parser_dispatch import parse_legal_reference, parse_legal_references


@dataclass(frozen=True)
class RelationIntent:
    requested_function: str = "unknown"
    target_reference: str | None = None
    relation_type: str | None = None
    legal_domain: str | None = None
    answerability: str = "unknown"
    rejection_reason: str | None = None
    route: str | None = None
    intent: str | None = None
    required_capabilities: tuple[str, ...] = ()


def classify_relation_intent(store, query: str) -> RelationIntent:
    if store is None:
        return RelationIntent()
    intent = intent_config_for(getattr(store.config, "query_strategy", "generic"), store.config)
    relation_config = intent.get("document_relation", {})
    for name, row in (relation_config.get("relation_families") or {}).items():
        explicit_mapping = (
            _explicit_renumbering_mapping(query, getattr(store.config, "corpus_id", "")) if name == "RENAME_PROVISION" else False
        )
        if contains_intent_phrase(query, tuple(row.get("terms") or ())) or explicit_mapping:
            return RelationIntent(
                "amendment_relation",
                _target_reference(store, query),
                relation_type=str(name),
                legal_domain="constitutional_amendment",
                answerability="answerable",
            )
    return RelationIntent()


def _explicit_renumbering_mapping(query: str, corpus_id: str) -> bool:
    folded = (query or "").casefold()
    return "menjadi" in folded and len(parse_legal_references(corpus_id, query)) >= 2


def _target_reference(store, query: str) -> str | None:
    try:
        ref = parse_legal_reference(getattr(store.config, "corpus_id", ""), query)
    except ValueError:
        return None
    pasal = ref.get("pasal")
    ayat = ref.get("ayat")
    return f"{pasal} ayat {ayat}" if pasal and ayat else pasal
