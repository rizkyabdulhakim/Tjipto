from __future__ import annotations

from dataclasses import dataclass

from tjipto.corpora.intent_config import contains_intent_phrase, intent_config_for
from tjipto.corpora.parser_dispatch import parse_legal_reference, parse_legal_references


@dataclass(frozen=True)
class LegalIntent:
    requested_function: str = "unknown"
    target_reference: str | None = None
    relation_type: str | None = None
    legal_domain: str | None = None
    answerability: str = "unknown"
    rejection_reason: str | None = None
    route: str | None = None
    intent: str | None = None


def classify_legal_intent(store, query: str) -> LegalIntent:
    guard = store.config.setting("scope_guard", {}) or {}
    current = contains_intent_phrase(query, tuple(guard.get("current_fact_terms") or ()))
    subject = contains_intent_phrase(query, tuple(guard.get("current_fact_subjects") or ()))
    identity = contains_intent_phrase(query, tuple(guard.get("identity_question_terms") or ()))
    legal_scope = contains_intent_phrase(query, tuple(guard.get("legal_scope_terms") or ()))
    if subject and (current or (identity and not legal_scope)):
        return LegalIntent(
            "current_fact",
            answerability="unsupported",
            rejection_reason="current_fact_unsupported",
            route="current_fact_unsupported",
            intent="current_fact_query",
        )
    for row in (guard.get("legal_intent_policy", {}) or {}).get("unsupported_functions", ()):
        topic = contains_intent_phrase(query, tuple(row.get("topic_terms") or ()))
        unsupported = contains_intent_phrase(query, tuple(row.get("unsupported_function_terms") or ()))
        ambiguous = contains_intent_phrase(query, tuple(row.get("ambiguous_criminal_terms") or ()))
        supported = contains_intent_phrase(query, tuple(row.get("supported_function_terms") or ()))
        target = contains_intent_phrase(query, tuple(row.get("target_reference_terms") or ()))
        if unsupported and (topic or target) or (topic and ambiguous and not supported):
            return LegalIntent(
                str(row.get("requested_function") or "out_of_corpus_domain"),
                _target_reference(store, query),
                legal_domain=row.get("legal_domain"),
                answerability="unsupported",
                rejection_reason=str(row.get("rejection_reason") or "unsupported_scope"),
                route="unsupported_scope",
                intent="out_of_corpus",
            )
    out_of_corpus_terms = tuple(guard.get("out_of_corpus_terms") or ())
    if out_of_corpus_terms and not _target_reference(store, query) and contains_intent_phrase(query, out_of_corpus_terms):
        return LegalIntent(
            "out_of_corpus_domain",
            answerability="unsupported",
            rejection_reason="unsupported_scope",
            route="unsupported_scope",
            intent="out_of_corpus",
        )
    return LegalIntent()


def classify_relation_intent(store, query: str) -> LegalIntent:
    if store is None:
        return LegalIntent()
    intent = intent_config_for(getattr(store.config, "query_strategy", "generic"), store.config)
    relation_config = intent.get("document_relation", {})
    for name, row in (relation_config.get("relation_families") or {}).items():
        explicit_mapping = (
            _explicit_renumbering_mapping(query, getattr(store.config, "corpus_id", "")) if name == "RENAME_PROVISION" else False
        )
        if contains_intent_phrase(query, tuple(row.get("terms") or ())) or explicit_mapping:
            return LegalIntent(
                "amendment_relation",
                _target_reference(store, query),
                relation_type=str(name),
                legal_domain="constitutional_amendment",
                answerability="answerable",
            )
    return LegalIntent()


def _explicit_renumbering_mapping(query: str, corpus_id: str) -> bool:
    folded = (query or "").casefold()
    if "menjadi" not in folded or any(term in folded for term in ("konflik", "anomali", "pasal iii", "sumber anomaly")):
        return False
    return len(parse_legal_references(corpus_id, query)) >= 2 and any(
        term in folded for term in ("pasal", "amandemen", "perubahan", "penomoran", "nomor")
    )


def _target_reference(store, query: str) -> str | None:
    try:
        ref = parse_legal_reference(getattr(store.config, "corpus_id", ""), query)
    except ValueError:
        return None
    pasal = ref.get("pasal")
    ayat = ref.get("ayat")
    return f"{pasal} ayat {ayat}" if pasal and ayat else pasal
