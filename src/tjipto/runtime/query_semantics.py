"""Immutable, corpus-aware interpretation used before retrieval arbitration."""

from __future__ import annotations

from dataclasses import dataclass
import re

from tjipto.corpora.intent_config import contains_intent_phrase
from tjipto.corpora.parser_dispatch import parse_legal_references, proposition_operator, resolve_navigation
from tjipto.retrieval.metadata import resolve_source_scope
from tjipto.runtime.intent import classify_relation_intent


@dataclass(frozen=True)
class PropositionClaim:
    predicate: str
    subject: str | None
    object: str
    polarity: str
    modality: str
    legal_references: tuple[str, ...]
    source_role: str | None
    temporal_context: str | None
    conditions: tuple[str, ...] = ()
    exceptions: tuple[str, ...] = ()


@dataclass(frozen=True)
class QuerySemantics:
    requested_function: str
    legal_references: tuple[str, ...]
    requested_proposition: PropositionClaim | None
    source_role: str | None
    temporal_context: str | None
    navigation_operation: str | None
    relation_intent: str | None
    discrepancy_intent: bool
    available_corpora: tuple[str, ...]
    needed_corpora: tuple[str, ...]
    missing_corpora: tuple[str, ...]
    answer_permission: str
    reason_code: str | None
    trace: tuple[str, ...]


def interpret_query(
    store,
    corpus_id: str,
    query: str,
    *,
    available_corpora: tuple[str, ...] | None = None,
) -> QuerySemantics:
    config = getattr(store, "config", None)
    references = _references(corpus_id, query)
    scope = resolve_source_scope(query, strategy=getattr(config, "query_strategy", "generic"), config=config)
    temporal = scope.temporal
    source_role = scope.role if scope.temporal or scope.explicit else None
    temporal_context = source_role if temporal else None
    navigation = resolve_navigation(corpus_id, query) if references and not temporal else None
    proposition = _proposition(corpus_id, query, references, source_role, temporal_context)
    discrepancy = len(references) >= 2 and contains_intent_phrase(query, discrepancy_terms(config))
    relation_intent = None if scope.state == "generic_post_amendment" else classify_relation_intent(store, query).relation_type
    requested_function = (
        "source_discrepancy"
        if discrepancy
        else "temporal_quotation"
        if temporal and references and not relation_intent
        else "amendment_relation"
        if relation_intent
        else "proposition_verification"
        if proposition
        else "structural_navigation"
        if navigation
        else "direct_quotation"
        if references
        else "retrieval"
    )
    trace = (
        (f"function:{requested_function}",)
        + ((f"source_role:{source_role}",) if source_role else ())
        + ((f"relation:{relation_intent}",) if relation_intent else ())
    )
    return QuerySemantics(
        requested_function=requested_function,
        legal_references=references,
        requested_proposition=proposition,
        source_role=source_role,
        temporal_context=temporal_context,
        navigation_operation=navigation[1] if navigation else None,
        relation_intent=relation_intent,
        discrepancy_intent=discrepancy,
        available_corpora=available_corpora if available_corpora is not None else ((corpus_id,) if store is not None else ()),
        needed_corpora=(),
        missing_corpora=(),
        answer_permission="verify" if proposition else "quote" if references else "retrieve",
        reason_code=None,
        trace=trace,
    )


def discrepancy_terms(config) -> tuple[str, ...]:
    return tuple(config.setting("source_conflict_intent", {}).get("discrepancy_terms") or ()) if config else ()


def _references(corpus_id: str, query: str) -> tuple[str, ...]:
    try:
        parsed = parse_legal_references(corpus_id, query)
    except ValueError:
        return ()
    labels = []
    for row in parsed:
        reference = row.get("reference")
        if reference:
            labels.append(str(reference))
            continue
        pasal = row.get("pasal")
        ayat = row.get("ayat")
        if not pasal:
            continue
        labels.append(f"{pasal} ayat {ayat}" if ayat else str(pasal))
    return tuple(dict.fromkeys(labels))


def _proposition(
    corpus_id: str,
    query: str,
    references: tuple[str, ...],
    source_role: str | None,
    temporal_context: str | None,
) -> PropositionClaim | None:
    normalized = " ".join(re.findall(r"[a-z0-9]+", str(query or "").casefold()))
    try:
        parsed = proposition_operator(corpus_id, normalized)
    except ValueError:
        parsed = None
    if parsed is None:
        return None
    operator, predicate, modality = parsed
    text = re.sub(r"\bpasal\s+\d+[a-z]?(?:\s+ayat\s*\(?\d+\)?)?", "", normalized, flags=re.IGNORECASE)
    for term in (operator, "apakah", "apa", "isi"):
        text = re.sub(rf"\b{re.escape(term)}\b", " ", text)
    object_tokens = tuple(token for token in text.split() if token)
    if not object_tokens:
        return None
    return PropositionClaim(
        predicate=predicate,
        subject=references[0] if len(references) == 1 else None,
        object=" ".join(object_tokens),
        polarity="negative" if re.search(rf"\btidak\s+{re.escape(operator)}\b", normalized) else "positive",
        modality=modality,
        legal_references=references,
        source_role=source_role,
        temporal_context=temporal_context,
    )
