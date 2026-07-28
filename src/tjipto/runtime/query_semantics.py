"""Immutable, corpus-aware interpretation used before retrieval arbitration."""

from __future__ import annotations

from dataclasses import dataclass
import re

from tjipto.corpora.intent_config import contains_intent_phrase, intent_config_for
from tjipto.corpora.parser_dispatch import parse_legal_references, resolve_navigation
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
    evidence_terms: tuple[str, ...]


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
    intent = intent_config_for(getattr(config, "query_strategy", "generic"), config)
    references = _references(corpus_id, query)
    temporal = contains_intent_phrase(query, intent["temporal_current_terms"])
    source_role = getattr(config, "preferred_source_role", None) if temporal else None
    temporal_context = source_role if temporal else None
    navigation = resolve_navigation(corpus_id, query) if references and not temporal else None
    proposition = _proposition(query, references, source_role, temporal_context, intent["proposition_operators"])
    discrepancy = len(references) >= 2 and contains_intent_phrase(query, discrepancy_terms(config))
    relation_intent = None if temporal else classify_relation_intent(store, query).relation_type
    requested_function = (
        "source_discrepancy"
        if discrepancy
        else "temporal_quotation"
        if temporal and references
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
    query: str,
    references: tuple[str, ...],
    source_role: str | None,
    temporal_context: str | None,
    operators: dict[str, dict],
) -> PropositionClaim | None:
    normalized = " ".join(re.findall(r"[a-z0-9]+", str(query or "").casefold()))
    matches = [(operator, policy) for operator, policy in operators.items() if contains_intent_phrase(normalized, (operator,))]
    if not matches:
        return None
    operator, policy = max(matches, key=lambda item: len(item[0]))
    text = re.sub(r"\bpasal\s+\d+[a-z]?(?:\s+ayat\s*\(?\d+\)?)?", "", normalized, flags=re.IGNORECASE)
    for term in (*operators, "apakah", "apa", "isi", "tidak"):
        text = re.sub(rf"\b{re.escape(term)}\b", " ", text)
    object_tokens = tuple(token for token in text.split() if token)
    if not object_tokens:
        return None
    return PropositionClaim(
        predicate=str(policy.get("predicate") or operator),
        subject=references[0] if len(references) == 1 else None,
        object=" ".join(object_tokens),
        polarity="negative" if contains_intent_phrase(normalized, ("tidak",)) else "positive",
        modality=str(policy.get("modality") or "statement"),
        legal_references=references,
        source_role=source_role,
        temporal_context=temporal_context,
        evidence_terms=tuple(str(term) for term in policy.get("evidence_terms") or ()),
    )
