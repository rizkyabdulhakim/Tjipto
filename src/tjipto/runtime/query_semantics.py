"""Immutable, corpus-aware interpretation used before retrieval arbitration."""

from __future__ import annotations

from dataclasses import dataclass
import re

from tjipto.corpora.parser_dispatch import parse_legal_references, resolve_navigation
from tjipto.runtime.intent import classify_relation_intent


@dataclass(frozen=True)
class QuerySemantics:
    requested_function: str
    legal_references: tuple[str, ...]
    requested_proposition: str | None
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


_TEMPORAL_CURRENT = ("setelah perubahan", "sesudah perubahan", "pasca perubahan", "saat ini", "satu naskah")
_PROPOSITION = ("apakah", "mengatur", "melarang", "mewajibkan", "memperbolehkan", "tidak mengatur")
_DISCREPANCY = ("tapi", "tetapi", "kenapa", "mengapa", "berbeda", "beda", "namun")


def interpret_query(
    store,
    corpus_id: str,
    query: str,
    *,
    available_corpora: tuple[str, ...] | None = None,
) -> QuerySemantics:
    config = getattr(store, "config", None)
    folded = str(query or "").casefold()
    references = _references(corpus_id, query)
    temporal = any(term in folded for term in _TEMPORAL_CURRENT)
    navigation = resolve_navigation(corpus_id, query) if references and not temporal else None
    proposition = _proposition(query) if references and any(term in folded for term in _PROPOSITION) else None
    discrepancy = len(references) >= 2 and any(term in folded for term in _DISCREPANCY)
    relation_intent = classify_relation_intent(store, query).relation_type
    source_role = getattr(config, "preferred_source_role", None) if temporal else None
    temporal_context = source_role if temporal else None
    requested_function = (
        "source_discrepancy"
        if discrepancy
        else "amendment_relation"
        if relation_intent
        else "proposition_verification"
        if proposition
        else "temporal_quotation"
        if temporal and references
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


def _proposition(query: str) -> str | None:
    text = re.sub(r"\bpasal\s+\d+[a-z]?(?:\s+ayat\s*\(?\d+\)?)?", "", query or "", flags=re.IGNORECASE)
    text = re.sub(r"\b(apakah|mengatur|melarang|mewajibkan|memperbolehkan|menyebut)\b", " ", text, flags=re.IGNORECASE)
    tokens = re.findall(r"[A-Za-zÀ-ÿ0-9]+", text.casefold())
    return " ".join(tokens) or None
