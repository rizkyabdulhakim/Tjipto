"""Immutable, corpus-aware interpretation used before retrieval arbitration."""

from __future__ import annotations

from dataclasses import dataclass
import re

from tjipto.corpora.capabilities import CapabilityDecision, resolve_capability
from tjipto.corpora.intent_config import contains_intent_phrase, intent_config_for
from tjipto.corpora.parser_dispatch import parse_legal_reference, parse_legal_references, proposition_operator, resolve_navigation
from tjipto.corpora.source_arbitration import resolve_source_scope
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
    operation: str
    operation_query: str | None
    targets: tuple[str, ...]
    source_scopes: tuple[str, ...]
    source_scope_state: str
    temporal_scope: str | None
    output_mode: str
    requires_multiple_supports: bool
    requires_comparison: bool
    requires_decomposition: bool
    requires_graph: bool
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
    capability_decision: CapabilityDecision
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
    references = _references(corpus_id, query, config=config)
    scope = resolve_source_scope(query, strategy=getattr(config, "query_strategy", "generic"), config=config)
    temporal = scope.temporal
    named_roles = scope.roles
    # Multiple explicitly named instruments are a comparison scope, not a
    # single-source filter.  Requirement generation owns the per-role split.
    source_role = scope.role if len(named_roles) <= 1 and (scope.temporal or scope.explicit) else None
    temporal_context = source_role if temporal else None
    # Navigation can be anchored to a division (for example, a BAB) as well as
    # a provision.  The corpus parser is the owner of both forms.
    navigation = _navigation(corpus_id, query, temporal=temporal, config=config)
    proposition = _proposition(corpus_id, query, references, source_role, temporal_context, config=config)
    discrepancy = len(references) >= 2 and contains_intent_phrase(query, discrepancy_terms(config))
    relation_intent = None if scope.state == "generic_post_amendment" else classify_relation_intent(store, query).relation_type
    temporal_scope = "historical_pre_change" if _historical_pre_change(query) else temporal_context
    operation = _operation(query, config, references, named_roles, navigation, relation_intent, proposition, temporal_scope)
    operation_query = _operation_query(query, config, operation, scope)
    navigation_anchor = _navigation_anchor(query) if navigation else None
    comparison = operation == "compare"
    requires_multiple_supports = comparison or temporal_scope == "historical_pre_change" or operation == "analyze"
    requires_decomposition = comparison or operation == "analyze"
    requires_graph = operation in {"navigate", "trace"} or temporal_scope == "historical_pre_change"
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
    if operation == "analyze":
        requested_function = "retrieval"
    strategy = getattr(config, "strategy", None)
    resolver = getattr(strategy, "capability_resolver", None) or resolve_capability
    capability = resolver(config, query, requested_function, available_corpora or ((corpus_id,) if store is not None else ()))
    trace = (
        (f"function:{requested_function}",)
        + ((f"source_role:{source_role}",) if source_role else ())
        + ((f"relation:{relation_intent}",) if relation_intent else ())
    )
    return QuerySemantics(
        requested_function=requested_function,
        operation=operation,
        operation_query=operation_query,
        targets=_targets(references, query, config, navigation_anchor=navigation_anchor),
        source_scopes=named_roles,
        source_scope_state=scope.state,
        temporal_scope=temporal_scope,
        output_mode=_output_mode(operation),
        requires_multiple_supports=requires_multiple_supports,
        requires_comparison=comparison,
        requires_decomposition=requires_decomposition,
        requires_graph=requires_graph,
        legal_references=references,
        requested_proposition=proposition,
        source_role=source_role,
        temporal_context=temporal_context,
        navigation_operation=("next" if navigation and navigation[1] == "direct" else navigation[1]) if navigation else None,
        relation_intent=relation_intent,
        discrepancy_intent=discrepancy,
        available_corpora=available_corpora if available_corpora is not None else ((corpus_id,) if store is not None else ()),
        needed_corpora=capability.missing_corpora,
        missing_corpora=capability.missing_corpora,
        capability_decision=capability,
        answer_permission="verify" if proposition else "analyze" if operation == "analyze" else "quote" if references else "retrieve",
        reason_code=capability.reason_code,
        trace=trace,
    )


def _operation(
    query: str,
    config,
    references: tuple[str, ...],
    source_scopes: tuple[str, ...],
    navigation: tuple[str, str] | None,
    relation_intent: str | None,
    proposition: PropositionClaim | None,
    temporal_scope: str | None,
) -> str:
    normalized = " ".join(re.findall(r"[a-z0-9]+", str(query or "").casefold()))
    if navigation:
        return "navigate"
    setting = getattr(config, "setting", None)
    summary = setting("document_summary", {}) if callable(setting) else {}
    summary_terms = summary.get("query_terms", ()) if isinstance(summary, dict) else ()
    if contains_intent_phrase(query, summary_terms):
        return "summarize"
    open_terms = setting("document_open_terms", ()) if callable(setting) else ()
    if contains_intent_phrase(query, open_terms or ()):
        return "open_document"
    if len(source_scopes) > 1 and re.search(r"\b(vs|versus|perbandingan|perbedaan|bandingkan|beda)\b", normalized):
        return "compare"
    if re.search(r"\b(analisis|analisa|legal opinion|pendapat hukum)\b", normalized):
        return "analyze"
    if relation_intent:
        return "quote_or_explain" if temporal_scope == "historical_pre_change" else "trace"
    if proposition or references:
        return "quote_or_explain"
    return "search"


def _operation_query(query: str, config, operation: str, scope) -> str | None:
    """Resolve an operation's canonical follow-up query once at the control boundary."""
    if operation != "summarize":
        return None
    setting = getattr(config, "setting", None)
    policy = setting("document_summary", {}) if callable(setting) else {}
    if not isinstance(policy, dict):
        return None
    role_queries = policy.get("source_role_queries", {}) or {}
    if scope.explicit:
        normalized = role_queries.get(scope.role)
        return str(normalized) if normalized else None
    if not contains_intent_phrase(query, policy.get("document_terms", ())):
        return None
    normalized = policy.get("default_query")
    return str(normalized) if normalized else None


def _targets(references: tuple[str, ...], query: str, config, *, navigation_anchor: str | None = None) -> tuple[str, ...]:
    intent = intent_config_for(getattr(config, "query_strategy", "generic"), config)
    metadata_fields = intent.get("metadata_fields", {}) if isinstance(intent, dict) else {}
    normalized = " ".join(re.findall(r"[a-z0-9]+", str(query or "").casefold()))
    metadata = next(
        (
            field
            for field, terms in metadata_fields.items()
            if any(str(term).casefold() in normalized for term in terms)
        ),
        None,
    )
    target = "signatory_metadata" if metadata == "signatories" else metadata
    return tuple(dict.fromkeys((*references, *((target,) if target else ()), *((navigation_anchor,) if navigation_anchor else ()))))


def _output_mode(operation: str) -> str:
    return {
        "open_document": "document",
        "summarize": "summary",
        "compare": "comparison",
        "analyze": "analysis",
        "navigate": "navigation",
        "trace": "trace",
    }.get(operation, "evidence")


def _historical_pre_change(query: str) -> bool:
    normalized = " ".join(re.findall(r"[a-z0-9]+", str(query or "").casefold()))
    return bool(re.search(r"\b(sebelum|pra)\b.*\b(dihapus|hapus|dicabut)\b", normalized))


def _navigation_anchor(query: str) -> str | None:
    reference = r"(?:BAB\s+[IVXLCDM]+[A-Z]?|Pasal\s+\d+[A-Z]?)"
    patterns = (
        rf"\b({reference})\s+(?:setelah|sesudah|sebelum|sebelumnya)\b",
        rf"\b(?:setelah|sesudah|sebelum|sebelumnya)\s+({reference})\b",
    )
    for pattern in patterns:
        match = re.search(pattern, query or "", re.IGNORECASE)
        if match:
            return " ".join(match.group(1).split())
    return None


def _navigation(corpus_id: str, query: str, *, temporal: bool, config) -> tuple[str, str] | None:
    if temporal:
        return None
    try:
        return resolve_navigation(corpus_id, query, config=config)
    except ValueError:
        return None


def discrepancy_terms(config) -> tuple[str, ...]:
    return tuple(config.setting("source_conflict_intent", {}).get("discrepancy_terms") or ()) if config else ()


def _references(corpus_id: str, query: str, *, config=None) -> tuple[str, ...]:
    try:
        parsed = parse_legal_references(corpus_id, query, config=config)
    except ValueError:
        return ()
    if not parsed:
        try:
            single = parse_legal_reference(corpus_id, query, allow_roman_pasal=True, config=config)
        except ValueError:
            single = {}
        if any(single.values()):
            parsed = (single,)
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
    *,
    config=None,
) -> PropositionClaim | None:
    normalized = " ".join(re.findall(r"[a-z0-9]+", str(query or "").casefold()))
    try:
        strategy = getattr(config, "strategy", None)
        parsed = strategy.proposition_operator(normalized) if strategy is not None else proposition_operator(corpus_id, normalized)
    except ValueError:
        parsed = None
    if parsed is None:
        return None
    operator, predicate, modality = parsed
    text = re.sub(r"\bpasal\s+\d+[a-z]?(?:\s+ayat\s*\(?\d+\)?)?", "", normalized, flags=re.IGNORECASE)
    negative = bool(re.search(rf"\btidak\s+{re.escape(operator)}\b", normalized))
    for term in (operator, "apakah", "apa", "isi", "tidak" if negative else ""):
        text = re.sub(rf"\b{re.escape(term)}\b", " ", text)
    object_tokens = tuple(token for token in text.split() if token)
    if not object_tokens:
        return None
    return PropositionClaim(
        predicate=predicate,
        # A legal reference scopes retrieval; it is not the grammatical
        # subject of the proposition asserted by the source clause.
        subject=None,
        object=" ".join(object_tokens),
        polarity="negative" if negative else "positive",
        modality=modality,
        legal_references=references,
        source_role=source_role,
        temporal_context=temporal_context,
    )
