"""Immutable, corpus-aware interpretation used before retrieval arbitration."""

from __future__ import annotations

from dataclasses import dataclass
import re

from tjipto.corpora.capabilities import CapabilityDecision, resolve_capability
from tjipto.corpora.intent_config import contains_intent_phrase, intent_config_for
from tjipto.corpora.parser_dispatch import (
    normalize_metadata_intent,
    parse_legal_reference,
    parse_legal_references,
    proposition_operator,
    resolve_navigation,
)
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
    intent = intent_config_for(getattr(config, "query_strategy", "generic"), config)
    if scope.state == "generic_post_amendment" and scope.role:
        named_roles = (scope.role,)
    if contains_intent_phrase(query, intent.get("all_source_scope_terms", ())):
        named_roles = tuple(getattr(config, "source_roles", ()))
    preferred_role = getattr(config, "preferred_source_role", None)
    if (
        named_roles
        and preferred_role
        and preferred_role not in named_roles
        and contains_intent_phrase(query, intent.get("temporal_current_terms", ()))
    ):
        named_roles = (*named_roles, preferred_role)
    before_after = re.search(
        r"\bsebelum\b.*\b(?:sesudah|setelah)\b|\b(?:sesudah|setelah)\b.*\bsebelum\b",
        query or "",
        re.IGNORECASE,
    )
    if before_after:
        explicit_change_role = next(
            (role for role in named_roles if role not in {preferred_role, "original_historical"}),
            None,
        )
        comparison_role = explicit_change_role or (named_roles[0] if len(named_roles) == 1 else preferred_role)
        predecessor = intent.get("source_role_predecessors", {}).get(comparison_role)
        if predecessor:
            named_roles = (str(predecessor), str(comparison_role))
    # Multiple explicitly named instruments are a comparison scope, not a
    # single-source filter.  Requirement generation owns the per-role split.
    source_role = scope.role if len(named_roles) <= 1 and (scope.temporal or scope.explicit) else None
    temporal_context = source_role if temporal else None
    # Navigation can be anchored to a division (for example, a BAB) as well as
    # a provision.  The corpus parser is the owner of both forms.
    navigation = _navigation(corpus_id, query, temporal=temporal, config=config)
    proposition = _proposition(corpus_id, query, references, source_role, temporal_context, config=config)
    discrepancy = len(references) >= 2 and contains_intent_phrase(query, discrepancy_terms(config))
    metadata_target = _metadata_target_requested(query, config, references)
    relation_intent = (
        None
        if scope.state == "generic_post_amendment" or metadata_target
        else classify_relation_intent(store, query).relation_type
    )
    temporal_scope = "historical_pre_change" if _historical_pre_change(query) else temporal_context
    operation = _operation(query, config, references, named_roles, navigation, relation_intent, proposition, temporal_scope)
    if operation == "analyze" and source_role is None and not named_roles:
        source_role = getattr(config, "preferred_source_role", None)
        temporal_context = source_role
    operation_query = _operation_query(query, config, operation, scope)
    navigation_anchor = _navigation_anchor(query) if navigation else None
    comparison = operation == "compare"
    multiple_scopes = len(named_roles) > 1
    requires_multiple_supports = comparison or multiple_scopes or temporal_scope == "historical_pre_change" or operation == "analyze"
    requires_decomposition = comparison or multiple_scopes or operation == "analyze"
    requires_graph = operation in {"navigate", "trace"} or temporal_scope == "historical_pre_change"
    # An explicit source-occurrence request ("Pasal 28 ada di naskah mana
    # saja") is a retrieval problem over every named source, even when the
    # parser also finds a legal reference.  Keep direct citation ownership for
    # the ordinary single-source form.
    source_occurrence = operation == "search" and contains_intent_phrase(
        query, intent.get("all_source_scope_terms", ())
    )
    requested_function = (
        "source_discrepancy"
        if discrepancy
        else "temporal_quotation"
        if temporal and references and not relation_intent
        else "amendment_relation"
        if relation_intent
        else "retrieval"
        if source_occurrence
        else "retrieval"
        if operation == "trace"
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
        targets=_targets(corpus_id, references, query, config, navigation_anchor=navigation_anchor),
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
    if _document_operation_requested(
        query,
        config,
        references=references,
        source_scopes=source_scopes,
        relation_intent=relation_intent,
        open_terms=open_terms or (),
    ):
        return "open_document"
    comparison = re.search(r"\b(vs|versus|perbandingan|perbedaan|bandingkan|beda)\b", normalized)
    if not comparison and len(source_scopes) > 1:
        comparison = re.search(r"\bsebelum\b.*\b(?:sesudah|setelah)\b|\b(?:sesudah|setelah)\b.*\bsebelum\b", normalized)
    if comparison:
        return "compare"
    intent = intent_config_for(getattr(config, "query_strategy", "generic"), config)
    if contains_intent_phrase(query, intent.get("all_source_scope_terms", ())):
        return "search"
    trace_terms = tuple(
        str(term)
        for term in intent.get("source_occurrence_query_wrappers", ())
        if isinstance(term, str) and term.casefold().strip() == "telusuri"
    )
    if len(source_scopes) > 1 and trace_terms and contains_intent_phrase(query, trace_terms):
        return "trace"
    # A configured issue phrase with at least two matching terms is an
    # analysis request even when the user omits the words "legal opinion".
    # The threshold keeps a single broad concept (for example, "kebebasan
    # beragama") on the ordinary lexical path instead of applying the speech
    # issue policy to an unrelated topic.
    research = setting("research", {}) if callable(setting) else {}
    analysis_policy = research.get("operation_requirements", {}).get("analyze", {}) if isinstance(research, dict) else {}
    issue_terms = analysis_policy.get("issue_reference_terms", ()) if isinstance(analysis_policy, dict) else ()
    if (
        len(
            tuple(
                term
                for term in issue_terms
                if isinstance(term, str) and contains_intent_phrase(query, (term,))
            )
        ) >= 2
    ):
        return "analyze"
    if len(references) > 1 and contains_intent_phrase(query, intent.get("relation_words", ())):
        return "compare"
    if re.search(r"\b(analisis|analisa|legal opinion|legal research|riset hukum|penelitian hukum|kajian hukum|pendapat hukum)\b", normalized):
        return "analyze"
    if relation_intent:
        return "quote_or_explain" if temporal_scope == "historical_pre_change" else "trace"
    if proposition or references:
        return "quote_or_explain"
    return "search"


def _document_operation_requested(
    query: str,
    config,
    *,
    references: tuple[str, ...],
    source_scopes: tuple[str, ...],
    relation_intent: str | None,
    open_terms: tuple[str, ...],
) -> bool:
    if references or relation_intent:
        return False
    if contains_intent_phrase(query, open_terms):
        return True
    setting = getattr(config, "setting", None)
    if not callable(setting):
        return False
    intent = intent_config_for(getattr(config, "query_strategy", "generic"), config)
    relation_policy = intent.get("document_relation", {}) or {}
    normalized = " ".join(re.findall(r"[a-z0-9]+", str(query or "").casefold()))
    if (
        re.search(r"\b(vs|versus|perbandingan|perbedaan|bandingkan|beda)\b", normalized)
        or re.search(r"\b(analisis|analisa|legal opinion|legal research|riset hukum|penelitian hukum|kajian hukum|pendapat hukum)\b", normalized)
        or contains_intent_phrase(query, intent.get("all_source_scope_terms", ()))
        or contains_intent_phrase(query, intent.get("metadata_candidate_signals", ()))
        or contains_intent_phrase(query, intent.get("relation_words", ()))
        or contains_intent_phrase(query, intent.get("direct_relation_words", ()))
        or contains_intent_phrase(query, relation_policy.get("add_terms", ()))
        or contains_intent_phrase(query, intent.get("instrument_content_signals", ()))
        or contains_intent_phrase(query, intent.get("instrument_effect_signals", ()))
        or contains_intent_phrase(query, intent.get("instrument_legal_object_signals", ()))
    ):
        return False
    target_words = tuple(intent.get("document_target_words", ()) or ())
    has_document_target = contains_intent_phrase(query, target_words)
    if not has_document_target:
        return False
    action_tokens = {
        str(term).casefold().split()[0]
        for term in open_terms
        if isinstance(term, str) and term.strip()
    }
    if len(source_scopes) == 1 and normalized.split() and normalized.split()[0] in {"naskah", "dokumen"}:
        return True
    if len(source_scopes) > 1 and normalized.split() and normalized.split()[0] in {"naskah", "dokumen"}:
        if re.search(r"\bsebelum\b.*\b(?:sesudah|setelah)\b|\b(?:sesudah|setelah)\b.*\bsebelum\b", normalized):
            return True
    return bool(normalized.split() and normalized.split()[0] in action_tokens)


def _metadata_target_requested(query: str, config, references: tuple[str, ...]) -> bool:
    """Treat explicit metadata questions as metadata retrieval, not relations."""
    if references:
        return False
    intent = intent_config_for(getattr(config, "query_strategy", "generic"), config)
    relation_policy = intent.get("document_relation", {}) if isinstance(intent, dict) else {}
    relation_terms = (
        tuple(intent.get("change_terms", ()) or ())
        + tuple(relation_policy.get("add_terms", ()) or ())
        + tuple(
            term
            for family in (relation_policy.get("relation_families", {}) or {}).values()
            if isinstance(family, dict)
            for term in (family.get("terms", ()) or ())
        )
    )
    # A change operation owns the query even when its target happens to be a
    # metadata word such as ``lembaga``.  This keeps relation policy ahead of
    # field lookup without adding a second vocabulary.
    if contains_intent_phrase(query, relation_terms):
        return False
    fields = intent.get("metadata_fields", {}) if isinstance(intent, dict) else {}
    return any(contains_intent_phrase(query, terms or ()) for terms in fields.values())


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


def _targets(
    corpus_id: str,
    references: tuple[str, ...],
    query: str,
    config,
    *,
    navigation_anchor: str | None = None,
) -> tuple[str, ...]:
    intent = intent_config_for(getattr(config, "query_strategy", "generic"), config)
    metadata_fields = intent.get("metadata_fields", {}) if isinstance(intent, dict) else {}
    normalized = normalize_metadata_intent(corpus_id, query, config=config)
    metadata = next(
        (
            field
            for field, terms in metadata_fields.items()
            if any(normalize_metadata_intent(corpus_id, str(term), config=config) in normalized for term in terms)
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
    shorthand = re.search(
        r"\b(?:pasal|ketentuan)\s+(?:setelah|sesudah|sebelum|sebelumnya)\s+(\d+[a-z]?)\b",
        query or "",
        re.IGNORECASE,
    )
    if shorthand:
        return f"Pasal {shorthand.group(1).upper()}"
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
