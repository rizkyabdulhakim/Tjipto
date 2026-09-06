from __future__ import annotations

from collections.abc import Callable
import re
from typing import Any

from tjipto.corpora.intent_config import contains_intent_phrase, intent_config_for, normalize_intent_text, resolve_instrument_intent
from tjipto.evidence.store import EvidenceStore
from tjipto.retrieval.answer import validate_answer_candidate
from tjipto.retrieval.bm25 import lexical_aliases, meaningful_tokens
from tjipto.retrieval.research import ResearchPlan
from tjipto.retrieval.metadata import has_metadata_target, metadata_lookup
from tjipto.retrieval.relations import has_relation_target
from tjipto.corpora.source_arbitration import resolve_source_scope
from tjipto.retrieval.requirements import semantic_supports_text
from tjipto.runtime.viewer import _source_status_label

ANSWER_TEMPLATES = {
    "insufficient": "Bukti tidak cukup atau database belum tersedia dalam korpus terverifikasi saat ini.",
    "legal_relation": "Dukungan relasi hukum berbasis bukti tersedia; sistem tidak menghasilkan kesimpulan hukum.",
    "citation": "Dukungan sitasi berbasis bukti tersedia untuk {citation}; sistem tidak menghasilkan kesimpulan hukum.",
}


def document_open_requested(query: str, *, config: object) -> bool:
    """Recognize only an adapter-declared document navigation operation."""
    terms = getattr(config, "setting", lambda *_: ())("document_open_terms", ()) or ()
    return contains_intent_phrase(query, terms)


def document_summary_query(query: str, *, strategy: str, config: object, semantics=None) -> str | None:
    """Normalize a summary operation into one adapter-owned retrieval query."""
    if semantics is not None:
        return getattr(semantics, "operation_query", None) if getattr(semantics, "operation", None) == "summarize" else None
    policy: dict = getattr(config, "setting", lambda *_: {})("document_summary", {}) or {}
    if not contains_intent_phrase(query, policy.get("query_terms", ())):
        return None
    scope = resolve_source_scope(query, strategy=strategy, config=config)
    role_queries = policy.get("source_role_queries", {}) or {}
    if scope.explicit:
        # A multi-document request is decomposed by the runtime.  Selecting
        # the first role here would silently discard every other requested
        # source.
        if len(scope.roles) > 1:
            return None
        normalized = role_queries.get(scope.role)
        return str(normalized) if normalized else None
    if not contains_intent_phrase(query, policy.get("document_terms", ())):
        return None
    normalized = policy.get("default_query")
    return str(normalized) if normalized else None


def compound_query_parts(query: str, *, semantics, config: object) -> tuple[str, ...]:
    """Split only an explicit mixed quote-and-summary request.

    The corpus parser already owns legal references and source scopes.  This
    small adapter preserves each explicit target instead of letting one summary
    normalization consume the rest of the request.
    """
    source_scopes = tuple(getattr(semantics, "source_scopes", ()) or ())
    references = tuple(getattr(semantics, "legal_references", ()) or ())
    alternatives = " atau " in f" {normalize_intent_text(query)} "
    quotes = (
        tuple(f"berikan {reference}" for reference in references)
        if len(references) > 1 and getattr(semantics, "operation", None) in {"quote_or_explain", "summarize"} and not alternatives
        else ()
    )
    summaries: tuple[str, ...] = ()
    if getattr(semantics, "operation", None) == "summarize" and len(source_scopes) > 1:
        policy: dict = getattr(config, "setting", lambda *_: {})("document_summary", {}) or {}
        role_queries = policy.get("source_role_queries", {}) if isinstance(policy, dict) else {}
        summaries = tuple(
            f"ringkas {normalized}"
            for role in source_scopes
            if isinstance(normalized := role_queries.get(role), str) and normalized.strip()
        )
        if len(summaries) < 2:
            summaries = ()
    if not quotes and not summaries:
        return ()
    return tuple(dict.fromkeys((*quotes, *summaries)))


def instrument_intent_context(store: Any, query: str) -> tuple[dict | None, str, str] | None:
    """Resolve an adapter-owned instrument lookup before general retrieval."""
    config = getattr(store, "config", None)
    intent = intent_config_for(getattr(config, "structured_strategy", "generic"), config)
    decision = resolve_instrument_intent(query, intent, corpus=getattr(config, "corpus_id", ""))
    if metadata_lookup(store, query, 1) and decision.reason not in {
        "analysis_metadata_conflict",
        "unsupported_analysis_intent",
    }:
        return None
    if decision.target_status == "not_instrument":
        return None
    if decision.target_status == "instrument_unresolved":
        return None, "instrument_unresolved", decision.reason
    row = next(
        (
            item
            for item in store.evidence
            if item.get("source_role") == decision.amendment and item.get("citation") == decision.target_citation
        ),
        None,
    )
    if row is None:
        return None, "instrument_unresolved", decision.reason
    row = row | {
        "route_sources": ("structured",),
        "candidate_type": f"instrument_{decision.role_family}_candidate",
    }
    if validate_answer_candidate(store, row)[0]:
        return row, "instrument_resolved_answerable", "answer_evidence"
    return (
        row | {"forced_rejection_reason": "instrument_resolved_fail_closed"},
        "instrument_resolved_fail_closed",
        "instrument_resolved_fail_closed",
    )


def source_document_response(
    store: Any,
    corpus_id: str,
    query: str,
    *,
    has_resolved_target: bool,
    document_title: Callable[[object, dict], str],
    insufficient_answer: str,
    semantics=None,
) -> dict | None:
    """Resolve only an explicit document-open operation to a verified source."""
    config = getattr(store, "config", None)
    strategy = getattr(config, "query_strategy", "generic")
    intent = intent_config_for(strategy, config)
    if semantics is None:
        if not document_open_requested(query, config=config):
            return None
        scope = resolve_source_scope(query, strategy=strategy, config=config)
        scope_explicit = scope.explicit
        scope_role = scope.role
        source_scopes = tuple(scope.roles or ())
    else:
        if getattr(semantics, "operation", None) != "open_document":
            return None
        source_scopes = tuple(getattr(semantics, "source_scopes", ()) or ())
        scope_explicit = bool(source_scopes)
        scope_role = getattr(semantics, "source_role", None) or (source_scopes[0] if source_scopes else None)
    if has_resolved_target:
        return None
    if semantics is None and has_metadata_target(query, strategy=strategy, config=config, store=store):
        return None
    if has_relation_target(query, strategy=strategy, config=config):
        return None
    if contains_intent_phrase(query, intent.get("instrument_analysis_signals", ())) or contains_intent_phrase(
        query, intent.get("instrument_effect_signals", ())
    ):
        return None
    if not scope_explicit:
        documents = tuple(source_document_payload(store, source, document_title) for source in store.source_documents)
        return {
            "status": "answer_ready",
            "route": "source_document_collection",
            "intent": "document_delivery",
            "corpus_id": corpus_id,
            "original_query": query,
            "normalized_query": query.strip(),
            "reason": None,
            "answer_type": "source_document_collection",
            "answer": "Naskah sumber terverifikasi tersedia.",
            "document_source": None,
            "document_sources": documents,
            "citations": (),
            "final_citations": (),
            "historical_citations": (),
            "metadata_support": (),
            "structural_support": (),
            "trace_support": (),
            "viewer_refs": (),
            "metadata_facts": (),
            "evidence": (),
            "warnings": ("document_sources_have_no_legal_citation",),
            "insufficient_reasons": (),
        }
    if len(source_scopes) > 1:
        requested = set(source_scopes)
        sources = tuple(
            source_document_payload(store, source, document_title)
            for source in store.source_documents
            if source.get("source_role") in requested
        )
        if not sources:
            return None
        return {
            "status": "answer_ready",
            "route": "source_document_collection",
            "intent": "document_delivery",
            "corpus_id": corpus_id,
            "original_query": query,
            "normalized_query": query.strip(),
            "reason": None,
            "answer_type": "source_document_collection",
            "answer": "Naskah sumber terverifikasi tersedia.",
            "document_source": None,
            "document_sources": sources,
            "citations": (),
            "final_citations": (),
            "historical_citations": (),
            "metadata_support": (),
            "structural_support": (),
            "trace_support": (),
            "viewer_refs": (),
            "metadata_facts": (),
            "evidence": (),
            "warnings": ("document_sources_have_no_legal_citation",),
            "insufficient_reasons": (),
        }
    source = next((row for row in store.source_documents if row.get("source_role") == scope_role), None)
    if source is None:
        reason = "source_document_not_found"
        return {
            "status": "insufficient_evidence",
            "route": "source_document",
            "intent": "document_delivery",
            "corpus_id": corpus_id,
            "original_query": query,
            "normalized_query": query.strip(),
            "reason": reason,
            "answer_type": "none",
            "answer": insufficient_answer,
            "document_source": None,
            "citations": (),
            "final_citations": (),
            "historical_citations": (),
            "metadata_support": (),
            "structural_support": (),
            "trace_support": (),
            "viewer_refs": (),
            "metadata_facts": (),
            "evidence": (),
            "warnings": (),
            "insufficient_reasons": (reason,),
        }
    document_source = source_document_payload(store, source, document_title)
    title = document_source["document_title"]
    return {
        "status": "answer_ready",
        "route": "source_document",
        "intent": "document_delivery",
        "corpus_id": corpus_id,
        "original_query": query,
        "normalized_query": query.strip(),
        "reason": None,
        "answer_type": "source_document",
        "answer": f"Naskah sumber terverifikasi: {title}.",
        "document_source": document_source,
        "citations": (),
        "final_citations": (),
        "historical_citations": (),
        "metadata_support": (),
        "structural_support": (),
        "trace_support": (),
        "viewer_refs": (),
        "metadata_facts": (),
        "evidence": (),
        "warnings": ("document_source_has_no_legal_citation",),
        "insufficient_reasons": (),
    }


def source_document_payload(store: Any, source: dict, document_title: Callable[[object, dict], str]) -> dict:
    return {
        "source_document_id": source.get("source_document_id"),
        "source_role": source.get("source_role"),
        "temporal_context": source.get("temporal_context"),
        "document_title": document_title(store, source),
        "intent": "document_delivery",
        "viewer_target": {
            "action": "open_document",
            "source_document_id": source.get("source_document_id"),
        },
    }


def _compact_text(value: object) -> str:
    return " ".join(str(value or "").split())


def _legal_reference_label(store, row: dict) -> str:
    """Render a unit label using corpus-declared structural labels."""
    citation = str(row.get("citation") or "").strip()
    hierarchy = tuple(str(value).strip() for value in row.get("hierarchy") or () if str(value).strip())
    unit: dict = next(
        (item for item in store.legal_units if item.get("legal_unit_id") == row.get("legal_unit_id")),
        {},
    )
    if unit.get("structural_role") == "subprovision" and len(hierarchy) >= 2:
        intent = intent_config_for(getattr(store.config, "structured_strategy", "generic"), store.config)
        rules = intent.get("structure_count_units", {}) or {}
        label = next(
            (
                str(rule.get("label") or name)
                for name, rule in rules.items()
                if isinstance(rule, dict) and rule.get("unit_type") == unit.get("unit_type")
            ),
            "",
        )
        return " ".join(value for value in (hierarchy[-2], label, hierarchy[-1]) if value)
    return citation or (hierarchy[-1] if hierarchy else "")


def _restore_corpus_labels(answer: str, evidence: tuple[dict, ...]) -> str:
    """Keep corpus-provided legal labels stable after natural-language rewriting."""
    labels = {
        str(value).strip()
        for row in evidence
        for value in (row.get("citation"), *(row.get("hierarchy") or ()))
        if isinstance(value, str) and value.strip()
    }
    rendered = answer
    for label in sorted(labels, key=len, reverse=True):
        rendered = re.sub(rf"(?<!\w){re.escape(label)}(?!\w)", label, rendered, flags=re.IGNORECASE)
    return rendered


def _wording_preserves_evidence(answer: str, evidence: tuple[dict, ...], claims: object) -> bool:
    """Reject a fluent rewrite that drops the source's verified proposition."""
    if claims:
        # Claim answers carry their own deterministic support wording; do not
        # let a provider replace that verdict with an unsupported paraphrase.
        return False
    folded_answer = normalize_intent_text(answer)
    for row in evidence:
        quote = _compact_text(row.get("quoted_text") or row.get("display_text") or "")
        citation = normalize_intent_text(str(row.get("citation") or ""))
        if citation and quote.casefold().startswith(citation.casefold()):
            quote = quote[len(str(row.get("citation") or "")) :].lstrip(" :.-")
        anchor = re.split(r"[.!?]", quote, maxsplit=1)[0].strip()
        if anchor and normalize_intent_text(anchor) not in folded_answer:
            return False
    return True


def _lexical_fallback_is_limited(
    store,
    query: str,
    evidence: tuple[dict, ...],
    semantic_plan: ResearchPlan | None,
    route: str,
    operation: str | None = None,
) -> bool:
    """Keep lexical-only or numerically unsupported claims explicitly limited."""
    if route != "lexical_fallback":
        return False
    if semantic_plan is None:
        # Summaries are grounded by the document-summary owner after lexical
        # candidate routing; the absence of a planner round is intentional.
        return operation != "summarize"
    plan_intent = semantic_plan.intent
    provider_complete = bool(
        semantic_plan.provider_status == "accepted"
        and semantic_plan.requirements
        and not semantic_plan.rejection_reasons
    )
    aliases = lexical_aliases(store.config)
    normalized_query = normalize_intent_text(query)
    numeric_values = {
        str(value) for key, value in aliases.items() if str(value).isdigit() and re.search(rf"\b{re.escape(key)}\b", normalized_query)
    }
    numeric_values.update(re.findall(r"\b\d+\b", normalized_query))
    if not numeric_values:
        return not provider_complete and not any(
            (
                plan_intent.multiple_supports,
                plan_intent.comparison,
                plan_intent.decomposition,
                plan_intent.relation_traversal,
            )
        )
    supported_numbers = set(
        re.findall(
            r"\b\d+\b",
            normalize_intent_text(" ".join(str(row.get("quoted_text") or "") for row in evidence)),
        )
    )
    return not numeric_values <= supported_numbers


def _structure_outline_item(store, row: dict) -> str:
    citation = str(row.get("citation") or row.get("label") or "").strip()
    if not citation:
        return ""
    lines = tuple(
        " ".join(line.split()).replace("\ufffd", "\u2014") for line in str(row.get("quoted_text") or "").splitlines() if line.strip()
    )
    try:
        start = lines.index(citation) + 1
    except ValueError:
        start = 0
    labels = {str(unit.get("unit_label") or "").strip().casefold() for unit in store.legal_units}
    title = []
    for line in lines[start:]:
        if line.casefold() in labels:
            break
        title.append(line)
    return f"{citation} \u2014 {' '.join(title)}" if title else citation


def _document_summary_answer(store, rows: tuple[dict, ...]) -> str:
    """Describe verified document coverage without inventing legal conclusions."""
    units = {str(unit.get("legal_unit_id")): unit for unit in store.legal_units}
    intent = intent_config_for(getattr(store.config, "structured_strategy", "generic"), store.config)
    outline_type = intent.get("structure_unit_type")
    outline_label = next(
        (
            str(rule.get("label") or name)
            for name, rule in (intent.get("structure_count_units", {}) or {}).items()
            if isinstance(rule, dict) and rule.get("unit_type") == outline_type
        ),
        "bagian",
    )
    grouped: dict[str, list[str]] = {}
    for row in rows:
        role = str(row.get("source_role") or "")
        citation = _legal_reference_label(store, row) or str(row.get("label") or "").strip()
        text = _compact_text(row.get("quoted_text") or row.get("display_text"))
        if citation and text.casefold().startswith(citation.casefold()):
            text = text[len(citation) :].lstrip(" :.-")
        sentence = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0].strip()
        unit = units.get(str(row.get("legal_unit_id")), {})
        unit_type = unit.get("unit_type")
        structural_role = unit.get("structural_role")
        if structural_role in {"document", "division"}:
            detail = _document_summary_outline_item(store, row, unit_type, structural_role)
        else:
            detail = f"{citation}: {sentence[:240]}" if citation and sentence else citation
        if detail:
            grouped.setdefault(role, []).append(detail)
    sections = []
    for role, details in grouped.items():
        source = next((row for row in rows if row.get("source_role") == role), {"source_role": role})
        title = _document_title(store, source)
        unique_details = tuple(dict.fromkeys(details))
        outline_details = tuple(detail for detail in unique_details if detail.casefold().startswith(f"{outline_label.casefold()} "))
        document_labels = tuple(
            dict.fromkeys(
                str(row.get("citation") or "").strip()
                for row in rows
                if row.get("source_role") == role
                and units.get(str(row.get("legal_unit_id")), {}).get("structural_role") == "document"
                and str(row.get("citation") or "").strip()
            )
        )
        if outline_details:
            coverage = " dan ".join((*document_labels, f"{len(outline_details)} {outline_label}"))
            sections.append(
                f"{title} memuat {coverage}. Pokok ketentuan yang terverifikasi: {'; '.join(unique_details)}."
            )
        else:
            sections.append(f"{title}: {'; '.join(unique_details)}.")
    return "\n\n".join(sections)


def _document_summary_outline_item(store, row: dict, unit_type: str | None, structural_role: str | None) -> str:
    citation = str(row.get("citation") or row.get("label") or "").strip()
    if structural_role == "document":
        return citation
    outline = _structure_outline_item(store, row)
    intent = intent_config_for(getattr(store.config, "structured_strategy", "generic"), store.config)
    if unit_type != intent.get("structure_unit_type") or not outline:
        return outline
    child_ids = {
        unit.get("legal_unit_id")
        for unit in store.legal_units
        if unit.get("parent_legal_unit_id") == row.get("legal_unit_id") and unit.get("structural_role") == "provision"
    }
    support = next(
        (item for item in store.evidence if item.get("legal_unit_id") in child_ids and item.get("citation_final") is True),
        None,
    )
    if support is None:
        return outline
    proposition = _compact_text(support.get("quoted_text") or "")
    support_label = str(support.get("citation") or "").strip()
    if support_label and proposition.casefold().startswith(support_label.casefold()):
        proposition = proposition[len(support_label) :].lstrip(" :.-")
    proposition = re.split(r"(?<=[.!?])\s+", proposition, maxsplit=1)[0].strip()
    return f"{outline}; pokok: {proposition[:240]}"


def _version_comparison_answer(
    store,
    roles: tuple[str, ...],
    rows: tuple[dict, ...],
    *,
    references: tuple[str, ...] = (),
) -> str:
    """Compose a bounded comparison from exact normative propositions.

    Parent provision spans contain their child subprovisions as well. Prefer the parent
    only when the request is document-level; explicit references keep their
    leaf span.  All rows remain in ``comparison_support`` for grounding, while
    the answer uses one concise proposition per unit.
    """
    sections: list[str] = []
    propositions: dict[str, dict[str, str]] = {role: {} for role in roles}
    units = {str(unit.get("legal_unit_id")): unit for unit in store.legal_units}
    for role in roles:
        role_rows = tuple(row for row in rows if row.get("source_role") == role)
        parent_units = {
            str(row.get("legal_unit_id") or "")
            for row in role_rows
            if units.get(str(row.get("legal_unit_id")), {}).get("structural_role") == "provision"
        }
        for row in role_rows:
            citation = _legal_reference_label(store, row) or str(row.get("label") or "").strip()
            if not citation:
                continue
            unit = units.get(str(row.get("legal_unit_id")), {})
            is_leaf = unit.get("structural_role") == "subprovision"
            if not references and is_leaf and str(unit.get("parent_legal_unit_id") or "") in parent_units:
                continue
            text = _compact_text(row.get("quoted_text") or row.get("display_text"))
            if text.casefold().startswith(citation.casefold()):
                text = text[len(citation) :].lstrip(" :.-")
            sentence = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0].strip().rstrip(".")
            if sentence:
                propositions[role].setdefault(citation, sentence[:240])
        details = tuple(
            f"{citation}: {text}"
            for citation, text in sorted(
                propositions[role].items(),
                key=lambda item: _natural_label_sort_key(item[0]),
            )
        )
        if details:
            # Keep document-level answers reviewable; the complete row set is
            # still exposed as citation/evidence support in the response.
            sections.append(f"{_document_title(store, {'source_role': role})}: {'; '.join(details[:12])}.")
    if len(roles) > 1:
        labels = tuple(_document_title(store, {"source_role": role}) for role in roles)
        if references:
            common = set.intersection(*(set(propositions.get(role, {})) for role in roles)) if propositions else set()
            changed = tuple(
                citation
                for citation in sorted(common, key=_natural_label_sort_key)
                if len({propositions.get(role, {}).get(citation) for role in roles}) > 1
            )
            prefix = (
                f"Perbandingan substantif antara {labels[0]} dan {labels[1]} menunjukkan redaksi berbeda pada {', '.join(changed)}."
                if changed
                else f"Perbandingan substantif antara {labels[0]} dan {labels[1]} didukung oleh ketentuan yang tercantum pada kedua sumber."
            )
        else:
            prefix = f"Perbandingan substantif antara {labels[0]} dan {labels[1]} menunjukkan cakupan dan rumusan yang berbeda."
        sections.insert(0, prefix)
    return "\n\n".join(sections)


def _semantic_supports_query(store: EvidenceStore, query: str, row: dict) -> bool:
    """Guard the single-row lexical fallback after safe lexical normalization."""
    source = " ".join(str(value or "") for value in (row.get("citation"), " ".join(row.get("hierarchy") or ()), row.get("quoted_text")))
    # Typed EvidenceRequirement assignment owns complex-answer completeness.
    # This guard is only for a one-row lexical fallback, where every retained
    # content token must be grounded after safe aliases and question auxiliaries
    # have been removed by the lexical policy.
    return semantic_supports_text(store, query, source)


def _semantic_support_rank(row: dict) -> tuple[int, int, str]:
    return (
        1 if row.get("authority_kind") == "structural_context" else 0,
        len(_compact_text(row.get("quoted_text"))),
        str(row.get("evidence_id") or ""),
    )


def _semantic_specificity(store: EvidenceStore, row: dict) -> int:
    """Prefer rare corpus terms over generic legal framing in BM25 ties."""
    normalization: dict[str, object] = getattr(store.config, "setting", lambda *_: {})("lexical_normalization", {}) or {}
    raw_aliases = normalization.get("aliases") if isinstance(normalization, dict) else None
    aliases: dict[str, str] = {
        str(key).casefold(): str(value).casefold() for key, value in (raw_aliases.items() if isinstance(raw_aliases, dict) else ())
    }
    terms = tuple(row.get("lexical_supported_terms") or ())
    if not terms:
        return 0
    frequency: dict[str, int] = {}
    for evidence in store.evidence:
        values = (evidence.get("citation"), evidence.get("quoted_text"), " ".join(evidence.get("hierarchy") or ()))
        present = {token for value in values for token in meaningful_tokens(str(value or ""), aliases=aliases)}
        for token in present:
            frequency[token] = frequency.get(token, 0) + 1
    threshold = max(10, len(store.evidence) // 20)
    score = sum(frequency.get(str(term), 0) <= threshold for term in terms)
    row_text = re.sub(r"[^\w]+", " ", str(row.get("quoted_text") or "").casefold())
    # When several provisions contain the same concept, prefer the shortest
    # complete provision. Repeated generic words in a chapter container must
    # not outrank a focused article/ayat statement.
    term_hits = sum(bool(row_text.count(_compact_text(term).casefold())) for term in terms)
    score += term_hits
    score += min(4, int(term_hits * 20 / max(1, len(row_text.split()))))
    citation = str(row.get("citation") or "").strip()
    hierarchy = tuple(str(item) for item in row.get("hierarchy") or ())
    # Prefer an isolated provision over a chapter/preamble container when
    # both satisfy the same semantic terms.  This is structural metadata,
    # not corpus-specific legal vocabulary.
    if re.match(r"^Pasal\b", citation, re.IGNORECASE):
        score += 2
    elif citation.startswith("(") and any(re.match(r"^Pasal\b", item, re.IGNORECASE) for item in hierarchy):
        score += 3
    return score


def _metadata_answer(store, rows: tuple[dict, ...]) -> str:
    page_suffix = _metadata_page_suffix(rows)
    roles = tuple(dict.fromkeys(str(row.get("source_role") or "") for row in rows if row.get("source_role")))
    if len(roles) > 1 and all(row.get("field") == "signatories" and row.get("fact_kind") != "person_role" for row in rows):
        documents = {str(row.get("source_role")): row for row in store.document_metadata if row.get("source_role") in roles}
        signatories = {role: tuple(documents.get(role, {}).get("signatories") or ()) for role in roles}
        labels = {str(row.get("source_role")): str(row.get("source_label") or row.get("source_role")) for row in rows}
        lines = []
        identities = {
            role: {str(person.get("entity_identity") or normalize_intent_text(person.get("name_text"))): person for person in people}
            for role, people in signatories.items()
        }
        for role in roles:
            people = signatories[role]
            grouped: dict[str, list[str]] = {}
            for person in people:
                grouped.setdefault(str(person.get("role_text") or "Penandatangan"), []).append(str(person.get("name_text") or ""))
            summary = "; ".join(f"{title}: {', '.join(name for name in names if name)}" for title, names in grouped.items())
            lines.append(f"{labels[role]}: {summary}.")
        if len(roles) == 2:
            left, right = roles
            only_left = [str(person.get("name_text")) for key, person in identities[left].items() if key not in identities[right]]
            only_right = [str(person.get("name_text")) for key, person in identities[right].items() if key not in identities[left]]
            differences = []
            if only_left:
                differences.append(f"hanya {labels[left]} mencantumkan {', '.join(only_left)}")
            if only_right:
                differences.append(f"hanya {labels[right]} mencantumkan {', '.join(only_right)}")
            lines.append(f"Perbedaannya: {'; '.join(differences) if differences else 'susunan penandatangan sama'}.")
        return "\n\n".join(lines) + page_suffix
    names = _unique_printed_names(rows)
    printed_roles = tuple(dict.fromkeys(str(row.get("printed_role") or "") for row in rows if row.get("printed_role")))
    if names and len(names) == 1 and len(roles) > 1 and len(printed_roles) == 1:
        sources = ", ".join(str(row.get("source_label") or row.get("source_role")) for row in rows)
        return f"{names[0]} tercantum sebagai {printed_roles[0]} dalam {sources}." + page_suffix
    if names and len(roles) == 1 and len(printed_roles) == 1:
        source = str(rows[0].get("source_label") or rows[0].get("source_role"))
        status = _source_status_label(rows[0], store)
        suffix = f" Sumber: {status}." if status else ""
        return f"{printed_roles[0]} yang tercantum dalam {source}: {', '.join(names)}.{suffix}" + page_suffix
    values = names or tuple(
        dict.fromkeys(
            _compact_text(row.get("display_text") or row.get("answer") or row.get("metadata_answer"))
            for row in rows
            if row.get("display_text") or row.get("answer") or row.get("metadata_answer")
        )
    )
    source_label = _source_status_label(rows[0], store) if rows else None
    suffix = f" Sumber: {source_label}." if source_label else ""
    return f"{', '.join(values)}.{suffix}{page_suffix}" if values else "Bukti metadata terverifikasi tidak memuat nilai."


def _metadata_page_suffix(rows: tuple[dict, ...]) -> str:
    pages = sorted({int(page) for row in rows if row.get("page_query") for page in row.get("page_numbers") or ()})
    return f" Halaman sumber: {', '.join(str(page) for page in pages)}." if pages else ""


def _claim_answer(claims) -> str:
    claim = claims[0]
    if claim.status == "contradicted":
        return f"Klaim �{claim.claim_text}� bertentangan dengan segmen terverifikasi."
    return f"Klaim �{claim.claim_text}� tidak didukung oleh segmen terverifikasi dalam korpus ini."


def _empty_citation_fields() -> dict:
    return {
        "final_citations": (),
        "historical_citations": (),
        "metadata_support": (),
        "structural_support": (),
        "trace_support": (),
        "citation_payloads": (),
        "viewer_refs": (),
        "validation_reasons": {},
    }


def _answer_templates(store) -> dict[str, str]:
    configured: dict = getattr(getattr(store, "config", None), "setting", lambda *args: {})("answer_templates", {})
    return ANSWER_TEMPLATES | dict(configured or {})


def _document_title(store, source: dict) -> str:
    catalog = store.config.setting("document_catalog", {}) or {}
    return (catalog.get("titles") or {}).get(source.get("source_role")) or source.get("filename") or source["source_document_id"]


def _unique_printed_names(rows: tuple[dict, ...]) -> tuple[str, ...]:
    """Collapse formatting variants without discarding their source supports."""
    names: dict[str, str] = {}
    for row in rows:
        value = str(row.get("printed_name") or "").strip()
        if row.get("fact_kind") != "person_role" or not value:
            continue
        names.setdefault(normalize_intent_text(value), value)
    return tuple(names.values())


def _natural_label_sort_key(value: str) -> tuple[tuple[int, object], ...]:
    """Sort configured legal labels without knowing a corpus's vocabulary."""
    return tuple((1, int(part)) if part.isdigit() else (0, part.casefold()) for part in re.split(r"(\d+)", value) if part)
