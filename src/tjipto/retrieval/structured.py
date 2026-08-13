from __future__ import annotations

import re
from dataclasses import dataclass

from tjipto.corpora.intent_config import intent_config_for, resolve_instrument_intent
from tjipto.corpora.parser_dispatch import (
    label_keys,
    parse_ayat_reference,
    parse_bab_reference,
    parse_legal_references,
    parse_pasal_reference,
    resolve_navigation,
)
from tjipto.evidence.store import EvidenceStore
from tjipto.corpora.source_arbitration import resolve_source_scope


@dataclass(frozen=True)
class StructuralRequest:
    operation: str
    unit: str
    inclusion: str
    bab: str | None
    pasal: str | None
    ayat: str | None
    include_hierarchy: bool = False


def structured_lookup(
    store: EvidenceStore,
    query: str,
    limit: int = 10,
    *,
    strategy: str = "uud_1945",
    source_role: str | None = None,
    allow_navigation: bool = True,
) -> tuple[dict, ...]:
    config = getattr(store, "config", None)
    intent = intent_config_for(strategy, config)
    corpus_id = _corpus_id(config)
    if not intent["structured_lookup_enabled"]:
        return ()
    structure_list = _structure_list_rows(store, query, limit, intent, corpus_id, source_role)
    if structure_list:
        return structure_list
    instrument = _instrument_rows(store, query, limit, strategy=strategy, config=config)
    if instrument:
        return instrument
    navigation = _navigation_rows(store, query, limit, corpus_id) if allow_navigation else ()
    if navigation:
        return navigation
    targets = _targets(query, intent, corpus_id)
    if not targets:
        return ()
    legal_unit_ids = {
        row["legal_unit_id"]
        for row in (*getattr(store, "legal_units", ()), *getattr(store, "chunks", ()))
        if row.get("legal_unit_id") and _matches_unit(row, targets, corpus_id)
    }
    scope = resolve_source_scope(query, strategy=strategy, config=config)
    requested_role = source_role
    if requested_role is None and not scope.unresolved:
        requested_role = scope.role or getattr(config, "preferred_source_role", None)
    request = _structural_request(query, intent, corpus_id)
    if request.bab:
        rows = _bab_request_rows(store, request, requested_role)
        if rows:
            return rows
    preferred_unit_ids = _preferred_unit_ids(store, targets, requested_role, corpus_id)
    fallback_rows = [
        row
        for row in store.evidence
        if row.get("status") == "final"
        and store.bboxes_for(row["evidence_id"])
        and (requested_role is None or not row.get("source_role") or row.get("source_role") == requested_role)
        and (
            row.get("legal_unit_id") in preferred_unit_ids
            if preferred_unit_ids
            else row.get("legal_unit_id") in legal_unit_ids or _matches(row, targets, corpus_id)
        )
    ]
    return tuple(fallback_rows[:limit])


def has_structured_target(query: str, *, strategy: str = "uud_1945", config=None) -> bool:
    intent = intent_config_for(strategy, config)
    if not intent["structured_lookup_enabled"]:
        return False
    if _instrument_target(query, strategy=strategy, config=config):
        return True
    return bool(_targets(query, intent, _corpus_id(config))) or _has_incomplete_pasal(query)


def _structural_request(query: str, intent: dict, corpus_id: str) -> StructuralRequest:
    """Classify only the corpus-configured structural granularity request."""
    bab = parse_bab_reference(corpus_id, query)
    pasal = parse_pasal_reference(corpus_id, query, allow_roman=True)
    ayat = parse_ayat_reference(corpus_id, query)
    terms = intent.get("structure_request_terms") or {}
    operation = (
        "title" if _contains_any(query, terms.get("title", ()))
        else "enumerate" if _contains_any(query, terms.get("enumerate", ()))
        else "content" if _contains_any(query, terms.get("content", ()))
        else "reference"
    )
    unit = "ayat" if ayat else "pasal" if pasal else "bab" if bab else ""
    inclusion = "descendants" if bab and not pasal and operation in {"content", "enumerate"} else "exact"
    return StructuralRequest(
        operation=operation,
        unit=unit,
        inclusion=inclusion,
        bab=bab,
        pasal=pasal,
        ayat=ayat,
        include_hierarchy=operation == "enumerate" and _contains_any(query, terms.get("hierarchy", ())),
    )


def _contains_any(query: str, terms: tuple[str, ...] | list[str]) -> bool:
    folded = f" {str(query or '').casefold()} "
    return any(f" {str(term).casefold()} " in folded for term in terms)


def _bab_request_rows(store: EvidenceStore, request: StructuralRequest, requested_role: str | None) -> tuple[dict, ...]:
    units = tuple(store.legal_units)
    headings = tuple(
        unit for unit in units
        if unit.get("unit_type") == "bab_record"
        and str(unit.get("unit_label") or "").casefold() == str(request.bab).casefold()
        and (requested_role is None or unit.get("source_role") == requested_role)
    )
    if not headings:
        return ()
    heading = headings[0]
    heading_row = _heading_projection(store, heading)
    if heading_row is None:
        return ()
    heading_is_answer = request.inclusion == "exact" and request.pasal is None
    if heading_is_answer:
        return (heading_row | _quote_projection(),)

    children = _descendant_units(units, str(heading.get("legal_unit_id") or ""))
    selected = _requested_units(children, request, str(heading.get("legal_unit_id") or ""))
    legal_rows = _normative_rows(store, selected, heading.get("source_role"))
    if not legal_rows:
        return ()
    return tuple(
        row | {"route_sources": ("structured",), "candidate_type": "structural_complete_set", "presentation_order": index}
        for index, row in enumerate((heading_row, *legal_rows))
    )


def _heading_projection(store: EvidenceStore, heading: dict) -> dict | None:
    evidence = next(
        (
            row for row in store.evidence
            if row.get("legal_unit_id") == heading.get("legal_unit_id")
            and row.get("status") == "final"
            and store.bboxes_for(row.get("evidence_id"))
        ),
        None,
    )
    if evidence is None:
        return None
    child_ids = {
        unit.get("legal_unit_id")
        for unit in store.legal_units
        if unit.get("parent_legal_unit_id") == heading.get("legal_unit_id")
    }
    child_span_ids = {
        span_id
        for unit in store.legal_units
        if unit.get("legal_unit_id") in child_ids
        for span_id in unit.get("text_span_ids") or ()
    }
    span_ids = tuple(span_id for span_id in heading.get("text_span_ids") or () if span_id not in child_span_ids)
    bbox_refs = tuple(evidence.get("bbox_refs") or ())[:len(span_ids)]
    if not span_ids or len(span_ids) != len(bbox_refs):
        return None
    spans = {span.get("text_span_id"): span for span in store.page_text_spans}
    heading_text = "\n".join(
        str(spans[span_id].get("exact_quote") or spans[span_id].get("text") or "").rstrip()
        for span_id in span_ids
        if span_id in spans
    )
    if not heading_text:
        return None
    return evidence | {
        "quoted_text": heading_text,
        "display_text": heading_text,
        "copy_text": heading_text,
        "text_span_ids": span_ids,
        "bbox_refs": bbox_refs,
        "page_numbers": tuple(dict.fromkeys(
            span.get("page_number")
            for span in store.page_text_spans
            if span.get("text_span_id") in set(span_ids) and span.get("page_number") is not None
        )),
        "route_sources": ("structured",),
        "candidate_type": "structural_heading_candidate",
    }


def _quote_projection() -> dict:
    return {
        "authority_kind": "normative_legal_text",
        "citation_final": True,
        "citable": True,
        "relevant_quote_eligible": True,
        "presentation_as_legal_quote": True,
        "candidate_type": "structural_heading_answer",
        "route_sources": ("structured",),
    }


def _descendant_units(units: tuple[dict, ...], parent_id: str) -> tuple[dict, ...]:
    children: dict[str, list[dict]] = {}
    for unit in units:
        children.setdefault(str(unit.get("parent_legal_unit_id") or ""), []).append(unit)
    result: list[dict] = []
    pending = list(sorted(children.get(parent_id, ()), key=_unit_order))
    while pending:
        unit = pending.pop(0)
        result.append(unit)
        pending[0:0] = sorted(children.get(str(unit.get("legal_unit_id") or ""), ()), key=_unit_order)
    return tuple(result)


def _requested_units(children: tuple[dict, ...], request: StructuralRequest, parent_id: str) -> tuple[dict, ...]:
    if request.pasal:
        matched = tuple(unit for unit in children if str(unit.get("unit_label") or "").casefold() == request.pasal.casefold())
        if request.ayat:
            matched_ids = {unit.get("legal_unit_id") for unit in matched}
            return tuple(
                unit for unit in children
                if unit.get("parent_legal_unit_id") in matched_ids
                and str(unit.get("unit_label") or "").casefold() == request.ayat.casefold()
            )
        return tuple(unit for unit in matched if unit.get("unit_type") == "pasal_record")
    direct_children = tuple(unit for unit in children if unit.get("parent_legal_unit_id") == parent_id)
    direct = tuple(unit for unit in direct_children if unit.get("unit_type") == "pasal_record") or direct_children
    if not request.include_hierarchy:
        return direct
    parent_ids = {unit.get("parent_legal_unit_id") for unit in children}
    return tuple(
        unit for unit in children
        if unit.get("unit_type") == "ayat_record" or unit.get("legal_unit_id") not in parent_ids
    )


def _normative_rows(store: EvidenceStore, units: tuple[dict, ...], source_role: object) -> tuple[dict, ...]:
    unit_ids = {unit.get("legal_unit_id") for unit in units}
    rows = [
        row for row in store.evidence
        if row.get("legal_unit_id") in unit_ids
        and row.get("source_role") == source_role
        and row.get("status") == "final"
        and row.get("authority_kind") == "normative_legal_text"
        and row.get("citation_eligibility") == "eligible"
        and row.get("relevant_quote_eligible") is True
        and store.bboxes_for(row.get("evidence_id"))
    ]
    order = {unit.get("legal_unit_id"): index for index, unit in enumerate(units)}
    return tuple(sorted(rows, key=lambda row: (order.get(row.get("legal_unit_id"), len(order)), row.get("evidence_id", ""))))


def _unit_order(unit: dict) -> tuple[int, str]:
    return int(unit.get("sibling_order") or 0), str(unit.get("legal_unit_id") or "")


def has_instrument_target(query: str, *, strategy: str = "uud_1945", config=None) -> bool:
    """Whether a corpus-configured instrument operation owns this query."""
    return _instrument_target(query, strategy=strategy, config=config)


def _structure_list_rows(store: EvidenceStore, query: str, limit: int, intent: dict, corpus_id: str, source_role: str | None) -> tuple[dict, ...]:
    folded = (query or "").casefold()
    terms = tuple(str(term).casefold() for term in intent.get("structure_list_terms") or ())
    if not terms or not any(term in folded for term in terms):
        return ()
    if any(re.search(r"\bbab\s+[ivxlcdm]+[a-z]?\b", folded) for _ in (0,)):
        return ()
    unit_type = intent.get("structure_unit_type")
    units = [
        row for row in store.legal_units
        if row.get("unit_type") == unit_type
        and (source_role is None or row.get("source_role") == source_role)
        and row.get("source_role") == getattr(store.config, "preferred_source_role", row.get("source_role"))
    ]
    evidence_by_unit = {row.get("legal_unit_id"): row for row in store.evidence if row.get("status") == "final"}
    return tuple(
        evidence_by_unit[unit["legal_unit_id"]] | {"candidate_type": "structural_list_candidate", "route_sources": ("structured",)}
        for unit in sorted(units, key=lambda row: (row.get("page_start", 0), row.get("unit_label", "")))
        if unit.get("legal_unit_id") in evidence_by_unit
    )[:limit]


def structured_failure_reason(store: EvidenceStore, query: str, *, strategy: str = "uud_1945") -> str | None:
    corpus_id = _corpus_id(getattr(store, "config", None))
    if _has_incomplete_pasal(query):
        return "incomplete_legal_reference"
    if not _is_parent_reference(query, corpus_id):
        return None
    pasal = parse_pasal_reference(corpus_id, query, allow_roman=True)
    scope = resolve_source_scope(query, strategy=strategy, config=getattr(store, "config", None))
    role = None if scope.unresolved else scope.role
    parents = [
        row
        for row in store.legal_units
        if row.get("unit_type") == "pasal_record"
        and row.get("unit_label", "").casefold() == str(pasal).casefold()
        and (role is None or row.get("source_role") == role)
    ]
    if not parents:
        return "pasal_aggregate_source_missing"
    return next(
        (
            row.get("aggregate_failure_reason")
            for parent in parents
            for row in (parent, _chunk_for_unit(store, parent.get("legal_unit_id")))
            if row and row.get("aggregate_failure_reason")
        ),
        "pasal_aggregate_geometry_unavailable",
    )


def _instrument_target(query: str, *, strategy: str, config=None) -> bool:
    return bool(_instrument_rows(None, query, 1, strategy=strategy, config=config, probe_only=True))


def _instrument_rows(
    store: EvidenceStore | None,
    query: str,
    limit: int,
    *,
    strategy: str,
    config=None,
    probe_only: bool = False,
) -> tuple[dict, ...]:
    folded = (query or "").casefold()
    intent = intent_config_for(strategy, config)
    corpus_id = _corpus_id(config)
    bab = parse_bab_reference(corpus_id, query)
    if bab and any(pattern in folded for pattern in intent["instrument_deletion_words"]) and not any(
        pattern in folded for pattern in intent["instrument_change_context_words"]
    ):
        if probe_only:
            return ({"probe": True},)
        matches: list[dict] = []
        for row in getattr(store, "evidence", ()):
            hierarchy = {str(value).casefold() for value in row.get("hierarchy") or ()}
            text = str(row.get("quoted_text") or "").casefold()
            if bab.casefold() not in hierarchy or not any(word in text for word in intent["instrument_deletion_evidence_words"]):
                continue
            if row.get("authority_kind") != "normative_legal_text" or row.get("citation_eligibility") != "eligible":
                continue
            candidate = _candidate(row, "normative_deletion_candidate")
            if candidate is not None:
                matches.append(candidate)
        if matches:
            return tuple(matches[:limit])
    if (
        bab
        and any(pattern in folded for pattern in intent["instrument_deletion_words"])
        and any(pattern in folded for pattern in intent["instrument_change_context_words"])
    ):
        if probe_only:
            return ({"probe": True},)
        clause_matches: list[dict] = []
        prefix = intent["instrument_citation_templates"].get("prefix", "")
        clause_marker = intent["instrument_citation_templates"].get("clause_marker", "")
        for row in getattr(store, "evidence", ()):
            citation = str(row.get("citation") or "")
            if not (prefix and clause_marker and citation.startswith(prefix) and clause_marker in citation):
                continue
            text = str(row.get("quoted_text") or "")
            if bab.casefold() in text.casefold() and any(word in text.casefold() for word in intent["instrument_deletion_evidence_words"]):
                candidate = _candidate(row, "instrument_clause_candidate")
                if candidate is not None:
                    clause_matches.append(candidate)
        return tuple(clause_matches[:limit])
    decision = resolve_instrument_intent(query, intent, corpus=corpus_id)
    if decision.target_status == "instrument_unresolved":
        return ({"probe": True},) if probe_only else ()
    if decision.target_status.startswith("instrument_resolved") and decision.target_citation:
        if probe_only:
            return ({"probe": True},)
        row = _instrument_evidence(store, decision.amendment or "", decision.target_citation)
        candidate = _candidate(row, f"instrument_{decision.role_family}_candidate")
        return (candidate,) if candidate is not None else ()
    return ()


def _targets(query: str, intent: dict, corpus_id: str) -> tuple[str, ...]:
    text = query or ""
    folded = text.casefold()
    for section in intent["structured_sections"]:
        if any(alias in folded for alias in section.get("aliases", ())):
            target = section["target"]
            return _with_pasal(target, text, corpus_id) if section.get("with_pasal") else (target,)
    bab = parse_bab_reference(corpus_id, text)
    if bab:
        return (bab.casefold(),)
    pasal = parse_pasal_reference(corpus_id, text, allow_roman=True)
    if pasal:
        targets = [pasal.casefold()]
        ayat = parse_ayat_reference(corpus_id, text)
        if ayat:
            targets.append(ayat)
        return tuple(targets)
    return ()


def _is_parent_reference(query: str, corpus_id: str) -> bool:
    return (
        len(parse_legal_references(corpus_id, query)) == 1
        and parse_pasal_reference(corpus_id, query, allow_roman=True) is not None
        and not parse_ayat_reference(corpus_id, query)
    )


def _has_incomplete_pasal(query: str) -> bool:
    return bool(re.fullmatch(r"\s*pasal(?:\s*[?!.]+)?\s*", query or "", flags=re.IGNORECASE))


def _chunk_for_unit(store: EvidenceStore, legal_unit_id: str | None) -> dict | None:
    return next((row for row in store.chunks if row.get("legal_unit_id") == legal_unit_id), None)


def _navigation_rows(
    store: EvidenceStore,
    query: str,
    limit: int,
    corpus_id: str,
) -> tuple[dict, ...]:
    navigation = resolve_navigation(corpus_id, query)
    if navigation is None:
        return ()
    label, direction = navigation
    scope = resolve_source_scope(query, strategy=getattr(store.config, "query_strategy", "generic"), config=store.config)
    preferred_role = None if scope.unresolved else scope.role
    source = next(
        (
            row
            for row in store.legal_units
            if row.get("unit_label", "").casefold() == label.casefold()
            and row.get("structural_role") in {"provision", "subprovision"}
            and (preferred_role is None or row.get("source_role") == preferred_role)
        ),
        None,
    )
    references = parse_legal_references(corpus_id, query)
    referenced = str(references[0].get("reference") or "") if references else ""
    _, separator, ayat_label = referenced.partition(" ayat ")
    if source is not None and separator:
        source = next(
            (
                row
                for row in store.legal_units
                if row.get("unit_label") == ayat_label
                and source.get("legal_unit_id") in (row.get("parent_legal_unit_ids") or ())
                and row.get("structural_role") == "subprovision"
            ),
            None,
        )
    if source is None:
        return ()
    # Article sequence may cross a chapter boundary, while paragraph sequence
    # remains inside its article.  Artifact order is the source-derived order
    # for this document; sibling_order alone resets at each parent.
    if source.get("structural_role") == "subprovision":
        ordered = [
            row
            for row in store.legal_units
            if row.get("source_document_id") == source.get("source_document_id")
            and row.get("parent_legal_unit_id") == source.get("parent_legal_unit_id")
            and row.get("structural_role") == "subprovision"
        ]
    else:
        ordered = [
            row
            for row in store.legal_units
            if row.get("source_document_id") == source.get("source_document_id")
            and row.get("structural_role") == "provision"
        ]
    try:
        source_index = next(index for index, row in enumerate(ordered) if row.get("legal_unit_id") == source.get("legal_unit_id"))
    except StopIteration:
        return ()
    target_index = source_index + (1 if direction == "next" else -1)
    target = ordered[target_index] if 0 <= target_index < len(ordered) else None
    if target is None:
        return ()
    rows = [
        row | {"candidate_type": "structural_navigation_candidate", "navigation_direction": direction}
        for row in store.evidence
        if target is not None and row.get("legal_unit_id") == target.get("legal_unit_id") and row.get("status") == "final" and store.bboxes_for(row["evidence_id"])
    ]
    return tuple(rows[:limit])


def _with_pasal(section: str, text: str, corpus_id: str) -> tuple[str, ...]:
    pasal = parse_pasal_reference(corpus_id, text, allow_roman=True)
    return (section, pasal.casefold()) if pasal else (section,)


def _matches(row: dict, targets: tuple[str, ...], corpus_id: str) -> bool:
    values = [row.get("citation", ""), *(row.get("hierarchy") or ())]
    haystack = {key for value in values for key in _label_keys(value, corpus_id)}
    return all(target in haystack for target in targets)


def _matches_unit(row: dict, targets: tuple[str, ...], corpus_id: str) -> bool:
    values = [row.get("unit_label", ""), *(row.get("hierarchy") or ())]
    haystack = {key for value in values for key in _label_keys(value, corpus_id)}
    return all(target in haystack for target in targets)


def _preferred_unit_ids(store: EvidenceStore, targets: tuple[str, ...], requested_role: str | None, corpus_id: str) -> set[str]:
    if not targets:
        return set()
    leaf = targets[-1]
    return {
        row["legal_unit_id"]
        for row in getattr(store, "legal_units", ())
        if leaf in _label_keys(row.get("unit_label"), corpus_id)
        and _matches_unit(row, targets, corpus_id)
        and (requested_role is None or not row.get("source_role") or row.get("source_role") == requested_role)
    }


def _label_keys(value: object, corpus_id: str) -> set[str]:
    return label_keys(corpus_id, value)


def _corpus_id(config) -> str:
    return str(getattr(config, "corpus_id", ""))


def _instrument_evidence(store: EvidenceStore | None, source_role: str, citation: str) -> dict | None:
    if store is None:
        return None
    return next(
        (
            row
            for row in store.evidence
            if row.get("source_role") == source_role and row.get("citation") == citation and row.get("status") == "final"
        ),
        None,
    )


def _candidate(row: dict | None, candidate_type: str) -> dict | None:
    return row | {"candidate_type": candidate_type} if row else None
