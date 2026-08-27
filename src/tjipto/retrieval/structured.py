from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

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


def structural_count(store: EvidenceStore, query: str, *, strategy: str = "uud_1945") -> dict | None:
    """Return one source-derived aggregate only when every counted unit is grounded."""
    config = getattr(store, "config", None)
    intent = intent_config_for(strategy, config)
    folded = (query or "").casefold()
    page_count = _document_page_count(store, query, folded, strategy=strategy, config=config)
    if page_count is not None:
        return page_count
    configured_units = intent.get("structure_count_units") or {}
    if not isinstance(configured_units, dict):
        return None
    count_intent = bool(re.search(r"\b(?:berapa|jumlah)\b", folded))
    has_count_unit_phrase = any(
        str(term).casefold() in folded
        for rule in configured_units.values()
        if isinstance(rule, dict)
        for term in rule.get("terms") or ()
    )
    requested = tuple(
        (name, rule)
        for name, rule in configured_units.items()
        if isinstance(rule, dict)
        and (
            any(str(term).casefold() in folded for term in rule.get("terms") or ())
            or (
                count_intent
                and has_count_unit_phrase
                and re.search(rf"\b{re.escape(str(name).casefold())}\b", folded)
            )
        )
    )
    if not requested:
        terms = tuple(str(term).casefold() for term in intent.get("structure_count_terms") or ())
        if not terms or not any(term in folded for term in terms):
            return None
        requested = (("pasal", {
            "unit_type": intent.get("structure_count_unit_type") or "pasal_record",
            "label_pattern": r"Pasal\s+\d+[A-Z]?",
            "authority_kinds": ("normative_legal_text",),
            "label": "Pasal",
        }),)
    article_count = _article_ayat_count(store, query, strategy=strategy, requested=requested)
    if article_count is not None:
        return article_count
    document_count = _source_document_count(store, requested)
    if document_count is not None:
        return document_count
    scope = resolve_source_scope(query, strategy=strategy, config=config)
    all_sources = any(str(term).casefold() in folded for term in intent.get("structure_count_all_source_terms") or ())
    roles = (
        tuple(getattr(config, "source_roles", ()))
        if all_sources
        else (() if scope.unresolved else (scope.role or getattr(config, "preferred_source_role", None),))
    )
    roles = tuple(role for role in roles if role)
    if not roles:
        return None
    catalog = config.setting("document_catalog", {}) if config is not None else {}
    titles = catalog.get("titles", {}) if isinstance(catalog, dict) else {}
    sources = {str(row.get("source_role")): row for row in store.source_documents}
    if any(role not in sources for role in roles):
        return None
    member_ids: list[str] = []
    support_ids: list[str] = []
    counts: dict[str, dict[str, int]] = {}
    units_by_key: dict[tuple[str, str], tuple[dict, ...]] = {}
    for role in roles:
        counts[role] = {}
        for name, rule in requested:
            pattern = str(rule.get("label_pattern") or ".+")
            units = tuple(
                row for row in store.legal_units
                if row.get("source_role") == role
                and row.get("unit_type") == rule.get("unit_type")
                and re.fullmatch(pattern, str(row.get("unit_label") or ""), re.IGNORECASE)
            )
            authority_kinds = set(rule.get("authority_kinds") or ())
            evidence_by_unit = {
                row.get("legal_unit_id"): row
                for row in store.evidence
                if row.get("status") == "final"
                and row.get("authority_kind") in authority_kinds
                and (
                    row.get("citation_eligibility") == "eligible"
                    or row.get("authority_kind") == "structural_context"
                )
            }
            if any(unit.get("legal_unit_id") not in evidence_by_unit for unit in units):
                return None
            counts[role][name] = len(units)
            units_by_key[(role, name)] = units
            member_ids.extend(str(unit["legal_unit_id"]) for unit in units)
            support_ids.extend(str(evidence_by_unit[unit["legal_unit_id"]]["evidence_id"]) for unit in units)

    lines = []
    for role in roles:
        source = sources[role]
        title = str(titles.get(role) or source.get("document_title") or source.get("title") or "Dokumen sumber")
        parts = [f"{counts[role][name]} {rule.get('label') or name}" for name, rule in requested]
        summary = ", ".join(parts[:-1]) + (f", dan {parts[-1]}" if len(parts) > 1 else parts[0])
        lines.append(f"{title} memuat {summary}.")
    answer = "\n".join(lines)
    source_role = roles[0] if len(roles) == 1 else None
    source = sources[roles[0]] if len(roles) == 1 else {}
    unit_type = str(requested[0][1].get("unit_type") or "document_structure")
    if len(roles) == 1 and len(requested) == 1 and requested[0][0] == "pasal":
        ordered = tuple(sorted(units_by_key[(roles[0], "pasal")], key=lambda row: _pasal_number(str(row.get("unit_label") or ""))))
        if ordered:
            first, last = str(ordered[0]["unit_label"]), str(ordered[-1]["unit_label"])
            title = str(titles.get(roles[0]) or source.get("document_title") or source.get("title") or "Dokumen sumber")
            answer = f"{title} memuat {len(ordered)} ketentuan berlabel Pasal, dari {first} sampai {last}."
    source_supports = _document_level_supports(sources, roles, titles)
    return {
        "evidence_id": f"structural_count::{source_role or 'all'}::{unit_type}",
        "status": "final",
        "authority_kind": "structural_context",
        "citation_final": False,
        "support_kind": "deterministic_structure",
        "fact_kind": "document_structure",
        "display_label": "Struktur dokumen",
        "display_text": answer,
        "quoted_text": answer,
        "source_document_id": source.get("source_document_id"),
        "source_role": source_role,
        "temporal_context": source_role,
        "source_label": str(titles.get(source_role) or source.get("document_title") or "Semua naskah terverifikasi"),
        "document_title": str(titles.get(source_role) or source.get("document_title") or "Semua naskah terverifikasi"),
        "structural_count": sum(sum(values.values()) for values in counts.values()),
        "structural_counts": counts,
        "structural_member_ids": tuple(dict.fromkeys(member_ids)),
        "structural_support_ids": tuple(dict.fromkeys(support_ids)),
        "viewer_highlightable": False,
        "viewer_ref": None,
        "source_supports": source_supports,
    }


def _document_page_count(store, query: str, folded: str, *, strategy: str, config) -> dict | None:
    """Count pages from the verified source manifest, not retrieved text."""
    if not re.search(r"\b(?:berapa|jumlah|total)\b(?:\s+\w+){0,4}\s+halaman\b", folded):
        return None
    scope = resolve_source_scope(query, strategy=strategy, config=config)
    if scope.unresolved:
        return None
    intent = intent_config_for(strategy, config)
    all_sources = any(str(term).casefold() in folded for term in intent.get("structure_count_all_source_terms") or ())
    roles = tuple(getattr(config, "source_roles", ()) or ()) if all_sources else (
        scope.role or getattr(config, "preferred_source_role", None),
    )
    roles = tuple(role for role in roles if role)
    sources = {str(row.get("source_role")): row for row in store.source_documents}
    if not roles or any(role not in sources or not int(sources[role].get("page_count") or 0) for role in roles):
        return None
    catalog = config.setting("document_catalog", {}) if config is not None else {}
    titles = catalog.get("titles", {}) if isinstance(catalog, dict) else {}
    lines = tuple(
        f"{titles.get(role) or sources[role].get('document_title') or role} memuat {int(sources[role]['page_count'])} halaman."
        for role in roles
    )
    answer = "\n".join(lines)
    return {
        "evidence_id": f"source_page_count::{'all' if all_sources else roles[0]}",
        "status": "final",
        "authority_kind": "structural_context",
        "citation_final": False,
        "support_kind": "verified_source_manifest",
        "fact_kind": "document_page_count",
        "display_label": "Jumlah halaman sumber",
        "display_text": answer,
        "quoted_text": answer,
        "source_document_id": sources[roles[0]].get("source_document_id") if len(roles) == 1 else None,
        "source_role": roles[0] if len(roles) == 1 else None,
        "temporal_context": roles[0] if len(roles) == 1 else None,
        "source_label": "Korpus terverifikasi" if len(roles) > 1 else str(titles.get(roles[0]) or roles[0]),
        "document_title": "Korpus terverifikasi" if len(roles) > 1 else str(titles.get(roles[0]) or roles[0]),
        "page_count": sum(int(sources[role]["page_count"]) for role in roles),
        "structural_count": sum(int(sources[role]["page_count"]) for role in roles),
        "structural_counts": {"source_documents": {role: int(sources[role]["page_count"]) for role in roles}},
        "structural_member_ids": tuple(sources[role].get("source_document_id") for role in roles),
        "structural_support_ids": (),
        "viewer_highlightable": False,
        "viewer_ref": None,
        "source_supports": _document_level_supports(sources, roles, titles),
    }


def _article_ayat_count(
    store: EvidenceStore,
    query: str,
    *,
    strategy: str,
    requested: tuple[tuple[str, dict], ...],
) -> dict | None:
    """Count ayat belonging to an explicitly referenced Pasal."""
    if not any(name == "ayat" for name, _ in requested):
        return None
    config = getattr(store, "config", None)
    corpus_id = _corpus_id(config)
    pasal = parse_pasal_reference(corpus_id, query, allow_roman=True)
    if not pasal:
        return None
    scope = resolve_source_scope(query, strategy=strategy, config=config)
    if scope.unresolved:
        return None
    role = scope.role or getattr(config, "preferred_source_role", None)
    parent = next(
        (
            unit
            for unit in store.legal_units
            if unit.get("unit_type") == "pasal_record"
            and str(unit.get("unit_label") or "").casefold() == pasal.casefold()
            and (role is None or unit.get("source_role") == role)
        ),
        None,
    )
    if parent is None:
        return None
    children = tuple(
        unit
        for unit in store.legal_units
        if unit.get("unit_type") == "ayat_record"
        and parent.get("legal_unit_id") in (unit.get("parent_legal_unit_ids") or ())
    )
    evidence_by_unit = {
        row.get("legal_unit_id"): row
        for row in store.evidence
        if row.get("status") == "final"
        and row.get("authority_kind") == "normative_legal_text"
        and row.get("citation_eligibility") == "eligible"
        and store.bboxes_for(row.get("evidence_id"))
    }
    if any(child.get("legal_unit_id") not in evidence_by_unit for child in children):
        return None
    parent_evidence = next(
        (
            row
            for row in store.evidence
            if row.get("legal_unit_id") == parent.get("legal_unit_id")
            and row.get("status") == "final"
            and store.bboxes_for(row.get("evidence_id"))
        ),
        None,
    )
    if parent_evidence is None:
        return None
    setting: Any = getattr(config, "setting", lambda *_: {})
    catalog = setting("document_catalog", {}) or {}
    title = str(
        catalog.get("titles", {}).get(role)
        or next((row.get("document_title") for row in store.source_documents if row.get("source_role") == role), None)
        or "Dokumen sumber"
    )
    count = len(children)
    answer = (
        f"{title} memuat {pasal} dengan {count} ayat bernomor."
        if count
        else f"{title} memuat {pasal} tanpa ayat bernomor."
    )
    support_ids = tuple(
        dict.fromkeys(
            [str(parent_evidence.get("evidence_id")), *(str(evidence_by_unit[child["legal_unit_id"]].get("evidence_id")) for child in children)]
        )
    )
    return {
        "evidence_id": f"structural_count::{role or 'preferred'}::ayat::{pasal.casefold()}",
        "status": "final",
        "authority_kind": "structural_context",
        "citation_final": False,
        "support_kind": "deterministic_structure",
        "fact_kind": "article_structure",
        "display_label": "Struktur ketentuan",
        "display_text": answer,
        "quoted_text": answer,
        "source_document_id": parent.get("source_document_id"),
        "source_role": role,
        "temporal_context": role,
        "source_label": title,
        "document_title": title,
        "structural_count": count,
        "structural_counts": {role: {pasal: count}},
        "structural_member_ids": tuple(str(child.get("legal_unit_id")) for child in children),
        "structural_support_ids": support_ids,
        "viewer_highlightable": False,
        "viewer_ref": None,
        "source_supports": _document_level_supports(
            {str(row.get("source_role")): row for row in store.source_documents},
            (role,) if role else (),
            catalog.get("titles", {}),
        ),
    }


def _source_document_count(store: EvidenceStore, requested: tuple[tuple[str, dict], ...]) -> dict | None:
    """Count only corpus-declared source documents, never inferred amendments."""
    if len(requested) != 1:
        return None
    name, rule = requested[0]
    roles = tuple(str(role) for role in rule.get("source_roles") or () if role)
    if not roles:
        return None
    sources = {str(row.get("source_role")): row for row in store.source_documents}
    if any(role not in sources for role in roles):
        return None
    config = getattr(store, "config", None)
    catalog = config.setting("document_catalog", {}) if config is not None else {}
    titles = catalog.get("titles", {}) if isinstance(catalog, dict) else {}
    label = str(rule.get("label") or name)
    documents = tuple(str(titles.get(role) or sources[role].get("document_title") or role) for role in roles)
    answer = f"Korpus terverifikasi memuat {len(roles)} {label}: {', '.join(documents)}."
    return {
        "evidence_id": f"source_document_count::{name}",
        "status": "final",
        "authority_kind": "structural_context",
        "citation_final": False,
        "support_kind": "deterministic_structure",
        "fact_kind": "source_document_count",
        "display_label": "Struktur korpus",
        "display_text": answer,
        "quoted_text": answer,
        "source_document_id": None,
        "source_role": None,
        "temporal_context": None,
        "source_label": "Korpus terverifikasi",
        "document_title": "Korpus terverifikasi",
        "structural_count": len(roles),
        "structural_counts": {"source_documents": {name: len(roles)}},
        "structural_member_ids": tuple(sources[role].get("source_document_id") for role in roles),
        "structural_support_ids": (),
        "viewer_highlightable": False,
        "viewer_ref": None,
        "source_supports": _document_level_supports(sources, roles, titles),
    }


def _document_level_supports(sources: dict[str, dict], roles: tuple[str, ...], titles: dict) -> tuple[dict, ...]:
    """Project source documents as traceable, deliberately non-highlighted support."""
    return tuple(
        {
            "evidence_id": f"source_document::{sources[role]['source_document_id']}",
            "status": "final",
            "authority_kind": "structural_context",
            "citation_final": False,
            "support_kind": "document_structure_source",
            "fact_kind": "source_document",
            "display_label": "Sumber dokumen",
            "display_text": "",
            "source_document_id": sources[role]["source_document_id"],
            "source_role": role,
            "temporal_context": sources[role].get("temporal_context") or role,
            "source_label": str(titles.get(role) or sources[role].get("document_title") or role),
            "document_title": str(titles.get(role) or sources[role].get("document_title") or role),
            "page_numbers": (1,),
            "viewer_highlightable": False,
            "viewer_target": {
                "action": "open_document",
                "page_numbers": (1,),
                "can_resolve": True,
            },
        }
        for role in roles
    )


def _pasal_number(label: str) -> tuple[int, str]:
    match = re.fullmatch(r"Pasal\s+(\d+)([A-Z]?)", label, re.IGNORECASE)
    return (int(match.group(1)), match.group(2).upper()) if match else (10**9, label)


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
    if parse_bab_reference(corpus_id, query):
        return ()
    unit_type = intent.get("structure_unit_type")
    requested_role = source_role or getattr(store.config, "preferred_source_role", None)
    units = [
        row for row in store.legal_units
        if row.get("unit_type") == unit_type
        and (requested_role is None or row.get("source_role") == requested_role)
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
            and row.get("structural_role") in {"division", "provision", "subprovision"}
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
    if direction == "direct":
        rows = [
            row | {"candidate_type": "structural_navigation_candidate", "navigation_direction": direction}
            for row in store.evidence
            if row.get("legal_unit_id") == source.get("legal_unit_id") and row.get("status") == "final" and store.bboxes_for(row["evidence_id"])
        ]
        return tuple(rows[:limit])
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
    elif source.get("structural_role") == "division":
        ordered = sorted(
            (
                row
                for row in store.legal_units
                if row.get("source_document_id") == source.get("source_document_id")
                and row.get("structural_role") == "division"
                and row.get("parent_legal_unit_id") == source.get("parent_legal_unit_id")
            ),
            key=lambda row: (row.get("page_start") or 0, row.get("sibling_order") or 0, row.get("legal_unit_id") or ""),
        )
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
    target = None
    if direction == "previous" and source.get("structural_role") == "division":
        # Inserted BAB labels (for example ``BAB XA``) are ordered after the
        # unsuffixed chapter they extend.  Prefer that corpus-derived base
        # label over sibling order, which can place another inserted chapter
        # first when page/sibling metadata is shared.
        match = re.fullmatch(r"(BAB\s+[IVXLCDM]+)[A-Z]", str(source.get("unit_label") or ""), re.IGNORECASE)
        if match:
            predecessor_label = match.group(1).casefold()
            target = next(
                (row for row in ordered if str(row.get("unit_label") or "").casefold() == predecessor_label),
                None,
            )
    if target is None:
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
