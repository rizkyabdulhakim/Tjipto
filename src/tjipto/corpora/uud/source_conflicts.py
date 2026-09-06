"""UUD-specific source-conflict arbitration and public anomaly projection."""

from __future__ import annotations

import re

from tjipto.corpora.intent_config import normalize_intent_text
from tjipto.corpora.parser_dispatch import parse_legal_reference
from tjipto.corpora.source_arbitration import source_reference_mappings_for_query
from tjipto.retrieval.answer import assemble_context_pack, empty_context_pack
from tjipto.retrieval.research import QueryVariant, ResearchIntent, ResearchPlan
from tjipto.runtime.answer_arbitration import _document_title
from tjipto.runtime.viewer import _authority_policy, _citation_with_authority, _source_mapping_semantics


def _source_anomaly_clarification(store, query: str) -> ResearchPlan | None:
    """Ask for the missing section when a source marker has two meanings."""
    if _source_anomaly_comparison_query(store, query):
        return None
    # An explicit corpus-declared mapping (for example the printed Pasal III
    # typo with its configured context) is already disambiguated by source
    # policy and should proceed to canonical lookup.
    if source_reference_mappings_for_query(query, config=store.config):
        return None
    folded = normalize_intent_text(query)
    for conflict in getattr(store, "source_conflicts", ()):
        anchors = tuple(normalize_intent_text(value) for value in conflict.get("query_anchor_terms") or ())
        if not any(_query_contains_term(folded, anchor) for anchor in anchors if anchor == "pasal iii"):
            continue
        sections = tuple(
            dict.fromkeys(
                term
                for term in (*conflict.get("query_exclusion_terms", ()), *conflict.get("query_anchor_terms", ()))
                if str(term).casefold().startswith("aturan ") and "pasal" not in str(term).casefold()
            )
        )
        if len(sections) < 2:
            continue
        if any(_query_contains_term(folded, section) for section in sections):
            continue
        reference = next((anchor for anchor in anchors if anchor.startswith("pasal ")), "")
        options = " atau ".join(f"{section.title()} {reference.title()}" for section in sections)
        question = f"Apakah yang dimaksud {options}?"
        plan = ResearchPlan(
            query,
            ResearchIntent(),
            (QueryVariant(query),),
            clarification_question=question,
            missing_dimensions=("legal_target",),
        )
        return plan
    return None


def _source_anomaly_response(store, corpus_id: str, query: str) -> dict | None:
    if store is None:
        return None
    if not _is_source_anomaly_query(store, query):
        return None
    conflict = _matched_source_conflict(store, query)
    if conflict is None:
        return _source_anomaly_fallback()
    support = _source_conflict_support(store, conflict)
    reasons = _source_conflict_reasons(store, query)
    exact_provenance = bool(support["evidence"])
    trace_only = bool(support["trace_support"]) and not exact_provenance
    answer = _source_anomaly_answer(store, conflict, query, exact_provenance=exact_provenance, trace_only=trace_only)
    return {
        "status": "limited_answer" if exact_provenance or trace_only else "insufficient_evidence",
        "route": "source_anomaly_explanation",
        "intent": "structured_lookup",
        "answer_type": "source_conflict_provenance" if exact_provenance or trace_only else "none",
        "answer": answer,
        "context_pack": support["context_pack"],
        "evidence": support["evidence"],
        "citations": (),
        "final_citations": (),
        "historical_citations": (),
        "metadata_support": (),
        "structural_support": (),
        "viewer_refs": (),
        "metadata_facts": (),
        "legal_relations": (),
        "trace_support": support["trace_support"],
        "answer_scope": support["answer_scope"],
        "warnings": support["warnings"],
        "insufficient_reasons": tuple(dict.fromkeys(reasons)) if not exact_provenance else (),
        "source_conflict": _public_source_conflict(conflict),
    }


def _matched_source_conflict(store, query: str) -> dict | None:
    folded = (query or "").casefold()
    if not _is_source_anomaly_query(store, query):
        return None
    intent = _source_conflict_intent(store)
    matches = [(score, row) for row in store.source_conflicts if (score := _source_conflict_match_score(store, row, folded, intent)) > 0]
    if not matches:
        return None
    matches.sort(key=lambda item: (-item[0], item[1].get("source_conflict_id") or ""))
    return matches[0][1]


def _is_source_anomaly_query(store, query: str) -> bool:
    folded = (query or "").casefold()
    intent = _source_conflict_intent(store)
    terms = tuple(str(term).casefold() for term in intent.get("query_terms") or ())
    if not any(_query_contains_term(folded, term) for term in terms):
        return False
    unresolved_terms = tuple(str(term).casefold() for term in intent.get("unresolved_query_terms") or ())
    if any(_query_contains_term(folded, term) for term in unresolved_terms):
        return True
    discrepancy_markers = tuple(str(marker).casefold() for marker in intent.get("discrepancy_terms") or ())
    if any(_query_contains_term(folded, marker) for marker in discrepancy_markers):
        return any(
            sum(_query_contains_term(folded, str(anchor).casefold()) for anchor in conflict.get("query_anchor_terms") or ()) >= 2
            for conflict in store.source_conflicts
        )
    return any(
        sum(_query_contains_term(folded, str(anchor).casefold()) for anchor in conflict.get("query_anchor_terms") or ()) >= 2
        for conflict in store.source_conflicts
    )


def _source_anomaly_comparison_query(store, query: str) -> bool:
    if not _is_source_anomaly_query(store, query):
        return False
    folded = (query or "").casefold()
    intent = _source_conflict_intent(store)
    unresolved_terms = tuple(str(term).casefold() for term in intent.get("unresolved_query_terms") or ())
    if any(_query_contains_term(folded, term) for term in unresolved_terms):
        return True
    references = re.findall(r"\b(?:pasal|bab)\s+(?:\d+[a-z]?|[ivxlcdm]+[a-z]?)\b", folded)
    if len(set(references)) >= 2:
        return True
    return any(_query_contains_term(folded, str(marker).casefold()) for marker in intent.get("discrepancy_terms") or ())


def _source_anomaly_fallback() -> dict:
    return {
        "status": "insufficient_evidence",
        "route": "source_anomaly_explanation",
        "intent": "structured_lookup",
        "answer_type": "none",
        "answer": "Bukti tidak cukup untuk mengaitkan pertanyaan ini dengan catatan konflik sumber tertentu.",
        "context_pack": empty_context_pack("source_anomaly_unresolved"),
        "evidence": (),
        "citations": (),
        "viewer_refs": (),
        "metadata_facts": (),
        "legal_relations": (),
        "answer_scope": "insufficient_evidence",
        "warnings": (),
        "insufficient_reasons": ("source_anomaly", "source_anomaly_unresolved"),
        "source_conflict": None,
    }


def _source_conflict_reasons(store, query: str) -> list[str]:
    intent = _source_conflict_intent(store)
    folded = (query or "").casefold()
    reasons = ["source_anomaly", "canonical_conflict"]
    for rule in intent.get("reason_rules") or ():
        terms = tuple(str(term).casefold() for term in rule.get("query_terms") or ())
        if any(_query_contains_term(folded, term) for term in terms):
            reasons.extend(str(reason) for reason in rule.get("reasons") or ())
            return reasons
    reasons.extend(str(reason) for reason in intent.get("default_reasons") or ())
    return reasons


def _source_conflict_match_score(store, conflict: dict, folded_query: str, intent: dict) -> int:
    exclusions = tuple(str(term).casefold() for term in conflict.get("query_exclusion_terms") or ())
    if any(_query_contains_term(folded_query, term) for term in exclusions):
        return 0
    required = tuple(str(term).casefold() for term in conflict.get("query_required_terms") or ())
    anchors = {str(term).casefold() for term in conflict.get("query_anchor_terms") or ()}
    source_role = str(_source_document_meta(store, conflict.get("source_document_id")).get("source_role") or "")
    role_label = str((intent.get("role_labels") or {}).get(source_role) or source_role).casefold()
    role_anchor_match = _query_contains_term(folded_query, role_label) and any(
        _query_contains_term(folded_query, anchor) for anchor in anchors
    )
    natural_discrepancy = (
        any(_query_contains_term(folded_query, str(marker).casefold()) for marker in intent.get("discrepancy_terms") or ())
        and sum(_query_contains_term(folded_query, anchor) for anchor in anchors) >= 2
    ) or sum(_query_contains_term(folded_query, anchor) for anchor in anchors) >= 2
    source_marker_context = (
        conflict.get("source_anomaly_kind") in {"source_marker_sequence_anomaly", "typed_source_discrepancy"} and role_anchor_match
    )
    if (
        required
        and not any(_query_contains_term(folded_query, term) for term in required)
        and not source_marker_context
        and not natural_discrepancy
    ):
        return 0
    explicit_anchor_match = any(len(anchor.split()) > 1 and _query_contains_term(folded_query, anchor) for anchor in anchors)
    semantic_required = tuple(term for term in required if term not in anchors or not _is_legal_reference_term(store, term))
    marker_context = role_anchor_match or natural_discrepancy or any(_query_contains_term(folded_query, term) for term in semantic_required)
    if (
        semantic_required
        and not any(_query_contains_term(folded_query, term) for term in semantic_required)
        and not role_anchor_match
        and not (
            conflict.get("source_anomaly_kind") in {"source_marker_sequence_anomaly", "typed_source_discrepancy"}
            and (explicit_anchor_match or role_anchor_match)
        )
    ):
        return 0
    if conflict.get("source_anomaly_kind") in {"source_marker_sequence_anomaly", "typed_source_discrepancy"} and not marker_context:
        return 0
    score = 0
    if _query_contains_term(folded_query, role_label):
        score += 4
    for token in (str(value).casefold() for value in (conflict.get("query_anchor_terms") or conflict.get("anchor_terms") or ())):
        if _query_contains_term(folded_query, token):
            score += 3
    if score:
        return score
    haystack = (
        " ".join(
            str(value or "")
            for value in (
                conflict.get("source_conflict_id"),
                conflict.get("type"),
                conflict.get("source_anomaly_kind"),
                conflict.get("source_mapping_kind"),
                conflict.get("classification"),
                conflict.get("source_document_id"),
                role_label,
            )
        )
        .replace("_", " ")
        .casefold()
    )
    query_tokens = _meaningful_conflict_tokens(folded_query, intent)
    conflict_tokens = {token for token in re.findall(r"[a-z0-9]+", haystack) if len(token) > 2}
    overlap = query_tokens & conflict_tokens
    return len(overlap) if len(overlap) >= 2 else 0


def _query_contains_term(query: str, term: str) -> bool:
    """Match policy terms on token boundaries so reference suffixes cannot alias."""
    if not term:
        return False
    return re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", query) is not None


def _is_legal_reference_term(store, term: str) -> bool:
    try:
        parsed = parse_legal_reference(
            store.config.corpus_id,
            term,
            allow_roman_pasal=True,
            config=store.config,
        )
    except ValueError:
        return False
    return any(parsed.values())


def _source_document_meta(store, source_document_id: object) -> dict:
    return next(
        (row for row in store.source_documents if row.get("source_document_id") == source_document_id),
        {},
    )


def _meaningful_conflict_tokens(text: str, intent: dict) -> set[str]:
    generic = set(intent.get("generic_tokens") or ())
    return {token for token in re.findall(r"[a-z0-9]+", text) if len(token) > 2 and token not in generic}


def _source_conflict_intent(store) -> dict:
    return store.config.setting("source_conflict_intent", {})


def _public_source_conflict(conflict: dict) -> dict:
    return {
        "source_conflict_id": conflict.get("source_conflict_id"),
        "type": conflict.get("type"),
        "classification": conflict.get("classification"),
        "source_document_id": conflict.get("source_document_id"),
        "status": conflict.get("status"),
    } | _source_conflict_contract_fields(conflict)


def _source_conflict_support(store, conflict: dict) -> dict:
    evidence_rows = tuple(
        row | {"route_sources": ("exact",), "candidate_type": "source_conflict_provenance"}
        for evidence_id in conflict.get("evidence_ids") or ()
        if (row := store.get(evidence_id)) is not None
    )
    context_pack = assemble_context_pack(store, evidence_rows) if evidence_rows else empty_context_pack("source_anomaly")
    evidence = context_pack["answer_evidence"]
    synthetic_support = _synthetic_source_conflict_support(store, conflict) if not evidence else None
    trace_support: tuple[dict, ...] = ()
    answer_scope = "insufficient_evidence"
    if evidence:
        trace_support = tuple(_citation_with_authority(store, row, conflict=conflict) for row in evidence)
        answer_scope = "source_conflict_exact_provenance"
    elif synthetic_support is not None:
        trace_support = tuple(synthetic_support["citations"])
        answer_scope = "source_conflict_exact_provenance"
    else:
        trace_support = (_source_conflict_trace_support(store, conflict, context_pack["validation_reasons"]),)
        answer_scope = "source_conflict_trace" if trace_support else "insufficient_evidence"
    viewer_refs = (
        tuple(
            ref | _authority_policy(store, evidence[0], can_resolve=ref.get("can_resolve") is True, conflict=conflict)
            for ref in context_pack["viewer_refs"]
        )
        if evidence
        else synthetic_support["viewer_refs"]
        if synthetic_support is not None
        else ()
    )
    citations = ()
    public_context_pack = (
        context_pack | {"citation_payloads": (), "viewer_refs": ()}
        if evidence
        else synthetic_support["context_pack"]
        if synthetic_support is not None
        else context_pack
    )
    return {
        "context_pack": public_context_pack,
        "evidence": synthetic_support["evidence"] if synthetic_support is not None else evidence,
        "citations": citations,
        "viewer_refs": viewer_refs,
        "trace_support": trace_support,
        "answer_scope": answer_scope,
        "warnings": (("source_conflict_not_final_legal_authority",) if evidence or synthetic_support is not None or trace_support else ()),
    }


def _source_conflict_trace_support(store, conflict: dict, validation_reasons: dict) -> dict:
    first_reason = next(iter(validation_reasons.values()), None)
    authority = _authority_policy(store, conflict, can_resolve=False, conflict=conflict)
    return (
        {
            "support_class": authority.get("support_kind") or "source_conflict_trace",
            "evidence_id": conflict.get("source_conflict_id"),
            "source_conflict_id": conflict.get("source_conflict_id"),
            "type": conflict.get("type"),
            "classification": conflict.get("classification"),
            "source_document_id": conflict.get("source_document_id"),
            "page_numbers": tuple(conflict.get("page_numbers") or conflict.get("affected_pages") or ()),
            "text_span_ids": tuple(conflict.get("text_span_ids") or ()),
            "evidence_ids": tuple(conflict.get("evidence_ids") or ()),
            "bbox_ids": tuple(conflict.get("bbox_ids") or ()),
            "bbox_count": len(conflict.get("raw_provenance_bbox_ids") or ()),
            "citation_available": False,
            "viewer_highlightable": False,
            "viewer_ref": None,
            "failure_reason": conflict.get("failure_reason") or first_reason or "source_conflict_trace_only",
        }
        | _source_conflict_contract_fields(conflict)
        | authority
    )


def _source_anomaly_answer(store, conflict: dict, query: str, *, exact_provenance: bool, trace_only: bool) -> str:
    intent = _source_conflict_intent(store)
    decision = conflict.get("resolution_decision") or {}
    folded = (query or "").casefold()
    classification = conflict.get("classification") or "source_conflict_recorded"
    summary = str(conflict.get("provenance_summary") or classification).strip()
    authority_policy = str(
        conflict.get("final_authority_policy")
        or "Sistem menampilkan provenance sumber ini sebagai jejak audit, bukan kesimpulan hukum final."
    ).strip()
    role_label = _source_conflict_role_label(store, conflict)
    reviewer_suffix = _source_conflict_reviewer_suffix(decision.get("reviewer_decision"))
    policy = conflict.get("source_anomaly_policy") or {}
    if exact_provenance:
        provenance_note = _source_conflict_provenance_note(conflict)
        return _source_anomaly_policy_answer(
            policy,
            role_label=role_label,
            summary=summary,
            authority_policy=authority_policy,
            provenance_note=provenance_note,
            reviewer_suffix=reviewer_suffix,
        )
    if trace_only:
        return _source_anomaly_policy_answer(
            policy,
            role_label=role_label,
            summary=summary,
            authority_policy=authority_policy,
            provenance_note="Jejak sumber tersedia, tetapi belum memenuhi syarat sitasi atau highlight exact.",
            reviewer_suffix=reviewer_suffix,
        )
    values = {
        "classification": classification,
        "reviewer_decision": decision.get("reviewer_decision") or "Reviewer decision unavailable",
    }
    for rule in intent.get("answer_rules") or ():
        terms = tuple(str(term).casefold() for term in rule.get("query_terms") or ())
        types = tuple(str(item) for item in rule.get("conflict_types") or ())
        if (terms and any(term in folded for term in terms)) or (types and conflict.get("type") in types):
            return str(rule.get("template") or "").format_map(values)
    return str(intent.get("default_answer_template") or "").format_map(values)


def _source_anomaly_policy_answer(
    policy: dict,
    *,
    role_label: str,
    summary: str,
    authority_policy: str,
    provenance_note: str,
    reviewer_suffix: str,
) -> str:
    values = {
        "anomaly_kind": policy.get("anomaly_kind") or "source_anomaly_provenance",
        "mapping_kind": policy.get("mapping_kind") or "source_anomaly_provenance",
        "role_label": role_label,
        "summary": summary,
        "authority_policy": authority_policy,
        "provenance_note": provenance_note,
        "reviewer_suffix": reviewer_suffix,
    }
    template = str(
        policy.get("public_wording_template")
        or "Catatan provenance sumber ({anomaly_kind}) pada {role_label}: {summary}. {authority_policy} {provenance_note}{reviewer_suffix}"
    )
    return template.format_map(values)


def _source_conflict_role_label(store, conflict: dict) -> str:
    source = _source_document_meta(store, conflict.get("source_document_id"))
    source_role = str(source.get("source_role") or "")
    labels = _source_conflict_intent(store).get("role_labels") or {}
    return str(labels.get(source_role) or source_role or conflict.get("source_document_id") or "sumber historis")


def _source_conflict_reviewer_suffix(reviewer_decision: object) -> str:
    text = str(reviewer_decision or "").strip()
    if not text:
        return ""
    return f" Reviewer decision: {text}."


def _source_conflict_contract_fields(conflict: dict) -> dict:
    raw_bbox_ids = tuple(conflict.get("raw_provenance_bbox_ids") or ())
    raw_text_span_ids = tuple(conflict.get("raw_provenance_text_span_ids") or ())
    blocked_text_span_ids = tuple(conflict.get("blocked_raw_provenance_text_span_ids") or ())
    fields = {
        "final_evidence_available": bool(conflict.get("final_evidence_available")),
        "source_anomaly_kind": conflict.get("source_anomaly_kind"),
        "source_mapping_kind": conflict.get("source_mapping_kind"),
        "provenance_bbox_status": conflict.get("provenance_bbox_status"),
        "provenance_highlight_scope": conflict.get("provenance_highlight_scope"),
        "raw_provenance_bbox_count": len(raw_bbox_ids),
        "raw_provenance_text_span_count": len(raw_text_span_ids),
        "blocked_raw_provenance_text_span_count": len(blocked_text_span_ids),
        "blocked_raw_provenance_reason": conflict.get("blocked_raw_provenance_reason"),
    }
    fields.update(_source_mapping_semantics(conflict))
    return fields


def _source_conflict_provenance_note(conflict: dict) -> str:
    if conflict.get("provenance_highlight_scope") == "all_relevant_spans":
        return "Highlight viewer tersedia untuk semua span provenance exact yang relevan."
    if conflict.get("provenance_highlight_scope") == "anchor_span_only":
        return "Highlight viewer saat ini terbatas pada span anchor exact yang tersedia; span anomali lain tetap tercatat sebagai trace tanpa highlight palsu."
    return "Viewer tidak menampilkan highlight exact karena belum ada span/BBox provenance yang aman."


def _source_conflict_viewer_evidence(store, evidence_id: str | None) -> tuple[dict | None, list[dict] | None]:
    if store is None or not evidence_id:
        return None, None
    conflict = next((row for row in store.source_conflicts if row.get("source_conflict_id") == evidence_id), None)
    if conflict is None:
        return None, None
    synthetic = _synthetic_source_conflict_support(store, conflict)
    if synthetic is None:
        raw_bboxes = tuple(store.bboxes_for_refs(tuple(conflict.get("raw_provenance_bbox_ids") or ())))
        if raw_bboxes:
            synthetic = {
                "evidence": (_synthetic_source_conflict_evidence(store, conflict, raw_bboxes),),
                "bboxes": raw_bboxes,
            }
    if synthetic is None:
        return None, None
    return synthetic["evidence"][0], list(synthetic["bboxes"])


def _synthetic_source_conflict_support(store, conflict: dict) -> dict | None:
    bboxes = tuple(store.exact_bboxes_for_text_spans(tuple(conflict.get("text_span_ids") or ())))
    if not bboxes:
        return None
    evidence = _synthetic_source_conflict_evidence(store, conflict, bboxes)
    viewer_ref = {
        "action": "viewer",
        "evidence_id": evidence["evidence_id"],
        "source_document_id": evidence.get("source_document_id"),
        "page_numbers": evidence["page_numbers"],
        "bbox_count": len(bboxes),
        "can_resolve": True,
    } | _authority_policy(store, evidence, can_resolve=True, conflict=conflict)
    citation = evidence | {
        "label": (_source_conflict_intent(store).get("public_labels") or {}).get(
            conflict.get("classification"), conflict.get("classification")
        ),
        "document_title": _document_title(store, _source_document_meta(store, conflict.get("source_document_id"))),
        "viewer_ref": viewer_ref,
        "bbox_count": len(bboxes),
        "evidence_status": conflict.get("status"),
    }
    return {
        "context_pack": {
            "answer_evidence": (evidence,),
            "supporting_context": (),
            "excluded_results": (),
            "citation_payloads": (),
            "viewer_refs": (viewer_ref,),
            "validation_reasons": {evidence["evidence_id"]: "source_conflict_exact_span_bbox"},
        },
        "evidence": (evidence,),
        "citations": (_citation_with_authority(store, citation, conflict=conflict),),
        "viewer_refs": (viewer_ref,),
        "bboxes": bboxes,
    }


def _attach_source_reference_provenance(store, query: str, response: dict) -> dict:
    """Expose an explicitly requested printed occurrence as non-final trace."""
    mappings = source_reference_mappings_for_query(query, store.config)
    if not mappings:
        return response
    for mapping in mappings:
        conflict_id = str(mapping.get("provenance") or "")
        conflict = next(
            (row for row in store.source_conflicts if row.get("source_conflict_id") == conflict_id),
            None,
        )
        synthetic = _source_reference_synthetic_support(store, conflict) if conflict is not None else None
        if synthetic is None:
            continue
        citation = synthetic["citations"][0] | {
            "source_reference_mapping": mapping.get("mapping_kind"),
            "printed_reference": mapping.get("raw_reference"),
            "canonical_reference": _public_canonical_reference(mapping.get("canonical_target"), mapping.get("context_terms"), query),
            "citation_final": False,
            "authority_kind": "source_anomaly",
        }
        viewer_ref = synthetic["viewer_refs"][0]
        trace_support = tuple(response.get("trace_support") or ()) + (citation,)
        viewer_refs = tuple(response.get("viewer_refs") or ()) + (viewer_ref,)
        context_pack = response.get("context_pack") or empty_context_pack("source_reference_provenance")
        context_pack = context_pack | {
            "trace_support": tuple(context_pack.get("trace_support") or ()) + (citation,),
            "viewer_refs": tuple(context_pack.get("viewer_refs") or ()) + (viewer_ref,),
        }
        return response | {
            "context_pack": context_pack,
            "trace_support": trace_support,
            "viewer_refs": viewer_refs,
            "source_reference_provenance": (citation,),
            "warnings": tuple(dict.fromkeys((*response.get("warnings", ()), "source_reference_mapping_not_final_authority"))),
        }
    return response


def _public_canonical_reference(value: object, context_terms: object, query: str) -> str | None:
    """Use a section-qualified label unless the query names that section explicitly."""
    text = str(value or "").strip()
    marker = text.casefold().find("pasal ")
    terms = context_terms if isinstance(context_terms, (list, tuple)) else ()
    context = tuple(option.strip() for term in terms for option in str(term).split("|") if option.strip())
    explicit_section = any(" " in term and re.search(rf"(?<!\w){re.escape(term)}(?!\w)", query or "", re.IGNORECASE) for term in context)
    return text[marker:] if marker >= 0 and explicit_section else (text or None)


def _source_reference_synthetic_support(store, conflict: dict) -> dict | None:
    """Build query-scoped provenance from the configured raw occurrence BBoxes."""
    synthetic = _synthetic_source_conflict_support(store, conflict)
    if synthetic is not None:
        return synthetic
    bboxes = tuple(store.bboxes_for_refs(tuple(conflict.get("raw_provenance_bbox_ids") or ())))
    if not bboxes:
        return None
    evidence = _synthetic_source_conflict_evidence(store, conflict, bboxes)
    viewer_ref = {
        "action": "viewer",
        "evidence_id": evidence["evidence_id"],
        "source_document_id": evidence.get("source_document_id"),
        "page_numbers": evidence["page_numbers"],
        "bbox_count": len(bboxes),
        "can_resolve": True,
    } | _authority_policy(store, evidence, can_resolve=True, conflict=conflict)
    citation = evidence | {
        "label": (_source_conflict_intent(store).get("public_labels") or {}).get(
            conflict.get("classification"), conflict.get("classification")
        ),
        "document_title": _document_title(store, _source_document_meta(store, conflict.get("source_document_id"))),
        "viewer_ref": viewer_ref,
        "bbox_count": len(bboxes),
        "evidence_status": conflict.get("status"),
    }
    return {
        "context_pack": {
            "answer_evidence": (evidence,),
            "supporting_context": (),
            "excluded_results": (),
            "final_citations": (),
            "historical_citations": (),
            "metadata_support": (),
            "structural_support": (),
            "trace_support": (),
            "citation_payloads": (),
            "viewer_refs": (viewer_ref,),
            "validation_reasons": {evidence["evidence_id"]: "source_conflict_raw_provenance_bbox"},
        },
        "evidence": (evidence,),
        "citations": (_citation_with_authority(store, citation, conflict=conflict),),
        "viewer_refs": (viewer_ref,),
        "bboxes": bboxes,
    }


def _synthetic_source_conflict_evidence(store, conflict: dict, bboxes: tuple[dict, ...]) -> dict:
    source = _source_document_meta(store, conflict.get("source_document_id"))
    pages = tuple(dict.fromkeys(int(row["page_number"]) for row in bboxes if row.get("page_number")))
    quoted_text = "\n".join(dict.fromkeys(str(row.get("text") or "").strip() for row in bboxes if str(row.get("text") or "").strip()))
    return {
        "evidence_id": conflict.get("source_conflict_id"),
        "legal_unit_id": None,
        "citation": f"Source anomaly: {conflict.get('classification')}",
        "quoted_text": quoted_text,
        "bbox_refs": tuple(row["bbox_id"] for row in bboxes if row.get("bbox_id")),
        "bbox_precision": "exact",
        "viewer_highlightable": True,
        "page_numbers": pages,
        "source_document_id": conflict.get("source_document_id"),
        "source_pdf_path": bboxes[0].get("source_pdf_path"),
        "source_sha256": bboxes[0].get("source_sha256"),
        "source_role": source.get("source_role"),
        "temporal_context": source.get("temporal_context"),
    }
