"""One deterministic projection for the common public answer envelope."""

from __future__ import annotations

from dataclasses import dataclass
import re
from collections.abc import Callable
from typing import Any

from tjipto.retrieval.answer import assemble_context_pack, empty_context_pack
from tjipto.retrieval.sufficiency import EvidenceRequirement
from tjipto.runtime.answer_arbitration import ANSWER_TEMPLATES, _answer_templates, _natural_label_sort_key
from tjipto.runtime.viewer import _citation_with_authority, _public_evidence_row


@dataclass(frozen=True)
class AnswerDecision:
    status: str
    route: str
    answer_type: str
    answer: str
    context_pack: dict[str, Any]
    evidence: tuple[dict, ...] = ()
    citations: tuple[dict, ...] = ()
    final_citations: tuple[dict, ...] = ()
    historical_citations: tuple[dict, ...] = ()
    metadata_support: tuple[dict, ...] = ()
    structural_support: tuple[dict, ...] = ()
    trace_support: tuple[dict, ...] = ()
    viewer_refs: tuple[dict, ...] = ()
    metadata_facts: tuple[dict, ...] = ()
    legal_relations: tuple[dict, ...] = ()
    answer_scope: str = "insufficient_evidence"
    warnings: tuple[str, ...] = ()
    insufficient_reasons: tuple[str, ...] = ()
    reason_code: str | None = None
    claim_support: tuple[dict, ...] = ()
    document_relations: tuple[dict, ...] | None = None
    article_amendment_relations: tuple[dict, ...] | None = None
    relation_support: tuple[dict, ...] | None = None


def project_response(base: dict[str, Any], decision: AnswerDecision) -> dict[str, Any]:
    """Preserve route diagnostics while projecting one complete answer shape."""
    response = base | decision.__dict__
    if decision.reason_code is None:
        response.pop("reason_code", None)
    for key in ("document_relations", "article_amendment_relations", "relation_support"):
        if response[key] is None:
            response.pop(key)
    return response


def instrument_response(
    store,
    corpus_id: str,
    query: str,
    semantics,
    instrument: tuple[dict | None, str, str],
    answer_text: Callable[[str, tuple[dict, ...], dict[str, str]], str],
) -> dict:
    """Project an instrument lookup through the common public envelope."""
    row, route, reason = instrument
    templates = _answer_templates(store)
    base = {
        "route": route,
        "intent": "instrument_unit_lookup",
        "corpus_id": corpus_id,
        "original_query": query,
        "normalized_query": query.strip(),
        "matches": (row,) if row is not None else (),
        "reason": reason if row is None else None,
    }
    context_pack = assemble_context_pack(store, (row,)) if row is not None else empty_context_pack(reason)
    evidence = context_pack["answer_evidence"]
    if not evidence:
        return project_response(
            base,
            AnswerDecision(
                "insufficient_evidence",
                route,
                "none",
                templates["insufficient"],
                context_pack,
                historical_citations=context_pack.get("historical_citations", ()),
                metadata_support=context_pack.get("metadata_support", ()),
                structural_support=context_pack.get("structural_support", ()),
                trace_support=context_pack.get("trace_support", ()),
                insufficient_reasons=(reason,),
            ),
        )
    public_evidence = tuple(_public_evidence_row(store, item) for item in evidence)
    historical_citations = (
        tuple(_citation_with_authority(store, item) for item in context_pack.get("trace_support", ()))
        if semantics.operation == "summarize"
        else ()
    )
    citations = tuple(_citation_with_authority(store, item) for item in context_pack["citation_payloads"])
    status = "answer_ready" if citations or historical_citations else "limited_answer"
    return project_response(
        base,
        AnswerDecision(
            status,
            route,
            "quoted_evidence",
            answer_text(status, public_evidence, templates),
            context_pack,
            evidence=public_evidence,
            citations=citations,
            final_citations=citations,
            historical_citations=historical_citations,
            metadata_support=tuple(_citation_with_authority(store, item) for item in context_pack.get("metadata_support", ())),
            structural_support=tuple(_citation_with_authority(store, item) for item in context_pack.get("structural_support", ())),
            trace_support=tuple(_citation_with_authority(store, item) for item in context_pack.get("trace_support", ())),
            viewer_refs=context_pack["viewer_refs"] if citations else (),
            answer_scope="direct_evidence" if status == "answer_ready" else "limited_evidence",
        ),
    )


def structural_response(corpus_id: str, query: str, semantics, aggregate: dict) -> dict:
    """Project one verified structural aggregate through the public envelope."""
    source_supports = tuple(aggregate.get("source_supports") or (aggregate,))
    context_pack = empty_context_pack("structural_aggregate") | {
        "answer_evidence": (aggregate,),
        "structural_support": source_supports,
    }
    return project_response(
        {
            "route": "structure_count",
            "intent": "structured_lookup",
            "corpus_id": corpus_id,
            "original_query": query,
            "normalized_query": query.strip(),
            "operation": semantics.operation,
            "source_scopes": semantics.source_scopes,
            "temporal_scope": semantics.temporal_scope,
            "matches": (aggregate,),
        },
        AnswerDecision(
            "answer_ready",
            "structure_count",
            "structural_aggregate",
            aggregate["display_text"],
            context_pack,
            evidence=(aggregate,),
            structural_support=source_supports,
            answer_scope="deterministic_structure",
        ),
    )
def compose_research_answer(
    evidence: tuple[dict, ...],
    evidence_set,
    requirements: tuple[EvidenceRequirement, ...],
    assessment,
    *,
    preferred_source_role: str | None = None,
) -> str:
    """Compose only requirement-assigned, source-backed findings."""
    by_id = {str(row.get("evidence_id")): row for row in evidence}
    requirement_by_id = {row.requirement_id: row for row in requirements}
    findings: list[str] = []
    grouped_findings: dict[str, list[str]] = {}
    for requirement_id, support_ids in evidence_set.assignments:
        requirement = requirement_by_id.get(requirement_id)
        heading = requirement.description if requirement and requirement.description else requirement_id.replace("_", " ")
        for row in (by_id[support_id] for support_id in support_ids if support_id in by_id):
            quote = " ".join(str(row.get("quoted_text") or row.get("display_text") or "").split())
            citation = str(row.get("citation") or "").strip()
            if citation and quote.casefold().startswith(citation.casefold()):
                quote = quote[len(citation) :].lstrip(" :.-")
            if requirement_id.startswith("instrument_"):
                scope = re.search(r"\bmengubah\b.*?(?=\s+sehingga\s+selengkapnya)", quote, re.IGNORECASE)
                quote = scope.group(0).rstrip(" .:") if scope else quote
            if requirement_id.startswith("source_occurrence_"):
                document = _source_occurrence_document(requirement, heading)
                provision = _provision_reference(row, citation)
                location = f" pada {provision}" if provision else ""
                findings.append(f"Dalam {document}, ketentuan tersebut tercantum{location}: {quote}")
            else:
                # Several analysis requirements intentionally share one
                # configured heading (issue provisions).  Emit that heading
                # once while retaining every source-backed quote, avoiding a
                # repetitive template without inventing connective text.
                grouped_findings.setdefault(heading, []).append(quote)
    findings.extend(f"{heading}: {' '.join(quotes)}" for heading, quotes in grouped_findings.items())
    roles = tuple(dict.fromkeys(str(row.get("source_role") or "") for row in evidence))
    qualification = (
        "Sumber yang digunakan merupakan naskah historis, bukan naskah konsolidasi yang berlaku saat ini."
        if preferred_source_role and roles and all(role != preferred_source_role for role in roles)
        else ""
    )
    missing = tuple(
        requirement_by_id[requirement_id].description
        for requirement_id in (assessment.missing_requirement_ids if assessment is not None else ())
        if requirement_id in requirement_by_id
    )
    missing_ids = assessment.missing_requirement_ids if assessment is not None else ()
    if missing and all(requirement_id.startswith("source_occurrence_") for requirement_id in missing_ids):
        documents = tuple(
            _source_occurrence_document(requirement_by_id.get(requirement_id), requirement_id)
            for requirement_id in missing_ids
            if requirement_id in requirement_by_id
        )
        limitation = f"Belum ditemukan dukungan terverifikasi pada {', '.join(documents)}."
    else:
        limitation = f"Keterbatasan: dukungan untuk {', '.join(missing)} belum terverifikasi." if missing else ""
    paragraphs = tuple((*findings, qualification, limitation))
    return "\n\n".join(paragraph for paragraph in paragraphs if paragraph)


def _provision_reference(row: dict, fallback: str) -> str:
    hierarchy = tuple(str(value) for value in row.get("hierarchy") or () if value)
    for index, value in enumerate(hierarchy):
        if value.casefold().startswith("pasal "):
            return " / ".join(hierarchy[index:])
    return fallback


def _source_occurrence_document(requirement: EvidenceRequirement | None, fallback: str) -> str:
    if requirement is None:
        return fallback
    description = requirement.description or fallback
    prefix = f"{requirement.retrieval_query} dalam " if requirement.retrieval_query else ""
    return description[len(prefix) :] if prefix and description.startswith(prefix) else description


def _relation_response(store, routed: dict) -> dict:
    templates = _answer_templates(store)
    target = routed.get("relation_target") or {"mode": None}
    graph_edges = tuple(routed.get("matches") or ())
    if target["mode"] == "article":
        return _project_article_relation(
            store, routed, target, templates, _relation_support(graph_edges, "article_amendment_relation_graph")
        )
    if target["mode"] == "unsupported":
        return _relation_not_promoted(routed, templates)
    support = _relation_support(graph_edges, "document_relation_graph")
    if not support:
        reason = "document_relation_not_found"
        return project_response(
            routed | {"matches": (), "reason": reason},
            AnswerDecision(
                "insufficient_evidence",
                "document_relation",
                "none",
                templates["insufficient"],
                empty_context_pack(reason),
                document_relations=(),
                insufficient_reasons=(reason,),
            ),
        )
    relations = tuple(_public_document_relation(row) for row in support)
    # Document-level graph edges are provenance traces, not publishable legal
    # support. Keep the relation available for audit/UI context but never
    # promote a trace-only result to an answer-ready publication.
    return project_response(
        routed | {"matches": support},
        AnswerDecision(
            "limited_answer",
            "document_relation",
            "document_relation",
            _document_relation_answer(store, relations),
            empty_context_pack("document_relation_source_role_trace"),
            document_relations=relations,
            trace_support=relations,
            answer_scope="source_role_document_relation",
            warnings=("document_relation_not_exact_highlightable", "document_relation_trace_only"),
        ),
    )


def _project_article_relation(store, routed: dict, target: dict, templates: dict[str, str], support: tuple[dict, ...]) -> dict:
    if not support:
        if target.get("target_citation"):
            return _relation_not_promoted(routed, templates, reason="relation_target_not_found")
        return _relation_not_promoted(routed, templates)
    exact_support = tuple(row for row in support if _is_exact_article_relation(row))
    exact_targets = {row.get("target_legal_unit_id") for row in exact_support}
    trace_support = tuple(
        row for row in support if not _is_exact_article_relation(row) and row.get("target_legal_unit_id") not in exact_targets
    )
    requested_targets = {_normalize_relation_citation(value) for value in target.get("target_citations") or () if value}
    exact_citations = {_normalize_relation_citation(row.get("target_citation") or row.get("target_reference")) for row in exact_support}
    if requested_targets - exact_citations:
        trace_support = tuple(
            row
            for row in support
            if _normalize_relation_citation(row.get("target_citation") or row.get("target_reference"))
            in requested_targets - exact_citations
        )
    # Exact source relations are the publishable article targets. Trace rows
    # remain available for the trace-only path, but must not be projected as
    # neighboring article answers when an exact target already satisfies the
    # request.
    public_relation_rows = exact_support if exact_support else trace_support
    if exact_support and not requested_targets and any(row.get("relation_type") == "RENAMES" for row in exact_support):
        public_relation_rows = (*exact_support, *trace_support)
    public_relations = tuple(public_article_relation(row) for row in public_relation_rows)
    answer_evidence = tuple(
        evidence
        for relation in exact_support
        for evidence in (
            _article_relation_evidence(store, relation),
            _article_relation_target_evidence(store, relation),
        )
        if evidence is not None
    )
    if not answer_evidence:
        if not trace_support:
            return _relation_not_promoted(routed, templates)
        return project_response(
            routed | {"matches": support, "reason": "relation_trace_only"},
            AnswerDecision(
                "limited_answer",
                "document_relation",
                "article_amendment_relation",
                _article_relation_answer(store, (), trace_support),
                empty_context_pack("relation_trace_only"),
                article_amendment_relations=public_relations,
                relation_support=(),
                trace_support=tuple(public_article_relation(row) for row in trace_support),
                answer_scope="trace_article_relation",
                warnings=("article_relation_trace_only_not_citable",),
            ),
        )
    citations = _deduplicated_article_relation_citations(store, answer_evidence)
    final_citations = tuple(row for row in citations if row.get("citation_final") is True)
    historical_citations = tuple(row for row in citations if row.get("citation_final") is False)
    viewer_refs = tuple(row["viewer_ref"] for row in answer_evidence if row.get("citation_final") is True)
    public_trace_support = trace_support
    partial = bool(public_trace_support)
    public_evidence = answer_evidence
    context_pack = {
        "answer_evidence": public_evidence,
        "supporting_context": (),
        "excluded_results": (),
        "citation_payloads": final_citations,
        "historical_citations": historical_citations,
        "viewer_refs": viewer_refs,
        "validation_reasons": {row["evidence_id"]: "article_amendment_relation_exact_source_text" for row in public_evidence},
    }
    return project_response(
        routed | {"matches": support},
        AnswerDecision(
            "limited_answer" if partial else "answer_ready",
            "document_relation",
            "article_amendment_relation",
            _article_relation_answer(store, exact_support, public_trace_support),
            context_pack,
            evidence=public_evidence,
            citations=final_citations,
            final_citations=final_citations,
            historical_citations=historical_citations,
            viewer_refs=viewer_refs,
            article_amendment_relations=public_relations,
            relation_support=answer_evidence,
            trace_support=tuple(public_article_relation(row) for row in public_trace_support),
            answer_scope="partial_exact_article_relation" if partial else "exact_article_relation",
            warnings=("article_relation_exact_support_partial_trace_omitted",) if public_trace_support else (),
        ),
    )


def _normalize_relation_citation(value: object) -> str:
    return " ".join(str(value or "").casefold().replace("(", "").replace(")", "").split())


def _relation_not_promoted(routed: dict, templates: dict[str, str], *, reason: str = "relation_not_promoted") -> dict:
    return project_response(
        routed | {"matches": (), "reason": reason},
        AnswerDecision(
            "insufficient_evidence",
            "document_relation",
            "none",
            templates["insufficient"],
            empty_context_pack(reason),
            document_relations=(),
            article_amendment_relations=(),
            insufficient_reasons=(reason,),
        ),
    )


def _relation_support(graph_edges: tuple[dict, ...], route_source: str) -> tuple[dict, ...]:
    return tuple(
        edge["relation_projection"] | {"route_sources": edge.get("route_sources") or (route_source,)}
        for edge in graph_edges
        if edge.get("relation_projection")
    )


def _article_relation_evidence(store, relation: dict) -> dict | None:
    if not _is_exact_article_relation(relation):
        return None
    row = store.get(relation["evidence_id"])
    if row is None:
        return None
    if store.lineage_error(row):
        return None
    proof_bbox_refs = tuple(relation.get("bbox_refs") or row.get("bbox_refs") or ())
    proof_text_span_ids = tuple(relation.get("text_span_ids") or row.get("text_span_ids") or ())
    source_quote = _source_quote_for_spans(store, proof_text_span_ids)
    target_bbox_refs = tuple(relation.get("target_bbox_refs") or ())
    proof_bboxes = store.bboxes_for_refs(proof_bbox_refs)
    if not proof_bboxes or not set(proof_bbox_refs) <= {bbox["bbox_id"] for bbox in proof_bboxes}:
        return None
    if row.get("bbox_precision") != "exact" or row.get("viewer_highlightable") is not True:
        return None
    relation_final = relation.get("citation_final") is True
    relation_citable = relation_final and relation.get("citation_available") is True
    return {
        **row,
        "relation_id": relation.get("relation_id"),
        "support_kind": "article_relation",
        "fact_kind": "article_relation",
        "display_label": relation.get("target_citation") or relation.get("target_label") or relation.get("relation_type") or "Relasi hukum",
        "display_text": source_quote or relation.get("quoted_text") or row.get("quoted_text"),
        "bbox_refs": proof_bbox_refs,
        "text_span_ids": proof_text_span_ids,
        "relation_source_proof_bbox_refs": proof_bbox_refs,
        "relation_source_proof_text_span_ids": proof_text_span_ids,
        "relation_target_bbox_refs": target_bbox_refs,
        "relation_target_text_span_ids": tuple(relation.get("target_text_span_ids") or ()),
        "quoted_text": source_quote or relation.get("quoted_text") or row.get("quoted_text"),
        "bbox_count": len(proof_bboxes),
        "citation_final": relation_final,
        "citable": relation_citable,
        "citable_status": "citable_exact" if relation_citable else row.get("citable_status"),
        "citation_eligibility": "eligible" if relation_citable else row.get("citation_eligibility"),
        "route_sources": ("article_amendment_relation",),
        "article_amendment_relation": relation,
        "viewer_ref": {
            "action": "viewer",
            "evidence_id": row["evidence_id"],
            "relation_id": relation.get("relation_id"),
            "source_document_id": row.get("source_document_id"),
            "page_numbers": tuple(row.get("page_numbers") or ()),
            "text_span_ids": proof_text_span_ids,
            "bbox_count": len(proof_bboxes),
            "bbox_refs": proof_bbox_refs,
            "source_proof_text_span_ids": proof_text_span_ids,
            "source_proof_bbox_refs": proof_bbox_refs,
            "target_text_span_ids": tuple(relation.get("target_text_span_ids") or ()),
            "target_bbox_refs": target_bbox_refs,
            "can_resolve": True,
        },
    }


def _source_quote_for_spans(store, text_span_ids: tuple[str, ...]) -> str:
    by_id = {span.get("text_span_id"): span for span in store.page_text_spans}
    quotes = [str(by_id[span_id].get("exact_quote") or "") for span_id in text_span_ids if span_id in by_id]
    return "\n".join(quote for quote in quotes if quote)


def _article_relation_target_evidence(store, relation: dict) -> dict | None:
    """Return the versioned normative target as its own citable evidence."""
    target_unit_id = str(relation.get("target_legal_unit_id") or "")
    target_spans = set(relation.get("target_text_span_ids") or ())
    target_role = str(relation.get("target_source_role") or "")
    if not target_unit_id or not target_spans:
        return None
    candidate = next(
        (
            row
            for row in store.evidence
            if row.get("legal_unit_id") == target_unit_id
            and (not target_role or row.get("source_role") == target_role)
            and row.get("citation_final") is True
            and row.get("bbox_precision") == "exact"
            and row.get("viewer_highlightable") is True
            and target_spans.intersection(row.get("text_span_ids") or ())
        ),
        None,
    )
    if candidate is None:
        return None
    target_bbox_refs = tuple(candidate.get("bbox_refs") or ())
    target_viewer_ref = {
        "action": "viewer",
        "evidence_id": candidate.get("evidence_id"),
        "source_document_id": candidate.get("source_document_id"),
        "page_numbers": tuple(candidate.get("page_numbers") or ()),
        "text_span_ids": tuple(candidate.get("text_span_ids") or ()),
        "bbox_refs": target_bbox_refs,
        "bbox_count": len(target_bbox_refs),
        "can_resolve": bool(target_bbox_refs),
    }
    return candidate | {
        "relation_id": relation.get("relation_id"),
        "support_kind": "article_relation_target",
        "fact_kind": "article_relation_target",
        "route_sources": ("article_relation_target",),
        "article_amendment_relation": relation,
        "display_label": relation.get("target_citation") or relation.get("target_label") or candidate.get("citation"),
        "display_text": _source_quote_for_spans(store, tuple(target_spans)) or candidate.get("quoted_text"),
        "viewer_ref": target_viewer_ref,
    }


def _is_exact_article_relation(row: dict) -> bool:
    if not (
        row.get("support_class") == "exact_article_relation"
        and row.get("grounding_level") == "exact_source_text"
        and row.get("bbox_precision") == "exact"
        and row.get("viewer_highlightable") is True
        and row.get("citation_available") is True
    ):
        return False
    relation_type = str(row.get("relation_type") or "")
    if relation_type in {"ADDS", "MODIFIES"}:
        if (
            not row.get("successor_legal_unit_id")
            or not row.get("successor_text_span_ids")
            or not row.get("target_text_span_ids")
            or row.get("comparison_basis") != "versioned_normative_text"
        ):
            return False
        if relation_type == "MODIFIES" and not row.get("predecessor_legal_unit_id"):
            return False
    return True


def _public_document_relation(row: dict) -> dict:
    return {
        "relation_id": row.get("relation_id"),
        "relation_type": row.get("relation_type"),
        "operation_candidates": tuple(row.get("operation_candidates") or ()),
        "source_document_id": row.get("source_document_id"),
        "source_role": row.get("source_role"),
        "target_source_role": row.get("target_source_role"),
        "target_document_id": row.get("target_document_id"),
        "support_type": row.get("support_type"),
        "reason": row.get("reason"),
        "highlightable": row.get("viewer_highlightable") is True,
    }


def public_article_relation(row: dict) -> dict:
    inverse = row.get("projection_direction") == "inverse"
    return {
        "relation_id": row.get("relation_id"),
        "relation_type": row.get("relation_type"),
        "operation_candidates": tuple(row.get("operation_candidates") or ()),
        "source_document_id": row.get("support_document_id") or row.get("source_document_id"),
        "source_role": row.get("support_source_role") or row.get("source_role"),
        "source_legal_unit_id": row.get("source_legal_unit_id"),
        "source_legal_unit_role": row.get("source_legal_unit_role"),
        "source_label": row.get("source_label"),
        "source_reference": row.get("new_reference" if inverse else "old_reference") or row.get("source_reference"),
        "source_reference_range": row.get("new_reference_range" if inverse else "old_reference_range") or row.get("source_reference_range"),
        "source_reference_range_kind": row.get("new_reference_range_kind" if inverse else "old_reference_range_kind")
        or row.get("source_reference_range_kind"),
        "target_legal_unit_id": row.get("target_legal_unit_id"),
        "target_label": row.get("target_label") or row.get("target_citation"),
        "target_citation": row.get("target_citation"),
        "target_reference": row.get("old_reference" if inverse else "new_reference") or row.get("target_reference"),
        "target_reference_range": row.get("old_reference_range" if inverse else "new_reference_range") or row.get("target_reference_range"),
        "target_reference_range_kind": row.get("old_reference_range_kind" if inverse else "new_reference_range_kind")
        or row.get("target_reference_range_kind"),
        "target_source_role": row.get("target_source_role"),
        "evidence_id": row.get("evidence_id"),
        "bbox_refs": tuple(row.get("bbox_refs") or ()),
        "source_proof_text_span_ids": tuple(row.get("text_span_ids") or ()),
        "source_proof_bbox_refs": tuple(row.get("bbox_refs") or ()),
        "target_bbox_refs": tuple(row.get("target_bbox_refs") or ()),
        "target_precision": row.get("target_precision"),
        "source_support_exact": row.get("source_support_exact") is True,
        "text_span_ids": tuple(row.get("text_span_ids") or ()),
        "target_text_span_ids": tuple(row.get("target_text_span_ids") or ()),
        "support_class": row.get("support_class"),
        "grounding_level": row.get("grounding_level"),
        "authority_kind": row.get("authority_kind"),
        "citation_final": row.get("citation_final") is True,
        "recovery_capability": row.get("recovery_capability"),
        "recovery_status": row.get("recovery_status"),
        "target_geometry_method": row.get("target_geometry_method"),
        "trace_only_reason": row.get("trace_only_reason"),
        "citation_available": row.get("citation_available") is True,
        "viewer_highlightable": row.get("viewer_highlightable") is True,
    }


def _article_relation_citation(row: dict) -> dict:
    return {
        "corpus_id": row.get("corpus_id"),
        "evidence_id": row["evidence_id"],
        "relation_id": row.get("relation_id"),
        "legal_unit_id": row.get("legal_unit_id"),
        "target_citation": row.get("target_citation"),
        "target_label": row.get("target_label"),
        "source_document_id": row.get("source_document_id"),
        "citation": row.get("citation"),
        "label": row.get("citation"),
        "hierarchy": tuple(row.get("hierarchy") or ()),
        "quoted_text": row.get("quoted_text"),
        "source_role": row.get("source_role"),
        "temporal_context": row.get("temporal_context"),
        "source_pdf_path": row.get("source_pdf_path"),
        "source_sha256": row.get("source_sha256"),
        "page_numbers": tuple(row.get("page_numbers") or ()),
        "bbox_count": row.get("bbox_count"),
        "citation_final": row.get("citation_final") is True,
        "citable": row.get("citation_final") is True and row.get("citation_available") is True,
        "fact_kind": "article_relation",
        "support_kind": "article_relation",
        "viewer_ref": row.get("viewer_ref"),
        "evidence_status": row.get("status"),
    }


def _deduplicated_article_relation_citations(store, rows: tuple[dict, ...]) -> tuple[dict, ...]:
    grouped: dict[object, dict] = {}
    for row in rows:
        # The operation clause is the single public relation citation.  The
        # versioned target remains attached as lineage/support evidence so it
        # can ground the viewer without creating a duplicate footnote.
        if row.get("support_kind") != "article_relation":
            continue
        citation = _article_relation_citation(row)
        key = citation.get("evidence_id")
        grouped.setdefault(key, citation)
    return tuple(_citation_with_authority(store, row) for row in grouped.values())


def _document_relation_answer(store, relations: tuple[dict, ...]) -> str:
    intent = store.config.setting("intent_config", {}) or {}
    relation_config = _document_relation_config(store)
    labels = intent.get("source_role_labels", {}) or {}
    prefix = str(relation_config.get("source_role_label_prefix", ""))
    amendment_roles = [
        role
        for row in relations
        if (
            role := next(
                (
                    candidate
                    for candidate in (str(row.get("source_role") or ""), str(row.get("target_source_role") or ""))
                    if candidate in labels
                ),
                None,
            )
        )
    ]
    amendment_roles = [role for role in amendment_roles if role]
    names = [f"{prefix}{labels.get(role, role)}" for role in amendment_roles]
    if len(names) > 1:
        listed = ", ".join(names[:-1]) + f", dan {names[-1]}"
        return str(relation_config.get("document_answer_template", "{relations}")).format(relations=listed)
    name = names[0] if names else "Perubahan"
    return str(relation_config.get("single_document_answer_template", "{relation}")).format(relation=name)


def _document_relation_config(store) -> dict:
    return (store.config.setting("intent_config", {}) or {}).get("document_relation", {}) or {}


def _article_relation_answer(store, relations: tuple[dict, ...], trace_support: tuple[dict, ...]) -> str:
    relation_config = _document_relation_config(store)
    relation_labels = relation_config.get("public_relation_labels") or {}
    ambiguous_label = str(relation_config.get("ambiguous_operation_label") or "AMBIGUOUS_OPERATION")

    def labels_for(rows: tuple[dict, ...]) -> list[str]:
        by_target: dict[str, set[str]] = {}
        for row in rows:
            target = str(row.get("new_reference") or row.get("target_citation") or "")
            if target:
                by_target.setdefault(target, set()).add(str(row.get("relation_type") or ""))
        labels = []
        for target in sorted(by_target, key=_natural_label_sort_key):
            types = by_target[target]
            suffix = " / ".join(
                str(relation_labels.get(relation) or relation)
                for relation in ("DELETES", "MODIFIES", "ADDS", "RENAMES", "RENUMBERED_TO")
                if relation in types
            )
            if "AMBIGUOUS_OPERATION" in types:
                suffix = " / ".join((*filter(None, (suffix,)), ambiguous_label))
            labels.append(f"{target} ({suffix})" if suffix else target)
        return labels

    exact_labels = labels_for(tuple(relations))
    trace_labels = labels_for(tuple(trace_support))
    if not exact_labels and not trace_labels:
        return "Sumber terverifikasi tidak memuat relasi hukum yang dapat dipublikasikan."
    source_label = next(
        (
            re.sub(
                r"\s+(?:Scope|Clause\s+\([^)]+\))$",
                "",
                str(row.get("source_label") or ""),
                flags=re.IGNORECASE,
            )
            for row in (*relations, *trace_support)
            if row.get("source_label")
        ),
        "Sumber perubahan",
    )
    if exact_labels and trace_labels:
        return (
            f"Berdasarkan ketentuan perubahan, {source_label} memuat perubahan pada {', '.join(exact_labels)}. "
            f"Keterbatasan: {', '.join(trace_labels)} hanya tersedia sebagai jejak sumber."
        )
    if trace_labels:
        return f"{source_label} menyebut {', '.join(trace_labels)}, tetapi dukungan yang tersedia hanya berupa jejak sumber."
    return f"Berdasarkan ketentuan perubahan, {source_label} memuat relasi: {', '.join(exact_labels)}."


def _empty_query_response(corpus_id: str) -> dict:
    return _clarification_invalid_response(corpus_id) | {
        "route": "empty_query",
        "intent": "empty_query",
        "reason": "empty_query",
        "insufficient_reasons": ("empty_query",),
    }


def _clarification_invalid_response(corpus_id: str) -> dict:
    return {
        "status": "insufficient_evidence",
        "route": "planner_clarification",
        "intent": "clarification",
        "corpus_id": corpus_id,
        "reason": "clarification_session_invalid",
        "answer": ANSWER_TEMPLATES["insufficient"],
        "evidence": (),
        "citations": (),
        "final_citations": (),
        "historical_citations": (),
        "metadata_support": (),
        "structural_support": (),
        "trace_support": (),
        "viewer_refs": (),
        "context_pack": empty_context_pack("clarification_session_invalid"),
        "answer_scope": "insufficient_evidence",
        "answer_type": "none",
        "warnings": (),
        "insufficient_reasons": ("clarification_session_invalid",),
    }


def _ask_route(route: str) -> str:
    return {
        "exact": "legal_reference",
        "structured": "legal_reference",
        "structural_navigation": "structural_navigation",
        "structure_list": "structural_navigation",
        "metadata": "metadata_fact",
        "metadata_not_found": "metadata_fact",
        "metadata_scope_unresolved": "metadata_fact",
        "relation": "legal_relation",
        "relation_not_found": "legal_relation",
        "citation_not_found": "legal_reference",
        "structured_not_found": "legal_reference",
        "scope_unresolved": "legal_reference",
        "bm25": "lexical_fallback",
        "hybrid": "lexical_fallback",
        "hybrid_degraded_sparse": "lexical_fallback",
    }.get(route, route)


def _answer_type(route: str, status: str) -> str:
    if status != "answer_ready":
        return "limited_evidence_summary"
    return {
        "metadata_fact": "metadata_fact",
        "legal_relation": "legal_relation",
    }.get(route, "quoted_evidence")


def _unique_response_rows(groups) -> tuple[dict, ...]:
    rows: list[dict] = []
    seen: set[str] = set()
    for group in groups:
        for row in group or ():
            if not isinstance(row, dict):
                continue
            identity = str(row.get("evidence_id") or row.get("relation_id") or row.get("source_document_id") or row.get("citation") or "")
            if identity and identity in seen:
                continue
            if identity:
                seen.add(identity)
            rows.append(row)
    return tuple(rows)


def _integrity_failure(corpus_id: str, query: str, error_code: str | None) -> dict:
    unknown = error_code in {"unknown_corpus", "registry_unavailable"}
    route = "unsupported_corpus" if unknown else "corpus_integrity"
    return {
        "status": "unsupported_corpus" if unknown else "corpus_not_ready",
        "route": route,
        "intent": route,
        "corpus_id": corpus_id,
        "original_query": query,
        "normalized_query": query.strip(),
        "reason": error_code or "corpus_load_failure",
        "reason_code": error_code or "corpus_load_failure",
        "readiness": False,
        "evidence": (),
        "citations": (),
        "final_citations": (),
        "historical_citations": (),
        "metadata_support": (),
        "structural_support": (),
        "trace_support": (),
        "viewer_refs": (),
        "context_pack": empty_context_pack(error_code),
        "answer_scope": "insufficient_evidence",
        "answer_type": "none",
        "answer": ANSWER_TEMPLATES["insufficient"],
    }


def _compound_response(corpus_id: str, query: str, semantics, parts: tuple[str, ...], responses: tuple[dict, ...]) -> dict:
    """Project independently grounded subanswers without inventing joint proof."""
    successful = tuple(
        (part, response)
        for part, response in zip(parts, responses, strict=True)
        if response.get("status") in {"answer_ready", "limited_answer"}
    )
    if not successful:
        return responses[0] | {"original_query": query} if responses else _empty_query_response(corpus_id)
    complete = len(successful) == len(responses) and all(response.get("status") == "answer_ready" for _, response in successful)
    answer = "\n\n".join(
        f"{part}: {str(response.get('answer') or '').strip()}" for part, response in successful if str(response.get("answer") or "").strip()
    )
    fields = (
        "matches",
        "evidence",
        "citations",
        "final_citations",
        "historical_citations",
        "metadata_support",
        "structural_support",
        "trace_support",
        "viewer_refs",
        "metadata_facts",
        "legal_relations",
    )
    merged = {field: _unique_response_rows(response.get(field, ()) for _, response in successful) for field in fields}
    warnings = _unique_response_values(response.get("warnings", ()) for response in responses)
    missing = _unique_response_values(response.get("insufficient_reasons", ()) for response in responses)
    if len(successful) != len(responses):
        warnings = (*warnings, "compound_partial")
    return {
        "status": "answer_ready" if complete else "limited_answer",
        "route": "compound",
        "intent": "compound_research",
        "corpus_id": corpus_id,
        "original_query": query,
        "normalized_query": query.strip(),
        "operation": "multiple",
        "source_scopes": semantics.source_scopes,
        "temporal_scope": semantics.temporal_scope,
        "answer_type": "compound_evidence",
        "answer": answer,
        "context_pack": empty_context_pack("compound"),
        "answer_scope": "direct_evidence" if complete else "limited_evidence",
        "warnings": warnings,
        "insufficient_reasons": missing,
        **merged,
    }


def _unique_response_values(groups) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value) for group in groups for value in group or () if value))


def _clarification_exhausted_response(corpus_id: str) -> dict:
    return _clarification_invalid_response(corpus_id) | {
        "reason": "clarification_unresolved",
        "insufficient_reasons": ("clarification_unresolved",),
    }
