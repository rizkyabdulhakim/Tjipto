"""Turn verified retrieval results into one public, citation-safe answer."""

from __future__ import annotations

from tjipto.corpora.source_arbitration import attach_source_reference_provenance
from tjipto.retrieval.answer import assemble_context_pack, empty_context_pack, validate_answer_candidate
from tjipto.retrieval.requirements import ambiguity_reason, semantic_support_score
from tjipto.runtime.answer_arbitration import (
    _answer_templates,
    _claim_answer,
    _compact_text,
    _lexical_fallback_is_limited,
    _metadata_answer,
    _semantic_specificity,
    _semantic_support_rank,
    _semantic_supports_query,
    _structure_outline_item,
)
from tjipto.runtime.claim_support import all_supported, verify_claims
from tjipto.runtime.public_document import _metadata_fact, _metadata_support
from tjipto.runtime.response import AnswerDecision, _answer_type, _ask_route, _relation_response, compose_research_answer, project_response
from tjipto.runtime.viewer import _citation_with_authority, _claim_citations, _public_evidence_row, _source_status_label


def evidence_answer(
    store,
    query: str,
    semantics,
    routed: dict,
    requirements: tuple,
    evidence_set,
    assessment,
    research_plan,
) -> dict:
    """Apply evidence sufficiency, claim verification, and public projection."""
    ask_route = _ask_route(routed["route"])
    templates = _answer_templates(store)
    routed["original_query"] = query
    if (
        semantics.operation == "search"
        and routed.get("route") in {"bm25", "hybrid", "hybrid_degraded_sparse"}
        and routed.get("matches")
        and not any(str(item.requirement_id).startswith("source_occurrence_") for item in requirements)
        and not any(_semantic_supports_query(store, query, row) for row in routed["matches"])
    ):
        return _insufficient(routed | {"reason": "semantic_support_missing"}, ask_route, templates, "semantic_support_missing")
    ambiguous = ambiguity_reason(semantics, routed)
    if ambiguous:
        return _insufficient(routed | {"reason": ambiguous}, ask_route, templates, ambiguous, reason_code=ambiguous)
    if routed.get("route") == "document_relation":
        return _relation_response(store, routed)
    if routed["status"] != "found":
        public_status = (
            "insufficient_evidence"
            if routed.get("route")
            in {"metadata_not_found", "metadata_scope_unresolved", "relation_not_found", "structured_not_found", "scope_unresolved"}
            else routed["status"]
        )
        reason = routed.get("reason") or routed["status"]
        missing = assessment.missing_requirement_ids if assessment is not None and assessment.missing_requirement_ids else (reason,)
        return project_response(
            routed,
            AnswerDecision(public_status, ask_route, "none", templates["insufficient"], empty_context_pack(reason), insufficient_reasons=missing),
        )
    if routed.get("semantic_scope_loss"):
        return _insufficient(routed, ask_route, templates, "semantic_scope_loss")
    if evidence_set is not None and assessment is not None and assessment.status == "insufficient":
        return project_response(
            routed,
            AnswerDecision(
                "insufficient_evidence",
                ask_route,
                "none",
                templates["insufficient"],
                empty_context_pack("required_evidence_missing"),
                insufficient_reasons=tuple(assessment.missing_requirement_ids),
            ),
        )

    answer_matches = evidence_set.supports if evidence_set is not None else routed["matches"]
    if ask_route == "lexical_fallback" and evidence_set is None:
        answer_matches = _best_lexical_match(store, query, semantics, answer_matches)
    context_pack = assemble_context_pack(store, answer_matches)
    evidence = context_pack["answer_evidence"]
    if not evidence:
        reasons = tuple(sorted(set(context_pack["validation_reasons"].values()))) or ("semantic_support_missing",)
        return project_response(
            routed,
            AnswerDecision(
                "insufficient_evidence",
                ask_route,
                "none",
                templates["insufficient"],
                context_pack,
                insufficient_reasons=reasons,
            ),
        )
    claim_support = verify_claims(semantics, evidence, store)
    if not all_supported(claim_support):
        claim_reason = next(claim.reason_code for claim in claim_support if claim.reason_code)
        return project_response(
            routed,
            AnswerDecision(
                "insufficient_evidence",
                ask_route,
                "none",
                _claim_answer(claim_support),
                empty_context_pack(claim_reason),
                insufficient_reasons=(claim_reason,),
                reason_code=claim_reason,
                claim_support=tuple(claim.public() for claim in claim_support),
            ),
        )

    status = _answer_status(store, query, semantics, evidence, requirements, evidence_set, assessment, research_plan, ask_route, context_pack)
    public_evidence = tuple(_public_evidence_row(store, row) for row in evidence)
    metadata_support = tuple(_metadata_support(store, row) for row in evidence if row.get("metadata_field"))
    if metadata_support:
        citations = ()
        viewer_refs = ()
    else:
        citations = _claim_citations(
            tuple(_citation_with_authority(store, row) for row in context_pack["citation_payloads"]),
            claim_support,
        )
        viewer_refs = tuple(row["viewer_ref"] for row in citations)
    if metadata_support:
        deterministic_answer = _metadata_answer(store, metadata_support)
    elif evidence_set is not None:
        deterministic_answer = compose_research_answer(
            public_evidence,
            evidence_set,
            requirements,
            assessment,
            preferred_source_role=getattr(store.config, "preferred_source_role", None),
        )
    else:
        deterministic_answer = answer_text(store, status, evidence, templates, claim_support)
        if routed.get("route") == "structure_list":
            outline = tuple(dict.fromkeys(_structure_outline_item(store, row) for row in evidence))
            outline = tuple(item for item in outline if item)
            if outline:
                deterministic_answer = f"Struktur naskah meliputi: {'; '.join(outline)}."
    response = project_response(
        routed,
        AnswerDecision(
            status,
            ask_route,
            _answer_type(ask_route, status),
            deterministic_answer,
            context_pack,
            evidence=public_evidence,
            citations=citations,
            final_citations=citations,
            historical_citations=context_pack.get("historical_citations", ()),
            viewer_refs=viewer_refs,
            metadata_facts=tuple(_metadata_fact(row) for row in evidence if row.get("metadata_field")),
            metadata_support=metadata_support,
            structural_support=tuple(_citation_with_authority(store, row) for row in context_pack.get("structural_support", ())),
            trace_support=tuple(_citation_with_authority(store, row) for row in context_pack.get("trace_support", ())),
            legal_relations=tuple(row["legal_relation"] for row in evidence if row.get("legal_relation")),
            answer_scope="direct_evidence" if status == "answer_ready" else "limited_evidence",
            warnings=("metadata_support_not_exact_highlightable",)
            if any(row.get("viewer_highlightable") is not True for row in metadata_support)
            else (),
            claim_support=tuple(claim.public() for claim in claim_support),
        ),
    )
    return attach_source_reference_provenance(store, query, response)


def answer_text(store, status: str, evidence: tuple[dict, ...], templates: dict[str, str], claims=()) -> str:
    if evidence[0].get("metadata_answer"):
        return _metadata_answer(store, evidence)
    if evidence[0].get("legal_relation"):
        relations = tuple(row["legal_relation"] for row in evidence)
        sources = tuple(dict.fromkeys(str(row.get("source_label") or "") for row in relations if row.get("source_label")))
        targets = tuple(dict.fromkeys(str(row.get("target_label") or "") for row in relations if row.get("target_label")))
        return f"{sources[0]} memuat: {', '.join(targets)}." if len(sources) == 1 and targets else templates["legal_relation"]
    quote = " ".join(_compact_text(row.get("quoted_text") or row.get("display_text") or "") for row in evidence).strip()
    if claims:
        claim = claims[0]
        segment = next((item.get("exact_quote") for item in claim.support_segments if item.get("exact_quote")), None)
        return f"Klaim ‘{claim.claim_text}’ didukung oleh segmen terverifikasi: {segment or quote}."
    source_label = _source_status_label(evidence[0], store)
    citation = evidence[0].get("display_label") or evidence[0].get("label") or evidence[0].get("citation") or "Bukti"
    preferred_role = getattr(store.config, "preferred_source_role", None)
    prefix = f"{source_label} — " if evidence[0].get("source_role") != preferred_role and source_label else ""
    return f"{prefix}{citation}: {quote}" if quote else templates["citation"].format(citation=citation)


def _insufficient(base: dict, route: str, templates: dict[str, str], reason: str, *, reason_code: str | None = None) -> dict:
    return project_response(
        base,
        AnswerDecision(
            "insufficient_evidence",
            route,
            "none",
            templates["insufficient"],
            empty_context_pack(reason),
            insufficient_reasons=(reason,),
            reason_code=reason_code,
        ),
    )


def _best_lexical_match(store, query: str, semantics, matches: tuple[dict, ...]) -> tuple[dict, ...]:
    candidates = tuple(
        row
        for row in matches
        if validate_answer_candidate(store, row)[0]
        and (semantics.requested_function == "proposition_verification" or _semantic_supports_query(store, query, row))
    )
    normative = tuple(row for row in candidates if row.get("authority_kind") not in {"structural_context", "structural_support"})
    candidates = normative or candidates
    if not candidates:
        return ()
    best = max(
        candidates,
        key=lambda row: (
            _semantic_specificity(store, row),
            semantic_support_score(
                store,
                query,
                " ".join((str(row.get("citation") or ""), " ".join(row.get("hierarchy") or ()), str(row.get("quoted_text") or ""))),
            ),
            -_semantic_support_rank(row)[0],
            -_semantic_support_rank(row)[1],
            str(row.get("evidence_id") or ""),
        ),
    )
    return (best,)


def _answer_status(store, query, semantics, evidence, requirements, evidence_set, assessment, research_plan, route, context_pack) -> str:
    if _lexical_fallback_is_limited(store, query, evidence, research_plan, route, semantics.operation):
        return "limited_answer"
    if assessment is not None and assessment.status == "partial":
        return "limited_answer"
    if assessment is not None and assessment.status == "complete":
        return "limited_answer" if any(item.allow_partial for item in requirements) else "answer_ready"
    if context_pack["trace_support"] and not context_pack["citation_payloads"]:
        return "limited_answer"
    return "answer_ready"
