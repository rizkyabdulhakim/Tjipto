from __future__ import annotations

DIRECT_ROUTES = {"exact", "metadata", "relation", "structured", "bm25"}
REQUIRED_FIELDS = (
    "citation",
    "quoted_text",
    "source_pdf_path",
    "source_sha256",
    "source_role",
    "temporal_context",
)
PUBLIC_REJECTION_REASONS = {
    "runtime_not_loadable",
    "linked_legal_unit_not_runtime_loadable",
    "linked_chunk_not_runtime_loadable",
    "page_grounded_only_not_answerable",
    "viewer_not_highlightable",
    "missing_exact_grounding",
    "missing_exact_text_span_support",
    "missing_bbox",
    "invalid_bbox",
    "retrieval_unit_backing_record_not_answerable",
    "noncanonical_trace_not_answerable",
}


def assemble_context_pack(store, matches: tuple[dict, ...]) -> dict:
    answer_evidence = []
    supporting_context = []
    excluded = []
    reasons = {}
    for row in matches:
        accepted, reason = validate_answer_candidate(store, row)
        payload = _payload(store, row)
        reasons[row["evidence_id"]] = reason
        if accepted:
            answer_evidence.append(payload)
        else:
            excluded.append(payload | {"reason": reason})
            if row.get("route_sources") == ("graph",):
                supporting_context.append(payload | {"context_type": "graph_supporting"})
    return {
        "answer_evidence": tuple(answer_evidence),
        "supporting_context": tuple(supporting_context),
        "excluded_results": tuple(excluded),
        "citation_payloads": tuple(_citation_payload(row) for row in answer_evidence),
        "viewer_refs": tuple(row["viewer_ref"] for row in answer_evidence),
        "validation_reasons": reasons,
    }


def empty_context_pack(reason: str | None) -> dict:
    return {
        "answer_evidence": (),
        "supporting_context": (),
        "excluded_results": (),
        "citation_payloads": (),
        "viewer_refs": (),
        "validation_reasons": {"request": reason or "no_final_evidence"},
    }


def validate_answer_candidate(store, row: dict) -> tuple[bool, str]:
    if row.get("forced_rejection_reason"):
        return False, row["forced_rejection_reason"]
    if row.get("runtime_loadable") is False:
        return False, "runtime_not_loadable"
    if not row.get("metadata_grounding"):
        lineage_error = store.lineage_error(row)
        if lineage_error:
            return False, f"source_lineage_invalid:{lineage_error}"
    legal_unit = _legal_unit(store, row.get("legal_unit_id"))
    chunk = _chunk_for_unit(store, row.get("legal_unit_id"))
    retrieval_unit = _retrieval_unit(store, row.get("evidence_id"))
    historical_instrument = row.get("authority_kind") == "instrument_provenance"
    if not row.get("metadata_grounding"):
        if row.get("bbox_precision") == "page_grounded_only":
            return False, "page_grounded_only_not_answerable"
        if row.get("bbox_precision") != "exact":
            return False, "missing_exact_grounding"
        if row.get("viewer_highlightable") is not True:
            return False, "viewer_not_highlightable"
        if not row.get("text_span_ids"):
            return False, "missing_exact_text_span_support"
        bbox_ids = set(row.get("bbox_ids") or row.get("bbox_refs") or ())
        actual_bbox_ids = {bbox.get("bbox_id") for bbox in store.bboxes_for(row["evidence_id"])}
        if not bbox_ids:
            return False, "missing_bbox"
        if not bbox_ids <= actual_bbox_ids:
            return False, "invalid_bbox"
        if legal_unit and legal_unit.get("runtime_loadable") is False and not historical_instrument:
            return False, "linked_legal_unit_not_runtime_loadable"
        if chunk and chunk.get("runtime_loadable") is False and not historical_instrument:
            return False, "linked_chunk_not_runtime_loadable"
        if (legal_unit or chunk) and not ((legal_unit and legal_unit.get("text_span_ids")) or (chunk and chunk.get("text_span_ids"))):
            return False, "missing_exact_text_span_support"
        if retrieval_unit and retrieval_unit.get("status") != "accepted" and not historical_instrument:
            return False, "retrieval_unit_backing_record_not_answerable"
        if _noncanonical_trace(legal_unit, chunk, row):
            return False, "noncanonical_trace_not_answerable"
    if not (DIRECT_ROUTES & set(row.get("route_sources") or ())):
        return False, "graph_only"
    if "bm25" in set(row.get("route_sources") or ()) and row.get("lexical_relevance_ok") is False:
        return False, row.get("lexical_relevance_reason") or "weak_lexical_match"
    if row.get("status") != "final":
        return False, "not_final"
    if row.get("metadata_grounding"):
        if not row.get("bbox_refs"):
            return False, "missing_metadata_grounding"
    elif not store.bboxes_for(row["evidence_id"]):
        return False, "missing_bbox"
    if not row.get("page_numbers"):
        return False, "missing_page_numbers"
    for field in REQUIRED_FIELDS:
        if not row.get(field):
            return False, f"missing_{field}"
    return True, "answer_evidence"


def _payload(store, row: dict) -> dict:
    bboxes = store.metadata_bboxes_for(row["evidence_id"]) if row.get("metadata_grounding") else store.bboxes_for(row["evidence_id"])
    label = _evidence_label(row)
    hierarchy = _evidence_hierarchy(row)
    can_resolve = (
        row.get("status") == "final" and row.get("bbox_precision") == "exact" and row.get("viewer_highlightable") is True and bool(bboxes)
    ) and (not row.get("metadata_grounding") or row.get("metadata_viewer_resolvable") is True)
    return {
        "corpus_id": row.get("corpus_id"),
        "evidence_id": row["evidence_id"],
        "legal_unit_id": row.get("legal_unit_id"),
        "source_document_id": row.get("source_document_id"),
        "citation": row.get("citation"),
        "label": label,
        "hierarchy": hierarchy,
        "source_role": row.get("source_role"),
        "temporal_context": row.get("temporal_context"),
        "source_pdf_path": row.get("source_pdf_path"),
        "source_sha256": row.get("source_sha256"),
        "page_numbers": tuple(row.get("page_numbers") or ()),
        "bbox_count": len(bboxes),
        "quoted_text": row.get("quoted_text"),
        "metadata_answer": row.get("metadata_answer"),
        "metadata_field": row.get("metadata_field"),
        "legal_relation": row.get("legal_relation"),
        "candidate_type": row.get("candidate_type"),
        "route_sources": tuple(row.get("route_sources") or ()),
        "evidence_status": row.get("status"),
        "authority_kind": row.get("authority_kind"),
        "citation_final": row.get("citation_final"),
        "citable_status": row.get("citable_status"),
        "viewer_ref": {
            "action": "viewer",
            "evidence_id": row["evidence_id"],
            "page_numbers": tuple(row.get("page_numbers") or ()),
            "bbox_count": len(bboxes),
            "can_resolve": can_resolve,
        },
    }


def _citation_payload(row: dict) -> dict:
    return {
        "corpus_id": row.get("corpus_id"),
        "evidence_id": row["evidence_id"],
        "legal_unit_id": row.get("legal_unit_id"),
        "source_document_id": row.get("source_document_id"),
        "citation": row.get("citation"),
        "label": row.get("label") or _evidence_label(row),
        "hierarchy": _evidence_hierarchy(row),
        "quoted_text": row.get("quoted_text"),
        "metadata_answer": row.get("metadata_answer"),
        "metadata_field": row.get("metadata_field"),
        "legal_relation": row.get("legal_relation"),
        "candidate_type": row.get("candidate_type"),
        "source_role": row.get("source_role"),
        "temporal_context": row.get("temporal_context"),
        "source_pdf_path": row.get("source_pdf_path"),
        "source_sha256": row.get("source_sha256"),
        "page_numbers": row.get("page_numbers"),
        "bbox_count": row.get("bbox_count"),
        "viewer_ref": row.get("viewer_ref"),
        "evidence_status": row.get("evidence_status"),
        "authority_kind": row.get("authority_kind"),
        "citation_final": row.get("citation_final"),
        "citable_status": row.get("citable_status"),
    }


def _evidence_label(row: dict) -> str | None:
    hierarchy = _evidence_hierarchy(row)
    if hierarchy:
        return " / ".join(str(item) for item in hierarchy)
    return row.get("citation") or row.get("legal_unit_id")


def _evidence_hierarchy(row: dict) -> tuple:
    return tuple(item for item in (row.get("hierarchy") or ()) if item)


def _legal_unit(store, legal_unit_id: str | None) -> dict | None:
    return next((row for row in _optional_rows(store, "legal_units") if row.get("legal_unit_id") == legal_unit_id), None)


def _chunk_for_unit(store, legal_unit_id: str | None) -> dict | None:
    return next((row for row in _optional_rows(store, "chunks") if row.get("legal_unit_id") == legal_unit_id), None)


def _retrieval_unit(store, evidence_id: str | None) -> dict | None:
    return next((row for row in _optional_rows(store, "retrieval_units") if row.get("evidence_id") == evidence_id), None)


def _noncanonical_trace(legal_unit: dict | None, chunk: dict | None, row: dict) -> bool:
    historical_exact = (
        row.get("authority_kind") == "normative_legal_text"
        and row.get("exactness") == "exact"
        and row.get("citation_final") is False
        and str(row.get("source_role") or "").endswith("_historical")
    )
    if historical_exact:
        return any(item and item.get("provenance_exception_category") == "accepted_noncanonical_source_conflict_trace_only" for item in (legal_unit, chunk, row))
    return any(
        item
        and (
            item.get("canonical_use_allowed") is False
            or item.get("provenance_exception_category") == "accepted_noncanonical_source_conflict_trace_only"
        )
        for item in (legal_unit, chunk, row)
    )


def _optional_rows(store, attr: str) -> tuple | list:
    try:
        return getattr(store, attr, ())
    except (KeyError, OSError, ValueError):
        return ()
