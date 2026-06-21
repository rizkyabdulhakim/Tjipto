from __future__ import annotations


DIRECT_ROUTES = {"exact", "structured", "bm25"}
BAB_XA_PASAL_PREFIXES = tuple(f"Pasal 28{letter}" for letter in "ABCDEFGHIJ")
REQUIRED_FIELDS = (
    "citation",
    "quoted_text",
    "source_pdf_path",
    "source_sha256",
    "source_role",
    "temporal_context",
)


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
    if row.get("runtime_loadable") is False:
        return False, "runtime_not_loadable"
    if not (DIRECT_ROUTES & set(row.get("route_sources") or ())):
        return False, "graph_only"
    if (
        "bm25" in set(row.get("route_sources") or ())
        and row.get("lexical_relevance_ok") is False
    ):
        return False, row.get("lexical_relevance_reason") or "weak_lexical_match"
    if row.get("status") != "final":
        return False, "not_final"
    if not store.bboxes_for(row["evidence_id"]):
        return False, "missing_bbox"
    if not row.get("page_numbers"):
        return False, "missing_page_numbers"
    for field in REQUIRED_FIELDS:
        if not row.get(field):
            return False, f"missing_{field}"
    return True, "answer_evidence"


def _payload(store, row: dict) -> dict:
    bboxes = store.bboxes_for(row["evidence_id"])
    label = _evidence_label(row)
    hierarchy = _evidence_hierarchy(row)
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
        "route_sources": tuple(row.get("route_sources") or ()),
        "evidence_status": row.get("status"),
        "viewer_ref": {
            "action": "viewer",
            "evidence_id": row["evidence_id"],
            "page_numbers": tuple(row.get("page_numbers") or ()),
            "bbox_count": len(bboxes),
            "source_pdf_path": row.get("source_pdf_path"),
            "source_sha256": row.get("source_sha256"),
            "can_resolve": row.get("status") == "final" and bool(bboxes),
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
        "source_role": row.get("source_role"),
        "temporal_context": row.get("temporal_context"),
        "source_pdf_path": row.get("source_pdf_path"),
        "source_sha256": row.get("source_sha256"),
        "page_numbers": row.get("page_numbers"),
        "bbox_count": row.get("bbox_count"),
        "viewer_ref": row.get("viewer_ref"),
        "evidence_status": row.get("evidence_status"),
    }


def _evidence_label(row: dict) -> str | None:
    hierarchy = _evidence_hierarchy(row)
    if hierarchy:
        return " / ".join(str(item) for item in hierarchy)
    return row.get("citation") or row.get("legal_unit_id")


def _evidence_hierarchy(row: dict) -> tuple:
    hierarchy = tuple(item for item in (row.get("hierarchy") or ()) if item)
    if hierarchy and hierarchy[0] == "BAB X" and _is_bab_xa_article(hierarchy):
        return ("BAB XA", *hierarchy[1:])
    return hierarchy


def _is_bab_xa_article(hierarchy: tuple) -> bool:
    return any(str(item).startswith(BAB_XA_PASAL_PREFIXES) for item in hierarchy[1:])
