from __future__ import annotations


DIRECT_ROUTES = {"exact", "structured", "bm25"}
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
        "excluded_candidates": tuple(excluded),
        "citation_payloads": tuple(
            {"citation": row.get("citation"), "evidence_id": row["evidence_id"]}
            for row in answer_evidence
        ),
        "viewer_refs": tuple(row["viewer_ref"] for row in answer_evidence),
        "validation_reasons": reasons,
    }


def validate_answer_candidate(store, row: dict) -> tuple[bool, str]:
    if not (DIRECT_ROUTES & set(row.get("route_sources") or ())):
        return False, "graph_only"
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
    return {
        "evidence_id": row["evidence_id"],
        "citation": row.get("citation"),
        "source_role": row.get("source_role"),
        "temporal_context": row.get("temporal_context"),
        "source_pdf_path": row.get("source_pdf_path"),
        "source_sha256": row.get("source_sha256"),
        "page_numbers": tuple(row.get("page_numbers") or ()),
        "bbox_count": len(bboxes),
        "quoted_text": row.get("quoted_text"),
        "route_sources": tuple(row.get("route_sources") or ()),
        "viewer_ref": {"action": "viewer", "evidence_id": row["evidence_id"]},
    }
