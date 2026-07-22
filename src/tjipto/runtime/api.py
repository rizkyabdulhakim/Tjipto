from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from tjipto.runtime.service import LegalRuntimeService, public_article_relation


class BadRequest(ValueError):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def handle_request(
    corpus_id: str,
    action: str,
    payload: dict,
    repo_root: Path | None = None,
    service: LegalRuntimeService | None = None,
) -> dict:
    service = service or _service_for(repo_root)
    if action == "search":
        return _public_search(
            service.search(
                corpus_id,
                str(payload.get("query", "")),
                _limit(payload, default=10),
                _filters(payload),
            )
        )
    if action == "citation":
        return _public_citation_response(
            service.citation(
                corpus_id,
                str(payload.get("query", "")),
                _optional_str(payload, "source_role"),
                _filters(payload),
            )
        )
    if action == "viewer":
        return _public_viewer(
            service.viewer(
                corpus_id,
                _optional_str(payload, "evidence_id"),
                relation_id=_optional_str(payload, "relation_id"),
                source_document_id=_optional_str(payload, "source_document_id"),
                page_number=_optional_int(payload, "page_number"),
                bbox_id=_optional_str(payload, "bbox_id"),
                source_pdf_path=_optional_str(payload, "source_pdf_path"),
            )
        )
    if action == "ask":
        return _public_ask(
            service.ask(
                corpus_id,
                str(payload.get("query", "")),
                _limit(payload, default=3),
                _filters(payload),
            )
        )
    if action == "capabilities":
        return service.capabilities(corpus_id)
    if action == "bookmarks":
        return service.bookmarks(corpus_id)
    if action == "bookmark":
        return service.bookmark(
            corpus_id,
            _required_str(payload, "evidence_id"),
            _optional_str(payload, "note"),
            _optional_str(payload, "citation_id"),
            _optional_str(payload, "viewer_ref_id"),
        )
    return {"status": "unsupported_action"}


def _public_search(result: dict) -> dict:
    if result.get("readiness") is False:
        return _public_integrity(result) | {"results": ()}
    return {
        "status": result.get("status"),
        "public_status": result.get("public_status"),
        "corpus_id": result.get("corpus_id"),
        "reason": _public_reason(result.get("reason")),
        "applied_filters": result.get("applied_filters", {}),
        "results": tuple(_public_search_result(row) for row in result.get("results", ())),
    }


def _public_ask(result: dict) -> dict:
    if result.get("readiness") is False:
        return _public_integrity(result)
    support_rows = (
        *((row, "legal") for row in result.get("final_citations", result.get("citations", ()))),
        *(() if result.get("answer_type") == "article_amendment_relation" else ((row, "legal") for row in result.get("historical_citations", ()))),
        *((row, "metadata") for row in result.get("metadata_support", ())),
        *((row, "structure") for row in result.get("structural_support", ())),
        *((row, "trace") for row in result.get("relation_support", ())),
        *((row, "trace") for row in result.get("trace_support", ())),
    )
    public = {
        "status": result.get("status"),
        "answer": result.get("answer"),
        "intent": result.get("intent"),
        "route": result.get("route"),
        "legal_relations": tuple(_public_legal_relation(row) for row in result.get("legal_relations", ())),
        "document_relations": tuple(result.get("document_relations", ())),
        "answer_scope": result.get("answer_scope"),
        "warnings": tuple(result.get("warnings", ())),
        "insufficient_reasons": tuple(_public_reason(row) or row for row in result.get("insufficient_reasons", ())),
        "supports": tuple(_public_support(row, panel_section) for row, panel_section in support_rows),
    }
    if result.get("document_source") is not None:
        source = result["document_source"]
        public["answer_type"] = "source_document"
        public["document_source"] = {
            "source_document_id": source.get("source_document_id"),
            "source_role": source.get("source_role"),
            "temporal_context": source.get("temporal_context"),
            "document_title": source.get("document_title"),
            "viewer_target": {
                "action": "open_document",
                "source_document_id": source.get("source_document_id"),
            },
        }
    for key in ("requested_function", "target_reference", "legal_domain"):
        if result.get(key) is not None:
            public[key] = result[key]
    if result.get("clarification_options"):
        public["clarification_options"] = tuple(result["clarification_options"])
    return public


def _public_citation_response(result: dict) -> dict:
    if result.get("readiness") is False:
        return _public_integrity(result) | {"citation_payloads": ()}
    public = {
        "status": result.get("status"),
        "public_status": result.get("public_status", result.get("status")),
        "answer_type": result.get("answer_type"),
        "reason": _public_reason(result.get("reason")),
        "applied_filters": result.get("applied_filters", {}),
        "citation_payloads": tuple(_public_citation(row) for row in result.get("citation_payloads", ())),
        "viewer_refs": tuple(_public_viewer_ref(row) for row in result.get("viewer_refs", ())),
        "validation_reasons": result.get("validation_reasons", {}),
    }
    for key in ("requested_function", "target_reference", "legal_domain"):
        if result.get(key) is not None:
            public[key] = result[key]
    return public


def _public_viewer(result: dict) -> dict:
    if result.get("readiness") is False:
        return _public_integrity(result)
    public = {
        "status": result.get("status"),
        "corpus_id": result.get("corpus_id"),
        "evidence_id": result.get("evidence_id"),
        "legal_unit_id": result.get("legal_unit_id"),
        "source_document_id": result.get("source_document_id"),
        "source_url": result.get("source_url"),
        "citation": result.get("citation"),
        "quoted_text": result.get("quoted_text"),
        "source_role": result.get("source_role"),
        "temporal_context": result.get("temporal_context"),
        "source_status_label": result.get("source_status_label"),
        "authority_kind": result.get("authority_kind"),
        "authority_label": result.get("authority_label"),
        "citation_final": result.get("citation_final"),
        "page_numbers": result.get("page_numbers", ()),
        "bbox_count": result.get("bbox_count"),
        "bbox_precision": result.get("bbox_precision"),
        "viewer_highlightable": result.get("viewer_highlightable"),
        "bbox_rectangles": tuple(_public_bbox(row) for row in result.get("bbox_rectangles", ())),
        "pdf_access_available": result.get("pdf_access_available", False),
        "rendering_available": result.get("rendering_available", False),
        "render_status": result.get("render_status"),
        "reason": _public_reason(result.get("reason")) or result.get("reason"),
    }
    if result.get("pdf"):
        public["pdf"] = {
            "mime_type": result["pdf"].get("mime_type"),
            "access_url": result["pdf"].get("access_url"),
        }
    return public


def _public_integrity(result: dict) -> dict:
    return {
        "status": result["status"],
        "route": result.get("route", "corpus_integrity"),
        "reason_code": result.get("reason_code") or result.get("reason"),
        "corpus_id": result.get("corpus_id"),
        "readiness": False,
        "answer_type": "none",
        "evidence": (),
        "citations": (),
        "viewer_refs": (),
        "context_pack": result.get("context_pack", {}),
    }


def _public_bbox(row: dict) -> dict:
    precision = _public_bbox_precision(row.get("bbox_precision"))
    return {
        "bbox_id": row.get("bbox_id"),
        "bbox_precision": precision,
        "page_number": row.get("page_number"),
        "viewer_highlightable": _public_viewer_highlightable(precision, row.get("viewer_highlightable")),
        "x0": row.get("x0"),
        "y0": row.get("y0"),
        "x1": row.get("x1"),
        "y1": row.get("y1"),
        "coordinate_space": row.get("coordinate_space"),
        "coordinate_origin": row.get("coordinate_origin"),
        "page_width": row.get("page_width"),
        "page_height": row.get("page_height"),
        "page_rotation": row.get("page_rotation"),
        "page_box_basis": row.get("page_box_basis"),
        "transform_version": row.get("transform_version"),
    }


def _public_bbox_precision(value) -> str:
    return value if value in {"exact", "coarse", "page_grounded_only"} else "page_grounded_only"


def _public_viewer_highlightable(precision: str, value) -> bool:
    return precision == "exact" and value is True


def _public_reason(reason):
    if reason in {"metadata_not_found", "relation_not_found"}:
        return "insufficient_evidence"
    return (
        reason
        if reason
        in {
            "invalid_query",
            "unsupported_corpus",
            "citation_not_found",
            "insufficient_evidence",
            "exact_instrument_unit_fail_closed",
            "neighbor_substitution_not_allowed",
            "page_grounded_only_not_answerable",
            "viewer_not_highlightable",
            "missing_exact_grounding",
            "instrument_unresolved",
            "content_signal_unresolved",
            "effect_signal_unsupported",
            "unsupported_instrument_analysis",
            "unsupported_analysis_intent",
            "analysis_metadata_conflict",
            "intent_arbitration_analysis_wins",
            "metadata_candidate_signal",
            "instrument_resolved_fail_closed",
            "exact_label_target_not_answerable",
            "lexical_fallback_blocked_by_instrument_intent",
            "current_fact_unsupported",
            "unsupported_scope",
            "document_not_found",
            "unresolved_source_scope",
        }
        else None
    )


def _public_search_result(row: dict) -> dict:
    return {
        "corpus_id": row.get("corpus_id"),
        "legal_unit_id": row.get("legal_unit_id"),
        "evidence_id": row.get("evidence_id"),
        "document_id": row.get("document_id"),
        "citation_id": row.get("citation_id"),
        "viewer_ref_id": row.get("viewer_ref_id"),
        "source_document_id": row.get("source_document_id"),
        "source_url": row.get("source_url"),
        "title": row.get("title"),
        "document_title": row.get("document_title"),
        "citation": row.get("citation"),
        "label": row.get("label"),
        "snippet": row.get("snippet"),
        "source_role": row.get("source_role"),
        "temporal_context": row.get("temporal_context"),
        "authority_kind": row.get("authority_kind"),
        "authority_label": row.get("authority_label"),
        "citation_final": row.get("citation_final"),
        "page_numbers": row.get("page_numbers", ()),
        "bbox_count": row.get("bbox_count"),
        "viewer_ref": _public_viewer_ref(row.get("viewer_ref") or {}),
        "status": row.get("status"),
    }


def _public_citation(row: dict) -> dict:
    viewer_ref = dict(row.get("viewer_ref") or {})
    viewer_ref.setdefault("source_document_id", row.get("source_document_id"))
    viewer_ref.setdefault("evidence_id", row.get("evidence_id"))
    viewer_ref.setdefault("page_numbers", tuple(row.get("page_numbers") or ()))
    public = {
        "corpus_id": row.get("corpus_id"),
        "evidence_id": row.get("evidence_id"),
        "legal_unit_id": row.get("legal_unit_id"),
        "source_document_id": row.get("source_document_id"),
        "source_url": row.get("source_url"),
        "citation": row.get("citation"),
        "label": row.get("label"),
        "hierarchy": row.get("hierarchy", ()),
        "document_title": row.get("document_title"),
        "quoted_text": row.get("quoted_text"),
        "metadata_answer": row.get("metadata_answer"),
        "metadata_field": row.get("metadata_field"),
        "source_role": row.get("source_role"),
        "temporal_context": row.get("temporal_context"),
        "authority_kind": row.get("authority_kind"),
        "authority_label": row.get("authority_label"),
        "citation_final": row.get("citation_final"),
        "page_numbers": row.get("page_numbers", ()),
        "bbox_count": row.get("bbox_count"),
        "viewer_ref": _public_viewer_ref(viewer_ref),
        "evidence_status": row.get("evidence_status"),
        "support_kind": row.get("support_kind"),
        "relevant_quote_eligible": row.get("relevant_quote_eligible") is True,
        "display_text": row.get("display_text") or row.get("quoted_text") or "",
        "copy_text": row.get("copy_text") or row.get("quoted_text") or "",
        "layout_lines": row.get("layout_lines") or tuple(str(row.get("quoted_text") or "").splitlines()),
        "viewer_target": _public_viewer_ref(row.get("viewer_target") or viewer_ref),
    }
    if row.get("legal_relation"):
        public["legal_relation"] = _public_legal_relation(row["legal_relation"])
    return public


def _public_support(row: dict, panel_section: str) -> dict:
    support_kind = row.get("support_kind") or ("metadata_source" if panel_section == "metadata" else "structural_provenance" if panel_section == "structure" else "trace_support")
    viewer_target = dict(row.get("viewer_target") or row.get("viewer_ref") or {})
    viewer_target.setdefault("source_document_id", row.get("source_document_id"))
    labels = {
        "legal": "Kutipan Relevan",
        "metadata": "Sumber Dokumen",
        "structure": "Struktur Dokumen",
        "trace": "Catatan Sumber",
    }
    legal = panel_section == "legal"
    linkable = row.get("viewer_highlightable") is True and viewer_target.get("can_resolve") is True
    return {
        "support_id": row.get("evidence_id") or row.get("source_conflict_id") or row.get("relation_id"),
        "support_kind": support_kind,
        "panel_section": labels[panel_section],
        "fact_kind": row.get("fact_kind") or ("legal_text" if legal else "document_structure" if panel_section == "structure" else "source_fact" if panel_section == "metadata" else "source_discrepancy"),
        "display_label": row.get("display_label") or row.get("label") or row.get("citation") or labels[panel_section],
        "display_text": row.get("display_text") or row.get("quoted_text") or row.get("answer") or "",
        "layout_lines": tuple(row.get("layout_lines") or ()),
        "copy_text": row.get("copy_text") or row.get("quoted_text") or "",
        "source_document": row.get("source_document_id"),
        "source_role": row.get("source_role"),
        "page_numbers": tuple(row.get("page_numbers") or ()),
        "legal_citation_available": legal and row.get("citation_final") is True,
        "linkable": linkable,
        "highlightable": linkable,
        "viewer_target": _public_viewer_ref(viewer_target),
    }


def _public_metadata_fact(row: dict) -> dict:
    return {
        "field": row.get("field"),
        "answer": row.get("answer"),
        "evidence_id": row.get("evidence_id"),
    }


def _public_legal_relation(row: dict) -> dict:
    return {
        "relation_type": row.get("relation_type"),
        "source_legal_unit_id": row.get("source_legal_unit_id"),
        "source_legal_unit_role": row.get("source_legal_unit_role"),
        "source_label": row.get("source_label"),
        "target_legal_unit_id": row.get("target_legal_unit_id"),
        "target_label": row.get("target_label"),
    }


def _public_article_relation(row: dict) -> dict:
    return public_article_relation(row)


def _public_viewer_ref(row: dict) -> dict:
    return {
        "action": row.get("action"),
        "source_document_id": row.get("source_document_id"),
        "page_numbers": row.get("page_numbers", ()),
        "bbox_count": row.get("bbox_count"),
        "can_resolve": row.get("can_resolve"),
    }


def handle_pdf_request(
    corpus_id: str,
    payload: dict,
    repo_root: Path | None = None,
    service: LegalRuntimeService | None = None,
) -> dict:
    return (service or _service_for(repo_root)).pdf_access(
        corpus_id,
        _optional_str(payload, "evidence_id"),
        source_document_id=_required_str(payload, "source_document_id"),
        page_number=_required_int(payload, "page_number"),
        source_sha256=_optional_str(payload, "source_sha256"),
        bbox_id=_optional_str(payload, "bbox_id"),
        source_pdf_path=_optional_str(payload, "source_pdf_path"),
    )


@lru_cache(maxsize=1)
def _service_for(repo_root: Path | None) -> LegalRuntimeService:
    return LegalRuntimeService(repo_root)


def _limit(payload: dict, *, default: int) -> int:
    value = payload.get("limit", default)
    try:
        limit = int(value)
    except (TypeError, ValueError):
        raise BadRequest("invalid_limit")
    if limit < 1 or limit > 50:
        raise BadRequest("invalid_limit")
    return limit


def _filters(payload: dict) -> dict | None:
    filters = payload.get("filters")
    if filters is not None and not isinstance(filters, dict):
        raise BadRequest("invalid_filters")
    return filters


def _required_str(payload: dict, field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise BadRequest(f"missing_{field}")
    return value


def _optional_str(payload: dict, field: str) -> str | None:
    value = payload.get(field)
    if value is None:
        return None
    if not isinstance(value, str):
        raise BadRequest(f"invalid_{field}")
    return value


def _optional_int(payload: dict, field: str) -> int | None:
    value = payload.get(field)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        raise BadRequest(f"invalid_{field}")


def _required_int(payload: dict, field: str) -> int:
    value = _optional_int(payload, field)
    if value is None:
        raise BadRequest(f"missing_{field}")
    return value
