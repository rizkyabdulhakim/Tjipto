from __future__ import annotations

from pathlib import Path

from tjipto.runtime.service import LegalRuntimeService


class BadRequest(ValueError):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def handle_request(corpus_id: str, action: str, payload: dict, repo_root: Path | None = None) -> dict:
    service = LegalRuntimeService(repo_root)
    if action == "search":
        return _public_search(service.search(
            corpus_id,
            str(payload.get("query", "")),
            _limit(payload, default=10),
            _filters(payload),
        ))
    if action == "citation":
        return _public_citation_response(service.citation(
            corpus_id,
            str(payload.get("query", "")),
            _optional_str(payload, "source_role"),
            _filters(payload),
        ))
    if action == "viewer":
        return _public_viewer(service.viewer(
            corpus_id,
            _required_str(payload, "evidence_id"),
            source_document_id=_optional_str(payload, "source_document_id"),
            page_number=_optional_int(payload, "page_number"),
            bbox_id=_optional_str(payload, "bbox_id"),
            source_pdf_path=_optional_str(payload, "source_pdf_path"),
        ))
    if action == "ask":
        return _public_ask(service.ask(
            corpus_id,
            str(payload.get("query", "")),
            _limit(payload, default=3),
            _filters(payload),
        ))
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
    return {
        "status": result.get("status"),
        "public_status": result.get("public_status"),
        "corpus_id": result.get("corpus_id"),
        "reason": _public_reason(result.get("reason")),
        "applied_filters": result.get("applied_filters", {}),
        "results": tuple(_public_search_result(row) for row in result.get("results", ())),
    }


def _public_ask(result: dict) -> dict:
    return {
        "status": result.get("status"),
        "public_status": result.get("public_status", result.get("status")),
        "answer_type": result.get("answer_type"),
        "answer": result.get("answer"),
        "reason": _public_reason(result.get("reason")),
        "applied_filters": result.get("applied_filters", {}),
        "citations": tuple(_public_citation(row) for row in result.get("citations", ())),
        "viewer_refs": tuple(_public_viewer_ref(row) for row in result.get("viewer_refs", ())),
    }


def _public_citation_response(result: dict) -> dict:
    return {
        "status": result.get("status"),
        "public_status": result.get("public_status", result.get("status")),
        "answer_type": result.get("answer_type"),
        "reason": _public_reason(result.get("reason")),
        "applied_filters": result.get("applied_filters", {}),
        "citation_payloads": tuple(_public_citation(row) for row in result.get("citation_payloads", ())),
        "viewer_refs": tuple(_public_viewer_ref(row) for row in result.get("viewer_refs", ())),
        "validation_reasons": result.get("validation_reasons", {}),
    }


def _public_viewer(result: dict) -> dict:
    public = {
        "status": result.get("status"),
        "corpus_id": result.get("corpus_id"),
        "evidence_id": result.get("evidence_id"),
        "legal_unit_id": result.get("legal_unit_id"),
        "source_document_id": result.get("source_document_id"),
        "citation": result.get("citation"),
        "quoted_text": result.get("quoted_text"),
        "source_role": result.get("source_role"),
        "temporal_context": result.get("temporal_context"),
        "source_status_label": result.get("source_status_label"),
        "page_numbers": result.get("page_numbers", ()),
        "bbox_count": result.get("bbox_count"),
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


def _public_bbox(row: dict) -> dict:
    return {
        "bbox_id": row.get("bbox_id"),
        "page_number": row.get("page_number"),
        "x0": row.get("x0"),
        "y0": row.get("y0"),
        "x1": row.get("x1"),
        "y1": row.get("y1"),
    }


def _public_reason(reason):
    return reason if reason in {"invalid_query", "unsupported_corpus", "citation_not_found", "insufficient_evidence"} else None


def _public_search_result(row: dict) -> dict:
    return {
        "corpus_id": row.get("corpus_id"),
        "legal_unit_id": row.get("legal_unit_id"),
        "evidence_id": row.get("evidence_id"),
        "citation_id": row.get("citation_id"),
        "viewer_ref_id": row.get("viewer_ref_id"),
        "source_document_id": row.get("source_document_id"),
        "title": row.get("title"),
        "document_title": row.get("document_title"),
        "citation": row.get("citation"),
        "label": row.get("label"),
        "snippet": row.get("snippet"),
        "source_role": row.get("source_role"),
        "temporal_context": row.get("temporal_context"),
        "page_numbers": row.get("page_numbers", ()),
        "bbox_count": row.get("bbox_count"),
        "viewer_ref": _public_viewer_ref(row.get("viewer_ref") or {}),
        "status": row.get("status"),
    }


def _public_citation(row: dict) -> dict:
    return {
        "corpus_id": row.get("corpus_id"),
        "evidence_id": row.get("evidence_id"),
        "legal_unit_id": row.get("legal_unit_id"),
        "source_document_id": row.get("source_document_id"),
        "citation": row.get("citation"),
        "label": row.get("label"),
        "hierarchy": row.get("hierarchy", ()),
        "document_title": row.get("document_title"),
        "quoted_text": row.get("quoted_text"),
        "metadata_answer": row.get("metadata_answer"),
        "metadata_field": row.get("metadata_field"),
        "source_role": row.get("source_role"),
        "temporal_context": row.get("temporal_context"),
        "page_numbers": row.get("page_numbers", ()),
        "bbox_count": row.get("bbox_count"),
        "viewer_ref": _public_viewer_ref(row.get("viewer_ref") or {}),
        "evidence_status": row.get("evidence_status"),
    }


def _public_viewer_ref(row: dict) -> dict:
    return {
        "action": row.get("action"),
        "evidence_id": row.get("evidence_id"),
        "page_numbers": row.get("page_numbers", ()),
        "bbox_count": row.get("bbox_count"),
        "can_resolve": row.get("can_resolve"),
    }


def handle_pdf_request(corpus_id: str, payload: dict, repo_root: Path | None = None) -> dict:
    return LegalRuntimeService(repo_root).pdf_access(
        corpus_id,
        _required_str(payload, "evidence_id"),
        source_document_id=_required_str(payload, "source_document_id"),
        page_number=_required_int(payload, "page_number"),
        source_sha256=_optional_str(payload, "source_sha256"),
        bbox_id=_optional_str(payload, "bbox_id"),
        source_pdf_path=_optional_str(payload, "source_pdf_path"),
    )


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
