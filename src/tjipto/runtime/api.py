from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from tjipto.runtime.service import LegalRuntimeService


class BadRequest(ValueError):
    def __init__(self, reason: str = "invalid_request"):
        self.reason = reason
        super().__init__(reason)


_PUBLIC_REQUEST_FIELDS = {
    "ask": {"query", "limit", "filters"},
    "search": {"query", "limit", "filters"},
    "citation": {"query", "source_role", "filters"},
    "viewer": {"target"},
    "bookmark": {"target", "note"},
    "bookmarks": set(),
    "capabilities": set(),
}


def handle_request(
    corpus_id: str,
    action: str,
    payload: dict,
    repo_root: Path | None = None,
    service: LegalRuntimeService | None = None,
) -> dict:
    _validate_payload(action, payload)
    service = service or _service_for(repo_root)
    if action == "search":
        return _public_search(service.search(corpus_id, _query(payload), _limit(payload, default=10), _filters(payload)), service, corpus_id)
    if action == "citation":
        result = service.citation(corpus_id, _query(payload), _optional_str(payload, "source_role"), _filters(payload))
        return _public_citation_response(result, service, corpus_id)
    if action == "viewer":
        return _public_viewer(service.viewer_public(corpus_id, _required_str(payload, "target")), corpus_id)
    if action == "ask":
        result = service.ask(corpus_id, _query(payload), _limit(payload, default=3), _filters(payload))
        return _public_ask(result, service, corpus_id)
    if action == "capabilities":
        return _public_capabilities(service.capabilities(corpus_id))
    if action == "bookmarks":
        return _public_bookmarks(service.bookmarks(corpus_id))
    if action == "bookmark":
        return _public_bookmark(service.bookmark_public(corpus_id, _required_str(payload, "target"), _optional_str(payload, "note")))
    return {"status": "unsupported_action"}


def _public_search(result: dict, service: LegalRuntimeService, corpus_id: str) -> dict:
    if result.get("readiness") is False:
        return _public_integrity(result) | {"results": ()}
    return {
        "status": result.get("status"),
        "reason": _public_reason(result.get("reason")),
        "results": tuple(_public_search_result(row, service, corpus_id) for row in result.get("results", ())),
    }


def _public_ask(result: dict, service: LegalRuntimeService, corpus_id: str) -> dict:
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
    supports = tuple(_public_support(row, section, service, corpus_id) for row, section in support_rows)
    public = {
        "status": result.get("status"),
        "answer": result.get("answer"),
        "answer_type": "source_document" if result.get("document_source") is not None else "answer",
        "answer_scope": result.get("answer_scope"),
        "reason": _public_reason(result.get("reason")),
        "supports": supports,
        "support_groups": _support_groups(supports, service, corpus_id),
    }
    if result.get("clarification_options"):
        public["clarification_options"] = tuple(_public_clarification_option(row) for row in result["clarification_options"])
    if result.get("document_source") is not None:
        source = result["document_source"]
        target = service.register_public_target(corpus_id, {"evidence_id": None, "source_document_id": source.get("source_document_id")})
        public["document_source"] = {
            "label": source.get("document_title"),
            "source_role": source.get("source_role"),
            "viewer_target": _public_target(target, "open_document", (1,), True),
        }
    return public


def _public_citation_response(result: dict, service: LegalRuntimeService, corpus_id: str) -> dict:
    if result.get("readiness") is False:
        return _public_integrity(result) | {"supports": (), "support_groups": ()}
    supports = tuple(_public_support(row, "legal", service, corpus_id) for row in result.get("citation_payloads", ()))
    return {
        "status": result.get("status"),
        "reason": _public_reason(result.get("reason")),
        "supports": supports,
        "support_groups": _support_groups(supports, service, corpus_id),
    }


def _public_viewer(result: dict, corpus_id: str) -> dict:
    if result.get("readiness") is False:
        return _public_integrity(result)
    public = {
        "status": result.get("status"),
        "citation": result.get("citation"),
        "quoted_text": result.get("quoted_text"),
        "source_role": result.get("source_role"),
        "source_status_label": result.get("source_status_label"),
        "page_numbers": tuple(result.get("page_numbers") or ()),
        "bbox_rectangles": tuple(_public_bbox(row, index) for index, row in enumerate(result.get("bbox_rectangles", ()), start=1)),
        "viewer_highlightable": result.get("viewer_highlightable") is True,
        "pdf_access_available": result.get("pdf_access_available") is True,
        "rendering_available": result.get("rendering_available") is True,
        "reason": _public_reason(result.get("reason")),
    }
    if result.get("pdf_access_available") and result.get("public_pdf_target"):
        public["pdf"] = {
            "mime_type": "application/pdf",
            "access_url": f"/legal/{corpus_id}/pdf?target={result['public_pdf_target']}",
        }
    return public


def _public_integrity(result: dict) -> dict:
    return {
        "status": "unavailable",
        "reason": "service_unavailable",
        "supports": (),
        "support_groups": (),
    }


def _public_bbox(row: dict, index: int = 1) -> dict:
    precision = _public_bbox_precision(row.get("bbox_precision"))
    return {
        "public_rectangle_id": f"r{index}",
        "page_number": row.get("page_number"),
        "x0": row.get("x0"),
        "y0": row.get("y0"),
        "x1": row.get("x1"),
        "y1": row.get("y1"),
        "page_width": row.get("page_width"),
        "page_height": row.get("page_height"),
        "bbox_precision": precision,
        "viewer_highlightable": precision == "exact" and row.get("viewer_highlightable") is True,
        "coordinate_space": "pdf_user_space",
        "coordinate_origin": "top_left",
        "page_rotation": 0,
        "page_box_basis": "media_box",
        "transform_version": "pymupdf_top_left_v1",
    }


def _public_bbox_precision(value: object) -> str:
    return value if value in {"exact", "coarse", "page_grounded_only"} else "page_grounded_only"


def _public_reason(reason: object) -> str | None:
    return str(reason) if reason in {
        "invalid_query", "unsupported_corpus", "citation_not_found", "insufficient_evidence",
        "page_grounded_only_not_answerable", "viewer_not_highlightable", "missing_exact_grounding",
        "document_not_found", "unresolved_source_scope", "invalid_viewer_target",
    } else None


def _public_search_result(row: dict, service: LegalRuntimeService, corpus_id: str) -> dict:
    document = row.get("status") == "document"
    target = service.register_public_target(corpus_id, {"evidence_id": None if document else row.get("evidence_id"), "source_document_id": row.get("source_document_id")})
    return {
        "title": row.get("title"),
        "label": row.get("label"),
        "snippet": row.get("snippet"),
        "source_role": row.get("source_role"),
        "page_numbers": tuple(row.get("page_numbers") or ()),
        "viewer_target": _public_target(target, "viewer", row.get("page_numbers") or (), True),
    }


def _public_support(row: dict, panel_section: str, service: LegalRuntimeService, corpus_id: str) -> dict:
    labels = {"legal": "Kutipan Relevan", "metadata": "Sumber Dokumen", "structure": "Struktur Dokumen", "trace": "Catatan Sumber"}
    legal = panel_section == "legal"
    fact_kind = row.get("fact_kind") or ("legal_text" if legal else "document_structure" if panel_section == "structure" else "source_fact" if panel_section == "metadata" else "source_discrepancy")
    role_label = row.get("printed_role") if fact_kind == "person_role" else None
    target_source = dict(row.get("viewer_target") or row.get("viewer_ref") or {})
    linkable = row.get("viewer_highlightable") is True and target_source.get("can_resolve") is True
    target = service.register_public_target(corpus_id, {
        "evidence_id": row.get("evidence_id") or row.get("source_conflict_id") or row.get("relation_id"),
        "relation_id": row.get("relation_id"),
        "source_document_id": row.get("source_document_id"),
    }) if linkable else None
    layout = tuple(_public_layout_line(item, index) for index, item in enumerate(row.get("layout_lines") or ()))
    return {
        "public_support_id": service.public_identifier(corpus_id, "support", target or (row.get("display_label"), row.get("source_role"), panel_section)),
        "support_kind": row.get("support_kind") or ("metadata_source" if panel_section == "metadata" else "structural_provenance" if panel_section == "structure" else "trace_support"),
        "panel_section": labels[panel_section],
        "fact_kind": fact_kind,
        "label": role_label or row.get("display_label") or row.get("label") or row.get("citation") or labels[panel_section],
        "role_label": role_label,
        "text": row.get("display_text") or row.get("quoted_text") or row.get("answer") or "",
        "layout_lines": layout,
        "copy_text": row.get("copy_text") or row.get("quoted_text") or "",
        "source_label": row.get("document_title") or row.get("source_label"),
        "source_role": row.get("source_role"),
        "page_numbers": tuple(row.get("page_numbers") or ()),
        "legal_citation_available": legal and row.get("citation_final") is True,
        "relevant_quote_eligible": legal and row.get("relevant_quote_eligible") is True,
        "viewer_target": _public_target(target, "viewer", row.get("page_numbers") or (), linkable),
    }


def _public_layout_line(row: dict, index: int) -> dict:
    alignment = row.get("alignment") if row.get("alignment") in {"left", "center", "right", "justify"} else "unknown"
    return {
        "text": str(row.get("text") or ""),
        "line_order": int(row.get("line_order") or index),
        "paragraph_id": str(row.get("paragraph_id") or index),
        "alignment": alignment,
        "indent": float(row.get("indent") or 0.0),
    }


def _public_target(target: str | None, action: str, pages: object, resolvable: bool) -> dict:
    page_numbers = tuple(pages) if isinstance(pages, (list, tuple)) else ()
    return {
        "action": action,
        "public_target_id": target,
        "page_numbers": page_numbers,
        "can_resolve": resolvable and target is not None,
    }


def _support_groups(supports: tuple[dict, ...], service: LegalRuntimeService, corpus_id: str) -> tuple[dict, ...]:
    grouped: dict[tuple[str, ...], list[dict]] = {}
    for support in supports:
        key = (
            str(support.get("source_label") or "").casefold(),
            str(support.get("panel_section") or ""),
            str(support.get("fact_kind") or ""),
            str(support.get("role_label") or support.get("label") or "").casefold(),
            str(support.get("source_role") or ""),
        )
        grouped.setdefault(key, []).append(support)
    return tuple(
        {
            "public_group_id": service.public_identifier(corpus_id, "support-group", key),
            "panel_section": members[0]["panel_section"],
            "label": members[0]["label"],
            "summary": members[0]["source_label"] or members[0]["label"],
            "member_count": len(members),
            "members": tuple(members),
        }
        for key, members in grouped.items()
    )


def _public_clarification_option(row: dict) -> dict:
    return {"source_role": row.get("source_role"), "label": row.get("label")}


def _public_capabilities(result: dict) -> dict:
    if result.get("readiness") is False:
        return _public_integrity(result)
    return {"status": "ok", "capabilities": ("search", "ask", "citation", "viewer", "bookmarks")}


def _public_bookmarks(result: dict) -> dict:
    if result.get("readiness") is False:
        return _public_integrity(result) | {"bookmarks": ()}
    return {
        "status": result.get("status"),
        "bookmarks": tuple({
            "public_bookmark_id": row.get("bookmark_id"),
            "public_target_id": row.get("public_target"),
            "note": row.get("note"),
            "created_at": row.get("created_at"),
            "status": row.get("status"),
        } for row in result.get("bookmarks", ())),
    }


def _public_bookmark(result: dict) -> dict:
    if result.get("status") != "saved":
        return {"status": "unavailable", "reason": "bookmark_target_unavailable"}
    row = result["bookmark"]
    return {
        "status": "saved",
        "bookmark": {
            "public_bookmark_id": row.get("bookmark_id"),
            "public_target_id": row.get("public_target"),
            "note": row.get("note"),
            "created_at": row.get("created_at"),
            "status": row.get("status"),
        },
    }


def handle_pdf_request(corpus_id: str, payload: dict, repo_root: Path | None = None, service: LegalRuntimeService | None = None) -> dict:
    _validate_payload("pdf", payload)
    return (service or _service_for(repo_root)).pdf_public(corpus_id, _required_str(payload, "target"))


@lru_cache(maxsize=1)
def _service_for(repo_root: Path | None) -> LegalRuntimeService:
    return LegalRuntimeService(repo_root)


def _validate_payload(action: str, payload: object) -> None:
    allowed = _PUBLIC_REQUEST_FIELDS.get(action, {"target"} if action == "pdf" else None)
    if not isinstance(payload, dict) or allowed is None or set(payload) - allowed:
        raise BadRequest()
    filters = payload.get("filters")
    if filters is not None and (not isinstance(filters, dict) or set(filters) - {"source_role"}):
        raise BadRequest()


def _query(payload: dict) -> str:
    value = payload.get("query", "")
    if not isinstance(value, str):
        raise BadRequest()
    return value


def _limit(payload: dict, *, default: int) -> int:
    value = payload.get("limit", default)
    if not isinstance(value, int) or not 1 <= value <= 50:
        raise BadRequest()
    return value


def _filters(payload: dict) -> dict | None:
    return payload.get("filters")


def _required_str(payload: dict, field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise BadRequest()
    return value


def _optional_str(payload: dict, field: str) -> str | None:
    value = payload.get(field)
    if value is None:
        return None
    if not isinstance(value, str):
        raise BadRequest()
    return value
