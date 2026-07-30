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
        return _public_integrity(result)
    return {
        "kind": "document",
        "status": result.get("status"),
        "reason": _public_reason(result.get("reason")),
        "results": tuple(_public_search_result(row, service, corpus_id) for row in result.get("results", ())),
    }


def _public_ask(result: dict, service: LegalRuntimeService, corpus_id: str) -> dict:
    if result.get("readiness") is False:
        return _public_integrity(result)
    support_rows = (
        *result.get("final_citations", result.get("citations", ())),
        *(() if result.get("answer_type") == "article_amendment_relation" else result.get("historical_citations", ())),
        *result.get("metadata_support", ()),
        *result.get("structural_support", ()),
        *result.get("relation_support", ()),
        *result.get("trace_support", ()),
    )
    projected = tuple((_public_support(row, service, corpus_id), row) for row in support_rows)
    supports = tuple(support for support, _ in projected)
    if result.get("clarification_options"):
        return {
            "kind": "clarification",
            "status": result.get("status"),
            "answer": result.get("answer"),
            "reason": _public_reason(result.get("reason")),
            "clarification_options": tuple(_public_clarification_option(row) for row in result["clarification_options"]),
        }
    if result.get("document_source") is not None:
        source = result["document_source"]
        target = service.register_public_target(corpus_id, {"evidence_id": None, "source_document_id": source.get("source_document_id")})
        return {
            "kind": "document",
            "status": result.get("status"),
            "document": {
                "label": source.get("document_title"),
                "source_role": source.get("source_role"),
                "viewer_target": _public_target(target, "open_document", (1,), True),
            },
        }
    return {
        "kind": "answer",
        "status": result.get("status"),
        "answer": result.get("answer"),
        "answer_scope": result.get("answer_scope"),
        "reason": _public_reason(result.get("reason")),
        "supports": supports,
        "support_groups": _support_groups(projected, service, corpus_id),
    }


def _public_citation_response(result: dict, service: LegalRuntimeService, corpus_id: str) -> dict:
    if result.get("readiness") is False:
        return _public_integrity(result)
    projected = tuple((_public_support(row, service, corpus_id), row) for row in result.get("citation_payloads", ()))
    supports = tuple(support for support, _ in projected)
    return {
        "kind": "answer",
        "status": result.get("status"),
        "reason": _public_reason(result.get("reason")),
        "supports": supports,
        "support_groups": _support_groups(projected, service, corpus_id),
    }


def _public_viewer(result: dict, corpus_id: str) -> dict:
    if result.get("readiness") is False:
        return _public_integrity(result)
    public = {
        "kind": "document",
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
        "kind": "unavailable",
        "status": "unavailable",
        "reason": "service_unavailable",
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


def _public_support(row: dict, service: LegalRuntimeService, corpus_id: str) -> dict:
    authority = row.get("authority_kind")
    authority_kind = authority if isinstance(authority, str) and authority in {
        "legal_citation", "metadata_source", "metadata_trace", "source_conflict_provenance",
        "source_anomaly", "structural_context", "instrument_provenance", "source_text",
    } else "source_text"
    fact_kind = row.get("fact_kind") or {
        "legal_citation": "legal_text",
        "metadata_source": "source_fact",
        "metadata_trace": "source_fact",
        "structural_context": "document_structure",
        "instrument_provenance": "source_provenance",
        "source_conflict_provenance": "source_discrepancy",
        "source_anomaly": "source_discrepancy",
        "source_text": "source_trace",
    }.get(authority_kind, "source_trace")
    role_label = row.get("printed_role") if fact_kind == "person_role" else None
    target_source = dict(row.get("viewer_target") or row.get("viewer_ref") or {})
    linkable = row.get("viewer_highlightable") is True and target_source.get("can_resolve") is True
    target = service.register_public_target(corpus_id, {
        "evidence_id": row.get("evidence_id") or row.get("source_conflict_id") or row.get("relation_id"),
        "proposition_id": row.get("proposition_id"),
        "relation_id": row.get("relation_id"),
        "source_document_id": row.get("source_document_id"),
        "bbox_refs": tuple(row.get("bbox_refs") or ()),
        "quoted_text": row.get("display_text") or row.get("quoted_text") or "",
        "support_projection": {
            key: row[key]
            for key in (
                "display_text", "presentation_as_legal_quote", "citation_final",
            )
            if key in row
        },
    }) if linkable else None
    return {
        "public_support_id": service.public_identifier(corpus_id, "support", target or (row.get("display_label"), row.get("source_role"), authority_kind)),
        "authority_kind": authority_kind,
        "citation_final": row.get("citation_final") is True and authority_kind == "legal_citation",
        "support_kind": row.get("support_kind") or "trace_support",
        "fact_kind": fact_kind,
        "label": row.get("printed_name") or role_label or row.get("display_label") or row.get("label") or row.get("citation") or row.get("authority_label") or "Bukti sumber",
        "role_label": role_label,
        "text": row.get("display_text") or row.get("quoted_text") or row.get("answer") or "",
        "source_label": row.get("document_title") or row.get("source_label"),
        "source_role": row.get("source_role"),
        "page_numbers": tuple(row.get("page_numbers") or ()),
        "viewer_target": _public_target(target, "viewer", row.get("page_numbers") or (), linkable),
    }


def _public_target(target: str | None, action: str, pages: object, resolvable: bool) -> dict:
    page_numbers = tuple(pages) if isinstance(pages, (list, tuple)) else ()
    return {
        "action": action,
        "public_target_id": target,
        "page_numbers": page_numbers,
        "can_resolve": resolvable and target is not None,
    }


def _support_groups(projected: tuple[tuple[dict, dict], ...], service: LegalRuntimeService, corpus_id: str) -> tuple[dict, ...]:
    grouped: dict[tuple[str, ...], list[dict]] = {}
    role_counts: dict[tuple[str, str, str], int] = {}
    entity_counts: dict[str, int] = {}
    for support, row in projected:
        if support.get("authority_kind") != "metadata_source" or support.get("fact_kind") != "person_role":
            continue
        role_key = (str(support.get("source_label") or ""), str(support.get("source_role") or ""), str(support.get("role_label") or ""))
        role_counts[role_key] = role_counts.get(role_key, 0) + 1
        identity = row.get("entity_identity")
        if identity:
            entity_counts[identity] = entity_counts.get(identity, 0) + 1
    for support, row in projected:
        key, group_kind = _support_group_key(support, row, role_counts, entity_counts)
        if group_kind:
            grouped.setdefault((group_kind, *key), []).append(support)
    return tuple(
        {
            "public_group_id": service.public_identifier(corpus_id, "support-group", key),
            "group_kind": key[0],
            "label": _support_group_label(key[0], members),
            "summary": members[0]["source_label"] or members[0]["label"],
            "member_count": len(members),
            "members": tuple(members),
        }
        for key, members in sorted(grouped.items())
    )


def _support_group_key(
    support: dict,
    row: dict,
    role_counts: dict[tuple[str, str, str], int],
    entity_counts: dict[str, int],
) -> tuple[tuple[str, ...], str]:
    """Group only source facts with an explicit, deterministic common owner."""
    authority_kind = str(support.get("authority_kind") or "")
    source = str(support.get("source_label") or "")
    source_role = str(support.get("source_role") or "")
    fact_kind = str(support.get("fact_kind") or "")
    if authority_kind == "metadata_source" and fact_kind == "person_role":
        role_key = (source, source_role, str(support.get("role_label") or ""))
        if role_counts.get(role_key, 0) > 1:
            return role_key, "role_members"
        name = row.get("entity_identity")
        if name and entity_counts.get(name, 0) > 1:
            return (name, str(support.get("role_label") or "")), "entity_occurrences"
        return (), ""
    if authority_kind == "metadata_source":
        return (source, source_role), "document_metadata"
    return (), ""


def _support_group_label(group_kind: str, members: list[dict]) -> str:
    if group_kind == "role_members" and members[0].get("role_label"):
        return f"{members[0]['role_label']} · {len(members)} orang"
    if group_kind == "entity_occurrences":
        return str(members[0].get("label") or "Sumber dokumen")
    return str(members[0].get("source_label") or members[0].get("label") or "Sumber dokumen")


def _public_clarification_option(row: dict) -> dict:
    return {"source_role": row.get("source_role"), "label": row.get("label")}


def _public_capabilities(result: dict) -> dict:
    if result.get("readiness") is False:
        return _public_integrity(result)
    return {"status": "ok", "capabilities": ("search", "ask", "citation", "viewer", "bookmarks")}


def _public_bookmarks(result: dict) -> dict:
    if result.get("readiness") is False:
        return _public_integrity(result)
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
