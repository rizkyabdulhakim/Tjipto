from __future__ import annotations

from urllib.parse import quote, urlencode

from tjipto.core.manifest import file_sha256

_SOURCE_STATUS_LABELS = {
    "current_consolidated": "Berlaku (konsolidasi saat ini)",
    "amendment_*": "Historis (sumber perubahan)",
    "original_historical": "Historis (naskah asli)",
    "default": "Status sumber tidak tersedia",
}


def viewer_payload(
    store,
    corpus_id: str,
    evidence: dict,
    bboxes: list[dict],
    *,
    source_document_id: str | None = None,
    page_number: int | None = None,
    bbox_id: str | None = None,
    source_pdf_path: str | None = None,
) -> dict:
    base = _base_payload(store, corpus_id, evidence, bboxes)
    pdf = resolve_pdf_access(
        store,
        corpus_id,
        evidence,
        bboxes,
        source_document_id=source_document_id,
        page_number=page_number,
        bbox_id=bbox_id,
        source_pdf_path=source_pdf_path,
    )
    if pdf["status"] != "pdf_access_ready":
        return base | _unavailable(pdf["reason"])
    if evidence.get("bbox_precision") == "page_grounded_only":
        return base | _trace_only(pdf, "source_page_trace_only", "page_grounded_only_not_answerable")
    if evidence.get("viewer_highlightable") is False:
        return base | _trace_only(pdf, "non_highlightable_trace", "viewer_not_highlightable")

    return base | {
        "status": "viewer_payload_ready",
        "pdf_access_available": True,
        "rendering_available": True,
        "render_status": "pdf_access_available",
        "page_number": pdf["page_number"],
        "bbox_rectangles": tuple(bboxes),
        "pdf": {
            "mime_type": "application/pdf",
            "access_url": pdf["access_url"],
        },
    }


def resolve_pdf_access(
    store,
    corpus_id: str,
    evidence: dict,
    bboxes: list[dict],
    *,
    source_document_id: str | None = None,
    page_number: int | None = None,
    bbox_id: str | None = None,
    source_sha256: str | None = None,
    source_pdf_path: str | None = None,
) -> dict:
    error = _validate_request(evidence, bboxes, source_document_id, page_number, bbox_id, source_pdf_path)
    if error:
        return _pdf_unavailable(error)

    page = page_number or int((evidence.get("page_numbers") or [0])[0])
    page_bboxes = tuple(row for row in bboxes if row.get("page_number") == page)
    if bbox_id:
        page_bboxes = tuple(row for row in page_bboxes if row.get("bbox_id") == bbox_id)
    if not page_bboxes:
        return _pdf_unavailable("invalid_bbox")
    if source_sha256 is not None and source_sha256 != evidence.get("source_sha256"):
        return _pdf_unavailable("source_hash_mismatch")

    source = _source_document(store, evidence)
    if source is None:
        return _pdf_unavailable("invalid_source")
    pdf_path = _safe_pdf_path(store, evidence, source)
    if pdf_path is None:
        return _pdf_unavailable("invalid_source")

    try:
        if file_sha256(pdf_path) != evidence.get("source_sha256") or source.get("sha256") != evidence.get("source_sha256"):
            return _pdf_unavailable("source_hash_mismatch")
    except (OSError, ValueError):
        return _pdf_unavailable("render_failed")

    first_box = page_bboxes[0]
    return {
        "status": "pdf_access_ready",
        "path": pdf_path,
        "mime_type": "application/pdf",
        "page_number": page,
        "page_width": first_box.get("page_width"),
        "page_height": first_box.get("page_height"),
        "bbox_rectangles": page_bboxes,
        "source_sha256": evidence.get("source_sha256"),
        "access_url": _pdf_access_url(corpus_id, evidence, page, bbox_id),
    }


def _base_payload(store, corpus_id: str, evidence: dict, bboxes: list[dict]) -> dict:
    return {
        "status": "viewer_payload_ready",
        "corpus_id": corpus_id,
        "evidence_id": evidence["evidence_id"],
        "legal_unit_id": evidence.get("legal_unit_id"),
        "source_document_id": evidence.get("source_document_id"),
        "citation": evidence["citation"],
        "quoted_text": evidence["quoted_text"],
        "source_role": evidence.get("source_role"),
        "temporal_context": evidence.get("temporal_context"),
        "source_status_label": _source_status_label(evidence, store),
        "page_numbers": evidence["page_numbers"],
        "bbox_count": len(bboxes),
        "bbox_rectangles": tuple(bboxes),
        "pdf_access_available": False,
        "rendering_available": False,
        "render_status": "render_unavailable",
        "reason": None,
        "bbox_precision": evidence.get("bbox_precision"),
        "viewer_highlightable": evidence.get("viewer_highlightable"),
    }


def _validate_request(
    evidence: dict,
    bboxes: list[dict],
    source_document_id: str | None,
    page_number: int | None,
    bbox_id: str | None,
    source_pdf_path: str | None,
) -> str | None:
    if source_pdf_path is not None:
        return "invalid_source"
    if source_document_id is not None and source_document_id != evidence.get("source_document_id"):
        return "invalid_source"
    pages = set(evidence.get("page_numbers") or ())
    if page_number is not None and page_number not in pages:
        return "invalid_page"
    bbox_refs = set(evidence.get("bbox_refs") or ())
    if bbox_id is not None and bbox_id not in bbox_refs:
        return "invalid_bbox"
    if any(row.get("bbox_id") not in bbox_refs for row in bboxes):
        return "invalid_bbox"
    return None


def _source_document(store, evidence: dict) -> dict | None:
    for row in store.source_documents:
        if row.get("source_document_id") == evidence.get("source_document_id"):
            return row
    return None


def _safe_pdf_path(store, evidence: dict, source: dict):
    if source.get("path") != evidence.get("source_pdf_path"):
        return None
    try:
        path = store.config.source_path(source["path"])
    except ValueError:
        return None
    if not path.exists() or path.suffix.casefold() != ".pdf":
        return None
    return path


def _unavailable(reason: str) -> dict:
    return {
        "status": "viewer_payload_ready",
        "rendering_available": False,
        "render_status": "render_unavailable" if reason != "render_failed" else "render_failed_safe",
        "reason": reason,
    }


def _trace_only(pdf: dict, status: str, reason: str) -> dict:
    return {
        "status": status,
        "pdf_access_available": True,
        "rendering_available": False,
        "render_status": status,
        "reason": reason,
        "page_number": pdf["page_number"],
        "bbox_rectangles": (),
        "pdf": {
            "mime_type": "application/pdf",
            "access_url": pdf["access_url"],
        },
    }


def _pdf_unavailable(reason: str) -> dict:
    return {"status": "pdf_access_unavailable", "reason": reason}


def _pdf_access_url(corpus_id: str, evidence: dict, page_number: int, bbox_id: str | None) -> str:
    query = {
        "evidence_id": evidence["evidence_id"],
        "source_document_id": evidence["source_document_id"],
        "page_number": str(page_number),
    }
    if bbox_id:
        query["bbox_id"] = bbox_id
    return f"/legal/{quote(corpus_id, safe='')}/pdf?{urlencode(query)}"


def _source_status_label(evidence: dict, store=None) -> str:
    role = str(evidence.get("source_role") or evidence.get("temporal_context") or "")
    labels = _SOURCE_STATUS_LABELS | dict(
        getattr(getattr(store, "config", None), "setting", lambda *args: {})("viewer_source_status_labels", {})
        or {}
    )
    if role in labels:
        return labels[role]
    if role.startswith("amendment_"):
        return labels["amendment_*"]
    return labels["default"]
