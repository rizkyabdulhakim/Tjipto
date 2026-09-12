from __future__ import annotations

from urllib.parse import quote, urlencode
import re
from typing import Any

from tjipto.core.manifest import file_sha256
from tjipto.corpora.source_arbitration import (
    source_conflict_viewer_evidence,
    source_document_by_id,
)
from tjipto.evidence.bbox import viewer_overlay_rectangles
from tjipto.retrieval.answer import validate_answer_candidate

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
    highlightable = _highlightable_bboxes(evidence, bboxes)
    if evidence.get("bbox_precision") == "page_grounded_only":
        return base | _trace_only(pdf, "source_page_trace_only", "page_grounded_only_not_answerable")
    if not highlightable:
        return base | _trace_only(pdf, "non_highlightable_trace", "viewer_not_highlightable")

    return base | {
        "status": "viewer_payload_ready",
        "pdf_access_available": True,
        "rendering_available": True,
        "render_status": "pdf_access_available",
        "page_number": pdf["page_number"],
        "bbox_rectangles": highlightable,
        "pdf": {
            "mime_type": "application/pdf",
            "access_url": pdf["access_url"],
        },
    }


def document_viewer_payload(
    store,
    corpus_id: str,
    source: dict,
    *,
    page_number: int | None = None,
    source_pdf_path: str | None = None,
) -> dict:
    pdf = resolve_document_pdf_access(
        store,
        corpus_id,
        source,
        page_number=page_number,
        source_pdf_path=source_pdf_path,
    )
    base = {
        "status": "viewer_payload_ready",
        "corpus_id": corpus_id,
        "source_document_id": source.get("source_document_id"),
        "citation": _document_title(store, source),
        "quoted_text": "",
        "source_role": source.get("source_role"),
        "temporal_context": source.get("temporal_context"),
        "source_status_label": _source_status_label(source, store),
        "page_numbers": (page_number or 1,),
        "bbox_count": 0,
        "bbox_rectangles": (),
        "pdf_access_available": False,
        "rendering_available": False,
        "render_status": "render_unavailable",
        "reason": None,
        "bbox_precision": "page_grounded_only",
        "viewer_highlightable": False,
    }
    if pdf["status"] != "pdf_access_ready":
        return base | _unavailable(pdf["reason"])
    return base | {
        "pdf_access_available": True,
        "rendering_available": True,
        "render_status": "pdf_access_available",
        "page_number": pdf["page_number"],
        "pdf": {
            "mime_type": "application/pdf",
            "access_url": pdf["access_url"],
        },
    }


def viewer_request(
    store,
    corpus_id: str,
    evidence_id: str | None = None,
    *,
    support_unit_id: str | None = None,
    source_support_id: str | None = None,
    relation_id: str | None = None,
    source_document_id: str | None = None,
    page_number: int | None = None,
    bbox_id: str | None = None,
    bbox_refs: tuple[str, ...] = (),
    proposition_id: str | None = None,
    quoted_text: str | None = None,
    support_projection: dict | None = None,
    source_pdf_path: str | None = None,
) -> dict:
    """Resolve one validated viewer request against an already verified store."""
    support = store.meaningful_support_unit(support_unit_id) if support_unit_id else None
    if support_unit_id and (
        support is None or support.get("decision_kind") == "typed_exclusion" or support.get("viewer_eligible") is not True
    ):
        return {"status": "not_found", "reason": "invalid_support_target", "corpus_id": corpus_id}
    synthetic_bboxes: list[dict] | None = list(support.get("bbox_rectangles") or ()) if support is not None else None
    if support is not None:
        source_document_id = str(support["source_document_id"])
        page_number = int(support["page_numbers"][0])
        bbox_refs = tuple(support.get("bbox_refs") or ())
        quoted_text = "\n".join(
            str(store.page_text_span(span_id).get("exact_quote") or "")
            for span_id in support.get("text_span_ids") or ()
            if store.page_text_span(span_id) is not None
        )
        if support.get("owner_type") == "review_decision":
            raw = store.source_span(str((support.get("raw_source_span_ids") or ("",))[0]))
            evidence_id = str(raw.get("source_support_id")) if raw else None
        else:
            evidence_id = str(support["owner_id"])
        if support.get("bbox_precision") == "page_grounded_only":
            source = source_document_by_id(store, source_document_id)
            if source is None:
                return {"status": "not_found", "reason": "invalid_source", "corpus_id": corpus_id}
            return document_viewer_payload(store, corpus_id, source, page_number=page_number) | {
                "quoted_text": quoted_text,
                "page_numbers": tuple(support["page_numbers"]),
                "source_role": support["source_role"],
                "temporal_context": support["temporal_context"],
            }
    evidence_id = evidence_id or source_support_id
    if evidence_id is None:
        source = source_document_by_id(store, source_document_id)
        if source is None:
            return {"status": "not_found", "reason": "invalid_source", "corpus_id": corpus_id}
        return document_viewer_payload(
            store,
            corpus_id,
            source,
            page_number=page_number,
            source_pdf_path=source_pdf_path,
        )
    evidence = store.get(evidence_id) or _metadata_grounding_evidence(store, evidence_id)
    if evidence is None:
        evidence, synthetic_bboxes = source_conflict_viewer_evidence(store, evidence_id)
    if evidence is None:
        evidence = _source_span_evidence(store, evidence_id)
        synthetic_bboxes = store.source_span_bboxes(evidence_id) if evidence is not None else None
    if evidence is None:
        return {"status": "not_found", "reason": "invalid_evidence", "corpus_id": corpus_id}
    if proposition_id is not None:
        proposition = next(
            (
                row
                for row in store.propositions
                if row.get("proposition_id") == proposition_id
                and row.get("legal_unit_id") == evidence.get("legal_unit_id")
                and row.get("source_document_id") == evidence.get("source_document_id")
                and tuple(row.get("bbox_refs") or ()) == bbox_refs
            ),
            None,
        )
        overlay = viewer_overlay_rectangles(proposition or {})
        if proposition is None or not overlay:
            return {"status": "not_found", "reason": "invalid_viewer_target", "corpus_id": corpus_id}
        evidence = evidence | {
            "bbox_refs": bbox_refs,
            "quoted_text": proposition.get("exact_quote"),
            "page_numbers": tuple(proposition.get("page_numbers") or ()),
        }
        synthetic_bboxes = list(overlay)
    relation = _relation_for_evidence(store, evidence_id, relation_id) if relation_id is not None else None
    if relation_id is not None and relation is None:
        return {"status": "not_found", "reason": "invalid_viewer_target", "corpus_id": corpus_id}
    if relation is not None:
        evidence = evidence | {
            "bbox_refs": tuple(relation.get("bbox_refs") or ()),
            "quoted_text": relation.get("quoted_text") or evidence.get("quoted_text"),
        }
    if quoted_text is not None:
        evidence = evidence | {"quoted_text": quoted_text}
    if support is not None:
        evidence = evidence | {
            "bbox_refs": bbox_refs,
            "bbox_precision": support["bbox_precision"],
            "viewer_highlightable": support.get("highlight_eligible") is True,
            "page_numbers": tuple(support["page_numbers"]),
            "source_document_id": support["source_document_id"],
            "source_role": support["source_role"],
            "temporal_context": support["temporal_context"],
        }
        synthetic_bboxes = list(support.get("bbox_rectangles") or ())
    if support_projection:
        evidence = evidence | {
            key: support_projection[key]
            for key in (
                "display_text",
                "copy_text",
                "layout_lines",
                "presentation_as_legal_quote",
                "citation_final",
                "relevant_quote_eligible",
            )
            if key in support_projection
        }
    override_refs = bbox_refs if proposition_id is None and support is None else ()
    bboxes = _request_bboxes(store, evidence_id, evidence, relation, synthetic_bboxes, override_refs)
    if proposition_id is None:
        bboxes = _select_viewer_bboxes(bboxes, bbox_refs)
    if bbox_refs and not bboxes:
        return {"status": "not_found", "reason": "invalid_viewer_target", "corpus_id": corpus_id}
    return _viewer_with_authority(
        store,
        evidence,
        viewer_payload(
            store,
            corpus_id,
            evidence,
            bboxes,
            source_document_id=source_document_id,
            page_number=page_number,
            bbox_id=bbox_id,
            source_pdf_path=source_pdf_path,
        ),
    )


def pdf_access_request(
    store,
    corpus_id: str,
    evidence_id: str | None,
    *,
    source_support_id: str | None = None,
    relation_id: str | None = None,
    source_document_id: str,
    page_number: int,
    source_sha256: str | None = None,
    bbox_id: str | None = None,
    bbox_refs: tuple[str, ...] = (),
    source_pdf_path: str | None = None,
) -> dict:
    """Resolve one validated PDF access request against a verified store."""
    evidence_id = evidence_id or source_support_id
    if evidence_id is None:
        source = source_document_by_id(store, source_document_id)
        if source is None:
            return {"status": "not_found", "reason": "invalid_source", "corpus_id": corpus_id}
        return resolve_document_pdf_access(
            store,
            corpus_id,
            source,
            page_number=page_number,
            source_pdf_path=source_pdf_path,
        )
    evidence = store.get(evidence_id) or _metadata_grounding_evidence(store, evidence_id)
    synthetic_bboxes: list[dict] | None = None
    if evidence is None:
        evidence, synthetic_bboxes = source_conflict_viewer_evidence(store, evidence_id)
    if evidence is None:
        evidence = _source_span_evidence(store, evidence_id)
        synthetic_bboxes = store.source_span_bboxes(evidence_id) if evidence is not None else None
    if evidence is None:
        return {"status": "not_found", "reason": "invalid_evidence", "corpus_id": corpus_id}
    relation = _relation_for_evidence(store, evidence_id, relation_id)
    if relation_id is not None and relation is None:
        return {"status": "not_found", "reason": "invalid_viewer_target", "corpus_id": corpus_id}
    if relation is not None:
        evidence = evidence | {
            "bbox_refs": tuple(relation.get("bbox_refs") or ()),
            "quoted_text": relation.get("quoted_text") or evidence.get("quoted_text"),
        }
    bboxes = _request_bboxes(store, evidence_id, evidence, relation, synthetic_bboxes, bbox_refs)
    bboxes = _select_viewer_bboxes(bboxes, bbox_refs)
    if bbox_refs and not bboxes:
        return {"status": "not_found", "reason": "invalid_viewer_target", "corpus_id": corpus_id}
    return resolve_pdf_access(
        store,
        corpus_id,
        evidence,
        bboxes,
        source_document_id=source_document_id,
        page_number=page_number,
        bbox_id=bbox_id,
        source_sha256=source_sha256,
        source_pdf_path=source_pdf_path,
    )


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
    requires_exact_geometry = (
        bbox_id is not None
        or evidence.get("bbox_precision") == "exact"
        and evidence.get("viewer_highlightable") is True
        and bool(evidence.get("bbox_refs"))
    )
    if requires_exact_geometry and not page_bboxes:
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

    first_box = page_bboxes[0] if page_bboxes else {}
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


def resolve_document_pdf_access(
    store,
    corpus_id: str,
    source: dict,
    *,
    page_number: int | None = None,
    source_pdf_path: str | None = None,
) -> dict:
    if source_pdf_path is not None:
        return _pdf_unavailable("invalid_source")
    page = page_number or 1
    if page < 1 or page > int(source.get("page_count") or 0):
        return _pdf_unavailable("invalid_page")
    try:
        pdf_path = store.config.source_path(source["path"])
    except (KeyError, ValueError):
        return _pdf_unavailable("invalid_source")
    if not pdf_path.exists() or pdf_path.suffix.casefold() != ".pdf":
        return _pdf_unavailable("invalid_source")
    try:
        if file_sha256(pdf_path) != source.get("sha256"):
            return _pdf_unavailable("source_hash_mismatch")
    except (OSError, ValueError):
        return _pdf_unavailable("render_failed")
    return {
        "status": "pdf_access_ready",
        "path": pdf_path,
        "mime_type": "application/pdf",
        "page_number": page,
        "source_sha256": source.get("sha256"),
        "access_url": _document_pdf_access_url(corpus_id, source, page),
    }


def _base_payload(store, corpus_id: str, evidence: dict, bboxes: list[dict]) -> dict:
    source = _source_document(store, evidence) or {}
    return {
        "status": "viewer_payload_ready",
        "corpus_id": corpus_id,
        "evidence_id": evidence["evidence_id"],
        "legal_unit_id": evidence.get("legal_unit_id"),
        "source_document_id": evidence.get("source_document_id"),
        "source_url": evidence.get("source_url")
        or source.get("source_page_url")
        or source.get("final_download_url")
        or source.get("download_url"),
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


def _highlightable_bboxes(evidence: dict, bboxes: list[dict]) -> tuple[dict, ...]:
    if evidence.get("bbox_precision") != "exact" or evidence.get("viewer_highlightable") is not True:
        return ()
    bbox_refs = set(evidence.get("bbox_refs") or ())
    return tuple(
        row
        for row in bboxes
        if row.get("bbox_id") in bbox_refs and row.get("bbox_precision") == "exact" and row.get("viewer_highlightable") is True
    )


def _source_document(store, evidence: dict) -> dict | None:
    for row in store.source_documents:
        if row.get("source_document_id") == evidence.get("source_document_id"):
            return row
    return None


def _document_title(store, source: dict) -> str:
    catalog: dict = getattr(getattr(store, "config", None), "setting", lambda *args: {})("document_catalog", {}) or {}
    titles = dict(catalog.get("titles") or {})
    return titles.get(source.get("source_role")) or source.get("filename") or source.get("source_document_id") or "Document"


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
        "rendering_available": True,
        "render_status": "pdf_access_available",
        "viewer_highlightable": False,
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


def _document_pdf_access_url(corpus_id: str, source: dict, page_number: int) -> str:
    query = {
        "source_document_id": source["source_document_id"],
        "page_number": str(page_number),
    }
    return f"/legal/{quote(corpus_id, safe='')}/pdf?{urlencode(query)}"


def _source_status_label(evidence: dict, store=None) -> str:
    role = str(evidence.get("source_role") or evidence.get("temporal_context") or "")
    labels = dict(getattr(getattr(store, "config", None), "setting", lambda *args: {})("viewer_source_status_labels", {}) or {})
    if role in labels:
        return labels[role]
    wildcard = next((label for prefix, label in labels.items() if prefix.endswith("*") and role.startswith(prefix[:-1])), None)
    return wildcard or labels.get("default", "Source status unavailable")


def _public_evidence_row(store, row: dict) -> dict:
    """Apply corpus-owned display labels without changing source identifiers."""
    labels = store.config.setting("public_evidence_labels", {}) or {}
    label = row.get("display_label") or row.get("label") or row.get("citation")
    public_label = labels.get(label) if isinstance(labels, dict) else None
    return row | {"display_label": public_label} if public_label else row


def _scope_has_verified_support(store, routed: dict) -> bool:
    return any(validate_answer_candidate(store, row)[0] for row in routed.get("matches", ()))


def _authority_policy(store, row: dict, *, can_resolve: bool | None = None, conflict: dict | None = None) -> dict:
    owner = store.get(row.get("evidence_id")) if store is not None and row.get("evidence_id") else None
    source_row = {**(owner or {}), **row}
    authority_kind = _authority_kind(store, row, can_resolve=can_resolve, conflict=conflict)
    conflict_row = conflict or _source_conflict_by_evidence(store, row.get("evidence_id"))
    non_final_conflict = conflict_row is not None or row.get("source_conflict_id")
    citation_final = row.get("citation_final") if isinstance(row.get("citation_final"), bool) else authority_kind == "legal_citation"
    if non_final_conflict and authority_kind in {"source_anomaly", "source_conflict_provenance"}:
        citation_final = False
    layout_lines = _layout_lines(store, source_row)
    copy_text, layout_lines = _canonical_text_projection(row.get("copy_text") or row.get("quoted_text") or "", layout_lines)
    payload = {
        "authority_kind": authority_kind,
        "authority_label": {
            "legal_citation": "Sitasi hukum",
            "metadata_source": "Metadata sumber",
            "metadata_trace": "Metadata trace",
            "source_conflict_provenance": "Jejak audit sumber",
            "source_anomaly": "Source anomaly",
            "structural_context": "Provenance struktural",
            "instrument_provenance": "Instrument provenance",
            "source_text": "Sumber teks PDF",
        }[authority_kind],
        "citation_final": citation_final,
        "source_url": row.get("source_url") or _source_url(store, row),
        "support_kind": "legal_unit"
        if source_row.get("evidence_owner_kind") == "legal_unit_source" and authority_kind == "legal_citation"
        else row.get("support_kind") or _support_kind_for_authority(authority_kind),
        "relevant_quote_eligible": source_row.get("relevant_quote_eligible") is True and authority_kind == "legal_citation",
        "display_text": row.get("display_text") or row.get("quoted_text") or "",
        "source_label": row.get("document_title") or _source_label(store, row),
        "copy_text": copy_text,
        "layout_lines": layout_lines,
        "viewer_target": _viewer_target(row),
    }
    if conflict_row is not None or row.get("source_conflict_id"):
        payload |= _source_conflict_taxonomy_fields(conflict_row or row)
    return payload


def _canonical_text_projection(value: str, layout_lines: tuple[dict, ...] = ()) -> tuple[str, tuple[dict, ...]]:
    """Build semantic copy text and visual-line ranges in one ordered traversal."""
    if layout_lines:
        pieces: list[str] = []
        projected: list[dict] = []
        current_id = object()
        for line in layout_lines:
            paragraph_id = line.get("paragraph_id")
            text = " ".join(str(line.get("text") or "").split())
            if not text:
                continue
            if pieces:
                pieces.append("\n\n" if paragraph_id != current_id else " ")
            current_id = paragraph_id
            start = sum(len(piece) for piece in pieces)
            pieces.append(text)
            projected.append(line | {"text": text, "canonical_start": start, "canonical_end": start + len(text)})
        if projected:
            return "".join(pieces), tuple(projected)
    text = "\n".join(line.lstrip(" \t") for line in str(value).replace("\r\n", "\n").replace("\r", "\n").split("\n")).strip()
    return text, tuple(layout_lines)


def _copy_text(value: str, layout_lines: tuple[dict, ...] = ()) -> str:
    return _canonical_text_projection(value, layout_lines)[0]


def _viewer_target(row: dict) -> dict:
    target = dict(row.get("viewer_target") or row.get("viewer_ref") or {})
    target.setdefault("source_document_id", row.get("source_document_id"))
    target.setdefault("evidence_id", row.get("evidence_id"))
    target.setdefault("page_numbers", tuple(row.get("page_numbers") or ()))
    return target


def _select_viewer_bboxes(bboxes: list[dict], bbox_refs: tuple[str, ...]) -> list[dict]:
    """Honor the verified support subset stored behind an opaque target."""
    if not bbox_refs:
        return bboxes
    expected = set(bbox_refs)
    selected = [bbox for bbox in bboxes if bbox.get("bbox_id") in expected]
    return selected if {bbox.get("bbox_id") for bbox in selected} == expected else []


def _request_bboxes(
    store,
    evidence_id: str,
    evidence: dict,
    relation: dict | None,
    synthetic_bboxes: list[dict] | None,
    bbox_refs: tuple[str, ...],
) -> list[dict]:
    if synthetic_bboxes is not None:
        bboxes = synthetic_bboxes
    elif evidence.get("metadata_grounding"):
        bboxes = store.metadata_bboxes_for(evidence_id)
    elif relation is not None:
        bboxes = store.bboxes_for_refs(tuple(relation.get("bbox_refs") or ()))
    else:
        bboxes = store.bboxes_for(evidence_id)
    return store.bboxes_for_refs(bbox_refs) if bbox_refs else bboxes


def _metadata_grounding_evidence(store, metadata_grounding_id: str | None) -> dict | None:
    row = next(
        (row for row in store.metadata_grounding if row.get("metadata_grounding_id") == metadata_grounding_id),
        None,
    )
    if row is None:
        return None
    return {
        "evidence_id": row["metadata_grounding_id"],
        "metadata_grounding": True,
        "legal_unit_id": None,
        "citation": f"Metadata {row.get('source_role')}: {row.get('metadata_field') or 'block'}",
        "hierarchy": (),
        "quoted_text": row.get("quoted_text"),
        "bbox_refs": tuple(row.get("bbox_refs") or ()),
        "bbox_precision": row.get("bbox_precision"),
        "viewer_highlightable": row.get("viewer_highlightable"),
        "page_numbers": tuple(row.get("page_numbers") or ()),
        "source_document_id": row.get("source_document_id"),
        "source_pdf_path": row.get("source_pdf_path"),
        "source_sha256": row.get("source_sha256"),
        "source_role": row.get("source_role"),
        "temporal_context": row.get("temporal_context"),
    }


def _relation_for_evidence(store, evidence_id: str | None, relation_id: str | None = None) -> dict | None:
    if not evidence_id:
        return None
    return next(
        (
            projection
            for edge in store.graph_edges
            for projection in (edge.get("relation_projection") or {},)
            if projection.get("target_legal_unit_id")
            and projection.get("evidence_id") == evidence_id
            and (relation_id is None or projection.get("relation_id") == relation_id)
        ),
        None,
    )


def _layout_lines(store, row: dict) -> tuple[dict, ...]:
    """Expose source-derived line layout without making the UI infer it."""
    configured = row.get("layout_lines")
    if isinstance(configured, (list, tuple)) and configured and isinstance(configured[0], dict):
        return tuple(configured)
    spans = {item.get("text_span_id"): item for item in (getattr(store, "page_text_spans", ()) if store is not None else ())}
    fragments = []
    for order, span_id in enumerate(row.get("text_span_ids") or ()):
        span = spans.get(span_id)
        if not span:
            continue
        boxes = store.exact_bboxes_for_text_spans((span_id,)) if store is not None else ()
        geometry = boxes or (span,)
        width = next((float(box["page_width"]) for box in geometry if isinstance(box.get("page_width"), (int, float))), 0.0)
        x0 = min((float(box["x0"]) for box in geometry if isinstance(box.get("x0"), (int, float))), default=0.0)
        x1 = max((float(box["x1"]) for box in geometry if isinstance(box.get("x1"), (int, float))), default=0.0)
        y0 = min((float(box["y0"]) for box in geometry if isinstance(box.get("y0"), (int, float))), default=0.0)
        y1 = max((float(box["y1"]) for box in geometry if isinstance(box.get("y1"), (int, float))), default=0.0)
        fragments.append(
            {
                "text": span.get("exact_quote") or span.get("text") or "",
                "order": order,
                "page": span.get("page_number"),
                "width": width,
                "x0": x0,
                "x1": x1,
                "y0": y0,
                "y1": y1,
                "refs": [box["bbox_id"] for box in boxes if box.get("bbox_id")],
            }
        )
    if fragments:
        # Page-text spans may be fragments of one visual source line.  Merge
        # only exact same-baseline fragments in extraction order.
        visual: list[dict[str, Any]] = []
        for fragment in fragments:
            previous = visual[-1] if visual else None
            previous_part = previous["parts"][-1] if previous is not None else None
            same_line = (
                previous is not None
                and previous_part is not None
                and fragment["page"] == previous["page"]
                and abs(fragment["y0"] - previous_part["y0"]) <= 1.0
                and abs(fragment["y1"] - previous_part["y1"]) <= 1.0
            )
            if same_line and previous is not None:
                previous["parts"].append(fragment)
            else:
                visual.append({"page": fragment["page"], "parts": [fragment]})
        page_left = {
            page: min(part["x0"] for line in visual if line["page"] == page for part in line["parts"])
            for page in {line["page"] for line in visual}
        }
        result = []
        paragraph = 0
        previous = None
        for line_order, line in enumerate(visual):
            parts = line["parts"]
            x0, x1 = min(part["x0"] for part in parts), max(part["x1"] for part in parts)
            y0, y1 = min(part["y0"] for part in parts), max(part["y1"] for part in parts)
            width = next((part["width"] for part in parts if part["width"]), 0.0)
            text = " ".join(str(part["text"]).strip() for part in parts if str(part["text"]).strip())
            centered = width and (x1 - x0) <= width * 0.55 and abs(((x0 + x1) / 2) - width / 2) <= max(8.0, width * 0.04)
            alignment = "center" if centered else "left"
            # A numbered line is a semantic boundary even when its baseline
            # follows the prior line closely (for example consecutive ayat).
            numbered = bool(re.match(r"^\s*(?:\(\d+\)|[A-Za-z]\.|\d+[.)])\s+", text))
            if previous and (
                line["page"] != previous["page"]
                or y0 - previous["y1"] > max(20.0, 1.5 * (y1 - y0))
                or alignment == "center"
                or previous["alignment"] == "center"
                or numbered
            ):
                paragraph += 1
            result.append(
                {
                    "text": text,
                    "line_order": line_order,
                    "paragraph_id": str(paragraph),
                    "alignment": alignment,
                    "indent": max(0.0, x0 - page_left[line["page"]]) if alignment == "left" else 0.0,
                    "source_bbox_refs": [ref for part in parts for ref in part["refs"]],
                }
            )
            previous = {"page": line["page"], "y1": y1, "alignment": alignment}
        return tuple(result)
    text = str(row.get("display_text") or row.get("quoted_text") or "")
    return (
        {
            "text": text,
            "line_order": 0,
            "paragraph_id": str(row.get("evidence_id") or "support"),
            "alignment": "unknown",
            "indent": 0.0,
            "source_bbox_refs": [],
        },
    )


def _support_kind_for_authority(authority_kind: str) -> str:
    return {
        "legal_citation": "legal_citation",
        "metadata_source": "metadata_source",
        "metadata_trace": "metadata_trace",
        "source_conflict_provenance": "source_anomaly_provenance",
        "source_anomaly": "source_anomaly_provenance",
        "structural_context": "structural_provenance",
        "instrument_provenance": "instrument_provenance",
        "source_text": "source_text",
    }[authority_kind]


def _source_url(store, row: dict) -> str | None:
    source_id = row.get("source_document_id")
    return next(
        (
            source.get("source_page_url") or source.get("final_download_url") or source.get("download_url")
            for source in getattr(store, "source_documents", ())
            if source.get("source_document_id") == source_id
        ),
        None,
    )


def _source_label(store, row: dict) -> str | None:
    source_id = row.get("source_document_id")
    source: dict[str, Any] = next(
        (item for item in getattr(store, "source_documents", ()) if item.get("source_document_id") == source_id), {}
    )
    catalog: dict[str, Any] = getattr(getattr(store, "config", None), "setting", lambda *args: {})("document_catalog", {}) or {}
    return (catalog.get("titles") or {}).get(source.get("source_role")) or source.get("filename")


def _source_conflict_taxonomy_fields(conflict: dict | None) -> dict:
    if not conflict:
        return {}
    policy = conflict.get("source_anomaly_policy") or {}
    fields = {
        "source_anomaly_kind": conflict.get("source_anomaly_kind") or policy.get("anomaly_kind"),
        "source_mapping_kind": conflict.get("source_mapping_kind") or policy.get("mapping_kind"),
        "provenance_highlight_scope": conflict.get("provenance_highlight_scope") or policy.get("provenance_highlight_scope"),
        "finality_policy": policy.get("finality_policy"),
        "support_type": conflict.get("type"),
    }
    fields = {key: value for key, value in fields.items() if value is not None}
    if fields.get("finality_policy"):
        fields["support_kind"] = fields["finality_policy"]
    fields.update(_source_mapping_semantics(conflict))
    return fields


def _authority_kind(store, row: dict, *, can_resolve: bool | None = None, conflict: dict | None = None) -> str:
    viewer_resolvable = (
        can_resolve
        if can_resolve is not None
        else row.get("viewer_ref", {}).get("can_resolve") is True or row.get("viewer_highlightable") is True
    )
    if row.get("metadata_grounding") or row.get("metadata_field"):
        return "metadata_source" if viewer_resolvable else "metadata_trace"
    if row.get("evidence_owner_kind") == "source_span" or row.get("authority_kind") == "source_text":
        return "source_text"
    if row.get("authority_kind") == "source_anomaly_trace" and row.get("citation_final") is False:
        return "source_anomaly"
    if row.get("presentation_as_legal_quote") is True:
        return "legal_citation"
    if row.get("authority_kind") == "structural_context":
        return "structural_context"
    conflict_row = conflict or _source_conflict_by_evidence(store, row.get("evidence_id"))
    if conflict_row is not None or row.get("source_conflict_id"):
        return "source_anomaly" if _is_source_anomaly_conflict(conflict_row or row) else "source_conflict_provenance"
    if _row_is_historical_anomaly(store, row):
        return "source_anomaly"
    if _row_is_instrument_provenance(store, row):
        return "instrument_provenance"
    return "legal_citation"


def _source_conflict_by_evidence(store, evidence_id: object) -> dict | None:
    if store is None or not evidence_id:
        return None
    return next(
        (
            row
            for row in store.source_conflicts
            if evidence_id == row.get("source_conflict_id") or evidence_id in set(row.get("evidence_ids") or ())
        ),
        None,
    )


def _is_source_anomaly_conflict(row: dict) -> bool:
    if row.get("source_anomaly_kind") == "renumbering_provenance":
        return False
    if row.get("source_anomaly_kind") == "source_marker_sequence_anomaly":
        return True
    classification = str(row.get("classification") or "").casefold()
    return (
        row.get("provenance_exception_category") == "accepted_noncanonical_source_conflict_trace_only"
        or "anomaly" in classification
        or "typo" in classification
    )


def _row_is_instrument_provenance(store, row: dict) -> bool:
    candidate_type = str(row.get("candidate_type") or "")
    if candidate_type == "article_amendment_relation" or candidate_type.startswith("instrument_"):
        return True
    if store is None:
        return False
    try:
        units = store.legal_units
    except (KeyError, OSError, ValueError):
        return False
    unit: dict = next((item for item in units if item.get("legal_unit_id") == row.get("legal_unit_id")), {})
    return bool(unit) and _is_instrument_unit(store, unit)


def _row_is_historical_anomaly(store, row: dict) -> bool:
    if store is None:
        return False
    try:
        units = store.legal_units
    except (KeyError, OSError, ValueError):
        return False
    unit: dict = next((item for item in units if item.get("legal_unit_id") == row.get("legal_unit_id")), {})
    return unit.get("status") == "active_historical_record" and bool(unit.get("exclusion_ref"))


def _citation_with_authority(store, row: dict, *, conflict: dict | None = None) -> dict:
    return _public_evidence_row(store, row | _authority_policy(store, row, conflict=conflict))


def _claim_citations(citations: tuple[dict, ...], claim_support) -> tuple[dict, ...]:
    segments = tuple(segment for claim in claim_support for segment in claim.support_segments if segment.get("evidence_id"))
    return tuple(
        _claim_citation(
            row,
            next(
                (
                    segment
                    for segment in segments
                    if segment.get("source_document_id") == row.get("source_document_id")
                    and (segment.get("evidence_id") == row.get("evidence_id") or segment.get("legal_unit_id") == row.get("legal_unit_id"))
                ),
                None,
            ),
        )
        for row in citations
    )


def _claim_citation(citation: dict, segment: dict | None) -> dict:
    if segment is None:
        return citation
    bbox_refs = tuple(segment.get("bbox_refs") or ())
    exact_quote = segment.get("exact_quote")
    return citation | {
        "proposition_id": segment.get("proposition_id"),
        "quoted_text": exact_quote,
        "display_text": exact_quote,
        "copy_text": exact_quote,
        "layout_lines": (),
        "text_span_ids": tuple(segment.get("text_span_ids") or ()),
        "bbox_refs": bbox_refs,
        "page_numbers": tuple(segment.get("page_numbers") or ()),
        "bbox_count": len(bbox_refs),
        "viewer_overlay": segment.get("viewer_overlay"),
        "viewer_ref": {
            **dict(citation.get("viewer_ref") or {}),
            "bbox_count": len(bbox_refs),
            "can_resolve": bool(bbox_refs),
        },
    }


def _source_span_evidence(store, support_id: str) -> dict | None:
    span = store.source_span_for_support(support_id)
    if not span or not span.get("semantic_text") or span.get("citation_eligible") is not True:
        return None
    if not store.source_span_bboxes(support_id):
        return None
    return {
        "corpus_id": getattr(store.config, "corpus_id", None),
        "evidence_id": support_id,
        "source_support_id": support_id,
        "source_document_id": span.get("source_document_id"),
        "source_pdf_path": span.get("source_pdf_path"),
        "source_sha256": span.get("source_sha256"),
        "source_role": span.get("source_role"),
        "temporal_context": span.get("source_role"),
        "citation": span.get("semantic_exact_quote"),
        "quoted_text": span.get("semantic_exact_quote"),
        "page_numbers": [span.get("page_number")],
        "bbox_refs": [support_id],
        "bbox_precision": "exact",
        "viewer_highlightable": True,
        "status": "final",
        "citation_final": False,
        "citation_eligible": True,
        "relevant_quote_eligible": False,
        "authority_kind": "source_text",
        "support_kind": "source_text",
        "evidence_owner_kind": "source_span",
        "text_span_ids": (),
    }


def _viewer_with_authority(store, evidence: dict, payload: dict) -> dict:
    return payload | _authority_policy(store, evidence, can_resolve=payload.get("viewer_highlightable") is True)


def _is_instrument_unit(store, unit: dict) -> bool:
    schema: dict = getattr(getattr(store, "config", None), "setting", lambda *args: {})("schema", {}) or {}
    return unit.get("unit_type") in set(schema.get("instrument_unit_types") or ())


def _source_mapping_semantics(conflict: dict) -> dict:
    if conflict.get("source_anomaly_kind") != "renumbering_provenance":
        return {}
    return {
        "relation_type": "renumbered_to",
        "substantive_change": False,
        "anomaly": False,
        "source_conflict": False,
    }
