from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re

from tjipto.catalog import CATALOG_FILTERS
from tjipto.evidence.legal_citation import FootnoteBook, IndonesianLegalCitationProfile
from tjipto.runtime.public_document import project_legal_document
from tjipto.runtime.service import LegalRuntimeService
from tjipto.retrieval.research import research_planning_provider_from_environment


class BadRequest(ValueError):
    def __init__(self, reason: str = "invalid_request"):
        self.reason = reason
        super().__init__(reason)


_PUBLIC_REQUEST_FIELDS = {
    "ask": {"query", "limit", "filters", "clarification_id", "clarification_answer"},
    "search": {"query", "limit", "filters"},
    "citation": {"query", "source_role", "filters"},
    "viewer": {"target"},
    "bookmark": {"target", "note"},
    "delete_bookmark": {"bookmark"},
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
        result = service.catalog_search(
            _query(payload),
            _limit(payload, default=10),
            _filters(payload),
            corpus_id=corpus_id,
        )
        if result.get("readiness") is False:
            return _public_integrity(result)
        return _public_catalog_result(result, service, kind="document")
    if action == "citation":
        result = service.citation(corpus_id, _query(payload), _optional_str(payload, "source_role"), _filters(payload))
        return _public_citation_response(result, service, corpus_id)
    if action == "viewer":
        target = _required_str(payload, "target")
        return _public_viewer(service.viewer_public(corpus_id, target), service, corpus_id, target)
    if action == "ask":
        filters = dict(_filters(payload) or {})
        result = service.ask(
            corpus_id,
            _query(payload),
            _limit(payload, default=3),
            filters or None,
            clarification_id=_optional_str(payload, "clarification_id"),
            clarification_answer=_optional_str(payload, "clarification_answer"),
        )
        return _public_ask(result, service, corpus_id)
    if action == "capabilities":
        return _public_capabilities(service.capabilities(corpus_id))
    if action == "bookmarks":
        return _public_bookmarks(service.bookmarks(corpus_id), service, corpus_id)
    if action == "bookmark":
        return _public_bookmark(
            service.bookmark_public(corpus_id, _required_str(payload, "target"), _optional_str(payload, "note")),
            service,
            corpus_id,
        )
    if action == "delete_bookmark":
        return service.delete_bookmark_public(corpus_id, _required_str(payload, "bookmark"))
    return {"status": "unsupported_action"}


def _public_catalog_result(result: dict, service: LegalRuntimeService, *, kind: str) -> dict:
    documents = service.catalog_documents()
    return {
        "kind": kind,
        "status": result["status"],
        "total": result["total"],
        "applied_filters": result.get("applied_filters", {}),
        "facets": result["facets"],
        "results": tuple(
            project_legal_document(
                document,
                documents,
                viewer_target=_public_target(document.public_target_id, "open_document", (1,), True),
            )
            for document in result["results"]
        ),
    }


def _public_ask(result: dict, service: LegalRuntimeService, corpus_id: str) -> dict:
    if result.get("readiness") is False:
        return _public_integrity(result)
    support_rows = _unique_support_rows(
        (
            *result.get("final_citations", result.get("citations", ())),
            *(() if result.get("answer_type") == "article_amendment_relation" else result.get("historical_citations", ())),
            *result.get("metadata_support", ()),
            *result.get("structural_support", ()),
            *result.get("relation_support", ()),
            *result.get("trace_support", ()),
            *result.get("summary_support", ()),
            *result.get("comparison_support", ()),
        )
    )
    footnotes = FootnoteBook()
    projected = tuple((_public_support(row, service, corpus_id, footnotes), row) for row in support_rows)
    supports = tuple(support for support, _ in projected)
    if result.get("document_source") is not None:
        source = result["document_source"]
        target = service.register_public_target(corpus_id, {"evidence_id": None, "source_document_id": source.get("source_document_id")})
        viewer_target = _public_target(target, "open_document", (1,), True)
        document = service.catalog_document_for_source(source.get("source_role"))
        return {
            "kind": "document",
            "status": result.get("status"),
            "document": (
                project_legal_document(document, service.catalog_documents(), viewer_target=viewer_target)
                if document is not None
                else {"title": source.get("document_title"), "viewer_target": viewer_target}
            ),
        }
    if result.get("document_sources") is not None:
        documents = []
        for source in result["document_sources"]:
            target = service.register_public_target(corpus_id, {"evidence_id": None, "source_document_id": source.get("source_document_id")})
            viewer_target = _public_target(target, "open_document", (1,), True)
            document = service.catalog_document_for_source(source.get("source_role"))
            documents.append(
                project_legal_document(document, service.catalog_documents(), viewer_target=viewer_target)
                if document is not None
                else {"title": source.get("document_title"), "viewer_target": viewer_target}
            )
        return {"kind": "documents", "status": result.get("status"), "documents": tuple(documents)}
    public = {
        "kind": "answer",
        "status": result.get("status"),
        "answer": _answer_with_footnotes(result.get("answer"), projected),
        "supports": supports,
        "support_groups": _support_groups(projected, service, corpus_id),
    }
    if result.get("clarification_id"):
        public["clarification"] = {
            "id": result["clarification_id"],
            "missing_dimensions": tuple(result.get("missing_dimensions") or ()),
        }
    if result.get("operation"):
        public["operation"] = result["operation"]
    if result.get("source_scopes"):
        public["source_scopes"] = tuple(
            {
                "label": (
                    project_legal_document(document, service.catalog_documents()).get("title")
                    if (document := service.catalog_document_for_source(role)) is not None
                    else service.public_source_status_label(corpus_id, role) or str(role).replace("_", " ")
                ),
            }
            for role in result["source_scopes"]
        )
    sufficiency = result.get("sufficiency")
    if isinstance(sufficiency, dict) and sufficiency.get("status"):
        public["sufficiency"] = {
            "status": sufficiency["status"],
            "missing_requirement_ids": tuple(sufficiency.get("missing_requirement_ids") or ()),
        }
    return public


def _unique_support_rows(rows: tuple[dict, ...]) -> tuple[dict, ...]:
    output: list[dict] = []
    seen: set[str] = set()
    for row in rows or ():
        if not isinstance(row, dict):
            continue
        # Relation claims sharing one instrument clause still represent
        # distinct targets and viewer refs.  Keep relation identity primary;
        # ordinary evidence remains deduplicated by its stable evidence id.
        identity = (
            str(row.get("relation_id") or row.get("evidence_id") or "")
            if row.get("support_kind") == "article_relation"
            else str(row.get("evidence_id") or row.get("relation_id") or "")
        )
        if identity and identity in seen:
            continue
        if identity:
            seen.add(identity)
        output.append(row)
    return tuple(output)


def _answer_with_footnotes(answer: object, projected: tuple[tuple[dict, dict], ...]) -> object:
    if not isinstance(answer, str) or not answer.strip():
        return answer
    citations = tuple(
        (str(source.get("evidence_id") or ""), citation["number"])
        for support, source in projected
        if isinstance((citation := support.get("citation")), dict)
        and isinstance(citation.get("number"), int)
    )
    numbers = tuple(dict.fromkeys(number for _, number in citations))
    if not numbers:
        return re.sub(r"\s*\[\[support:[^\]]+\]\]", "", answer).strip()
    had_markers = "[[support:" in answer
    rendered = answer
    for evidence_id, number in citations:
        if evidence_id:
            rendered = rendered.replace(f"[[support:{evidence_id}]]", f"[{number}]")
    rendered = re.sub(r"\s*\[\[support:[^\]]+\]\]", "", rendered).strip()
    if not had_markers:
        paragraphs = answer.strip().split("\n\n")
        if len(paragraphs) == len(numbers):
            return "\n\n".join(f"{paragraph.rstrip()} [{number}]" for paragraph, number in zip(paragraphs, numbers, strict=True))
        existing = {int(value) for value in re.findall(r"\[(\d+)\]", rendered)}
        missing = tuple(number for number in numbers if number not in existing)
        if missing:
            rendered = f"{rendered} {' '.join(f'[{number}]' for number in missing)}"
    return rendered


def _public_citation_response(result: dict, service: LegalRuntimeService, corpus_id: str) -> dict:
    if result.get("readiness") is False:
        return _public_integrity(result)
    footnotes = FootnoteBook()
    projected = tuple((_public_support(row, service, corpus_id, footnotes), row) for row in result.get("citation_payloads", ()))
    supports = tuple(support for support, _ in projected)
    return {
        "kind": "answer",
        "status": result.get("status"),
        "supports": supports,
        "support_groups": _support_groups(projected, service, corpus_id),
    }


def _public_viewer(
    result: dict,
    service: LegalRuntimeService,
    corpus_id: str,
    target: str,
) -> dict:
    if result.get("readiness") is False:
        return _public_integrity(result)
    public = {
        "kind": "document",
        "status": result.get("status"),
        "citation": result.get("citation"),
        "quoted_text": result.get("quoted_text"),
        "source_status_label": result.get("source_status_label"),
        "page_numbers": tuple(result.get("page_numbers") or ()),
        "bbox_rectangles": tuple(_public_bbox(row, index) for index, row in enumerate(result.get("bbox_rectangles", ()), start=1)),
        "viewer_highlightable": result.get("viewer_highlightable") is True,
        "pdf_access_available": result.get("pdf_access_available") is True,
        "rendering_available": result.get("rendering_available") is True,
    }
    if result.get("pdf_access_available") and result.get("public_pdf_target"):
        public["pdf"] = {
            "mime_type": "application/pdf",
            "access_url": f"/legal/{corpus_id}/pdf?target={result['public_pdf_target']}",
        }
    document = service.catalog_document_for_target(corpus_id, target)
    if document is not None:
        public["source_status_label"] = document.document_role_label
        public["document"] = project_legal_document(
            document,
            service.catalog_documents(),
            viewer_target=_public_target(target, "viewer", result.get("page_numbers") or (), True),
        )
    return public


def _public_integrity(result: dict) -> dict:
    return {
        "kind": "unavailable",
        "status": "unavailable",
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
    }


def _public_bbox_precision(value: object) -> str:
    return value if value in {"exact", "coarse", "page_grounded_only"} else "page_grounded_only"


def _public_support(row: dict, service: LegalRuntimeService, corpus_id: str, footnotes: FootnoteBook | None = None) -> dict:
    authority = row.get("authority_kind")
    authority_kind = authority if isinstance(authority, str) and authority in {
        "legal_citation", "metadata_source", "metadata_trace", "source_conflict_provenance",
        "source_anomaly", "structural_context", "instrument_provenance", "source_text",
        "source_annotation",
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
        "source_annotation": "source_annotation",
    }.get(authority_kind, "source_trace")
    role_label = row.get("printed_role") if fact_kind == "person_role" else None
    target_source = dict(row.get("viewer_target") or row.get("viewer_ref") or {})
    linkable = target_source.get("can_resolve") is True and bool(
        row.get("source_document_id") and row.get("page_numbers")
    )
    target_action = "open_document" if target_source.get("action") == "open_document" else "viewer"
    target = service.register_public_target(corpus_id, {
        "evidence_id": None if target_action == "open_document" else row.get("evidence_id") or row.get("source_conflict_id") or row.get("relation_id"),
        "proposition_id": row.get("proposition_id"),
        "relation_id": row.get("relation_id") or target_source.get("relation_id"),
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
    unit = service.citation_unit(corpus_id, row)
    citation = None
    if unit is not None:
        number, text = (footnotes or FootnoteBook()).cite(unit)
        citation = {
            "number": number,
            "text": text,
            "official_url": unit.official_url,
            "citation_final": unit.citation_final,
        }
    legal_document = service.catalog_document_for_source(row.get("source_role"))
    relation_claim = row.get("fact_kind") == "article_relation" or row.get("support_kind") == "article_relation"
    relation_label = row.get("target_citation") or row.get("target_label") if relation_claim else None
    result = {
        "public_support_id": service.public_identifier(corpus_id, "support", target or (row.get("display_label"), row.get("source_role"), authority_kind)),
        "authority_kind": authority_kind,
        # Relation finality is evaluated at the claim level.  The supporting
        # clause remains instrument provenance, but an exact, isolated
        # article relation can be published as a final relation claim.
        "citation_final": row.get("citation_final") is True and (authority_kind == "legal_citation" or relation_claim),
        "support_kind": row.get("support_kind") or "trace_support",
        "fact_kind": fact_kind,
        "label": relation_label or row.get("printed_name") or role_label or row.get("display_label") or row.get("label") or row.get("citation") or row.get("authority_label") or "Bukti sumber",
        "role_label": role_label,
        "text": row.get("display_text") or row.get("quoted_text") or row.get("answer") or "",
        "source_label": row.get("document_title") or row.get("source_label"),
        "source_status_label": (
            legal_document.document_role_label
            if legal_document is not None
            else row.get("source_status_label") or service.public_source_status_label(corpus_id, row.get("source_role"))
        ),
        "page_numbers": tuple(row.get("page_numbers") or ()),
        "viewer_target": _public_target(target, target_action, row.get("page_numbers") or (), linkable),
        "citation": citation,
    }
    if legal_document is not None:
        result["document"] = project_legal_document(
            legal_document,
            service.catalog_documents(),
            viewer_target=result["viewer_target"],
        )
    return result


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
        role_key = (str(support.get("source_label") or ""), str(row.get("source_role") or ""), str(support.get("role_label") or ""))
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
    source_role = str(row.get("source_role") or "")
    fact_kind = str(support.get("fact_kind") or "")
    if authority_kind == "metadata_source" and fact_kind == "person_role":
        role_key = (source, source_role, str(support.get("role_label") or ""))
        if role_counts.get(role_key, 0) > 1:
            return role_key, "role_members"
        name = row.get("entity_identity")
        if name and entity_counts.get(name, 0) > 1:
            return (name, str(support.get("role_label") or "")), "entity_occurrences"
    if support.get("support_kind") == "article_relation":
        return (str(row.get("evidence_id") or ""), source_role), "article_relation_members"
    if authority_kind == "metadata_source":
        return (source, source_role), "document_metadata"
    return (), ""


def _support_group_label(group_kind: str, members: list[dict]) -> str:
    if group_kind == "role_members" and members[0].get("role_label"):
        return f"{members[0]['role_label']} · {len(members)} orang"
    if group_kind == "entity_occurrences":
        return str(members[0].get("label") or "Sumber dokumen")
    if group_kind == "article_relation_members":
        return f"{members[0].get('source_label') or 'Sumber perubahan'} · {len(members)} ketentuan"
    return str(members[0].get("source_label") or members[0].get("label") or "Sumber dokumen")


def _public_capabilities(result: dict) -> dict:
    if result.get("readiness") is False:
        return _public_integrity(result)
    return {"status": "ok", "capabilities": ("search", "ask", "citation", "viewer", "bookmarks")}


def _public_bookmarks(result: dict, service: LegalRuntimeService, corpus_id: str) -> dict:
    if result.get("readiness") is False:
        return _public_integrity(result)
    return {
        "status": result.get("status"),
        "bookmarks": tuple(_public_bookmark_row(row, service, corpus_id) for row in result.get("bookmarks", ())),
    }


def _public_bookmark(result: dict, service: LegalRuntimeService, corpus_id: str) -> dict:
    if result.get("status") != "saved":
        return {"status": "unavailable"}
    row = result["bookmark"]
    return {
        "status": "saved",
        "bookmark": _public_bookmark_row(row, service, corpus_id),
    }


def _public_bookmark_row(row: dict, service: LegalRuntimeService, corpus_id: str) -> dict:
    public_target = row.get("public_target")
    result = {
        "public_bookmark_id": service.public_identifier(corpus_id, "bookmark", row.get("bookmark_id")),
        "public_target_id": public_target,
        "note": row.get("note"),
        "created_at": row.get("created_at"),
        "status": row.get("status"),
    }
    document = service.catalog_document_for_target(corpus_id, public_target)
    if document is not None:
        result["document"] = project_legal_document(
            document,
            service.catalog_documents(),
            viewer_target=_public_target(public_target, "viewer", (), True),
        )
    return result


def handle_catalog_request(
    action: str,
    payload: dict,
    repo_root: Path | None = None,
    service: LegalRuntimeService | None = None,
) -> dict:
    service = service or _service_for(repo_root)
    if action == "search":
        _validate_catalog_payload(payload, {"query", "limit", "filters"})
        result = service.catalog_search(_query(payload), _limit(payload, default=10), _filters(payload))
        return _public_catalog_result(result, service, kind="catalog")
    if action == "viewer":
        _validate_catalog_payload(payload, {"target"})
        target = _required_str(payload, "target")
        document = service.catalog_viewer(target)
        if document is None:
            return {"kind": "unavailable", "status": "not_found"}
        projection = project_legal_document(
            document,
            service.catalog_documents(),
            viewer_target=_public_target(target, "open_document", (1,), True),
        )
        return {
            "kind": "document",
            "status": "found",
            "citation": IndonesianLegalCitationProfile().full(document.citation_unit),
            "document": projection,
            **{
                key: projection[key]
                for key in (
                    "title", "legal_identity", "legal_status", "legal_status_scope", "document_role", "issuer",
                    "establishment_date", "establishment_place", "signatories", "promulgation_date", "effective_date", "publication",
                    "official_url", "relations", "provision_effects",
                )
            },
            "page_numbers": (1,),
            "bbox_rectangles": (),
            "viewer_highlightable": False,
            "pdf_access_available": True,
            "rendering_available": True,
            "pdf": {
                "mime_type": "application/pdf",
                "access_url": f"/legal/catalog/pdf?target={target}",
            },
        }
    if action == "facets":
        _validate_catalog_payload(payload, set())
        corpus_ids = service.registry.corpus_ids()
        return {
            "kind": "catalog_facets",
            "facets": service.catalog_search("", 1)["facets"],
            "default_corpus": corpus_ids[0] if len(corpus_ids) == 1 else None,
        }
    return {"kind": "unavailable", "status": "unsupported_action"}


def handle_catalog_pdf_request(
    payload: dict,
    repo_root: Path | None = None,
    service: LegalRuntimeService | None = None,
) -> dict:
    _validate_catalog_payload(payload, {"target"})
    return (service or _service_for(repo_root)).catalog_pdf(_required_str(payload, "target"))


def handle_pdf_request(corpus_id: str, payload: dict, repo_root: Path | None = None, service: LegalRuntimeService | None = None) -> dict:
    _validate_payload("pdf", payload)
    return (service or _service_for(repo_root)).pdf_public(corpus_id, _required_str(payload, "target"))


@lru_cache(maxsize=1)
def _service_for(repo_root: Path | None) -> LegalRuntimeService:
    return LegalRuntimeService(repo_root, planning_provider=research_planning_provider_from_environment())


def _validate_payload(action: str, payload: object) -> None:
    allowed = _PUBLIC_REQUEST_FIELDS.get(action, {"target"} if action == "pdf" else None)
    if not isinstance(payload, dict) or allowed is None or set(payload) - allowed:
        raise BadRequest()


def _validate_catalog_payload(payload: object, allowed: set[str]) -> None:
    if not isinstance(payload, dict) or set(payload) - allowed:
        raise BadRequest()
    filters = payload.get("filters")
    if filters is not None and (not isinstance(filters, dict) or set(filters) - set(CATALOG_FILTERS)):
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
