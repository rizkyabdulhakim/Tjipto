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
        return service.search(
            corpus_id,
            str(payload.get("query", "")),
            _limit(payload, default=10),
            _filters(payload),
        )
    if action == "citation":
        return service.citation(
            corpus_id,
            str(payload.get("query", "")),
            _optional_str(payload, "source_role"),
            _filters(payload),
        )
    if action == "viewer":
        return service.viewer(
            corpus_id,
            _required_str(payload, "evidence_id"),
            source_document_id=_optional_str(payload, "source_document_id"),
            page_number=_optional_int(payload, "page_number"),
            bbox_id=_optional_str(payload, "bbox_id"),
            source_pdf_path=_optional_str(payload, "source_pdf_path"),
        )
    if action == "ask":
        return service.ask(
            corpus_id,
            str(payload.get("query", "")),
            _limit(payload, default=3),
            _filters(payload),
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
