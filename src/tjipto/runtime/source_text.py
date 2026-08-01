from __future__ import annotations

from tjipto.contracts.source_text import (
    SourceSelector,
    SourceTextCapability,
    SourceTextDisposition,
    SourceTextQueryResult,
    SourceTextRecord,
)
from tjipto.retrieval.answer import empty_context_pack


def source_text_response(store, corpus_id: str, query: str) -> dict | None:
    handler = getattr(getattr(store.config, "strategy", None), "source_text_query", None)
    result = handler(store, query) if handler is not None else None
    if not isinstance(result, SourceTextQueryResult):
        return None
    context = empty_context_pack(None)
    return {
        "status": "answer_ready",
        "route": result.route,
        "intent": result.route,
        "corpus_id": corpus_id,
        "original_query": query,
        "normalized_query": query.strip(),
        "matches": (),
        "reason": None,
        "answer_type": "source_annotation",
        "answer": result.answer,
        "context_pack": context,
        "evidence": (),
        "citations": (),
        "final_citations": (),
        "historical_citations": (),
        "metadata_support": (),
        "structural_support": (),
        "relation_support": (),
        "trace_support": result.supports,
        "viewer_refs": (),
        "metadata_facts": (),
        "legal_relations": (),
        "answer_scope": "source_provenance",
        "warnings": (),
        "insufficient_reasons": (),
    }


def source_text_record(row: dict) -> SourceTextRecord:
    semantic = str(row.get("semantic_text") or "").strip()
    source_value = str(row.get("raw_text") or "")
    normalized_value = semantic or " ".join(source_value.split())
    return SourceTextRecord(
        raw_source_span_id=str(row.get("raw_source_span_id") or ""),
        source_value=source_value,
        normalized_value=normalized_value,
        source_document_id=str(row.get("source_document_id") or ""),
        source_sha256=str(row.get("source_sha256") or ""),
        page_number=int(row.get("page_number") or 0),
        extraction_order=int(row.get("extraction_order") or 0),
        selector=SourceSelector(
            str(row.get("raw_stream_id") or ""),
            int(row.get("raw_text_start") or 0),
            int(row.get("raw_text_end") or 0),
        ),
        geometry_available=all(row.get(key) is not None for key in ("x0", "y0", "x1", "y1")),
        semantic_text_span_id=str(row["semantic_text_span_id"]) if row.get("semantic_text_span_id") else None,
        semantic_classification=str(row["semantic_classification"]) if row.get("semantic_classification") else None,
        semantic_join_status=str(row.get("semantic_join_status") or "missing"),
        source_role=str(row.get("source_role") or ""),
        temporal_context=str(row.get("temporal_context") or row.get("source_role") or ""),
        disposition=SourceTextDisposition(str(row.get("disposition") or "")),
        legal_force=str(row.get("legal_force") or ""),
        capabilities=tuple(SourceTextCapability(str(value)) for value in row.get("capabilities") or ()),
        legal_answer_eligible=row.get("legal_answer_eligible") is True,
        source_answer_eligible=row.get("source_answer_eligible") is True,
        legal_citation_eligible=row.get("legal_citation_eligible") is True,
        source_citation_eligible=row.get("source_citation_eligible") is True,
        default_highlight_eligible=row.get("default_highlight_eligible") is True,
        abstention_reason=str(row["abstention_reason"]) if row.get("abstention_reason") else None,
    )


def source_text_health(store) -> dict[str, int]:
    records = tuple(source_text_record(row) for row in store.raw_source_spans if str(row.get("raw_text") or "").strip())
    record_count = len(records)
    without_route = 0
    without_selector = 0
    without_geometry_or_reason = 0
    for record in records:
        without_route += not record.capabilities and not record.abstention_reason
        without_selector += not record.selector.stream_id
        without_geometry_or_reason += not record.geometry_available and not record.abstention_reason
    annotation_health = getattr(getattr(store.config, "strategy", None), "source_text_health", None)
    annotation = annotation_health(store) if annotation_health is not None else {}
    return {
        "nonempty_source_span_count": record_count,
        "meaningful_source_span_without_route_count": without_route,
        "source_text_without_selector_count": without_selector,
        "source_text_without_geometry_or_reason_count": without_geometry_or_reason,
        "semantic_join_missing_count": sum(record.semantic_join_status == "missing" for record in records),
        "semantic_join_duplicate_count": sum(record.semantic_join_status == "duplicate" for record in records),
        **annotation,
    }
