from __future__ import annotations

import json

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
    marker = row.get("classification") == "source_annotation_marker"
    legal = row.get("legal_text") is True
    semantic = str(row.get("semantic_text") or "").strip()
    if marker:
        disposition = SourceTextDisposition.SOURCE_ANNOTATION
        capabilities = (SourceTextCapability.ANNOTATION_ANSWER,)
    elif legal:
        disposition = SourceTextDisposition.LEGAL_TEXT
        capabilities = (SourceTextCapability.LEGAL_ANSWER,)
    elif semantic and row.get("citation_eligible") is True:
        disposition = SourceTextDisposition.SOURCE_FACT
        capabilities = (SourceTextCapability.SOURCE_FACT_ANSWER,)
    else:
        disposition = SourceTextDisposition.EXTRACTION_ARTIFACT
        capabilities = (SourceTextCapability.AUDIT_ONLY,)
    source_value = str(row.get("raw_text") or "")
    normalized_value = semantic or " ".join(source_value.split())
    return SourceTextRecord(
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
        disposition=disposition,
        legal_force="legal_norm" if legal else "source_annotation" if marker else "source_fact" if semantic else "none",
        capabilities=capabilities,
        legal_answer_eligible=legal,
        source_answer_eligible=legal or marker or bool(semantic),
        legal_citation_eligible=legal and row.get("citation_eligible") is True,
        source_citation_eligible=not marker and row.get("citation_eligible") is True,
        default_highlight_eligible=not marker and row.get("default_highlight_eligible") is True,
        abstention_reason=None if legal or marker or semantic else str(row.get("disposition_reason") or "audit_only"),
    )


def source_text_health(store) -> dict[str, int]:
    record_count = 0
    without_route = 0
    without_selector = 0
    without_geometry_or_reason = 0
    with store.config.artifact_path("raw_source_spans").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not str(row.get("raw_text") or "").strip():
                continue
            record = source_text_record(row)
            record_count += 1
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
        **annotation,
    }
