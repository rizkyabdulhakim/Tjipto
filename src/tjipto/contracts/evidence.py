from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence

EVIDENCE_DECISION_FIELDS = (
    "classification",
    "legal_force",
    "evidence_ids",
    "span_bbox_ids",
    "object_role",
    "linked_authority",
    "viewer_highlightable",
    "reason_code",
    "reason",
)


def normalize_source_text(text: object) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or "")).replace("\xad", "").replace("\xa0", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"\s+([,.;:])", r"\1", normalized)
    return normalized.strip().casefold()


def exact_quote_support_reason(
    *,
    quoted_text: object,
    source_document_id: object,
    page_numbers: Sequence[object],
    text_span_ids: Sequence[object],
    bbox_refs: Sequence[object],
    spans_by_id: Mapping[object, Mapping],
    bboxes_by_id: Mapping[object, Mapping],
) -> str | None:
    quote = normalize_source_text(quoted_text)
    if not quote:
        return "quote_missing"
    pages = {page for page in page_numbers if isinstance(page, int)}
    spans = [spans_by_id.get(span_id) for span_id in text_span_ids]
    if not text_span_ids or any(span is None for span in spans):
        return "quote_span_reference_missing"
    if any(span.get("source_document_id") != source_document_id or span.get("page_number") not in pages for span in spans if span):
        return "quote_span_provenance_mismatch"
    bboxes = [bboxes_by_id.get(bbox_id) for bbox_id in bbox_refs]
    if not bbox_refs or any(bbox is None for bbox in bboxes):
        return "quote_bbox_reference_missing"
    if any(bbox.get("source_document_id") != source_document_id or bbox.get("page_number") not in pages for bbox in bboxes if bbox):
        return "quote_bbox_provenance_mismatch"
    if quote not in normalize_source_text(" ".join(str(span.get("text") or "") for span in spans if span)):
        return "quote_not_in_text_spans"
    if quote not in normalize_source_text(" ".join(str(bbox.get("text") or "") for bbox in bboxes if bbox)):
        return "quote_not_in_bbox_text"
    return None


def source_lineage_reason(
    *,
    evidence: Mapping,
    source_documents_by_id: Mapping[object, Mapping],
    spans_by_id: Mapping[object, Mapping],
    bboxes_by_id: Mapping[object, Mapping],
) -> str | None:
    source_id = evidence.get("source_document_id")
    source = source_documents_by_id.get(source_id)
    if source is None:
        return "source_document_unresolved"
    if evidence.get("source_sha256") != source.get("sha256"):
        return "source_sha256_mismatch"
    if evidence.get("source_pdf_path") != source.get("path"):
        return "source_pdf_path_mismatch"
    if evidence.get("source_role") != source.get("source_role"):
        return "source_role_mismatch"
    if evidence.get("temporal_context") != source.get("temporal_context"):
        return "temporal_context_mismatch"
    reason = exact_quote_support_reason(
        quoted_text=evidence.get("quoted_text"),
        source_document_id=source_id,
        page_numbers=evidence.get("page_numbers") or (),
        text_span_ids=evidence.get("text_span_ids") or (),
        bbox_refs=evidence.get("bbox_refs") or (),
        spans_by_id=spans_by_id,
        bboxes_by_id=bboxes_by_id,
    )
    if reason:
        return reason
    for bbox_id in evidence.get("bbox_refs") or ():
        bbox = bboxes_by_id.get(bbox_id)
        if bbox is None or bbox.get("source_sha256") != source.get("sha256"):
            return "bbox_source_sha256_mismatch"
    for span_id in evidence.get("text_span_ids") or ():
        span = spans_by_id.get(span_id)
        if span is None or span.get("source_sha256") != source.get("sha256"):
            return "span_source_sha256_mismatch"
    return None
