from __future__ import annotations

from collections import defaultdict, deque
import re

from tjipto.contracts.source_text import SourceTextCapability, SourceTextDisposition


_ALLOWED_CLASS_FORCES = {
    "normative_constitutional_text": {"canonical_normative", "historical_normative"},
    "structural_heading": {"metadata_only"},
    "amendment_instrument_text": {"amendment_instrument"},
    "decision_clause": {"amendment_instrument"},
    "session_institution_metadata": {"amendment_instrument", "metadata_only"},
    "signatory_block": {"amendment_instrument"},
    "header_footer": {"nonlegal"},
    "separator": {"nonlegal"},
}


def project_source_text_rows(raw_rows: list[dict], page_text_spans: list[dict]) -> list[dict]:
    """Join immutable raw source rows to accepted semantics once for runtime publication."""
    semantic_index: dict[tuple[object, ...], list[dict]] = defaultdict(list)
    for span in page_text_spans:
        semantic_index[_semantic_key(span)].append(span)
    projected = []
    for raw in raw_rows:
        if not str(raw.get("raw_text") or "").strip():
            continue
        semantic_text = str(raw.get("semantic_text") or "").strip()
        if semantic_text:
            matches = semantic_index[_raw_semantic_key(raw)]
            if len(matches) != 1:
                semantics = _fail_closed_semantics(raw, "semantic_join_missing" if not matches else "semantic_join_duplicate")
            else:
                semantics = source_text_semantics(raw, matches[0])
        else:
            semantics = source_text_semantics(raw, None)
        projected.append(raw | semantics)
    return projected


def source_text_semantics(raw: dict, semantic_span: dict | None) -> dict:
    marker = raw.get("classification") == "source_annotation_marker"
    if marker:
        return _semantics(
            raw,
            disposition=SourceTextDisposition.SOURCE_ANNOTATION,
            legal_force="nonlegal",
            capabilities=(SourceTextCapability.ANNOTATION_ANSWER,),
            source_answer_eligible=True,
            semantic_join_status="not_applicable",
        )
    if semantic_span is None:
        return _fail_closed_semantics(raw, str(raw.get("disposition_reason") or "nonsemantic_source_text"))

    classification = str(semantic_span.get("semantic_classification") or "")
    legal_force = str(semantic_span.get("legal_force") or "")
    if legal_force not in _ALLOWED_CLASS_FORCES.get(classification, set()):
        return _fail_closed_semantics(raw, "unsupported_semantic_mapping")

    if classification == "normative_constitutional_text":
        disposition = SourceTextDisposition.LEGAL_TEXT
        capabilities = (SourceTextCapability.LEGAL_ANSWER,)
    elif classification == "structural_heading":
        disposition = SourceTextDisposition.STRUCTURAL_TEXT
        capabilities = (SourceTextCapability.STRUCTURAL_ANSWER,)
    elif classification in {"amendment_instrument_text", "decision_clause", "signatory_block"} or (
        classification == "session_institution_metadata" and legal_force == "amendment_instrument"
    ):
        disposition = SourceTextDisposition.INSTRUMENT_TEXT
        capabilities = (SourceTextCapability.INSTRUMENT_ANSWER,)
    elif classification == "session_institution_metadata":
        disposition = SourceTextDisposition.SOURCE_FACT
        capabilities = (SourceTextCapability.SOURCE_FACT_ANSWER,)
    elif classification == "header_footer":
        disposition = SourceTextDisposition.DOCUMENT_FURNITURE
        capabilities = (SourceTextCapability.SOURCE_FORMAT_ANSWER,)
    else:
        disposition = SourceTextDisposition.LAYOUT_SEPARATOR
        capabilities = (SourceTextCapability.AUDIT_ONLY,)

    canonical = classification == "normative_constitutional_text" and legal_force == "canonical_normative"
    audit_only = disposition is SourceTextDisposition.LAYOUT_SEPARATOR
    geometry = all(raw.get(key) is not None for key in ("x0", "y0", "x1", "y1"))
    return _semantics(
        raw,
        disposition=disposition,
        legal_force=legal_force,
        capabilities=capabilities,
        source_answer_eligible=not audit_only,
        source_citation_eligible=geometry and not audit_only,
        legal_answer_eligible=canonical,
        legal_citation_eligible=canonical and semantic_span.get("viewer_highlightable") is True,
        default_highlight_eligible=canonical and semantic_span.get("viewer_highlightable") is True,
        abstention_reason="audit_only_layout_separator" if audit_only else None,
        semantic_span=semantic_span,
        semantic_join_status="exact",
    )


def _semantics(
    raw: dict,
    *,
    disposition: SourceTextDisposition,
    legal_force: str,
    capabilities: tuple[SourceTextCapability, ...],
    source_answer_eligible: bool,
    source_citation_eligible: bool = False,
    legal_answer_eligible: bool = False,
    legal_citation_eligible: bool = False,
    default_highlight_eligible: bool = False,
    abstention_reason: str | None = None,
    semantic_span: dict | None = None,
    semantic_join_status: str,
) -> dict:
    temporal_context = str((semantic_span or {}).get("temporal_context") or raw.get("source_role") or "")
    return {
        "semantic_text_span_id": (semantic_span or {}).get("text_span_id"),
        "semantic_classification": (semantic_span or {}).get("semantic_classification"),
        "semantic_join_status": semantic_join_status,
        "temporal_context": temporal_context,
        "disposition": disposition.value,
        "legal_force": legal_force,
        "capabilities": [capability.value for capability in capabilities],
        "legal_answer_eligible": legal_answer_eligible,
        "source_answer_eligible": source_answer_eligible,
        "legal_citation_eligible": legal_citation_eligible,
        "source_citation_eligible": source_citation_eligible,
        "default_highlight_eligible": default_highlight_eligible,
        "abstention_reason": abstention_reason,
        # Compatibility fields are narrowed in the runtime projection only;
        # immutable raw artifact rows retain their extraction-time values.
        "legal_text": legal_answer_eligible,
        "citation_eligible": source_citation_eligible,
        "relevant_quote_eligible": legal_citation_eligible,
        "viewer_eligible": source_citation_eligible,
        "viewer_highlightable": source_citation_eligible,
    }


def _fail_closed_semantics(raw: dict, reason: str) -> dict:
    return _semantics(
        raw,
        disposition=SourceTextDisposition.EXTRACTION_ARTIFACT,
        legal_force="nonlegal",
        capabilities=(SourceTextCapability.AUDIT_ONLY,),
        source_answer_eligible=False,
        abstention_reason=reason,
        semantic_join_status=reason.removeprefix("semantic_join_") if reason.startswith("semantic_join_") else "not_applicable",
    )


def _semantic_key(span: dict) -> tuple[object, ...]:
    return (span.get("source_document_id"), span.get("page_number"), span.get("text_start"), span.get("text_end"))


def _raw_semantic_key(raw: dict) -> tuple[object, ...]:
    return (
        raw.get("source_document_id"),
        raw.get("page_number"),
        raw.get("semantic_text_start"),
        raw.get("semantic_text_end"),
    )


def validate_source_text_closure(rows: list[dict]) -> tuple[str, ...]:
    legends = _legend_markers(rows)
    errors = []
    for row in rows:
        row_id = str(row.get("raw_source_span_id") or "unknown")
        value = str(row.get("raw_text") or "")
        if not value.strip():
            continue
        if not row.get("raw_stream_id") or row.get("raw_text_start") is None or row.get("raw_text_end") is None:
            errors.append(f"source_text_selector_missing:{row_id}")
        if not all(row.get(field) is not None for field in ("x0", "y0", "x1", "y1")) and not row.get("disposition_reason"):
            errors.append(f"source_text_geometry_or_reason_missing:{row_id}")
        marker = row.get("classification") == "source_annotation_marker"
        if marker:
            if value.strip() == ":":
                errors.append(f"ordinary_punctuation_annotation:{row_id}")
            parts = {f"{stars})" for stars in re.findall(r"\*+", value)}
            if not parts or not parts <= legends:
                errors.append(f"source_annotation_unmapped:{row_id}")
            if any(row.get(field) is not False for field in ("legal_text", "citation_eligible", "default_highlight_eligible")):
                errors.append(f"source_annotation_authority_leak:{row_id}")
        elif not str(row.get("semantic_text") or "").strip() and not row.get("disposition_reason"):
            errors.append(f"meaningful_source_text_without_route_or_review:{row_id}")
    return tuple(errors)


def _legend_markers(rows: list[dict]) -> set[str]:
    candidates: dict[str, set[str]] = defaultdict(set)
    window: deque[dict] = deque(maxlen=3)
    identity = None
    for row in rows:
        row_identity = (row.get("source_document_id"), row.get("page_number"))
        if row_identity != identity:
            window.clear()
            identity = row_identity
        window.append(row)
        if len(window) != 3:
            continue
        marker, separator, meaning = window
        if (
            marker.get("classification") == "source_annotation_marker"
            and separator.get("raw_text") == ":"
            and str(meaning.get("semantic_text") or "").casefold().startswith("perubahan ")
        ):
            candidates[str(marker.get("raw_text") or "")].add(str(meaning.get("semantic_text") or "").casefold())
    return {marker for marker, meanings in candidates.items() if marker and len(meanings) == 1}
