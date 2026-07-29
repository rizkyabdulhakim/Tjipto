from __future__ import annotations

from collections import defaultdict

from tjipto.corpora.uud.provenance_exceptions import (
    ACCEPTED_FALSE_POSITIVE_SEGMENTATION_PUNCTUATION,
    apply_review_category,
    review_category,
)
from tjipto.corpora.uud.structure_builder import compact, matching_sequence


def build_retrieval_units(evidence: list[dict], chunks: list[dict]) -> list[dict]:
    chunks_by_unit = {row["legal_unit_id"]: row for row in chunks}
    rows = [
        {
            "chunk_id": chunks_by_unit[row["legal_unit_id"]]["chunk_id"],
            "evidence_id": row["evidence_id"],
            "legal_unit_id": row["legal_unit_id"],
            "retrieval_unit_id": f"uud_retrieval_unit::{row['evidence_id']}",
            "source_role": row["source_role"],
            "object_role": "retrieval_index_record",
            "artifact_status": "published",
            "page_locator": {"source_document_id": row["source_document_id"], "page_numbers": row["page_numbers"]},
            "retrieval_terms": compact(row.get("quoted_text") or ""),
            "temporal_context": row["temporal_context"],
            "text": _retrieval_unit_text(row),
        }
        | _constitutional_retrieval_fields(row)
        | _retrieval_answerability(row, chunks_by_unit[row["legal_unit_id"]])
        for row in sorted(evidence, key=lambda item: item["evidence_id"])
    ]
    return [row for row in rows if row["artifact_status"] == "published"]


def _constitutional_retrieval_fields(row: dict) -> dict:
    hierarchy = row.get("hierarchy") or ()
    if hierarchy[:1] != ["ATURAN TAMBAHAN"] or not any(label in {"Pasal I", "Pasal II"} for label in hierarchy):
        return {}
    return {
        "provision_kind": "normative_constitutional_text",
        "anomaly": False,
        "source_conflict": False,
        "citation_eligible": True,
        "viewer_eligible": True,
        "relevant_quote_eligible": True,
    }


def _retrieval_answerability(evidence: dict, chunk: dict) -> dict:
    if evidence.get("bbox_precision") == "page_grounded_only":
        return {"artifact_status": "excluded"}
    if evidence.get("bbox_precision") != "exact":
        return {"artifact_status": "excluded"}
    if evidence.get("viewer_highlightable") is False:
        return {"artifact_status": "excluded"}
    if not evidence.get("text_span_ids"):
        return {"artifact_status": "excluded"}
    if not (evidence.get("bbox_ids") or evidence.get("bbox_refs")):
        return {"artifact_status": "excluded"}
    if chunk.get("runtime_loadable") is False:
        return {"artifact_status": "excluded"}
    return {"artifact_status": "published"}


def apply_chunk_grounding(
    chunks: list[dict],
    legal_units: list[dict],
    evidence: list[dict],
    page_text_spans: list[dict],
) -> None:
    units_by_id = {row["legal_unit_id"]: row for row in legal_units}
    evidence_by_unit: dict[str, list[dict]] = defaultdict(list)
    spans_by_page: dict[tuple[str, int], list[dict]] = defaultdict(list)
    source_meta: dict[str, dict] = {}
    for row in evidence:
        evidence_by_unit[row["legal_unit_id"]].append(row)
    for row in page_text_spans:
        spans_by_page[(row["source_document_id"], row["page_number"])].append(row)
        source_meta.setdefault(row["source_document_id"], row)
    for row in evidence:
        row["bbox_ids"] = list(row.get("bbox_refs") or ())
        row["text_span_ids"] = _text_span_ids_for_text(
            spans_by_page,
            row["source_document_id"],
            list(row.get("page_numbers") or ()),
            row.get("quoted_text") or "",
        )
    for chunk in chunks:
        unit = units_by_id[chunk["legal_unit_id"]]
        source = source_meta[unit["source_document_id"]]
        page_range = chunk["page_range"]
        page_numbers = list(range(page_range["start_page_number"], page_range["end_page_number"] + 1))
        chunk_evidence = evidence_by_unit.get(chunk["legal_unit_id"], [])
        chunk["page_numbers"] = page_numbers
        chunk["source_document_id"] = unit["source_document_id"]
        chunk["source_role"] = source["source_role"]
        chunk["temporal_context"] = source.get("temporal_context", source["source_role"])
        chunk["evidence_ids"] = [row["evidence_id"] for row in chunk_evidence]
        chunk["bbox_ids"] = [bbox_id for row in chunk_evidence for bbox_id in row.get("bbox_refs") or ()]
        chunk["text_span_ids"], chunk_grounding_status = _text_span_match_for_text(
            spans_by_page,
            unit["source_document_id"],
            page_numbers,
            chunk["text"],
            allow_containing_span=not chunk_evidence and review_category(chunk) == ACCEPTED_FALSE_POSITIVE_SEGMENTATION_PUNCTUATION,
        )
        if not chunk["text_span_ids"] and chunk_evidence:
            evidence_span_ids = [span_id for item in chunk_evidence for span_id in item.get("text_span_ids") or ()]
            if evidence_span_ids:
                chunk["text_span_ids"] = list(dict.fromkeys(evidence_span_ids))
                chunk_grounding_status = "text_span_exact_from_evidence"
        chunk["grounding_status"] = chunk_grounding_status
        if not chunk["text_span_ids"]:
            chunk["failure_reason"] = "text_span_exact_match_unavailable"
        chunk["runtime_loadable"] = unit.get("runtime_loadable") is not False and bool(chunk_evidence) and bool(chunk["text_span_ids"])
        if (
            chunk["runtime_loadable"]
            and unit.get("unit_type") == "bab_record"
            and "dihapus" in compact(unit.get("text"))
            and unit.get("source_role") != "current_consolidated"
        ):
            chunk["canonical_use_allowed"] = True
            chunk["status"] = "active_canonical_record"
        chunk["validation_status"], chunk["validation_basis"] = _chunk_validation(chunk)
        apply_review_category(chunk)
    chunks_by_unit = {row["legal_unit_id"]: row for row in chunks}
    descendants_by_parent = _descendants_by_parent(legal_units)
    for unit in legal_units:
        unit_chunk = chunks_by_unit.get(unit["legal_unit_id"])
        unit_evidence = evidence_by_unit.get(unit["legal_unit_id"], [])
        page_numbers = list(range(unit["page_start"], unit["page_end"] + 1))
        text_span_ids, grounding_status = _text_span_match_for_text(
            spans_by_page,
            unit["source_document_id"],
            page_numbers,
            unit["text"],
            allow_containing_span=not unit_evidence and review_category(unit) == ACCEPTED_FALSE_POSITIVE_SEGMENTATION_PUNCTUATION,
        )
        source = source_meta[unit["source_document_id"]]
        unit["source_role"] = source["source_role"]
        unit["temporal_context"] = source.get("temporal_context", unit["source_role"])
        unit["page_numbers"] = page_numbers
        unit["text_span_ids"] = text_span_ids
        unit["evidence_ids"] = [row["evidence_id"] for row in unit_evidence]
        unit["bbox_ids"] = [bbox_id for row in unit_evidence for bbox_id in row.get("bbox_refs") or ()]
        if not text_span_ids and unit.get("status") == "active_historical_record" and unit_evidence:
            text_span_ids = list(dict.fromkeys(span_id for row in unit_evidence for span_id in row.get("text_span_ids") or ()))
            unit["text_span_ids"] = text_span_ids
            grounding_status = "text_span_exact_from_evidence"
        if descendants_by_parent.get(unit["legal_unit_id"]):
            aggregate_span_ids = list(
                dict.fromkeys(span_id for row in unit_evidence for span_id in row.get("text_span_ids") or ())
            )
            if aggregate_span_ids:
                text_span_ids = aggregate_span_ids
                unit["text_span_ids"] = aggregate_span_ids
                grounding_status = "text_span_aggregate_from_evidence"
        unit["grounding_status"] = grounding_status
        unit["validation_status"] = "accepted_grounding" if text_span_ids else "grounding_unavailable"
        if not text_span_ids:
            unit["failure_reason"] = "text_span_exact_match_unavailable"
        unit["runtime_loadable"] = (
            unit.get("runtime_loadable") is not False
            and bool(text_span_ids)
            and bool(unit_evidence or (unit_chunk and unit_chunk.get("runtime_loadable")))
        )
        apply_review_category(unit)
    for parent_id in descendants_by_parent:
        parent = units_by_id.get(parent_id)
        aggregate_span_ids = list(
            dict.fromkeys(span_id for row in evidence_by_unit.get(parent_id, ()) for span_id in row.get("text_span_ids") or ())
        )
        if parent is not None and aggregate_span_ids:
            parent["text_span_ids"] = aggregate_span_ids
            parent["grounding_status"] = "text_span_aggregate_from_evidence"
            parent["validation_status"] = "accepted_grounding"
            parent.pop("failure_reason", None)


def _descendants_by_parent(legal_units: list[dict]) -> dict[str, list[dict]]:
    descendants: dict[str, list[dict]] = defaultdict(list)
    for unit in legal_units:
        for parent_id in unit.get("parent_legal_unit_ids") or ():
            descendants[parent_id].append(unit)
    return descendants


def _text_span_ids_for_text(
    spans_by_page: dict[tuple[str, int], list[dict]],
    source_document_id: str,
    page_numbers: list[int],
    text: str,
    *,
    allow_containing_span: bool = False,
) -> list[str]:
    return _text_span_match_for_text(
        spans_by_page,
        source_document_id,
        page_numbers,
        text,
        allow_containing_span=allow_containing_span,
    )[0]


def _text_span_match_for_text(
    spans_by_page: dict[tuple[str, int], list[dict]],
    source_document_id: str,
    page_numbers: list[int],
    text: str,
    *,
    allow_containing_span: bool = False,
) -> tuple[list[str], str]:
    expected = [compact(line) for line in (text or "").splitlines() if compact(line)]
    if not expected:
        return [], "text_span_unavailable"
    rows = [span for page_number in page_numbers for span in spans_by_page.get((source_document_id, page_number), [])]
    matched_sequence = matching_sequence(rows, text)
    if matched_sequence:
        return [row["text_span_id"] for row in matched_sequence], "text_span_exact"
    if allow_containing_span:
        wanted = compact(text)
        containing = [row["text_span_id"] for row in rows if wanted and wanted in compact(row.get("text"))]
        if containing:
            return containing, "text_span_containing_match"
        matched_containing: list[str] = []
        target_index = 0
        for row in rows:
            if target_index >= len(expected):
                break
            if expected[target_index] in compact(row.get("text")):
                matched_containing.append(row["text_span_id"])
                target_index += 1
        if target_index >= len(expected):
            return matched_containing, "text_span_containing_match"
    matched: list[str] = []
    target_index = 0
    for page_number in page_numbers:
        for span in spans_by_page.get((source_document_id, page_number), []):
            if target_index >= len(expected):
                break
            if compact(span.get("text")) != expected[target_index]:
                continue
            matched.append(span["text_span_id"])
            target_index += 1
        if target_index >= len(expected):
            return matched, "text_span_exact"
    return [], "text_span_unavailable"


def _chunk_validation(chunk: dict) -> tuple[str, str]:
    if chunk.get("runtime_loadable") is True:
        if chunk.get("evidence_ids") and chunk.get("bbox_ids") and chunk.get("text_span_ids"):
            return "validated_grounded_chunk", "evidence_bbox_text_span"
        return "validation_error_missing_grounding", "missing_required_grounding"
    if chunk.get("failure_reason"):
        return "excluded_or_non_runtime_chunk", "failure_reason"
    return "structural_context_validated", "legal_unit_text_span_context"


def rebuild_retrieval(existing: dict, chunk: dict, retrieval_units: list[dict]) -> None:
    quoted = existing["quoted_text"]
    for row in retrieval_units:
        if row["evidence_id"] == existing["evidence_id"]:
            row["page_numbers"] = existing["page_numbers"]
            row["bbox_sample_refs"] = existing["bbox_refs"][:1]
            row["bbox_total_count"] = len(existing["bbox_refs"])
            row["text"] = retrieval_text(existing["citation"], existing.get("hierarchy") or [], quoted)
            chunk["text"] = quoted if chunk["status"] == "active_canonical_record" else chunk["text"]
            break


def retrieval_text(citation: str | None, hierarchy: list[str] | tuple[str, ...], quoted_text: str) -> str:
    prefix = " ".join([item for item in [citation, *hierarchy] if item])
    return f"{prefix}\n{quoted_text}".strip()


def _retrieval_unit_text(row: dict) -> str:
    hierarchy = row.get("hierarchy") or []
    if row["evidence_id"].startswith("uud_instrument_final_citation_evidence::"):
        if hierarchy == [row.get("citation")]:
            hierarchy = []
        return retrieval_text(row.get("citation"), hierarchy, row["quoted_text"])
    first_line = row["quoted_text"].splitlines()[0]
    prefix = " ".join([item for item in [row.get("citation"), *hierarchy] if item])
    separator = "\n" if row["source_role"] == "current_consolidated" and "\u00a0" in first_line else " "
    return f"{prefix}{separator}{row['quoted_text']}".strip()


def _bbox_sample_refs(row: dict) -> list[str]:
    refs = row.get("bbox_refs", [])
    return [next((ref for ref in refs if ref.endswith("::0000")), refs[0])] if refs else []
