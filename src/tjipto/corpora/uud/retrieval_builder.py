from __future__ import annotations

from collections import defaultdict

from tjipto.corpora.uud.provenance_exceptions import apply_review_category
from tjipto.corpora.uud.structure_builder import compact


def build_retrieval_units(evidence: list[dict], chunks: list[dict]) -> list[dict]:
    chunks_by_unit = {row["legal_unit_id"]: row for row in chunks}
    return [
        {
            "bbox_sample_refs": _bbox_sample_refs(row),
            "bbox_total_count": len(row.get("bbox_refs", [])),
            "chunk_id": chunks_by_unit[row["legal_unit_id"]]["chunk_id"],
            "corpus_id": "uud",
            "evidence_id": row["evidence_id"],
            "legal_unit_id": row["legal_unit_id"],
            "page_numbers": row["page_numbers"],
            "retrieval_unit_id": f"uud_retrieval_unit::{row['evidence_id']}",
            "source_pdf_path": row["source_pdf_path"],
            "source_role": row["source_role"],
            "source_sha256": row["source_sha256"],
            **_retrieval_answerability(row, chunks_by_unit[row["legal_unit_id"]]),
            "temporal_context": row["temporal_context"],
            "text": _retrieval_unit_text(row),
        }
        for row in sorted(evidence, key=lambda item: item["evidence_id"])
    ]


def _retrieval_answerability(evidence: dict, chunk: dict) -> dict:
    if evidence.get("bbox_precision") == "page_grounded_only":
        return {"status": "excluded_public_answer", "rejection_reason": "page_grounded_only_not_answerable"}
    if evidence.get("bbox_precision") != "exact":
        return {"status": "excluded_public_answer", "rejection_reason": "missing_exact_grounding"}
    if evidence.get("viewer_highlightable") is False:
        return {"status": "excluded_public_answer", "rejection_reason": "viewer_not_highlightable"}
    if not evidence.get("text_span_ids"):
        return {"status": "excluded_public_answer", "rejection_reason": "missing_exact_text_span_support"}
    if not (evidence.get("bbox_ids") or evidence.get("bbox_refs")):
        return {"status": "excluded_public_answer", "rejection_reason": "missing_bbox"}
    if chunk.get("runtime_loadable") is False:
        return {"status": "excluded_public_answer", "rejection_reason": "linked_chunk_not_runtime_loadable"}
    return {"status": "accepted"}


def apply_chunk_grounding(
    chunks: list[dict],
    legal_units: list[dict],
    evidence: list[dict],
    page_text_spans: list[dict],
) -> None:
    units_by_id = {row["legal_unit_id"]: row for row in legal_units}
    evidence_by_unit: dict[str, list[dict]] = defaultdict(list)
    spans_by_page: dict[tuple[str, int], list[dict]] = defaultdict(list)
    source_meta = {}
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
        chunk["text_span_ids"] = _text_span_ids_for_text(spans_by_page, unit["source_document_id"], page_numbers, chunk["text"])
        chunk["grounding_status"] = "text_span_exact" if chunk["text_span_ids"] else "text_span_unavailable"
        if not chunk["text_span_ids"]:
            chunk["failure_reason"] = "text_span_exact_match_unavailable"
        chunk["runtime_loadable"] = unit.get("runtime_loadable") is not False and bool(chunk_evidence) and bool(chunk["text_span_ids"])
        chunk["validation_status"], chunk["validation_basis"] = _chunk_validation(chunk)
        apply_review_category(chunk)
    chunks_by_unit = {row["legal_unit_id"]: row for row in chunks}
    for unit in legal_units:
        chunk = chunks_by_unit.get(unit["legal_unit_id"])
        unit_evidence = evidence_by_unit.get(unit["legal_unit_id"], [])
        page_numbers = list(range(unit["page_start"], unit["page_end"] + 1))
        text_span_ids = _text_span_ids_for_text(spans_by_page, unit["source_document_id"], page_numbers, unit["text"])
        source = source_meta[unit["source_document_id"]]
        unit["source_role"] = source["source_role"]
        unit["temporal_context"] = source.get("temporal_context", unit["source_role"])
        unit["page_numbers"] = page_numbers
        unit["text_span_ids"] = text_span_ids
        unit["bbox_ids"] = [bbox_id for row in unit_evidence for bbox_id in row.get("bbox_refs") or ()]
        unit["grounding_status"] = "text_span_exact" if text_span_ids else "text_span_unavailable"
        unit["validation_status"] = "accepted_grounding" if text_span_ids else "grounding_unavailable"
        if not text_span_ids:
            unit["failure_reason"] = "text_span_exact_match_unavailable"
        unit["runtime_loadable"] = unit.get("runtime_loadable") is not False and bool(text_span_ids) and bool(unit_evidence or (chunk and chunk.get("runtime_loadable")))
        apply_review_category(unit)


def _text_span_ids_for_text(
    spans_by_page: dict[tuple[str, int], list[dict]],
    source_document_id: str,
    page_numbers: list[int],
    text: str,
) -> list[str]:
    expected = [compact(line) for line in (text or "").splitlines() if compact(line)]
    if not expected:
        return []
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
            return matched
    return []


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
