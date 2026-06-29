from __future__ import annotations

from collections import defaultdict

from tjipto.corpora.uud.structure_builder import compact


def apply_chunk_grounding(
    chunks: list[dict],
    legal_units: list[dict],
    evidence: list[dict],
    page_text_spans: list[dict],
) -> None:
    units_by_id = {row["legal_unit_id"]: row for row in legal_units}
    evidence_by_unit: dict[str, list[dict]] = defaultdict(list)
    spans_by_page: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in evidence:
        evidence_by_unit[row["legal_unit_id"]].append(row)
    for row in page_text_spans:
        spans_by_page[(row["source_document_id"], row["page_number"])].append(row)
    for chunk in chunks:
        unit = units_by_id[chunk["legal_unit_id"]]
        page_range = chunk["page_range"]
        page_numbers = list(range(page_range["start_page_number"], page_range["end_page_number"] + 1))
        chunk_evidence = evidence_by_unit.get(chunk["legal_unit_id"], [])
        chunk["page_numbers"] = page_numbers
        chunk["evidence_ids"] = [row["evidence_id"] for row in chunk_evidence]
        chunk["bbox_ids"] = [bbox_id for row in chunk_evidence for bbox_id in row.get("bbox_refs") or ()]
        chunk["text_span_ids"] = _text_span_ids_for_text(spans_by_page, unit["source_document_id"], page_numbers, chunk["text"])
        chunk["grounding_status"] = "text_span_exact" if chunk["text_span_ids"] else "text_span_unavailable"
        if not chunk["text_span_ids"]:
            chunk["failure_reason"] = "text_span_exact_match_unavailable"
        chunk["runtime_loadable"] = unit.get("runtime_loadable") is not False and bool(chunk_evidence) and bool(chunk["text_span_ids"])
    chunks_by_unit = {row["legal_unit_id"]: row for row in chunks}
    for unit in legal_units:
        chunk = chunks_by_unit.get(unit["legal_unit_id"])
        unit_evidence = evidence_by_unit.get(unit["legal_unit_id"], [])
        page_numbers = list(range(unit["page_start"], unit["page_end"] + 1))
        text_span_ids = _text_span_ids_for_text(spans_by_page, unit["source_document_id"], page_numbers, unit["text"])
        unit["source_role"] = unit["source_document_id"].split("::", 1)[1]
        unit["temporal_context"] = unit["source_role"]
        unit["page_numbers"] = page_numbers
        unit["text_span_ids"] = text_span_ids
        unit["bbox_ids"] = [bbox_id for row in unit_evidence for bbox_id in row.get("bbox_refs") or ()]
        unit["grounding_status"] = "text_span_exact" if text_span_ids else "text_span_unavailable"
        unit["validation_status"] = "accepted_grounding" if text_span_ids else "grounding_unavailable"
        if not text_span_ids:
            unit["failure_reason"] = "text_span_exact_match_unavailable"
        unit["runtime_loadable"] = unit.get("runtime_loadable") is not False and bool(text_span_ids) and bool(unit_evidence or (chunk and chunk.get("runtime_loadable")))


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
