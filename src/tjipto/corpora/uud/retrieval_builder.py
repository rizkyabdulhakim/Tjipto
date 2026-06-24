from __future__ import annotations

from collections import defaultdict


def apply_chunk_grounding(
    chunks: list[dict],
    legal_units: list[dict],
    evidence: list[dict],
    page_text_spans: list[dict],
) -> None:
    units_by_id = {row["legal_unit_id"]: row for row in legal_units}
    evidence_by_unit: dict[str, list[dict]] = defaultdict(list)
    spans_by_page: dict[tuple[str, int], list[str]] = defaultdict(list)
    for row in evidence:
        evidence_by_unit[row["legal_unit_id"]].append(row)
    for row in page_text_spans:
        spans_by_page[(row["source_document_id"], row["page_number"])].append(row["text_span_id"])
    for chunk in chunks:
        unit = units_by_id[chunk["legal_unit_id"]]
        page_range = chunk["page_range"]
        page_numbers = list(range(page_range["start_page_number"], page_range["end_page_number"] + 1))
        chunk_evidence = evidence_by_unit.get(chunk["legal_unit_id"], [])
        chunk["page_numbers"] = page_numbers
        chunk["evidence_ids"] = [row["evidence_id"] for row in chunk_evidence]
        chunk["bbox_ids"] = [bbox_id for row in chunk_evidence for bbox_id in row.get("bbox_refs") or ()]
        chunk["text_span_ids"] = [
            span_id
            for page_number in page_numbers
            for span_id in spans_by_page.get((unit["source_document_id"], page_number), [])
        ]
        chunk["grounding_status"] = "text_span_page_grounded" if chunk["text_span_ids"] else "text_span_unavailable"
        chunk["runtime_loadable"] = unit.get("runtime_loadable") is not False and bool(chunk_evidence)
