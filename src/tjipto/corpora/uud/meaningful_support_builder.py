from __future__ import annotations

from collections import defaultdict
from hashlib import sha256


REVIEWED_HEADINGS = {
    "uud_text_span::current_consolidated::0002::0010": (
        "document_title",
        "reviewed_publication_title_block_before_preamble",
    ),
    "uud_text_span::current_consolidated::0003::0017": (
        "structural",
        "reviewed_document_title_before_operational_body",
    ),
}
PASAL_III_SPAN = "uud_text_span::amendment_4_historical::0006::0000"
PASAL_III_CONFLICT = "uud_1945_amendment_4_aturan_tambahan_pasal_ii_iii_conflict"


def build_meaningful_support_units(
    *,
    page_text_spans: list[dict],
    raw_source_spans: list[dict],
    evidence: list[dict],
    metadata_grounding: list[dict],
    source_conflicts: list[dict],
    bbox_registry: list[dict],
    word_bboxes: list[dict],
) -> list[dict]:
    """Project reviewed support decisions without becoming an authority source."""
    spans = {row["text_span_id"]: row for row in page_text_spans if row["legal_force"] != "nonlegal"}
    evidence_by_id = {row["evidence_id"]: row for row in evidence}
    metadata_by_id = {row["metadata_grounding_id"]: row for row in metadata_grounding}
    conflicts_by_id = {row["source_conflict_id"]: row for row in source_conflicts}
    raw_by_selector = {
        (
            row["source_document_id"],
            row["page_number"],
            row.get("semantic_text_start"),
            row.get("semantic_text_end"),
        ): row
        for row in raw_source_spans
        if row.get("semantic_text")
    }
    words_by_page: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in word_bboxes:
        words_by_page[(row["source_document_id"], row["page_number"])].append(row)
    geometry_by_id: dict[str, dict] = {}
    for row in (*bbox_registry, *word_bboxes):
        geometry_id = row.get("bbox_id") or row.get("word_bbox_id")
        if geometry_id:
            geometry_by_id[str(geometry_id)] = row

    assignments: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for span in spans.values():
        span_id = span["text_span_id"]
        evidence_ids = span.get("evidence_ids") or []
        if evidence_ids:
            owner_id = min(evidence_ids, key=lambda item: (len(evidence_by_id[item]["text_span_ids"]), item))
            assignments[("evidence_registry", owner_id)].append(span)
        elif span_id in REVIEWED_HEADINGS:
            assignments[("page_text_span_review", span_id)].append(span)
        elif span_id == PASAL_III_SPAN:
            assignments[("source_conflict", PASAL_III_CONFLICT)].append(span)
        else:
            owner_id = span.get("promotion_target_id")
            if not isinstance(owner_id, str) or owner_id not in metadata_by_id:
                raise ValueError(f"meaningful span has no reviewed owner: {span_id}")
            assignments[("metadata_grounding", owner_id)].append(span)

    rows: list[dict] = []
    for (owner_type, owner_id), owned_spans in assignments.items():
        for segment in _segments(owned_spans):
            raw_rows = [
                raw_by_selector[(span["source_document_id"], span["page_number"], span["text_start"], span["text_end"])]
                for span in segment
            ]
            owner = (
                evidence_by_id[owner_id]
                if owner_type == "evidence_registry"
                else metadata_by_id[owner_id]
                if owner_type == "metadata_grounding"
                else conflicts_by_id[owner_id]
                if owner_type == "source_conflict"
                else segment[0]
            )
            bbox_refs = _bbox_refs(owner_type, owner, raw_rows, words_by_page, geometry_by_id)
            classification = segment[0]["semantic_classification"]
            legal_force = segment[0]["legal_force"]
            citation_final = owner.get("citation_final") is True if owner_type != "page_text_span_review" else False
            authority_kind = owner.get("authority_kind") or "structural_context"
            support_kind, reason = _support_decision(owner_type, owner_id, authority_kind)
            row_key = sha256("\n".join(span["text_span_id"] for span in segment).encode()).hexdigest()[:16]
            rows.append({
                "support_unit_id": f"uud_meaningful_support::{owner_type}::{row_key}",
                "decision_kind": "reviewed_support" if owner_type == "page_text_span_review" else "canonical_owner_support",
                "support_kind": support_kind,
                "owner_type": owner_type,
                "owner_id": owner_id,
                "source_document_id": segment[0]["source_document_id"],
                "source_role": segment[0]["source_role"],
                "temporal_context": segment[0]["temporal_context"],
                "semantic_classification": classification,
                "legal_force": legal_force,
                "authority_kind": authority_kind,
                "citation_final": citation_final,
                "text_span_ids": [span["text_span_id"] for span in segment],
                "raw_source_span_ids": [row["raw_source_span_id"] for row in raw_rows],
                "page_numbers": sorted({span["page_number"] for span in segment}),
                "selector_refs": [row["raw_source_span_id"] for row in raw_rows],
                "bbox_refs": bbox_refs,
                "bbox_precision": "exact" if bbox_refs else "page_grounded_only",
                "quoted_text_sha256": sha256("\n".join(span["exact_quote"] for span in segment).encode()).hexdigest(),
                "answer_eligible": authority_kind == "normative_legal_text" and legal_force == "canonical_normative",
                "citation_eligible": owner.get("citable") is True if owner_type == "evidence_registry" else False,
                "viewer_eligible": bool(bbox_refs) and (
                    owner_type == "page_text_span_review" or owner.get("viewer_highlightable") is not False
                ),
                "highlight_eligible": bool(bbox_refs) and (
                    owner_type == "page_text_span_review" or owner.get("viewer_highlightable") is not False
                ),
                "decision_status": "reviewed",
                "decision_reason": reason,
            })
    return sorted(rows, key=lambda row: row["support_unit_id"])


def _segments(spans: list[dict]) -> list[list[dict]]:
    ordered = sorted(spans, key=lambda row: (row["source_document_id"], row["page_number"], row["text_start"]))
    segments: list[list[dict]] = []
    for span in ordered:
        key = (
            span["source_document_id"], span["source_role"], span["temporal_context"],
            span["semantic_classification"], span["legal_force"], span["page_number"],
        )
        if not segments or key != (
            segments[-1][-1]["source_document_id"], segments[-1][-1]["source_role"],
            segments[-1][-1]["temporal_context"], segments[-1][-1]["semantic_classification"],
            segments[-1][-1]["legal_force"], segments[-1][-1]["page_number"],
        ) or int(span["text_span_id"].rsplit("::", 1)[1]) != int(
            segments[-1][-1]["text_span_id"].rsplit("::", 1)[1]
        ) + 1:
            segments.append([])
        segments[-1].append(span)
    return segments


def _bbox_refs(
    owner_type: str,
    owner: dict,
    raw_rows: list[dict],
    words_by_page: dict[tuple[str, int], list[dict]],
    geometry_by_id: dict[str, dict],
) -> list[str]:
    pages = {(row["source_document_id"], row["page_number"]) for row in raw_rows}
    if owner_type == "source_conflict":
        candidates = list(owner.get("raw_provenance_bbox_ids") or [])
        return [item for item in candidates if _geometry_page(geometry_by_id, item) in pages]
    if owner_type != "page_text_span_review":
        candidates = list(owner.get("bbox_refs") or owner.get("bbox_ids") or [])
        return [item for item in candidates if _geometry_page(geometry_by_id, item) in pages]
    refs = []
    for raw in raw_rows:
        for word in words_by_page[(raw["source_document_id"], raw["page_number"])]:
            if (
                word["x0"] >= raw["x0"] - 0.01 and word["x1"] <= raw["x1"] + 0.01
                and word["y0"] >= raw["y0"] - 0.01 and word["y1"] <= raw["y1"] + 0.01
            ):
                refs.append(word["word_bbox_id"])
    return refs


def _geometry_page(geometry_by_id: dict[str, dict], bbox_id: str) -> tuple[str, int] | None:
    row = geometry_by_id.get(bbox_id)
    return (row["source_document_id"], row["page_number"]) if row else None


def _support_decision(owner_type: str, owner_id: str, authority_kind: str) -> tuple[str, str]:
    if owner_type == "page_text_span_review":
        return REVIEWED_HEADINGS[owner_id]
    if owner_type == "metadata_grounding":
        return "metadata", "existing_exact_metadata_grounding"
    if owner_type == "source_conflict":
        return "trace", "reviewed_printed_numbering_anomaly"
    return {
        "normative_legal_text": "normative",
        "structural_context": "structural",
        "instrument_provenance": "instrument",
    }[authority_kind], "existing_canonical_evidence_owner"
