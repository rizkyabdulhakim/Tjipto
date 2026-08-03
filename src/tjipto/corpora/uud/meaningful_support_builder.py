from __future__ import annotations

from collections import defaultdict
from hashlib import sha256

from tjipto.ingestion.pdf.words import align_text_to_word_bboxes


REVIEW_CONTRACT_ID = "tjipto.uud.meaningful-support-review"
REVIEW_CONTRACT_VERSION = 1


def build_meaningful_support_units(
    *,
    page_text_spans: list[dict],
    raw_source_spans: list[dict],
    evidence: list[dict],
    metadata_grounding: list[dict],
    source_conflicts: list[dict],
    bbox_registry: list[dict],
    word_bboxes: list[dict],
    review_decisions: dict,
) -> list[dict]:
    """Project canonical owners and reviewed exclusions without becoming authority."""
    reviews = _review_decisions(review_decisions)
    spans = {
        row["text_span_id"]: row
        for row in page_text_spans
        if row["legal_force"] != "nonlegal" or row["text_span_id"] in reviews
    }
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
    geometry_by_id = {
        str(row.get("bbox_id") or row.get("word_bbox_id")): row
        for row in (*bbox_registry, *word_bboxes)
        if row.get("bbox_id") or row.get("word_bbox_id")
    }
    canonical_owners = _canonical_owners(evidence, metadata_grounding, source_conflicts)

    assignments: dict[tuple[str, str], list[dict]] = defaultdict(list)
    owners: dict[tuple[str, str], dict] = {}
    for span in spans.values():
        span_id = span["text_span_id"]
        if span_id in reviews:
            review = reviews[span_id]
            for field in (
                "source_document_id", "source_role", "temporal_context", "semantic_classification", "legal_force"
            ):
                if review[field] != span[field]:
                    raise ValueError(f"review decision {review['review_decision_id']} mismatches {field}")
            key = ("review_decision", review["review_decision_id"])
            owners[key] = review
        else:
            key, owner = _select_canonical_owner(span, canonical_owners)
            owners[key] = owner
        assignments[key].append(span)

    rows: list[dict] = []
    for key, owned_spans in assignments.items():
        owner_type, owner_id = key
        owner = owners[key]
        for segment in _segments(owned_spans):
            raw_rows = [
                raw_by_selector[(span["source_document_id"], span["page_number"], span["text_start"], span["text_end"])]
                for span in segment
            ]
            review_exclusion = owner_type == "review_decision" and owner["decision_kind"] == "typed_exclusion"
            bbox_refs = [] if review_exclusion else _segment_bbox_refs(
                segment, raw_rows, words_by_page, geometry_by_id
            )
            classification = segment[0]["semantic_classification"]
            legal_force = segment[0]["legal_force"]
            authority_kind = owner.get("authority_kind") or "structural_context"
            support_kind, reason = _support_decision(owner_type, owner, authority_kind)
            citation_final = owner.get("citation_final") is True if owner_type != "review_decision" else False
            exact_overlay = bool(bbox_refs) and owner.get("viewer_highlightable") is not False
            answer_eligible = (
                not review_exclusion
                and authority_kind == "normative_legal_text"
                and legal_force == "canonical_normative"
            )
            citation_eligible = (
                not review_exclusion
                and owner_type == "evidence_registry"
                and owner.get("citable") is True
            )
            row_key = sha256("\n".join(span["text_span_id"] for span in segment).encode()).hexdigest()[:16]
            rows.append({
                "support_unit_id": f"uud_meaningful_support::{owner_type}::{row_key}",
                "decision_kind": owner.get("decision_kind", "canonical_owner_support"),
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
                "bbox_precision": "not_applicable" if review_exclusion else "exact" if bbox_refs else "page_grounded_only",
                "quoted_text_sha256": sha256("\n".join(span["exact_quote"] for span in segment).encode()).hexdigest(),
                "answer_eligible": answer_eligible,
                "citation_eligible": citation_eligible,
                "viewer_eligible": False if review_exclusion else True,
                "highlight_eligible": False if review_exclusion else exact_overlay,
                "decision_status": "reviewed",
                "decision_reason": reason,
            })
    return sorted(rows, key=lambda row: row["support_unit_id"])


def _review_decisions(value: dict) -> dict[str, dict]:
    if value.get("contract_id") != REVIEW_CONTRACT_ID or value.get("contract_version") != REVIEW_CONTRACT_VERSION:
        raise ValueError("unsupported meaningful-support review contract")
    required = {
        "review_decision_id", "text_span_id", "decision_kind", "support_kind", "source_document_id",
        "source_role", "temporal_context", "semantic_classification", "legal_force", "authority_kind",
        "decision_reason",
    }
    decisions: dict[str, dict] = {}
    for row in value.get("decisions") or ():
        if required - row.keys() or row["decision_kind"] not in {"reviewed_support", "typed_exclusion"}:
            raise ValueError(f"invalid meaningful-support review decision: {row}")
        span_id = row["text_span_id"]
        if span_id in decisions:
            raise ValueError(f"duplicate meaningful-support review decision: {span_id}")
        decisions[span_id] = row
    return decisions


def _canonical_owners(evidence: list[dict], metadata: list[dict], conflicts: list[dict]) -> list[tuple[str, str, dict]]:
    return [
        *(("evidence_registry", row["evidence_id"], row) for row in evidence),
        *(("metadata_grounding", row["metadata_grounding_id"], row) for row in metadata),
        *(("source_conflict", row["source_conflict_id"], row) for row in conflicts),
    ]


def _select_canonical_owner(span: dict, owners: list[tuple[str, str, dict]]) -> tuple[tuple[str, str], dict]:
    candidates = [
        (owner_type, owner_id, owner)
        for owner_type, owner_id, owner in owners
        if span["text_span_id"] in (owner.get("text_span_ids") or ()) and _owner_compatible(span, owner_type, owner)
    ]
    if not candidates:
        raise ValueError(f"meaningful span has no compatible canonical owner: {span['text_span_id']}")
    owner_type, owner_id, owner = min(
        candidates,
        key=lambda item: (len(item[2].get("text_span_ids") or ()), item[1], item[0]),
    )
    return (owner_type, owner_id), owner


def _owner_compatible(span: dict, owner_type: str, owner: dict) -> bool:
    if owner.get("source_document_id") != span["source_document_id"]:
        return False
    owner_role = owner.get("source_role") or (owner.get("source_anomaly_policy") or {}).get("source_role")
    owner_temporal = owner.get("temporal_context") or owner_role
    if owner_role != span["source_role"] or owner_temporal != span["temporal_context"]:
        return False
    authority = str(owner.get("authority_kind") or "")
    if owner_type != "source_conflict" and authority != span.get("linked_authority"):
        return False
    compatible_forces = {
        "normative_legal_text": {"canonical_normative", "historical_normative"},
        "structural_context": {"canonical_normative", "historical_normative", "metadata_only"},
        "instrument_provenance": {"amendment_instrument"},
        "metadata": {"metadata_only"},
        "source_anomaly_provenance": {"historical_normative", "amendment_instrument", "metadata_only"},
    }
    return span["legal_force"] in compatible_forces.get(authority, set())


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


def _segment_bbox_refs(
    spans: list[dict],
    raw_rows: list[dict],
    words_by_page: dict[tuple[str, int], list[dict]],
    geometry_by_id: dict[str, dict],
) -> list[str]:
    refs: list[str] = []
    for span, raw in zip(spans, raw_rows):
        span_refs = list(dict.fromkeys(span.get("span_bbox_ids") or ()))
        if span_refs and all(_geometry_matches_span(geometry_by_id.get(ref), span) for ref in span_refs):
            refs.extend(span_refs)
            continue
        match = align_text_to_word_bboxes(
            text=span.get("exact_quote"),
            source_document_id=span["source_document_id"],
            page_numbers=[span["page_number"]],
            words_by_page=words_by_page,
            reference_bbox=raw,
        )
        fallback = list(match.get("matched_word_bbox_ids") or ()) if match else []
        if not fallback or not all(_geometry_inside_raw(geometry_by_id.get(ref), raw) for ref in fallback):
            return []
        refs.extend(fallback)
    return list(dict.fromkeys(refs))


def _geometry_matches_span(geometry: dict | None, span: dict) -> bool:
    return bool(
        geometry
        and geometry.get("source_document_id") == span["source_document_id"]
        and geometry.get("page_number") == span["page_number"]
    )


def _geometry_inside_raw(geometry: dict | None, raw: dict) -> bool:
    return bool(
        geometry
        and geometry.get("source_document_id") == raw["source_document_id"]
        and geometry.get("page_number") == raw["page_number"]
        and geometry["x0"] >= raw["x0"] - 0.01
        and geometry["x1"] <= raw["x1"] + 0.01
        and geometry["y0"] >= raw["y0"] - 0.01
        and geometry["y1"] <= raw["y1"] + 0.01
    )


def _support_decision(owner_type: str, owner: dict, authority_kind: str) -> tuple[str, str]:
    if owner_type == "review_decision":
        return owner["support_kind"], owner["decision_reason"]
    if owner_type == "metadata_grounding":
        return "metadata", "most_specific_canonical_metadata_owner"
    if owner_type == "source_conflict":
        return "trace", "reviewed_source_conflict_owner"
    return {
        "normative_legal_text": "normative",
        "structural_context": "structural",
        "instrument_provenance": "instrument",
    }[authority_kind], "most_specific_canonical_evidence_owner"
