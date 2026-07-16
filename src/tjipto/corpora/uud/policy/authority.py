from __future__ import annotations

from collections import defaultdict

from tjipto.contracts.authority import authority_decision
from tjipto.contracts.evidence import EVIDENCE_DECISION_FIELDS


NODE_AUTHORITY = {
    "bbox": "endpoint_provenance",
    "final_evidence": "normative_legal_text",
    "legal_unit": "normative_legal_text",
    "page": "endpoint_provenance",
    "source_pdf": "endpoint_provenance",
    "source_role": "metadata",
    "excluded_record": "nonlegal",
    "source_conflict": "source_anomaly_trace",
}
EDGE_AUTHORITY = {
    "exact_source_relation": "exact_relation_support",
    "deterministic_structure": "deterministic_structure",
    "endpoint_provenance": "endpoint_provenance",
    "instrument_provenance": "instrument_provenance",
    "historical_mapping": "historical_mapping",
    "source_anomaly_trace": "source_anomaly_trace",
    "nonlegal": "nonlegal",
}


def apply_authority_contract(
    *,
    spans: list[dict],
    evidence: list[dict],
    bboxes: list[dict],
    units: list[dict],
    chunks: list[dict],
    nodes: list[dict],
    edges: list[dict],
) -> None:
    evidence_by_id = {row["evidence_id"]: row for row in evidence}
    evidence_by_span: dict[str, list[str]] = defaultdict(list)
    bbox_by_span: dict[str, list[str]] = defaultdict(list)
    for row in sorted(evidence, key=lambda item: item["evidence_id"]):
        for span_id in row.get("text_span_ids") or ():
            evidence_by_span[span_id].append(row["evidence_id"])
            bbox_by_span[span_id].extend(row.get("bbox_refs") or ())
    for span in spans:
        evidence_ids = list(dict.fromkeys(evidence_by_span[span["text_span_id"]]))
        evidence_bbox_ids = list(dict.fromkeys(bbox_by_span[span["text_span_id"]]))
        span_bbox_ids = list(dict.fromkeys(span.get("span_bbox_ids") or ()))
        exact = span.get("promotion_status") == "promoted_legal_unit" and bool(evidence_ids and span_bbox_ids)
        for field in tuple(span):
            if field.startswith("exposure_") or field in {"target_evidence_ids", "target_bbox_ids", "word_bbox_ids"}:
                span.pop(field)
        span.update(
            {
                "classification": span.get("semantic_classification") or "unclassified_source_text",
                "legal_force": span.get("legal_force") or "nonlegal",
                "highlightable": exact,
                "evidence_ids": evidence_ids,
                "span_bbox_ids": span_bbox_ids,
                "evidence_bbox_ids": evidence_bbox_ids,
                "context_bbox_ids": [],
                "reason": span.get("exclusion_reason") or ("exact_evidence_unavailable" if not exact else "exact_evidence"),
            }
        )
        span.update(_decision("normative_legal_text" if exact else _span_nonfinal_kind(span), exact, exact, span["reason"]))
        if any(field not in span or span[field] is None for field in EVIDENCE_DECISION_FIELDS):
            raise ValueError("incomplete_span_decision")
    for row in evidence:
        exact = row.get("bbox_precision") == "exact" and row.get("viewer_highlightable") is True
        row.update(
            _decision(
                "normative_legal_text" if exact else "page_only",
                exact,
                exact,
                "exact_evidence" if exact else row.get("failure_reason", "not_exact"),
            )
        )
    for row in bboxes:
        exact = row.get("bbox_precision") == "exact" and row.get("viewer_highlightable") is True
        row.update(_decision("endpoint_provenance" if exact else "page_only", False, False, "bbox_support_only"))
    for row in units:
        kind = "normative_legal_text" if row.get("runtime_loadable") is True else _inactive_kind(row)
        row.update(_decision(kind, False, False, "legal_unit_requires_exact_evidence"))
    for row in chunks:
        kind = "normative_legal_text" if row.get("runtime_loadable") is True else _inactive_kind(row)
        row.update(_decision(kind, False, False, "chunk_requires_exact_evidence"))
    for row in nodes:
        node_kind = NODE_AUTHORITY.get(str(row.get("node_type")))
        if node_kind is None:
            raise ValueError(f"unknown_uud_graph_node:{row.get('node_type')}")
        if row.get("node_type") == "final_evidence":
            evidence_id = row.get("evidence_id") or str(row.get("node_id", "")).removeprefix("final_evidence::")
            linked = evidence_by_id.get(evidence_id, {})
            if linked.get("bbox_precision") != "exact" or linked.get("viewer_highlightable") is not True:
                node_kind = "page_only"
        row.update(_decision(node_kind, False, False, "graph_node_not_citation"))
    for row in edges:
        edge_kind = EDGE_AUTHORITY.get(str(row.get("support_kind")))
        if edge_kind is None:
            raise ValueError(f"unknown_uud_graph_support:{row.get('support_kind')}")
        row.update(_decision(edge_kind, False, False, "graph_edge_not_citation"))


def _decision(kind: str, citable: bool, final: bool, reason: str) -> dict:
    return authority_decision(
        authority_kind=kind,
        citable=citable,
        citation_final=final,
        exactness="exact" if citable else "not_applicable",
        evidence_exists=citable,
        reason_code=reason,
    )


def _span_nonfinal_kind(span: dict) -> str:
    if span.get("promotion_status") == "excluded_nonlegal":
        return "nonlegal"
    if span.get("promotion_status") == "promoted_metadata":
        return "metadata"
    if span.get("promotion_status") == "promoted_source_conflict":
        return "source_anomaly_trace"
    return "structural_context" if span.get("promotion_status") == "excluded_structural" else "rejected"


def _inactive_kind(row: dict) -> str:
    if row.get("exclusion_ref") or row.get("status") in {"inactive_source_typo_reference", "excluded"}:
        return "rejected"
    return "structural_context"
