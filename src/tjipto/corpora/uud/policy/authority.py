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
    evidence_by_unit = {row["legal_unit_id"]: row for row in evidence}
    units_by_id = {row["legal_unit_id"]: row for row in units}
    evidence_by_span: dict[str, list[str]] = defaultdict(list)
    bbox_by_span: dict[str, list[str]] = defaultdict(list)
    for row in sorted(evidence, key=lambda item: item["evidence_id"]):
        for span_id in row.get("text_span_ids") or ():
            evidence_by_span[span_id].append(row["evidence_id"])
            bbox_by_span[span_id].extend(row.get("bbox_refs") or ())
    unit_by_id = {row["legal_unit_id"]: row for row in units}
    context_by_span: dict[str, list[str]] = defaultdict(list)
    for unit in units:
        ancestors = unit.get("ancestor_legal_unit_ids") or []
        context_ids = [bbox_id for ancestor_id in ancestors for bbox_id in (unit_by_id.get(ancestor_id, {}).get("bbox_ids") or ())]
        for span_id in unit.get("text_span_ids") or ():
            context_by_span[span_id].extend(context_ids)
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
                "context_bbox_ids": [
                    bbox_id
                    for bbox_id in dict.fromkeys(context_by_span[span["text_span_id"]])
                    if bbox_id not in span_bbox_ids and bbox_id not in evidence_bbox_ids
                ],
                "reason": span.get("exclusion_reason") or ("exact_evidence_unavailable" if not exact else "exact_evidence"),
            }
        )
        span.update(
            _decision(
                "normative_legal_text" if exact else _span_nonfinal_kind(span),
                exact,
                exact,
                span["reason"],
                evidence_exists=bool(evidence_ids),
            )
        )
        if any(field not in span or span[field] is None for field in EVIDENCE_DECISION_FIELDS):
            raise ValueError("incomplete_span_decision")
    for row in evidence:
        exact = row.get("bbox_precision") == "exact" and row.get("viewer_highlightable") is True
        historical_anomaly = bool(units_by_id.get(row.get("legal_unit_id"), {}).get("exclusion_ref"))
        instrument_trace = units_by_id.get(row.get("legal_unit_id"), {}).get("unit_type") in {
            "amendment_recital_record",
            "amendment_scope_record",
            "instrument_clause_record",
            "instrument_closing_record",
            "decision_clause_record",
            "determination_clause_record",
            "signatory_block_record",
        }
        row.update(
            _decision(
                "source_anomaly_trace" if historical_anomaly else ("instrument_provenance" if instrument_trace else ("normative_legal_text" if exact else "page_only")),
                exact and not instrument_trace and not historical_anomaly,
                exact and not instrument_trace and not historical_anomaly,
                "historical_source_anomaly_not_final"
                if historical_anomaly
                else "instrument_trace_only_not_public_citation"
                if instrument_trace
                else ("exact_evidence" if exact else row.get("failure_reason", "not_exact")),
                evidence_exists=True,
                exactness="exact" if exact else "not_applicable",
            )
        )
    for row in bboxes:
        exact = row.get("bbox_precision") == "exact" and row.get("viewer_highlightable") is True
        row.update(_decision("endpoint_provenance" if exact else "page_only", False, False, "bbox_support_only", evidence_exists=False))
    for row in units:
        kind = "normative_legal_text" if row.get("runtime_loadable") is True else _inactive_kind(row)
        has_evidence = bool(row.get("evidence_ids"))
        row.update(
            _decision(
                kind,
                False,
                False,
                "evidence_present_not_citation" if has_evidence else "legal_unit_requires_exact_evidence",
                evidence_exists=has_evidence,
            )
        )
    for row in chunks:
        kind = "normative_legal_text" if row.get("runtime_loadable") is True else _inactive_kind(row)
        has_evidence = bool(row.get("evidence_ids"))
        row.update(
            _decision(
                kind,
                False,
                False,
                "evidence_present_not_citation" if has_evidence else "chunk_requires_exact_evidence",
                evidence_exists=has_evidence,
            )
        )
    for row in nodes:
        node_kind = NODE_AUTHORITY.get(str(row.get("node_type")))
        if node_kind is None:
            raise ValueError(f"unknown_uud_graph_node:{row.get('node_type')}")
        if row.get("node_type") == "final_evidence":
            evidence_id = row.get("evidence_id") or str(row.get("node_id", "")).removeprefix("final_evidence::")
            linked = evidence_by_id.get(evidence_id, {})
            if linked.get("bbox_precision") != "exact" or linked.get("viewer_highlightable") is not True:
                node_kind = "page_only"
            has_evidence = bool(linked)
            row.update(
                _decision(
                    node_kind,
                    False,
                    False,
                    "evidence_present_not_citation" if has_evidence else "graph_node_not_citation",
                    evidence_exists=has_evidence,
                )
            )
            continue
        has_evidence = bool(
            row.get("final_evidence_id")
            or row.get("evidence_ids")
            or (row.get("node_type") == "legal_unit" and row.get("legal_unit_id") in evidence_by_unit)
        )
        row.update(
            _decision(
                node_kind,
                False,
                False,
                "evidence_present_not_citation" if has_evidence else "graph_node_not_citation",
                evidence_exists=has_evidence,
            )
        )
    for row in edges:
        if row.get("relation_id"):
            continue
        edge_kind = EDGE_AUTHORITY.get(str(row.get("support_kind")))
        if edge_kind is None:
            raise ValueError(f"unknown_uud_graph_support:{row.get('support_kind')}")
        has_evidence = bool(row.get("supporting_evidence_ids"))
        row.update(
            _decision(
                edge_kind,
                False,
                False,
                "evidence_present_not_citation" if has_evidence else "graph_edge_not_citation",
                evidence_exists=has_evidence,
            )
        )


def _decision(
    kind: str,
    citable: bool,
    final: bool,
    reason: str,
    *,
    evidence_exists: bool = False,
    exactness: str | None = None,
) -> dict:
    return authority_decision(
        authority_kind=kind,
        citable=citable,
        citation_final=final,
        exactness=exactness or ("exact" if citable else "not_applicable"),
        evidence_exists=evidence_exists,
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
