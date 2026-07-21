from __future__ import annotations

from collections import defaultdict

from tjipto.contracts.authority import authority_decision
from tjipto.contracts.evidence import EVIDENCE_DECISION_FIELDS
from tjipto.corpora.uud.span_disposition_policy import role_for_legal_unit


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
    for row in sorted(evidence, key=lambda item: item["evidence_id"]):
        for span_id in row.get("text_span_ids") or ():
            evidence_by_span[span_id].append(row["evidence_id"])
    unit_by_id = {row["legal_unit_id"]: row for row in units}
    bbox_by_id = {row["bbox_id"]: row for row in bboxes}
    context_by_span: dict[str, list[str]] = defaultdict(list)
    for unit in units:
        ancestors = unit.get("ancestor_legal_unit_ids") or []
        context_ids = [bbox_id for ancestor_id in ancestors for bbox_id in (unit_by_id.get(ancestor_id, {}).get("bbox_ids") or ())]
        for span_id in unit.get("text_span_ids") or ():
            span = next((item for item in spans if item.get("text_span_id") == span_id), None)
            context_by_span[span_id].extend(
                bbox_id
                for bbox_id in context_ids
                if bbox_by_id.get(bbox_id, {}).get("source_document_id") == (span or {}).get("source_document_id")
                and bbox_by_id.get(bbox_id, {}).get("page_number") == (span or {}).get("page_number")
            )
    for span in spans:
        evidence_ids = list(dict.fromkeys(evidence_by_span[span["text_span_id"]]))
        span_bbox_ids = list(dict.fromkeys(span.get("span_bbox_ids") or ()))
        marker = str(span.get("text") or "").strip() in {"*)", "**)", "***)", "****)"}
        exact = not marker and span.get("promotion_status") == "promoted_legal_unit" and bool(evidence_ids and span_bbox_ids)
        for field in tuple(span):
            if field.startswith("exposure_") or field in {"target_evidence_ids", "target_bbox_ids", "word_bbox_ids"}:
                span.pop(field)
        span.update(
            {
                "object_role": "source_span",
                "linked_authority": (
                    "normative_legal_text"
                    if span.get("semantic_classification") == "normative_constitutional_text"
                    and span.get("linked_authority") == "rejected"
                    else span.get("authority_kind") or _span_nonfinal_kind(span)
                ),
                "classification": "footnote_marker" if marker else span.get("semantic_classification") or "unclassified_source_text",
                "legal_force": "nonlegal" if marker else span.get("legal_force") or "nonlegal",
                "viewer_highlightable": exact,
                "evidence_ids": evidence_ids,
                "span_bbox_ids": span_bbox_ids,
                "reason": "footnote_marker" if marker else span.get("exclusion_reason") or ("exact_evidence_unavailable" if not exact else "exact_evidence"),
            }
        )
        if marker:
            span["linked_authority"] = "nonlegal"
        span.update(
            _decision(
                "normative_legal_text" if exact else _span_nonfinal_kind(span),
                exact,
                exact,
                span["reason"],
                evidence_exists=bool(evidence_ids),
            )
        )
        for field in ("authority_kind", "citable", "citable_status", "citation_final", "citation_finality_reason", "exactness", "evidence_exists"):
            span.pop(field, None)
        if any(field not in span or span[field] is None for field in EVIDENCE_DECISION_FIELDS):
            missing = set(EVIDENCE_DECISION_FIELDS) - set(span)
            if missing - {"authority_kind", "citable", "citation_final", "exactness", "evidence_exists"}:
                raise ValueError("incomplete_span_decision")
    for row in evidence:
        exact = row.get("bbox_precision") == "exact" and row.get("viewer_highlightable") is True
        structural_provenance = row.get("evidence_owner_kind") == "metadata_source" or role_for_legal_unit(
            units_by_id.get(row.get("legal_unit_id"), {})
        ) == "structural_heading"
        historical_anomaly = bool(units_by_id.get(row.get("legal_unit_id"), {}).get("exclusion_ref"))
        historical_pasali = row.get("hierarchy") == ["ATURAN TAMBAHAN", "Pasal I"] and row.get("source_role") == "amendment_4_historical"
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
                "structural_context"
                if structural_provenance
                else "source_anomaly_trace"
                if historical_anomaly
                else ("instrument_provenance" if instrument_trace else ("normative_legal_text" if exact else "page_only")),
                exact and not structural_provenance and not instrument_trace and not historical_anomaly,
                exact and not historical_pasali and not structural_provenance and not instrument_trace and not historical_anomaly,
                "historical_source_anomaly_not_final"
                if historical_anomaly
                else "historical_exact_nonfinal"
                if historical_pasali
                else "structural_source_provenance_not_final"
                if structural_provenance
                else "instrument_trace_only_not_public_citation"
                if instrument_trace
                else ("exact_evidence" if exact else row.get("failure_reason", "not_exact")),
                evidence_exists=True,
                exactness="exact" if exact else "not_applicable",
            )
        )
        row.update({"object_role": "evidence", "linked_authority": row.get("authority_kind"), "is_citation_object": bool(row.get("citable"))})
        row.update(
            {
                "temporal_role": row.get("source_role"),
                "citation_eligibility": "eligible" if row.get("citable") is True else "ineligible",
                "relevant_quote_eligible": row.get("citable") is True and row.get("authority_kind") == "normative_legal_text" and row.get("evidence_owner_kind") == "legal_unit_source",
                "support_kind": "direct_evidence" if row.get("text_span_ids") else "page_support",
            }
        )
    for row in bboxes:
        row.update(
            {
                "object_role": "geometry",
                "evidence_exists": bool(row.get("evidence_id")),
                "reason_code": "bbox_support_only",
            }
        )
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
        row["object_role"] = "graph_projection"
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
    # Projection rows are intentionally authority-free in schema 6.  Their
    # linked evidence/relation IDs are the only route back to the owner.
    for rows in (units, chunks, nodes):
        for row in rows:
            for field in (
                "authority_kind",
                "citable_status",
                "citable",
                "citation_final",
                "citation_finality_reason",
                "exactness",
                "evidence_exists",
            ):
                row.pop(field, None)
    for row in edges:
        relation_id = row.get("relation_id")
        support_ids = list(dict.fromkeys(row.get("support_evidence_ids") or row.get("supporting_evidence_ids") or ()))
        if relation_id and not row.get("support_relation_ids"):
            row["support_relation_ids"] = [relation_id]
        row.setdefault("support_relation_ids", [])
        row["support_evidence_ids"] = support_ids
        row["support_exception_ids"] = list(row.get("support_exception_ids") or ())
        row["support_kind"] = row.get("support_kind") or "relation_reference"
        row["object_role"] = "graph_projection"
        for field in (
            "supporting_evidence_ids",
            "bbox_refs",
            "text_span_ids",
            "source_document_ids",
            "page_numbers",
            "citation_available",
            "authority_kind",
            "citable_status",
            "citable",
            "citation_final",
            "citation_finality_reason",
            "exactness",
            "evidence_exists",
        ):
            row.pop(field, None)


def _decision(
    kind: str,
    citable: bool,
    final: bool,
    reason: str,
    *,
    evidence_exists: bool = False,
    exactness: str | None = None,
) -> dict:
    decision = authority_decision(
        authority_kind=kind,
        citable=citable,
        citation_final=final,
        exactness=exactness or ("exact" if citable else "not_applicable"),
        evidence_exists=evidence_exists,
        reason_code=reason,
    )
    decision["artifact_status"] = "published" if evidence_exists else "rejected"
    decision.pop("status", None)
    return decision


def _span_nonfinal_kind(span: dict) -> str:
    if span.get("semantic_classification") == "normative_constitutional_text":
        return "normative_legal_text"
    if span.get("promotion_status") == "excluded_nonlegal":
        return "nonlegal"
    if span.get("promotion_status") == "promoted_metadata":
        return "metadata"
    if span.get("promotion_status") == "promoted_source_conflict":
        return "source_anomaly_trace"
    if span.get("legal_force") == "amendment_instrument" or span.get("semantic_classification") == "amendment_instrument_text":
        return "instrument_provenance"
    return "structural_context" if span.get("promotion_status") == "excluded_structural" else "rejected"


def _inactive_kind(row: dict) -> str:
    if row.get("exclusion_ref") or row.get("status") in {"inactive_source_typo_reference", "excluded"}:
        return "rejected"
    return "structural_context"
