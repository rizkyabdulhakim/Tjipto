from __future__ import annotations


def build_document_relations(source_documents: list[dict]) -> list[dict]:
    source_by_role = {row["source_role"]: row for row in source_documents}
    original = source_by_role["original_historical"]
    rows = []
    for role in sorted(role for role in source_by_role if role.startswith("amendment_")):
        source = source_by_role[role]
        rows.append(_document_relation("AMENDS", source, original))
        rows.append(_document_relation("AMENDED_BY", original, source, support_role=role))
    return sorted(rows, key=lambda row: row["relation_id"])


def build_article_amendment_relations(
    *,
    graph_edges: list[dict],
    legal_units: list[dict],
    evidence: list[dict],
    bbox_rows: list[dict],
) -> list[dict]:
    units = {row["legal_unit_id"]: row for row in legal_units}
    evidence_by_id = {row["evidence_id"]: row for row in evidence}
    bbox_ids = {row["bbox_id"] for row in bbox_rows}
    rows = []
    for edge in graph_edges:
        relation_type = edge.get("edge_type")
        if relation_type not in {"MODIFIES", "DELETES"}:
            continue
        supporting_ids = edge.get("supporting_evidence_ids") or ()
        evidence_row = evidence_by_id.get(supporting_ids[0]) if supporting_ids else None
        source_unit_id = _legal_unit_id(edge.get("source_id"))
        target_unit_id = _legal_unit_id(edge.get("target_id"))
        target = units.get(target_unit_id or "")
        target_citation = target.get("unit_label") if target else None
        if not str(target_citation or "").startswith(("Pasal ", "Ayat ")):
            continue
        if not evidence_row or not target or not all(ref in bbox_ids for ref in evidence_row.get("bbox_refs") or ()):
            continue
        bbox_refs = list(evidence_row.get("bbox_refs") or ())
        instrument_unit = units.get(evidence_row.get("legal_unit_id"), {}).get("unit_type") in {
            "amendment_recital_record",
            "amendment_scope_record",
            "instrument_clause_record",
            "instrument_closing_record",
            "decision_clause_record",
            "determination_clause_record",
            "signatory_block_record",
        }
        exact_support = (
            evidence_row.get("bbox_precision") == "exact"
            and evidence_row.get("viewer_highlightable") is True
            and (not instrument_unit or target_citation == "Pasal 16")
            and all(ref in bbox_ids for ref in bbox_refs)
        )
        support_class = "exact_article_relation" if exact_support else "trace_article_relation"
        trace_only_reason = None if exact_support else evidence_row.get("failure_reason") or "blocked_by_missing_exact_bbox"
        rows.append(
            {
                "relation_id": f"uud_article_amendment_relation::{relation_type.lower()}::{evidence_row['evidence_id']}::{target_unit_id}",
                "corpus_id": "uud",
                "source_document_id": evidence_row["source_document_id"],
                "source_role": evidence_row["source_role"],
                "relation_type": relation_type,
                "target_legal_unit_id": target_unit_id,
                "target_citation": target_citation,
                "source_legal_unit_id": source_unit_id,
                "evidence_id": evidence_row["evidence_id"],
                "bbox_refs": bbox_refs,
                "page_number": (evidence_row.get("page_numbers") or [None])[0],
                "quoted_text": evidence_row.get("quoted_text"),
                "source_pdf_sha256": evidence_row.get("source_sha256"),
                "grounding_level": "exact_source_text" if exact_support else "page_grounded_trace",
                "support_class": support_class,
                "bbox_precision": evidence_row.get("bbox_precision"),
                "viewer_highlightable": exact_support,
                "citation_available": exact_support,
                "trace_only_reason": trace_only_reason,
                "runtime_loadable": True,
                "validator_status": "valid",
            }
        )
    return sorted(rows, key=lambda row: row["relation_id"])


def _document_relation(relation_type: str, source: dict, target: dict, *, support_role: str | None = None) -> dict:
    source_role = source["source_role"]
    target_role = target["source_role"]
    support = support_role or source_role
    return {
        "relation_id": f"uud_document_relation::{source_role.lower()}::{relation_type.lower()}::{target_role.lower()}",
        "corpus_id": "uud",
        "relation_type": relation_type,
        "source_document_id": source["source_document_id"],
        "source_role": source_role,
        "target_document_id": target["source_document_id"],
        "target_source_role": target_role,
        "support_type": "source_role_grounded",
        "support_refs": [_support_ref(relation_type, source_role, target_role, support)],
        "runtime_loadable": True,
        "viewer_highlightable": False,
        "citation_available": False,
        "article_level": False,
        "reason": "source_role_document_level_relation_without_pasal_ayat_evidence_bbox",
    }


def _legal_unit_id(node_id: object) -> str | None:
    text = str(node_id or "")
    return text.split("legal_unit::", 1)[1] if text.startswith("legal_unit::") else None


def _support_ref(relation_type: str, source_role: str, target_role: str, support_role: str) -> str:
    if relation_type == "AMENDS":
        return f"uud_legal_graph_edge::{source_role}::amends_original_record"
    return f"uud_legal_graph_edge::{source_role}::amended_by_{target_role if target_role.startswith('amendment_') else support_role}_record"
