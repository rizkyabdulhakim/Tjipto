from __future__ import annotations

import hashlib
import re

from tjipto.corpora.uud.specs import UUD_INSERTED_BAB_PREDECESSORS, UUD_LEGAL_GRAPH_EDGE_SCHEMA


def build_graph_artifacts(
    *,
    source_documents: list[dict],
    pages: list[dict],
    legal_units: list[dict],
    evidence: list[dict],
    bbox_rows: list[dict],
    excluded_records: list[dict],
    source_conflicts: list[dict],
    metadata_grounding: list[dict],
) -> tuple[list[dict], list[dict]]:
    nodes: list[dict] = []
    edges: list[dict] = []
    seen_nodes: set[str] = set()
    seen_edges: set[str] = set()
    unit_node_ids: dict[str, str] = {}
    bab_nodes_by_source_label: dict[tuple[str, str], str] = {}
    legal_units_by_id = {row["legal_unit_id"]: row for row in legal_units}
    evidence_by_unit = {row["legal_unit_id"]: row for row in evidence}
    evidence_by_id = {row["evidence_id"]: row for row in evidence}
    source_by_id = {row["source_document_id"]: row for row in source_documents}
    metadata_by_key = {(row.get("source_role"), row.get("metadata_field")): row for row in metadata_grounding}

    def add_node(node_id: str, **payload) -> None:
        if node_id in seen_nodes:
            return
        seen_nodes.add(node_id)
        nodes.append({"node_id": node_id, **payload})

    def add_edge(source_id: str, target_id: str, edge_type: str, **payload) -> None:
        edge_id = _edge_id(edge_type, source_id, target_id, payload.get("evidence_ref"))
        if edge_id in seen_edges:
            return
        seen_edges.add(edge_id)
        authority = _edge_authority_payload(edge_type, payload, evidence_by_id, source_by_id)
        edges.append(
            {
                "edge_id": edge_id,
                "edge_type": edge_type,
                "relation_type": edge_type,
                "source_id": source_id,
                "target_id": target_id,
                **payload,
                **authority,
            }
        )

    for row in source_documents:
        source_role = row["source_role"]
        add_node(
            f"source_role::{source_role}",
            node_type="source_role",
            source_role=source_role,
        )
        add_node(
            f"source_pdf::{row['sha256'][:16]}",
            node_type="source_pdf",
            source_document_id=row["source_document_id"],
            source_pdf=row["filename"],
            source_pdf_path=row["path"],
            source_role=source_role,
            source_sha256=row["sha256"],
        )

    for row in pages:
        source_meta = source_by_id[row["source_document_id"]]
        add_node(
            f"page::{source_meta['sha256'][:16]}::{row['page_number']:04d}",
            node_type="page",
            page_number=row["page_number"],
            source_document_id=row["source_document_id"],
            source_pdf=source_meta["filename"],
            source_pdf_path=source_meta["path"],
            source_role=source_meta["source_role"],
            source_sha256=source_meta["sha256"],
        )

    for row in legal_units:
        source_role = row.get("source_role") or source_by_id[row["source_document_id"]]["source_role"]
        node_id = f"legal_unit::{row['legal_unit_id']}"
        unit_node_ids[row["legal_unit_id"]] = node_id
        if row.get("unit_type") == "bab_record" and row.get("unit_label"):
            bab_nodes_by_source_label[(row["source_document_id"], row["unit_label"])] = node_id
        add_node(
            node_id,
            node_type="legal_unit",
            legal_unit_id=row["legal_unit_id"],
            source_document_id=row["source_document_id"],
            source_role=source_role,
            unit_label=row.get("unit_label"),
            unit_type=row.get("unit_type"),
            hierarchy_path=row.get("hierarchy") or ([row["unit_label"]] if row.get("unit_label") else []),
            runtime_loadable=row.get("runtime_loadable") is not False,
        )

    for row in evidence:
        add_node(
            f"final_evidence::{row['evidence_id']}",
            node_type="final_evidence",
            final_evidence_id=row["evidence_id"],
            citation=row.get("citation"),
            source_document_id=row["source_document_id"],
            source_role=row.get("source_role"),
            source_sha256=row.get("source_sha256"),
            viewer_highlightable=row.get("viewer_highlightable"),
            bbox_precision=row.get("bbox_precision"),
        )

    for row in bbox_rows:
        source_meta = source_by_id[row["source_document_id"]]
        add_node(
            f"bbox::{row['bbox_id']}",
            node_type="bbox",
            bbox_status="final_accepted_bbox",
            bbox_id=row["bbox_id"],
            final_evidence_id=row["evidence_id"],
            page_number=row["page_number"],
            rectangle_index=_numeric_suffix(row["bbox_id"]),
            source_document_id=row["source_document_id"],
            source_pdf=row["source_pdf"],
            source_pdf_path=row["source_pdf_path"],
            source_role=source_meta["source_role"],
            source_sha256=row["source_sha256"],
        )

    for row in excluded_records:
        add_node(
            f"excluded_record::{row['legacy_chunk_id']}",
            node_type="excluded_record",
            excluded_reason=row["reason"],
            excluded_status=row["status"],
            runtime_loadable=False,
            source_role=row["source_role"],
        )

    for row in source_conflicts:
        source_meta = source_by_id[row["source_document_id"]]
        add_node(
            f"source_conflict::{row['source_conflict_id']}",
            node_type="source_conflict",
            source_conflict_id=row["source_conflict_id"],
            source_document_id=row["source_document_id"],
            source_role=source_meta["source_role"],
            conflict_type=row["type"],
            classification=row["classification"],
            runtime_loadable=False,
            status=row["status"],
        )

    for row in evidence:
        evidence_node = f"final_evidence::{row['evidence_id']}"
        source_meta = source_by_id[row["source_document_id"]]
        add_edge(unit_node_ids[row["legal_unit_id"]], evidence_node, "HAS_FINAL_EVIDENCE")
        add_edge(evidence_node, f"source_role::{row['source_role']}", "BELONGS_TO_SOURCE_ROLE")
        add_edge(evidence_node, f"source_pdf::{source_meta['sha256'][:16]}", "USES_SOURCE_PDF")
        for page_number in row.get("page_numbers") or ():
            add_edge(
                evidence_node,
                f"page::{source_meta['sha256'][:16]}::{page_number:04d}",
                "PAGE_GROUNDED_AT",
            )
        for bbox_id in row.get("bbox_refs") or ():
            add_edge(evidence_node, f"bbox::{bbox_id}", "HAS_BBOX")

    for row in excluded_records:
        add_edge(
            f"excluded_record::{row['legacy_chunk_id']}",
            f"source_role::{row['source_role']}",
            "EXCLUDED_BECAUSE",
        )

    for row in legal_units:
        child_node = unit_node_ids[row["legal_unit_id"]]
        evidence_row = evidence_by_unit.get(row["legal_unit_id"])
        evidence_ref = evidence_row["evidence_id"] if evidence_row else None
        runtime_loadable = evidence_ref is not None
        for parent_id in row.get("parent_legal_unit_ids") or ():
            parent_node = unit_node_ids.get(parent_id)
            if not parent_node:
                continue
            if _is_false_inserted_bab_parent(row, legal_units_by_id[parent_id]):
                continue
            payload = {
                "source_document_id": row["source_document_id"],
                "evidence_ref": evidence_ref,
                "runtime_loadable": runtime_loadable,
                "validation_status": "accepted_structural_hierarchy",
                "confidence_policy": "legal_unit_parent_child_artifact",
            }
            add_edge(parent_node, child_node, "CONTAINS", **payload)
            add_edge(child_node, parent_node, "PART_OF", **payload)

    for (source_document_id, inserted_label), inserted_node in bab_nodes_by_source_label.items():
        source_meta = source_by_id[source_document_id]
        sequence_payload = {
            "source_document_id": source_document_id,
            "source_role": source_meta["source_role"],
            "temporal_context": source_meta.get("temporal_context"),
        }
        for predecessor_label in UUD_INSERTED_BAB_PREDECESSORS.get(inserted_label, ()):
            predecessor_node = bab_nodes_by_source_label.get((source_document_id, predecessor_label))
            if not predecessor_node:
                continue
            _add_structural_sequence_edge(add_edge, predecessor_node, inserted_node, "PRECEDES", sequence_payload)
            _add_structural_sequence_edge(add_edge, inserted_node, predecessor_node, "FOLLOWS", sequence_payload)
            _add_structural_sequence_edge(add_edge, inserted_node, predecessor_node, "INSERTED_AFTER", sequence_payload)
            break

    for row in evidence:
        if row.get("citation", "").endswith(" Scope"):
            source_node = unit_node_ids.get(row["legal_unit_id"])
            for label in _scope_target_labels(row.get("quoted_text")):
                target = _resolve_legal_unit_by_label(
                    legal_units,
                    row["source_document_id"],
                    label,
                )
                if source_node and target:
                    add_edge(
                        source_node,
                        unit_node_ids[target["legal_unit_id"]],
                        "MODIFIES",
                        source_document_id=row["source_document_id"],
                        evidence_ref=row["evidence_id"],
                        source_legal_unit_id=row["legal_unit_id"],
                        target_legal_unit_id=target["legal_unit_id"],
                        target_citation=target.get("unit_label"),
                        article_relation_ref=_article_relation_ref("MODIFIES", row["evidence_id"], target["legal_unit_id"], target.get("unit_label")),
                        runtime_loadable=True,
                        validation_status="accepted_instrument_scope",
                        confidence_policy="explicit_scope_article_reference",
                    )

    delete_clause = next((row for row in evidence if row.get("citation") == "Perubahan Keempat Clause (d)"), None)
    if delete_clause:
        source_node = unit_node_ids[delete_clause["legal_unit_id"]]
        for label in ("BAB IV", "Pasal 16"):
            target = _resolve_legal_unit_by_label(legal_units, delete_clause["source_document_id"], label)
            if target:
                add_edge(
                    source_node,
                    unit_node_ids[target["legal_unit_id"]],
                    "DELETES",
                    source_document_id=delete_clause["source_document_id"],
                    evidence_ref=delete_clause["evidence_id"],
                    source_legal_unit_id=delete_clause["legal_unit_id"],
                    target_legal_unit_id=target["legal_unit_id"],
                    target_citation=target.get("unit_label"),
                    article_relation_ref=_article_relation_ref("DELETES", delete_clause["evidence_id"], target["legal_unit_id"], target.get("unit_label")),
                    runtime_loadable=True,
                    validation_status="accepted_instrument_clause",
                    confidence_policy="explicit_delete_clause_reference",
                )

    for ordinal in ("Pertama", "Kedua", "Ketiga", "Keempat"):
        role = _source_role_for_ordinal(ordinal)
        signatory = next((row for row in evidence if row.get("citation") == f"Perubahan {ordinal} Signatories"), None)
        if signatory:
            add_edge(
                f"source_role::{role}",
                unit_node_ids[signatory["legal_unit_id"]],
                "HAS_SIGNATORY",
                source_document_id=signatory["source_document_id"],
                evidence_ref=signatory["evidence_id"],
                runtime_loadable=True,
                validation_status="accepted_signatory_block",
                confidence_policy="explicit_signatory_block_evidence",
            )
        decision = next((row for row in evidence if row.get("citation") == f"Perubahan {ordinal} Decision"), None)
        if decision:
            add_edge(
                f"source_role::{role}",
                unit_node_ids[decision["legal_unit_id"]],
                "HAS_DECISION_SESSION",
                source_document_id=decision["source_document_id"],
                evidence_ref=decision["evidence_id"],
                runtime_loadable=True,
                validation_status="accepted_decision_clause",
                confidence_policy="explicit_decision_clause_evidence",
            )
        effective_unit = next((row for row in legal_units if row.get("unit_label") == f"Perubahan {ordinal} Effective"), None)
        effective_grounding = metadata_by_key.get((role, "effective_rule"))
        if effective_unit and effective_grounding:
            add_edge(
                f"source_role::{role}",
                unit_node_ids[effective_unit["legal_unit_id"]],
                "HAS_EFFECTIVE_RULE",
                source_document_id=effective_unit["source_document_id"],
                provenance_ref=effective_grounding["metadata_grounding_id"],
                provenance_ref_kind="metadata_grounding",
                provenance_support="page_grounded",
                runtime_loadable=False,
                validation_status="grounded_metadata_only",
                confidence_policy="field_level_metadata_grounding",
            )

    for row in source_conflicts:
        role = source_by_id[row["source_document_id"]]["source_role"]
        add_edge(
            f"source_role::{role}",
            f"source_conflict::{row['source_conflict_id']}",
            "HAS_SOURCE_ANOMALY",
            source_document_id=row["source_document_id"],
            provenance_ref=row["source_conflict_id"],
            provenance_ref_kind="source_conflict",
            provenance_support="trace_only",
            runtime_loadable=False,
            validation_status="recorded_source_conflict",
            confidence_policy="source_conflict_artifact_only",
        )

    nodes.sort(key=lambda row: (row["node_type"], row["node_id"]))
    edges.sort(key=lambda row: (row["edge_type"], row["source_id"], row["target_id"], row["edge_id"]))
    return nodes, edges


def _edge_id(edge_type: str, source_id: str, target_id: str, evidence_ref: str | None) -> str:
    digest = hashlib.md5(
        f"{edge_type}|{source_id}|{target_id}|{evidence_ref or ''}".encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()
    return f"edge::{digest}"


def _edge_authority_payload(edge_type: str, payload: dict, evidence_by_id: dict[str, dict], source_by_id: dict[str, dict]) -> dict:
    evidence = evidence_by_id.get(str(payload.get("evidence_ref") or ""))
    exact = bool(evidence and evidence.get("bbox_precision") == "exact" and evidence.get("viewer_highlightable") is True)
    trace = bool(evidence and not exact)
    source_document_id = payload.get("source_document_id") or (evidence or {}).get("source_document_id")
    source_role = _source_role_class(source_by_id.get(str(source_document_id or ""), {}).get("source_role"))
    if edge_type in {"MODIFIES", "DELETES"}:
        return {
            "edge_authority_level": "evidence_backed_relation" if exact else "trace",
            "graph_finality_policy": "evidence_backed_relation" if exact else "trace_only_relation",
            "citation_final": False,
            "viewer_highlightable": exact,
            "evidence_requirement": "exact_bbox" if exact else "trace_only",
            "source_role": source_role,
            "relation_support": "exact" if exact else "trace_only",
            "reason": "exact_article_relation_evidence_bbox" if exact else (evidence or {}).get("failure_reason", "trace_only_relation_not_citable"),
            "bbox_refs": list((evidence or {}).get("bbox_refs") or ()),
        }
    if edge_type in {"HAS_EFFECTIVE_RULE"}:
        support = "metadata"
        authority = "provenance"
        finality = "metadata_provenance"
        requirement = "page_grounded"
    elif edge_type in {"HAS_SOURCE_ANOMALY"}:
        support = "source_anomaly"
        authority = "provenance"
        finality = "source_anomaly_provenance"
        requirement = "trace_only"
    elif edge_type in {"CONTAINS", "PART_OF", "PRECEDES", "FOLLOWS", "INSERTED_AFTER"}:
        support = "structural"
        authority = "evidence_backed_relation" if evidence else "trace"
        finality = "structural_relation"
        requirement = "exact_bbox" if exact else ("page_grounded" if trace else "none")
    elif edge_type in {"HAS_SIGNATORY", "HAS_DECISION_SESSION"}:
        support = "exact" if exact else "trace_only"
        authority = "evidence_backed_relation" if exact else "trace"
        finality = "instrument_provenance"
        requirement = "exact_bbox" if exact else "trace_only"
    elif edge_type == "EXCLUDED_BECAUSE":
        support = "structural"
        authority = "nonlegal"
        finality = "nonlegal"
        requirement = "none"
    else:
        support = "structural"
        authority = "provenance"
        finality = "instrument_provenance"
        requirement = "exact_bbox" if exact else ("page_grounded" if trace else "none")
    return {
        "edge_authority_level": authority,
        "graph_finality_policy": finality,
        "citation_final": False,
        "viewer_highlightable": False,
        "evidence_requirement": requirement,
        "source_role": source_role,
        "relation_support": support,
        "reason": payload.get("confidence_policy") or payload.get("validation_status") or "graph_provenance_edge",
        "bbox_refs": list((evidence or {}).get("bbox_refs") or ()),
    }


def _source_role_class(source_role: str | None) -> str:
    if source_role == "current_consolidated":
        return "consolidated"
    if source_role == "original_historical":
        return "historical"
    if str(source_role or "").startswith("amendment_"):
        return "amendment"
    return "canonical"


def _article_relation_ref(relation_type: str, evidence_id: str, target_unit_id: str, target_citation: str | None) -> str | None:
    if not str(target_citation or "").startswith(("Pasal ", "Ayat ")):
        return None
    return f"uud_article_amendment_relation::{relation_type.lower()}::{evidence_id}::{target_unit_id}"


def _is_false_inserted_bab_parent(child: dict, parent: dict) -> bool:
    if child.get("unit_type") != "bab_record" or parent.get("unit_type") != "bab_record":
        return False
    if child.get("source_document_id") != parent.get("source_document_id"):
        return False
    return parent.get("unit_label") in UUD_INSERTED_BAB_PREDECESSORS.get(child.get("unit_label"), ())


def _add_structural_sequence_edge(add_edge, source_id: str, target_id: str, edge_type: str, payload: dict) -> None:
    schema = UUD_LEGAL_GRAPH_EDGE_SCHEMA[edge_type]
    add_edge(
        source_id,
        target_id,
        edge_type,
        **payload,
        runtime_loadable=schema["runtime_loadable"],
        validation_status=schema["validation_status"],
        confidence_policy=schema["confidence_policy"],
        derivation_basis=schema["derivation_basis"],
    )


def _numeric_suffix(value: str) -> int:
    match = re.search(r"(\d+)$", value)
    return int(match.group(1)) if match else 0


def _scope_target_labels(text: str | None) -> list[str]:
    labels = []
    seen = set()
    for match in re.finditer(r"\bPasal\s+\d+[A-Z]?\b", text or ""):
        label = match.group(0)
        if label not in seen:
            seen.add(label)
            labels.append(label)
    return labels


def _resolve_legal_unit_by_label(legal_units: list[dict], source_document_id: str, label: str) -> dict | None:
    return next(
        (row for row in legal_units if row["source_document_id"] == source_document_id and row.get("unit_label") == label),
        None,
    )


def _source_role_for_ordinal(ordinal: str) -> str:
    return {
        "Pertama": "amendment_1_historical",
        "Kedua": "amendment_2_historical",
        "Ketiga": "amendment_3_historical",
        "Keempat": "amendment_4_historical",
    }[ordinal]
