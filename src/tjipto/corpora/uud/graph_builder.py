from __future__ import annotations

import hashlib
import re

from tjipto.corpora.uud.specs import UUD_INSERTED_BAB_PREDECESSORS, UUD_LEGAL_GRAPH_EDGE_SCHEMA
from tjipto.corpora.parser_dispatch import parse_legal_references


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
    source_by_id = {row["source_document_id"]: row for row in source_documents}
    metadata_by_key = {(row.get("source_role"), row.get("metadata_field")): row for row in metadata_grounding}
    evidenced_pages = {(row["source_document_id"], page_number) for row in evidence for page_number in row.get("page_numbers") or ()}

    def add_node(node_id: str, **payload) -> None:
        if node_id in seen_nodes:
            return
        seen_nodes.add(node_id)
        nodes.append({"node_id": node_id, **payload})

    def add_edge(source_id: str, target_id: str, edge_type: str, **payload) -> None:
        evidence_ids = payload.get("supporting_evidence_ids") or ()
        identity_evidence_id = payload.pop("_identity_evidence_id", None)
        edge_id = _edge_id(edge_type, source_id, target_id, identity_evidence_id or (evidence_ids[0] if evidence_ids else None))
        if edge_id in seen_edges:
            return
        seen_edges.add(edge_id)
        edges.append(
            {
                "edge_id": edge_id,
                "edge_type": edge_type,
                "relation_type": edge_type,
                "source_id": source_id,
                "target_id": target_id,
                **payload,
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
            runtime_loadable=(row["source_document_id"], row["page_number"]) in evidenced_pages,
            orphan_policy=(
                None
                if (row["source_document_id"], row["page_number"]) in evidenced_pages
                else "page_without_retrieval_evidence_excluded_from_runtime_graph"
            ),
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
        runtime_loadable = evidence_row is not None
        for parent_id in row.get("parent_legal_unit_ids") or ():
            parent_node = unit_node_ids.get(parent_id)
            if not parent_node:
                continue
            if _is_false_inserted_bab_parent(row, legal_units_by_id[parent_id]):
                continue
            payload = {
                "source_document_id": row["source_document_id"],
                "_identity_evidence_id": evidence_row["evidence_id"] if evidence_row else None,
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
                        supporting_evidence_ids=[row["evidence_id"]],
                        source_legal_unit_id=row["legal_unit_id"],
                        target_legal_unit_id=target["legal_unit_id"],
                        target_citation=target.get("unit_label"),
                        article_relation_ref=_article_relation_ref(
                            "MODIFIES", row["evidence_id"], target["legal_unit_id"], target.get("unit_label")
                        ),
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
                    supporting_evidence_ids=[delete_clause["evidence_id"]],
                    source_legal_unit_id=delete_clause["legal_unit_id"],
                    target_legal_unit_id=target["legal_unit_id"],
                    target_citation=target.get("unit_label"),
                    article_relation_ref=_article_relation_ref(
                        "DELETES", delete_clause["evidence_id"], target["legal_unit_id"], target.get("unit_label")
                    ),
                    runtime_loadable=True,
                    validation_status="accepted_instrument_clause",
                    confidence_policy="explicit_delete_clause_reference",
                )

    renumber_clause = next((row for row in evidence if row.get("citation") == "Perubahan Keempat Clause (c)"), None)
    if renumber_clause:
        renumberings = (
            ("amendment_3_historical", "Pasal 3", "current_consolidated", "Pasal 3"),
            ("amendment_2_historical", "Pasal 25E", "current_consolidated", "Pasal 25A"),
        )
        source_node = unit_node_ids.get(renumber_clause["legal_unit_id"])
        for source_role, source_label, target_role, target_label in renumberings:
            source_unit = next(
                (row for row in legal_units if row.get("source_role") == source_role and row.get("unit_label") == source_label),
                None,
            )
            target_unit = next(
                (row for row in legal_units if row.get("source_role") == target_role and row.get("unit_label") == target_label),
                None,
            )
            if source_node and source_unit and target_unit:
                add_edge(
                    unit_node_ids[source_unit["legal_unit_id"]],
                    unit_node_ids[target_unit["legal_unit_id"]],
                    "RENAMES",
                    source_document_id=renumber_clause["source_document_id"],
                    supporting_evidence_ids=[renumber_clause["evidence_id"]],
                    source_legal_unit_id=source_unit["legal_unit_id"],
                    target_legal_unit_id=target_unit["legal_unit_id"],
                    target_citation=target_unit.get("unit_label"),
                    article_relation_ref=_article_relation_ref(
                        "RENAMES", renumber_clause["evidence_id"], target_unit["legal_unit_id"], target_unit.get("unit_label")
                    ),
                    runtime_loadable=True,
                    validation_status="accepted_renumbering_clause",
                    confidence_policy="explicit_renumbering_clause_reference",
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
                supporting_evidence_ids=[signatory["evidence_id"]],
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
                supporting_evidence_ids=[decision["evidence_id"]],
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


def _edge_id(edge_type: str, source_id: str, target_id: str, supporting_evidence_id: str | None) -> str:
    digest = hashlib.md5(
        f"{edge_type}|{source_id}|{target_id}|{supporting_evidence_id or ''}".encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()
    return f"edge::{digest}"


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
    labels: list[str] = []
    seen: set[str] = set()
    for row in parse_legal_references("uud", text or ""):
        label = str(row["reference"])
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
