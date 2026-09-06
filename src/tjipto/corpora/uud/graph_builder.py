from __future__ import annotations

import hashlib
import re

from tjipto.corpora.parser_dispatch import parse_legal_references
from tjipto.corpora.uud.policy.relations import is_deletion_provision, is_renumbering_provision, is_scope_provision
from tjipto.corpora.uud.relation_builder import (
    classify_scope_operation,
    legal_unit_reference,
    parse_renumbering_mappings,
    predecessor_unit_for_reference,
    resolve_relation_unit,
)
from tjipto.corpora.uud.specs import UUD_INSERTED_BAB_PREDECESSORS, UUD_LEGAL_GRAPH_EDGE_SCHEMA
from tjipto.corpora.uud.structure_builder import numeric_suffix


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
    return _GraphArtifactBuilder(
        source_documents=source_documents,
        pages=pages,
        legal_units=legal_units,
        evidence=evidence,
        bbox_rows=bbox_rows,
        excluded_records=excluded_records,
        source_conflicts=source_conflicts,
        metadata_grounding=metadata_grounding,
    ).build()


class _GraphArtifactBuilder:
    """Build the UUD graph while keeping node and edge ownership local."""

    def __init__(
        self,
        *,
        source_documents: list[dict],
        pages: list[dict],
        legal_units: list[dict],
        evidence: list[dict],
        bbox_rows: list[dict],
        excluded_records: list[dict],
        source_conflicts: list[dict],
        metadata_grounding: list[dict],
    ) -> None:
        self.source_documents = source_documents
        self.pages = pages
        self.legal_units = legal_units
        self.evidence = evidence
        self.bbox_rows = bbox_rows
        self.excluded_records = excluded_records
        self.source_conflicts = source_conflicts
        self.nodes: list[dict] = []
        self.edges: list[dict] = []
        self.seen_nodes: set[str] = set()
        self.seen_edges: set[str] = set()
        self.unit_node_ids: dict[str, str] = {}
        self.bab_nodes_by_source_label: dict[tuple[str, str], str] = {}
        self.legal_units_by_id = {row["legal_unit_id"]: row for row in self.legal_units}
        self.evidence_by_unit = {row["legal_unit_id"]: row for row in self.evidence}
        self.source_by_id = {row["source_document_id"]: row for row in self.source_documents}
        self.metadata_by_key = {(row.get("source_role"), row.get("metadata_field")): row for row in metadata_grounding}
        self.evidenced_pages = {
            (row["source_document_id"], page_number) for row in self.evidence for page_number in row.get("page_numbers") or ()
        }

    def build(self) -> tuple[list[dict], list[dict]]:
        self._add_document_nodes()
        self._add_content_nodes()
        self._add_evidence_edges()
        self._add_structure_edges()
        self._add_scope_edges()
        self._add_deletion_edge()
        self._add_renumbering_edges()
        self._add_document_metadata_edges()
        self._add_conflict_edges()
        self.nodes.sort(key=lambda row: (row["node_type"], row["node_id"]))
        self.edges.sort(key=lambda row: (row["edge_type"], row["source_id"], row["target_id"], row["edge_id"]))
        return self.nodes, self.edges

    def add_node(self, node_id: str, **payload) -> None:
        if node_id in self.seen_nodes:
            return
        self.seen_nodes.add(node_id)
        self.nodes.append({"node_id": node_id, **payload})

    def add_edge(self, source_id: str, target_id: str, edge_type: str, **payload) -> None:
        evidence_ids = payload.get("supporting_evidence_ids") or ()
        identity = payload.pop("_identity_evidence_id", None)
        edge_id = _edge_id(edge_type, source_id, target_id, identity or (evidence_ids[0] if evidence_ids else None))
        if edge_id in self.seen_edges:
            return
        self.seen_edges.add(edge_id)
        self.edges.append(
            {
                "edge_id": edge_id,
                "edge_type": edge_type,
                "relation_type": edge_type,
                "source_id": source_id,
                "target_id": target_id,
                **payload,
            }
        )

    def _add_document_nodes(self) -> None:
        for row in self.source_documents:
            role = row["source_role"]
            self.add_node(f"source_role::{role}", node_type="source_role", source_role=role)
            self.add_node(
                f"source_pdf::{row['sha256'][:16]}",
                node_type="source_pdf",
                source_document_id=row["source_document_id"],
                source_pdf=row["filename"],
                source_pdf_path=row["path"],
                source_role=role,
                source_sha256=row["sha256"],
            )
        for row in self.pages:
            source = self.source_by_id[row["source_document_id"]]
            evidenced = (row["source_document_id"], row["page_number"]) in self.evidenced_pages
            self.add_node(
                f"page::{source['sha256'][:16]}::{row['page_number']:04d}",
                node_type="page",
                page_number=row["page_number"],
                source_document_id=row["source_document_id"],
                source_pdf=source["filename"],
                source_pdf_path=source["path"],
                source_role=source["source_role"],
                source_sha256=source["sha256"],
                runtime_loadable=evidenced,
                orphan_policy=None if evidenced else "page_without_retrieval_evidence_excluded_from_runtime_graph",
            )

    def _add_content_nodes(self) -> None:
        for row in self.legal_units:
            role = row.get("source_role") or self.source_by_id[row["source_document_id"]]["source_role"]
            node_id = f"legal_unit::{row['legal_unit_id']}"
            self.unit_node_ids[row["legal_unit_id"]] = node_id
            if row.get("unit_type") == "bab_record" and row.get("unit_label"):
                self.bab_nodes_by_source_label[(row["source_document_id"], row["unit_label"])] = node_id
            self.add_node(
                node_id,
                node_type="legal_unit",
                legal_unit_id=row["legal_unit_id"],
                source_document_id=row["source_document_id"],
                source_role=role,
                unit_label=row.get("unit_label"),
                unit_type=row.get("unit_type"),
                hierarchy_path=row.get("hierarchy") or ([row["unit_label"]] if row.get("unit_label") else []),
                runtime_loadable=row.get("runtime_loadable") is not False,
            )
        for row in self.evidence:
            self.add_node(
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
        for row in self.bbox_rows:
            source = self.source_by_id[row["source_document_id"]]
            self.add_node(
                f"bbox::{row['bbox_id']}",
                node_type="bbox",
                bbox_status="final_accepted_bbox",
                bbox_id=row["bbox_id"],
                page_number=row["page_number"],
                rectangle_index=numeric_suffix(row["bbox_id"]),
                source_document_id=row["source_document_id"],
                source_pdf=row["source_pdf"],
                source_pdf_path=row["source_pdf_path"],
                source_role=source["source_role"],
                source_sha256=row["source_sha256"],
            )
        for row in self.excluded_records:
            self.add_node(
                f"excluded_record::{row['legacy_chunk_id']}",
                node_type="excluded_record",
                excluded_reason=row["reason"],
                excluded_status=row["status"],
                runtime_loadable=False,
                source_role=row["source_role"],
            )
        for row in self.source_conflicts:
            source = self.source_by_id[row["source_document_id"]]
            self.add_node(
                f"source_conflict::{row['source_conflict_id']}",
                node_type="source_conflict",
                source_conflict_id=row["source_conflict_id"],
                source_document_id=row["source_document_id"],
                source_role=source["source_role"],
                conflict_type=row["type"],
                classification=row["classification"],
                runtime_loadable=False,
                status=row["status"],
            )

    def _add_evidence_edges(self) -> None:
        for row in self.evidence:
            evidence_node = f"final_evidence::{row['evidence_id']}"
            source = self.source_by_id[row["source_document_id"]]
            self.add_edge(self.unit_node_ids[row["legal_unit_id"]], evidence_node, "HAS_FINAL_EVIDENCE")
            self.add_edge(evidence_node, f"source_role::{row['source_role']}", "BELONGS_TO_SOURCE_ROLE")
            self.add_edge(evidence_node, f"source_pdf::{source['sha256'][:16]}", "USES_SOURCE_PDF")
            for page in row.get("page_numbers") or ():
                self.add_edge(evidence_node, f"page::{source['sha256'][:16]}::{page:04d}", "PAGE_GROUNDED_AT")
            for bbox_id in row.get("bbox_refs") or ():
                self.add_edge(evidence_node, f"bbox::{bbox_id}", "HAS_BBOX")
        for row in self.excluded_records:
            self.add_edge(f"excluded_record::{row['legacy_chunk_id']}", f"source_role::{row['source_role']}", "EXCLUDED_BECAUSE")

    def _add_structure_edges(self) -> None:
        for row in self.legal_units:
            child = self.unit_node_ids[row["legal_unit_id"]]
            evidence = self.evidence_by_unit.get(row["legal_unit_id"])
            for parent_id in row.get("parent_legal_unit_ids") or ():
                parent = self.legal_units_by_id.get(parent_id)
                parent_node = self.unit_node_ids.get(parent_id)
                if not parent_node or not parent or _is_false_inserted_bab_parent(row, parent):
                    continue
                payload = {
                    "source_document_id": row["source_document_id"],
                    "_identity_evidence_id": evidence["evidence_id"] if evidence else None,
                    "runtime_loadable": evidence is not None,
                    "validation_status": "accepted_structural_hierarchy",
                    "confidence_policy": "legal_unit_parent_child_artifact",
                }
                self.add_edge(parent_node, child, "CONTAINS", **payload)
                self.add_edge(child, parent_node, "PART_OF", **payload)
        for (source_id, inserted), inserted_node in self.bab_nodes_by_source_label.items():
            source = self.source_by_id[source_id]
            payload = {
                "source_document_id": source_id,
                "source_role": source["source_role"],
                "temporal_context": source.get("temporal_context"),
            }
            for predecessor in UUD_INSERTED_BAB_PREDECESSORS.get(inserted, ()):
                predecessor_node = self.bab_nodes_by_source_label.get((source_id, predecessor))
                if not predecessor_node:
                    continue
                for edge_type, source_node, target_node in (
                    ("PRECEDES", predecessor_node, inserted_node),
                    ("FOLLOWS", inserted_node, predecessor_node),
                    ("INSERTED_AFTER", inserted_node, predecessor_node),
                ):
                    _add_structural_sequence_edge(self.add_edge, source_node, target_node, edge_type, payload)
                break

    def _add_scope_edges(self) -> None:
        for row in self.evidence:
            if not is_scope_provision(row):
                continue
            source_node = self.unit_node_ids.get(row["legal_unit_id"])
            for label, relation_type, candidates in _scope_target_operations(row.get("quoted_text")):
                target, relation_type, candidates, anomaly = self._scope_target(row, label, relation_type, candidates)
                if not source_node or not target:
                    continue
                target_id = target["legal_unit_id"]
                payload = {
                    "source_document_id": row["source_document_id"],
                    "supporting_evidence_ids": [row["evidence_id"]],
                    "source_legal_unit_id": row["legal_unit_id"],
                    "target_legal_unit_id": target_id,
                    "target_citation": label,
                    "article_relation_ref": _article_relation_ref(relation_type, row["evidence_id"], target_id, label),
                    "operation_candidates": candidates,
                    "runtime_loadable": True,
                    "validation_status": "accepted_instrument_scope",
                    "confidence_policy": "explicit_scope_article_reference",
                }
                if anomaly:
                    payload.update(
                        provenance_ref=anomaly["source_conflict_id"], provenance_ref_kind="source_conflict", provenance_support="trace_only"
                    )
                self.add_edge(source_node, self.unit_node_ids[target_id], relation_type, **payload)

    def _scope_target(self, row: dict, label: str, relation_type: str, candidates: tuple[str, ...]):
        target = resolve_relation_unit(self.legal_units, label, source_document_id=row["source_document_id"])
        target = target or resolve_relation_unit(self.legal_units, label, source_role="current_consolidated")
        if relation_type in {"ADDS", "MODIFIES", "AMBIGUOUS_OPERATION"}:
            comparison = classify_scope_operation(self.legal_units, row["source_role"], label, canonical_target=target)
            relation_type = str(comparison.get("relation_type") or relation_type)
            candidates = tuple(comparison.get("operation_candidates") or ())
            target = self.legal_units_by_id.get(comparison.get("successor_legal_unit_id"), target)
        printed = legal_unit_reference(target, self.legal_units_by_id) if target else ""
        anomaly = _numbering_anomaly(self.source_conflicts, row["source_document_id"], printed, label)
        if printed != label and anomaly is None:
            relation_type, candidates = "AMBIGUOUS_OPERATION", ("MODIFIES", "ADDS")
        return target, relation_type, candidates, anomaly

    def _add_deletion_edge(self) -> None:
        clause = next((row for row in self.evidence if is_deletion_provision(row)), None)
        if not clause:
            return
        source_node = self.unit_node_ids[clause["legal_unit_id"]]
        for label in _deletion_target_labels(clause.get("quoted_text")):
            target = predecessor_unit_for_reference(self.legal_units, label, clause["source_role"])
            target = target or resolve_relation_unit(self.legal_units, label, source_document_id=clause["source_document_id"])
            if not target or not str(target.get("unit_label") or "").startswith(("Pasal ", "Ayat ")):
                continue
            target_id = target["legal_unit_id"]
            citation = legal_unit_reference(target, self.legal_units_by_id)
            self.add_edge(
                source_node,
                self.unit_node_ids[target_id],
                "DELETES",
                source_document_id=clause["source_document_id"],
                supporting_evidence_ids=[clause["evidence_id"]],
                source_legal_unit_id=clause["legal_unit_id"],
                target_legal_unit_id=target_id,
                target_citation=citation,
                article_relation_ref=_article_relation_ref("DELETES", clause["evidence_id"], target_id, citation),
                runtime_loadable=True,
                validation_status="accepted_instrument_clause",
                confidence_policy="explicit_delete_clause_reference",
            )

    def _add_renumbering_edges(self) -> None:
        clause = next((row for row in self.evidence if is_renumbering_provision(row)), None)
        if not clause:
            return
        for mapping in parse_renumbering_mappings(str(clause.get("quoted_text") or "")):
            source = resolve_relation_unit(self.legal_units, str(mapping["old_reference"]), source_role=mapping["source_role"])
            target = resolve_relation_unit(self.legal_units, str(mapping["new_reference"]), source_role="current_consolidated")
            if not source or not target:
                continue
            mapping_key = f"{mapping['old_reference']}->{mapping['new_reference']}"
            relation_type = (
                "RENUMBERED_TO"
                if mapping["old_reference"].startswith("Pasal 25E") and mapping["new_reference"].startswith("Pasal 25A")
                else "RENAMES"
            )
            target_id = target["legal_unit_id"]
            self.add_edge(
                self.unit_node_ids[source["legal_unit_id"]],
                self.unit_node_ids[target_id],
                relation_type,
                source_document_id=clause["source_document_id"],
                supporting_evidence_ids=[clause["evidence_id"]],
                source_legal_unit_id=source["legal_unit_id"],
                target_legal_unit_id=target_id,
                target_citation=mapping["new_reference"],
                source_legal_unit_role=source.get("source_role"),
                reference_mapping=mapping,
                article_relation_ref=_article_relation_ref(
                    relation_type, clause["evidence_id"], target_id, mapping["new_reference"], mapping_key
                ),
                _identity_evidence_id=f"{clause['evidence_id']}::{mapping_key}",
                runtime_loadable=True,
                validation_status="accepted_renumbering_clause",
                confidence_policy="explicit_renumbering_clause_reference",
            )

    def _add_document_metadata_edges(self) -> None:
        for source in self.source_documents:
            source_id, role = source["source_document_id"], source["source_role"]
            signatory = _event_evidence(self.evidence, source_id, "Signatories")
            decision = _event_evidence(self.evidence, source_id, "Decision")
            prefix = _event_prefix(signatory or decision)
            if not prefix:
                continue
            if signatory:
                self.add_edge(
                    f"source_role::{role}",
                    self.unit_node_ids[signatory["legal_unit_id"]],
                    "HAS_SIGNATORY",
                    source_document_id=source_id,
                    supporting_evidence_ids=[signatory["evidence_id"]],
                    runtime_loadable=True,
                    validation_status="accepted_signatory_block",
                    confidence_policy="explicit_signatory_block_evidence",
                )
            if decision:
                self.add_edge(
                    f"source_role::{role}",
                    self.unit_node_ids[decision["legal_unit_id"]],
                    "HAS_DECISION_SESSION",
                    source_document_id=source_id,
                    supporting_evidence_ids=[decision["evidence_id"]],
                    runtime_loadable=True,
                    validation_status="accepted_decision_clause",
                    confidence_policy="explicit_decision_clause_evidence",
                )
            effective = next(
                (
                    row
                    for row in self.legal_units
                    if row.get("source_document_id") == source_id and row.get("unit_label") == f"{prefix} Effective"
                ),
                None,
            )
            grounding = self.metadata_by_key.get((role, "effective_rule"))
            if effective and grounding:
                self.add_edge(
                    f"source_role::{role}",
                    self.unit_node_ids[effective["legal_unit_id"]],
                    "HAS_EFFECTIVE_RULE",
                    source_document_id=source_id,
                    provenance_ref=grounding["metadata_grounding_id"],
                    provenance_ref_kind="metadata_grounding",
                    provenance_support="page_grounded",
                    runtime_loadable=False,
                    validation_status="grounded_metadata_only",
                    confidence_policy="field_level_metadata_grounding",
                )

    def _add_conflict_edges(self) -> None:
        for row in self.source_conflicts:
            role = self.source_by_id[row["source_document_id"]]["source_role"]
            self.add_edge(
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


def _edge_id(edge_type: str, source_id: str, target_id: str, supporting_evidence_id: str | None) -> str:
    digest = hashlib.md5(
        f"{edge_type}|{source_id}|{target_id}|{supporting_evidence_id or ''}".encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()
    return f"edge::{digest}"


def _article_relation_ref(
    relation_type: str,
    evidence_id: str,
    target_unit_id: str,
    target_citation: str | None,
    mapping_key: str | None = None,
) -> str | None:
    if not str(target_citation or "").startswith(("Pasal ", "Ayat ")):
        return None
    suffix = f"::{mapping_key}" if mapping_key else ""
    return f"uud_article_amendment_relation::{relation_type.lower()}::{evidence_id}::{target_unit_id}{suffix}"


def _numbering_anomaly(
    source_conflicts: list[dict],
    source_document_id: str,
    printed_label: str,
    canonical_label: str,
) -> dict | None:
    if not printed_label or printed_label == canonical_label:
        return None
    return next(
        (
            row
            for row in source_conflicts
            if row.get("source_document_id") == source_document_id
            and row.get("printed_label") == printed_label
            and row.get("canonical_label") == canonical_label
        ),
        None,
    )


def _is_false_inserted_bab_parent(child: dict, parent: dict) -> bool:
    return (
        child.get("unit_type") == parent.get("unit_type") == "bab_record"
        and child.get("source_document_id") == parent.get("source_document_id")
        and parent.get("unit_label") in UUD_INSERTED_BAB_PREDECESSORS.get(child.get("unit_label"), ())
    )


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


def _event_evidence(evidence: list[dict], source_document_id: str, suffix: str) -> dict | None:
    return next(
        (
            row
            for row in evidence
            if row.get("source_document_id") == source_document_id and str(row.get("citation") or "").endswith(f" {suffix}")
        ),
        None,
    )


def _event_prefix(row: dict | None) -> str | None:
    if not row:
        return None
    citation = str(row.get("citation") or "")
    return citation.rsplit(" ", 1)[0] if citation else None


def _scope_target_operations(text: str | None) -> list[tuple[str, str, tuple[str, ...]]]:
    source = str(text or "")
    segments = {
        marker: segment for marker, segment in re.findall(r"\(([a-z])\)\s*(.*?)(?=\s*\([a-z]\)\s|$)", source, re.IGNORECASE | re.DOTALL)
    }
    relation_type = "MODIFIES"
    candidates: tuple[str, ...] = ()
    if segments:
        source = segments.get("e") or segments.get("a") or source
        if "e" in segments and _has_ambiguous_operation(source):
            relation_type, candidates = "AMBIGUOUS_OPERATION", ("MODIFIES", "ADDS")
    else:
        if _has_ambiguous_operation(source):
            relation_type, candidates = "AMBIGUOUS_OPERATION", ("MODIFIES", "ADDS")
        operations = tuple(re.finditer(r"\bmengubah\b", source, re.IGNORECASE))
        if operations:
            source = source[operations[-1].end() :]
    labels = []
    seen: set[str] = set()
    for row in parse_legal_references("uud", source):
        label = str(row["reference"])
        if label not in seen:
            seen.add(label)
            labels.append(label)
    return [(label, relation_type, candidates) for label in labels]


def _has_ambiguous_operation(text: str) -> bool:
    return bool(re.search(r"(?:mengubah|pengubahan)\s+dan/atau\s+(?:menambah|penambahan)", text, re.IGNORECASE))


def _deletion_target_labels(text: str | None) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for row in parse_legal_references("uud", str(text or "")):
        label = str(row.get("reference") or "")
        if label and label not in seen:
            seen.add(label)
            labels.append(label)
    return labels
