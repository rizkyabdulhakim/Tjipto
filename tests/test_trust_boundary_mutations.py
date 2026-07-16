from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import unittest

from tjipto.contracts.authority import authority_decision
from tjipto.core.manifest import read_jsonl
from tjipto.corpora.uud.policy.validation import validate_uud_trust_boundary


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "data/final/uud"


class TrustBoundaryMutationTest(unittest.TestCase):
    base: dict[str, list[dict]]

    @classmethod
    def setUpClass(cls) -> None:
        cls.base = {
            "legal_units": read_jsonl(FINAL / "legal_units.jsonl"),
            "chunks": read_jsonl(FINAL / "chunks.jsonl"),
            "graph_nodes": read_jsonl(FINAL / "graph_nodes.jsonl"),
            "graph_edges": read_jsonl(FINAL / "graph_edges.jsonl"),
            "retrieval_units": read_jsonl(FINAL / "retrieval_units.jsonl"),
            "evidence": read_jsonl(FINAL / "evidence_registry.jsonl"),
            "bbox_rows": read_jsonl(FINAL / "bbox_registry.jsonl"),
            "word_bboxes": read_jsonl(FINAL / "word_bboxes.jsonl"),
            "page_text_spans": read_jsonl(FINAL / "page_text_spans.jsonl"),
            "source_documents": read_jsonl(FINAL / "source_documents.jsonl"),
            "pages": read_jsonl(FINAL / "pages.jsonl"),
        }

    def test_baseline_has_no_trust_boundary_violations(self) -> None:
        self.assertEqual(validate_uud_trust_boundary(**self.base), [])

    def test_required_mutations_fail_with_stable_codes(self) -> None:
        cases = (
            ("AUTHORITY_MISSING", self._missing_node_authority),
            ("AUTHORITY_STATE_CONTRADICTION", self._final_graph_node),
            ("RETRIEVAL_TRACE_AUTHORITY_MISSING", self._null_retrieval_trace_authority),
            ("RETRIEVAL_TRACE_FINAL", self._final_retrieval_trace),
            ("COORDINATE_METADATA_MISSING", self._missing_coordinate),
            ("REFERENCE_UNRESOLVED_EVIDENCE", self._unresolved_evidence),
            ("REFERENCE_UNRESOLVED_SOURCE", self._unresolved_source),
            ("REFERENCE_UNRESOLVED_PAGE", self._unresolved_page),
            ("REFERENCE_UNRESOLVED_SPAN", self._unresolved_span),
            ("REFERENCE_UNRESOLVED_BBOX", self._unresolved_bbox),
            ("SIBLING_ORDER_DUPLICATE", self._duplicate_sibling_order),
            ("SIBLING_ORDER_NONCONTIGUOUS", self._noncontiguous_sibling_order),
            ("STABLE_UNIT_ID_MISMATCH", self._wrong_stable_id),
            ("CONTRIBUTING_CHILD_MISSING", self._missing_child),
            ("PARENT_UNRESOLVED", self._missing_parent),
            ("ANCESTOR_PATH_INCORRECT", self._wrong_ancestor_path),
            ("GRAPH_NODE_LEGAL_UNIT_UNRESOLVED", self._node_missing_unit),
            ("GRAPH_NODE_ORPHAN", self._orphan_node),
            ("GRAPH_EDGE_ENDPOINT_UNRESOLVED", self._unresolved_endpoint),
            ("DERIVATION_METHOD_UNKNOWN", self._unknown_derivation),
            ("RELATION_SUPPORT_MISMATCH", self._relation_support_mismatch),
            ("CITABLE_STATUS_CONFLICT", self._citable_status_conflict),
            ("FINALITY_REASON_CONFLICT", self._finality_reason_conflict),
            ("RETRIEVAL_EVIDENCE_UNRESOLVED", self._missing_retrieval_evidence),
            ("CHUNK_EVIDENCE_UNRESOLVED", self._missing_chunk_evidence),
            ("NORMATIVE_SPAN_REJECTED", self._reject_normative_span),
        )
        for expected, mutate in cases:
            with self.subTest(expected=expected):
                artifacts = deepcopy(self.base)
                mutate(artifacts)
                codes = {violation.code for violation in validate_uud_trust_boundary(**artifacts)}
                self.assertIn(expected, codes)

    @staticmethod
    def _missing_node_authority(rows):
        rows["graph_nodes"][0].pop("authority_kind")

    @staticmethod
    def _final_graph_node(rows):
        rows["graph_nodes"][0]["citation_final"] = True

    @staticmethod
    def _null_retrieval_trace_authority(rows):
        rows["retrieval_units"][0]["retrieval_trace"]["authority_kind"] = None

    @staticmethod
    def _final_retrieval_trace(rows):
        rows["retrieval_units"][0]["retrieval_trace"]["citation_final"] = True

    @staticmethod
    def _missing_coordinate(rows):
        next(row for row in rows["bbox_rows"] if row["viewer_highlightable"]).pop("page_width")

    @staticmethod
    def _unresolved_evidence(rows):
        next(row for row in rows["graph_edges"] if row["supporting_evidence_ids"])["supporting_evidence_ids"] = ["missing"]

    @staticmethod
    def _unresolved_source(rows):
        rows["evidence"][0]["source_document_id"] = "missing"

    @staticmethod
    def _unresolved_page(rows):
        rows["evidence"][0]["page_numbers"] = [999]

    @staticmethod
    def _unresolved_span(rows):
        rows["evidence"][0]["text_span_ids"] = ["missing"]

    @staticmethod
    def _unresolved_bbox(rows):
        rows["evidence"][0]["bbox_refs"] = ["missing"]

    @staticmethod
    def _duplicate_sibling_order(rows):
        siblings = _sibling_pair(rows["legal_units"])
        siblings[1]["sibling_order"] = siblings[0]["sibling_order"]

    @staticmethod
    def _noncontiguous_sibling_order(rows):
        _sibling_pair(rows["legal_units"])[1]["sibling_order"] = 999

    @staticmethod
    def _wrong_stable_id(rows):
        rows["legal_units"][0]["stable_unit_id"] = "wrong"

    @staticmethod
    def _missing_child(rows):
        rows["chunks"][0]["contributing_child_legal_unit_ids"] = []

    @staticmethod
    def _missing_parent(rows):
        row = next(row for row in rows["legal_units"] if row.get("parent_legal_unit_id"))
        row["parent_legal_unit_id"] = "missing"

    @staticmethod
    def _wrong_ancestor_path(rows):
        row = next(row for row in rows["legal_units"] if row.get("ancestor_legal_unit_ids"))
        row["ancestor_legal_unit_ids"] = []

    @staticmethod
    def _node_missing_unit(rows):
        next(row for row in rows["graph_nodes"] if row["node_type"] == "legal_unit")["legal_unit_id"] = "missing"

    @staticmethod
    def _orphan_node(rows):
        node = deepcopy(rows["graph_nodes"][0])
        node["node_id"] = "orphan::synthetic"
        node["runtime_loadable"] = True
        node.update(
            authority_decision(
                authority_kind="endpoint_provenance",
                citable=False,
                citation_final=False,
                exactness="not_applicable",
                evidence_available=False,
                reason_code="mutation_fixture",
            )
        )
        rows["graph_nodes"].append(node)

    @staticmethod
    def _unresolved_endpoint(rows):
        rows["graph_edges"][0]["source_id"] = "missing"

    @staticmethod
    def _unknown_derivation(rows):
        rows["graph_edges"][0]["derivation_method"] = "unknown"

    @staticmethod
    def _relation_support_mismatch(rows):
        row = next(row for row in rows["graph_edges"] if row["support_kind"] == "deterministic_structure")
        row["supporting_evidence_ids"] = [rows["evidence"][0]["evidence_id"]]

    @staticmethod
    def _citable_status_conflict(rows):
        rows["evidence"][0]["citable_status"] = "not_citable"

    @staticmethod
    def _finality_reason_conflict(rows):
        rows["evidence"][0]["citation_finality_reason"] = "wrong"

    @staticmethod
    def _missing_retrieval_evidence(rows):
        rows["retrieval_units"][0]["evidence_id"] = "missing"

    @staticmethod
    def _missing_chunk_evidence(rows):
        row = next(row for row in rows["chunks"] if row.get("evidence_ids"))
        row["evidence_ids"] = ["missing"]

    @staticmethod
    def _reject_normative_span(rows):
        row = next(row for row in rows["page_text_spans"] if row.get("citation_final"))
        row.update(
            authority_decision(
                authority_kind="rejected",
                citable=False,
                citation_final=False,
                exactness="rejected",
                evidence_available=False,
                reason_code="mutation_fixture",
            )
        )


def _sibling_pair(units: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str | None], list[dict]] = {}
    for row in units:
        group = groups.setdefault((row["source_document_id"], row.get("parent_legal_unit_id")), [])
        group.append(row)
        if len(group) == 2:
            return group
    raise AssertionError("missing sibling pair")


if __name__ == "__main__":
    unittest.main()
