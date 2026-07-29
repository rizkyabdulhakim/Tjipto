from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import unittest

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
            "word_bboxes": [],
            "page_text_spans": read_jsonl(FINAL / "page_text_spans.jsonl"),
            "source_documents": read_jsonl(FINAL / "source_documents.jsonl"),
            "pages": read_jsonl(FINAL / "pages.jsonl"),
        }

    def test_baseline_has_no_trust_boundary_violations(self) -> None:
        self.assertEqual(validate_uud_trust_boundary(**self.base), [])

    def test_required_mutations_fail_with_stable_codes(self) -> None:
        cases = (
            ("AUTHORITY_MISSING", self._missing_evidence_authority),
            ("AUTHORITY_STATE_CONTRADICTION", self._invalid_evidence_finality),
            ("COORDINATE_METADATA_MISSING", self._missing_coordinate),
            ("REFERENCE_UNRESOLVED_EVIDENCE", self._unresolved_evidence),
            ("REFERENCE_UNRESOLVED_SOURCE", self._unresolved_source),
            ("REFERENCE_UNRESOLVED_PAGE", self._unresolved_page),
            ("REFERENCE_UNRESOLVED_SPAN", self._unresolved_span),
            ("REFERENCE_UNRESOLVED_BBOX", self._unresolved_bbox),
            ("GRAPH_EDGE_ENDPOINT_UNRESOLVED", self._unresolved_endpoint),
            ("RELATION_SUPPORT_MISMATCH", self._relation_support_mismatch),
            ("RETRIEVAL_EVIDENCE_UNRESOLVED", self._missing_retrieval_evidence),
            ("CHUNK_EVIDENCE_UNRESOLVED", self._missing_chunk_evidence),
        )
        for expected, mutate in cases:
            with self.subTest(expected=expected):
                artifacts = deepcopy(self.base)
                mutate(artifacts)
                codes = {violation.code for violation in validate_uud_trust_boundary(**artifacts)}
                self.assertIn(expected, codes)

    @staticmethod
    def _missing_evidence_authority(rows):
        rows["evidence"][0].pop("authority_kind")

    @staticmethod
    def _invalid_evidence_finality(rows):
        rows["evidence"][0]["citation_final"] = "yes"

    @staticmethod
    def _missing_coordinate(rows):
        next(row for row in rows["bbox_rows"] if row["viewer_highlightable"]).pop("page_width")

    @staticmethod
    def _unresolved_evidence(rows):
        rows["bbox_rows"][0]["evidence_id"] = "missing"

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
    def _unresolved_endpoint(rows):
        rows["graph_edges"][0]["source_id"] = "missing"

    @staticmethod
    def _unknown_derivation(rows):
        rows["graph_edges"][0]["derivation_method"] = "unknown"

    @staticmethod
    def _relation_support_mismatch(rows):
        rows["graph_edges"][0].pop("support_kind")

    @staticmethod
    def _missing_retrieval_evidence(rows):
        rows["retrieval_units"][0]["evidence_id"] = "missing"

    @staticmethod
    def _missing_chunk_evidence(rows):
        row = next(row for row in rows["chunks"] if row.get("evidence_ids"))
        row["evidence_ids"] = ["missing"]

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
