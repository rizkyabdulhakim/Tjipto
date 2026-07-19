from __future__ import annotations

from pathlib import Path
import unittest

from tjipto.core.manifest import read_json, read_jsonl
from tjipto.evidence.bbox import bbox_is_accepted


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "data/final/uud"


class BBoxContractTest(unittest.TestCase):
    def test_bbox_rows_are_accepted_and_page_grounded(self) -> None:
        rows = read_jsonl(FINAL / "bbox_registry.jsonl")
        self.assertTrue(rows)
        for row in rows:
            self.assertTrue(bbox_is_accepted(row))
            self.assertIn(row["bbox_precision"], {"exact", "coarse", "page_grounded_only"})
            if row["viewer_highlightable"]:
                self.assertEqual(row["bbox_precision"], "exact")
            if row["bbox_precision"] != "exact" or row["viewer_highlightable"] is not True:
                self.assertIn("failure_reason", row, row["bbox_id"])
            self.assertGreaterEqual(row["x1"], row["x0"])
            self.assertGreaterEqual(row["y1"], row["y0"])
            self.assertTrue(row["text"])
            self.assertTrue((ROOT / row["source_pdf_path"]).exists())

    def test_parent_aggregate_owns_complete_child_span_closure(self) -> None:
        units = read_jsonl(FINAL / "legal_units.jsonl")
        evidence = {row["evidence_id"]: row for row in read_jsonl(FINAL / "evidence_registry.jsonl")}
        children: dict[str, list[dict]] = {}
        for unit in units:
            for parent_id in unit.get("parent_legal_unit_ids") or ():
                children.setdefault(parent_id, []).append(unit)
        for parent in (unit for unit in units if unit["legal_unit_id"] in children):
            owners = [evidence[evidence_id] for evidence_id in parent.get("evidence_ids") or () if evidence_id in evidence]
            self.assertEqual(len(owners), 1, parent["legal_unit_id"])
            owner = owners[0]
            self.assertTrue(owner.get("text_span_ids"), parent["legal_unit_id"])
            self.assertTrue(owner.get("bbox_refs"), parent["legal_unit_id"])
            self.assertEqual(owner.get("source_document_id"), parent.get("source_document_id"))
            self.assertEqual(owner.get("source_role"), parent.get("source_role"))
            parent_spans = set(owner["text_span_ids"])
            for child in children[parent["legal_unit_id"]]:
                self.assertTrue(set(child.get("text_span_ids") or ()) <= parent_spans, child["legal_unit_id"])

    def test_metadata_grounding_is_non_normative_and_bbox_linked(self) -> None:
        exact_bbox_ids = {row["bbox_id"] for row in read_jsonl(FINAL / "bbox_registry.jsonl")} | {
            row["word_bbox_id"] for row in read_jsonl(FINAL / "word_bboxes.jsonl")
        }
        metadata_grounding_ids = {row["bbox_id"] for row in read_jsonl(FINAL / "metadata_grounding_registry.jsonl")}
        metadata_registry_rows = read_jsonl(FINAL / "metadata_grounding_registry.jsonl")
        rows = read_jsonl(FINAL / "metadata_grounding.jsonl")
        report = read_json(FINAL / "validation_report.json")
        self.assertEqual(len(rows), 37)
        self.assertEqual(
            len(metadata_registry_rows),
            report["metadata_bbox_registry_health"]["metadata_grounding_registry_rows"],
        )
        for row in rows:
            self.assertEqual(row["status"], "accepted_metadata_grounding")
            self.assertTrue(row["quoted_text"])
            self.assertTrue(row["bbox_refs"])
            self.assertTrue(set(row["bbox_refs"]) <= metadata_grounding_ids)
            if row["bbox_precision"] == "exact":
                self.assertTrue(row["viewer_highlightable"])
                self.assertTrue(set(row["bbox_ids"]) <= exact_bbox_ids)
                self.assertTrue(row["text_span_ids"])
            else:
                self.assertFalse(row["viewer_highlightable"])
                self.assertEqual(row["bbox_precision"], "page_grounded_only")
            self.assertTrue((ROOT / row["source_pdf_path"]).exists())

    def test_inserted_bab_heading_bboxes_attach_to_matching_structure(self) -> None:
        evidence = {row["evidence_id"]: row for row in read_jsonl(FINAL / "evidence_registry.jsonl")}
        legal_units = {row["legal_unit_id"]: row for row in read_jsonl(FINAL / "legal_units.jsonl")}
        expected = {
            "BAB IXA": "BAB IXA",
            "BAB XA": "BAB XA",
            "BAB VIIA": "BAB VIIA",
            "BAB VIIB": "BAB VIIB",
            "BAB VIIIA": "BAB VIIIA",
        }
        heading_rows = [row for row in read_jsonl(FINAL / "bbox_registry.jsonl") if row["text"] in expected]
        self.assertEqual(len(heading_rows), 5)
        for row in heading_rows:
            target = evidence[row["evidence_id"]]
            owner = legal_units[target["legal_unit_id"]]
            self.assertIn(row["bbox_id"], target["bbox_refs"])
            self.assertEqual(target["hierarchy"][0], expected[row["text"]], row["bbox_id"])
            self.assertEqual(owner["unit_type"], "bab_record", row["bbox_id"])
            self.assertEqual(target["authority_kind"], "structural_context", row["bbox_id"])
            self.assertFalse(target["citation_final"], row["bbox_id"])
            self.assertTrue(target["viewer_highlightable"], row["bbox_id"])
            self.assertEqual(target["evidence_owner_kind"], "legal_unit_source", row["bbox_id"])
            self.assertGreaterEqual(len(target["bbox_refs"]), 1, row["bbox_id"])

    def test_decision_bbox_precision_is_exact_or_non_highlightable(self) -> None:
        bbox_by_id = {row["bbox_id"]: row for row in read_jsonl(FINAL / "bbox_registry.jsonl")}
        for label in (
            "Perubahan Pertama Decision",
            "Perubahan Ketiga Decision",
            "Perubahan Keempat Decision",
        ):
            evidence = next(row for row in read_jsonl(FINAL / "evidence_registry.jsonl") if row["citation"] == label)
            self.assertIn(evidence["bbox_precision"], {"exact", "page_grounded_only", "coarse"})
            if evidence["viewer_highlightable"]:
                self.assertEqual(evidence["bbox_precision"], "exact")
            for bbox_id in evidence["bbox_refs"]:
                bbox = bbox_by_id[bbox_id]
                self.assertIn(bbox["bbox_precision"], {"exact", "page_grounded_only", "coarse"})
                if bbox["viewer_highlightable"]:
                    self.assertEqual(bbox["bbox_precision"], "exact")
                if bbox["viewer_highlightable"]:
                    self.assertNotIn("Pasal ", bbox["text"])

    def test_span_disposition_and_local_bbox_contract_complete(self) -> None:
        report = read_json(FINAL / "validation_report.json")["viewer_provenance_coverage_health"]
        self.assertGreater(report["bbox_key_absent_span_count"], 0)
        self.assertEqual(report["incomplete_disposition_count"], 0)
        self.assertEqual(report["highlight_without_span_bbox_count"], 0)
        self.assertEqual(report["final_without_exact_span_bbox_count"], 0)

    def test_word_bbox_rows_are_valid_and_nonempty(self) -> None:
        rows = read_jsonl(FINAL / "word_bboxes.jsonl")
        self.assertEqual(len(rows), 11336)
        self.assertEqual({row["extractor_version"] for row in rows}, {"pymupdf_words"})
        for row in rows:
            self.assertTrue(row["normalized_text"])
            self.assertGreaterEqual(row["x1"], row["x0"])
            self.assertGreaterEqual(row["y1"], row["y0"])
            self.assertGreater(row["page_width"], 0)
            self.assertGreater(row["page_height"], 0)

    def test_highlight_registry_contract_includes_word_bboxes(self) -> None:
        report = read_json(FINAL / "validation_report.json")["highlight_registry_contract"]
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["architecture"], "bbox_registry_union_word_bboxes")
        self.assertEqual(report["official_viewer_highlight_ref_sources"], ["bbox_registry", "word_bboxes"])
        self.assertEqual(report["bbox_registry_layer"], "materialized_final_and_provenance_bboxes")
        self.assertEqual(report["word_bbox_layer"], "word_bbox_exact_highlight")
        self.assertEqual(report["viewer_highlightable_union"], "bbox_registry_union_word_bboxes")
        self.assertGreater(report["bbox_key_absent_span_count"], 0)
        spans = read_jsonl(FINAL / "page_text_spans.jsonl")
        words = {row["word_bbox_id"]: row for row in read_jsonl(FINAL / "word_bboxes.jsonl")}
        evidence = {row["evidence_id"]: row for row in read_jsonl(FINAL / "evidence_registry.jsonl")}
        expected = {
            row["text_span_id"]
            for row in spans
            if row.get("promotion_status") == "promoted_legal_unit"
            and row.get("bbox_registry_coverage_reason") == "exact_word_bbox_available"
            and row.get("evidence_ids")
            and row.get("span_bbox_ids")
            and all(
                word_id in words
                and words[word_id]["source_document_id"] == row["source_document_id"]
                and words[word_id]["page_number"] == row["page_number"]
                for word_id in row["span_bbox_ids"]
            )
            and all(evidence.get(evidence_id, {}).get("exactness") == "exact" for evidence_id in row["evidence_ids"])
        }
        self.assertEqual(report["exact_safe_word_highlight_count"], len(expected))
        self.assertEqual(len(expected), sum(1 for row in spans if row.get("highlightable") is True))
        self.assertGreaterEqual(report["non_citable_absent_span_count"], 633)
        self.assertEqual(report["false_highlight_count"], 0)


if __name__ == "__main__":
    unittest.main()
