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
        self.assertEqual(len(rows), 1584)
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
            self.assertIn("heading_bab_", row["bbox_id"])
            self.assertEqual(target["hierarchy"][0], expected[row["text"]], row["bbox_id"])
            if owner["unit_type"] != "bab_record":
                self.assertFalse(row["viewer_highlightable"], row["bbox_id"])

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

    def test_634_span_exposure_policy_complete(self) -> None:
        report = read_json(FINAL / "validation_report.json")["viewer_provenance_coverage_health"]
        self.assertEqual(report["bbox_key_absent_span_count"], 634)
        self.assertEqual(report["missing_exposure_policy_count"], 0)
        self.assertEqual(report["missing_exposure_target_count"], 0)
        self.assertEqual(report["missing_field_bbox_feasibility_count"], 0)
        self.assertEqual(report["clickable_absent_span_count"], 178)
        self.assertEqual(report["final_citation_absent_span_count"], 0)
        self.assertEqual(report["false_exact_absent_span_count"], 0)
        self.assertEqual(report["false_highlight_exposure_policy_count"], 0)
        self.assertEqual(report["legal_citation_highlight_count"], 153)
        self.assertEqual(report["metadata_source_highlight_count"], 8)
        self.assertEqual(report["source_anomaly_provenance_highlight_count"], 17)
        self.assertEqual(
            sum(
                report[f"{policy}_count"]
                for policy in (
                    "legal_citation_highlight",
                    "metadata_source_highlight",
                    "nonlegal_excluded_position",
                    "source_anomaly_provenance_highlight",
                    "structural_provenance_position",
                )
            ),
            634,
        )
        self.assertEqual(report["exact_word_bbox_available_count"], 178)
        self.assertEqual(report["exact_safe_feasibility_count"], 178)
        self.assertEqual(report["requires_word_level_bbox_feasibility_count"], 0)

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


if __name__ == "__main__":
    unittest.main()
