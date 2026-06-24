from __future__ import annotations

import json
from pathlib import Path
import unittest

from tjipto.core.manifest import file_sha256, validate_manifest


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "data/final/uud"


class ManifestContractTest(unittest.TestCase):
    def test_manifest_is_clean_and_hash_valid(self) -> None:
        self.assertEqual(validate_manifest(FINAL), ())
        manifest = json.loads((FINAL / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["corpus_id"], "uud")
        self.assertEqual(manifest["status"], "final")
        self.assertEqual(
            manifest["counts"],
            {
                "article_versions": 218,
                "bbox_records": 1542,
                "chunks": 651,
                "document_metadata": 6,
                "evidence_records": 464,
                "excluded_records": 6,
                "graph_edges": len((FINAL / "graph_edges.jsonl").read_text(encoding="utf-8").splitlines()),
                "graph_nodes": len((FINAL / "graph_nodes.jsonl").read_text(encoding="utf-8").splitlines()),
                "legal_units": 651,
                "metadata_assertions": 1319,
                "metadata_grounding": 37,
                "metadata_grounding_records": 37,
                "metadata_graph_edges": 449,
                "pages": 63,
                "retrieval_units": 464,
                "source_conflicts": 2,
                "source_documents": 6,
                "validation_alignment_results": 610,
                "validation_exception_review_labels": 9,
                "validation_exceptions": 19,
                "not_promoted_amends_edges": 8,
            },
        )
        for source_path, expected_sha in manifest["source_files"].items():
            self.assertEqual(file_sha256(ROOT / source_path), expected_sha)
        self.assertFalse((FINAL / "eval_fixtures.jsonl").exists())
        self.assertTrue((ROOT / "tests/fixtures/uud/eval_fixtures.jsonl").exists())

    def test_validation_report_references_existing_artifacts(self) -> None:
        report = json.loads((FINAL / "validation_report.json").read_text(encoding="utf-8"))
        manifest = json.loads((FINAL / "manifest.json").read_text(encoding="utf-8"))
        for name in report["referenced_artifacts"]:
            self.assertTrue((FINAL / name).exists(), name)
        self.assertEqual(report["structure_fidelity"]["status"], "corrected")
        self.assertEqual(report["metadata_grounding_contract"]["status"], "field_grounded")
        self.assertEqual(report["legal_graph_baseline"]["status"], "evidence_backed_minimal_baseline")
        actual_counts = {
            "chunks": len((FINAL / "chunks.jsonl").read_text(encoding="utf-8").splitlines()),
            "legal_units": len((FINAL / "legal_units.jsonl").read_text(encoding="utf-8").splitlines()),
            "evidence_records": len((FINAL / "evidence_registry.jsonl").read_text(encoding="utf-8").splitlines()),
            "bbox_records": len((FINAL / "bbox_registry.jsonl").read_text(encoding="utf-8").splitlines()),
            "retrieval_units": len((FINAL / "retrieval_units.jsonl").read_text(encoding="utf-8").splitlines()),
            "graph_nodes": len((FINAL / "graph_nodes.jsonl").read_text(encoding="utf-8").splitlines()),
            "graph_edges": len((FINAL / "graph_edges.jsonl").read_text(encoding="utf-8").splitlines()),
        }
        for key, value in actual_counts.items():
            self.assertEqual(report["final_artifact_counts"][key], value)
            self.assertEqual(manifest["counts"][key], value)
        bbox_rows = [json.loads(line) for line in (FINAL / "bbox_registry.jsonl").read_text(encoding="utf-8").splitlines() if line]
        self.assertEqual(
            report["bbox_precision_counts"],
            {
                "exact": sum(1 for row in bbox_rows if row["bbox_precision"] == "exact"),
                "coarse": sum(1 for row in bbox_rows if row["bbox_precision"] == "coarse"),
                "page_grounded_only": sum(1 for row in bbox_rows if row["bbox_precision"] == "page_grounded_only"),
            },
        )
        self.assertEqual(
            report["bbox_highlightability_counts"]["non_highlightable"],
            sum(1 for row in bbox_rows if row["viewer_highlightable"] is not True),
        )
        self.assertEqual(report["instrument_baseline"]["status"], "corrected")
        self.assertFalse(report["instrument_baseline"]["metadata_viewer_highlightable"])
        self.assertEqual(report["bbox_precision_policy"]["status"], "corrected")
        self.assertEqual(report["artifact_governance"]["status"], "current_final_artifacts_present")
        self.assertIn(
            "Pasal 22D ayat (3) bbox/text exception remains tracked",
            report["artifact_governance"]["reviewed_exceptions_preserved"][0],
        )
        report_text = json.dumps(report).casefold()
        self.assertNotIn("no_final_legal_units_created", report_text)
        self.assertNotIn("no_bbox_created", report_text)


if __name__ == "__main__":
    unittest.main()
