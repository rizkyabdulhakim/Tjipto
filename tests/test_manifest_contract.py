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
            _expected_manifest_counts(),
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
        self.assertEqual(report["metadata_grounding_contract"]["status"], "mixed_exact_and_field_grounded")
        self.assertEqual(report["legal_graph_baseline"]["status"], "authority_aware_evidence_gated")
        actual_counts = {
            "chunks": len((FINAL / "chunks.jsonl").read_text(encoding="utf-8").splitlines()),
            "legal_units": len((FINAL / "legal_units.jsonl").read_text(encoding="utf-8").splitlines()),
            "evidence_records": len((FINAL / "evidence_registry.jsonl").read_text(encoding="utf-8").splitlines()),
            "bbox_records": len((FINAL / "bbox_registry.jsonl").read_text(encoding="utf-8").splitlines()),
            "retrieval_units": len((FINAL / "retrieval_units.jsonl").read_text(encoding="utf-8").splitlines()),
            "graph_nodes": len((FINAL / "graph_nodes.jsonl").read_text(encoding="utf-8").splitlines()),
            "graph_edges": len((FINAL / "graph_edges.jsonl").read_text(encoding="utf-8").splitlines()),
            "page_text_spans": len((FINAL / "page_text_spans.jsonl").read_text(encoding="utf-8").splitlines()),
            "promotion_decisions": len((FINAL / "promotion_decisions.jsonl").read_text(encoding="utf-8").splitlines()),
            "propositions": len((FINAL / "propositions.jsonl").read_text(encoding="utf-8").splitlines()),
            "word_bboxes": len((FINAL / "word_bboxes.jsonl").read_text(encoding="utf-8").splitlines()),
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
        self.assertEqual(report["artifact_origin_health"]["files_missing_origin"], 0)
        self.assertIn(
            "Pasal 22D ayat (3) bbox/text exception remains tracked",
            report["artifact_governance"]["reviewed_exceptions_preserved"][0],
        )
        report_text = json.dumps(report).casefold()
        self.assertNotIn("no_final_legal_units_created", report_text)
        self.assertNotIn("no_bbox_created", report_text)


if __name__ == "__main__":
    unittest.main()


def _jsonl_count(name: str) -> int:
    return len((FINAL / name).read_text(encoding="utf-8").splitlines())


def _expected_manifest_counts() -> dict[str, int]:
    return {
        "article_amendment_relations": _jsonl_count("article_amendment_relations.jsonl"),
        "article_versions": _jsonl_count("article_versions.jsonl"),
        "bbox_records": _jsonl_count("bbox_registry.jsonl"),
        "chunks": _jsonl_count("chunks.jsonl"),
        "document_metadata": _jsonl_count("document_metadata.jsonl"),
        "document_relations": _jsonl_count("document_relations.jsonl"),
        "evidence_records": _jsonl_count("evidence_registry.jsonl"),
        "excluded_records": _jsonl_count("excluded_records.jsonl"),
        "graph_edges": _jsonl_count("graph_edges.jsonl"),
        "graph_nodes": _jsonl_count("graph_nodes.jsonl"),
        "legal_units": _jsonl_count("legal_units.jsonl"),
        "metadata_assertions": _jsonl_count("metadata.jsonl"),
        "metadata_graph_edges": _jsonl_count("metadata_graph_edges.jsonl"),
        "metadata_grounding": _jsonl_count("metadata_grounding.jsonl"),
        "metadata_grounding_records": _jsonl_count("metadata_grounding_registry.jsonl"),
        "not_promoted_amends_edges": 8,
        "page_text_spans": _jsonl_count("page_text_spans.jsonl"),
        "raw_source_spans": _jsonl_count("raw_source_spans.jsonl"),
        "pages": _jsonl_count("pages.jsonl"),
        "promotion_decisions": _jsonl_count("promotion_decisions.jsonl"),
        "propositions": _jsonl_count("propositions.jsonl"),
        "retrieval_units": _jsonl_count("retrieval_units.jsonl"),
        "source_conflicts": _jsonl_count("source_conflicts.jsonl"),
        "source_documents": _jsonl_count("source_documents.jsonl"),
        "validation_alignment_results": _jsonl_count("validation_alignment_results.jsonl"),
        "validation_exception_review_labels": _jsonl_count("validation_exception_review_labels.jsonl"),
        "validation_exceptions": _jsonl_count("validation_exceptions.jsonl"),
        "word_bboxes": _jsonl_count("word_bboxes.jsonl"),
    }
