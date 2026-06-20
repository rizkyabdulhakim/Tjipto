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
                "bbox_records": 1388,
                "chunks": 609,
                "document_metadata": 6,
                "evidence_records": 438,
                "excluded_records": 6,
                "graph_edges": 3150,
                "graph_nodes": 2339,
                "legal_units": 609,
                "metadata_assertions": 1319,
                "metadata_grounding": 5,
                "metadata_grounding_records": 5,
                "metadata_graph_edges": 449,
                "pages": 63,
                "retrieval_units": 438,
                "source_conflicts": 1,
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
        for name in report["referenced_artifacts"]:
            self.assertTrue((FINAL / name).exists(), name)


if __name__ == "__main__":
    unittest.main()
