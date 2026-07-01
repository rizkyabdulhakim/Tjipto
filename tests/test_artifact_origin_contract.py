from __future__ import annotations

import json
from pathlib import Path
import unittest

from tjipto.corpora.uud.artifact_policy import ALLOWED_ARTIFACT_ORIGINS, UUD_ARTIFACT_ORIGIN_POLICY
from tjipto.core.manifest import validate_manifest


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "data/final/uud"


class ArtifactOriginContractTest(unittest.TestCase):
    def test_manifest_file_records_have_origin_metadata(self) -> None:
        manifest = json.loads((FINAL / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(len(manifest["files"]), 23)
        self.assertEqual(set(manifest["files"]), set(UUD_ARTIFACT_ORIGIN_POLICY))
        self.assertEqual(validate_manifest(FINAL), ())
        for rel, row in manifest["files"].items():
            self.assertIn(row["origin"], ALLOWED_ARTIFACT_ORIGINS, rel)
            self.assertTrue(row["producer"], rel)
            self.assertTrue(row["build_stage"], rel)
            if row["origin"] == "generated":
                self.assertTrue(row["producer"], rel)
                self.assertTrue(row["build_stage"], rel)
            else:
                self.assertTrue(row["origin_reason"], rel)

    def test_validation_report_records_artifact_origin_health(self) -> None:
        report = json.loads((FINAL / "validation_report.json").read_text(encoding="utf-8"))
        health = report["artifact_origin_health"]
        self.assertEqual(health["manifest_file_rows"], 23)
        self.assertEqual(health["files_with_origin"], 23)
        self.assertEqual(health["files_missing_origin"], 0)
        self.assertEqual(health["invalid_origin_values"], 0)
        self.assertEqual(health["generated_missing_producer_count"], 0)
        self.assertEqual(health["generated_missing_build_stage_count"], 0)
        self.assertEqual(health["non_generated_missing_origin_reason_count"], 0)


if __name__ == "__main__":
    unittest.main()
