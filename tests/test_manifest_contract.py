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
                "bbox_records": 1388,
                "evidence_records": 438,
                "graph_edges": 3150,
                "graph_nodes": 2339,
                "source_documents": 6,
            },
        )
        for source_path, expected_sha in manifest["source_files"].items():
            self.assertEqual(file_sha256(ROOT / source_path), expected_sha)


if __name__ == "__main__":
    unittest.main()
