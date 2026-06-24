from __future__ import annotations

import json
from pathlib import Path
import unittest

from tjipto.core.manifest import file_sha256, validate_manifest


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "data/final/uud"


class CleanArchiveManifestContractTest(unittest.TestCase):
    def test_manifest_files_are_lf_only_and_hash_current_bytes(self) -> None:
        manifest = json.loads((FINAL / "manifest.json").read_text(encoding="utf-8"))
        for rel, expected in manifest["files"].items():
            path = FINAL / rel
            data = path.read_bytes()
            self.assertNotIn(b"\r\n", data, rel)
            self.assertEqual(len(data), expected["bytes"], rel)
            self.assertEqual(file_sha256(path), expected["sha256"], rel)
        self.assertEqual(validate_manifest(FINAL), ())


if __name__ == "__main__":
    unittest.main()
