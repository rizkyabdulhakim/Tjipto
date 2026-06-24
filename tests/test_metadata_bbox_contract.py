from __future__ import annotations

from pathlib import Path
import unittest

from tjipto.core.manifest import read_jsonl


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "data/final/uud"


class MetadataBBoxContractTest(unittest.TestCase):
    def test_metadata_grounding_stays_fail_closed_until_exact_bbox_exists(self) -> None:
        registry_ids = {row["bbox_id"] for row in read_jsonl(FINAL / "metadata_grounding_registry.jsonl")}
        for row in read_jsonl(FINAL / "metadata_grounding.jsonl"):
            self.assertEqual(row["bbox_precision"], "page_grounded_only", row["metadata_grounding_id"])
            self.assertFalse(row["viewer_highlightable"], row["metadata_grounding_id"])
            self.assertTrue(set(row["bbox_refs"]) <= registry_ids, row["metadata_grounding_id"])
            self.assertIn("grounding_status", row)


if __name__ == "__main__":
    unittest.main()
