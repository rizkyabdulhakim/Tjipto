from __future__ import annotations

from pathlib import Path
import unittest

from tjipto.core.manifest import read_jsonl


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "data/final/uud"


class ChunkGroundingContractTest(unittest.TestCase):
    def test_chunks_have_direct_grounding_fields(self) -> None:
        text_span_ids = {row["text_span_id"] for row in read_jsonl(FINAL / "page_text_spans.jsonl")}
        evidence_ids = {row["evidence_id"] for row in read_jsonl(FINAL / "evidence_registry.jsonl")}
        bbox_ids = {row["bbox_id"] for row in read_jsonl(FINAL / "bbox_registry.jsonl")}
        for row in read_jsonl(FINAL / "chunks.jsonl"):
            self.assertIn("text_span_ids", row)
            self.assertIn("evidence_ids", row)
            self.assertIn("bbox_ids", row)
            self.assertIn("page_numbers", row)
            self.assertIn("grounding_status", row)
            self.assertIn("runtime_loadable", row)
            self.assertTrue(set(row["text_span_ids"]) <= text_span_ids)
            self.assertTrue(set(row["evidence_ids"]) <= evidence_ids)
            self.assertTrue(set(row["bbox_ids"]) <= bbox_ids)
            if row["runtime_loadable"]:
                self.assertTrue(row["evidence_ids"], row["chunk_id"])
                self.assertTrue(row["bbox_ids"], row["chunk_id"])


if __name__ == "__main__":
    unittest.main()
