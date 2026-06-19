from __future__ import annotations

from pathlib import Path
import unittest

from tjipto.core.manifest import read_jsonl


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "data/final/uud"


class EvidenceContractTest(unittest.TestCase):
    def test_final_evidence_rows_are_grounded(self) -> None:
        rows = read_jsonl(FINAL / "evidence_registry.jsonl")
        self.assertEqual(len(rows), 438)
        for row in rows:
            self.assertEqual(row["corpus_id"], "uud")
            self.assertEqual(row["status"], "final")
            self.assertTrue(row["evidence_id"])
            self.assertTrue(row["source_sha256"])
            self.assertTrue((ROOT / row["source_pdf_path"]).exists())
            self.assertTrue(row["page_numbers"])
            self.assertTrue(row["quoted_text"])
            self.assertTrue(row["bbox_refs"])
            self.assertNotIn("uud_chunk_candidate_", str(row))

    def test_legal_units_and_chunks_are_linked(self) -> None:
        units = read_jsonl(FINAL / "legal_units.jsonl")
        chunks = read_jsonl(FINAL / "chunks.jsonl")
        self.assertEqual(len(units), 609)
        self.assertEqual(len(chunks), 609)
        unit_ids = {row["legal_unit_id"] for row in units}
        for row in chunks:
            self.assertIn(row["legal_unit_id"], unit_ids)
            self.assertTrue(row["chunk_id"].startswith("uud_chunk_"))
            self.assertNotIn("candidate", row["chunk_id"])
        for row in units:
            for parent_id in row["parent_legal_unit_ids"]:
                self.assertIn(parent_id, unit_ids)


if __name__ == "__main__":
    unittest.main()
