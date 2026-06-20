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
        source_docs = read_jsonl(FINAL / "source_documents.jsonl")
        self.assertEqual(len(units), 609)
        self.assertEqual(len(chunks), 609)
        unit_ids = {row["legal_unit_id"] for row in units}
        source_doc_ids = {row["source_document_id"] for row in source_docs}
        for row in chunks:
            self.assertIn(row["legal_unit_id"], unit_ids)
            self.assertTrue(row["chunk_id"].startswith("uud_chunk_"))
            self.assertNotIn("candidate", row["chunk_id"])
        for row in units:
            self.assertIn(row["source_document_id"], source_doc_ids)
            for parent_id in row["parent_legal_unit_ids"]:
                self.assertIn(parent_id, unit_ids)

    def test_source_integrity_references_source_documents(self) -> None:
        source_ids = {
            row["source_document_id"]
            for row in read_jsonl(FINAL / "source_documents.jsonl")
        }
        import json

        source_integrity = json.loads(
            (FINAL / "source_integrity.json").read_text(encoding="utf-8")
        )
        self.assertEqual(source_integrity["source_count"], 6)
        for doc in source_integrity["source_documents"]:
            self.assertIn(doc["source_document_id"], source_ids)


if __name__ == "__main__":
    unittest.main()
