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
        source_docs = read_jsonl(FINAL / "source_documents.jsonl")
        source_ids = {row["source_document_id"] for row in source_docs}
        import json

        source_integrity = json.loads(
            (FINAL / "source_integrity.json").read_text(encoding="utf-8")
        )
        self.assertEqual(source_integrity["source_count"], 6)
        for doc in source_integrity["source_documents"]:
            self.assertIn(doc["source_document_id"], source_ids)
            self.assertTrue((ROOT / doc["path"]).exists())
            self.assertTrue(doc["sha256_match"])
            self.assertTrue(doc["page_count_match"])
            self.assertTrue(doc["file_size_match"])
            self.assertEqual(doc["reproducibility_status"], "passed")
            self.assertTrue(doc["source_page_url"])
            self.assertTrue(doc["download_url"])
        for doc in source_docs:
            self.assertIn("source_page_url", doc)
            self.assertIn("content_fingerprint", doc)

    def test_metadata_grounding_and_document_metadata_reference_sources(self) -> None:
        source_ids = {
            row["source_document_id"]
            for row in read_jsonl(FINAL / "source_documents.jsonl")
        }
        for filename in (
            "metadata_grounding.jsonl",
            "metadata_grounding_registry.jsonl",
            "document_metadata.jsonl",
        ):
            rows = read_jsonl(FINAL / filename)
            self.assertTrue(rows)
            for row in rows:
                self.assertIn(row["source_document_id"], source_ids)
        grounding = {
            row["metadata_grounding_id"]: row
            for row in read_jsonl(FINAL / "metadata_grounding.jsonl")
        }
        docs = read_jsonl(FINAL / "document_metadata.jsonl")
        self.assertEqual(len(docs), 6)
        for row in docs:
            for refs in row["grounded_fields"].values():
                for ref in refs:
                    self.assertIn(ref, grounding)
            if row["source_role"].startswith("amendment_"):
                if row["source_role"] == "amendment_1_historical":
                    self.assertEqual(row["date"], "19 Oktober 1999")
                if row["source_role"] == "amendment_2_historical":
                    self.assertEqual(row["date"], "18 Agustus 2000")
                if row["source_role"] == "amendment_3_historical":
                    self.assertEqual(row["date"], "9 November 2001")
                if row["source_role"] == "amendment_4_historical":
                    self.assertEqual(row["date"], "10 Agustus 2002")
                if row["date"]:
                    quote = grounding[row["grounded_fields"]["date"][0]]["quoted_text"]
                    self.assertIn(row["date"], quote)
                    self.assertIn(row["place"], quote)
                    self.assertIn(row["institution"], quote)
            if row["source_role"] == "original_historical":
                self.assertEqual(row["status"], "not_found_in_source")
                self.assertIsNone(row["official_title"])
            self.assertIn(
                row["field_statuses"].get("ln_tln"),
                {"not_found_in_source", None},
            )


if __name__ == "__main__":
    unittest.main()
