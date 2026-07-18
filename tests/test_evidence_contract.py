from __future__ import annotations

from pathlib import Path
import unittest

from tjipto.core.manifest import read_jsonl
from tjipto.corpora.uud.evidence_bbox_builder import _admit_evidence


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "data/final/uud"


class EvidenceContractTest(unittest.TestCase):
    def test_final_evidence_rows_are_grounded(self) -> None:
        rows = read_jsonl(FINAL / "evidence_registry.jsonl")
        self.assertGreaterEqual(len(rows), 472)
        for row in rows:
            self.assertEqual(row["corpus_id"], "uud")
            self.assertEqual(row["status"], "final")
            self.assertTrue(row["evidence_id"])
            self.assertTrue(row["source_sha256"])
            self.assertTrue((ROOT / row["source_pdf_path"]).exists())
            self.assertTrue(row["page_numbers"])
            self.assertTrue(row["quoted_text"])
            self.assertTrue(row["bbox_refs"])
            self.assertEqual(row["bbox_ids"], row["bbox_refs"])
            self.assertNotIn("uud_chunk_candidate_", str(row))

    def test_legal_units_and_chunks_are_linked(self) -> None:
        units = read_jsonl(FINAL / "legal_units.jsonl")
        chunks = read_jsonl(FINAL / "chunks.jsonl")
        source_docs = read_jsonl(FINAL / "source_documents.jsonl")
        self.assertEqual(len(units), 651)
        self.assertEqual(len(chunks), 651)
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

    def test_ordinal_exclusion_records_are_not_used_for_admission(self) -> None:
        excluded = read_jsonl(FINAL / "excluded_records.jsonl")
        self.assertEqual(excluded, [])
        evidence = {row["legal_unit_id"] for row in read_jsonl(FINAL / "evidence_registry.jsonl")}
        for unit_id in (
            "uud_legal_unit_00383",
            "uud_legal_unit_00400",
            "uud_legal_unit_00475",
            "uud_legal_unit_00479",
            "uud_legal_unit_00521",
            "uud_legal_unit_00566",
        ):
            self.assertIn(unit_id, evidence)

    def test_evidence_admission_is_stable_when_chunk_ordinals_change(self) -> None:
        units = {row["legal_unit_id"]: row for row in read_jsonl(FINAL / "legal_units.jsonl")}
        chunks = {row["legal_unit_id"]: row for row in read_jsonl(FINAL / "chunks.jsonl")}
        for unit_id in (
            "uud_legal_unit_00383",
            "uud_legal_unit_00400",
            "uud_legal_unit_00475",
            "uud_legal_unit_00479",
            "uud_legal_unit_00521",
            "uud_legal_unit_00566",
        ):
            unit = units[unit_id]
            chunk = chunks[unit_id] | {"chunk_id": "uud_chunk_reordered"}
            self.assertTrue(_admit_evidence(unit, chunk), unit_id)

    def test_bab_records_do_not_point_to_child_bab(self) -> None:
        for row in read_jsonl(FINAL / "legal_units.jsonl"):
            if row["unit_type"] == "bab_record":
                self.assertFalse(
                    [item for item in row.get("hierarchy", ()) if str(item).startswith("BAB")],
                    row["legal_unit_id"],
                )

    def test_source_integrity_references_source_documents(self) -> None:
        source_docs = read_jsonl(FINAL / "source_documents.jsonl")
        source_ids = {row["source_document_id"] for row in source_docs}
        import json

        source_integrity = json.loads((FINAL / "source_integrity.json").read_text(encoding="utf-8"))
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
        source_ids = {row["source_document_id"] for row in read_jsonl(FINAL / "source_documents.jsonl")}
        for filename in (
            "metadata_grounding.jsonl",
            "metadata_grounding_registry.jsonl",
            "document_metadata.jsonl",
        ):
            rows = read_jsonl(FINAL / filename)
            self.assertTrue(rows)
            for row in rows:
                self.assertIn(row["source_document_id"], source_ids)
        grounding = {row["metadata_grounding_id"]: row for row in read_jsonl(FINAL / "metadata_grounding.jsonl")}
        field_grounding = [row for row in grounding.values() if row.get("metadata_field")]
        docs = read_jsonl(FINAL / "document_metadata.jsonl")
        self.assertEqual(len(docs), 6)
        self.assertGreater(len(field_grounding), 5)
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
                self.assertIn("penetapan", row["grounded_fields"])
                self.assertIn("institution", row["grounded_fields"])
                self.assertIn("signatories", row["grounded_fields"])
            if row["source_role"] == "amendment_3_historical":
                self.assertEqual(row["decision_date"], "9 November 2001")
                self.assertEqual(row["field_statuses"]["decision_date"], "grounded")
                self.assertEqual(row["field_statuses"]["decision_session"], "grounded")
                self.assertEqual(row["field_statuses"]["effective_rule"], "grounded")
                self.assertIn("decision_date", row["grounded_fields"])
                self.assertIn("decision_session", row["grounded_fields"])
                self.assertIn("effective_rule", row["grounded_fields"])
            if row["source_role"] == "amendment_2_historical":
                self.assertEqual(row["source_anomaly_status"], "source_article_renumbering_provenance")
                self.assertEqual(row["field_statuses"]["source_anomaly_status"], "artifact_recorded")
            if row["source_role"] == "original_historical":
                self.assertEqual(row["status"], "not_found_in_source")
                self.assertIsNone(row["official_title"])
            self.assertIn(
                row["field_statuses"].get("ln_tln"),
                {"not_found_in_source", None},
            )
        for row in field_grounding:
            self.assertFalse(row["runtime_loadable"])
            self.assertTrue(row["quote"])
            if row["bbox_precision"] == "exact":
                self.assertTrue(row["viewer_highlightable"])
                self.assertEqual(row["grounding_status"], "text_bbox_exact")
                self.assertTrue(row["bbox_ids"])
                self.assertTrue(row["text_span_ids"])
            else:
                self.assertFalse(row["viewer_highlightable"])
                self.assertEqual(row["grounding_status"], "field_level_grounded")
                self.assertEqual(row["bbox_precision"], "page_grounded_only")
                self.assertIn("failure_reason", row)

    def test_instrument_units_and_historical_anomaly_are_preserved(self) -> None:
        units = read_jsonl(FINAL / "legal_units.jsonl")
        chunks = {row["legal_unit_id"]: row for row in read_jsonl(FINAL / "chunks.jsonl")}
        labels = {row.get("unit_label") for row in units}
        for label in (
            "Perubahan Pertama Recital",
            "Perubahan Kedua Scope",
            "Perubahan Ketiga Closing",
            "Perubahan Keempat Clause (a)",
            "Perubahan Keempat Clause (b)",
            "Perubahan Keempat Clause (c)",
            "Perubahan Keempat Clause (d)",
            "Perubahan Keempat Clause (e)",
        ):
            self.assertIn(label, labels)

        anomaly_rows = [row for row in units if row.get("exclusion_ref") == "source_typo_reference::uud_source_typo_reference_00001"]
        self.assertEqual(
            {row["unit_label"] for row in anomaly_rows},
            {"ATURAN TAMBAHAN source typo reference", "Pasal III"},
        )
        for row in anomaly_rows:
            chunk = chunks[row["legal_unit_id"]]
            self.assertFalse(chunk["canonical_use_allowed"])
            if row["unit_label"] == "ATURAN TAMBAHAN source typo reference":
                self.assertFalse(row["runtime_loadable"])
                self.assertEqual(row["status"], "inactive_source_typo_reference")
            else:
                self.assertTrue(row["runtime_loadable"])
                self.assertEqual(row["status"], "active_historical_record")
                self.assertEqual(chunk["status"], "active_historical_record")
                self.assertEqual(row["provenance_review_status"], "resolved_exact_historical_evidence")
        pasal_i = next(row for row in units if row.get("hierarchy") == ["ATURAN TAMBAHAN", "Pasal I"])
        self.assertIsNone(pasal_i.get("exclusion_ref"))
        self.assertTrue(pasal_i["runtime_loadable"])
        self.assertFalse(chunks[pasal_i["legal_unit_id"]]["canonical_use_allowed"])

    def test_closing_clauses_are_separated_from_normative_units(self) -> None:
        forbidden = (
            "Naskah perubahan ini merupakan bagian tak terpisahkan",
            "Perubahan tersebut diputuskan",
            "Ditetapkan di Jakarta",
        )
        for row in read_jsonl(FINAL / "legal_units.jsonl"):
            if row["unit_type"] not in {"pasal_record", "ayat_record"}:
                continue
            self.assertFalse(any(marker in row["text"] for marker in forbidden), row["legal_unit_id"])

    def test_amendment_2_bab_xv_parent_context_is_clean(self) -> None:
        units = read_jsonl(FINAL / "legal_units.jsonl")
        unit = next(
            row for row in units if row["source_document_id"] == "uud::amendment_2_historical" and row.get("unit_label") == "BAB XV"
        )
        self.assertNotIn("Ditetapkan di Jakarta", unit["text"])
        self.assertNotIn("MAJELIS PERMUSYAWARATAN RAKYAT", unit["text"])
        chunk = next(row for row in read_jsonl(FINAL / "chunks.jsonl") if row["legal_unit_id"] == unit["legal_unit_id"])
        self.assertEqual(chunk["chunk_type"], "bab_structural_context_record")
        self.assertNotIn("Ditetapkan di Jakarta", chunk["text"])
        self.assertNotIn("MAJELIS PERMUSYAWARATAN RAKYAT", chunk["text"])


if __name__ == "__main__":
    unittest.main()
