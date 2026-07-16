from __future__ import annotations

import json
from pathlib import Path
import unittest

from tjipto.core.manifest import read_jsonl


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "data/final/uud"


class ChunkGroundingContractTest(unittest.TestCase):
    def test_chunks_are_self_contained_and_match_legal_units(self) -> None:
        chunks = read_jsonl(FINAL / "chunks.jsonl")
        units = {row["legal_unit_id"]: row for row in read_jsonl(FINAL / "legal_units.jsonl")}
        self.assertEqual(len(chunks), 651)
        self.assertEqual(sum(1 for row in chunks if row["runtime_loadable"] is True), 469)
        self.assertEqual(sum(1 for row in chunks if row["runtime_loadable"] is False), 182)
        self.assertEqual(sum(1 for row in chunks if row.get("source_document_id")), len(chunks))
        self.assertEqual(sum(1 for row in chunks if row.get("source_role")), len(chunks))
        self.assertEqual(sum(1 for row in chunks if row.get("temporal_context")), len(chunks))
        self.assertEqual(sum(1 for row in chunks if row.get("validation_status")), len(chunks))
        self.assertEqual(sum(1 for row in chunks if row.get("validation_basis")), len(chunks))
        self.assertFalse([row for row in chunks if row["legal_unit_id"] not in units])
        for row in chunks:
            unit = units[row["legal_unit_id"]]
            self.assertEqual(row["source_document_id"], unit["source_document_id"], row["chunk_id"])
            self.assertEqual(row["source_role"], unit["source_role"], row["chunk_id"])
            self.assertEqual(row["temporal_context"], unit["temporal_context"], row["chunk_id"])
            if row["runtime_loadable"]:
                self.assertTrue(row["evidence_ids"], row["chunk_id"])
                self.assertTrue(row["bbox_ids"], row["chunk_id"])
                self.assertTrue(row["text_span_ids"], row["chunk_id"])
                self.assertNotEqual(row["validation_status"], "validation_error_missing_grounding", row["chunk_id"])

    def test_chunks_have_direct_grounding_fields(self) -> None:
        text_span_ids = {row["text_span_id"] for row in read_jsonl(FINAL / "page_text_spans.jsonl")}
        evidence_ids = {row["evidence_id"] for row in read_jsonl(FINAL / "evidence_registry.jsonl")}
        bbox_ids = {row["bbox_id"] for row in read_jsonl(FINAL / "bbox_registry.jsonl")}
        spans_by_page: dict[tuple[str, int], set[str]] = {}
        units = {row["legal_unit_id"]: row for row in read_jsonl(FINAL / "legal_units.jsonl")}
        for span in read_jsonl(FINAL / "page_text_spans.jsonl"):
            spans_by_page.setdefault((span["source_document_id"], span["page_number"]), set()).add(span["text_span_id"])
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
                self.assertEqual(row["grounding_status"], "text_span_exact", row["chunk_id"])
                unit = units[row["legal_unit_id"]]
                page_span_ids = set().union(
                    *[spans_by_page.get((unit["source_document_id"], page_number), set()) for page_number in row["page_numbers"]]
                )
                if page_span_ids != set(row["text_span_ids"]):
                    continue
                self.assertLessEqual(len(page_span_ids), len(row["bbox_ids"]), row["chunk_id"])

    def test_fixture_chunk_keeps_stable_grounding_ids(self) -> None:
        rows = {row["chunk_id"]: row for row in read_jsonl(FINAL / "chunks.jsonl")}
        for case in _chunk_grounding_cases():
            row = rows[case["chunk_id"]]
            for field in (
                "legal_unit_id",
                "grounding_status",
                "page_numbers",
                "text_span_ids",
                "evidence_ids",
                "bbox_ids",
            ):
                self.assertEqual(row[field], case[field], case["chunk_id"])


def _chunk_grounding_cases() -> tuple[dict, ...]:
    path = ROOT / "tests/fixtures/uud/chunk_grounding_cases.jsonl"
    return tuple(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


if __name__ == "__main__":
    unittest.main()
