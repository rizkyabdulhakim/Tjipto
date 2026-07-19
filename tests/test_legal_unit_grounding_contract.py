from __future__ import annotations

import json
from pathlib import Path
import unittest

from tjipto.core.manifest import read_json, read_jsonl


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "data/final/uud"


class LegalUnitGroundingContractTest(unittest.TestCase):
    def test_runtime_loadable_legal_units_have_direct_grounding(self) -> None:
        text_span_ids = {row["text_span_id"] for row in read_jsonl(FINAL / "page_text_spans.jsonl")}
        bbox_ids = {row["bbox_id"] for row in read_jsonl(FINAL / "bbox_registry.jsonl")}
        for row in read_jsonl(FINAL / "legal_units.jsonl"):
            if row.get("runtime_loadable") is not True:
                continue
            for field in (
                "source_document_id",
                "source_role",
                "temporal_context",
                "page_numbers",
                "text",
                "text_span_ids",
                "bbox_ids",
                "grounding_status",
                "validation_status",
            ):
                self.assertIn(field, row, row["legal_unit_id"])
            self.assertIn(
                row["grounding_status"],
                {"text_span_exact", "text_span_aggregate_from_evidence"},
                row["legal_unit_id"],
            )
            self.assertTrue(set(row["text_span_ids"]) <= text_span_ids, row["legal_unit_id"])
            self.assertTrue(set(row["bbox_ids"]) <= bbox_ids, row["legal_unit_id"])
            self.assertTrue(row["text_span_ids"], row["legal_unit_id"])
            self.assertTrue(row["bbox_ids"], row["legal_unit_id"])

    def test_fixture_legal_unit_keeps_stable_grounding_ids(self) -> None:
        rows = {row["legal_unit_id"]: row for row in read_jsonl(FINAL / "legal_units.jsonl")}
        for case in _legal_unit_grounding_cases():
            row = rows[case["legal_unit_id"]]
            for field in (
                "unit_label",
                "source_document_id",
                "grounding_status",
                "validation_status",
                "text_span_ids",
                "bbox_ids",
            ):
                self.assertEqual(row[field], case[field], case["legal_unit_id"])

    def test_stage3_legal_unit_chunk_span_closure_health_is_complete(self) -> None:
        health = read_json(FINAL / "validation_report.json")["legal_unit_chunk_span_closure_health"]
        self.assertEqual(health["legal_unit_count"], 651)
        self.assertEqual(health["chunk_count"], 651)
        non_error_count_keys = {
            "legal_unit_count",
            "chunk_count",
            "legal_unit_exact_span_link_count",
            "chunk_exact_span_link_count",
            "legal_unit_containing_span_link_count",
            "chunk_containing_span_link_count",
        }
        for key, value in health.items():
            if key.endswith("_count") and key not in non_error_count_keys:
                self.assertEqual(value, 0, key)
        self.assertEqual(health["active_legal_units_without_span_ids"], 0)
        self.assertEqual(health["active_chunks_without_span_ids"], 0)
        self.assertEqual(health["source_text_backed_legal_units_without_span_ids_count"], 0)
        self.assertEqual(health["source_text_backed_chunks_without_span_ids_count"], 0)
        self.assertEqual(health["legal_unit_containing_span_link_count"], 3)
        self.assertEqual(health["chunk_containing_span_link_count"], 3)
        self.assertNotIn("reviewed_nonruntime_canonical_chunks_without_span_ids", health)
        self.assertEqual(health["status"], "complete")

    def test_effective_clause_rows_are_source_span_linked(self) -> None:
        units = {row["legal_unit_id"]: row for row in read_jsonl(FINAL / "legal_units.jsonl")}
        chunks = {row["chunk_id"]: row for row in read_jsonl(FINAL / "chunks.jsonl")}
        spans = {row["text_span_id"]: row for row in read_jsonl(FINAL / "page_text_spans.jsonl")}
        cases = {
            "uud_chunk_00624": (
                "uud_legal_unit_00624",
                ["uud_text_span::amendment_1_historical::0003::0006"],
            ),
            "uud_chunk_00635": (
                "uud_legal_unit_00635",
                [
                    "uud_text_span::amendment_3_historical::0008::0031",
                    "uud_text_span::amendment_3_historical::0008::0032",
                ],
            ),
            "uud_chunk_00649": (
                "uud_legal_unit_00649",
                ["uud_text_span::amendment_4_historical::0006::0005"],
            ),
        }
        for chunk_id, (unit_id, span_ids) in cases.items():
            unit = units[unit_id]
            chunk = chunks[chunk_id]
            self.assertEqual(unit["text_span_ids"], span_ids, unit_id)
            self.assertEqual(chunk["text_span_ids"], span_ids, chunk_id)
            self.assertEqual(unit["grounding_status"], "text_span_containing_match", unit_id)
            self.assertEqual(chunk["grounding_status"], "text_span_containing_match", chunk_id)
            for span_id in span_ids:
                span = spans[span_id]
                self.assertEqual(span["source_document_id"], unit["source_document_id"], span_id)
                self.assertIn(span["page_number"], unit["page_numbers"], span_id)
                self.assertNotIn(span["promotion_status"], {"excluded_nonlegal", "needs_review"}, span_id)


def _legal_unit_grounding_cases() -> tuple[dict, ...]:
    path = ROOT / "tests/fixtures/uud/legal_unit_grounding_cases.jsonl"
    return tuple(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


if __name__ == "__main__":
    unittest.main()
