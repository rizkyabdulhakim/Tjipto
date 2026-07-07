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
            self.assertEqual(row["grounding_status"], "text_span_exact", row["legal_unit_id"])
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
        for key, value in health.items():
            if key.endswith("_count") and key not in {"legal_unit_count", "chunk_count"}:
                self.assertEqual(value, 0, key)
        self.assertEqual(health["active_legal_units_without_span_ids"], 0)
        self.assertEqual(health["active_chunks_without_span_ids"], 0)
        self.assertEqual(health["reviewed_nonruntime_canonical_chunks_without_span_ids"], 3)
        self.assertEqual(health["status"], "complete")


def _legal_unit_grounding_cases() -> tuple[dict, ...]:
    path = ROOT / "tests/fixtures/uud/legal_unit_grounding_cases.jsonl"
    return tuple(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


if __name__ == "__main__":
    unittest.main()
