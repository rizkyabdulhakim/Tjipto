from __future__ import annotations

from pathlib import Path
import unittest

from tjipto.core.manifest import read_jsonl
from tjipto.ingestion.pdf.source_objects import TERMINAL_DISPOSITIONS, _disposition


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "data/final/uud"


class SourceObjectDispositionContractTest(unittest.TestCase):
    def test_every_published_pdf_object_has_one_terminal_disposition(self) -> None:
        rows = read_jsonl(FINAL / "source_objects.jsonl")
        self.assertTrue(rows)
        self.assertEqual(len(rows), len({row["source_object_id"] for row in rows}))
        self.assertTrue(all(row["disposition"] in TERMINAL_DISPOSITIONS for row in rows))
        self.assertTrue(all(row["source_sha256"] and row["payload_sha256"] for row in rows))
        self.assertTrue(all(row["object_role"] == "source_object" for row in rows))

    def test_current_resolvable_objects_have_typed_terminal_dispositions(self) -> None:
        rows = read_jsonl(FINAL / "source_objects.jsonl")
        reasons = {row["reason"]: row["disposition"] for row in rows}
        self.assertFalse(any(row["disposition"] == "needs_review" for row in rows))
        self.assertEqual(sum(row["reason"] == "all_source_spans_have_terminal_dispositions" for row in rows), 26)
        self.assertEqual(sum(row["reason"] == "whitespace_only_text_block" for row in rows), 5)
        self.assertEqual(sum(row["reason"] == "decorative_soft_hyphen_block" for row in rows), 2)
        self.assertEqual(sum(row["reason"] == "source_annotation_marker_block" for row in rows), 1)
        self.assertEqual(reasons["source_annotation_marker_block"], "source_annotation_object")

    def test_unknown_meaningful_text_remains_explicit_review(self) -> None:
        source_object = {"pdf_block_type": 0, "_raw_character_text": "unknown meaningful text"}
        self.assertEqual(
            _disposition(source_object, [], []),
            ("needs_review", "unclassified_meaningful_text_block"),
        )

    def test_object_and_line_order_is_deterministic(self) -> None:
        rows = read_jsonl(FINAL / "source_objects.jsonl")
        self.assertEqual(rows, sorted(rows, key=lambda row: row["source_object_id"]))
        for row in rows:
            self.assertEqual(row["source_line_refs"], sorted(row["source_line_refs"]))


if __name__ == "__main__":
    unittest.main()
