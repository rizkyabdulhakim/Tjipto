from __future__ import annotations

from pathlib import Path
import unittest

import fitz

from tjipto.core.manifest import read_jsonl
from tjipto.ingestion.pdf.bbox import extract_pdf
from tjipto.ingestion.pdf.source_objects import TERMINAL_DISPOSITIONS, build_source_object_inventory


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "data/final/uud"


def _expected(source_object: dict, spans: list[dict], raw_spans: list[dict]) -> tuple[str, str]:
    if source_object.get("extraction_error"):
        return "extraction_failed", "raw_pdf_extraction_failed"
    if source_object.get("pdf_block_type") != 0:
        return "unsupported_nontext_object", "nontext_pdf_block"
    if not spans:
        raw_text = "".join(str(row.get("raw_text") or "") for row in raw_spans) or str(
            source_object.get("_raw_character_text") or ""
        )
        if raw_text and raw_text.isspace():
            return "excluded_nonlegal_object", "whitespace_only_text_block"
        if raw_text and not raw_text.replace("\u00ad", ""):
            return "excluded_nonlegal_object", "decorative_soft_hyphen_block"
        if raw_spans and all(row.get("classification") == "source_annotation_marker" for row in raw_spans):
            return "source_annotation_object", "source_annotation_marker_block"
        return "needs_review", "unclassified_meaningful_text_block"
    statuses = {str(span.get("promotion_status") or "") for span in spans}
    roles = {str(span.get("span_role") or "") for span in spans}
    if statuses == {"promoted_legal_unit"}:
        disposition = "promoted_normative_evidence" if roles == {"normative_text"} else "promoted_structural_evidence"
        return disposition, "source_span_promotion"
    if statuses == {"promoted_metadata"}:
        return "promoted_metadata", "source_span_promotion"
    if statuses == {"promoted_source_conflict"}:
        return "promoted_source_anomaly", "source_span_promotion"
    if all(span.get("exclusion_reason") for span in spans):
        return "excluded_nonlegal_object", "all_source_spans_excluded"
    if all(status and status != "needs_review" for status in statuses):
        return "resolved_mixed_dispositions", "all_source_spans_have_terminal_dispositions"
    return "needs_review", "mixed_or_unresolved_source_span_dispositions"


class SourceObjectDispositionContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.published = read_jsonl(FINAL / "source_objects.jsonl")
        cls.spans = read_jsonl(FINAL / "page_text_spans.jsonl")
        cls.raw_spans = read_jsonl(FINAL / "raw_source_spans.jsonl")
        cls.sources = {row["source_document_id"]: row for row in read_jsonl(FINAL / "source_documents.jsonl")}
        objects = []
        for source_id, source in cls.sources.items():
            with fitz.open(ROOT / source["path"]) as document:
                objects.extend(extract_pdf(document, source_id).source_objects)
        cls.raw_objects = tuple(objects)

    def test_every_published_pdf_object_has_one_terminal_disposition(self) -> None:
        rows = self.published
        self.assertTrue(rows)
        self.assertEqual(len(rows), len({row["source_object_id"] for row in rows}))
        self.assertTrue(all(row["disposition"] in TERMINAL_DISPOSITIONS for row in rows))
        self.assertTrue(all(row["source_sha256"] and row["payload_sha256"] for row in rows))
        self.assertTrue(all(row["object_role"] == "source_object" for row in rows))

    def test_current_inventory_matches_independent_row_derivation(self) -> None:
        built = build_source_object_inventory(
            source_objects=self.raw_objects,
            page_text_spans=self.spans,
            raw_source_spans=self.raw_spans,
            source_documents=self.sources,
        )
        self.assertEqual(
            {row["source_object_id"] for row in built},
            {row["source_object_id"] for row in self.published},
        )
        spans_by_object: dict[str, list[dict]] = {}
        for span in self.spans:
            spans_by_object.setdefault(str(span["source_object_id"]), []).append(span)
        raw_by_object: dict[tuple[str, int, int], list[dict]] = {}
        for row in self.raw_spans:
            raw_by_object.setdefault((row["source_document_id"], row["page_number"], row["block_index"]), []).append(row)
        expected = {
            row["source_object_id"]: _expected(
                row,
                spans_by_object.get(row["source_object_id"], []),
                raw_by_object.get((row["source_document_id"], row["page_number"], row["block_index"]), []),
            )
            for row in self.raw_objects
        }
        actual = {row["source_object_id"]: (row["disposition"], row["reason"]) for row in self.published}
        self.assertEqual(actual, expected)

    def test_public_builder_covers_terminal_and_review_cases(self) -> None:
        cases = (
            ("mixed", "", "resolved_mixed_dispositions", "all_source_spans_have_terminal_dispositions"),
            ("whitespace", " \n\t", "excluded_nonlegal_object", "whitespace_only_text_block"),
            ("soft-hyphen", "\u00ad\u00ad", "excluded_nonlegal_object", "decorative_soft_hyphen_block"),
            ("marker", "***)", "source_annotation_object", "source_annotation_marker_block"),
            ("unknown", "unknown meaningful text", "needs_review", "unclassified_meaningful_text_block"),
        )
        objects = tuple(
            {
                "source_object_id": f"synthetic::{name}", "source_document_id": "synthetic", "page_number": 1,
                "block_index": index, "pdf_block_type": 0, "payload_sha256": "a" * 64, "source_line_refs": (),
                "_raw_character_text": raw_text,
            }
            for index, (name, raw_text, _, _) in enumerate(cases)
        )
        spans = [
            {
                "text_span_id": f"synthetic-span::{index}", "source_object_id": "synthetic::mixed",
                "promotion_status": status, "span_role": "normative_text",
            }
            for index, status in enumerate(("promoted_legal_unit", "promoted_metadata"))
        ]
        raw_spans = [{
            "source_document_id": "synthetic", "page_number": 1, "block_index": 3,
            "raw_text": "***)", "classification": "source_annotation_marker",
        }]
        rows = build_source_object_inventory(
            source_objects=objects,
            page_text_spans=spans,
            raw_source_spans=raw_spans,
            source_documents={"synthetic": {"sha256": "b" * 64, "path": "synthetic.pdf"}},
        )
        actual = {row["source_object_id"].removeprefix("synthetic::"): (row["disposition"], row["reason"]) for row in rows}
        expected = {name: (disposition, reason) for name, _, disposition, reason in cases}
        self.assertEqual(actual, expected)

    def test_object_and_line_order_is_deterministic(self) -> None:
        self.assertEqual(self.published, sorted(self.published, key=lambda row: row["source_object_id"]))
        for row in self.published:
            self.assertEqual(row["source_line_refs"], sorted(row["source_line_refs"]))


if __name__ == "__main__":
    unittest.main()
