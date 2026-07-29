from __future__ import annotations

from pathlib import Path
from hashlib import sha256
import unittest

from tjipto.core.manifest import read_json, read_jsonl
from tjipto.corpora.uud.proposition_builder import source_marker_character_boxes
from tjipto.evidence.bbox import positive_area_intersection, viewer_overlay_rectangles
from tjipto.ingestion.pdf.text_spans import build_pdf_text_spans


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "data/final/uud"


class PdfTextSpanCoverageTest(unittest.TestCase):
    def test_page_text_spans_cover_all_source_documents(self) -> None:
        source_ids = {row["source_document_id"] for row in read_jsonl(FINAL / "source_documents.jsonl")}
        spans = read_jsonl(FINAL / "page_text_spans.jsonl")
        self.assertTrue(spans)
        self.assertEqual({row["source_document_id"] for row in spans}, source_ids)
        for row in spans:
            self.assertEqual(row["status"], "accepted_text_span")
            self.assertTrue(row["text"].strip())
            self.assertEqual(row["bbox_precision"], "exact")
            self.assertIsInstance(row["viewer_highlightable"], bool)
            self.assertEqual(row["object_role"], "source_span")
            self.assertNotIn("highlightable", row)

    def test_text_span_builder_uses_source_metadata_not_id_format(self) -> None:
        rows = build_pdf_text_spans(
            source_documents={
                "synthetic-source": {
                    "filename": "synthetic.pdf",
                    "path": "data/sources/synthetic.pdf",
                    "source_role": "synthetic_role",
                    "temporal_context": "synthetic_time",
                    "sha256": "abc123",
                }
            },
            pdf_lines={"synthetic-source": {1: [{"text": "hello", "x0": 1, "x1": 2, "y0": 3, "y1": 4}]}},
            corpus_id="synthetic",
            text_span_id_prefix="span",
        )
        self.assertEqual(rows[0]["source_document_id"], "synthetic-source")
        self.assertEqual(rows[0]["source_role"], "synthetic_role")
        self.assertEqual(rows[0]["temporal_context"], "synthetic_time")
        self.assertEqual(rows[0]["text_span_id"], "span::synthetic_role::0001::0000")

    def test_selectors_reconstruct_the_canonical_page_stream(self) -> None:
        spans = read_jsonl(FINAL / "page_text_spans.jsonl")
        streams: dict[str, list[dict]] = {}
        for row in spans:
            streams.setdefault(row["stream_id"], []).append(row)
        for stream_id, rows in streams.items():
            rows.sort(key=lambda row: row["text_start"])
            stream = "\n".join(row["text"] for row in rows)
            self.assertEqual(sha256(stream.encode("utf-8")).hexdigest(), rows[0]["page_text_hash"], stream_id)
            for row in rows:
                self.assertEqual(stream[row["text_start"] : row["text_end"]], row["exact_quote"], row["text_span_id"])

    def test_source_markers_have_raw_disposition_and_never_enter_semantic_text(self) -> None:
        markers = {"*)", "**)", "***)", "****)", "***/****)"}
        raw = read_jsonl(FINAL / "raw_source_spans.jsonl")
        marker_rows = [row for row in raw if row["classification"] == "source_annotation_marker"]
        self.assertEqual({row["raw_text"] for row in marker_rows}, markers)
        self.assertTrue(all(row["disposition_reason"] == "source_annotation_marker" for row in marker_rows))
        self.assertTrue(all(row["legal_text"] is False for row in marker_rows))
        self.assertTrue(all(row["citation_eligible"] is False for row in marker_rows))
        self.assertTrue(all(row["relevant_quote_eligible"] is False for row in marker_rows))
        self.assertTrue(all(row["default_highlight_eligible"] is False for row in marker_rows))
        for name, field in (("page_text_spans", "text"), ("legal_units", "text"), ("evidence_registry", "quoted_text")):
            rows = read_jsonl(FINAL / f"{name}.jsonl")
            self.assertFalse(any(marker in str(row.get(field) or "") for row in rows for marker in markers), name)

    def test_nonempty_semantic_source_segments_have_exact_support_selectors(self) -> None:
        rows = read_jsonl(FINAL / "raw_source_spans.jsonl")
        semantic = [row for row in rows if row.get("semantic_text")]
        self.assertTrue(semantic)
        self.assertTrue(all(row.get("source_support_id") for row in semantic))
        self.assertTrue(all(row["semantic_text"] == row["semantic_exact_quote"] for row in semantic))
        self.assertTrue(all(row["semantic_text_start"] < row["semantic_text_end"] for row in semantic))
        streams: dict[str, list[dict]] = {}
        for row in semantic:
            streams.setdefault(row["semantic_stream_id"], []).append(row)
        for stream_id, stream_rows in streams.items():
            stream_rows.sort(key=lambda row: row["semantic_text_start"])
            stream = "\n".join(row["semantic_text"] for row in stream_rows)
            self.assertEqual(sha256(stream.encode("utf-8")).hexdigest(), stream_rows[0]["semantic_stream_sha256"], stream_id)
            for row in stream_rows:
                self.assertEqual(stream[row["semantic_text_start"] : row["semantic_text_end"]], row["semantic_exact_quote"])
        markers = [row for row in rows if row["classification"] == "source_annotation_marker"]
        self.assertTrue(all(not row.get("semantic_text") for row in markers))

    def test_proposition_selectors_use_trimmed_character_geometry(self) -> None:
        spans = {row["text_span_id"]: row for row in read_jsonl(FINAL / "page_text_spans.jsonl")}
        characters = {
            character["character_bbox_id"]: word | character
            for word in read_jsonl(FINAL / "word_bboxes.jsonl")
            for character in word.get("characters") or ()
        }
        propositions = read_jsonl(FINAL / "propositions.jsonl")
        self.assertTrue(propositions)
        for proposition in propositions:
            quotes = []
            selected_ids = []
            for selector in proposition["source_selectors"]:
                span = spans[selector["text_span_id"]]
                start, end = selector["start"], selector["end"]
                self.assertEqual(span["text"][start:end], span["text"][start:end].strip())
                self.assertEqual(selector["absolute_start"], span["text_start"] + start)
                self.assertEqual(selector["absolute_end"], span["text_start"] + end)
                self.assertFalse(set(selector["character_bbox_ids"]) & set(span["span_bbox_ids"]))
                self.assertTrue(set(selector["character_bbox_ids"]) <= characters.keys())
                quotes.append(span["text"][start:end])
                selected_ids.extend(selector["character_bbox_ids"])
            self.assertEqual(proposition["exact_quote"], "\n".join(quotes))
            self.assertEqual(proposition["bbox_refs"], list(dict.fromkeys(selected_ids)))
            overlay = proposition["viewer_overlay"]
            self.assertEqual(overlay["status"], "complete")
            self.assertEqual(overlay["proposition_id"], proposition["proposition_id"])
            self.assertEqual(overlay["source_document_id"], proposition["source_document_id"])
            self.assertEqual(overlay["source_sha256"], proposition["source_sha256"])
            self.assertEqual(overlay["selector_field"], "source_selectors")
            self.assertEqual(overlay["selected_character_field"], "bbox_refs")
            self.assertEqual(
                {
                    character_id
                    for row in viewer_overlay_rectangles(proposition, characters)
                    for character_id in row["character_bbox_ids"]
                },
                set(proposition["bbox_refs"]),
            )

    def test_pasal_7c_viewer_overlay_clips_marker_overlap_without_changing_raw_geometry(self) -> None:
        word_bboxes = read_jsonl(FINAL / "word_bboxes.jsonl")
        characters = {
            character["character_bbox_id"]: word | character
            for word in word_bboxes
            for character in word.get("characters") or ()
        }
        proposition = next(
            row
            for row in read_jsonl(FINAL / "propositions.jsonl")
            if row["source_document_id"] == "uud::current_consolidated"
            and row["page_numbers"] == [7]
            and row["exact_quote"].startswith("Pasal 7C\n")
        )
        selected_period = next(
            characters[character_id]
            for character_id in proposition["bbox_refs"]
            if characters[character_id].get("text") == "."
        )
        marker = next(
            row
            for row in source_marker_character_boxes(word_bboxes)
            if row["source_document_id"] == proposition["source_document_id"]
            and row["page_number"] == 7
            and positive_area_intersection(selected_period, row)
        )
        self.assertEqual(marker["text"], "*")
        period_overlay = [
            row
            for row in viewer_overlay_rectangles(proposition, characters)
            if selected_period["character_bbox_id"] in row["character_bbox_ids"]
        ]
        self.assertTrue(period_overlay)
        self.assertFalse(any(positive_area_intersection(row, marker) for row in period_overlay))

    def test_selector_health_reports_text_and_viewer_geometry_separately(self) -> None:
        health = read_json(FINAL / "validation_report.json")["selector_geometry_health"]
        self.assertNotIn("marker_highlight_intersection_count", health)
        for counter in (
            "selector_round_trip_mismatch_count",
            "absolute_selector_mismatch_count",
            "unknown_selected_character_id_count",
            "marker_text_in_exact_quote_count",
            "marker_viewer_geometry_intersection_count",
            "viewer_geometry_without_exact_selector_lineage_count",
        ):
            self.assertEqual(health[counter], 0, counter)


if __name__ == "__main__":
    unittest.main()
