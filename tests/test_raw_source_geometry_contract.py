from __future__ import annotations

import json
from pathlib import Path
import unittest

import fitz

from tjipto.ingestion.pdf.bbox import extract_pdf


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "data/final/uud"


class RawSourceGeometryContractTest(unittest.TestCase):
    def test_every_segment_reextracts_from_rawdict_character_lineage(self) -> None:
        sources = {
            row["source_document_id"]: row
            for row in map(json.loads, (FINAL / "source_documents.jsonl").read_text(encoding="utf-8").splitlines())
        }
        rows = [json.loads(line) for line in (FINAL / "raw_source_spans.jsonl").read_text(encoding="utf-8").splitlines()]
        line_cache: dict[tuple[str, int], list[dict]] = {}
        documents = {source_id: fitz.open(ROOT / source["path"]) for source_id, source in sources.items()}
        try:
            for source_id, document in documents.items():
                line_cache.update({(source_id, page): lines for page, lines in extract_pdf(document, source_id).lines.items()})
            for row in rows:
                lines = line_cache[(row["source_document_id"], row["page_number"])]
                line = next(item for item in lines if item["block_index"] == row["block_index"] and item["line_index"] == row["line_index"])
                selected = [
                    character
                    for character in line["characters"]
                    if character["character_id"] in set(row["character_ids"])
                ]
                self.assertEqual("".join(character["text"] for character in selected), row["raw_text"], row["raw_source_span_id"])
                self.assertEqual(row["character_ids"], [character["character_id"] for character in selected], row["raw_source_span_id"])
                union = (
                    min(character["x0"] for character in selected),
                    min(character["y0"] for character in selected),
                    max(character["x1"] for character in selected),
                    max(character["y1"] for character in selected),
                )
                for actual, stored in zip(union, (row[field] for field in ("x0", "y0", "x1", "y1"))):
                    self.assertAlmostEqual(actual, stored, places=5, msg=row["raw_source_span_id"])
                self.assertEqual(row["raw_geometry_method"], "pdf_rawdict_character_bbox")
        finally:
            for document in documents.values():
                document.close()


if __name__ == "__main__":
    unittest.main()
