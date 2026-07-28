from __future__ import annotations

from pathlib import Path
import unittest

from tjipto.core.manifest import read_jsonl
from tjipto.ingestion.pdf.source_objects import TERMINAL_DISPOSITIONS


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


if __name__ == "__main__":
    unittest.main()
