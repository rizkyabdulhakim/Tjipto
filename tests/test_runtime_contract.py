from __future__ import annotations

from pathlib import Path
import unittest

from tjipto.runtime.service import LegalRuntimeService


ROOT = Path(__file__).resolve().parents[1]


class RuntimeContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = LegalRuntimeService(ROOT)

    def test_search_citation_and_viewer_work(self) -> None:
        search = self.service.search("uud", "negara hukum", limit=3)
        self.assertEqual(search["status"], "found")
        self.assertTrue(search["matches"])

        citation = self.service.citation("uud", "Pasal 1 ayat (3)")
        self.assertEqual(citation["status"], "found")
        evidence = citation["matches"][0]
        self.assertEqual(evidence["source_role"], "current_consolidated")

        viewer = self.service.viewer("uud", evidence["evidence_id"])
        self.assertEqual(viewer["status"], "viewer_payload_ready")
        self.assertTrue(viewer["page_numbers"])
        self.assertGreater(viewer["bbox_count"], 0)
        self.assertTrue(viewer["bbox_rectangles"])

    def test_unsupported_corpus_fails_safely(self) -> None:
        self.assertEqual(
            self.service.search("unknown", "Pasal 1")["status"],
            "unsupported_corpus",
        )


if __name__ == "__main__":
    unittest.main()
