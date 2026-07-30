from __future__ import annotations

import json
from pathlib import Path
import unittest

from tjipto.evidence.store import EvidenceStore
from tjipto.corpora.uud.policy.source_text import validate_source_text_closure
from tjipto.runtime.api import handle_request
from tjipto.runtime.service import LegalRuntimeService
from tjipto.runtime.source_text import source_text_health, source_text_record


ROOT = Path(__file__).resolve().parents[1]


class SourceTextReachabilityContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.service = LegalRuntimeService(ROOT)
        cls.store = cls.service._store("uud")
        assert cls.store is not None

    @classmethod
    def tearDownClass(cls) -> None:
        EvidenceStore.clear_shared_cache()

    def test_every_nonempty_raw_span_has_typed_route_or_reviewed_abstention(self) -> None:
        records = []
        with self.store.config.artifact_path("raw_source_spans").open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if str(row.get("raw_text") or "").strip():
                    records.append(source_text_record(row))
        self.assertGreater(len(records), 0)
        self.assertFalse([record for record in records if not record.capabilities and not record.abstention_reason])

    def test_annotation_health_is_closed_without_promoting_markers(self) -> None:
        health = source_text_health(self.store)
        with self.store.config.artifact_path("raw_source_spans").open(encoding="utf-8") as handle:
            expected = sum(1 for line in handle if str(json.loads(line).get("raw_text") or "").strip())
        self.assertEqual(health["raw_nonempty_source_span_count"], expected)
        self.assertEqual(health["meaningful_source_span_without_route_count"], 0)
        self.assertEqual(health["unmapped_source_annotation_count"], 0)
        self.assertEqual(health["ordinary_punctuation_annotation_count"], 0)
        self.assertEqual(health["source_annotation_legal_citation_count"], 0)
        self.assertEqual(health["source_annotation_default_highlight_count"], 0)

    def test_required_marker_queries_are_legend_grounded_source_answers(self) -> None:
        cases = {
            "Apa perbedaan *, **, ***, dan **** dalam naskah UUD?": ("Perubahan Pertama", 4),
            "Apa arti tanda ** pada UUD?": ("Perubahan Kedua", 1),
            "Mengapa Pasal 36A diberi tanda **?": ("Perubahan Kedua", 1),
            "Pasal mana yang berasal dari Perubahan Ketiga?": ("Pasal 1", 1),
            "Apa arti marker gabungan pada sumber ini?": ("Perubahan Keempat", 2),
        }
        for query, (expected, support_count) in cases.items():
            with self.subTest(query=query):
                response = handle_request("uud", "ask", {"query": query}, ROOT, self.service)
                self.assertEqual(response["kind"], "answer")
                self.assertEqual(response["status"], "answer_ready")
                self.assertIn(expected, response["answer"])
                self.assertEqual(len(response["supports"]), support_count)
                self.assertTrue(all(row["authority_kind"] == "source_annotation" for row in response["supports"]))
                self.assertTrue(all(row["citation_final"] is False for row in response["supports"]))
                self.assertTrue(all(row["viewer_target"]["can_resolve"] is True for row in response["supports"]))

    def test_normal_article_quote_excludes_marker(self) -> None:
        response = handle_request("uud", "ask", {"query": "Apa isi Pasal 36A?"}, ROOT, self.service)
        self.assertEqual(response["kind"], "answer")
        self.assertTrue(response["supports"])
        self.assertNotIn("**)", response["supports"][0]["text"])

    def test_validator_rejects_unmapped_marker_and_colon_promotion(self) -> None:
        rows = []
        with self.store.config.artifact_path("raw_source_spans").open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        self.assertEqual(validate_source_text_closure(rows), ())
        colon = next(row for row in rows if row.get("raw_text") == ":")
        mutated = rows + [colon | {"classification": "source_annotation_marker", "raw_source_span_id": "mutation::colon"}]
        errors = validate_source_text_closure(mutated)
        self.assertIn("ordinary_punctuation_annotation:mutation::colon", errors)
        marker = next(row for row in rows if row.get("classification") == "source_annotation_marker")
        errors = validate_source_text_closure(
            rows + [marker | {"raw_text": "*****)", "raw_source_span_id": "mutation::unknown"}]
        )
        self.assertIn("source_annotation_unmapped:mutation::unknown", errors)


if __name__ == "__main__":
    unittest.main()
