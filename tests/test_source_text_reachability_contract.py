from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import unittest

from tjipto.evidence.store import EvidenceStore
from tjipto.corpora.uud.policy.source_text import project_source_text_rows, validate_source_text_closure
from tjipto.corpora.uud.source_annotations import source_annotation_occurrences
from tjipto.runtime.api import handle_request
from tjipto.runtime.service import LegalRuntimeService
from tjipto.runtime.source_text import source_text_health, source_text_record
from scripts.evaluate_source_text_reachability import evaluate_projection


ROOT = Path(__file__).resolve().parents[1]


class SourceTextReachabilityContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.service = LegalRuntimeService(ROOT)
        cls.store = cls.service._store("uud")
        assert cls.store is not None
        cls.raw_rows = [
            json.loads(line)
            for line in cls.store.config.artifact_path("raw_source_spans").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        cls.semantic_rows = [
            json.loads(line)
            for line in cls.store.config.artifact_path("page_text_spans").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    @classmethod
    def tearDownClass(cls) -> None:
        EvidenceStore.clear_shared_cache()

    def test_every_nonempty_raw_span_has_typed_route_or_reviewed_abstention(self) -> None:
        records = [source_text_record(row) for row in self.store.raw_source_spans]
        self.assertEqual(len(records), 2370)
        self.assertFalse([record for record in records if not record.capabilities and not record.abstention_reason])
        self.assertEqual(sum(record.semantic_join_status == "exact" for record in records), 2070)
        self.assertEqual(sum(record.semantic_join_status == "not_applicable" for record in records), 300)
        self.assertEqual(
            sum(
                record.semantic_classification != "normative_constitutional_text"
                and (record.legal_answer_eligible or record.legal_citation_eligible or record.default_highlight_eligible)
                for record in records
            ),
            0,
        )
        self.assertFalse(
            [
                record
                for record in records
                if record.legal_force == "historical_normative"
                and (record.legal_answer_eligible or record.legal_citation_eligible or record.default_highlight_eligible)
            ]
        )

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
        self.assertEqual(health["semantic_join_missing_count"], 0)
        self.assertEqual(health["semantic_join_duplicate_count"], 0)
        self.assertEqual(health["source_annotation_occurrence_without_selector_or_geometry_count"], 0)
        self.assertEqual(health["source_annotation_occurrence_without_target_or_reason_count"], 0)
        self.assertEqual(health["fabricated_annotation_target_count"], 0)

    def test_marker_occurrences_preserve_identity_and_target_or_reason(self) -> None:
        occurrences = source_annotation_occurrences(self.store)
        self.assertEqual(len(occurrences), 298)
        self.assertEqual(len({row.occurrence_id for row in occurrences}), 298)
        self.assertEqual(sum(row.target_reason == "legend_definition" for row in occurrences), 4)
        self.assertEqual(sum(row.target_reason == "ambiguous_target" for row in occurrences), 294)
        self.assertEqual(sum("/" in row.marker for row in occurrences), 2)
        self.assertTrue(all(row.selector.stream_id and len(row.geometry) == 4 for row in occurrences))
        self.assertTrue(all(bool(row.target_legal_unit_id) != bool(row.target_reason) for row in occurrences))

    def test_required_marker_queries_are_legend_grounded_source_answers(self) -> None:
        cases = {
            "Apa perbedaan *, **, ***, dan **** dalam naskah UUD?": ("Perubahan Pertama", 4),
            "Apa arti tanda ** pada UUD?": ("Perubahan Kedua", 1),
            "Mengapa Pasal 36A diberi tanda **?": ("Perubahan Kedua", 1),
            "Pasal mana yang berasal dari Perubahan Ketiga?": ("Perubahan Ketiga", 1),
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
                if "Pasal" in query:
                    self.assertIn("tidak dapat dipastikan", response["answer"])

    def test_normal_article_quote_excludes_marker(self) -> None:
        response = handle_request("uud", "ask", {"query": "Apa isi Pasal 36A?"}, ROOT, self.service)
        self.assertEqual(response["kind"], "answer")
        self.assertTrue(response["supports"])
        self.assertNotIn("**)", response["supports"][0]["text"])

    def test_validator_rejects_unmapped_marker_and_colon_promotion(self) -> None:
        rows = []
        with self.store.config.artifact_path("raw_source_spans").open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if (
                    row.get("source_role") == "current_consolidated"
                    and row.get("page_number") == 1
                    and int(row.get("extraction_order") or 0) <= 11
                ):
                    rows.append(row)
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

    def test_row_level_evaluator_rejects_unsafe_mutations(self) -> None:
        baseline = [dict(row) for row in self.store.raw_source_spans]
        self.assertEqual(evaluate_projection(self.raw_rows, self.semantic_rows, baseline)["status"], "PASS")
        mutations = []

        structural = next(index for index, row in enumerate(baseline) if row.get("disposition") == "structural_text")
        mutations.append(("structural_current_norm", structural, {"disposition": "legal_text", "legal_force": "canonical_normative", "legal_answer_eligible": True}))
        instrument = next(index for index, row in enumerate(baseline) if row.get("disposition") == "instrument_text")
        mutations.append(("instrument_current_norm", instrument, {"disposition": "legal_text", "legal_force": "canonical_normative", "legal_answer_eligible": True}))
        metadata = next(index for index, row in enumerate(baseline) if row.get("disposition") == "source_fact")
        mutations.append(("metadata_final_citation", metadata, {"legal_citation_eligible": True}))
        marker = next(index for index, row in enumerate(baseline) if row.get("disposition") == "source_annotation")
        mutations.append(("marker_citation_highlight", marker, {"legal_citation_eligible": True, "default_highlight_eligible": True}))
        historical = next(index for index, row in enumerate(baseline) if row.get("legal_force") == "historical_normative")
        mutations.append(("historical_current", historical, {"legal_force": "canonical_normative", "legal_answer_eligible": True}))
        mutations.append(("temporal_role", historical, {"source_role": "current_consolidated", "temporal_context": "current_consolidated"}))
        mutations.append(("fabricated_target", marker, {"target_legal_unit_id": "uud_legal_unit_00001", "annotation_target_basis": "adjacency"}))

        for name, index, mutation in mutations:
            with self.subTest(name=name):
                rows = [dict(row) for row in baseline]
                rows[index].update(mutation)
                self.assertEqual(evaluate_projection(self.raw_rows, self.semantic_rows, rows)["status"], "FAIL")

    def test_evaluator_rejects_missing_and_duplicate_semantic_joins(self) -> None:
        joined = next(row for row in self.raw_rows if str(row.get("semantic_text") or "").strip())
        key = (
            joined["source_document_id"],
            joined["page_number"],
            joined["semantic_text_start"],
            joined["semantic_text_end"],
        )
        target = next(
            row
            for row in self.semantic_rows
            if (row["source_document_id"], row["page_number"], row["text_start"], row["text_end"]) == key
        )
        missing_semantics = [row for row in self.semantic_rows if row is not target]
        missing_projection = project_source_text_rows(self.raw_rows, missing_semantics)
        missing = evaluate_projection(self.raw_rows, missing_semantics, missing_projection)
        self.assertEqual(missing["status"], "FAIL")
        self.assertEqual(missing["semantic_join_missing_count"], 1)

        duplicated_semantics = self.semantic_rows + [deepcopy(target)]
        duplicate_projection = project_source_text_rows(self.raw_rows, duplicated_semantics)
        duplicate = evaluate_projection(self.raw_rows, duplicated_semantics, duplicate_projection)
        self.assertEqual(duplicate["status"], "FAIL")
        self.assertEqual(duplicate["semantic_join_duplicate_count"], 1)

    def test_generic_source_text_contract_has_no_uud_vocabulary(self) -> None:
        generic = (ROOT / "src" / "tjipto" / "contracts" / "source_text.py").read_text(encoding="utf-8")
        runtime = (ROOT / "src" / "tjipto" / "runtime" / "source_text.py").read_text(encoding="utf-8")
        for term in ("normative_constitutional_text", "amendment_instrument_text", "current_consolidated"):
            self.assertNotIn(term, generic)
            self.assertNotIn(term, runtime)


if __name__ == "__main__":
    unittest.main()
