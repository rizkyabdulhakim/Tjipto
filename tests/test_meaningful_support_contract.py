from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import unittest
import unicodedata

from scripts.evaluate_meaningful_support import evaluate_rows, load_artifacts
from tjipto.corpora.uud.meaningful_support_builder import build_meaningful_support_units


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "data" / "final" / "uud"
ORACLE_PATH = ROOT / "tests" / "fixtures" / "uud" / "meaningful_support_oracle.json"
REVIEWS_PATH = ROOT / "data" / "review" / "uud" / "meaningful_support_review_decisions.json"


def _rows(name: str) -> list[dict]:
    return [json.loads(line) for line in (FINAL / name).read_text(encoding="utf-8").splitlines() if line]


def _visible(value: str) -> str:
    return "".join(
        character for character in unicodedata.normalize("NFKC", value).replace("\u00ad", "")
        if not character.isspace()
    )


class MeaningfulSupportContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = _rows("meaningful_support_units.jsonl")
        cls.artifacts = load_artifacts(FINAL)
        cls.oracle = json.loads(ORACLE_PATH.read_text(encoding="utf-8"))
        cls.reviews = json.loads(REVIEWS_PATH.read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls) -> None:
        del cls.rows, cls.artifacts, cls.oracle, cls.reviews

    def _build(self, artifacts: dict[str, list[dict]] | None = None) -> list[dict]:
        values = artifacts or self.artifacts
        return build_meaningful_support_units(
            page_text_spans=values["spans"],
            raw_source_spans=values["raw"],
            evidence=values["evidence"],
            metadata_grounding=values["metadata"],
            source_conflicts=values["conflicts"],
            bbox_registry=values["bboxes"],
            word_bboxes=values["words"],
            review_decisions=self.reviews,
        )

    def test_projection_is_byte_deterministic_and_matches_artifact(self) -> None:
        first = self._build()
        second = self._build()
        def encode(rows: list[dict]) -> bytes:
            return "".join(
                json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows
            ).encode()

        self.assertEqual(encode(first), encode(second))
        self.assertEqual(first, self.rows)

    def test_independent_evaluator_accepts_owner_geometry_and_support_universe(self) -> None:
        report = evaluate_rows(self.rows, self.artifacts, self.oracle)
        self.assertEqual(report["status"], "PASS", report)
        self.assertEqual(report["exclusion_span_count"], 1)
        self.assertEqual(report["decision_span_count"], report["support_span_count"] + 1)
        self.assertTrue(all(
            row["viewer_eligible"] and not row["highlight_eligible"]
            for row in self.rows if row["bbox_precision"] == "page_grounded_only"
        ))
        self.assertTrue(all(value == 0 for value in report["counters"].values()), report)

    def test_exact_geometry_is_segment_local_and_never_owner_wide(self) -> None:
        characters = {
            character["character_bbox_id"]: word | character
            for word in self.artifacts["words"]
            for character in word.get("characters") or ()
        }
        owners = {
            ("evidence_registry", row["evidence_id"]): row for row in self.artifacts["evidence"]
        }
        owners.update({("metadata_grounding", row["metadata_grounding_id"]): row for row in self.artifacts["metadata"]})
        owner_wide = 0
        for row in self.rows:
            if row["bbox_precision"] != "exact":
                continue
            self.assertTrue(set(row["bbox_refs"]) <= characters.keys(), row["support_unit_id"])
            owner = owners.get((row["owner_type"], row["owner_id"]))
            if owner and set(owner.get("text_span_ids") or ()) - set(row["text_span_ids"]):
                owner_wide += set(row["bbox_refs"]) == set(owner.get("bbox_refs") or owner.get("bbox_ids") or ())
        self.assertEqual(owner_wide, 0)

    def test_exact_character_refs_reconstruct_only_selected_quote(self) -> None:
        spans = {row["text_span_id"]: row for row in self.artifacts["spans"]}
        characters = {
            character["character_bbox_id"]: character
            for word in self.artifacts["words"]
            for character in word.get("characters") or ()
        }
        for row in self.rows:
            if row["bbox_precision"] != "exact":
                continue
            actual = "".join(characters[ref]["text"] for ref in row["bbox_refs"])
            expected = "".join(spans[span_id]["exact_quote"] for span_id in row["text_span_ids"])
            self.assertEqual(_visible(actual), _visible(expected), row["support_unit_id"])

    def test_marker_suffixes_are_not_selected(self) -> None:
        characters = {
            character["character_bbox_id"]: character
            for word in self.artifacts["words"]
            for character in word.get("characters") or ()
        }
        for suffix in ("9289abf66c0f03d8", "95cc3b98fa209081"):
            row = next(item for item in self.rows if item["support_unit_id"].endswith(suffix))
            selected = "".join(characters[ref]["text"] for ref in row["bbox_refs"])
            self.assertNotIn("*", selected)
            self.assertFalse(selected.endswith(")"))

    def test_heading_and_article_label_do_not_inherit_full_legal_unit_overlay(self) -> None:
        evidence = {row["evidence_id"]: row for row in self.artifacts["evidence"]}
        spans = {row["text_span_id"]: row for row in self.artifacts["spans"]}
        checked = 0
        for row in self.rows:
            if row["owner_type"] != "evidence_registry":
                continue
            owner = evidence[row["owner_id"]]
            selected = [spans[span_id] for span_id in row["text_span_ids"]]
            if not all(span["span_role"] == "structural_heading" or span["exact_quote"].startswith("Pasal") for span in selected):
                continue
            if set(owner.get("text_span_ids") or ()) - set(row["text_span_ids"]):
                checked += 1
                self.assertNotEqual(set(row["bbox_refs"]), set(owner.get("bbox_refs") or owner.get("bbox_ids") or ()))
        self.assertGreater(checked, 0)

    def test_layout_separator_is_a_nonpresentational_typed_exclusion(self) -> None:
        matches = [row for row in self.rows if row["support_kind"] == "layout_separator"]
        self.assertEqual(len(matches), 1)
        row = matches[0]
        self.assertEqual(row["text_span_ids"], ["uud_text_span::current_consolidated::0028::0018"])
        self.assertEqual(row["decision_kind"], "typed_exclusion")
        self.assertFalse(any(row[field] for field in (
            "answer_eligible", "citation_eligible", "viewer_eligible", "highlight_eligible", "citation_final"
        )))
        self.assertEqual(row["bbox_refs"], [])

    def test_owner_arbitration_rejects_broader_and_equal_incompatible_substitutions(self) -> None:
        canonical = chosen = broader = None
        for candidate in self.rows:
            if candidate["owner_type"] != "evidence_registry":
                continue
            candidate_owner = next(
                row for row in self.artifacts["evidence"] if row["evidence_id"] == candidate["owner_id"]
            )
            broad = next((
                row for row in self.artifacts["evidence"]
                if candidate["text_span_ids"][0] in (row.get("text_span_ids") or ())
                and row.get("source_document_id") == candidate["source_document_id"]
                and row.get("authority_kind") == candidate["authority_kind"]
                and len(row.get("text_span_ids") or ()) > len(candidate_owner.get("text_span_ids") or ())
            ), None)
            if broad:
                canonical, chosen, broader = candidate, candidate_owner, broad
                break
        self.assertIsNotNone(canonical)
        assert canonical is not None and chosen is not None and broader is not None
        rows = deepcopy(self.rows)
        target = next(row for row in rows if row["support_unit_id"] == canonical["support_unit_id"])
        target["owner_id"] = broader["evidence_id"]
        self.assertEqual(evaluate_rows(rows, self.artifacts, self.oracle)["status"], "FAIL")

        incompatible = next(
            row for row in self.artifacts["evidence"]
            if len(row.get("text_span_ids") or ()) == len(chosen.get("text_span_ids") or ())
            and row["source_document_id"] != canonical["source_document_id"]
        )
        rows = deepcopy(self.rows)
        target = next(row for row in rows if row["support_unit_id"] == canonical["support_unit_id"])
        target["owner_id"] = incompatible["evidence_id"]
        self.assertEqual(evaluate_rows(rows, self.artifacts, self.oracle)["status"], "FAIL")

    def test_missing_grounding_and_sibling_geometry_mutations_fail_closed(self) -> None:
        mutations = {
            "missing_owner": lambda row: row.update(owner_id="missing-owner"),
            "wrong_source": lambda row: row.update(source_document_id="uud::missing"),
            "missing_selector": lambda row: row["selector_refs"].__setitem__(0, "missing-selector"),
            "altered_quote": lambda row: row.update(quoted_text_sha256="0" * 64),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                rows = deepcopy(self.rows)
                target = next(row for row in rows if row["decision_kind"] == "canonical_owner_support")
                mutate(target)
                self.assertEqual(evaluate_rows(rows, self.artifacts, self.oracle)["status"], "FAIL")

        rows = deepcopy(self.rows)
        target = next(row for row in rows if row["bbox_precision"] == "exact" and len(row["text_span_ids"]) == 1)
        page = target["page_numbers"][0]
        selected = set(target["bbox_refs"])
        sibling_ref = next(
            ref for span in self.artifacts["spans"]
            if span["page_number"] == page and span["text_span_id"] not in target["text_span_ids"]
            for ref in span.get("span_bbox_ids") or () if ref not in selected
        )
        target["bbox_refs"].append(sibling_ref)
        self.assertEqual(evaluate_rows(rows, self.artifacts, self.oracle)["status"], "FAIL")

    def test_missing_overlay_preserves_normative_authority_and_page_viewer(self) -> None:
        target = next(
            row for row in self.rows
            if row["support_kind"] == "normative" and row["answer_eligible"] and row["citation_eligible"]
            and row["bbox_precision"] == "exact"
        )
        artifacts = {
            **self.artifacts,
            "spans": [dict(row) for row in self.artifacts["spans"]],
        }
        selected = set(target["text_span_ids"])
        pages = set(target["page_numbers"])
        source = target["source_document_id"]
        for span in artifacts["spans"]:
            if span["text_span_id"] in selected:
                span["span_bbox_ids"] = []
        artifacts["words"] = [
            row for row in artifacts["words"]
            if not (row["source_document_id"] == source and row["page_number"] in pages)
        ]
        rebuilt = self._build(artifacts)
        row = next(item for item in rebuilt if item["text_span_ids"] == target["text_span_ids"])
        self.assertTrue(row["answer_eligible"])
        self.assertTrue(row["citation_eligible"])
        self.assertTrue(row["viewer_eligible"])
        self.assertFalse(row["highlight_eligible"])
        self.assertEqual(row["bbox_refs"], [])
        self.assertEqual(row["bbox_precision"], "page_grounded_only")
        self.assertEqual(evaluate_rows(rebuilt, artifacts, self.oracle)["status"], "PASS")

    def test_upstream_span_geometry_injection_is_not_propagated_or_trusted(self) -> None:
        artifacts = {**self.artifacts, "spans": deepcopy(self.artifacts["spans"])}
        target = next(row for row in artifacts["spans"] if row.get("span_bbox_ids"))
        sibling = next(
            ref for row in artifacts["spans"]
            if row["page_number"] == target["page_number"] and row["text_span_id"] != target["text_span_id"]
            for ref in row.get("span_bbox_ids") or () if ref not in target["span_bbox_ids"]
        )
        target["span_bbox_ids"].append(sibling)
        rebuilt = self._build(artifacts)
        self.assertEqual(rebuilt, self.rows)
        self.assertEqual(evaluate_rows(rebuilt, artifacts, self.oracle)["status"], "PASS")

    def test_metadata_historical_and_source_conflict_support_never_gain_finality(self) -> None:
        trace_or_metadata = [row for row in self.rows if row["support_kind"] in {"trace", "metadata"}]
        historical = [row for row in self.rows if row["source_role"] != "current_consolidated"]
        self.assertTrue(trace_or_metadata)
        self.assertTrue(historical)
        self.assertTrue(all(not row["citation_final"] for row in trace_or_metadata))
        self.assertTrue(all(not row["answer_eligible"] for row in historical))

    def test_reviewed_identities_are_data_owned_and_obsolete_constants_have_zero_consumers(self) -> None:
        source = (ROOT / "src" / "tjipto" / "corpora" / "uud" / "meaningful_support_builder.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("REVIEWED_HEADINGS", source)
        self.assertNotIn("PASAL_III_SPAN", source)
        self.assertNotIn("PASAL_III_CONFLICT", source)
        self.assertNotIn("uud_text_span::current_consolidated::0002::0010", source)
        self.assertNotIn("uud_text_span::amendment_4_historical::0006::0000", source)

    def test_evaluator_does_not_import_or_call_projection_builder(self) -> None:
        source = (ROOT / "scripts" / "evaluate_meaningful_support.py").read_text(encoding="utf-8")
        self.assertNotIn("meaningful_support_builder", source)
        self.assertNotIn("build_meaningful_support_units", source)


if __name__ == "__main__":
    unittest.main()
