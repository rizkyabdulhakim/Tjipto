from __future__ import annotations

import json
from pathlib import Path
import unittest

from tjipto.corpora.intent_config import intent_config_for
from tjipto.corpora.registry import CorpusRegistry
from tjipto.retrieval.metadata import has_metadata_target
from tjipto.retrieval.relations import has_relation_target
from tjipto.retrieval.structured import has_structured_target


ROOT = Path(__file__).resolve().parents[1]


def _intent_cases() -> tuple[dict, ...]:
    path = ROOT / "tests/fixtures/uud/no_hardcoded_intent_cases.jsonl"
    return tuple(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


class RuntimeNoHardcodedIntentContractTest(unittest.TestCase):
    def test_generic_clarification_contains_no_corpus_vocabulary(self) -> None:
        source = (ROOT / "src/tjipto/runtime/clarification.py").read_text(encoding="utf-8").casefold()
        for term in ("atau", "pasal", "bab", "penandatangan", "wakil ketua"):
            self.assertNotIn(term, source)

    def test_uud_intent_terms_require_corpus_config(self) -> None:
        generic = intent_config_for("uud_1945")
        self.assertFalse(generic["metadata_fields"])
        self.assertFalse(generic["relation_words"])
        self.assertFalse(generic["instrument_scope_queries"])
        self.assertFalse(generic["structured_sections"])
        self.assertFalse(generic["structured_lookup_enabled"])

        config = CorpusRegistry(ROOT).resolve("uud")
        configured = intent_config_for(config.query_strategy, config)
        self.assertIn("penetapan", configured["metadata_fields"])
        self.assertIn("relasi", configured["relation_words"])
        self.assertTrue(configured["structured_lookup_enabled"])
        scope_guard = config.setting("scope_guard", {}) or {}
        criminal_policy = next(
            row for row in scope_guard["legal_intent_policy"]["unsupported_functions"] if row["requested_function"] == "criminal_punishment"
        )
        self.assertIn("korupsi", criminal_policy["topic_terms"])
        self.assertIn("pidana", criminal_policy["ambiguous_criminal_terms"])
        self.assertIn("sanksi", criminal_policy["unsupported_function_terms"])
        self.assertIn("pasal 7a", criminal_policy["target_reference_terms"])
        self.assertNotIn("pasal 7a", criminal_policy["supported_function_terms"])
        self.assertNotIn("presiden", criminal_policy["supported_function_terms"])
        relation_families = configured["document_relation"]["relation_families"]
        self.assertIn("dicabut", relation_families["DELETE_OR_REMOVE_PROVISION"]["terms"])
        self.assertIn("RENAME_PROVISION", relation_families)
        self.assertEqual(relation_families["RENAME_PROVISION"]["relation_types"], ["RENAMES"])
        self.assertNotIn("RENAMES", configured["document_relation"]["schema_only_relation_types"])
        for case in _intent_cases():
            if case["kind"] == "metadata":
                self.assertEqual(
                    has_metadata_target(case["query"], strategy="uud_1945"),
                    case["generic_expected"],
                    case["query"],
                )
                self.assertEqual(
                    has_metadata_target(case["query"], strategy=config.query_strategy, config=config),
                    case["configured_expected"],
                    case["query"],
                )
            elif case["kind"] == "relation":
                self.assertEqual(
                    has_relation_target(case["query"], strategy="uud_1945"),
                    case["generic_expected"],
                    case["query"],
                )
                self.assertEqual(
                    has_relation_target(case["query"], strategy=config.query_strategy, config=config),
                    case["configured_expected"],
                    case["query"],
                )
            else:
                self.assertEqual(
                    has_structured_target(case["query"], strategy="uud_1945"),
                    case["generic_expected"],
                    case["query"],
                )
                self.assertEqual(
                    has_structured_target(case["query"], strategy=config.structured_strategy, config=config),
                    case["configured_expected"],
                    case["query"],
                )


if __name__ == "__main__":
    unittest.main()
