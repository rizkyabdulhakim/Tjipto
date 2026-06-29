from __future__ import annotations

import json
from pathlib import Path
import unittest

from tjipto.corpora.intent_config import intent_config_for
from tjipto.corpora.registry import CorpusRegistry
from tjipto.retrieval.metadata import has_metadata_target
from tjipto.retrieval.relations import has_relation_target


ROOT = Path(__file__).resolve().parents[1]


def _intent_cases() -> tuple[dict, ...]:
    path = ROOT / "tests/fixtures/uud/no_hardcoded_intent_cases.jsonl"
    return tuple(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


class RuntimeNoHardcodedIntentContractTest(unittest.TestCase):
    def test_uud_intent_terms_require_corpus_config(self) -> None:
        generic = intent_config_for("uud_1945")
        self.assertFalse(generic["metadata_fields"])
        self.assertFalse(generic["relation_words"])
        self.assertFalse(generic["instrument_scope_queries"])

        config = CorpusRegistry(ROOT).resolve("uud")
        configured = intent_config_for(config.query_strategy, config)
        self.assertIn("penetapan", configured["metadata_fields"])
        self.assertIn("relasi", configured["relation_words"])
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
            else:
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


if __name__ == "__main__":
    unittest.main()
