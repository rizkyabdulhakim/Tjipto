from __future__ import annotations

import json
from pathlib import Path
import unittest

from tjipto.corpora.registry import CorpusRegistry


ROOT = Path(__file__).resolve().parents[1]


def _expectations() -> dict:
    path = ROOT / "tests/fixtures/uud/corpus_spec_expectations.json"
    return json.loads(path.read_text(encoding="utf-8"))


class CorpusSpecContractTest(unittest.TestCase):
    def test_uud_registry_exposes_source_conflict_intent_spec(self) -> None:
        config = CorpusRegistry(ROOT).resolve("uud")
        self.assertIsNotNone(config)
        intent = config.setting("source_conflict_intent")
        expected = _expectations()["source_conflict_intent"]
        for field in ("query_terms", "generic_tokens", "type_anchors"):
            for value in expected[field]:
                self.assertIn(value, intent[field])
        for key, value in expected["role_labels"].items():
            self.assertEqual(intent["role_labels"][key], value)
        for field in ("reason_rules", "default_reasons"):
            self.assertEqual(intent[field], expected[field])
        self.assertEqual(len(intent["answer_rules"]), expected["answer_rule_count"])
        self.assertIn(expected["default_answer_template_contains"], intent["default_answer_template"])

    def test_uud_registry_exposes_minimal_corpus_schema(self) -> None:
        config = CorpusRegistry(ROOT).resolve("uud")
        schema = config.setting("schema")
        expected = _expectations()["schema"]
        self.assertIn(config.preferred_source_role, schema["document_roles"])
        for field in ("unit_hierarchy", "metadata_fields", "relation_types", "source_conflict_types"):
            for value in expected[field]:
                self.assertIn(value, schema[field])
        self.assertEqual(schema["chunk_policy"]["direct_grounding"], expected["chunk_policy"]["direct_grounding"])

    def test_uud_registry_owns_runtime_intent_terms(self) -> None:
        config = CorpusRegistry(ROOT).resolve("uud")
        intent = config.setting("intent_config")
        expected = _expectations()["intent_config"]
        for field in (
            "document_target_words",
            "metadata_fields",
            "pasal_parent_words",
            "relation_child_words",
            "instrument_scope_queries",
            "instrument_deletion_words",
            "instrument_deletion_evidence_words",
            "instrument_change_context_words",
            "unsupported_relation_context_words",
        ):
            for value in expected[field]:
                self.assertIn(value, intent[field])
        for key, value in expected["source_role_labels"].items():
            self.assertEqual(intent["source_role_labels"][key], value)
        for expected_section in expected["structured_sections"]:
            self.assertIn(expected_section, intent["structured_sections"])
        for field, values in expected["metadata_rules"].items():
            for value in values:
                self.assertIn(value, intent["metadata_rules"][field])
        for row in expected["metadata_roles"]:
            self.assertEqual(intent["metadata_roles"][row["index"]]["role"], row["role"])
        source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (
                ROOT / "src/tjipto/corpora/intent_config.py",
                ROOT / "src/tjipto/retrieval/metadata.py",
                ROOT / "src/tjipto/retrieval/structured.py",
            )
        )
        for value in _expectations()["generic_intent_source_absent"]:
            self.assertNotIn(value, source)


if __name__ == "__main__":
    unittest.main()
