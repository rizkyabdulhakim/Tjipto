from __future__ import annotations

import json
from pathlib import Path
from shutil import copytree
import tempfile
import unittest

from tjipto.corpora.registry import CorpusRegistry
from tjipto.corpora.intent_config import validation_intent_config_for
from tjipto.corpora.uud.specs import SOURCE_CONFLICT_SPECS
from tjipto.corpora.uud.validation import validate_uud_artifact_dir


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
        for field in ("query_terms", "generic_tokens"):
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
        for field in (
            "unit_hierarchy",
            "metadata_fields",
            "relation_types",
            "source_conflict_types",
            "source_anomaly_kinds",
            "source_mapping_kinds",
        ):
            for value in expected[field]:
                self.assertIn(value, schema[field])
        self.assertEqual(schema["chunk_policy"]["direct_grounding"], expected["chunk_policy"]["direct_grounding"])
        aliases = config.setting("normalization_aliases")
        for alias in _expectations()["normalization_aliases"]:
            self.assertTrue(any(all(row.get(key) == value for key, value in alias.items()) for row in aliases))
        self.assertEqual(config.setting("query_normalization_enabled"), _expectations()["query_normalization_enabled"])
        self.assertEqual(config.setting("exact_citation_intent_enabled"), _expectations()["exact_citation_intent_enabled"])
        for key, value in _expectations()["answer_templates"].items():
            self.assertEqual(config.setting("answer_templates")[key], value)
        for key, value in _expectations()["viewer_source_status_labels"].items():
            self.assertEqual(config.setting("viewer_source_status_labels")[key], value)

    def test_uud_registry_owns_runtime_intent_terms(self) -> None:
        config = CorpusRegistry(ROOT).resolve("uud")
        intent = config.setting("intent_config")
        validation_intent = validation_intent_config_for(config.query_strategy, config)
        expected = _expectations()["intent_config"]
        for field in (
            "document_target_words",
            "metadata_fields",
            "pasal_parent_words",
            "relation_child_words",
            "relation_routes",
            "instrument_deletion_words",
            "instrument_deletion_evidence_words",
            "instrument_change_context_words",
            "instrument_citation_templates",
            "unsupported_relation_context_words",
        ):
            for value in expected[field]:
                self.assertIn(value, intent[field])
        for value in expected["instrument_scope_queries"]:
            self.assertIn(value, validation_intent["instrument_scope_queries"])
        for key, value in expected["source_role_labels"].items():
            self.assertEqual(intent["source_role_labels"][key], value)
        for key, value in expected["instrument_citation_templates"].items():
            self.assertEqual(intent["instrument_citation_templates"][key], value)
        for key, value in expected["relation_routes"].items():
            for field, expected_value in value.items():
                self.assertEqual(intent["relation_routes"][key][field], expected_value)
        for expected_section in expected["structured_sections"]:
            self.assertIn(expected_section, intent["structured_sections"])
        self.assertEqual(intent["structured_lookup_enabled"], expected["structured_lookup_enabled"])
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
                ROOT / "src/tjipto/retrieval/relations.py",
                ROOT / "src/tjipto/retrieval/structured.py",
            )
        )
        for value in _expectations()["generic_intent_source_absent"]:
            self.assertNotIn(value, source)
        relation_source = (ROOT / "src/tjipto/retrieval/relations.py").read_text(encoding="utf-8")
        for value in _expectations()["relation_source_absent"]:
            self.assertNotIn(value, relation_source)

    def test_uud_source_conflict_specs_own_taxonomy_and_anchor_terms(self) -> None:
        expected = _expectations()["source_conflict_specs"]
        rows = list(SOURCE_CONFLICT_SPECS)
        for row in rows:
            self.assertIsInstance(row.get("anchor_terms"), list)
            self.assertIsInstance(row.get("query_anchor_terms"), list)
            policy = row.get("source_anomaly_policy")
            self.assertIsInstance(policy, dict)
            for field in (
                "anomaly_kind",
                "source_role",
                "canonical_role",
                "anchor_terms",
                "affected_span_refs",
                "provenance_rules",
                "highlight_policy",
                "finality_policy",
                "corpus_id",
            ):
                self.assertIn(field, policy)
            self.assertEqual(policy["corpus_id"], row["corpus_id"])
            self.assertEqual(policy["anomaly_kind"], row["source_anomaly_kind"])
            self.assertEqual(policy["finality_policy"], "source_anomaly_provenance")
            for field in expected["policy_fields"]:
                self.assertTrue(str(row.get(field) or "").strip(), field)
        self.assertEqual({row["source_anomaly_kind"] for row in rows}, set(expected["source_anomaly_kinds"]))
        self.assertEqual(
            {row.get("source_mapping_kind") for row in rows if row.get("source_mapping_kind")},
            set(expected["source_mapping_kinds"]),
        )
        anchor_terms = {term for row in rows for term in row.get("anchor_terms", ())}
        query_anchor_terms = {term for row in rows for term in row.get("query_anchor_terms", ())}
        for value in expected["anchor_terms"]:
            self.assertIn(value, anchor_terms)
        for value in expected["query_anchor_terms"]:
            self.assertIn(value, query_anchor_terms)

    def test_validation_fails_when_source_conflict_policy_field_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "uud"
            copytree(ROOT / "data/final/uud", target)
            path = target / "source_conflicts.jsonl"
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            rows[0].pop("provenance_summary", None)
            path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
            errors = validate_uud_artifact_dir(target)
            self.assertIn(
                "missing_required_field:source_conflicts:<missing>:provenance_summary",
                errors,
            )


if __name__ == "__main__":
    unittest.main()
