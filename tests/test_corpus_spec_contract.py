from __future__ import annotations

from pathlib import Path
import unittest

from tjipto.corpora.registry import CorpusRegistry


ROOT = Path(__file__).resolve().parents[1]


class CorpusSpecContractTest(unittest.TestCase):
    def test_uud_registry_exposes_source_conflict_intent_spec(self) -> None:
        config = CorpusRegistry(ROOT).resolve("uud")
        self.assertIsNotNone(config)
        intent = config.setting("source_conflict_intent")
        self.assertIn("pasal 25e", intent["query_terms"])
        self.assertIn("source_marker_sequence_conflict", intent["type_anchors"])
        self.assertEqual(intent["role_labels"]["amendment_4_historical"], "perubahan keempat")

    def test_uud_registry_exposes_minimal_corpus_schema(self) -> None:
        config = CorpusRegistry(ROOT).resolve("uud")
        schema = config.setting("schema")
        self.assertIn(config.preferred_source_role, schema["document_roles"])
        self.assertIn("pasal_record", schema["unit_hierarchy"])
        self.assertIn("source_anomaly_status", schema["metadata_fields"])
        self.assertIn("HAS_SOURCE_ANOMALY", schema["relation_types"])
        self.assertIn("article_renumbering_conflict", schema["source_conflict_types"])
        self.assertEqual(schema["chunk_policy"]["direct_grounding"], "text_span_exact")

    def test_uud_registry_owns_runtime_intent_terms(self) -> None:
        config = CorpusRegistry(ROOT).resolve("uud")
        intent = config.setting("intent_config")
        self.assertIn("penetapan", intent["metadata_fields"])
        self.assertIn("berada di bab", intent["pasal_parent_words"])
        self.assertEqual(intent["metadata_roles"][1]["role"], "amendment_2_historical")
        source = (ROOT / "src/tjipto/corpora/intent_config.py").read_text(encoding="utf-8")
        self.assertNotIn("perubahan pertama", source)
        self.assertNotIn("berada di bab", source)


if __name__ == "__main__":
    unittest.main()
