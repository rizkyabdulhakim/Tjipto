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


if __name__ == "__main__":
    unittest.main()
