from __future__ import annotations

from pathlib import Path
import unittest

from tjipto.corpora.intent_config import intent_config_for
from tjipto.corpora.registry import CorpusRegistry
from tjipto.retrieval.metadata import has_metadata_target
from tjipto.retrieval.relations import has_relation_target


ROOT = Path(__file__).resolve().parents[1]


class RuntimeNoHardcodedIntentContractTest(unittest.TestCase):
    def test_uud_intent_terms_require_corpus_config(self) -> None:
        generic = intent_config_for("uud_1945")
        self.assertFalse(generic["metadata_fields"])
        self.assertFalse(generic["relation_words"])
        self.assertFalse(has_metadata_target("tanggal penetapan perubahan kedua UUD", strategy="uud_1945"))
        self.assertFalse(has_relation_target("relasi amandemen Pasal 1", strategy="uud_1945"))

        config = CorpusRegistry(ROOT).resolve("uud")
        configured = intent_config_for(config.query_strategy, config)
        self.assertIn("penetapan", configured["metadata_fields"])
        self.assertIn("relasi", configured["relation_words"])
        self.assertTrue(has_metadata_target("tanggal penetapan perubahan kedua UUD", strategy=config.query_strategy, config=config))
        self.assertTrue(has_relation_target("relasi amandemen Pasal 1", strategy=config.query_strategy, config=config))


if __name__ == "__main__":
    unittest.main()
