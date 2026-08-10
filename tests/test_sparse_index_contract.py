from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from dataclasses import FrozenInstanceError

from tjipto.retrieval.bm25 import SparseIndex, lexical_search, sparse_index_for_store
from tjipto.runtime.api import handle_request
from tjipto.runtime.service import LegalRuntimeService


ROOT = Path(__file__).resolve().parents[1]


def _config(**kwargs):
    values = {
        "corpus_id": "synthetic",
        "manifest_digest": "m" * 64,
        "artifact_set_digest": "a" * 64,
        "manifest_path": ROOT / "manifest.json",
        "settings": {},
    }
    values.update(kwargs)
    config = SimpleNamespace(**values)
    config.setting = lambda key, default=None: (config.settings or {}).get(key, default)
    return config


class SparseIndexContractTest(unittest.TestCase):
    def test_order_and_raw_provenance_are_deterministic(self) -> None:
        rows = [
            {"evidence_id": "b", "quoted_text": "alpha"},
            {"evidence_id": "a", "quoted_text": "alpha"},
        ]
        index = SparseIndex.build(rows, config=_config())
        results = index.search("alpha", limit=10)
        self.assertEqual([row["evidence_id"] for row in results], ["a", "b"])
        self.assertEqual([row["_bm25_provenance"]["rank"] for row in results], [1, 2])
        self.assertTrue(all(row["_bm25_provenance"]["retriever"] == "bm25" for row in results))
        self.assertTrue(all(row["_bm25_provenance"]["score_domain"] == "bm25" for row in results))
        self.assertEqual(
            [row["evidence_id"] for row in lexical_search(rows, "alpha", config=_config())],
            [row["evidence_id"] for row in results],
        )

    def test_store_reuses_and_invalidates_index(self) -> None:
        store = SimpleNamespace(config=_config(), evidence=[{"evidence_id": "a", "quoted_text": "alpha"}])
        first = sparse_index_for_store(store)
        self.assertIs(first, sparse_index_for_store(store))
        store.evidence.append({"evidence_id": "b", "quoted_text": "beta"})
        second = sparse_index_for_store(store)
        self.assertIsNot(first, second)
        self.assertEqual(second.record_count, 2)
        store.config = _config(settings={"lexical_normalization": {"aliases": {"a": "alpha"}}})
        self.assertIsNot(second, sparse_index_for_store(store))

    def test_index_state_is_read_only(self) -> None:
        index = SparseIndex.build([], config=_config())
        with self.assertRaises(FrozenInstanceError):
            index.k1 = 2.0

    def test_raw_provenance_is_not_public(self) -> None:
        service = LegalRuntimeService(ROOT)
        result = handle_request("uud", "ask", {"query": "hak memperoleh pengajaran"}, service=service)
        text = repr(result)
        self.assertNotIn("_bm25_provenance", text)
        self.assertNotIn("raw_score", text)
        self.assertNotIn('"rank"', text)


if __name__ == "__main__":
    unittest.main()
