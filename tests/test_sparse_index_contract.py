from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from dataclasses import FrozenInstanceError
from unittest.mock import patch

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

        store.config = _config()
        store.evidence[0]["quoted_text"] = "gamma"
        third = sparse_index_for_store(store)
        self.assertIsNot(second, third)
        store.evidence[0]["evidence_id"] = "c"
        fourth = sparse_index_for_store(store)
        self.assertIsNot(third, fourth)
        self.assertEqual(fourth.identity, SparseIndex.snapshot_identity(store.evidence, config=store.config))

    def test_nested_document_state_and_search_results_are_isolated(self) -> None:
        rows = [{"evidence_id": "a", "quoted_text": "alpha", "hierarchy": ["I"], "metadata": {"page": 1}}]
        index = SparseIndex.build(rows, config=_config())
        self.assertFalse(any(isinstance(document.fields, dict) for document in index.documents))
        result = index.search("alpha")[0]
        result["hierarchy"].append("mutated")
        result["metadata"]["page"] = 99
        fresh = index.search("alpha")[0]
        self.assertEqual(fresh["hierarchy"], ["I"])
        self.assertEqual(fresh["metadata"], {"page": 1})

    def test_unchanged_snapshot_does_not_rebuild_statistics(self) -> None:
        store = SimpleNamespace(config=_config(), evidence=[{"evidence_id": "a", "quoted_text": "alpha"}])
        first = sparse_index_for_store(store)
        with patch.object(SparseIndex, "build", side_effect=AssertionError("unexpected rebuild")):
            self.assertIs(first, sparse_index_for_store(store))

    def test_bm25_exposes_neutral_coverage_only(self) -> None:
        result = SparseIndex.build([{"evidence_id": "a", "quoted_text": "alpha"}], config=_config()).search("alpha")[0]
        self.assertTrue(result["lexical_complete_coverage"])
        self.assertNotIn("lexical_relevance_ok", result)
        self.assertNotIn("lexical_relevance_reason", result)
        self.assertNotIn("answer_evidence", repr(result))

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
