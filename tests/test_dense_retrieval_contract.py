from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch
import subprocess

from tjipto.retrieval.dense import (
    DENSE_DIMENSION,
    MODEL_ID,
    MODEL_REVISION,
    DenseEmbeddingBatch,
    DenseModelIdentity,
    DenseUnavailable,
    LocalDenseProvider,
    dense_index_for_store,
    dense_search,
)


def _identity() -> DenseModelIdentity:
    return DenseModelIdentity(model_id=MODEL_ID, revision=MODEL_REVISION, tokenizer_sha256="a" * 64, model_sha256="b" * 64)


def _vector(first: float, second: float = 0.0) -> tuple[float, ...]:
    return (first, second) + (0.0,) * (DENSE_DIMENSION - 2)


def _store(*, configured: bool = True):
    evidence = [
        {"evidence_id": "evidence-a", "quoted_text": "alpha", "source_document_id": "doc", "citation": "A"},
        {"evidence_id": "evidence-b", "quoted_text": "beta", "source_document_id": "doc", "citation": "B"},
    ]
    retrieval = [
        {"retrieval_unit_id": "unit-a", "evidence_id": "evidence-a", "legal_unit_id": "legal-a", "artifact_status": "published", "text": "alpha"},
        {"retrieval_unit_id": "unit-b", "evidence_id": "evidence-b", "legal_unit_id": "legal-b", "artifact_status": "published", "text": "beta"},
    ]
    config = SimpleNamespace(
        manifest={"dense_retrieval": configured},
        corpus_id="test",
        manifest_digest="m" * 64,
        artifact_set_digest="a" * 64,
    )
    return SimpleNamespace(config=config, evidence=evidence, retrieval_units=retrieval, legal_units=[{"legal_unit_id": "legal-a", "text": "alpha"}, {"legal_unit_id": "legal-b", "text": "beta"}])


class FakeProvider:
    def __init__(self, *, invalid: str | None = None):
        self.calls: list[tuple[str, ...]] = []
        self.invalid = invalid

    def identity(self) -> DenseModelIdentity:
        return _identity()

    def embed(self, texts: tuple[str, ...]) -> DenseEmbeddingBatch:
        self.calls.append(texts)
        if self.invalid == "dimension":
            return DenseEmbeddingBatch(((1.0,),) * len(texts), _identity())
        if self.invalid == "nan":
            return DenseEmbeddingBatch(((_vector(float("nan"))),) * len(texts), _identity())
        vectors = tuple(_vector(1.0) if "alpha" in text.casefold() else _vector(0.0, 1.0) for text in texts)
        return DenseEmbeddingBatch(vectors, _identity())


class DenseRetrievalContractTest(unittest.TestCase):
    def test_disabled_lane_is_fail_soft(self) -> None:
        result = dense_search(_store(configured=False), "alpha", provider=FakeProvider())
        self.assertEqual(result["status"], "dense_unavailable")
        self.assertEqual(result["reason"], "not_configured")

    def test_real_lane_is_one_to_one_and_hides_provenance(self) -> None:
        provider = FakeProvider()
        result = dense_search(_store(), "alpha", provider=provider)
        self.assertEqual(result["route"], "dense")
        self.assertEqual([row["evidence_id"] for row in result["matches"]], ["evidence-a", "evidence-b"])
        self.assertFalse(any(key.startswith("_") for key in result["matches"][0]))
        self.assertEqual(len(provider.calls), 2)  # one index build and one query batch

    def test_ties_are_stable_and_unchanged_index_reuses_embeddings(self) -> None:
        store = _store()
        provider = FakeProvider()
        first = dense_index_for_store(store, provider=provider)
        second = dense_index_for_store(store, provider=provider)
        self.assertIs(first, second)
        self.assertEqual(first.identity_record()["record_count"], 2)
        self.assertEqual(first.identity_record()["model"]["revision"], MODEL_REVISION)
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual([row["evidence_id"] for row in first.search(_vector(0.0, 1.0), 2)], ["evidence-b", "evidence-a"])

    def test_snapshot_mutation_rebuilds(self) -> None:
        store = _store()
        provider = FakeProvider()
        first = dense_index_for_store(store, provider=provider)
        store.retrieval_units[0]["text"] = "changed"
        second = dense_index_for_store(store, provider=provider)
        self.assertIsNot(first, second)
        self.assertEqual(len(provider.calls), 2)

    def test_embedding_source_mutation_rebuilds(self) -> None:
        store = _store()
        provider = FakeProvider()
        first = dense_index_for_store(store, provider=provider)
        store.evidence[0]["quoted_text"] = "changed"
        second = dense_index_for_store(store, provider=provider)
        self.assertIsNot(first, second)

    def test_invalid_provider_vectors_are_unavailable(self) -> None:
        for kind in ("dimension", "nan"):
            result = dense_search(_store(), "alpha", provider=FakeProvider(invalid=kind))
            self.assertEqual(result["status"], "dense_unavailable")

    def test_worker_timeout_and_malformed_response_are_typed(self) -> None:
        provider = LocalDenseProvider(timeout_seconds=0.1)
        with patch("tjipto.retrieval.dense.subprocess.run", side_effect=subprocess.TimeoutExpired("worker", 0.1)):
            with self.assertRaisesRegex(DenseUnavailable, "dense_timeout"):
                provider.embed(("alpha",))
        with patch("tjipto.retrieval.dense.subprocess.run", return_value=SimpleNamespace(returncode=0, stdout="not-json")):
            with self.assertRaisesRegex(DenseUnavailable, "worker_malformed"):
                provider.embed(("alpha",))


if __name__ == "__main__":
    unittest.main()
