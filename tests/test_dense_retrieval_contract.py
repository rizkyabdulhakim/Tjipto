from __future__ import annotations

from types import SimpleNamespace
import os
import unittest
from unittest.mock import patch
import subprocess

from tjipto.retrieval.dense import (
    DENSE_MAX_LENGTH,
    DENSE_POOLING,
    DENSE_DIMENSION,
    MODEL_ID,
    MODEL_REVISION,
    DenseEmbeddingBatch,
    DenseIndex,
    DenseModelIdentity,
    DenseUnavailable,
    LocalDenseProvider,
    _legal_embedding_text,
    _model_identity_digests,
    dense_index_for_store,
    dense_search,
    _embedding_text,
)
from tjipto.retrieval.dense_worker import _cls_pool
from tjipto.corpora.uud.source_policy import normalize_retrieval_text
from pathlib import Path
import tempfile
from tjipto.runtime.service import LegalRuntimeService


def _identity() -> DenseModelIdentity:
    return DenseModelIdentity(
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        tokenizer_sha256="a" * 64,
        model_sha256="b" * 64,
        pooling_config_sha256="c" * 64,
    )


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
    def __init__(self, *, invalid: str | None = None, max_length: int = DENSE_MAX_LENGTH):
        self.calls: list[tuple[str, ...]] = []
        self.invalid = invalid
        self.max_length = max_length

    def identity(self) -> DenseModelIdentity:
        identity = _identity()
        return DenseModelIdentity(**(identity.as_dict() | {"max_length": self.max_length}))

    def embed(self, texts: tuple[str, ...]) -> DenseEmbeddingBatch:
        self.calls.append(texts)
        if self.invalid == "dimension":
            return DenseEmbeddingBatch(((1.0,),) * len(texts), self.identity())
        if self.invalid == "nan":
            return DenseEmbeddingBatch(((_vector(float("nan"))),) * len(texts), self.identity())
        vectors = tuple(_vector(1.0) if "alpha" in text.casefold() else _vector(0.0, 1.0) for text in texts)
        return DenseEmbeddingBatch(vectors, self.identity())


class DenseRetrievalContractTest(unittest.TestCase):
    def test_pembukaan_current_query_uses_deterministic_current_support(self) -> None:
        result = LegalRuntimeService(Path(__file__).resolve().parents[1]).ask(
            "uud", "Apa isi Pembukaan UUD 1945 saat ini?", limit=3
        )
        self.assertEqual(result["status"], "answer_ready")
        self.assertEqual(result["route"], "legal_reference")
        self.assertTrue(result["citations"])
        self.assertEqual(result["citations"][0]["source_role"], "current_consolidated")
        self.assertEqual(result["citations"][0]["citation"], "PEMBUKAAN/Preambule")

    def test_model_identity_binds_cls_and_effective_length(self) -> None:
        identity = _identity().as_dict()
        self.assertEqual(identity["pooling"], DENSE_POOLING)
        self.assertEqual(identity["max_length"], DENSE_MAX_LENGTH)
        self.assertEqual(DenseModelIdentity.from_value(identity).pooling, "cls")

    def test_allowed_ablation_lengths_validate_as_distinct_index_identity(self) -> None:
        for length in (256, 512, 1024):
            index = dense_index_for_store(_store(), provider=FakeProvider(max_length=length))
            self.assertEqual(index.model_identity.max_length, length)

    def test_cls_pooling_is_not_mean_pooling(self) -> None:
        class Probe:
            def __getitem__(self, key):
                self.key = key
                return "cls"

        hidden = Probe()
        self.assertEqual(_cls_pool(hidden), "cls")
        self.assertEqual(hidden.key, (slice(None), 0, slice(None)))
        self.assertNotEqual((1.0, 0.0), (0.5, 0.5))

    def test_embedding_text_uses_source_fields_not_opaque_ids(self) -> None:
        text = _embedding_text(
            {"source_document_id": "doc", "evidence_id": "opaque", "text": "ignored"},
            {"evidence_id": "opaque", "quoted_text": "legal"},
            {"unit_label": "Pasal", "hierarchy": ["BAB I"], "text": "legal"},
        )
        self.assertNotIn("opaque", text)
        self.assertIn("doc", text)
        self.assertIn("legal", text)

    def test_source_aware_soft_hyphen_repairs_only_lexical_occurrences(self) -> None:
        self.assertEqual(normalize_retrieval_text("peri\u00adkemanusiaan"), "peri-kemanusiaan")
        self.assertEqual(normalize_retrieval_text("\u00ad\u00ad"), "")
        self.assertEqual(normalize_retrieval_text("normal orthography"), "normal orthography")

    def test_source_lineage_stitches_boundary_soft_hyphen_before_normalization(self) -> None:
        store = _store()
        store.page_text_spans = [
            {
                "text_span_id": "span-a",
                "source_document_id": "doc",
                "page_number": 1,
                "source_object_id": "pdf_object::doc::0001",
                "text": "Undang",
            },
            {
                "text_span_id": "span-b",
                "source_document_id": "doc",
                "page_number": 1,
                "source_object_id": "pdf_object::doc::0002",
                "text": "Undang Dasar",
            },
        ]
        store.raw_source_spans = [
            {
                "source_document_id": "doc",
                "page_number": 1,
                "block_index": 1,
                "raw_stream_id": "stream",
                "raw_text_start": 0,
                "raw_text_end": 7,
                "raw_text": "Undang\u00ad",
                "semantic_text": "Undang",
            },
            {
                "source_document_id": "doc",
                "page_number": 1,
                "block_index": 2,
                "raw_stream_id": "stream",
                "raw_text_start": 8,
                "raw_text_end": 20,
                "raw_text": "Undang Dasar",
                "semantic_text": "Undang Dasar",
            },
        ]
        legal = {"text_span_ids": ("span-a", "span-b"), "text": "Undang Undang Dasar"}
        self.assertEqual(
            _legal_embedding_text(store, legal, {"text": legal["text"]}, {"quoted_text": legal["text"]}),
            "Undang-Undang Dasar",
        )

    def test_corpus_boundary_orthography_is_reconstructed(self) -> None:
        store = LegalRuntimeService(Path(__file__).resolve().parents[1])._store("uud")
        legal = next(row for row in store.legal_units if "sebesar \n" in str(row.get("text")))
        retrieval = next(row for row in store.retrieval_units if row.get("legal_unit_id") == legal.get("legal_unit_id"))
        evidence = next(row for row in store.evidence if row.get("evidence_id") == retrieval.get("evidence_id"))
        text = _legal_embedding_text(store, legal, retrieval, evidence)
        self.assertNotIn("sebesar\nbesar", text)
        self.assertIn("sebesar-besar", text)

    def test_model_identity_uses_pinned_pooling_config_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text('{"model": "root"}', encoding="utf-8")
            pooling = root / "1_Pooling"
            pooling.mkdir()
            (pooling / "config.json").write_text(
                '{"word_embedding_dimension": 1024, "pooling_mode_cls_token": true}', encoding="utf-8"
            )
            _, _, digest = _model_identity_digests(root)
            import hashlib

            self.assertEqual(digest, hashlib.sha256((pooling / "config.json").read_bytes()).hexdigest())

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

    def test_persisted_index_is_identity_bound_and_reloadable(self) -> None:
        store = _store()
        provider = FakeProvider()
        index = dense_index_for_store(store, provider=provider)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dense.index"
            index.persist(path)
            loaded = DenseIndex.load(path, store, provider=provider)
            self.assertEqual(loaded.identity, index.identity)
            self.assertEqual(loaded.identity_record()["source_identity"], index.source_identity)
            store.evidence[0]["status"] = "draft"
            with self.assertRaisesRegex(DenseUnavailable, "dense_identity_mismatch"):
                DenseIndex.load(path, store, provider=provider)

    def test_configured_artifact_activates_dense_search(self) -> None:
        store = _store()
        provider = FakeProvider()
        index = dense_index_for_store(store, provider=provider)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dense.index"
            index.persist(path)
            fresh_store = _store()
            with patch.dict(os.environ, {"TJIPTO_DENSE_INDEX_PATH": str(path)}, clear=False):
                result = dense_search(fresh_store, "alpha", provider=FakeProvider())
            self.assertEqual(result["status"], "found")
            self.assertEqual(result["route"], "dense")

    def test_invalid_persisted_vectors_are_rejected(self) -> None:
        store = _store()
        provider = FakeProvider()
        index = dense_index_for_store(store, provider=provider)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dense.index"
            index.persist(path)
            payload = path.read_bytes().replace(b"TJIPTO_DENSE_INDEX", b"INVALID_DENSE_INDEX", 1)
            path.write_bytes(payload)
            with self.assertRaisesRegex(DenseUnavailable, "dense_artifact_magic_invalid"):
                DenseIndex.load(path, store, provider=provider)

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
