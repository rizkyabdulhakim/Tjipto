from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess  # nosec B404 - fixed module worker, no shell
import sys
from dataclasses import dataclass
from typing import Protocol


MODEL_ID = "BAAI/bge-m3"
MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
DENSE_DIMENSION = 1024
DENSE_DTYPE = "float32"
DENSE_NORMALIZATION = "l2"
DENSE_POOLING = "cls"
DENSE_MAX_LENGTH = 256
DENSE_BATCH_SIZE = 64
DENSE_TRUNCATION_POLICY = "explicit_max_length"
EMBEDDING_TEXT_POLICY = "source_document+breadcrumb+label+legal_text"
INDEX_BUILDER_ID = "tjipto.dense.index"


class DenseError(RuntimeError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


class DenseUnavailable(DenseError):
    pass


@dataclass(frozen=True)
class DenseModelIdentity:
    model_id: str = MODEL_ID
    revision: str = MODEL_REVISION
    dimension: int = DENSE_DIMENSION
    dtype: str = DENSE_DTYPE
    normalization: str = DENSE_NORMALIZATION
    pooling: str = DENSE_POOLING
    max_length: int = DENSE_MAX_LENGTH
    truncation_policy: str = DENSE_TRUNCATION_POLICY
    tokenizer_sha256: str | None = None
    model_sha256: str | None = None
    pooling_config_sha256: str | None = None
    builder_identity: str = INDEX_BUILDER_ID

    def as_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "dimension": self.dimension,
            "dtype": self.dtype,
            "normalization": self.normalization,
            "pooling": self.pooling,
            "max_length": self.max_length,
            "truncation_policy": self.truncation_policy,
            "tokenizer_sha256": self.tokenizer_sha256,
            "model_sha256": self.model_sha256,
            "pooling_config_sha256": self.pooling_config_sha256,
            "builder_identity": self.builder_identity,
        }

    @classmethod
    def from_value(cls, value: object) -> "DenseModelIdentity":
        if isinstance(value, DenseModelIdentity):
            return value
        if not isinstance(value, dict):
            raise DenseUnavailable("model_identity_missing")
        identity = cls(**{key: value[key] for key in cls.__dataclass_fields__ if key in value})
        if (
            identity.model_id != MODEL_ID
            or identity.revision != MODEL_REVISION
            or identity.dimension != DENSE_DIMENSION
            or identity.dtype != DENSE_DTYPE
            or identity.normalization != DENSE_NORMALIZATION
            or identity.pooling != DENSE_POOLING
            or identity.max_length != DENSE_MAX_LENGTH
            or identity.truncation_policy != DENSE_TRUNCATION_POLICY
        ):
            raise DenseUnavailable("noncanonical_model")
        return identity


@dataclass(frozen=True)
class DenseEmbeddingBatch:
    vectors: tuple[tuple[float, ...], ...]
    identity: DenseModelIdentity
    truncated_indices: tuple[int, ...] = ()
    worker_peak_rss_bytes: int | None = None


class DenseEmbeddingProvider(Protocol):
    def identity(self) -> DenseModelIdentity: ...

    def embed(self, texts: tuple[str, ...]) -> DenseEmbeddingBatch: ...


class LocalDenseProvider:
    """Run model inference in a short-lived worker, never in the API process."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 120.0,
        python_executable: str | None = None,
        batch_size: int = DENSE_BATCH_SIZE,
        max_length: int = DENSE_MAX_LENGTH,
    ):
        self.timeout_seconds = timeout_seconds
        self.python_executable = python_executable or sys.executable
        self.batch_size = batch_size
        self.max_length = max_length

    def identity(self) -> DenseModelIdentity:
        model_dir = _model_snapshot_path()
        tokenizer_digest, model_digest, pooling_digest = _model_identity_digests(model_dir)
        return DenseModelIdentity(
            tokenizer_sha256=os.environ.get("TJIPTO_DENSE_TOKENIZER_SHA256") or tokenizer_digest,
            model_sha256=os.environ.get("TJIPTO_DENSE_MODEL_SHA256") or model_digest,
            pooling_config_sha256=os.environ.get("TJIPTO_DENSE_POOLING_CONFIG_SHA256") or pooling_digest,
            pooling=DENSE_POOLING,
            max_length=self.max_length,
            truncation_policy=DENSE_TRUNCATION_POLICY,
        )

    def embed(self, texts: tuple[str, ...]) -> DenseEmbeddingBatch:
        if self.batch_size < 1 or self.max_length != DENSE_MAX_LENGTH:
            raise DenseUnavailable("dense_configuration_invalid")
        if not texts:
            return DenseEmbeddingBatch((), self.identity())
        return self._embed_chunk(texts)

    def _embed_chunk(self, texts: tuple[str, ...]) -> DenseEmbeddingBatch:
        payload = {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "texts": list(texts),
            "batch_size": self.batch_size,
            "max_length": self.max_length,
        }
        try:
            completed = subprocess.run(  # nosec B603 - fixed executable/module, no shell
                [self.python_executable, "-m", "tjipto.retrieval.dense_worker"],
                input=json.dumps(payload, ensure_ascii=False),
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise DenseUnavailable("dense_timeout" if isinstance(error, subprocess.TimeoutExpired) else "worker_unavailable") from error
        try:
            response = json.loads(completed.stdout)
        except (TypeError, json.JSONDecodeError) as error:
            raise DenseUnavailable("worker_malformed") from error
        if completed.returncode or not isinstance(response, dict):
            raise DenseUnavailable(str(response.get("error") or "worker_failed") if isinstance(response, dict) else "worker_failed")
        if response.get("error"):
            raise DenseUnavailable(str(response["error"]))
        identity = DenseModelIdentity.from_value(response.get("model_identity"))
        vectors = _parse_vectors(response.get("vectors"), identity.dimension)
        truncated = response.get("truncated_indices", [])
        if not isinstance(truncated, list) or any(not isinstance(index, int) for index in truncated):
            raise DenseUnavailable("worker_truncation_metadata_malformed")
        worker_peak_rss = response.get("worker_peak_rss_bytes")
        if worker_peak_rss is not None and (not isinstance(worker_peak_rss, int) or worker_peak_rss < 0):
            raise DenseUnavailable("worker_resource_malformed")
        return DenseEmbeddingBatch(vectors, identity, tuple(truncated), worker_peak_rss)


@dataclass(frozen=True)
class DenseDocument:
    retrieval_unit_id: str
    evidence_id: str
    row_json: bytes

    def row(self) -> dict:
        return json.loads(self.row_json.decode("utf-8"))


@dataclass(frozen=True)
class DenseIndex:
    """Immutable exact-search vectors and one-to-one evidence mapping."""

    identity: str
    source_identity: str
    corpus_id: str | None
    artifact_set_digest: str | None
    manifest_digest: str | None
    retrieval_units_digest: str
    model_identity: DenseModelIdentity
    record_count: int
    dimension: int
    retrieval_unit_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    documents: tuple[DenseDocument, ...]
    vector_bytes: bytes
    vector_mapping_digest: str
    embedding_text_policy: str
    batch_size: int
    truncation_count: int
    truncated_retrieval_unit_ids: tuple[str, ...]
    embedding_text_digest: str
    worker_peak_rss_bytes: int | None = None

    @classmethod
    def build(cls, store, provider: DenseEmbeddingProvider) -> "DenseIndex":
        records = _dense_records(store)
        texts = tuple(_embedding_text(row, evidence, legal, store=store) for row, evidence, legal in records)
        batch = provider.embed(texts)
        vectors = _validate_batch(batch, len(records))
        model = batch.identity
        embedding_text_digest = _digest(texts)
        source_identity = _source_identity(store, records, model, embedding_text_digest)
        corpus_id = getattr(store.config, "corpus_id", None)
        artifact_set_digest = getattr(store.config, "artifact_set_digest", None)
        manifest_digest = getattr(store.config, "manifest_digest", None)
        retrieval_units_digest = _digest([row for row, _, _ in records])
        retrieval_ids = tuple(str(row["retrieval_unit_id"]) for row, _, _ in records)
        evidence_ids = tuple(str(row["evidence_id"]) for row, _, _ in records)
        mapping_digest = _digest((retrieval_ids, evidence_ids))
        truncated_indices = tuple(batch.truncated_indices)
        if any(index < 0 or index >= len(records) for index in truncated_indices):
            raise DenseError("truncation_metadata_invalid")
        documents = tuple(
            DenseDocument(retrieval_id, evidence_id, _row_bytes(evidence))
            for retrieval_id, evidence_id, (_, evidence, _) in zip(retrieval_ids, evidence_ids, records)
        )
        vector_bytes = b"".join(struct.pack("<" + "f" * model.dimension, *vector) for vector in vectors)
        identity = _digest({"source": source_identity, "model": model.as_dict(), "mapping": mapping_digest, "count": len(records)})
        return cls(
            identity,
            source_identity,
            corpus_id,
            artifact_set_digest,
            manifest_digest,
            retrieval_units_digest,
            model,
            len(records),
            model.dimension,
            retrieval_ids,
            evidence_ids,
            documents,
            vector_bytes,
            mapping_digest,
            EMBEDDING_TEXT_POLICY,
            getattr(provider, "batch_size", DENSE_BATCH_SIZE),
            len(truncated_indices),
            tuple(retrieval_ids[index] for index in truncated_indices),
            embedding_text_digest,
            batch.worker_peak_rss_bytes,
        )

    def identity_record(self) -> dict[str, object]:
        return {
            "identity": self.identity,
            "corpus_id": self.corpus_id,
            "artifact_set_digest": self.artifact_set_digest,
            "manifest_digest": self.manifest_digest,
            "retrieval_units_digest": self.retrieval_units_digest,
            "model": self.model_identity.as_dict(),
            "record_count": self.record_count,
            "vector_mapping_digest": self.vector_mapping_digest,
            "embedding_text_policy": self.embedding_text_policy,
            "batch_size": self.batch_size,
            "truncation_count": self.truncation_count,
            "truncated_retrieval_unit_ids": self.truncated_retrieval_unit_ids,
            "embedding_text_digest": self.embedding_text_digest,
            "worker_peak_rss_bytes": self.worker_peak_rss_bytes,
            "worker_peak_rss_scope": "embedding_worker_peak_working_set",
        }

    def search(self, query_vector: tuple[float, ...], limit: int = 10) -> list[dict]:
        if len(query_vector) != self.dimension:
            raise DenseUnavailable("query_dimension_mismatch")
        _validate_vector(query_vector, self.dimension)
        norm = math.sqrt(math.fsum(value * value for value in query_vector))
        if abs(norm - 1.0) > 1e-3:
            raise DenseUnavailable("query_not_normalized")
        scored: list[tuple[float, str, int]] = []
        width = self.dimension * 4
        for index, evidence_id in enumerate(self.evidence_ids):
            values = struct.unpack_from("<" + "f" * self.dimension, self.vector_bytes, index * width)
            score = math.fsum(left * right for left, right in zip(values, query_vector))
            scored.append((score, evidence_id, index))
        results = []
        for rank, (score, _, index) in enumerate(sorted(scored, key=lambda item: (-item[0], item[1]))[: max(0, limit)], 1):
            row = self.documents[index].row()
            row["_dense_provenance"] = {"retriever": "bge-m3", "raw_score": score, "rank": rank, "score_domain": "inner_product"}
            results.append(row)
        return results


def dense_search(
    store,
    query: str,
    limit: int = 10,
    *,
    provider: DenseEmbeddingProvider | None = None,
    include_provenance: bool = False,
) -> dict:
    configured = bool(
        getattr(store.config, "manifest", {}).get("dense_retrieval")
        or getattr(store.config, "setting", lambda *_: False)("dense_retrieval", False)
    )
    if not configured:
        return _unavailable("not_configured")
    provider = provider or LocalDenseProvider()
    try:
        index = dense_index_for_store(store, provider=provider)
        query_batch = provider.embed((query,))
        if query_batch.identity != index.model_identity:
            raise DenseUnavailable("model_identity_mismatch")
        matches = index.search(query_batch.vectors[0], limit)
        if not include_provenance:
            for row in matches:
                row.pop("_dense_provenance", None)
        return {"status": "found" if matches else "no_results", "route": "dense", "matches": tuple(matches), "reason": None}
    except DenseError as error:
        return _unavailable(error.code)
    except (TypeError, ValueError, KeyError, OverflowError):
        return _unavailable("dense_invalid")


def dense_index_for_store(store, *, provider: DenseEmbeddingProvider | None = None) -> DenseIndex:
    provider = provider or LocalDenseProvider()
    records = _dense_records(store)
    texts = _embedding_texts(store, records)
    source_identity = _source_identity(store, records, provider.identity(), _digest(texts))
    index = getattr(store, "_dense_index", None)
    if isinstance(index, DenseIndex) and index.source_identity == source_identity:
        return index
    index = DenseIndex.build(store, provider)
    store._dense_index = index
    store._dense_index_cache_key = source_identity
    return index


def _dense_records(store) -> list[tuple[dict, dict, dict]]:
    evidence_by_id = {str(row.get("evidence_id")): row for row in store.evidence}
    legal_by_id = {str(row.get("legal_unit_id")): row for row in getattr(store, "legal_units", ())}
    records: list[tuple[dict, dict, dict]] = []
    seen_retrieval: set[str] = set()
    seen_evidence: set[str] = set()
    for row in store.retrieval_units:
        if row.get("artifact_status") != "published" or row.get("runtime_loadable", True) is not True:
            continue
        retrieval_id = str(row.get("retrieval_unit_id") or "")
        evidence_id = str(row.get("evidence_id") or "")
        evidence = evidence_by_id.get(evidence_id)
        legal = legal_by_id.get(str(row.get("legal_unit_id") or ""), {})
        if not retrieval_id or not evidence_id or evidence is None or retrieval_id in seen_retrieval or evidence_id in seen_evidence:
            raise DenseError("dense_mapping_invalid")
        seen_retrieval.add(retrieval_id)
        seen_evidence.add(evidence_id)
        records.append((row, evidence, legal))
    if not records:
        raise DenseError("dense_mapping_empty")
    return records


def _embedding_text(retrieval: dict, evidence: dict, legal: dict, *, store=None) -> str:
    breadcrumb = " ".join(str(value) for value in (legal.get("hierarchy") or retrieval.get("hierarchy") or ()))
    label = legal.get("canonical_label") or legal.get("unit_label") or evidence.get("citation") or ""
    legal_text = _legal_embedding_text(store, legal, retrieval, evidence)
    return "\n".join(
        (
            str(retrieval.get("source_document_id") or evidence.get("source_document_id") or ""),
            breadcrumb,
            str(label),
            legal_text,
        )
    )


def _legal_embedding_text(store, legal: dict, retrieval: dict, evidence: dict) -> str:
    """Prefer source-backed raw lineage when it repairs semantic SHY loss."""
    fallback = str(legal.get("text") or retrieval.get("text") or evidence.get("quoted_text") or "")
    if store is None or not legal.get("text_span_ids"):
        return _normalize_embedding_text(fallback, store)
    spans = {str(row.get("text_span_id")): row for row in getattr(store, "page_text_spans", ())}
    raw_rows = tuple(getattr(store, "raw_source_spans", ()) or ())
    repaired: list[tuple[str, dict]] = []
    for span_id in legal.get("text_span_ids") or ():
        span = spans.get(str(span_id))
        if not span:
            continue
        raw = _raw_row_for_span(span, raw_rows)
        repaired.append((str(raw.get("raw_text")), raw) if raw else (str(span.get("text") or ""), {}))
    if not repaired:
        return _normalize_embedding_text(fallback, store)
    stitched = repaired[0][0]
    previous_text, previous = repaired[0]
    for text, current in repaired[1:]:
        if _join_source_boundary(previous_text, previous, text, current):
            stitched += text
        else:
            stitched += "\n" + text
        previous_text, previous = text, current
    return _normalize_embedding_text(stitched, store) or _normalize_embedding_text(fallback, store)


def _embedding_texts(store, records: list[tuple[dict, dict, dict]]) -> tuple[str, ...]:
    return tuple(_embedding_text(row, evidence, legal, store=store) for row, evidence, legal in records)


def _join_source_boundary(previous_text: str, previous: dict, current_text: str, current: dict) -> bool:
    """Join a verified line-break soft hyphen before retrieval normalization."""
    if not previous_text.endswith("\u00ad") or not current_text[:1].isalnum():
        return False
    return bool(
        previous.get("raw_stream_id")
        and previous.get("raw_stream_id") == current.get("raw_stream_id")
        and isinstance(previous.get("raw_text_end"), int)
        and isinstance(current.get("raw_text_start"), int)
        and 0 <= current["raw_text_start"] - previous["raw_text_end"] <= 1
    )


def _raw_row_for_span(span: dict, raw_rows: tuple[dict, ...]) -> dict | None:
    source_id = span.get("source_document_id")
    page = span.get("page_number")
    object_id = str(span.get("source_object_id") or "")
    block = object_id.rsplit("::", 1)[-1] if object_id else ""
    candidates = [
        row
        for row in raw_rows
        if row.get("source_document_id") == source_id
        and row.get("page_number") == page
        and (not block or (block.isdigit() and str(row.get("block_index")) == str(int(block))))
    ]
    expected = str(span.get("text") or "")
    return next((row for row in candidates if str(row.get("semantic_text") or "") == expected), candidates[0] if len(candidates) == 1 else None)


def _normalize_embedding_text(value: str, store=None) -> str:
    normalizer = getattr(getattr(getattr(store, "config", None), "strategy", None), "embedding_text_normalizer", None)
    if normalizer is not None:
        return str(normalizer(value))
    return _source_aware_soft_hyphens(value)


def _source_aware_soft_hyphens(value: str) -> str:
    value = " ".join(str(value or "").replace("\xa0", " ").split())
    chars: list[str] = []
    for index, character in enumerate(value):
        if character != "\u00ad":
            chars.append(character)
            continue
        previous = value[index - 1] if index else ""
        following = value[index + 1] if index + 1 < len(value) else ""
        if previous.isalnum() and following.isalnum():
            chars.append("-")
    return "".join(chars)


def _source_identity(store, records, model: DenseModelIdentity, embedding_text_digest: str) -> str:
    retrieval_digest = _digest([row for row, _, _ in records])
    embedding_source_digest = _digest(
        [{"retrieval": row, "evidence": evidence, "legal_unit": legal} for row, evidence, legal in records]
    )
    artifact_digest = getattr(store.config, "artifact_set_digest", None)
    return _digest(
        {
            "corpus_id": getattr(store.config, "corpus_id", None),
            "artifact_digest": artifact_digest,
            "manifest_digest": getattr(store.config, "manifest_digest", None),
            "retrieval_units_digest": retrieval_digest,
            "embedding_source_digest": embedding_source_digest,
            "embedding_text_digest": embedding_text_digest,
            "embedding_text_policy": EMBEDDING_TEXT_POLICY,
            "model": model.as_dict(),
            "record_count": len(records),
        }
    )


def _validate_batch(batch: DenseEmbeddingBatch, expected: int) -> tuple[tuple[float, ...], ...]:
    if not isinstance(batch, DenseEmbeddingBatch) or len(batch.vectors) != expected:
        raise DenseError("embedding_count_mismatch")
    identity = DenseModelIdentity.from_value(batch.identity)
    if (
        identity.dimension != DENSE_DIMENSION
        or identity.dtype != DENSE_DTYPE
        or identity.normalization != DENSE_NORMALIZATION
        or identity.pooling != DENSE_POOLING
        or identity.max_length != DENSE_MAX_LENGTH
        or identity.truncation_policy != DENSE_TRUNCATION_POLICY
    ):
        raise DenseError("embedding_contract_invalid")
    if (
        not _valid_digest(identity.tokenizer_sha256)
        or not _valid_digest(identity.model_sha256)
        or not _valid_digest(identity.pooling_config_sha256)
    ):
        raise DenseError("model_file_digest_missing")
    for vector in batch.vectors:
        _validate_vector(vector, identity.dimension)
        norm = math.sqrt(math.fsum(value * value for value in vector))
        if not math.isfinite(norm) or abs(norm - 1.0) > 1e-3:
            raise DenseError("embedding_not_normalized")
    return batch.vectors


def _parse_vectors(value: object, dimension: int) -> tuple[tuple[float, ...], ...]:
    if not isinstance(value, list):
        raise DenseUnavailable("worker_vectors_missing")
    vectors = tuple(tuple(float(item) for item in vector) for vector in value if isinstance(vector, list))
    if len(vectors) != len(value):
        raise DenseUnavailable("worker_vectors_malformed")
    for vector in vectors:
        _validate_vector(vector, dimension)
    return vectors


def _validate_vector(vector: tuple[float, ...], dimension: int) -> None:
    if len(vector) != dimension or any(not math.isfinite(value) for value in vector):
        raise DenseError("embedding_vector_invalid")


def _valid_digest(value: str | None) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _model_snapshot_path() -> Path | None:
    configured = os.environ.get("TJIPTO_DENSE_MODEL_DIR")
    if configured:
        path = Path(configured)
        return path if path.is_dir() else None
    hub_root = os.environ.get("HF_HOME")
    if hub_root:
        candidate = Path(hub_root) / "hub" / "models--BAAI--bge-m3" / "snapshots" / MODEL_REVISION
    else:
        candidate = Path.home() / ".cache" / "huggingface" / "hub" / "models--BAAI--bge-m3" / "snapshots" / MODEL_REVISION
    return candidate if candidate.is_dir() else None


def _files_digest(path: Path | None, names: tuple[str, ...]) -> str | None:
    if path is None or not path.exists():
        return None
    root = path if path.is_dir() else path.parent
    paths = [path / name for name in names if (path / name).is_file()] if path.is_dir() else [path]
    if not paths:
        return None
    digest = hashlib.sha256()
    for file_path in sorted(paths):
        try:
            digest.update(file_path.relative_to(root).as_posix().encode("utf-8"))
            digest.update(file_path.read_bytes())
        except OSError:
            return None
    return digest.hexdigest()


def _model_identity_digests(model_dir: Path | None) -> tuple[str | None, str | None, str | None]:
    if model_dir is None:
        return None, None, None
    tokenizer = _files_digest(
        model_dir,
        ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "sentencepiece.bpe.model", "vocab.txt"),
    )
    model = _files_digest(model_dir, ("config.json", "pytorch_model.bin", "model.safetensors", "model.safetensors.index.json"))
    pooling_path = model_dir / "1_Pooling" / "config.json"
    pooling = _sha256_file(pooling_path)
    return tokenizer, model, pooling


def _sha256_file(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    except OSError:
        return None


def _row_bytes(row: dict) -> bytes:
    return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _unavailable(reason: str) -> dict:
    return {"status": "dense_unavailable", "route": "dense_unavailable", "matches": (), "reason": reason}


__all__ = [
    "DENSE_BATCH_SIZE",
    "DENSE_DIMENSION",
    "DENSE_MAX_LENGTH",
    "DENSE_POOLING",
    "DENSE_TRUNCATION_POLICY",
    "DenseEmbeddingBatch",
    "DenseIndex",
    "DenseModelIdentity",
    "DenseUnavailable",
    "LocalDenseProvider",
    "MODEL_ID",
    "MODEL_REVISION",
    "EMBEDDING_TEXT_POLICY",
    "dense_index_for_store",
    "dense_search",
]
