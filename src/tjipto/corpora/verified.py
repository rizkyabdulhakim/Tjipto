from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
from threading import RLock

from tjipto.contracts.artifacts import MINIMUM_ARTIFACT_FIELDS
from tjipto.contracts.evidence import exact_quote_support_reason
from tjipto.core.manifest import ALLOWED_ARTIFACT_ORIGINS, verified_file_bytes


class CorpusIntegrityError(ValueError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class CorpusReadiness:
    ready: bool
    manifest_digest: str | None = None
    artifact_set_digest: str | None = None
    error_code: str | None = None


@dataclass(frozen=True)
class VerifiedCorpusSnapshot:
    config: object
    artifacts: Mapping[str, object]
    readiness: CorpusReadiness


class VerifiedCorpusRepository:
    """Publishes a corpus only after trusted, semantic, read-once verification."""

    def __init__(self, registry):
        self.registry = registry
        self._snapshots: dict[str, VerifiedCorpusSnapshot] = {}
        self._lock = RLock()
        self.load_count = 0

    def load(self, corpus_id: str) -> VerifiedCorpusSnapshot:
        with self._lock:
            cached = self._snapshots.get(corpus_id)
            if cached is not None:
                return cached
            config = self.registry.resolve(corpus_id)
            if config is None:
                raise CorpusIntegrityError(self.registry.error_code or "corpus_load_failure")
            snapshot = _load_snapshot(config)
            self._snapshots[corpus_id] = snapshot
            self.load_count += 1
            return snapshot


def _load_snapshot(config) -> VerifiedCorpusSnapshot:
    expected_digest = config.setting("manifest_sha256")
    if not isinstance(expected_digest, str) or len(expected_digest) != 64:
        raise CorpusIntegrityError("trusted_manifest_missing")
    try:
        manifest_bytes = config.manifest_path.read_bytes()
    except OSError as error:
        raise CorpusIntegrityError("manifest_missing") from error
    manifest_digest = sha256(manifest_bytes).hexdigest()
    if manifest_digest != expected_digest:
        raise CorpusIntegrityError("trusted_manifest_mismatch")
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CorpusIntegrityError("malformed_manifest") from error
    if not isinstance(manifest, dict) or manifest.get("corpus_id") != config.corpus_id:
        raise CorpusIntegrityError("manifest_identity_mismatch")
    if manifest.get("schema_version") != 4:
        raise CorpusIntegrityError("unsupported_schema")
    artifacts = _verify_artifacts(config.manifest_path.parent.resolve(), manifest, config.setting("runtime_required_artifacts"))
    try:
        _validate_cross_artifact_references(manifest, artifacts)
    except (KeyError, TypeError) as error:
        raise CorpusIntegrityError("artifact_semantic_invalid") from error
    frozen_artifacts = _freeze(artifacts)
    frozen_manifest = _freeze(manifest)
    verified = replace(
        config,
        manifest=frozen_manifest,
        settings=_freeze(config.settings or {}),
        verified_artifacts=frozen_artifacts,
        manifest_digest=manifest_digest,
        artifact_set_digest=_artifact_digest(manifest),
    )
    readiness = CorpusReadiness(True, manifest_digest, verified.artifact_set_digest)
    return VerifiedCorpusSnapshot(verified, frozen_artifacts, readiness)


def _verify_artifacts(final_dir: Path, manifest: dict, required_value: object) -> dict[str, object]:
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise CorpusIntegrityError("manifest_files_missing")
    if not isinstance(required_value, (list, tuple)) or not required_value:
        raise CorpusIntegrityError("runtime_artifact_contract_missing")
    required = tuple(required_value)
    for logical_key in required:
        if not isinstance(logical_key, str) or not isinstance(manifest.get(logical_key), str):
            raise CorpusIntegrityError("required_artifact_missing")
    loaded: dict[str, object] = {}
    for rel, record in sorted(files.items()):
        if not isinstance(rel, str) or not isinstance(record, dict):
            raise CorpusIntegrityError("manifest_malformed")
        logical_key = record.get("logical_key")
        if not isinstance(logical_key, str) or manifest.get(logical_key) != rel:
            raise CorpusIntegrityError("semantic_artifact_identity_mismatch")
        _validate_record_identity(logical_key, rel, record)
        path = _contained_path(final_dir, rel)
        data, integrity_error = verified_file_bytes(path, record)
        if integrity_error:
            raise CorpusIntegrityError(integrity_error)
        if data is None:
            raise CorpusIntegrityError("artifact_missing")
        loaded[rel] = _parse_and_validate(data, record, logical_key)
    _validate_exact_evidence(manifest, loaded)
    return loaded


def _validate_record_identity(logical_key: str, rel: str, record: dict) -> None:
    expected_format = "json" if rel.endswith(".json") else "jsonl"
    if (
        record.get("logical_key") != logical_key
        or record.get("artifact_kind") != logical_key
        or record.get("artifact_schema") != 4
        or record.get("format") != expected_format
    ):
        raise CorpusIntegrityError("semantic_artifact_identity_mismatch")
    if record.get("origin") not in ALLOWED_ARTIFACT_ORIGINS or not all(
        isinstance(record.get(field), str) and record[field].strip() for field in ("producer", "build_stage")
    ):
        raise CorpusIntegrityError("artifact_provenance_invalid")


def _validate_cross_artifact_references(manifest: dict, artifacts: dict[str, object]) -> None:
    def rows(logical_key: str) -> list[dict]:
        value = artifacts.get(manifest.get(logical_key, ""), ())
        return value if isinstance(value, list) else []

    evidence_ids = {row["evidence_id"] for row in rows("evidence_registry")}
    unit_ids = {row["legal_unit_id"] for row in rows("legal_units")}
    chunk_ids = {row["chunk_id"] for row in rows("chunks")}
    node_ids = {row["node_id"] for row in rows("graph_nodes")}
    source_ids = {row["source_document_id"] for row in rows("source_documents")}
    span_ids = {row["text_span_id"] for row in rows("page_text_spans")}
    bbox_ids = {row["bbox_id"] for row in rows("bbox_registry")} | {row["word_bbox_id"] for row in rows("word_bboxes")}
    for row in rows("retrieval_units"):
        if row["evidence_id"] not in evidence_ids or row["legal_unit_id"] not in unit_ids or row["chunk_id"] not in chunk_ids:
            raise CorpusIntegrityError("semantic_cross_reference_unresolved")
    for row in rows("graph_edges"):
        if row["source_id"] not in node_ids or row["target_id"] not in node_ids:
            raise CorpusIntegrityError("semantic_cross_reference_unresolved")
    for row in rows("evidence_registry"):
        if (
            row["legal_unit_id"] not in unit_ids
            or row["source_document_id"] not in source_ids
            or any(span_id not in span_ids for span_id in row["text_span_ids"])
            or any(bbox_id not in bbox_ids for bbox_id in row["bbox_refs"])
        ):
            raise CorpusIntegrityError("semantic_cross_reference_unresolved")


def _parse_and_validate(data: bytes, record: dict, logical_key: str) -> object:
    try:
        text = data.decode("utf-8")
        value = json.loads(text) if record["format"] == "json" else [json.loads(line) for line in text.splitlines() if line]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CorpusIntegrityError("artifact_malformed") from error
    if record["format"] == "jsonl":
        if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
            raise CorpusIntegrityError("artifact_shape_invalid")
        primary_id = record.get("primary_id")
        if not isinstance(primary_id, str):
            raise CorpusIntegrityError("artifact_primary_id_missing")
        ids = [row.get(primary_id) for row in value]
        if any(not isinstance(row_id, str) or not row_id for row_id in ids):
            raise CorpusIntegrityError("artifact_primary_id_missing")
        if len(ids) != len(set(ids)):
            raise CorpusIntegrityError("artifact_primary_id_duplicate")
        declared_fields = record.get("required_fields")
        if declared_fields is not None and (
            not isinstance(declared_fields, list) or any(not isinstance(field, str) for field in declared_fields)
        ):
            raise CorpusIntegrityError("artifact_contract_malformed")
        minimum_fields = MINIMUM_ARTIFACT_FIELDS.get(logical_key, ())
        if declared_fields is not None and not set(minimum_fields).issubset(declared_fields):
            raise CorpusIntegrityError("artifact_contract_weakened")
        required_fields = tuple(minimum_fields) + tuple(declared_fields or ())
        if any(field not in row for row in value for field in required_fields):
            raise CorpusIntegrityError("artifact_required_field_missing")
    elif not isinstance(value, dict):
        raise CorpusIntegrityError("artifact_shape_invalid")
    return value


def _validate_exact_evidence(manifest: dict, artifacts: dict[str, object]) -> None:
    evidence = _rows_for(manifest, artifacts, "evidence_registry")
    spans = {row["text_span_id"]: row for row in _rows_for(manifest, artifacts, "page_text_spans")}
    bboxes = {row["bbox_id"]: row for row in _rows_for(manifest, artifacts, "bbox_registry")}
    for bbox in bboxes.values():
        if bbox.get("viewer_highlightable") is not True:
            continue
        fields = (
            "coordinate_space",
            "coordinate_origin",
            "page_width",
            "page_height",
            "page_rotation",
            "page_box_basis",
            "transform_version",
        )
        if any(bbox.get(field) is None for field in fields):
            raise CorpusIntegrityError("coordinate_metadata_missing")
        if bbox.get("page_rotation") != 0 or bbox.get("coordinate_origin") != "top_left" or bbox.get("page_box_basis") != "media_box":
            raise CorpusIntegrityError("coordinate_metadata_invalid")
    for row in evidence:
        if (
            row.get("exactness") != "exact"
            and row.get("citable") is not True
            and row.get("citation_final") is not True
            and row.get("viewer_highlightable") is not True
        ):
            continue
        reason = exact_quote_support_reason(
            quoted_text=row.get("quoted_text"),
            source_document_id=row.get("source_document_id"),
            page_numbers=row.get("page_numbers") or (),
            text_span_ids=row.get("text_span_ids") or (),
            bbox_refs=row.get("bbox_refs") or (),
            spans_by_id=spans,
            bboxes_by_id=bboxes,
        )
        if reason:
            raise CorpusIntegrityError("evidence_quote_source_mismatch")


def _rows_for(manifest: dict, artifacts: dict[str, object], logical_key: str) -> list[dict]:
    value = artifacts.get(manifest.get(logical_key, ""), ())
    return value if isinstance(value, list) else []


def _freeze(value):
    if isinstance(value, dict):
        return FrozenDict({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


class FrozenDict(dict):
    def _immutable(self, *args, **kwargs):
        raise TypeError("immutable corpus snapshot")

    __setitem__ = __delitem__ = __ior__ = clear = pop = popitem = setdefault = update = _immutable


def _contained_path(final_dir: Path, rel: str) -> Path:
    raw = Path(rel)
    if raw.is_absolute():
        raise CorpusIntegrityError("artifact_path_invalid")
    resolved = (final_dir / raw).resolve()
    if not resolved.is_relative_to(final_dir):
        raise CorpusIntegrityError("artifact_path_invalid")
    return resolved


def _artifact_digest(manifest: dict) -> str:
    rows = [(name, row.get("sha256"), row.get("bytes")) for name, row in sorted(manifest["files"].items())]
    return sha256(json.dumps(rows, separators=(",", ":")).encode("utf-8")).hexdigest()
