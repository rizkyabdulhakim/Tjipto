from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
from threading import RLock
from collections import OrderedDict

from tjipto.contracts.artifacts import ARTIFACT_ALLOWED_FIELDS, ARTIFACT_OPTIONAL_FIELDS, COMMON_ARTIFACT_FIELDS, CURRENT_ARTIFACT_SCHEMA, FORBIDDEN_ARTIFACT_FIELDS, MINIMUM_ARTIFACT_FIELDS
from tjipto.contracts.evidence import exact_quote_support_reason, source_lineage_reason
from tjipto.core.manifest import ALLOWED_ARTIFACT_ORIGINS, verified_file_bytes
from tjipto.core.manifest import artifact_set_digest as compute_artifact_set_digest
from tjipto.ingestion.pdf.fingerprint import extractor_fingerprint


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
    access_mode: str = "verified"
    canonical_build_eligible: bool = True


@dataclass(frozen=True)
class CorpusSemanticAttestation:
    status: str
    violation_codes: tuple[str, ...] = ()


_SNAPSHOT_TOKEN = object()


@dataclass(frozen=True)
class ValidatedCorpusSnapshot:
    config: object
    artifacts: Mapping[str, object]
    readiness: CorpusReadiness
    semantic_attestation: CorpusSemanticAttestation
    corpus_id: str
    schema_version: int
    manifest_digest: str
    artifact_set_digest: str
    _token: object = None

    def __post_init__(self) -> None:
        if self._token is not _SNAPSHOT_TOKEN:
            raise TypeError("validated_snapshot_requires_publication")


VerifiedCorpusSnapshot = ValidatedCorpusSnapshot


class CorpusPublicationService:
    def verify_and_publish(self, config) -> ValidatedCorpusSnapshot:
        manifest, manifest_digest = _read_trusted_manifest(config)
        build_eligible = _canonical_extractor_matches(manifest)
        access_mode = "verified" if build_eligible else "verified_read_only"
        runtime_required = tuple(config.setting("runtime_required_artifacts"))
        artifacts = _verify_artifacts(config.manifest_path.parent.resolve(), manifest, runtime_required)
        semantic_attestation = _runtime_attestation(manifest, artifacts)
        retained_paths = {manifest[logical_key] for logical_key in runtime_required}
        retained_artifacts = {path: artifacts[path] for path in retained_paths}
        # Keep only the runtime projection while freezing it.  The verifier
        # decoded every manifest artifact for attestation; retaining that
        # full map during the deep immutable conversion needlessly raises RSS.
        del artifacts
        frozen_artifacts = _freeze(retained_artifacts)
        del retained_artifacts
        frozen_manifest = _freeze(manifest)
        verified = replace(
            config,
            manifest=frozen_manifest,
            settings=_freeze(config.settings or {}),
            verified_artifacts=frozen_artifacts,
            manifest_digest=manifest_digest,
            artifact_set_digest=compute_artifact_set_digest(manifest),
            artifact_access_mode=access_mode,
            canonical_build_eligible=build_eligible,
        )
        artifact_set_digest = compute_artifact_set_digest(manifest)
        readiness = CorpusReadiness(
            True,
            manifest_digest,
            artifact_set_digest,
            access_mode=access_mode,
            canonical_build_eligible=build_eligible,
        )
        return ValidatedCorpusSnapshot(
            config=verified,
            artifacts=frozen_artifacts,
            readiness=readiness,
            semantic_attestation=semantic_attestation,
            corpus_id=config.corpus_id,
            schema_version=manifest["schema_version"],
            manifest_digest=manifest_digest,
            artifact_set_digest=artifact_set_digest,
            _token=_SNAPSHOT_TOKEN,
        )


class VerifiedCorpusRepository:
    """Publishes a corpus only after trusted, semantic, read-once verification."""

    # ponytail: process-local immutable snapshots; use an external cache only if
    # publication must be shared across processes.
    _published_snapshots: OrderedDict[tuple[str, str, str], ValidatedCorpusSnapshot] = OrderedDict()
    _published_lock = RLock()
    _published_snapshot_limit = 1

    @classmethod
    def clear_shared_cache(cls) -> None:
        with cls._published_lock:
            cls._published_snapshots.clear()

    def __init__(self, registry):
        self.registry = registry
        self.publication = CorpusPublicationService()
        self.load_count = 0
        self._validated: dict[str, tuple[ValidatedCorpusSnapshot, tuple[tuple[str, int, int], ...]]] = {}

    def load(self, corpus_id: str) -> VerifiedCorpusSnapshot:
        config = self.registry.resolve(corpus_id)
        if config is None:
            raise CorpusIntegrityError(self.registry.error_code or "corpus_load_failure")
        manifest, manifest_digest = _read_trusted_manifest(config)
        final_dir = config.manifest_path.parent.resolve()
        validated_entry = self._validated.get(corpus_id)
        if validated_entry is not None and validated_entry[0].manifest_digest == manifest_digest:
            signature = _artifact_stat_signature(final_dir, manifest)
            if signature == validated_entry[1]:
                return validated_entry[0]
        _verify_artifact_integrity(final_dir, manifest)
        cache_key = (str(config.manifest_path.resolve()), corpus_id, manifest_digest)
        with self._published_lock:
            published = self._published_snapshots.get(cache_key)
            if published is not None:
                self._published_snapshots.move_to_end(cache_key)
                self._validated[corpus_id] = (published, _artifact_stat_signature(final_dir, manifest))
                return published
            snapshot = self.publication.verify_and_publish(config)
            self._published_snapshots[cache_key] = snapshot
            self._published_snapshots.move_to_end(cache_key)
            while len(self._published_snapshots) > self._published_snapshot_limit:
                self._published_snapshots.popitem(last=False)
            self.load_count += 1
            self._validated[corpus_id] = (snapshot, _artifact_stat_signature(final_dir, manifest))
            return snapshot


def _artifact_stat_signature(final_dir: Path, manifest: dict) -> tuple[tuple[str, int, int], ...]:
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise CorpusIntegrityError("manifest_files_missing")
    signature: list[tuple[str, int, int]] = []
    for rel in sorted(files):
        path = _contained_path(final_dir, rel)
        try:
            stat = path.stat()
        except OSError as error:
            raise CorpusIntegrityError("artifact_missing") from error
        signature.append((rel, stat.st_size, stat.st_mtime_ns))
    return tuple(signature)


def _read_manifest(config) -> tuple[dict, str]:
    try:
        manifest_bytes = config.manifest_path.read_bytes()
    except OSError as error:
        raise CorpusIntegrityError("manifest_missing") from error
    manifest_digest = sha256(manifest_bytes).hexdigest()
    expected_digest = config.setting("manifest_sha256")
    if manifest_digest != expected_digest:
        raise CorpusIntegrityError("trusted_manifest_mismatch")
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CorpusIntegrityError("malformed_manifest") from error
    return manifest, manifest_digest


def _read_trusted_manifest(config) -> tuple[dict, str]:
    expected_digest = config.setting("manifest_sha256")
    if not isinstance(expected_digest, str) or len(expected_digest) != 64:
        raise CorpusIntegrityError("trusted_manifest_missing")
    manifest, manifest_digest = _read_manifest(config)
    if not isinstance(manifest, dict) or manifest.get("corpus_id") != config.corpus_id:
        raise CorpusIntegrityError("manifest_identity_mismatch")
    strategy = getattr(config, "strategy", None)
    if strategy is None:
        raise CorpusIntegrityError("unsupported_corpus_strategy")
    contract = strategy.contract
    if contract and (
        manifest.get("schema_version") != contract.schema_version
        or manifest.get("contract_id") != contract.contract_id
        or manifest.get("contract_version") != contract.contract_version
        or manifest.get("contract_fingerprint") != contract.contract_fingerprint
    ):
        raise CorpusIntegrityError("contract_fingerprint_mismatch")
    return manifest, manifest_digest


def _canonical_extractor_matches(manifest: dict) -> bool:
    expected = manifest.get("extractor_fingerprint")
    if expected is None:
        return True
    try:
        return expected == extractor_fingerprint()
    except RuntimeError:
        return False


def require_canonical_build_environment(config) -> None:
    manifest, _ = _read_trusted_manifest(config)
    if not _canonical_extractor_matches(manifest):
        raise CorpusIntegrityError("extractor_fingerprint_mismatch")


def _manifest_digest(config) -> str:
    return _read_trusted_manifest(config)[1]


def _run_semantic_validator(config, manifest: dict, artifacts: dict[str, object]) -> CorpusSemanticAttestation:
    validator = getattr(config.strategy, "semantic_validator", None)
    if validator is None:
        return CorpusSemanticAttestation("not_configured")
    try:
        logical_artifacts = {
            record["logical_key"]: artifacts[rel]
            for rel, record in manifest["files"].items()
            if rel in artifacts
        }
        violations = tuple(validator(config.manifest_path.parent.resolve(), logical_artifacts))
    except (KeyError, TypeError, ValueError) as error:
        raise CorpusIntegrityError("semantic_validator_unavailable") from error
    return CorpusSemanticAttestation("passed" if not violations else "failed", violations)


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
        _validate_record_identity(logical_key, rel, record, expected_schema=manifest.get("schema_version"))
        path = _contained_path(final_dir, rel)
        data, integrity_error = verified_file_bytes(path, record)
        if integrity_error:
            raise CorpusIntegrityError(integrity_error)
        if data is None:
            raise CorpusIntegrityError("artifact_missing")
        if logical_key in required:
            loaded[rel] = _parse_and_validate(data, record, logical_key)
    return loaded


def _runtime_attestation(manifest: dict, artifacts: dict[str, object]) -> CorpusSemanticAttestation:
    report = artifacts.get(manifest.get("validation_report", ""))
    expected = compute_artifact_set_digest(manifest, exclude=(manifest.get("validation_report"),))
    if (
        not isinstance(report, dict)
        or report.get("status") not in {"pass", "valid"}
        or report.get("validated_artifact_set_digest") != expected
    ):
        raise CorpusIntegrityError("runtime_validation_attestation_missing")
    return CorpusSemanticAttestation("passed")


def _verify_artifact_integrity(final_dir: Path, manifest: dict) -> None:
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise CorpusIntegrityError("manifest_files_missing")
    for rel, record in sorted(files.items()):
        if not isinstance(rel, str) or not isinstance(record, dict):
            raise CorpusIntegrityError("manifest_malformed")
        logical_key = record.get("logical_key")
        if not isinstance(logical_key, str) or manifest.get(logical_key) != rel:
            raise CorpusIntegrityError("semantic_artifact_identity_mismatch")
        _validate_record_identity(logical_key, rel, record, expected_schema=manifest.get("schema_version"))
        _, integrity_error = verified_file_bytes(_contained_path(final_dir, rel), record)
        if integrity_error:
            raise CorpusIntegrityError(integrity_error)


def _validate_record_identity(logical_key: str, rel: str, record: dict, *, expected_schema: int | None = None) -> None:
    expected_format = "json" if rel.endswith(".json") else "jsonl"
    if record.get("artifact_schema") != (expected_schema or CURRENT_ARTIFACT_SCHEMA):
        raise CorpusIntegrityError("artifact_schema_mismatch")
    if record.get("logical_key") != logical_key or record.get("artifact_kind") != logical_key or record.get("format") != expected_format:
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
    bbox_ids |= {character["character_bbox_id"] for word in rows("word_bboxes") for character in word.get("characters") or ()}
    relation_ids = {row["relation_id"] for row in rows("article_amendment_relations")}
    for row in rows("retrieval_units"):
        if row["evidence_id"] not in evidence_ids or row["legal_unit_id"] not in unit_ids or row["chunk_id"] not in chunk_ids:
            raise CorpusIntegrityError("semantic_cross_reference_unresolved")
    for row in rows("graph_edges"):
        if row["source_id"] not in node_ids or row["target_id"] not in node_ids:
            raise CorpusIntegrityError("semantic_cross_reference_unresolved")
        if row.get("edge_type") in {"MODIFIES", "DELETES", "RENAMES", "RENUMBERED_TO"} and row.get("relation_id") not in relation_ids:
            raise CorpusIntegrityError("semantic_cross_reference_unresolved")
    for row in rows("evidence_registry"):
        if (
            row["legal_unit_id"] not in unit_ids
            or row["source_document_id"] not in source_ids
            or any(span_id not in span_ids for span_id in row["text_span_ids"])
            or any(bbox_id not in bbox_ids for bbox_id in row["bbox_refs"])
            or not isinstance(row.get("runtime_loadable"), bool)
            or not isinstance(row.get("evidence_owner_kind"), str)
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
        forbidden = FORBIDDEN_ARTIFACT_FIELDS.get(logical_key, frozenset())
        if any(forbidden.intersection(row) for row in value):
            raise CorpusIntegrityError("mixed_schema_contract")
        allowed = set(ARTIFACT_ALLOWED_FIELDS.get(logical_key, ())) or (
            set(minimum_fields) | set(ARTIFACT_OPTIONAL_FIELDS.get(logical_key, ())) | set(COMMON_ARTIFACT_FIELDS)
        )
        if declared_fields is not None and not set(declared_fields) <= allowed:
            raise CorpusIntegrityError("unknown_field")
        if logical_key in MINIMUM_ARTIFACT_FIELDS and any(field not in allowed for row in value for field in row):
            raise CorpusIntegrityError("unknown_field")
        if declared_fields is not None and not set(minimum_fields).issubset(declared_fields):
            raise CorpusIntegrityError("artifact_contract_weakened")
        required_fields = tuple(minimum_fields)
        if any(field not in row for row in value for field in required_fields):
            raise CorpusIntegrityError("missing_required_field")
    elif not isinstance(value, dict):
        raise CorpusIntegrityError("artifact_shape_invalid")
    return value


def _validate_exact_evidence(manifest: dict, artifacts: dict[str, object]) -> None:
    evidence = _rows_for(manifest, artifacts, "evidence_registry")
    spans = {row["text_span_id"]: row for row in _rows_for(manifest, artifacts, "page_text_spans")}
    bboxes = {row["bbox_id"]: row for row in _rows_for(manifest, artifacts, "bbox_registry")}
    sources = {row["source_document_id"]: row for row in _rows_for(manifest, artifacts, "source_documents")}
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
        lineage = source_lineage_reason(
            evidence=row,
            source_documents_by_id=sources,
            spans_by_id=spans,
            bboxes_by_id=bboxes,
        )
        if lineage:
            raise CorpusIntegrityError("evidence_source_lineage_invalid")


def _rows_for(manifest: dict, artifacts: dict[str, object], logical_key: str) -> list[dict]:
    value = artifacts.get(manifest.get(logical_key, ""), ())
    return value if isinstance(value, list) else []


def _freeze(value):
    if isinstance(value, dict):
        for key in value:
            dict.__setitem__(value, key, _freeze(value[key]))
        return FrozenDict(value)
    if isinstance(value, list):
        for index, item in enumerate(value):
            value[index] = _freeze(item)
        return tuple(value)
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
