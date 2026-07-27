"""Small, local-only telemetry boundary for operational signals."""

from __future__ import annotations

import json
import math
import os
import re
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tjipto.corpora.registry import CorpusRegistry


EVENT_ATTRIBUTES = {
    "corpus_load": frozenset({"corpus_id", "status"}),
    "integrity_failure": frozenset({"corpus_id", "reason_code"}),
    "http_request": frozenset({"request_id", "method", "route", "status_code", "latency_ms"}),
    "retrieval_route": frozenset({"corpus_id", "route", "status"}),
    "ci_gate": frozenset({"gate", "status", "duration_ms"}),
    "release_validation": frozenset({"status", "forbidden_entry_count", "archive_sha256"}),
}
SENSITIVE_ATTRIBUTES = frozenset({"query", "query_text", "text", "quote", "quoted_text", "token", "tokens", "credential", "password", "secret", "path", "file_path", "email", "user_id"})
MAX_ATTRIBUTE_LENGTH = 96
_REQUEST_ID = re.compile(r"[0-9a-f]{32}")
_ARCHIVE_SHA256 = re.compile(r"[0-9a-f]{64}")

_HTTP_ROUTES = frozenset(
    {
        "health",
        "legal.ask",
        "legal.search",
        "legal.citation",
        "legal.viewer",
        "legal.pdf",
        "legal.bookmarks",
        "legal.capabilities",
        "legacy.ask",
        "legacy.search",
        "legacy.citation",
        "legacy.viewer",
        "legacy.pdf",
        "legacy.bookmarks",
        "legacy.capabilities",
        "not_found",
    }
)
_RETRIEVAL_ROUTES = frozenset(
    {
        "bm25",
        "citation_not_found",
        "exact",
        "metadata",
        "metadata_not_found",
        "metadata_scope_unresolved",
        "no_results",
        "relation",
        "relation_not_found",
        "scope_unresolved",
        "structural_navigation",
        "structure_list",
        "structured",
        "structured_not_found",
        "unsupported_corpus",
    }
)
_RETRIEVAL_STATUS = frozenset({"citation_not_found", "found", "invalid_filter", "no_results", "unsupported_corpus"})
_INTEGRITY_REASONS = frozenset(
    {
        "artifact_contract_malformed", "artifact_contract_weakened", "artifact_malformed", "artifact_missing", "artifact_path_invalid",
        "artifact_primary_id_duplicate", "artifact_primary_id_missing", "artifact_provenance_invalid", "artifact_schema_mismatch",
        "artifact_shape_invalid", "contract_fingerprint_mismatch", "coordinate_metadata_invalid", "coordinate_metadata_missing",
        "corpus_id_mismatch", "evidence_quote_source_mismatch", "evidence_source_lineage_invalid", "extractor_fingerprint_mismatch",
        "malformed_manifest", "malformed_registry", "malformed_registry_entry", "manifest_files_missing", "manifest_identity_mismatch",
        "manifest_malformed", "manifest_missing", "manifest_path_violation", "missing_required_field", "mixed_schema_contract",
        "registry_unavailable", "required_artifact_missing", "runtime_artifact_contract_missing", "runtime_validation_attestation_missing",
        "semantic_artifact_identity_mismatch", "semantic_cross_reference_unresolved", "semantic_validator_unavailable",
        "trusted_manifest_mismatch", "trusted_manifest_missing", "unknown_corpus", "unknown_field", "unsupported_schema",
    }
)
_CI_GATES = frozenset(
    {
        "compileall", "unittest", "pytest", "retrieval_evaluation", "artifact_validate", "artifact_rebuild", "ruff", "mypy",
        "bandit", "pip_check", "pip_audit", "clean_tree", "release_validation", "web_test", "web_lint", "web_typecheck",
        "web_build", "web_smoke",
    }
)


def event_record(event: str, *, registry: CorpusRegistry | None = None, **attributes: Any) -> dict[str, Any]:
    """Return a bounded record containing only the event's approved fields."""
    allowed = EVENT_ATTRIBUTES[event]
    values = {key: value for key, value in attributes.items() if key in allowed and key not in SENSITIVE_ATTRIBUTES and _safe_value(value)}
    if set(values) != allowed:
        missing = sorted(allowed - set(values))
        raise ValueError(f"telemetry attributes missing or unsafe: {', '.join(missing)}")
    if not all(_valid_attribute(event, key, value, registry) for key, value in values.items()):
        raise ValueError("telemetry attributes are outside their approved values")
    return {"event": event, "attributes": values}


def _safe_value(value: Any) -> bool:
    return (
        type(value) in {bool, int}
        or (type(value) is float and math.isfinite(value))
        or (type(value) is str and 0 < len(value) <= MAX_ATTRIBUTE_LENGTH)
    )


def _valid_attribute(event: str, key: str, value: Any, registry: CorpusRegistry | None = None) -> bool:
    if key == "corpus_id":
        from tjipto.corpora.registry import CorpusRegistry

        return value == "unknown" or value in (registry or CorpusRegistry()).corpus_ids()
    if event == "http_request":
        return {
            "request_id": type(value) is str and bool(_REQUEST_ID.fullmatch(value)),
            "method": value in {"GET", "POST", "OPTIONS"},
            "route": value in _HTTP_ROUTES,
            "status_code": type(value) is int and 100 <= value <= 599,
            "latency_ms": type(value) in {int, float} and value >= 0,
        }[key]
    if event == "retrieval_route":
        return value in (_RETRIEVAL_ROUTES if key == "route" else _RETRIEVAL_STATUS)
    if event == "corpus_load":
        return value == "loaded"
    if event == "integrity_failure":
        return value in _INTEGRITY_REASONS
    if event == "ci_gate":
        return value in _CI_GATES if key == "gate" else value in {"passed", "failed"} if key == "status" else value >= 0
    return value in {"passed", "failed"} if key == "status" else value >= 0 if key == "forbidden_entry_count" else bool(_ARCHIVE_SHA256.fullmatch(value))


class Telemetry:
    """Disabled by default; when enabled, sends JSON-safe records to one local sink."""

    def __init__(
        self,
        sink: Callable[[dict[str, Any]], None] | None = None,
        *,
        strict: bool = False,
        registry: CorpusRegistry | None = None,
    ):
        self._sink = sink
        self._strict = strict
        self._registry = registry

    @classmethod
    def from_environment(cls, registry: CorpusRegistry | None = None) -> Telemetry:
        return cls(_stderr_sink if os.environ.get("TJIPTO_TELEMETRY") == "stderr" else None, registry=registry)

    def bind_registry(self, registry: CorpusRegistry) -> None:
        """Bind this instance to one repository; reject cross-repository reuse."""
        if self._registry is None:
            self._registry = registry
        elif self._registry.repo_root.resolve() != registry.repo_root.resolve():
            raise ValueError("telemetry registry conflicts with runtime registry")

    def emit(self, event: str, **attributes: Any) -> None:
        if self._sink is not None:
            try:
                self._sink(event_record(event, registry=self._registry, **attributes))
            except Exception:
                if self._strict:
                    raise


def _stderr_sink(record: dict[str, Any]) -> None:
    print(json.dumps(record, separators=(",", ":"), sort_keys=True), file=sys.stderr)


DEFAULT_TELEMETRY = Telemetry.from_environment()
