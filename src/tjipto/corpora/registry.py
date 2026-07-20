from __future__ import annotations

from pathlib import Path
import json
import os

from tjipto.core.config import CorpusConfig
from tjipto.core.manifest import read_json
from tjipto.contracts.artifacts import CURRENT_ARTIFACT_SCHEMA

SUPPORTED_ARTIFACT_SCHEMA_VERSIONS = {5, CURRENT_ARTIFACT_SCHEMA}


class CorpusRegistry:
    def __init__(self, repo_root: Path | None = None):
        env_root = os.environ.get("TJIPTO_REPO_ROOT")
        self.repo_root = repo_root or (Path(env_root) if env_root else Path(__file__).resolve().parents[3])
        self.registry_path = self.repo_root / "data" / "corpus_registry.json"
        self.error_code: str | None = None

    def resolve(self, corpus_id: str) -> CorpusConfig | None:
        self.error_code = None
        try:
            registry = read_json(self.registry_path)
        except (OSError, json.JSONDecodeError):
            self.error_code = "registry_unavailable"
            return None
        if not isinstance(registry, dict):
            self.error_code = "malformed_registry"
            return None
        entry = registry.get(corpus_id)
        if entry is None:
            self.error_code = "unknown_corpus"
            return None
        rel = entry.get("manifest") if isinstance(entry, dict) else entry
        if not isinstance(rel, str):
            self.error_code = "malformed_registry_entry"
            return None
        manifest_path = self._safe_registry_path(rel)
        if manifest_path is None:
            self.error_code = "manifest_path_violation"
            return None
        if not manifest_path.exists():
            self.error_code = "manifest_missing"
            return None
        try:
            manifest = read_json(manifest_path)
        except (OSError, json.JSONDecodeError):
            self.error_code = "malformed_manifest"
            return None
        if not isinstance(manifest, dict):
            self.error_code = "malformed_manifest"
            return None
        if manifest.get("corpus_id") != corpus_id:
            self.error_code = "corpus_id_mismatch"
            return None
        if manifest.get("schema_version") not in SUPPORTED_ARTIFACT_SCHEMA_VERSIONS or (
            corpus_id == "uud" and manifest.get("schema_version") != CURRENT_ARTIFACT_SCHEMA
        ):
            self.error_code = "unsupported_schema"
            return None
        settings = {key: value for key, value in entry.items() if key != "manifest"} if isinstance(entry, dict) else {}
        return CorpusConfig(corpus_id, manifest_path, manifest, settings, self.repo_root)

    def _safe_registry_path(self, rel: str) -> Path | None:
        path = Path(rel)
        if path.is_absolute():
            return None
        root = self.repo_root.resolve()
        resolved = (root / path).resolve()
        if not resolved.is_relative_to(root):
            return None
        return resolved


def require_corpus(corpus_id: str, repo_root: Path | None = None):
    config = CorpusRegistry(repo_root).resolve(corpus_id)
    if config is None:
        return {"status": "unsupported_corpus", "corpus_id": corpus_id}
    return config
