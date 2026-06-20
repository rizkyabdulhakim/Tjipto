from __future__ import annotations

from pathlib import Path
import json
import os

from tjipto.core.config import CorpusConfig
from tjipto.core.manifest import read_json


class CorpusRegistry:
    def __init__(self, repo_root: Path | None = None):
        env_root = os.environ.get("TJIPTO_REPO_ROOT")
        self.repo_root = repo_root or (Path(env_root) if env_root else Path(__file__).resolve().parents[3])
        self.registry_path = self.repo_root / "data" / "corpus_registry.json"

    def resolve(self, corpus_id: str) -> CorpusConfig | None:
        try:
            registry = read_json(self.registry_path)
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(registry, dict):
            return None
        rel = registry.get(corpus_id)
        if not isinstance(rel, str):
            return None
        manifest_path = self._safe_registry_path(rel)
        if manifest_path is None:
            return None
        if not manifest_path.exists():
            return None
        try:
            manifest = read_json(manifest_path)
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(manifest, dict):
            return None
        if manifest.get("corpus_id") != corpus_id:
            return None
        return CorpusConfig(corpus_id, manifest_path, manifest)

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
