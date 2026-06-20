from __future__ import annotations

from pathlib import Path

from tjipto.core.config import CorpusConfig
from tjipto.core.manifest import read_json


class CorpusRegistry:
    def __init__(self, repo_root: Path | None = None):
        self.repo_root = repo_root or Path(__file__).resolve().parents[3]
        self.registry_path = self.repo_root / "data" / "corpus_registry.json"

    def resolve(self, corpus_id: str) -> CorpusConfig | None:
        registry = read_json(self.registry_path)
        rel = registry.get(corpus_id)
        if rel is None:
            return None
        manifest_path = self.repo_root / rel
        if not manifest_path.exists():
            return None
        manifest = read_json(manifest_path)
        if manifest.get("corpus_id") != corpus_id:
            return None
        return CorpusConfig(corpus_id, manifest_path, manifest)


def require_corpus(corpus_id: str, repo_root: Path | None = None):
    config = CorpusRegistry(repo_root).resolve(corpus_id)
    if config is None:
        return {"status": "unsupported_corpus", "corpus_id": corpus_id}
    return config
