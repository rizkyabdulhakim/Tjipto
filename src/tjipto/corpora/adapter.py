from __future__ import annotations

from pathlib import Path

from tjipto.corpora.registry import CorpusRegistry
from tjipto.core.config import CorpusConfig


def config_for(corpus_id: str, repo_root: Path | None = None) -> CorpusConfig | None:
    return CorpusRegistry(repo_root).resolve(corpus_id)
