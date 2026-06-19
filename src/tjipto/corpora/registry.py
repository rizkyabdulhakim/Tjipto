from __future__ import annotations

from pathlib import Path

from .adapter import config_for


def require_corpus(corpus_id: str, repo_root: Path | None = None):
    config = config_for(corpus_id, repo_root)
    if config is None:
        return {"status": "unsupported_corpus", "corpus_id": corpus_id}
    return config
