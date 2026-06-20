from __future__ import annotations

from pathlib import Path

from tjipto.corpora.reproducibility import validate_corpus_ingestion_artifacts


def validate_uud_ingestion_artifacts(repo_root: Path) -> dict:
    return validate_corpus_ingestion_artifacts("uud", repo_root)
