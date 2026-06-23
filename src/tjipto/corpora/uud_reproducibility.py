from __future__ import annotations

from pathlib import Path

from tjipto.corpora.reproducibility import validate_corpus_ingestion_artifacts
from tjipto.corpora.uud_artifact_baseline import validate_uud_artifact_baseline


def validate_uud_ingestion_artifacts(repo_root: Path) -> dict:
    result = validate_corpus_ingestion_artifacts("uud", repo_root)
    baseline_errors = validate_uud_artifact_baseline(repo_root)
    if baseline_errors:
        return {
            **result,
            "status": "invalid",
            "errors": tuple(sorted({*result["errors"], *baseline_errors})),
        }
    return result
