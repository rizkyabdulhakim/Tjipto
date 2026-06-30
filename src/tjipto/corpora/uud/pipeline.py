from __future__ import annotations

from pathlib import Path

from tjipto.artifacts.pipeline import atomic_promote_artifacts


def run_staged_uud_pipeline(
    final_dir: Path,
    build,
    validate,
) -> None:
    atomic_promote_artifacts(final_dir=final_dir, build=build, validate=validate, stage_prefix=".uud-stage-")
