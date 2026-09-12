from __future__ import annotations

import shutil
import tempfile
import time
from collections.abc import Callable
from pathlib import Path


def atomic_promote_artifacts(
    *,
    final_dir: Path,
    build: Callable[[Path], None],
    validate: Callable[[Path], tuple[str, ...]],
    stage_prefix: str = ".artifact-stage-",
) -> None:
    final_dir = final_dir.resolve()
    _remove_abandoned_stages(final_dir.parent, stage_prefix)
    tmp_dir = Path(tempfile.mkdtemp(prefix=stage_prefix, dir=final_dir.parent))
    try:
        stage_dir = tmp_dir / "stage"
        snapshot_dir = tmp_dir / "snapshot"
        shutil.copytree(final_dir, stage_dir)
        shutil.copytree(final_dir, snapshot_dir)
        build(stage_dir)
        errors = validate(stage_dir)
        if errors:
            raise ValueError(";".join(errors))
        promoted: list[str] = []
        try:
            for path in sorted(stage_dir.iterdir()):
                if path.is_file():
                    target = final_dir / path.name
                    for attempt in range(10):
                        try:
                            path.replace(target)
                            break
                        except PermissionError:
                            if attempt == 9:
                                raise
                            time.sleep(0.2)
                    promoted.append(path.name)
        except Exception:
            for name in promoted:
                (snapshot_dir / name).replace(final_dir / name)
            raise
    finally:
        _remove_stage_dir(tmp_dir)


def _remove_abandoned_stages(parent: Path, stage_prefix: str) -> None:
    # ponytail: one build per final directory; add a file lock if concurrent
    # writers are ever required.
    for path in parent.glob(f"{stage_prefix}*"):
        if path.is_dir():
            _remove_stage_dir(path)


def _remove_stage_dir(path: Path) -> None:
    for attempt in range(10):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.2)
