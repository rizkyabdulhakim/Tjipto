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
    with tempfile.TemporaryDirectory(prefix=stage_prefix, dir=final_dir.parent) as tmp:
        tmp_dir = Path(tmp)
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
                    for attempt in range(3):
                        try:
                            path.replace(target)
                            break
                        except PermissionError:
                            if attempt == 2:
                                raise
                            time.sleep(0.1)
                    promoted.append(path.name)
        except Exception:
            for name in promoted:
                (snapshot_dir / name).replace(final_dir / name)
            raise
