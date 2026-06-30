from __future__ import annotations

from pathlib import Path

from tjipto.artifacts.writer import write_json
from tjipto.core.manifest import file_sha256, read_jsonl


def refresh_manifest(
    final_dir: Path,
    manifest: dict,
    *,
    count_files: tuple[tuple[str, str], ...],
    legacy_counts: dict[str, int] | None = None,
) -> None:
    legacy_counts = legacy_counts or {}
    counts = manifest.setdefault("counts", {})
    for key, filename in count_files:
        path = final_dir / filename
        if path.exists():
            counts[key] = len(read_jsonl(path))
        elif key in legacy_counts:
            counts[key] = legacy_counts[key]
    for rel in manifest["files"]:
        path = final_dir / rel
        if path.exists():
            manifest["files"][rel]["bytes"] = path.stat().st_size
            manifest["files"][rel]["sha256"] = file_sha256(path)
    write_json(final_dir / "manifest.json", manifest)
