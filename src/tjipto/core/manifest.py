from __future__ import annotations

import hashlib
import json
from pathlib import Path


FORBIDDEN_ACTIVE_PATH_PARTS = (
    "candidate",
    "batch",
    "manual_review",
    "dry_run",
    "pilot",
    "remaining",
    "supplemental",
    "v1",
    "v2",
)
ALLOWED_ARTIFACT_ORIGINS = {
    "generated",
    "carried_forward",
    "manual_review_artifact",
    "deprecated",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_set_digest(manifest: dict, *, exclude: tuple[str, ...] = ()) -> str:
    rows = [
        (name, row.get("sha256"), row.get("bytes"))
        for name, row in sorted(manifest["files"].items())
        if name not in exclude
    ]
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode("utf-8")).hexdigest()


def verified_file_bytes(path: Path, record: dict) -> tuple[bytes | None, str | None]:
    try:
        data = path.read_bytes()
    except OSError:
        return None, "artifact_missing"
    if len(data) != record.get("bytes"):
        return None, "artifact_size_mismatch"
    if hashlib.sha256(data).hexdigest() != record.get("sha256"):
        return None, "artifact_sha256_mismatch"
    return data, None


def validate_manifest(final_dir: Path) -> tuple[str, ...]:
    manifest_path = final_dir / "manifest.json"
    manifest = read_json(manifest_path)
    errors: list[str] = []
    for rel, record in manifest["files"].items():
        lowered = rel.casefold()
        for part in FORBIDDEN_ACTIVE_PATH_PARTS:
            if part in lowered:
                errors.append(f"forbidden active path part:{rel}:{part}")
        path = final_dir / rel
        _, integrity_error = verified_file_bytes(path, record)
        if integrity_error:
            errors.append(f"{integrity_error}:{rel}")
        origin = record.get("origin")
        if origin not in ALLOWED_ARTIFACT_ORIGINS:
            errors.append(f"invalid_artifact_origin:{rel}:{origin}")
        for field in ("origin", "producer", "build_stage"):
            if not str(record.get(field) or "").strip():
                errors.append(f"manifest_file_missing_{field}:{rel}")
        if origin == "generated":
            if not str(record.get("producer") or "").strip():
                errors.append(f"generated_artifact_missing_producer:{rel}")
            if not str(record.get("build_stage") or "").strip():
                errors.append(f"generated_artifact_missing_build_stage:{rel}")
        elif not str(record.get("origin_reason") or "").strip():
            errors.append(f"non_generated_artifact_missing_origin_reason:{rel}")
    return tuple(errors)
