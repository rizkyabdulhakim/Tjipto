from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess  # nosec B404
import sys
import tempfile
import zipfile


FORBIDDEN_PATTERNS = (
    ".git",
    ".git/**",
    "node_modules",
    "node_modules/**",
    "apps/web/node_modules",
    "apps/web/node_modules/**",
    "dist",
    "dist/**",
    "apps/web/dist",
    "apps/web/dist/**",
    "__pycache__",
    "__pycache__/**",
    "*.pyc",
    "*.pyo",
    "*.key",
    "*.pem",
    "*.p12",
    "*.pfx",
    "credentials*.json",
    "id_rsa",
    "*.log",
    ".pytest_cache",
    ".pytest_cache/**",
    "tmp",
    "tmp/**",
    "temp",
    "temp/**",
    ".env",
    ".env.*",
    "coverage",
    "coverage/**",
    ".mypy_cache",
    ".mypy_cache/**",
    ".ruff_cache",
    ".ruff_cache/**",
)

_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo_root, text=True).strip()  # nosec B603 B607


def _exact_commit(repo_root: Path, commit_sha: str) -> tuple[str, str]:
    if not _COMMIT_SHA.fullmatch(commit_sha):
        raise ValueError("commit_sha must be a 40-character commit SHA")
    if _git(repo_root, "rev-parse", "--is-inside-work-tree") != "true":
        raise ValueError("archive creation requires a Git checkout")
    resolved = _git(repo_root, "rev-parse", commit_sha)
    if resolved != commit_sha:
        raise ValueError("commit_sha must be the exact resolved commit SHA")
    tree_sha = _git(repo_root, "rev-parse", f"{commit_sha}^{{tree}}")
    return resolved, tree_sha


def create_archive(repo_root: Path, commit_sha: str, archive_path: Path) -> dict[str, str]:
    """Create committed content for one exact commit; never reads worktree attributes."""
    commit_sha, tree_sha = _exact_commit(repo_root.resolve(), commit_sha)
    archive_path = archive_path.resolve()
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(  # nosec B603 B607 - exact commit SHA and fixed git subcommand.
        ["git", "-c", "core.autocrlf=false", "archive", "--format=zip", commit_sha, "-o", str(archive_path)],
        cwd=repo_root,
        check=True,
    )
    return {
        "commit_sha": commit_sha,
        "tree_sha": tree_sha,
        "archive_sha256": _sha256(archive_path),
    }


def archive_inventory(repo_root: Path, commit_sha: str, archive_path: Path) -> dict:
    expected = set(_git(repo_root, "ls-tree", "-r", "--name-only", commit_sha).splitlines())
    with zipfile.ZipFile(archive_path) as archive:
        actual = {name for name in archive.namelist() if not name.endswith("/")}
        mismatches = sorted(
            name for name in expected & actual
            if hashlib.sha256(archive.read(name)).hexdigest()
            != hashlib.sha256(
                subprocess.check_output(["git", "show", f"{commit_sha}:{name}"], cwd=repo_root)  # nosec B603 B607
            ).hexdigest()
        )
    inventory = {
        "expected_file_count": len(expected),
        "archive_file_count": len(actual),
        "missing_files": sorted(expected - actual),
        "extra_files": sorted(actual - expected),
        "digest_mismatches": mismatches,
    }
    inventory["status"] = "passed" if not any(inventory[key] for key in ("missing_files", "extra_files", "digest_mismatches")) else "failed"
    return inventory


def forbidden_entries(path: Path) -> list[str]:
    names = _zip_entries(path) if path.suffix.casefold() == ".zip" else _directory_entries(path)
    return sorted(name for name in names if _forbidden(name))


def verify_candidate(path: Path) -> list[str]:
    """Read-only validation of one ZIP or extracted candidate."""
    return forbidden_entries(path)


def _extract_archive(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as source:
        source.extractall(destination)


def run_candidate_checks(path: Path) -> dict[str, int]:
    """Run checks in a disposable copy; the verified candidate is never modified."""
    with tempfile.TemporaryDirectory(prefix="tjipto-release-candidate-") as tmp:
        candidate = Path(tmp) / "candidate"
        if path.suffix.casefold() == ".zip":
            _extract_archive(path, candidate)
        else:
            shutil.copytree(path, candidate)
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(candidate / "src")
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        subprocess.run(  # nosec B603 B607 - fixed Python module commands.
            [sys.executable, "-m", "compileall", "-q", "src", "tests", "scripts"],
            cwd=candidate,
            check=True,
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(  # nosec B603 B607 - fixed Python module commands.
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
            cwd=candidate,
            check=True,
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    return {"compileall": 0, "unittest": 0}


def release_candidate(repo_root: Path, commit_sha: str, archive_path: Path, corpus_ids: list[str] | None = None) -> dict:
    identity = create_archive(repo_root, commit_sha, archive_path)
    archive_forbidden = verify_candidate(archive_path)
    inventory = archive_inventory(repo_root, identity["commit_sha"], archive_path)
    checks = run_candidate_checks(archive_path) if not archive_forbidden and inventory["status"] == "passed" else {"compileall": 1, "unittest": 1}
    result = {
        **identity,
        "archive_forbidden_entries": archive_forbidden,
        "archive_inventory": inventory,
        "candidate_checks": checks,
    }
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from tjipto.telemetry import event_record
    result["telemetry"] = event_record(
        "release_validation",
        status="passed" if not archive_forbidden and inventory["status"] == "passed" and all(value == 0 for value in checks.values()) else "failed",
        forbidden_entry_count=len(archive_forbidden),
        archive_sha256=identity["archive_sha256"],
    )
    sidecar = archive_path.with_suffix(archive_path.suffix + ".sidecar.json")
    sidecar.write_text(json.dumps(_release_sidecar(repo_root, archive_path, result, corpus_ids), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result["sidecar_path"] = str(sidecar)
    result["sidecar_sha256"] = _sha256(sidecar)
    return result


def _release_sidecar(repo_root: Path, archive_path: Path, result: dict, corpus_ids: list[str] | None = None) -> dict:
    """Describe immutable archive bytes; this deliberately remains outside the archive."""
    with zipfile.ZipFile(archive_path) as archive:
        files = {name: hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist() if not name.endswith("/")}
        corpora = _archive_corpora(archive, corpus_ids)
        dense_promotions = _archive_dense_promotions(archive, result["commit_sha"], result["tree_sha"])
    sys.path.insert(0, str(repo_root / "src"))
    return {
        "archive_byte_representation": "git archive ZIP entry bytes",
        "archive_sha256": result["archive_sha256"],
        "archive_inventory": result["archive_inventory"],
        "candidate_checks": result["candidate_checks"],
        "commit_sha": result["commit_sha"],
        "corpora": corpora,
        "dense_promotions": dense_promotions,
        "source_file_digests": files,
        "tree_sha": result["tree_sha"],
        "worktree_status": _git(repo_root, "status", "--porcelain").splitlines(),
        "python_version": sys.version,
    }


def _archive_dense_promotions(archive: zipfile.ZipFile, commit_sha: str, tree_sha: str) -> dict[str, dict]:
    registry = json.loads(archive.read("data/corpus_registry.json"))
    promotions: dict[str, dict] = {}
    for corpus_id, entry in sorted(registry.items()):
        if not isinstance(entry, dict) or not entry.get("dense_index_path") or not entry.get("dense_promotion_path"):
            continue
        index_name = PurePosixPath(str(entry["dense_index_path"])).as_posix()
        promotion_name = PurePosixPath(str(entry["dense_promotion_path"])).as_posix()
        if (
            PurePosixPath(index_name).is_absolute()
            or ".." in PurePosixPath(index_name).parts
            or PurePosixPath(promotion_name).is_absolute()
            or ".." in PurePosixPath(promotion_name).parts
        ):
            raise ValueError("archive dense promotion has an unsafe path")
        try:
            promotion_bytes = archive.read(promotion_name)
            index_bytes = archive.read(index_name)
            promotion = json.loads(promotion_bytes)
        except (KeyError, json.JSONDecodeError) as error:
            raise ValueError("archive dense promotion is incomplete") from error
        if (
            not isinstance(promotion, dict)
            or promotion.get("status") != "promoted"
            or promotion.get("index_sha256") != hashlib.sha256(index_bytes).hexdigest()
        ):
            raise ValueError("archive dense promotion identity mismatch")
        promotions[corpus_id] = {
            "index_path": index_name,
            "index_sha256": promotion["index_sha256"],
            "promotion_path": promotion_name,
            "promotion_sha256": hashlib.sha256(promotion_bytes).hexdigest(),
            "index_identity": promotion.get("index_identity"),
            "recorded_runtime_commit": promotion.get("runtime_commit"),
            "recorded_runtime_tree_sha": promotion.get("runtime_tree_sha"),
            "runtime_commit": commit_sha,
            "runtime_tree_sha": tree_sha,
        }
    return promotions


def _archive_corpora(archive: zipfile.ZipFile, corpus_ids: list[str] | None = None) -> dict[str, dict]:
    registry = json.loads(archive.read("data/corpus_registry.json"))
    if not isinstance(registry, dict):
        raise ValueError("archive corpus registry is malformed")
    from tjipto.core.manifest import artifact_set_digest

    corpora = {}
    selected = corpus_ids or sorted(registry)
    for corpus_id in selected:
        entry = registry.get(corpus_id)
        rel = entry.get("manifest") if isinstance(entry, dict) else entry
        path = PurePosixPath(rel) if isinstance(rel, str) else None
        if not isinstance(corpus_id, str) or path is None or path.is_absolute() or ".." in path.parts:
            raise ValueError("archive corpus registry has an unsafe manifest path")
        manifest_bytes = archive.read(path.as_posix())
        manifest = json.loads(manifest_bytes)
        if not isinstance(manifest, dict) or manifest.get("corpus_id") != corpus_id:
            raise ValueError("archive manifest identity mismatch")
        corpora[corpus_id] = {
            "artifact_set_digest": artifact_set_digest(manifest),
            "contract": {key: manifest.get(key) for key in ("contract_id", "contract_version", "contract_fingerprint")},
            "extractor_fingerprint": manifest.get("extractor_fingerprint"),
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "runtime_projection_sha256": manifest.get("files", {}).get(manifest.get("runtime_projection", ""), {}).get("sha256"),
            "schema_version": manifest.get("schema_version"),
        }
    return corpora


def _directory_entries(path: Path) -> list[str]:
    if not path.is_dir():
        raise ValueError(f"expected directory or zip: {path}")
    return [item.relative_to(path).as_posix() for item in path.rglob("*")]


def _zip_entries(path: Path) -> list[str]:
    if not path.is_file():
        raise ValueError(f"expected directory or zip: {path}")
    with zipfile.ZipFile(path) as archive:
        return [name.rstrip("/") for name in archive.namelist() if name.rstrip("/")]


def _forbidden(name: str) -> bool:
    path = PurePosixPath(name)
    parts = path.parts
    if any(
        part in {".git", "node_modules", "dist", "__pycache__", ".pytest_cache", "tmp", "temp", "coverage", ".mypy_cache", ".ruff_cache"}
        for part in parts
    ):
        return True
    return (
        path.match("*.pyc")
        or path.match("*.pyo")
        or path.match("*.log")
        or path.match("*.key")
        or path.match("*.pem")
        or path.match("*.p12")
        or path.match("*.pfx")
        or path.match("credentials*.json")
        or path.name == "id_rsa"
        or path.match(".env")
        or path.match(".env.*")
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create and verify an immutable Tjipto release candidate.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create-archive")
    create.add_argument("commit_sha")
    create.add_argument("archive", type=Path)

    verify = subparsers.add_parser("verify-candidate")
    verify.add_argument("path", type=Path)

    checks = subparsers.add_parser("check-candidate")
    checks.add_argument("path", type=Path)

    release = subparsers.add_parser("release")
    release.add_argument("commit_sha")
    release.add_argument("archive", type=Path)
    release.add_argument("--corpus", action="append", default=[])

    args = parser.parse_args(argv)
    if args.command == "create-archive":
        print(json.dumps(create_archive(Path.cwd(), args.commit_sha, args.archive), sort_keys=True))
        return 0
    if args.command == "verify-candidate":
        noisy = verify_candidate(args.path)
        if noisy:
            print(f"clean_handoff: FAIL ({len(noisy)} forbidden entries)")
            print("\n".join(noisy[:50]))
            return 1
        print("clean_handoff: PASS (0 forbidden entries)")
        return 0
    if args.command == "check-candidate":
        print(json.dumps(run_candidate_checks(args.path), sort_keys=True))
        return 0
    result = release_candidate(Path.cwd(), args.commit_sha, args.archive, args.corpus)
    print(json.dumps(result, sort_keys=True))
    return 0 if not result["archive_forbidden_entries"] and result["archive_inventory"]["status"] == "passed" and all(value == 0 for value in result["candidate_checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
