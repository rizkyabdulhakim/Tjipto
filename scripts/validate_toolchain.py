"""Fail closed when the checked-in canonical toolchain drifts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess  # nosec B404
import sys
import tomllib
from typing import Callable

from tjipto.corpora.registry import CorpusRegistry


def _command_output(root: Path, *command: str) -> str:
    if sys.platform == "win32" and command[0] == "npm":
        command = ("cmd.exe", "/d", "/s", "/c", *command)
    return subprocess.check_output(command, cwd=root, text=True).strip()  # nosec B603 B607 - fixed tool commands.


def _lock_version(path: Path, package: str) -> str | None:
    match = re.search(rf"^{re.escape(package)}==([^\s]+)", path.read_text(encoding="utf-8"), re.MULTILINE)
    return match.group(1) if match else None


def package_origin_error(root: Path, package_file: Path | None = None) -> str | None:
    """Return an error when the imported package is not from this checkout."""
    expected = (root / "src" / "tjipto").resolve()
    if package_file is None:
        import tjipto

        package_file = Path(tjipto.__file__ or "")
    try:
        package_file.resolve().relative_to(expected)
    except ValueError:
        return f"package origin: expected {expected}, got {package_file.resolve()}"
    return None


def validate_toolchain(
    root: Path | None = None,
    corpus_ids: list[str] | None = None,
    *,
    python_version: str | None = None,
    command_output: Callable[..., str] = _command_output,
    extractor_versions: tuple[str, str] | None = None,
) -> list[str]:
    root = (root or Path(__file__).resolve().parents[1]).resolve()
    errors: list[str] = []
    expected_python = (root / ".python-version").read_text(encoding="utf-8").strip()
    active_python = python_version or sys.version.split()[0]
    if active_python != expected_python:
        errors.append(f"python: expected {expected_python}, got {active_python}")

    expected_node = (root / ".nvmrc").read_text(encoding="utf-8").strip()
    package = json.loads((root / "apps/web/package.json").read_text(encoding="utf-8"))
    lock = json.loads((root / "apps/web/package-lock.json").read_text(encoding="utf-8"))
    package_engines = package.get("engines", {})
    lock_engines = lock.get("packages", {}).get("", {}).get("engines", {})
    expected_npm = package_engines.get("npm")
    if package_engines.get("node") != expected_node or lock_engines.get("node") != expected_node:
        errors.append("node metadata: .nvmrc, package.json, and package-lock.json must match")
    if lock_engines.get("npm") != expected_npm:
        errors.append("npm metadata: package.json and package-lock.json must match")
    active_node = command_output(root, "node", "--version").removeprefix("v")
    active_npm = command_output(root, "npm", "--version")
    if active_node != expected_node:
        errors.append(f"node: expected {expected_node}, got {active_node}")
    if active_npm != expected_npm:
        errors.append(f"npm: expected {expected_npm}, got {active_npm}")

    ruff_lock = _lock_version(root / "requirements.lock", "ruff")
    pre_commit = (root / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    ruff_pre_commit = re.search(r"repo: https://github.com/astral-sh/ruff-pre-commit\s+rev: v([^\s]+)", pre_commit)
    if ruff_lock is None or ruff_pre_commit is None or ruff_pre_commit.group(1) != ruff_lock:
        errors.append("ruff pin: pre-commit and requirements.lock must match")

    dense_lock = root / "requirements-dense.lock"
    dense = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]["optional-dependencies"]["dense"]
    if not dense_lock.exists():
        errors.append("dense lock: requirements-dense.lock is missing")
        dense_lock = root / "requirements.lock"
    for requirement in dense:
        package_name, expected_version = requirement.split("==", maxsplit=1)
        if _lock_version(dense_lock, package_name) != expected_version:
            errors.append(f"dense lock: {requirement} must match requirements-dense.lock")

    if extractor_versions is None:
        import fitz

        extractor_versions = (fitz.VersionBind, fitz.VersionFitz)
    registry = CorpusRegistry(root)
    for corpus_id in corpus_ids or list(registry.corpus_ids()):
        config = registry.resolve(corpus_id)
        if config is None:
            errors.append(f"corpus {corpus_id}: {registry.error_code or 'unavailable'}")
            continue
        fingerprint = config.manifest.get("extractor_fingerprint", {})
        expected = (expected_python, *extractor_versions)
        actual = (fingerprint.get("python"), fingerprint.get("pymupdf"), fingerprint.get("mupdf"))
        if actual != expected:
            errors.append(f"corpus {corpus_id}: extractor fingerprint mismatch")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the checked-in canonical Python and Node toolchain.")
    parser.add_argument("--corpus", action="append", default=[])
    parser.add_argument("--package-origin", action="store_true")
    args = parser.parse_args(argv)
    if args.package_origin:
        error = package_origin_error(Path(__file__).resolve().parents[1])
        if error:
            print("package origin: FAIL")
            print(error)
            return 1
        print("package origin: PASS")
        return 0
    errors = validate_toolchain(corpus_ids=args.corpus)
    if errors:
        print("toolchain: FAIL")
        print("\n".join(errors))
        return 1
    print("toolchain: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
