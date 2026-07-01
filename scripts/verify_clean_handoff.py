from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
import sys
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


def forbidden_entries(path: Path) -> list[str]:
    names = _zip_entries(path) if path.suffix.casefold() == ".zip" else _directory_entries(path)
    return sorted(name for name in names if _forbidden(name))


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
    if any(part in {".git", "node_modules", "dist", "__pycache__", ".pytest_cache", "tmp", "temp", "coverage", ".mypy_cache", ".ruff_cache"} for part in parts):
        return True
    return path.match("*.pyc") or path.match("*.pyo") or path.match("*.log") or path.match(".env") or path.match(".env.*")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify a clean Tjipto handoff directory or zip.")
    parser.add_argument("path", type=Path)
    args = parser.parse_args(argv)
    noisy = forbidden_entries(args.path)
    if noisy:
        print(f"clean_handoff: FAIL ({len(noisy)} forbidden entries)")
        for name in noisy[:50]:
            print(name)
        return 1
    print("clean_handoff: PASS (0 forbidden entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
