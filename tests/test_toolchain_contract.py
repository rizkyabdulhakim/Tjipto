from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.validate_toolchain import package_origin_error, validate_toolchain


class ToolchainContractTest(unittest.TestCase):
    def test_rejects_python_node_npm_lock_and_extractor_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(Path(tmp))
            def command(_root: Path, tool: str, _flag: str) -> str:
                return "v24.14.0" if tool == "node" else "11.9.0"

            self.assertEqual(validate_toolchain(root, command_output=command, extractor_versions=("1.27.2.3", "1.27.2")), [])
            self.assertIn("python: expected 3.12.10, got 3.12.9", validate_toolchain(root, python_version="3.12.9", command_output=command, extractor_versions=("1.27.2.3", "1.27.2")))
            self.assertTrue(any(error.startswith("node:") for error in validate_toolchain(root, command_output=lambda *_: "v24.13.0", extractor_versions=("1.27.2.3", "1.27.2"))))
            self.assertTrue(any(error.startswith("npm:") for error in validate_toolchain(root, command_output=lambda _root, tool, _flag: "v24.14.0" if tool == "node" else "11.8.0", extractor_versions=("1.27.2.3", "1.27.2"))))
            lock = root / "apps/web/package-lock.json"
            payload = json.loads(lock.read_text(encoding="utf-8"))
            payload["packages"][""]["engines"]["node"] = "24.13.0"
            lock.write_text(json.dumps(payload), encoding="utf-8")
            self.assertIn("node metadata: .nvmrc, package.json, and package-lock.json must match", validate_toolchain(root, command_output=command, extractor_versions=("1.27.2.3", "1.27.2")))
            payload["packages"][""]["engines"]["node"] = "24.14.0"
            lock.write_text(json.dumps(payload), encoding="utf-8")
            self.assertIn("corpus demo: extractor fingerprint mismatch", validate_toolchain(root, command_output=command, extractor_versions=("1.27.2.3", "1.27.1")))

    def test_package_origin_must_be_current_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root / "src/tjipto/__init__.py"
            package.parent.mkdir(parents=True)
            package.write_text("", encoding="utf-8")
            self.assertIsNone(package_origin_error(root, package))
            self.assertIn("package origin: expected", package_origin_error(root, root / "elsewhere/tjipto/__init__.py") or "")

    @staticmethod
    def _root(root: Path) -> Path:
        (root / "apps/web").mkdir(parents=True)
        (root / "data/final/demo").mkdir(parents=True)
        (root / ".python-version").write_text("3.12.10\n", encoding="utf-8")
        (root / ".nvmrc").write_text("24.14.0\n", encoding="utf-8")
        (root / "requirements.lock").write_text("ruff==0.16.0\n", encoding="utf-8")
        (root / "requirements-dense.lock").write_text("torch==2.8.0\ntransformers==4.57.1\nsentencepiece==0.2.1\n", encoding="utf-8")
        (root / "pyproject.toml").write_text(
            "[project.optional-dependencies]\ndense = [\"torch==2.8.0\", \"transformers==4.57.1\", \"sentencepiece==0.2.1\"]\n",
            encoding="utf-8",
        )
        (root / ".pre-commit-config.yaml").write_text("repos:\n  - repo: https://github.com/astral-sh/ruff-pre-commit\n    rev: v0.16.0\n", encoding="utf-8")
        (root / "apps/web/package.json").write_text(json.dumps({"engines": {"node": "24.14.0", "npm": "11.9.0"}}), encoding="utf-8")
        (root / "apps/web/package-lock.json").write_text(json.dumps({"packages": {"": {"engines": {"node": "24.14.0", "npm": "11.9.0"}}}}), encoding="utf-8")
        (root / "data/corpus_registry.json").write_text(json.dumps({"demo": {"manifest": "data/final/demo/manifest.json"}}), encoding="utf-8")
        (root / "data/final/demo/manifest.json").write_text(json.dumps({"corpus_id": "demo", "schema_version": 5, "extractor_fingerprint": {"python": "3.12.10", "pymupdf": "1.27.2.3", "mupdf": "1.27.2"}}), encoding="utf-8")
        return root
