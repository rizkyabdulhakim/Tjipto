from __future__ import annotations

import hashlib
import io
from pathlib import Path
import subprocess  # nosec B404
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import scripts.verify_clean_handoff as handoff


class CleanHandoffContractTest(unittest.TestCase):
    def test_forbidden_patterns_are_centralized(self) -> None:
        self.assertIn("node_modules/**", handoff.FORBIDDEN_PATTERNS)
        self.assertIn("apps/web/dist/**", handoff.FORBIDDEN_PATTERNS)
        self.assertIn("*.pyc", handoff.FORBIDDEN_PATTERNS)
        self.assertIn("*.key", handoff.FORBIDDEN_PATTERNS)

    def test_candidate_rejects_credential_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "credentials-prod.json").write_text("{}", encoding="utf-8")
            (root / "private.key").write_text("secret", encoding="utf-8")
            self.assertEqual(handoff.forbidden_entries(root), ["credentials-prod.json", "private.key"])

    def test_archive_uses_exact_commit_and_ignores_dirty_attributes_and_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._init_git(root)
            (root / "src").mkdir()
            (root / "src/value.txt").write_text("committed\n", encoding="utf-8")
            (root / ".gitattributes").write_text("src/value.txt export-ignore\n", encoding="utf-8")
            self._git(root, "add", ".")
            self._git(root, "commit", "-m", "baseline")
            commit = self._git(root, "rev-parse", "HEAD")

            (root / "src/value.txt").write_text("dirty\n", encoding="utf-8")
            (root / ".gitattributes").write_text("\n", encoding="utf-8")
            archive = root / "candidate.zip"
            identity = handoff.create_archive(root, commit, archive)

            self.assertEqual(identity["commit_sha"], commit)
            self.assertEqual(identity["archive_sha256"], hashlib.sha256(archive.read_bytes()).hexdigest())
            with zipfile.ZipFile(archive) as source:
                self.assertNotIn("src/value.txt", source.namelist())
            self.assertEqual((root / "src/value.txt").read_text(encoding="utf-8"), "dirty\n")

    def test_archive_rejects_non_exact_ref(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._init_git(root)
            with self.assertRaisesRegex(ValueError, "SHA|exact"):
                handoff.create_archive(root, "HEAD", root / "candidate.zip")

    def test_candidate_verification_is_read_only_and_requires_no_git(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src/example.py").parent.mkdir()
            (root / "src/example.py").write_text("print('ok')\n", encoding="utf-8")
            before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
            self.assertEqual(handoff.verify_candidate(root), [])
            after = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
            self.assertEqual(before, after)
            self.assertFalse((root / ".git").exists())

    def test_candidate_checks_use_disposable_copy_without_recursive_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for directory in ("src", "tests", "scripts"):
                (root / directory).mkdir()
            (root / "src/example.py").write_text("VALUE = 1\n", encoding="utf-8")
            (root / "scripts/example.py").write_text("VALUE = 1\n", encoding="utf-8")
            (root / "tests/test_example.py").write_text(
                "import unittest\n\nclass CandidateTest(unittest.TestCase):\n    def test_candidate_without_git(self):\n        self.assertTrue(True)\n",
                encoding="utf-8",
            )
            before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
            with patch.object(handoff, "create_archive", side_effect=AssertionError("recursive archive")):
                self.assertEqual(handoff.run_candidate_checks(root), {"compileall": 0, "unittest": 0})
            after = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
            self.assertEqual(before, after)
            self.assertEqual(handoff.forbidden_entries(root), [])

    def test_release_sidecar_binds_archive_bytes_without_entering_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._init_git(root)
            (root / "data/final/uud").mkdir(parents=True)
            (root / "src").mkdir()
            (root / "tests").mkdir()
            (root / "tests/test_candidate.py").write_text("import unittest\n\nclass CandidateTest(unittest.TestCase):\n    def test_ok(self): self.assertTrue(True)\n", encoding="utf-8")
            (root / "data/final/uud/manifest.json").write_text(
                '{"contract_id":"uud","contract_version":7,"contract_fingerprint":"f","files":{}}\n', encoding="utf-8"
            )
            self._git(root, "add", ".")
            self._git(root, "commit", "-m", "candidate")
            archive = root / "candidate.zip"
            result = handoff.release_candidate(root, self._git(root, "rev-parse", "HEAD"), archive)
            sidecar = Path(result["sidecar_path"])
            self.assertTrue(sidecar.is_file())
            self.assertEqual(result["sidecar_sha256"], hashlib.sha256(sidecar.read_bytes()).hexdigest())
            payload = __import__("json").loads(sidecar.read_text(encoding="utf-8"))
            self.assertEqual(payload["archive_sha256"], result["archive_sha256"])
            with zipfile.ZipFile(archive) as source:
                self.assertNotIn(sidecar.name, source.namelist())

    def test_cli_verifier_rejects_noisy_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "apps/web/node_modules/pkg").mkdir(parents=True)
            (root / "apps/web/dist").mkdir(parents=True)
            (root / "debug.log").write_text("", encoding="utf-8")
            with patch("sys.stdout", io.StringIO()):
                self.assertEqual(handoff.main(["verify-candidate", str(root)]), 1)

    @staticmethod
    def _init_git(root: Path) -> None:
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)  # nosec B603 B607
        subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=root, check=True)  # nosec B603 B607
        subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)  # nosec B603 B607

    @staticmethod
    def _git(root: Path, *args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()  # nosec B603 B607


if __name__ == "__main__":
    unittest.main()
