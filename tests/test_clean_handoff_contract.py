from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from contextlib import redirect_stdout
import io
import unittest

from scripts.verify_clean_handoff import FORBIDDEN_PATTERNS, forbidden_entries, main


ROOT = Path(__file__).resolve().parents[1]


class CleanHandoffContractTest(unittest.TestCase):
    def test_forbidden_patterns_are_centralized(self) -> None:
        self.assertIn("node_modules/**", FORBIDDEN_PATTERNS)
        self.assertIn("apps/web/dist/**", FORBIDDEN_PATTERNS)
        self.assertIn("*.pyc", FORBIDDEN_PATTERNS)

    def test_git_archive_has_no_forbidden_entries(self) -> None:
        probe = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=ROOT, capture_output=True, text=True)
        if probe.returncode != 0:
            self.skipTest("git archive production check requires a git checkout")
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "tjipto-clean.zip"
            subprocess.run(
                ["git", "archive", "--format=zip", "--worktree-attributes", "HEAD", "-o", str(archive)],
                cwd=ROOT,
                check=True,
            )
            self.assertEqual(forbidden_entries(archive), [])

    def test_verifier_fails_on_noisy_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "apps/web/node_modules/pkg").mkdir(parents=True)
            (root / "apps/web/dist").mkdir(parents=True)
            (root / "debug.log").write_text("", encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                self.assertNotEqual(main([str(root)]), 0)

    def test_verifier_passes_on_clean_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "src/example.py").write_text("print('ok')\n", encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main([str(root)]), 0)


if __name__ == "__main__":
    unittest.main()
