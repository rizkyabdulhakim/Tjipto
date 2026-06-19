from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = (
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


class NoSlopLeakageTest(unittest.TestCase):
    def test_active_paths_are_clean(self) -> None:
        active_paths = [
            path.relative_to(ROOT).as_posix()
            for path in (ROOT / "data/final/uud").iterdir()
            if path.is_file()
        ]
        active_paths.extend(
            str(path.relative_to(ROOT)).replace("\\", "/")
            for path in (ROOT / "src/tjipto/runtime").rglob("*.py")
        )
        for path in active_paths:
            lowered = path.casefold()
            for word in FORBIDDEN:
                self.assertNotIn(word, lowered, path)

    def test_runtime_source_does_not_reference_process_artifacts(self) -> None:
        runtime_text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (ROOT / "src/tjipto/runtime").rglob("*.py")
        ).casefold()
        for word in FORBIDDEN:
            self.assertNotIn(word, runtime_text)
        self.assertNotIn("data/processed/constitutional/uud", runtime_text)


if __name__ == "__main__":
    unittest.main()
