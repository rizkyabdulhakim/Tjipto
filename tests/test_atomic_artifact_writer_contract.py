from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from tjipto.artifacts.pipeline import atomic_promote_artifacts
from tjipto.artifacts.writer import write_jsonl


class AtomicArtifactWriterContractTest(unittest.TestCase):
    def test_promote_rejects_invalid_stage_without_touching_final(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            final = Path(tmp) / "final"
            final.mkdir()
            original = [{"id": "old"}]
            write_jsonl(final / "rows.jsonl", original)

            def build(stage: Path) -> None:
                write_jsonl(stage / "rows.jsonl", [{"id": "bad"}])

            with self.assertRaises(ValueError):
                atomic_promote_artifacts(
                    final_dir=final,
                    build=build,
                    validate=lambda _: ("invalid_stage",),
                )

            rows = [json.loads(line) for line in (final / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows, original)

    def test_promote_writes_lf_only_when_stage_is_valid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            final = Path(tmp) / "final"
            final.mkdir()
            write_jsonl(final / "rows.jsonl", [{"id": "old"}])

            atomic_promote_artifacts(
                final_dir=final,
                build=lambda stage: write_jsonl(stage / "rows.jsonl", [{"id": "new"}]),
                validate=lambda _: (),
            )

            self.assertEqual((final / "rows.jsonl").read_bytes(), b'{"id": "new"}\n')

    def test_promote_clears_abandoned_stage_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            final = Path(tmp) / "final"
            final.mkdir()
            stale = Path(tmp) / ".artifact-stage-empty"
            stale.mkdir()
            write_jsonl(stale / "partial.jsonl", [{"id": "partial"}])
            atomic_promote_artifacts(
                final_dir=final,
                build=lambda stage: write_jsonl(stage / "rows.jsonl", [{"id": "new"}]),
                validate=lambda _: (),
            )
            self.assertFalse(stale.exists())


if __name__ == "__main__":
    unittest.main()
