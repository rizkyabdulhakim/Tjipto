from __future__ import annotations

from pathlib import Path
import io
import subprocess
import tarfile
import unittest
import json


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
        self.assertNotIn("data/final/uud", runtime_text)

    def test_core_boundary_has_no_runtime_or_corpus_dependency(self) -> None:
        core_text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (ROOT / "src/tjipto/core").rglob("*.py")
        ).casefold()
        self.assertNotIn("tjipto.runtime", core_text)
        self.assertNotIn("tjipto.retrieval", core_text)
        self.assertNotIn("tjipto.corpora", core_text)
        self.assertNotIn("tjipto.evidence", core_text)
        self.assertNotIn("tjipto.graph", core_text)
        self.assertNotIn("data/final/uud", core_text)

    def test_generic_reproducibility_has_no_uud_final_path(self) -> None:
        text = (ROOT / "src/tjipto/corpora/reproducibility.py").read_text(encoding="utf-8").casefold()
        self.assertNotIn("data/final/uud", text)
        self.assertNotIn(r"data\\final\\uud", text)

    def test_stale_runtime_structure_shim_is_absent(self) -> None:
        self.assertFalse((ROOT / "src/tjipto/retrieval/structure.py").exists())

    def test_git_archive_handoff_excludes_local_artifacts(self) -> None:
        if (ROOT / ".git").exists():
            archive = subprocess.check_output(
                ["git", "archive", "--format=tar", "--worktree-attributes", "HEAD"],
                cwd=ROOT,
            )
            names = set(tarfile.open(fileobj=io.BytesIO(archive)).getnames())
        else:
            names = {path.relative_to(ROOT).as_posix() for path in ROOT.rglob("*")}
        forbidden = (
            ".git",
            "node_modules",
            "apps/web/node_modules",
            "apps/web/dist",
            "__pycache__",
            ".pytest_cache",
            ".tjipto-http.pid",
        )
        for item in forbidden:
            self.assertNotIn(item, names)

    def test_runtime_facing_ids_are_clean(self) -> None:
        checked_files = (
            "evidence_registry.jsonl",
            "legal_units.jsonl",
            "chunks.jsonl",
            "metadata.jsonl",
            "metadata_grounding.jsonl",
            "metadata_grounding_registry.jsonl",
            "metadata_graph_edges.jsonl",
            "document_metadata.jsonl",
            "article_versions.jsonl",
            "retrieval_units.jsonl",
        )
        for filename in checked_files:
            rows = [
                json.loads(line)
                for line in (ROOT / "data/final/uud" / filename)
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            for row in rows:
                self._assert_id_values_clean(row, filename)

    def test_eval_fixture_fields_are_clean_and_not_final_artifacts(self) -> None:
        self.assertFalse((ROOT / "data/final/uud/eval_fixtures.jsonl").exists())
        rows = [
            json.loads(line)
            for line in (ROOT / "tests/fixtures/uud/eval_fixtures.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        self.assertEqual(len(rows), 175)
        text = "\n".join(json.dumps(row, sort_keys=True) for row in rows)
        self.assertNotIn("expected_candidate_id", text)
        self.assertNotIn("expected_top_candidate_id", text)
        self.assertNotIn("must_include_candidate_ids", text)
        self.assertNotIn("candidate_id", text)

    def _assert_id_values_clean(self, value, label: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {"text", "quoted_text", "source_pdf_path", "source_pdf"}:
                    continue
                self._assert_id_values_clean(child, label)
            return
        if isinstance(value, list):
            for child in value:
                self._assert_id_values_clean(child, label)
            return
        if isinstance(value, str):
            lowered = value.casefold()
            for word in FORBIDDEN:
                self.assertNotIn(word, lowered, label)


if __name__ == "__main__":
    unittest.main()
