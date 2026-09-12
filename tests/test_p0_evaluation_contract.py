from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "evaluate_p0_contract.py"
FIXTURES = ROOT / "tests/fixtures/uud/p0_pre_fix_fixtures.jsonl"
IDENTITY = ROOT / "tests/fixtures/uud/p0_evaluation_identity.json"
SPEC = importlib.util.spec_from_file_location("evaluate_p0_contract", RUNNER)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def _create_test_archive(root: Path, directory: Path) -> tuple[Path, Path]:
    frozen = json.loads(IDENTITY.read_text(encoding="utf-8"))["frozen_identity"]
    excluded_parts = {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        "node_modules",
        "reports",
        "reports-local",
    }
    files = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if path.is_file() and not any(part in excluded_parts for part in relative.parts):
            files.append(relative)
    files.sort(key=lambda path: path.as_posix())

    archive_path = directory / "p0-contract-test.zip"
    source_hashes = {}
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in files:
            data = (root / relative).read_bytes()
            archive.writestr(relative.as_posix(), data)
            source_hashes[relative.as_posix()] = hashlib.sha256(data).hexdigest()
    sidecar_path = directory / "p0-contract-test.zip.sidecar.json"
    sidecar = {
        "archive_kind": "test-clean-snapshot",
        "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "archive_file_count": len(source_hashes),
        "base_branch": frozen["branch"],
        "base_commit_sha": frozen["commit_sha"],
        "base_tree_sha": frozen["tree_sha"],
        "source_file_sha256": source_hashes,
    }
    sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return archive_path, sidecar_path


class P0EvaluationContractTest(unittest.TestCase):
    def test_fixture_set_is_adjudicated_and_covers_known_p0_families(self) -> None:
        rows = runner._read_jsonl(FIXTURES)
        families = {row["family"] for row in rows}
        for row in rows:
            families.update(row.get("coverage_families", ()))
        self.assertEqual(len(rows), len({row["fixture_id"] for row in rows}))
        self.assertTrue(runner.MANDATORY_FIXTURE_FAMILIES <= families)
        for row in rows:
            runner._validate_case(row)
            self.assertIn(row["adjudication"]["target_classification"], runner.CLASSIFICATIONS)
            self.assertIn(row["adjudication"]["pre_fix_expected"], runner.CLASSIFICATIONS)

    def test_answer_ready_alone_is_not_a_pass_rule(self) -> None:
        status_only = {
            "fixture_id": "status-only",
            "family": "test",
            "adjudication": {"target_classification": "FAIL", "pre_fix_expected": "FAIL"},
            "deterministic": {"status_in": ["answer_ready"]},
        }
        self.assertEqual(runner._classify(status_only, {"status": "answer_ready"}, [], []), "FAIL")
        with self.assertRaises(runner.ContractError):
            runner._validate_case(
                {
                    "fixture_id": "invalid-empty-contract",
                    "family": "test",
                    "adjudication": {"target_classification": "PASS", "pre_fix_expected": "PASS"},
                }
            )

    def test_historical_catalog_integrity_is_machine_verified(self) -> None:
        identity = json.loads(IDENTITY.read_text(encoding="utf-8"))
        self.assertEqual(runner._verify_historical_baselines(ROOT, identity), [])
        altered = json.loads(json.dumps(identity))
        altered["historical_baselines"][0]["sha256"] = "0" * 64
        self.assertTrue(
            any("digest_mismatch" in error for error in runner._verify_historical_baselines(ROOT, altered))
        )
        altered = json.loads(json.dumps(identity))
        altered["historical_baselines"][0]["row_count"] = 149
        self.assertTrue(
            any("row_count_mismatch" in error for error in runner._verify_historical_baselines(ROOT, altered))
        )
        altered = json.loads(json.dumps(identity))
        altered["historical_baselines"][0]["classification_counts"]["PASS"] = 22
        self.assertTrue(
            any(
                "classification_counts_mismatch" in error
                for error in runner._verify_historical_baselines(ROOT, altered)
            )
        )
        catalog = json.loads(
            (ROOT / "tests/fixtures/uud/p0_historical_150q_baseline.json").read_text(encoding="utf-8")
        )
        self.assertEqual(catalog["schema"], runner.HISTORICAL_CATALOG_SCHEMA)
        for index, supports, viewer in ((31, 1, 0), (47, 135, 134), (145, 1, 0)):
            row = catalog["rows"][index - 1]
            self.assertEqual((row["supports_count"], row["viewer_count"]), (supports, viewer))
            self.assertNotEqual(row["supports_count"], row["viewer_count"])
        catalog["rows"][0]["query"] = "altered"
        altered = json.loads(json.dumps(identity))
        catalog_bytes = (json.dumps(catalog, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        altered["historical_baselines"][0]["sha256"] = hashlib.sha256(catalog_bytes).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            temp_root = Path(directory)
            temp_path = temp_root / "tests/fixtures/uud/p0_historical_150q_baseline.json"
            temp_path.parent.mkdir(parents=True)
            temp_path.write_bytes(catalog_bytes)
            altered["historical_baselines"] = [altered["historical_baselines"][0]]
            self.assertTrue(
                any(
                    "catalog_digest_mismatch" in error
                    for error in runner._verify_historical_baselines(temp_root, altered)
                )
            )
            semantic_catalog = json.loads(
                (ROOT / "tests/fixtures/uud/p0_historical_150q_baseline.json").read_text(encoding="utf-8")
            )
            semantic_catalog["rows"][30]["supports_count"] = 0
            semantic_catalog["rows"][30]["viewer_count"] = 1
            semantic_bytes = (json.dumps(semantic_catalog, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
            semantic_identity = json.loads(json.dumps(identity))
            semantic_identity["historical_baselines"] = [semantic_identity["historical_baselines"][0]]
            semantic_identity["historical_baselines"][0]["sha256"] = hashlib.sha256(semantic_bytes).hexdigest()
            temp_path.write_bytes(semantic_bytes)
            self.assertTrue(
                any(
                    "support_viewer_semantics_invalid" in error
                    for error in runner._verify_historical_baselines(temp_root, semantic_identity)
                )
            )

    def test_missing_recorded_bbox_evidence_invalidates_contract(self) -> None:
        self._assert_invalid_recorded_bbox_evidence(None, "recorded_evidence_missing_file")

    def test_tampered_recorded_bbox_evidence_invalidates_contract(self) -> None:
        source = (ROOT / "tests/fixtures/uud/p0_bbox_browser_evidence.json").read_text(encoding="utf-8")
        self._assert_invalid_recorded_bbox_evidence(source.replace("machine_verifiable", "tampered"), "recorded_evidence_digest_mismatch")

    def _assert_invalid_recorded_bbox_evidence(self, evidence_text: str | None, expected_error: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temp_root = Path(directory)
            evidence_path = temp_root / "bbox-evidence.json"
            if evidence_text is not None:
                evidence_path.write_text(evidence_text, encoding="utf-8")
            fixture = {
                "fixture_id": "bbox-integrity-regression",
                "family": "BBox_browser_interaction",
                "probe": "recorded_evidence",
                "recorded_evidence": {
                    "path": "bbox-evidence.json",
                    "sha256": runner._digest(ROOT / "tests/fixtures/uud/p0_bbox_browser_evidence.json"),
                    "needles": ["machine_verifiable_pre_fix_browser_observation"],
                },
                "adjudication": {"target_classification": "PASS", "pre_fix_expected": "FAIL", "reason_code": "test"},
            }
            fixtures_path = temp_root / "fixtures.jsonl"
            fixtures_path.write_text(json.dumps(fixture) + "\n", encoding="utf-8")
            report_path = temp_root / "contract.json"
            with patch.object(runner, "_bind_identity", return_value=({"identity_mode": "worktree"}, [])), patch.object(
                runner, "_verify_historical_baselines", return_value=[]
            ), patch.object(
                runner,
                "_evaluate_cases",
                side_effect=lambda _repo, cases: [runner._evaluate_recorded_case(_repo, case) for case in cases],
            ):
                result = runner.main(
                    [
                        "--repo-root",
                        str(temp_root),
                        "--fixtures",
                        str(fixtures_path),
                        "--identity",
                        str(IDENTITY),
                        "--report",
                        str(report_path),
                    ]
                )
            report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(result, 2)
        self.assertEqual(report["status"], "invalid")
        self.assertTrue(any(expected_error in error for error in report["recorded_evidence_gate"]["errors"]))
        self.assertFalse(report["fixture_counts"]["pre_fix_mismatches"] == 0 and report["identity_match"])

    def test_runner_freezes_current_prefixed_behavior_without_rewriting_old_baselines(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "p0-contract-archive.json"
            archive_path, sidecar_path = _create_test_archive(ROOT, Path(directory))
            self.assertEqual(
                runner.main(
                    [
                        "--archive",
                        str(archive_path),
                        "--archive-sidecar",
                        str(sidecar_path),
                        "--trusted-archive-sha256",
                        hashlib.sha256(archive_path.read_bytes()).hexdigest(),
                        "--report",
                        str(report_path),
                    ]
                ),
                0,
            )
            report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "pre_fix_contract_frozen")
        self.assertTrue(report["identity_match"])
        self.assertEqual(report["identity_mode"], "archive")
        self.assertEqual(report["fixture_counts"]["pre_fix_mismatches"], 0)
        self.assertGreater(report["fixture_counts"]["intentionally_failing_against_target"], 0)
        self.assertTrue(report["classification_policy"]["answer_ready_is_not_pass"])
        self.assertTrue(report["deterministic"]["separate_from_semantic"])
        self.assertTrue(report["semantic"]["separate_from_deterministic"])

    def test_self_consistent_mutated_archive_is_rejected_by_trusted_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temp_root = Path(directory)
            archive_path, sidecar_path = _create_test_archive(ROOT, temp_root)
            trusted_archive_sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()
            mutated_archive = temp_root / "mutated.zip"
            with zipfile.ZipFile(archive_path) as source, zipfile.ZipFile(
                mutated_archive, "w", compression=zipfile.ZIP_DEFLATED
            ) as target:
                for info in source.infolist():
                    data = source.read(info.filename)
                    if info.filename == "src/tjipto/runtime/service.py":
                        data = data.replace(b"class LegalRuntimeService", b"class LegalRuntimeServicE", 1)
                    target.writestr(info, data)
            mutated_sidecar = temp_root / "mutated.zip.sidecar.json"
            with zipfile.ZipFile(mutated_archive) as source:
                source_hashes = {
                    info.filename: hashlib.sha256(source.read(info.filename)).hexdigest()
                    for info in source.infolist()
                    if not info.filename.endswith("/")
                }
            frozen = json.loads(IDENTITY.read_text(encoding="utf-8"))["frozen_identity"]
            mutated_sidecar.write_text(
                json.dumps(
                    {
                        "archive_kind": "test-clean-snapshot",
                        "archive_sha256": hashlib.sha256(mutated_archive.read_bytes()).hexdigest(),
                        "archive_file_count": len(source_hashes),
                        "base_branch": frozen["branch"],
                        "base_commit_sha": frozen["commit_sha"],
                        "base_tree_sha": frozen["tree_sha"],
                        "source_file_sha256": source_hashes,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            report_path = temp_root / "mutated-contract.json"
            result = runner.main(
                [
                    "--archive",
                    str(mutated_archive),
                    "--archive-sidecar",
                    str(mutated_sidecar),
                    "--trusted-archive-sha256",
                    trusted_archive_sha,
                    "--report",
                    str(report_path),
                ]
            )
            report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(result, 2)
        self.assertEqual(report["status"], "invalid")
        self.assertTrue(any("archive_identity_mismatch" in error for error in report["identity_errors"]))
        self.assertEqual(
            report["runtime_identity"]["claimed_base_commit_sha"],
            "0097802d8330f2ff9c578f408e40fff291e411a9",
        )
        self.assertEqual(
            report["runtime_identity"]["claimed_base_tree_sha"],
            "6b0692c7c247745765a180731c400574efacf25b",
        )
        self.assertIsNone(report["runtime_identity"]["commit_sha"])
        self.assertIsNone(report["runtime_identity"]["tree_sha"])

    def test_identity_lock_contains_exact_head_and_artifact_bindings(self) -> None:
        identity = json.loads(IDENTITY.read_text(encoding="utf-8"))
        frozen = identity["frozen_identity"]
        self.assertEqual(frozen["commit_sha"], "0097802d8330f2ff9c578f408e40fff291e411a9")
        self.assertEqual(frozen["branch"], "codex/hybrid-research")
        self.assertEqual(len(frozen["manifest_sha256"]), 64)
        self.assertEqual(len(frozen["validation_sha256"]), 64)
        self.assertEqual(len(frozen["case_set_sha256"]), 64)
        self.assertEqual(len(frozen["evaluator_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
