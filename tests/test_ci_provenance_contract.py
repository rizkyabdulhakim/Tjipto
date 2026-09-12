from __future__ import annotations

import os
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts import measure_command


ROOT = Path(__file__).resolve().parents[1]


class CiProvenanceContractTests(unittest.TestCase):
    def _environment(self, job: str, check_run_id: str) -> dict[str, str]:
        return {
            "GITHUB_ACTIONS": "true",
            "GITHUB_REPOSITORY": "rizkyabdulhakim/Tjipto",
            "GITHUB_SHA": "b" * 40,
            "TJIPTO_EXACT_HEAD_SHA": "d" * 40,
            "GITHUB_REF": "refs/heads/codex/p0-p1-source-text-closure",
            "GITHUB_REF_NAME": "codex/p0-p1-source-text-closure",
            "GITHUB_RUN_ID": "30730010658",
            "GITHUB_RUN_ATTEMPT": "1",
            "GITHUB_JOB": job,
            "TJIPTO_JOB_CHECK_RUN_ID": check_run_id,
            "TJIPTO_JOB_WORKFLOW_REF": "rizkyabdulhakim/Tjipto/.github/workflows/ci.yml@refs/heads/codex/p0-p1-source-text-closure",
            "TJIPTO_JOB_WORKFLOW_SHA": "a" * 40,
            "TJIPTO_JOB_WORKFLOW_REPOSITORY": "rizkyabdulhakim/Tjipto",
            "TJIPTO_JOB_WORKFLOW_FILE_PATH": ".github/workflows/ci.yml",
        }

    def _identity(self, environment: dict[str, str]) -> dict:
        with patch.dict(os.environ, {"PATH": os.environ["PATH"]} | environment, clear=True):
            with patch.object(measure_command, "_command_output", return_value="c" * 40):
                return measure_command._execution_identity(ROOT)

    def test_run_identity_is_shared_and_job_identity_is_distinct(self) -> None:
        backend = self._identity(self._environment("backend", "1001"))
        web = self._identity(self._environment("web", "1002"))
        self.assertEqual(backend["run_identity_id"], web["run_identity_id"])
        self.assertNotEqual(backend["job_identity_id"], web["job_identity_id"])
        self.assertEqual(backend["job_check_run_id"], "1001")
        self.assertEqual(backend["commit_sha"], "d" * 40)

    def test_missing_or_invalid_typed_identity_fails_closed(self) -> None:
        missing = self._environment("backend", "1001")
        del missing["TJIPTO_JOB_CHECK_RUN_ID"]
        with self.assertRaisesRegex(ValueError, "job_check_run_id"):
            self._identity(missing)
        invalid = self._environment("backend", "not-a-number")
        with self.assertRaisesRegex(ValueError, "job_check_run_id"):
            self._identity(invalid)

    def test_resource_policy_is_symmetric_and_wall_is_diagnostic(self) -> None:
        first = {"exit_code": 0, "peak_rss_bytes": 700 * 1024 * 1024, "wall_seconds": 10}
        second = {"exit_code": 0, "peak_rss_bytes": 680 * 1024 * 1024, "wall_seconds": 20}
        identity = {"run_identity_id": "identity"}
        forward = measure_command.compare_pytest_resources(first, second, identity)
        reverse = measure_command.compare_pytest_resources(second, first, identity)
        for key in ("rss_ratio", "rss_limit_pass", "rss_stability_pass", "wall_ratio", "wall_status"):
            self.assertEqual(forward[key], reverse[key])
        self.assertTrue(forward["rss_limit_pass"])
        self.assertTrue(forward["rss_stability_pass"])
        self.assertEqual(forward["wall_status"], "variable")
        self.assertEqual(forward["wall_policy"], "diagnostic_not_benchmark")
