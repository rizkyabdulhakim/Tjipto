from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts import assemble_closure_provenance as closure


class ClosureProvenanceContractTests(unittest.TestCase):
    uploads = {
        "backend": {"artifact_id": "201", "artifact_digest": "1" * 64},
        "web": {"artifact_id": "202", "artifact_digest": "2" * 64},
    }

    def _identity(self, job: str, check_run_id: str) -> dict:
        identity = {
            "repository": "rizkyabdulhakim/Tjipto",
            "workflow_ref": "rizkyabdulhakim/Tjipto/.github/workflows/ci.yml@refs/heads/main",
            "workflow_sha": "a" * 40,
            "workflow_repository": "rizkyabdulhakim/Tjipto",
            "workflow_file_path": ".github/workflows/ci.yml",
            "commit_sha": "b" * 40,
            "tree_sha": "c" * 40,
            "parent_sha": "d" * 40,
            "python_lock_sha256": "f" * 64,
            "dense_lock_sha256": "0" * 64,
            "ref": "refs/heads/main",
            "run_id": "123",
            "run_attempt": "1",
            "run_identity_id": "e" * 64,
            "job_key": job,
            "job_check_run_id": check_run_id,
        }
        identity["job_identity_id"] = closure._job_identity(identity)
        return identity

    def _write(self, path: Path, value: object) -> None:
        path.write_text(json.dumps(value), encoding="utf-8")

    def _evidence(self, root: Path) -> tuple[Path, Path]:
        backend, web = root / "backend", root / "web"
        backend.mkdir()
        web.mkdir()
        backend_identity = self._identity("backend", "101")
        web_identity = self._identity("web", "102")
        self._write(backend / "run-identity.json", backend_identity)
        self._write(web / "run-identity.json", web_identity)
        for directory, identity, gates in ((backend, backend_identity, closure.BACKEND_GATES), (web, web_identity, closure.WEB_GATES)):
            rows = [identity | {"event": "ci_gate", "gate": gate, "status": "passed", "exit_code": 0} for gate in sorted(gates)]
            (directory / "ci-gates.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        release = {
            "commit_sha": backend_identity["commit_sha"],
            "tree_sha": backend_identity["tree_sha"],
            "archive_forbidden_entries": [],
            "archive_inventory": {"status": "passed", "missing_files": [], "extra_files": [], "digest_mismatches": []},
            "candidate_checks": {"compileall": 0, "unittest": 0},
        }
        for name in ("release-one.json", "release-two.json"):
            self._write(backend / name, release)
        sidecar = {
            "corpora": {"uud_1945": {"artifact_set_digest": "f" * 64}},
            "dense_promotion_attestation": {
                "status": "valid",
                "runtime_identity": {"commit": backend_identity["commit_sha"], "tree": backend_identity["tree_sha"]},
            },
        }
        self._write(backend / "release-one.sidecar.json", sidecar)
        self._write(backend / "release-two.sidecar.json", sidecar)
        self._write(backend / "release-comparison.json", {"run_identity_id": backend_identity["run_identity_id"], "archive_sha256_equal": True, "corpora_equal": True, "sidecars_equal": True, "forbidden_entry_count": 0})
        command = {"exit_code": 0, "peak_rss_bytes": 1, "wall_seconds": 1}
        for name in closure.BACKEND_EVIDENCE:
            path = backend / name
            if path.exists():
                continue
            if path.suffix == ".txt":
                path.write_text("", encoding="utf-8")
            elif name == "pytest-resource-comparison.json":
                self._write(path, closure.compare_pytest_resources(command, command, backend_identity))
            elif name == "semantic-generalization.json":
                self._write(path, {
                    "status": "valid",
                    "runtime_identity": {"commit": backend_identity["commit_sha"], "tree": backend_identity["tree_sha"]},
                    "failures": [],
                    "metrics": {"hard_negative_fp": 0, "query_drift_rate": 0},
                })
            elif name == "live-planner-integration.json":
                self._write(path, {
                    "status": "valid",
                    "runtime_identity": {"commit": backend_identity["commit_sha"], "tree": backend_identity["tree_sha"]},
                })
            elif name == "dense-promotion-attestation.json":
                self._write(path, {
                    "schema_version": 1,
                    "kind": "post_build_dense_runtime_attestation",
                    "status": "valid",
                    "runtime_identity": {"commit": backend_identity["commit_sha"], "tree": backend_identity["tree_sha"]},
                    "activation": {
                        "dense_configured": True,
                        "dense_runtime_available": True,
                        "hybrid_active": True,
                        "route": "hybrid",
                        "contributing_lanes": ["bm25", "dense"],
                        "fusion": {
                            "algorithm": "rrf_rank_only",
                            "lane_candidate_counts": {"bm25": 1, "dense": 1},
                        },
                    },
                })
            else:
                self._write(path, command)
        for name in closure.WEB_EVIDENCE:
            self._write(web / name, {})
        return backend, web

    def test_assembles_complete_bound_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            backend, web = self._evidence(Path(temporary))
            result = closure.assemble(backend, web, artifact_uploads=self.uploads)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(
                sum(len(gates) for gates in result["gates"].values()),
                len(closure.BACKEND_GATES) + len(closure.WEB_GATES),
            )
            self.assertEqual(set(result["evidence"]), {"backend", "web"})

    def test_failed_upstream_produces_typed_partial_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            backend, web = self._evidence(Path(temporary))
            result = closure.assemble(
                backend, web, upstream_results={"backend": "failure", "web": "success"},
                artifact_uploads=self.uploads,
            )
            self.assertEqual(result["status"], "partial")
            self.assertEqual(result["upstream_results"], {"backend": "failure", "web": "success"})
            self.assertTrue(result["available_evidence"]["backend"])

    def test_identity_mutation_and_missing_artifact_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            backend, web = self._evidence(Path(temporary))
            rows = (backend / "ci-gates.jsonl").read_text(encoding="utf-8").splitlines()
            changed = json.loads(rows[0])
            changed["run_id"] = "different"
            rows[0] = json.dumps(changed)
            (backend / "ci-gates.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(closure.ClosureError, "gate identity mismatch"):
                closure.assemble(backend, web, artifact_uploads=self.uploads)
            (web / "ci-gates.jsonl").unlink()
            with self.assertRaisesRegex(closure.ClosureError, "missing artifact"):
                closure.assemble(backend, web, artifact_uploads=self.uploads)

    def test_duplicate_job_identity_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            backend, web = self._evidence(Path(temporary))
            duplicate = self._identity("web", "101")
            self._write(web / "run-identity.json", duplicate)
            with self.assertRaisesRegex(closure.ClosureError, "duplicate backend/web"):
                closure.assemble(backend, web, artifact_uploads=self.uploads)

    def test_dense_attestation_schema_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            backend, web = self._evidence(Path(temporary))
            attestation_path = backend / "dense-promotion-attestation.json"
            attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
            attestation["kind"] = "untrusted"
            self._write(attestation_path, attestation)
            with self.assertRaisesRegex(closure.ClosureError, "dense promotion attestation invalid or stale"):
                closure.assemble(backend, web, artifact_uploads=self.uploads)

    def test_mixed_run_attempts_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            backend, web = self._evidence(Path(temporary))
            mixed = self._identity("web", "102")
            mixed["run_attempt"] = "2"
            mixed["run_identity_id"] = "f" * 64
            mixed["job_identity_id"] = closure._job_identity(mixed)
            self._write(web / "run-identity.json", mixed)
            with self.assertRaisesRegex(closure.ClosureError, "backend and web run identities differ"):
                closure.assemble(backend, web, artifact_uploads=self.uploads)
