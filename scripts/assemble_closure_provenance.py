from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path
import sys

try:
    from scripts.measure_command import compare_pytest_resources
except ModuleNotFoundError:  # Direct script execution places scripts/ on sys.path.
    from measure_command import compare_pytest_resources


BACKEND_GATES = frozenset((
    "toolchain", "compileall", "unittest", "pytest_run_1", "retrieval_evaluation", "research_retrieval_evaluation", "semantic_generalization_evaluation", "answer_evaluation",
    "source_text_evaluation", "meaningful_support_evaluation", "support_reachability_evaluation",
    "artifact_validate", "clean_tree", "artifact_rebuild",
    "ruff", "mypy", "bandit", "pip_check", "pip_audit", "pytest_run_2",
    "live_planner_integration", "dense_promotion_attestation", "true_hybrid_activation",
    "release_a", "release_b",
))
WEB_GATES = frozenset(("toolchain", "web_test", "web_lint", "web_typecheck", "web_build", "web_smoke"))
SHARED_FIELDS = (
    "repository", "workflow_ref", "workflow_sha", "workflow_repository", "workflow_file_path",
    "commit_sha", "tree_sha", "parent_sha", "ref", "run_id", "run_attempt", "run_identity_id",
)
JOB_FIELDS = ("job_key", "job_check_run_id", "job_identity_id")
BACKEND_EVIDENCE = frozenset((
    "unittest.json", "pytest-run-1.json", "pytest-run-2.json", "pytest-resource-comparison.json",
    "retrieval-evaluation.json", "retrieval-evaluation-command.json", "research-retrieval.json", "research-retrieval-command.json", "semantic-generalization.json", "semantic-generalization-command.json", "answer-evaluation.json", "answer-evaluation-command.json",
    "source-text-evaluation.json", "meaningful-support-evaluation.json",
    "support-reachability-evaluation.json",
    "live-planner-integration.json", "dense-promotion-attestation.json",
    "artifact-validation.json", "artifact-rebuild.json", "pip-audit.json", "pip-inspect.json", "clean-tree.txt",
))
WEB_EVIDENCE = frozenset(("npm-audit.json", "browser.json"))


class ClosureError(ValueError):
    pass


def _load(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ClosureError(f"missing artifact: {path}") from error
    except json.JSONDecodeError as error:
        raise ClosureError(f"invalid JSON artifact: {path}") from error


def _load_lines(path: Path) -> list[dict]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as error:
        raise ClosureError(f"missing artifact: {path}") from error
    try:
        return [json.loads(line) for line in lines if line]
    except json.JSONDecodeError as error:
        raise ClosureError(f"invalid JSON artifact: {path}") from error


def _job_identity(identity: dict) -> str:
    fields = {key: identity.get(key) for key in ("run_identity_id", "job_key", "job_check_run_id")}
    return sha256(json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _validate_job(identity: dict, expected_key: str) -> None:
    missing = [key for key in (*SHARED_FIELDS, *JOB_FIELDS) if not identity.get(key)]
    if missing:
        raise ClosureError(f"missing identity fields for {expected_key}: {', '.join(missing)}")
    if identity["job_key"] != expected_key:
        raise ClosureError(f"wrong job key: expected {expected_key}")
    if not str(identity["job_check_run_id"]).isdigit() or int(identity["job_check_run_id"]) <= 0:
        raise ClosureError(f"invalid check run ID for {expected_key}")
    if identity["job_identity_id"] != _job_identity(identity):
        raise ClosureError(f"invalid job identity for {expected_key}")


def _validate_gates(gates: list[dict], identity: dict, expected: frozenset[str], job_key: str) -> None:
    names = {gate.get("gate") or gate.get("attributes", {}).get("gate") for gate in gates}
    if names != expected:
        raise ClosureError(f"missing or unexpected measured gates for {job_key}")
    for gate in gates:
        name = gate.get("gate") or gate.get("attributes", {}).get("gate")
        status = gate.get("status") or gate.get("attributes", {}).get("status")
        if gate.get("event") != "ci_gate" or gate.get("exit_code") != 0 or status != "passed":
            raise ClosureError(f"nonzero or invalid gate record: {name}")
        if any(gate.get(key) != identity.get(key) for key in (*SHARED_FIELDS, *JOB_FIELDS)):
            raise ClosureError(f"gate identity mismatch: {name}")


def _evidence_manifest(root: Path) -> dict[str, dict[str, int | str]]:
    return {
        path.relative_to(root).as_posix(): {
            "sha256": sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def _validate_evidence(root: Path, required: frozenset[str], job_key: str) -> None:
    missing = sorted(name for name in required if not (root / name).is_file())
    if missing:
        raise ClosureError(f"missing {job_key} evidence: {', '.join(missing)}")


def _validate_resource(backend: Path, identity: dict) -> dict:
    first = _load(backend / "pytest-run-1.json")
    second = _load(backend / "pytest-run-2.json")
    comparison = _load(backend / "pytest-resource-comparison.json")
    if first.get("exit_code") != 0 or second.get("exit_code") != 0:
        raise ClosureError("full pytest run failed")
    if comparison.get("run_identity_id") != identity["run_identity_id"]:
        raise ClosureError("resource comparison provenance mismatch")
    expected = compare_pytest_resources(first, second, identity)
    if any(comparison.get(key) != expected[key] for key in expected):
        raise ClosureError("resource comparison content mismatch")
    if not all(comparison.get(key) is True for key in ("both_pytest_runs_pass", "rss_limit_pass", "rss_stability_pass")):
        raise ClosureError("pytest RSS policy failed")
    if comparison.get("wall_status") not in {"stable", "variable", "unavailable"}:
        raise ClosureError("invalid diagnostic wall status")
    return {"run_1": first, "run_2": second, "comparison": comparison}


def _validate_stage_evidence(backend: Path, identity: dict) -> dict:
    semantic = _load(backend / "semantic-generalization.json")
    planner = _load(backend / "live-planner-integration.json")
    dense = _load(backend / "dense-promotion-attestation.json")
    semantic_identity = semantic.get("runtime_identity") or {}
    planner_identity = planner.get("runtime_identity") or {}
    dense_identity = dense.get("runtime_identity") or {}
    expected_identity = {"commit": identity["commit_sha"], "tree": identity["tree_sha"]}
    if semantic.get("status") != "valid" or semantic_identity != expected_identity:
        raise ClosureError("semantic end-to-end evidence invalid or stale")
    if (
        semantic.get("failures")
        or (semantic.get("metrics") or {}).get("hard_negative_fp") != 0
        or (semantic.get("metrics") or {}).get("query_drift_rate") != 0
    ):
        raise ClosureError("semantic end-to-end contract failed")
    if planner.get("status") != "valid" or planner_identity != expected_identity:
        raise ClosureError("live planner evidence invalid or stale")
    if dense.get("status") != "valid" or dense_identity != expected_identity:
        raise ClosureError("dense promotion attestation invalid or stale")
    activation = dense.get("activation") or {}
    fusion = activation.get("fusion") or {}
    if not (
        activation.get("dense_configured") is True
        and activation.get("dense_runtime_available") is True
        and activation.get("hybrid_active") is True
        and activation.get("route") == "hybrid"
        and fusion.get("algorithm") == "rrf_rank_only"
        and fusion.get("lane_candidate_counts", {}).get("bm25", 0) > 0
        and fusion.get("lane_candidate_counts", {}).get("dense", 0) > 0
        and {"bm25", "dense"} <= set(activation.get("contributing_lanes") or ())
    ):
        raise ClosureError("true hybrid activation evidence failed")
    return {
        "semantic_end_to_end": semantic,
        "live_planner": planner,
        "dense_promotion": dense,
    }


def _validate_uploads(uploads: dict[str, dict[str, str]]) -> None:
    for job_key in ("backend", "web"):
        upload = uploads.get(job_key, {})
        if not str(upload.get("artifact_id", "")).isdigit() or len(str(upload.get("artifact_digest", ""))) != 64:
            raise ClosureError(f"invalid {job_key} artifact upload identity")


def _release(backend: Path, identity: dict) -> dict:
    first = _load(backend / "release-one.json")
    second = _load(backend / "release-two.json")
    first_sidecar = _load(backend / "release-one.sidecar.json")
    second_sidecar = _load(backend / "release-two.sidecar.json")
    comparison = _load(backend / "release-comparison.json")
    for release in (first, second):
        if release.get("commit_sha") != identity["commit_sha"] or release.get("tree_sha") != identity["tree_sha"]:
            raise ClosureError("release identity mismatch")
        inventory = release.get("archive_inventory", {})
        if release.get("archive_forbidden_entries") or inventory.get("status") != "passed" or any(inventory.get(key) for key in ("missing_files", "extra_files", "digest_mismatches")) or any(release.get("candidate_checks", {}).values()):
            raise ClosureError("release validation failed")
    if not all(comparison.get(key) is True for key in ("archive_sha256_equal", "corpora_equal", "sidecars_equal")):
        raise ClosureError("release reproducibility comparison failed")
    if comparison.get("forbidden_entry_count") != 0 or comparison.get("run_identity_id") != identity["run_identity_id"]:
        raise ClosureError("release comparison provenance mismatch")
    if first_sidecar.get("corpora") != second_sidecar.get("corpora"):
        raise ClosureError("release corpus identity mismatch")
    expected_identity = {"commit": identity["commit_sha"], "tree": identity["tree_sha"]}
    for sidecar in (first_sidecar, second_sidecar):
        attestation = sidecar.get("dense_promotion_attestation") or {}
        if attestation.get("status") != "valid" or attestation.get("runtime_identity") != expected_identity:
            raise ClosureError("release dense attestation missing or stale")
    return {"a": first, "b": second, "a_sidecar": first_sidecar, "b_sidecar": second_sidecar, "comparison": comparison}


def assemble(
    backend: Path,
    web: Path,
    closure_identity_path: Path | None = None,
    *,
    upstream_results: dict[str, str] | None = None,
    artifact_uploads: dict[str, dict[str, str]] | None = None,
) -> dict:
    results = upstream_results or {"backend": "success", "web": "success"}
    uploads = artifact_uploads or {}
    if set(results) != {"backend", "web"}:
        raise ClosureError("missing upstream job result")
    if any(result != "success" for result in results.values()):
        jobs = {}
        if closure_identity_path and closure_identity_path.is_file():
            closure_identity = _load(closure_identity_path)
            _validate_job(closure_identity, "closure")
            jobs["closure"] = closure_identity
        return {
            "schema_version": 1,
            "status": "partial",
            "upstream_results": results,
            "jobs": jobs,
            "artifact_uploads": uploads,
            "available_evidence": {
                "backend": _evidence_manifest(backend) if backend.is_dir() else {},
                "web": _evidence_manifest(web) if web.is_dir() else {},
            },
        }
    backend_identity = _load(backend / "run-identity.json")
    web_identity = _load(web / "run-identity.json")
    _validate_job(backend_identity, "backend")
    _validate_job(web_identity, "web")
    if any(backend_identity[key] != web_identity[key] for key in SHARED_FIELDS):
        raise ClosureError("backend and web run identities differ")
    if backend_identity["job_check_run_id"] == web_identity["job_check_run_id"] or backend_identity["job_identity_id"] == web_identity["job_identity_id"]:
        raise ClosureError("duplicate backend/web job identity")
    backend_gates = _load_lines(backend / "ci-gates.jsonl")
    web_gates = _load_lines(web / "ci-gates.jsonl")
    _validate_gates(backend_gates, backend_identity, BACKEND_GATES, "backend")
    _validate_gates(web_gates, web_identity, WEB_GATES, "web")
    _validate_evidence(backend, BACKEND_EVIDENCE, "backend")
    _validate_evidence(web, WEB_EVIDENCE, "web")
    _validate_uploads(uploads)
    jobs = {"backend": backend_identity, "web": web_identity}
    if closure_identity_path:
        closure_identity = _load(closure_identity_path)
        _validate_job(closure_identity, "closure")
        if any(backend_identity[key] != closure_identity[key] for key in SHARED_FIELDS):
            raise ClosureError("closure run identity differs")
        if closure_identity["job_check_run_id"] in {backend_identity["job_check_run_id"], web_identity["job_check_run_id"]}:
            raise ClosureError("duplicate closure job identity")
        jobs["closure"] = closure_identity
    stage_evidence = _validate_stage_evidence(backend, backend_identity)
    return {
        "schema_version": 1,
        "status": "complete",
        "upstream_results": results,
        "run_identity": {key: backend_identity[key] for key in SHARED_FIELDS},
        "jobs": jobs,
        "gates": {"backend": backend_gates, "web": web_gates},
        "resource": _validate_resource(backend, backend_identity),
        "stage_6_11": stage_evidence,
        "artifact_uploads": uploads,
        "evidence": {"backend": _evidence_manifest(backend), "web": _evidence_manifest(web)},
        "release": _release(backend, backend_identity),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fail-closed closure provenance aggregation.")
    parser.add_argument("--backend-dir", type=Path, required=True)
    parser.add_argument("--web-dir", type=Path, required=True)
    parser.add_argument("--closure-identity", type=Path)
    parser.add_argument("--backend-result", required=True)
    parser.add_argument("--web-result", required=True)
    parser.add_argument("--backend-artifact-id", default="")
    parser.add_argument("--backend-artifact-digest", default="")
    parser.add_argument("--web-artifact-id", default="")
    parser.add_argument("--web-artifact-digest", default="")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    results = {"backend": args.backend_result, "web": args.web_result}
    uploads = {
        "backend": {"artifact_id": args.backend_artifact_id, "artifact_digest": args.backend_artifact_digest},
        "web": {"artifact_id": args.web_artifact_id, "artifact_digest": args.web_artifact_digest},
    }
    try:
        result = assemble(
            args.backend_dir, args.web_dir, args.closure_identity,
            upstream_results=results, artifact_uploads=uploads,
        )
    except ClosureError as error:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "schema_version": 1, "status": "invalid", "upstream_results": results,
            "artifact_uploads": uploads, "error": str(error),
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"closure provenance failed: {error}", file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "gate_count": sum(len(rows) for rows in result.get("gates", {}).values()),
        "run_identity_id": result.get("run_identity", {}).get("run_identity_id"),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
