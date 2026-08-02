from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path
import sys


BACKEND_GATES = frozenset((
    "toolchain", "compileall", "unittest", "pytest_run_1", "retrieval_evaluation",
    "source_text_evaluation", "artifact_validate", "clean_tree", "artifact_rebuild",
    "ruff", "mypy", "bandit", "pip_check", "pip_audit", "pytest_run_2", "release_a", "release_b",
))
WEB_GATES = frozenset(("toolchain", "web_test", "web_lint", "web_typecheck", "web_build", "web_smoke"))
SHARED_FIELDS = (
    "repository", "workflow_ref", "workflow_sha", "workflow_repository", "workflow_file_path",
    "commit_sha", "tree_sha", "parent_sha", "ref", "run_id", "run_attempt", "run_identity_id",
)
JOB_FIELDS = ("job_key", "job_check_run_id", "job_identity_id")


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
    return {"a": first, "b": second, "a_sidecar": first_sidecar, "b_sidecar": second_sidecar, "comparison": comparison}


def assemble(backend: Path, web: Path, closure_identity_path: Path | None = None) -> dict:
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
    jobs = {"backend": backend_identity, "web": web_identity}
    if closure_identity_path:
        closure_identity = _load(closure_identity_path)
        _validate_job(closure_identity, "closure")
        if any(backend_identity[key] != closure_identity[key] for key in SHARED_FIELDS):
            raise ClosureError("closure run identity differs")
        if closure_identity["job_check_run_id"] in {backend_identity["job_check_run_id"], web_identity["job_check_run_id"]}:
            raise ClosureError("duplicate closure job identity")
        jobs["closure"] = closure_identity
    return {
        "schema_version": 1,
        "run_identity": {key: backend_identity[key] for key in SHARED_FIELDS},
        "jobs": jobs,
        "gates": {"backend": backend_gates, "web": web_gates},
        "release": _release(backend, backend_identity),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fail-closed closure provenance aggregation.")
    parser.add_argument("--backend-dir", type=Path, required=True)
    parser.add_argument("--web-dir", type=Path, required=True)
    parser.add_argument("--closure-identity", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = assemble(args.backend_dir, args.web_dir, args.closure_identity)
    except ClosureError as error:
        print(f"closure provenance failed: {error}", file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"gate_count": sum(len(rows) for rows in result["gates"].values()), "run_identity_id": result["run_identity"]["run_identity_id"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
