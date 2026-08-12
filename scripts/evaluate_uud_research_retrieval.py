from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "tests/fixtures/uud/research_retrieval_cases.jsonl"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the legal-research retrieval benchmark.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--base-commit", default=None)
    parser.add_argument("--runtime-commit", default=None)
    parser.add_argument("--runtime-root", type=Path, default=ROOT)
    parser.add_argument("--runtime-digest", default=None)
    parser.add_argument("--materialize-commit", default=None)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    if args.materialize_commit:
        return _evaluate_materialized(args)
    return _evaluate_runtime(args)


def _evaluate_materialized(args: argparse.Namespace) -> int:
    with tempfile.TemporaryDirectory(prefix="tjipto-research-runtime-") as temp:
        runtime_root = Path(temp)
        archive = _git_archive(args.materialize_commit)
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as bundle:
            bundle.extractall(runtime_root, filter="data")
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--cases", str(args.cases.resolve()),
            "--runtime-root", str(runtime_root),
            "--runtime-commit", args.materialize_commit,
            "--runtime-digest", hashlib.sha256(archive).hexdigest(),
        ]
        if args.base_commit:
            command.extend(("--base-commit", args.base_commit))
        with tempfile.TemporaryDirectory(prefix="tjipto-research-report-") as report_dir:
            child_report = Path(report_dir) / "research.json"
            command.extend(("--report", str(child_report)))
            completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
            if completed.stdout:
                print(completed.stdout, end="")
            if completed.stderr:
                print(completed.stderr, file=sys.stderr, end="")
            if not child_report.is_file():
                return completed.returncode or 1
            report = json.loads(child_report.read_text(encoding="utf-8"))
            report["evaluation_identity"]["materialized_runtime"] = True
            if args.report:
                args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            return completed.returncode


def _evaluate_runtime(args: argparse.Namespace) -> int:
    runtime_root = args.runtime_root.resolve()
    sys.path.insert(0, str(runtime_root / "src"))
    from tjipto.runtime.service import LegalRuntimeService

    cases = _read_jsonl(args.cases)
    service = LegalRuntimeService(runtime_root)
    results = [_evaluate(case, service) for case in cases]
    invalid = sum(row["execution_status"] == "invalid" for row in results)
    gaps = sum(row["capability_status"] == "gap" for row in results)
    met = sum(row["capability_status"] == "met" for row in results)
    blocking_failures = sum(row["blocking"] and row["capability_status"] != "met" for row in results)
    runtime_commit = args.runtime_commit or _git_head(runtime_root)
    runtime_tree = _git_tree(runtime_commit) if runtime_commit != "unavailable" else "unavailable"
    runtime_digest = args.runtime_digest or _git_archive_digest(runtime_commit, runtime_root)
    report: dict[str, Any] = {
        "execution_status": "invalid" if invalid else "valid",
        "capability_status": "gap" if gaps else "met",
        "met": met,
        "gap": gaps,
        "invalid": invalid,
        "blocking_failures": blocking_failures,
        "counts": {"cases": len(results), "met": met, "gap": gaps, "invalid": invalid},
        "results": results,
        "evaluation_identity": {
            "base_commit": args.base_commit or _git_head(ROOT),
            "case_set_sha256": _sha256(args.cases),
            "evaluator_sha256": _sha256(Path(__file__)),
            "case_count": len(cases),
            "runtime_commit": runtime_commit,
            "runtime_tree_sha": runtime_tree,
            "runtime_snapshot_sha256": runtime_digest,
            "family_counts": {family: sum(row["family"] == family for row in cases) for family in sorted({row["family"] for row in cases})},
            "behavior_counts": {behavior: sum(row["expected_behavior"] == behavior for row in cases) for behavior in ("retrieve", "abstain", "clarify")},
        },
    }
    if args.report:
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"research_retrieval: execution={report['execution_status']} met={met} gap={gaps} invalid={invalid}")
    for row in results:
        if row["capability_status"] != "met":
            print(f"RESEARCH_GAP: {row['case_id']} :: {'; '.join(row['errors'])}")
    return 0 if not invalid and not blocking_failures else 1


def _evaluate(case: dict[str, Any], service: Any) -> dict[str, Any]:
    try:
        response = service.ask(case["corpus_id"], case["query"])
        citations = tuple(response.get("citations", ())) + tuple(response.get("historical_citations", ()))
        published_ids = [str(row.get("evidence_id")) for row in citations if row.get("evidence_id")]
        retrieved_ids = [
            str(row.get("evidence_id"))
            for row in response.get("matches", ())
            if row.get("evidence_id")
        ]
        actual_citations = [str(row.get("citation")) for row in citations if row.get("citation")]
        errors = _compare(case, response, published_ids, retrieved_ids, actual_citations)
        execution_status = "valid"
    except Exception as error:  # the report distinguishes evaluator/runtime failure from capability gaps
        response, published_ids, retrieved_ids, actual_citations = {}, [], [], []
        errors = [f"execution:{type(error).__name__}:{error}"]
        execution_status = "invalid"
    capability_status = "met" if not errors else "gap"
    return {
        "case_id": case["case_id"],
        "family": case["family"],
        "query": case["query"],
        "expected_behavior": case["expected_behavior"],
        "blocking": bool(case.get("blocking")),
        "execution_status": execution_status,
        "capability_status": capability_status,
        "met": capability_status == "met",
        "gap": capability_status == "gap",
        "invalid": execution_status == "invalid",
        "errors": errors,
        "gold": {key: case.get(key) for key in ("expected_behavior", "expected_status", "expected_route", "expected_clarification_kind", "gold_support_groups", "alternative_support_groups")},
            "observed": {
            "status": response.get("status"),
            "route": response.get("route"),
            "intent": response.get("intent"),
            "support_ids": published_ids,
            "retrieved_support_ids": retrieved_ids,
            "citations": actual_citations,
            "clarification_kind": response.get("clarification_kind"),
        },
    }


def _compare(
    case: dict[str, Any],
    response: dict,
    published_ids: list[str],
    retrieved_ids: list[str],
    actual_citations: list[str],
) -> list[str]:
    errors: list[str] = []
    if response.get("status") != case.get("expected_status"):
        errors.append(f"status:{response.get('status')}!={case.get('expected_status')}")
    if case.get("expected_route") and response.get("route") != case["expected_route"]:
        errors.append(f"route:{response.get('route')}!={case['expected_route']}")
    if case.get("expected_clarification_kind") and response.get("clarification_kind") != case["expected_clarification_kind"]:
        errors.append(f"clarification_kind:{response.get('clarification_kind')}")
    required_citations = set(case.get("expected_citations") or ())
    if required_citations and not required_citations <= set(actual_citations):
        errors.append("citations_missing")
    groups = [set(group) for group in (case.get("gold_support_groups") or ())]
    alternatives = [set(group) for group in (case.get("alternative_support_groups") or ())]
    if groups or alternatives:
        # Research support-group recall measures the verified candidate set,
        # while published_ids records the smaller server-selected citation
        # set.  Candidate presence is the retrieval contract; publication
        # remains guarded by the runtime's support validators.
        if not any(group <= set(retrieved_ids) for group in (*groups, *alternatives)):
            errors.append("support_group_missing")
    return errors


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _git_tree(commit: str) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", f"{commit}^{{tree}}"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _git_archive(commit: str) -> bytes:
    return subprocess.check_output(["git", "archive", "--format=tar", commit], cwd=ROOT)


def _git_archive_digest(commit: str, root: Path) -> str:
    if commit != "unavailable":
        try:
            return hashlib.sha256(_git_archive(commit)).hexdigest()
        except (OSError, subprocess.CalledProcessError):
            pass
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file() and ".git" not in path.parts:
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
