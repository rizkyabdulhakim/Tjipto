from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "tests/fixtures/uud/answer_cases.jsonl"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate grounded public answers without prose snapshots.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--runtime-commit")
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    cases = _read_jsonl(args.cases)
    repo_root = args.repo_root.resolve()
    service_type, request_handler = _runtime(repo_root)
    service = service_type(repo_root)
    results = [_evaluate(case, service, request_handler) for case in cases]
    report = {
        "status": "pass" if all(row["passed"] for row in results) else "fail",
        "counts": {"pass": sum(row["passed"] for row in results), "fail": sum(not row["passed"] for row in results)},
        "metrics": _metrics(cases, results),
        "identity": {
            "runtime_commit": args.runtime_commit,
            "runtime_root": str(repo_root),
            "case_set_sha256": _digest(args.cases),
            "evaluator_sha256": _digest(Path(__file__)),
            "case_count": len(cases),
        },
        "results": results,
    }
    if args.report:
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"answer_eval: PASS={report['counts']['pass']} FAIL={report['counts']['fail']}")
    for row in results:
        if not row["passed"]:
            print(f"FAIL: {row['case_id']} :: {'; '.join(row['errors'])}")
    return int(report["status"] != "pass")


def _evaluate(case: dict[str, Any], service, request_handler) -> dict[str, Any]:
    response = service.ask("uud", case["query"])
    public = request_handler("uud", "ask", {"query": case["query"]}, service=service)
    support_rows = tuple(
        row
        for field in ("evidence", "citations", "historical_citations", "metadata_support", "trace_support", "relation_support")
        for row in response.get(field, ())
        if isinstance(row, dict)
    )
    support_ids = set(_ids(support_rows)) | {
        str(item)
        for claim in response.get("claim_support", ())
        for item in claim.get("support_evidence_ids", ())
    }
    answer = " ".join(str(response.get("answer") or "").split()).casefold()
    roles = {str(row["source_role"]) for row in support_rows if row.get("source_role")}
    temporal = {str(row.get("temporal_context") or row.get("source_role")) for row in support_rows if row.get("temporal_context") or row.get("source_role")}
    public_targets = [
        support.get("viewer_target")
        for support in public.get("supports", ())
        if isinstance(support, dict) and support.get("viewer_target", {}).get("can_resolve") is True
    ]
    claim_statuses = [str(row.get("status")) for row in response.get("claim_support", ())]
    errors: list[str] = []
    _expect(errors, "status", case["expected_status"], response.get("status"))
    _expect(errors, "route", case["expected_route"], response.get("route"))
    _expect(errors, "behavior", case["behavior"], _behavior(response.get("status")))
    for item in case["required_support_ids"]:
        if item not in support_ids:
            errors.append(f"missing_support:{item}")
    for item in case["forbidden_support_ids"]:
        if item in support_ids:
            errors.append(f"forbidden_support:{item}")
    for term in case["required_answer_terms"]:
        if term.casefold() not in answer:
            errors.append(f"missing_fact:{term}")
    for term in case["forbidden_answer_terms"]:
        if term.casefold() in answer:
            errors.append(f"forbidden_fact:{term}")
    if case["expected_source_roles"] and not set(case["expected_source_roles"]) <= roles:
        errors.append("source_role_mismatch")
    if case["expected_temporal_scopes"] and not set(case["expected_temporal_scopes"]) <= temporal:
        errors.append("temporal_scope_mismatch")
    if len(public_targets) < case["minimum_public_targets"]:
        errors.append("public_target_missing")
    if case["expected_claim_status"] and case["expected_claim_status"] not in claim_statuses:
        errors.append("claim_status_mismatch")
    return {
        "case_id": case["case_id"],
        "category": case["category"],
        "behavior": case["behavior"],
        "passed": not errors,
        "errors": errors,
        "actual_status": response.get("status"),
        "actual_route": response.get("route"),
        "support_ids": sorted(support_ids),
        "fact_hits": sum(term.casefold() in answer for term in case["required_answer_terms"]),
        "fact_total": len(case["required_answer_terms"]),
        "required_support_hits": sum(item in support_ids for item in case["required_support_ids"]),
        "required_support_total": len(case["required_support_ids"]),
        "forbidden_support_hits": sum(item in support_ids for item in case["forbidden_support_ids"]),
        "source_role_ok": not case["expected_source_roles"] or set(case["expected_source_roles"]) <= roles,
        "temporal_ok": not case["expected_temporal_scopes"] or set(case["expected_temporal_scopes"]) <= temporal,
        "public_target_ok": len(public_targets) >= case["minimum_public_targets"],
        "claim_ok": not case["expected_claim_status"] or case["expected_claim_status"] in claim_statuses,
    }


def _metrics(cases: list[dict[str, Any]], results: list[dict[str, Any]]) -> dict[str, Any]:
    fact_total = sum(row["fact_total"] for row in results)
    support_total = sum(row["required_support_total"] for row in results)
    claim_cases = sum(bool(row["expected_claim_status"]) for row in cases)
    denominators = {
        "case_pass_rate": len(cases),
        "behavior_accuracy": len(cases),
        "factual_unit_recall": fact_total,
        "required_support_recall": support_total,
        "forbidden_support_false_positive_rate": sum(len(row["forbidden_support_ids"]) for row in cases),
        "source_role_accuracy": len(cases),
        "temporal_accuracy": len(cases),
        "public_target_accuracy": len(cases),
        "claim_verdict_accuracy": claim_cases,
    }
    return {
        "case_pass_rate": _ratio(sum(row["passed"] for row in results), denominators["case_pass_rate"]),
        "behavior_accuracy": _ratio(sum(row["actual_status"] == case["expected_status"] for case, row in zip(cases, results, strict=True)), denominators["behavior_accuracy"]),
        "factual_unit_recall": _ratio(sum(row["fact_hits"] for row in results), fact_total),
        "required_support_recall": _ratio(sum(row["required_support_hits"] for row in results), support_total),
        "forbidden_support_false_positive_rate": _ratio(sum(row["forbidden_support_hits"] for row in results), denominators["forbidden_support_false_positive_rate"]),
        "source_role_accuracy": _ratio(sum(row["source_role_ok"] for row in results), len(cases)),
        "temporal_accuracy": _ratio(sum(row["temporal_ok"] for row in results), len(cases)),
        "public_target_accuracy": _ratio(sum(row["public_target_ok"] for row in results), len(cases)),
        "claim_verdict_accuracy": _ratio(sum(row["claim_ok"] for row in results if row["case_id"] in {case["case_id"] for case in cases if case["expected_claim_status"]}), claim_cases),
        "denominators": denominators,
    }


def _ids(rows: tuple[dict, ...]) -> tuple[str, ...]:
    return tuple(str(row.get("evidence_id") or row.get("source_conflict_id") or row.get("relation_id")) for row in rows if row.get("evidence_id") or row.get("source_conflict_id") or row.get("relation_id"))


def _expect(errors: list[str], field: str, expected: object, actual: object) -> None:
    if expected != actual:
        errors.append(f"{field}:expected={expected!r}:actual={actual!r}")


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _behavior(status: object) -> str:
    if status == "insufficient_evidence":
        return "abstain"
    return "answer"


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _runtime(repo_root: Path):
    source = str(repo_root / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    from tjipto.runtime.api import handle_request
    from tjipto.runtime.service import LegalRuntimeService

    return LegalRuntimeService, handle_request


if __name__ == "__main__":
    sys.exit(main())
