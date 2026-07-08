from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from tjipto.runtime.api import handle_request


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "tests/fixtures/uud/retrieval_eval_cases.jsonl"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate UUD retrieval contract cases.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--strict-known-gaps", action="store_true")
    args = parser.parse_args(argv)

    cases = _read_jsonl(args.cases)
    results = [_evaluate(row) for row in cases]
    counts = {
        "pass": sum(row["outcome"] == "PASS" for row in results),
        "fail": sum(row["outcome"] == "FAIL" for row in results),
        "known_gap": sum(row["outcome"] == "KNOWN_GAP" for row in results),
    }
    report = {
        "status": "fail" if counts["fail"] or (args.strict_known_gaps and counts["known_gap"]) else "pass",
        "counts": counts,
        "results": results,
    }
    if args.report:
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"retrieval_eval: PASS={counts['pass']} FAIL={counts['fail']} KNOWN_GAP={counts['known_gap']}")
    for row in results:
        if row["outcome"] != "PASS":
            print(f"{row['outcome']}: {row['id']} :: {'; '.join(row['errors'])}")
    return 1 if report["status"] == "fail" else 0


def _evaluate(case: dict[str, Any]) -> dict[str, Any]:
    response = handle_request(case["corpus_id"], "ask", {"query": case["query"]}, ROOT)
    errors = _validate(case, response)
    if not errors:
        outcome = "PASS"
    elif case.get("case_status") == "known_gap":
        outcome = "KNOWN_GAP"
    else:
        outcome = "FAIL"
    return {
        "id": case["id"],
        "query": case["query"],
        "case_status": case.get("case_status", "accepted"),
        "outcome": outcome,
        "errors": errors,
        "actual": {
            "status": response.get("status"),
            "route": response.get("route"),
            "intent": response.get("intent"),
            "support_type": _support_type(response),
            "citation_evidence_ids": _ids(response.get("citations", ()), "evidence_id"),
            "citation_legal_unit_ids": _ids(response.get("citations", ()), "legal_unit_id"),
        },
    }


def _validate(case: dict[str, Any], response: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    _expect_equal(errors, "status", case.get("expected_status"), response.get("status"))
    _expect_equal(errors, "route", case.get("expected_route"), response.get("route"))
    _expect_equal(errors, "intent", case.get("expected_intent"), response.get("intent"))
    _expect_equal(errors, "support_type", case.get("expected_support_type"), _support_type(response))
    citations = tuple(response.get("citations", ()))
    metadata = tuple(response.get("metadata_support", ()))
    trace = tuple(response.get("trace_support", ()))
    documents = tuple(response.get("document_relations", ()))
    citation_evidence_ids = set(_ids(citations, "evidence_id"))
    citation_legal_unit_ids = set(_ids(citations, "legal_unit_id"))
    support_evidence_ids = citation_evidence_ids | set(_ids(metadata, "evidence_id")) | set(_ids(trace, "evidence_id"))
    for evidence_id in case.get("expected_evidence_ids", ()):
        if evidence_id not in support_evidence_ids:
            errors.append(f"missing_expected_evidence:{evidence_id}")
    for legal_unit_id in case.get("expected_legal_unit_ids", ()):
        if legal_unit_id not in citation_legal_unit_ids:
            errors.append(f"missing_expected_legal_unit:{legal_unit_id}")
    for evidence_id in case.get("forbidden_evidence_ids", ()):
        if evidence_id in support_evidence_ids:
            errors.append(f"forbidden_evidence_returned:{evidence_id}")
    for legal_unit_id in case.get("forbidden_legal_unit_ids", ()):
        if legal_unit_id in citation_legal_unit_ids:
            errors.append(f"forbidden_legal_unit_returned:{legal_unit_id}")
    if metadata and (citations or response.get("viewer_refs")):
        metadata_citations = [
            row
            for row in citations
            if row.get("authority_kind") == "metadata_source" and row.get("citation_final") is False
        ]
        if len(metadata_citations) != len(citations) or (
            response.get("viewer_refs") and len(tuple(response.get("viewer_refs", ()))) != len(metadata_citations)
        ):
            errors.append("metadata_support_exposed_as_exact_citation")
    if trace and any(row.get("citation_available") or row.get("viewer_highlightable") for row in trace):
        errors.append("trace_support_claims_citation_or_highlight")
    if documents and (citations or response.get("viewer_refs") or any(row.get("highlightable") for row in documents)):
        errors.append("document_relation_exposed_as_exact_citation")
    if case.get("expected_support_type") == "insufficient_evidence" and (
        citations or response.get("viewer_refs") or metadata or trace or documents
    ):
        errors.append("insufficient_evidence_has_public_support")
    return errors


def _support_type(response: dict[str, Any]) -> str:
    if response.get("metadata_support"):
        return "metadata_support"
    if response.get("citations"):
        return "citation"
    if response.get("trace_support"):
        return "trace_support"
    if response.get("document_relations"):
        return "document_relation"
    if response.get("status") in {"insufficient_evidence", "citation_not_found", "no_results"}:
        return "insufficient_evidence"
    return "none"


def _expect_equal(errors: list[str], field: str, expected: Any, actual: Any) -> None:
    if expected is not None and expected != actual:
        errors.append(f"{field}:expected={expected!r}:actual={actual!r}")


def _ids(rows: Any, key: str) -> tuple[str, ...]:
    return tuple(str(row[key]) for row in rows if isinstance(row, dict) and row.get(key))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    sys.exit(main())
