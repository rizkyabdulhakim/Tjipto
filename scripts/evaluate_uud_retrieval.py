from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

from tjipto.runtime.api import handle_request
from tjipto.runtime.service import LegalRuntimeService


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "tests/fixtures/uud/retrieval_eval_cases.jsonl"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate UUD retrieval contract cases.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--strict-known-gaps", action="store_true")
    parser.add_argument("--scope-performance", action="store_true")
    args = parser.parse_args(argv)

    cases = _read_jsonl(args.cases)
    service = LegalRuntimeService(ROOT)
    results = [_evaluate(row, service) for row in cases]
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
    if args.scope_performance:
        report["scope_performance"] = _scope_performance()
        if report["scope_performance"].get("status") != "pass":
            report["status"] = "fail"
    if args.report:
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"retrieval_eval: PASS={counts['pass']} FAIL={counts['fail']} KNOWN_GAP={counts['known_gap']}")
    if args.scope_performance:
        performance = report["scope_performance"]
        print(f"scope_performance: {json.dumps(performance, ensure_ascii=False, sort_keys=True)}")
    for row in results:
        if row["outcome"] != "PASS":
            print(f"{row['outcome']}: {row['id']} :: {'; '.join(row['errors'])}")
    return 1 if report["status"] == "fail" else 0


def _scope_performance() -> dict[str, Any]:
    service = LegalRuntimeService(ROOT)
    store = service._store("uud")
    if store is None:
        return {"status": "unavailable", "reason": "corpus_not_ready"}
    config = store.config
    labels = config.setting("intent_config", {}).get("source_role_labels", {})
    markers = {
        role: ("naskah asli" if role == "original_historical" else "saat ini" if role == "current_consolidated" else f"Perubahan {label}")
        for role, label in labels.items()
    }
    cases = []
    for role in config.source_roles:
        evidence = next(
            (
                row
                for row in store.evidence
                if row.get("source_role") == role
                and row.get("status") == "final"
                and row.get("citation")
                and row.get("viewer_highlightable") is True
            ),
            None,
        )
        if evidence is not None and role in markers:
            cases.append((f"Apa isi {evidence['citation']} {markers[role]}?", role))
    if not cases:
        return {"status": "unavailable", "reason": "no_scoped_exact_cases"}
    cold_start = time.perf_counter()
    cold = [service.ask("uud", query) for query, _ in cases]
    cold_seconds = time.perf_counter() - cold_start
    warm_samples = []
    warm_responses = []
    for _ in range(4):
        for query, _ in cases:
            started = time.perf_counter()
            warm_responses.append(service.ask("uud", query))
            warm_samples.append((time.perf_counter() - started) * 1000)
    expected_roles = [role for _, role in cases]
    responses = cold + warm_responses
    correct = 0
    leakage = 0
    for response, expected in zip(responses[: len(cases)], expected_roles):
        roles = {row.get("source_role") for row in response.get("citations", ())}
        correct += roles == {expected}
        leakage += bool(roles - {expected})
    warm_samples.sort()
    return {
        "status": "pass" if correct == len(cases) and leakage == 0 else "fail",
        "case_count": len(cases),
        "cold_seconds": round(cold_seconds, 3),
        "warm_p50_ms": round(_percentile(warm_samples, 0.50), 2),
        "warm_p95_ms": round(_percentile(warm_samples, 0.95), 2),
        "source_role_precision": round(correct / len(cases), 4),
        "cross_source_leakage": leakage,
    }


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    index = min(len(values) - 1, max(0, round((len(values) - 1) * fraction)))
    return values[index]


def _evaluate(case: dict[str, Any], service: LegalRuntimeService) -> dict[str, Any]:
    response = handle_request(case["corpus_id"], "ask", {"query": case["query"]}, ROOT, service=service)
    internal = service.ask(case["corpus_id"], case["query"])
    errors = _validate(case, response, internal)
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
            "requested_function": response.get("requested_function"),
            "support_type": _support_type(internal),
            "citation_evidence_ids": _ids(internal.get("citations", ()), "evidence_id"),
            "citation_legal_unit_ids": _ids(internal.get("citations", ()), "legal_unit_id"),
            "final_citation_count": _final_citation_count(internal),
        },
    }


def _validate(case: dict[str, Any], response: dict[str, Any], internal: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    _expect_equal(errors, "status", case.get("expected_status"), response.get("status"))
    _expect_equal(errors, "route", case.get("expected_route"), response.get("route"))
    _expect_equal(errors, "intent", case.get("expected_intent"), response.get("intent"))
    _expect_equal(errors, "requested_function", case.get("expected_requested_function"), response.get("requested_function"))
    _expect_equal(errors, "support_type", case.get("expected_support_type"), _support_type(internal))
    citations = tuple(internal.get("citations", ()))
    historical = tuple(internal.get("historical_citations", ()))
    metadata = tuple(internal.get("metadata_support", ()))
    trace = tuple(internal.get("trace_support", ()))
    documents = tuple(internal.get("document_relations", ()))
    citation_evidence_ids = set(_ids(citations, "evidence_id")) | set(_ids(historical, "evidence_id"))
    citation_legal_unit_ids = set(_ids(citations, "legal_unit_id")) | set(_ids(historical, "legal_unit_id"))
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
            row for row in citations if row.get("authority_kind") == "metadata_source" and row.get("citation_final") is False
        ]
        if len(metadata_citations) != len(citations) or (
            response.get("viewer_refs") and len(tuple(response.get("viewer_refs", ()))) != len(metadata_citations)
        ):
            errors.append("metadata_support_exposed_as_exact_citation")
    if trace and any(row.get("citation_final") is True for row in trace):
        errors.append("trace_support_claims_citation_or_highlight")
    if documents and (citations or response.get("viewer_refs") or any(row.get("highlightable") for row in documents)):
        errors.append("document_relation_exposed_as_exact_citation")
    if case.get("expected_support_type") == "insufficient_evidence" and (
        citations or response.get("viewer_refs") or metadata or trace or documents
    ):
        errors.append("insufficient_evidence_has_public_support")
    if "expected_final_citation_count" in case:
        _expect_equal(errors, "final_citation_count", case["expected_final_citation_count"], _final_citation_count(internal))
    return errors


def _support_type(response: dict[str, Any]) -> str:
    if response.get("metadata_support"):
        return "metadata_support"
    if response.get("citations"):
        return "citation"
    if response.get("historical_citations"):
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


def _final_citation_count(response: dict[str, Any]) -> int:
    return sum(1 for row in response.get("citations", ()) if isinstance(row, dict) and row.get("citation_final") is True)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    sys.exit(main())
