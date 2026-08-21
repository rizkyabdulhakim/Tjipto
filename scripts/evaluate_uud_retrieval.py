from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

from tjipto.runtime.service import LegalRuntimeService


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "tests/fixtures/uud/retrieval_cases.jsonl"
EVALUATION_CUTOFF = 10


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
    acceptance_counters = _acceptance_counters(results) | _artifact_acceptance_counters()
    report: dict[str, Any] = {
        "status": "fail"
        if counts["fail"] or (args.strict_known_gaps and counts["known_gap"]) or any(acceptance_counters.values())
        else "pass",
        "counts": counts,
        "acceptance_counters": acceptance_counters,
        "results": results,
        "metrics": _retrieval_metrics(cases, results),
        "evaluation_identity": {
            "case_set_sha256": _sha256(args.cases),
            "evaluator_sha256": _sha256(Path(__file__)),
            "case_count": len(cases),
            "behavior_counts": {
                behavior: sum(row["expected_behavior"] == behavior for row in cases)
                for behavior in ("retrieve", "abstain")
            },
            "cutoff": EVALUATION_CUTOFF,
        },
    }
    if args.scope_performance:
        performance = _scope_performance()
        report["scope_performance"] = performance
        if performance.get("status") != "pass":
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


def _acceptance_counters(results: list[dict[str, Any]]) -> dict[str, int]:
    """Aggregate only invariants demonstrated by the versioned case catalog."""
    counters = {
        "unsupported_answer_ready": 0,
        "claim_without_exact_grounded_segment": 0,
        "false_support_or_contradiction": 0,
        "source_precedence_or_navigation_error": 0,
        "capability_payload_inconsistency": 0,
    }
    for row in results:
        actual = row["actual"]
        expected_support = row["expected_claim_support"]
        if actual["status"] == "answer_ready" and expected_support == ["insufficient"]:
            counters["unsupported_answer_ready"] += 1
        if any(error.startswith(("claim_support:", "predicate:", "polarity:", "modality:")) for error in row["errors"]):
            counters["false_support_or_contradiction"] += 1
        if row["risk_family"] in {"temporal_arbitration", "explicit_navigation"} and row["errors"]:
            counters["source_precedence_or_navigation_error"] += 1
        if row["expected_needed_corpora"] is not None and row["expected_needed_corpora"] != actual["needed_corpora"]:
            counters["capability_payload_inconsistency"] += 1
        for claim in row["claim_records"]:
            if claim.get("status") == "supported" and not claim.get("support_segments"):
                counters["claim_without_exact_grounded_segment"] += 1
    return counters


def _artifact_acceptance_counters() -> dict[str, int]:
    report = json.loads((ROOT / "data/final/uud/validation_report.json").read_text(encoding="utf-8"))
    source = report.get("source_object_disposition_health", {})
    graph = report.get("legal_graph_authority_health", {})
    generic_files = (
        ROOT / "src/tjipto/corpora/parser_dispatch.py",
        ROOT / "src/tjipto/runtime/query_semantics.py",
        ROOT / "src/tjipto/runtime/service.py",
        ROOT / "src/tjipto/retrieval/router.py",
    )
    generic_text = "\n".join(path.read_text(encoding="utf-8") for path in generic_files)
    duplicate_owner_count = sum(
        generic_text.count(symbol)
        for symbol in ("classify_legal_intent", "_missing_corpus_requirements")
    )
    return {
        "graph_authority_incomplete": int(
            graph.get("status") != "complete"
            or graph.get("authority_without_evidence_count", 0) != 0
            or graph.get("authority_without_bbox_count", 0) != 0
            or graph.get("trace_promoted_count", 0) != 0
        ),
        "unclassified_or_silent_source_objects": int(
            source.get("status") != "complete"
            or source.get("source_object_count") != source.get("terminal_disposition_count")
            or any(source.get(key, 0) != 0 for key in ("duplicate_source_object_id_count", "missing_source_object_id_count", "invalid_disposition_count"))
        ),
        "generic_uud_fallback": int("tjipto.corpora.uud" in generic_text),
        "duplicate_policy_owner": duplicate_owner_count,
    }


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
        roles = {
            row.get("source_role")
            for field in ("citations", "historical_citations", "trace_support")
            for row in response.get(field, ())
        }
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
    response = service.ask(case["corpus_id"], case["query"])
    internal = response
    errors = _validate(case, response, response)
    if not errors:
        outcome = "PASS"
    elif case.get("known_gap") is True:
        outcome = "KNOWN_GAP"
    else:
        outcome = "FAIL"
    return {
        "id": case["id"],
        "query": case["query"],
        "outcome": outcome,
        "errors": errors,
        "expected_claim_support": case.get("expected_claim_support"),
        "expected_needed_corpora": case.get("expected_needed_corpora"),
        "risk_family": case.get("risk_family"),
        "claim_records": [dict(row) for row in response.get("claim_support", ()) if isinstance(row, dict)],
        "actual": {
            "status": response.get("status"),
            "route": response.get("route"),
            "intent": response.get("intent"),
            "requested_function": response.get("requested_function"),
            "support_type": _support_type(internal),
            "citation_evidence_ids": _ids(internal.get("citations", ()), "evidence_id"),
            "citation_legal_unit_ids": _ids(internal.get("citations", ()), "legal_unit_id"),
            "final_citation_count": _final_citation_count(internal),
            "claims": _claim_texts(internal),
            "claim_support": _claim_statuses(internal),
            "predicate": _claim_attributes(internal, "predicate"),
            "polarity": _claim_attributes(internal, "polarity"),
            "modality": _claim_attributes(internal, "modality"),
            "reason_code": _reason_code(internal),
            "source_role": _source_attribute(internal, "source_role"),
            "temporal_context": _source_attribute(internal, "temporal_context"),
            "needed_corpora": list(response.get("needed_corpora", ())),
            "support_ids": list(_support_ids(internal)),
            "text_span_ids": list(_support_span_ids(internal)),
            "public_targets": list(_public_targets(internal)),
            "source_roles": list(dict.fromkeys(str(row["source_role"]) for row in _support_rows(internal) if row.get("source_role"))),
            "temporal_scopes": list(
                dict.fromkeys(
                    str(row.get("temporal_context") or row.get("source_role"))
                    for row in _support_rows(internal)
                    if row.get("temporal_context") or row.get("source_role")
                )
            ),
        },
    }


def _retrieval_metrics(cases: list[dict[str, Any]], results: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {row["id"]: row for row in results}
    retrieve = [row for row in cases if row["expected_behavior"] == "retrieve"]
    retrieved_relevant = retrieved_total = relevant_total = reciprocal_rank = dcg = ideal_dcg = 0.0
    ranked_case_count = 0
    expected_spans: set[tuple[str, str]] = set()
    actual_spans: set[tuple[str, str]] = set()
    role_hits = temporal_hits = target_hits = 0
    for case in retrieve:
        actual = by_id[case["id"]]["actual"]
        relevant = set(case["gold_support_ids"]) | set(case["alternative_support_ids"])
        ranked = actual["support_ids"][:EVALUATION_CUTOFF]
        hits = [support_id for support_id in ranked if support_id in relevant]
        retrieved_relevant += len(hits)
        retrieved_total += len(ranked)
        relevant_total += len(relevant)
        ranked_case_count += bool(relevant)
        if hits:
            rank = next(index for index, support_id in enumerate(ranked, 1) if support_id in relevant)
            reciprocal_rank += 1 / rank
        for index, support_id in enumerate(ranked, 1):
            if support_id in relevant:
                dcg += 1 / math.log2(index + 1)
        ideal_dcg += sum(1 / math.log2(index + 1) for index in range(1, min(len(relevant), EVALUATION_CUTOFF) + 1))
        expected_spans.update((case["id"], span_id) for span_id in case["minimal_span_ids"])
        actual_spans.update((case["id"], span_id) for span_id in actual["text_span_ids"])
        role_hits += not case["expected_source_roles"] or bool(set(case["expected_source_roles"]) & set(actual["source_roles"]))
        temporal_hits += not case["expected_temporal_scopes"] or bool(set(case["expected_temporal_scopes"]) & set(actual["temporal_scopes"]))
        expected_targets = {(row["source_document_id"], tuple(row["page_numbers"])) for row in case["expected_public_targets"]}
        actual_targets = {(row["source_document_id"], tuple(row["page_numbers"])) for row in actual["public_targets"]}
        target_hits += not expected_targets or bool(expected_targets & actual_targets)
    negatives = [row for row in cases if row["expected_behavior"] == "abstain"]
    predicted_abstain = {row["id"] for row in results if row["actual"]["status"] in {"insufficient_evidence", "citation_not_found", "no_results"}}
    expected_abstain = {row["id"] for row in negatives}
    span_hits = expected_spans & actual_spans
    denominators = {
        "recall": relevant_total,
        "precision": retrieved_total,
        "mrr": ranked_case_count,
        "ndcg": ideal_dcg,
        "minimal_span_recall": len(expected_spans),
        "minimal_span_precision": len(actual_spans),
        "over_highlight_ratio": len(actual_spans),
        "wrong_page_rate": len(retrieve),
        "source_role_accuracy": len(retrieve),
        "temporal_accuracy": len(retrieve),
        "citation_target_accuracy": len(retrieve),
        "hard_negative_false_positive_rate": len(negatives),
        "abstention_precision": len(predicted_abstain),
        "abstention_recall": len(expected_abstain),
    }
    return {
        "recall": _ratio(retrieved_relevant, relevant_total),
        "precision": _ratio(retrieved_relevant, retrieved_total),
        "mrr": _ratio(reciprocal_rank, ranked_case_count),
        "ndcg": _ratio(dcg, ideal_dcg),
        "minimal_span_recall": _ratio(len(span_hits), len(expected_spans)),
        "minimal_span_precision": _ratio(len(span_hits), len(actual_spans)),
        "over_highlight_ratio": _ratio(len(actual_spans - expected_spans), len(actual_spans)),
        "wrong_page_rate": _ratio(len(retrieve) - target_hits, len(retrieve)),
        "source_role_accuracy": _ratio(role_hits, len(retrieve)),
        "temporal_accuracy": _ratio(temporal_hits, len(retrieve)),
        "citation_target_accuracy": _ratio(target_hits, len(retrieve)),
        "hard_negative_false_positive_rate": _ratio(sum(bool(by_id[row["id"]]["actual"]["support_ids"]) for row in negatives), len(negatives)),
        "abstention_precision": _ratio(len(predicted_abstain & expected_abstain), len(predicted_abstain)),
        "abstention_recall": _ratio(len(predicted_abstain & expected_abstain), len(expected_abstain)),
        "denominators": denominators,
    }


def _ratio(numerator: float, denominator: float) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate(case: dict[str, Any], response: dict[str, Any], internal: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    _expect_equal(errors, "status", case.get("expected_status"), response.get("status"))
    _expect_equal(errors, "route", case.get("expected_route"), response.get("route"))
    _expect_equal(errors, "intent", case.get("expected_intent"), response.get("intent"))
    _expect_equal(errors, "requested_function", case.get("expected_requested_function"), response.get("requested_function"))
    _expect_equal(errors, "claims", case.get("expected_claims"), _claim_texts(internal))
    _expect_equal(errors, "claim_support", case.get("expected_claim_support"), _claim_statuses(internal))
    _expect_equal(errors, "predicate", case.get("expected_predicate"), _claim_attributes(internal, "predicate"))
    _expect_equal(errors, "polarity", case.get("expected_polarity"), _claim_attributes(internal, "polarity"))
    _expect_equal(errors, "modality", case.get("expected_modality"), _claim_attributes(internal, "modality"))
    _expect_equal(errors, "reason_code", case.get("expected_reason_code"), _reason_code(internal))
    _expect_equal(errors, "source_role", case.get("expected_source_role"), _source_attribute(internal, "source_role"))
    _expect_equal(errors, "temporal_context", case.get("expected_temporal_context"), _source_attribute(internal, "temporal_context"))
    _expect_equal(errors, "needed_corpora", case.get("expected_needed_corpora"), list(response.get("needed_corpora", ())))
    _expect_equal(errors, "support_type", case.get("expected_support_type"), _support_type(internal))
    citations = tuple(internal.get("citations", ()))
    historical = tuple(internal.get("historical_citations", ()))
    metadata = tuple(internal.get("metadata_support", ()))
    trace = tuple(internal.get("trace_support", ()))
    documents = tuple(internal.get("document_relations", ()))
    citation_evidence_ids = set(_ids(citations, "evidence_id")) | set(_ids(historical, "evidence_id"))
    citation_legal_unit_ids = set(_ids(citations, "legal_unit_id")) | set(_ids(historical, "legal_unit_id"))
    support_evidence_ids = citation_evidence_ids | set(_ids(metadata, "evidence_id")) | set(_ids(trace, "evidence_id"))
    support_evidence_ids |= set(_claim_support_ids(internal))
    for evidence_id in case.get("expected_evidence_ids", ()):
        if evidence_id not in support_evidence_ids:
            errors.append(f"missing_expected_evidence:{evidence_id}")
    for legal_unit_id in case.get("expected_legal_unit_ids", ()):
        if legal_unit_id not in citation_legal_unit_ids:
            errors.append(f"missing_expected_legal_unit:{legal_unit_id}")
    for evidence_id in case.get("forbidden_evidence_ids", ()):
        if evidence_id in support_evidence_ids:
            errors.append(f"forbidden_evidence_returned:{evidence_id}")
    for evidence_id in case.get("forbidden_support_ids", ()):
        if evidence_id in support_evidence_ids:
            errors.append(f"forbidden_support_returned:{evidence_id}")
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


def _claim_texts(response: dict[str, Any]) -> list[str]:
    return [str(row.get("claim_text")) for row in response.get("claim_support", ()) if isinstance(row, dict)]


def _claim_statuses(response: dict[str, Any]) -> list[str]:
    return [str(row.get("status")) for row in response.get("claim_support", ()) if isinstance(row, dict)]


def _claim_attributes(response: dict[str, Any], field: str) -> list[str]:
    return [str(row.get(field)) for row in response.get("claim_support", ()) if isinstance(row, dict) and row.get(field)]


def _claim_support_ids(response: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        str(evidence_id)
        for row in response.get("claim_support", ())
        if isinstance(row, dict)
        for evidence_id in row.get("support_evidence_ids", ())
    )


def _support_rows(response: dict[str, Any]) -> tuple[dict, ...]:
    return tuple(
        row
        for field in ("citations", "historical_citations", "metadata_support", "trace_support")
        for row in response.get(field, ())
        if isinstance(row, dict)
    )


def _support_ids(response: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            str(row.get("evidence_id") or row.get("metadata_grounding_id") or row.get("source_conflict_id"))
            for row in _support_rows(response)
            if row.get("evidence_id") or row.get("metadata_grounding_id") or row.get("source_conflict_id")
        )
    )


def _support_span_ids(response: dict[str, Any]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(span_id) for row in _support_rows(response) for span_id in row.get("text_span_ids", ())))


def _public_targets(response: dict[str, Any]) -> tuple[dict, ...]:
    return tuple(
        {
            "source_document_id": row.get("source_document_id"),
            "page_numbers": list(row.get("page_numbers") or (() if row.get("page_number") is None else (row["page_number"],))),
        }
        for row in _support_rows(response)
        if row.get("source_document_id")
    )


def _reason_code(response: dict[str, Any]) -> str | None:
    if response.get("reason_code"):
        return str(response["reason_code"])
    return next(
        (str(row["reason_code"]) for row in response.get("claim_support", ()) if isinstance(row, dict) and row.get("reason_code")),
        None,
    )


def _source_attribute(response: dict[str, Any], field: str) -> str | None:
    for rows in (
        response.get("citations", ()),
        response.get("historical_citations", ()),
        response.get("metadata_support", ()),
        response.get("trace_support", ()),
        response.get("claim_support", ()),
    ):
        for row in rows:
            if isinstance(row, dict) and row.get(field):
                return str(row[field])
    return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    sys.exit(main())
