"""Freeze and execute the P0-0 evaluation contract without changing runtime behavior."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from hashlib import sha256
import json
from pathlib import Path
from pathlib import PurePosixPath
import tempfile
import subprocess
import sys
import zipfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURES = ROOT / "tests/fixtures/uud/p0_pre_fix_fixtures.jsonl"
DEFAULT_IDENTITY = ROOT / "tests/fixtures/uud/p0_evaluation_identity.json"
EXPECTED_COMMIT = "0097802d8330f2ff9c578f408e40fff291e411a9"
EXPECTED_TREE = "6b0692c7c247745765a180731c400574efacf25b"
CLASSIFICATIONS = {"PASS", "PARTIAL", "FAIL", "EXPECTED_CORPUS_GAP"}
HISTORICAL_CATALOG_SCHEMA = {
    "source_columns": ["Query", "Decision", "Status / route", "Supports", "Viewer"],
    "supports_field": "supports_count",
    "supports_semantics": "broader_public_support_count",
    "viewer_field": "viewer_count",
    "viewer_semantics": "viewer_resolvable_support_count",
}
HISTORICAL_CATALOG_ROW_FIELDS = {
    "index",
    "query",
    "classification",
    "status",
    "route",
    "supports_count",
    "viewer_count",
}
MANDATORY_FIXTURE_FAMILIES = {
    "conversation/greetings",
    "follow-up/thread references",
    "exact legal reference",
    "canonical spelling/number variants",
    "metadata/corpus counts",
    "structure/navigation",
    "source/version identity",
    "amendment relation",
    "relation analysis",
    "comparison",
    "legal research",
    "legal opinion",
    "memo",
    "IRAC",
    "chronology",
    "multi-objective",
    "clarification",
    "hypothetical/counterfactual",
    "DeliverySpec",
    "footnote preference",
    "source anomaly",
    "proposition coverage",
    "valid paraphrase",
    "unsupported claim",
    "support minimality",
    "BBox/browser interaction",
}
DEFAULT_PLANNER_PROPOSAL = {
    "variants": [],
    "information_needs": [],
    "status": "ready",
    "missing_dimensions": [],
    "clarification_question": None,
}


class ContractError(ValueError):
    """Raised when the frozen contract cannot be evaluated safely."""


class ProbePlanner:
    """Deterministic provider probe; records counts and safe metadata only."""

    def __init__(self, proposal: dict[str, Any] | None = None) -> None:
        self.proposal = proposal or DEFAULT_PLANNER_PROPOSAL
        self.calls = 0

    def propose(self, _request: dict[str, Any]) -> dict[str, Any]:
        self.calls += 1
        return self.proposal


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--identity", type=Path, default=DEFAULT_IDENTITY)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--archive-sidecar", type=Path)
    parser.add_argument(
        "--trusted-archive-sha256",
        help="Out-of-band trusted SHA-256 for the archive; sidecar identity claims are not trust anchors.",
    )
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.archive:
        with _archive_workspace(
            args.archive.resolve(), args.archive_sidecar, trusted_archive_sha256=args.trusted_archive_sha256
        ) as context:
            return _run_contract(
                context["repo_root"],
                context["fixtures_path"],
                context["identity_path"],
                args.report,
                archive_identity=context["archive_identity"],
            )
    return _run_contract(args.repo_root.resolve(), args.fixtures.resolve(), args.identity.resolve(), args.report)


def _run_contract(
    repo_root: Path,
    fixtures_path: Path,
    identity_path: Path,
    report_path: Path,
    *,
    archive_identity: dict[str, Any] | None = None,
) -> int:
    fixtures = _read_jsonl(fixtures_path)
    frozen = _read_json(identity_path)
    if archive_identity is None:
        identity, identity_errors = _bind_identity(repo_root, fixtures_path, identity_path, frozen)
    else:
        identity, identity_errors = _bind_archive_identity(
            repo_root, fixtures_path, identity_path, frozen, archive_identity
        )
    historical_errors = _verify_historical_baselines(repo_root, frozen)
    recorded_evidence_errors = _verify_recorded_evidence(repo_root, fixtures)
    missing_families = sorted(_missing_mandatory_families(fixtures))
    identity_errors.extend(historical_errors)
    identity_errors.extend(recorded_evidence_errors)
    identity_errors.extend(f"missing_mandatory_fixture_family:{family}" for family in missing_families)
    results = []
    if not identity_errors:
        results = _evaluate_cases(repo_root, fixtures)
        _apply_group_checks(results, fixtures)
        _refresh_classifications(results, fixtures)

    pre_fix_mismatches = [row["fixture_id"] for row in results if not row["pre_fix_match"]]
    target_passes = [row["fixture_id"] for row in results if row["target_match"]]
    deterministic_failed = sum(bool(row["deterministic_failures"]) for row in results)
    semantic_failed = sum(bool(row["semantic_failures"]) for row in results)
    report = {
        "contract_version": frozen["contract_version"],
        "status": "pre_fix_contract_frozen" if not identity_errors and not pre_fix_mismatches else "invalid",
        "identity_match": not identity_errors,
        "identity_errors": identity_errors,
        "identity_mode": identity["identity_mode"],
        "frozen_identity": frozen["frozen_identity"],
        "runtime_identity": identity,
        "historical_baselines": frozen["historical_baselines"],
        "historical_baseline_verification": {
            "status": "passed" if not historical_errors else "failed",
            "errors": historical_errors,
        },
        "recorded_evidence_gate": {
            "status": "passed" if not recorded_evidence_errors else "failed",
            "errors": recorded_evidence_errors,
        },
        "mandatory_fixture_families": sorted(MANDATORY_FIXTURE_FAMILIES),
        "missing_mandatory_fixture_families": missing_families,
        "catalog_policy": "historical_baseline_not_gold",
        "classification_policy": {
            "allowed": sorted(CLASSIFICATIONS),
            "answer_ready_is_not_pass": True,
            "pre_fix_match_means_observed_current_behavior_matches_adjudication": True,
        },
        "fixture_counts": {
            "total": len(results),
            "pre_fix_expected_failures": sum(row["pre_fix_expected"] == "FAIL" for row in results),
            "pre_fix_expected_corpus_gaps": sum(row["pre_fix_expected"] == "EXPECTED_CORPUS_GAP" for row in results),
            "observed_failures": sum(row["observed_classification"] == "FAIL" for row in results),
            "observed_expected_corpus_gaps": sum(row["observed_classification"] == "EXPECTED_CORPUS_GAP" for row in results),
            "intentionally_failing_against_target": len(
                [row for row in results if row["observed_classification"] != row["target_classification"]]
            ),
            "target_passes": len(target_passes),
            "pre_fix_mismatches": len(pre_fix_mismatches),
        },
        "deterministic": {
            "fixture_passes": len(results) - deterministic_failed,
            "fixture_failures": deterministic_failed,
            "separate_from_semantic": True,
        },
        "semantic": {
            "fixture_passes": len(results) - semantic_failed,
            "fixture_failures": semantic_failed,
            "separate_from_deterministic": True,
        },
        "results": results,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "p0_contract: "
        f"status={report['status']} fixtures={len(results)} "
        f"deterministic_failures={deterministic_failed} semantic_failures={semantic_failed} "
        f"pre_fix_mismatches={len(pre_fix_mismatches)}"
    )
    return 0 if report["status"] == "pre_fix_contract_frozen" else 2


@contextmanager
def _archive_workspace(archive_path: Path, sidecar_path: Path | None, *, trusted_archive_sha256: str | None):
    sidecar = (sidecar_path or Path(str(archive_path) + ".sidecar.json")).resolve()
    archive_identity = _verify_archive_sidecar(archive_path, sidecar)
    identity_errors: list[str] = []
    if not isinstance(trusted_archive_sha256, str) or len(trusted_archive_sha256) != 64:
        identity_errors.append("trusted_archive_sha256_required")
    elif archive_identity["verified_archive_sha256"] != trusted_archive_sha256.casefold():
        identity_errors.append(
            "archive_identity_mismatch:"
            f"expected={trusted_archive_sha256.casefold()}:actual={archive_identity['verified_archive_sha256']}"
        )
    archive_identity["trusted_archive_sha256"] = trusted_archive_sha256
    archive_identity["identity_errors"] = identity_errors
    with tempfile.TemporaryDirectory(prefix="tjipto-p0-archive-") as directory:
        repo_root = Path(directory)
        _extract_archive_verified(archive_path, repo_root)
        yield {
            "repo_root": repo_root,
            "fixtures_path": repo_root / "tests/fixtures/uud/p0_pre_fix_fixtures.jsonl",
            "identity_path": repo_root / "tests/fixtures/uud/p0_evaluation_identity.json",
            "archive_identity": archive_identity,
        }


def _extract_archive_verified(archive_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        for name in archive.namelist():
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts:
                raise ContractError(f"unsafe archive path: {name}")
        archive.extractall(destination)


def _verify_archive_sidecar(archive_path: Path, sidecar_path: Path) -> dict[str, Any]:
    if not archive_path.is_file() or not sidecar_path.is_file():
        raise ContractError("archive and immutable sidecar are required")
    sidecar = _read_json(sidecar_path)
    actual_archive_sha = _digest(archive_path)
    if actual_archive_sha != sidecar.get("archive_sha256"):
        raise ContractError("archive_sha256_mismatch")
    with zipfile.ZipFile(archive_path) as archive:
        actual = {
            name: sha256(archive.read(name)).hexdigest()
            for name in archive.namelist()
            if not name.endswith("/")
        }
    expected = sidecar.get("source_file_sha256") or sidecar.get("source_file_digests")
    if not isinstance(expected, dict) or actual != expected:
        raise ContractError("archive_file_hashes_mismatch")
    if sidecar.get("archive_file_count") is not None and sidecar["archive_file_count"] != len(actual):
        raise ContractError("archive_file_count_mismatch")
    content_manifest = "".join(f"{name}\t{digest}\n" for name, digest in sorted(actual.items()))
    sidecar["sidecar_sha256"] = _digest(sidecar_path)
    sidecar["verified_archive_sha256"] = actual_archive_sha
    sidecar["verified_archive_content_sha256"] = sha256(content_manifest.encode("utf-8")).hexdigest()
    sidecar["claimed_base_commit_sha"] = sidecar.get("base_commit_sha")
    sidecar["claimed_base_tree_sha"] = sidecar.get("base_tree_sha")
    sidecar["claimed_commit_sha"] = sidecar.get("commit_sha")
    sidecar["claimed_tree_sha"] = sidecar.get("tree_sha")
    return sidecar


def _evaluate_cases(repo_root: Path, fixtures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for case in fixtures:
        _validate_case(case)
        if case.get("probe") == "recorded_evidence":
            results.append(_evaluate_recorded_case(repo_root, case))
        elif case.get("probe") == "sequence":
            results.append(_evaluate_sequence(repo_root, case))
        else:
            results.append(_evaluate_single(repo_root, case))
    return results


def _evaluate_single(repo_root: Path, case: dict[str, Any]) -> dict[str, Any]:
    service_type, request_handler = _runtime(repo_root)
    planner = ProbePlanner(case.get("planner_proposal")) if case.get("planner_enabled") else None
    service = service_type(repo_root, answer_provider=None, planning_provider=planner)
    response = service.ask("uud", str(case["query"]))
    public = None
    public_error = None
    if case.get("public") or case.get("deterministic", {}).get("api_footnote_mode_accepted"):
        try:
            public = request_handler("uud", "ask", {"query": case["query"]}, service=service)
        except Exception as error:  # pragma: no cover - defensive boundary for probe evidence.
            public_error = type(error).__name__
    deterministic_failures = _deterministic_failures(case, response, public, planner, public_error)
    semantic_failures = _semantic_failures(case, response)
    result = _result_row(case, response, public, planner, deterministic_failures, semantic_failures)
    if case.get("deterministic", {}).get("api_footnote_mode_accepted"):
        result["api_footnote_mode"] = _probe_footnote_mode(request_handler, service, case)
        if not result["api_footnote_mode"]["accepted"]:
            result["deterministic_failures"].append("api_footnote_mode_rejected")
            result["observed_classification"] = "FAIL"
    return result


def _evaluate_sequence(repo_root: Path, case: dict[str, Any]) -> dict[str, Any]:
    service_type, _ = _runtime(repo_root)
    planner = ProbePlanner(case.get("planner_proposal")) if case.get("planner_enabled") else None
    service = service_type(repo_root, answer_provider=None, planning_provider=planner)
    turns: list[dict[str, Any]] = []
    deterministic_failures: list[str] = []
    semantic_failures: list[str] = []
    for index, turn in enumerate(case["turns"]):
        response = service.ask("uud", str(turn["query"]))
        d_failures = _deterministic_failures(turn, response, None, planner, None)
        s_failures = _semantic_failures(turn, response)
        if index and turn.get("requires_context_resolution"):
            if response.get("status") in {"no_results", "insufficient_evidence"}:
                d_failures.append("follow_up_context_not_resolved")
            if response.get("route") == "lexical_fallback":
                d_failures.append("follow_up_used_unrelated_lexical_fallback")
        deterministic_failures.extend(f"turn_{index + 1}:{item}" for item in d_failures)
        semantic_failures.extend(f"turn_{index + 1}:{item}" for item in s_failures)
        turns.append(_response_trace(response, None, planner, d_failures, s_failures))
    result = _result_row(case, turns[-1].get("response", {}) if turns else {}, None, planner, deterministic_failures, semantic_failures)
    result["turns"] = turns
    return result


def _verify_recorded_evidence(repo_root: Path, fixtures: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for case in fixtures:
        if case.get("probe") != "recorded_evidence":
            continue
        for error in _recorded_evidence_details(repo_root, case)["errors"]:
            errors.append(f"{case['fixture_id']}:{error}")
    return errors


def _recorded_evidence_details(repo_root: Path, case: dict[str, Any]) -> dict[str, Any]:
    evidence = case.get("recorded_evidence") or {}
    path = repo_root / str(evidence.get("path", ""))
    actual_digest = _digest(path)
    errors: list[str] = []
    missing: list[str] = []
    source_hashes: dict[str, str] = {}
    if not path.is_file():
        return {
            "path": path,
            "actual_digest": actual_digest,
            "errors": ["recorded_evidence_missing_file"],
            "missing": missing,
            "source_hashes": source_hashes,
        }
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return {
            "path": path,
            "actual_digest": actual_digest,
            "errors": ["recorded_evidence_not_machine_readable"],
            "missing": missing,
            "source_hashes": source_hashes,
        }
    expected_digest = evidence.get("sha256")
    if not expected_digest:
        errors.append("recorded_evidence_digest_missing")
    elif actual_digest != expected_digest:
        errors.append("recorded_evidence_digest_mismatch")
    needles = evidence.get("needles")
    if not isinstance(needles, list):
        errors.append("recorded_evidence_needles_invalid")
    else:
        missing = [needle for needle in needles if needle not in text]
        errors.extend(f"recorded_evidence_missing:{needle}" for needle in missing)
    if path.suffix.casefold() == ".json":
        try:
            payload = _read_json(path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            errors.append("recorded_evidence_not_machine_readable")
        else:
            if not isinstance(payload, dict):
                errors.append("recorded_evidence_not_machine_readable")
            else:
                runtime_identity = payload.get("runtime_identity", {})
                if not isinstance(runtime_identity, dict) or runtime_identity.get("commit_sha") != EXPECTED_COMMIT:
                    errors.append("recorded_evidence_runtime_identity_mismatch")
                source_hashes = payload.get("source_hashes", {})
                if not isinstance(source_hashes, dict):
                    errors.append("recorded_evidence_source_hashes_invalid")
                    source_hashes = {}
    for relative, expected in source_hashes.items():
        if _digest(repo_root / relative) != expected:
            errors.append(f"recorded_evidence_source_digest_mismatch:{relative}")
    return {
        "path": path,
        "actual_digest": actual_digest,
        "errors": errors,
        "missing": missing,
        "source_hashes": source_hashes,
    }


def _evaluate_recorded_case(repo_root: Path, case: dict[str, Any]) -> dict[str, Any]:
    evidence = case["recorded_evidence"]
    details = _recorded_evidence_details(repo_root, case)
    missing = details["missing"]
    actual_digest = details["actual_digest"]
    failures = list(details["errors"])
    actual = "FAIL"
    result = {
        "fixture_id": case["fixture_id"],
        "family": case["family"],
        "query": case.get("query"),
        "reason_code": case["adjudication"]["reason_code"],
        "target_classification": case["adjudication"]["target_classification"],
        "pre_fix_expected": case["adjudication"]["pre_fix_expected"],
        "observed_classification": actual,
        "deterministic_failures": failures,
        "semantic_failures": [],
        "pre_fix_match": actual == case["adjudication"]["pre_fix_expected"] and not failures,
        "target_match": actual == case["adjudication"]["target_classification"],
        "execution": "preserved_evidence_not_live",
        "recorded_evidence": {
            "path": evidence["path"],
            "sha256": actual_digest,
            "needles_found": not missing,
            "digest_match": bool(evidence.get("sha256")) and actual_digest == evidence.get("sha256"),
            "integrity_valid": not failures,
            "integrity_errors": failures,
        },
        "planner_trace": {"calls": 0},
        "tool_trace": {},
        "evidence_trace": {},
        "review_trace": {},
    }
    return result


def _result_row(
    case: dict[str, Any],
    response: dict[str, Any],
    public: dict[str, Any] | None,
    planner: ProbePlanner | None,
    deterministic_failures: list[str],
    semantic_failures: list[str],
) -> dict[str, Any]:
    actual = _classify(case, response, deterministic_failures, semantic_failures)
    row = {
        "fixture_id": case["fixture_id"],
        "family": case["family"],
        "query": case.get("query"),
        "reason_code": case["adjudication"]["reason_code"],
        "target_classification": case["adjudication"]["target_classification"],
        "pre_fix_expected": case["adjudication"]["pre_fix_expected"],
        "observed_classification": actual,
        "pre_fix_match": actual == case["adjudication"]["pre_fix_expected"],
        "target_match": actual == case["adjudication"]["target_classification"],
        "actual_status": response.get("status"),
        "actual_route": response.get("route"),
        "actual_operation": response.get("operation"),
        "answer_ready_is_not_pass": response.get("status") == "answer_ready" and actual != "PASS",
        "deterministic_failures": list(deterministic_failures),
        "semantic_failures": list(semantic_failures),
        "planner_trace": _planner_trace(response, planner),
        "tool_trace": {
            "route": response.get("route"),
            "operation": response.get("operation"),
            "source_scopes": tuple(response.get("source_scopes") or ()),
            "research_round": response.get("research_round"),
        },
        "evidence_trace": _evidence_trace(response, public),
        "review_trace": {
            "claim_statuses": tuple(
                str(item.get("status")) for item in response.get("claim_support", ()) if isinstance(item, dict)
            ),
            "review_issue_codes": tuple(str(item) for item in response.get("warnings", ()) if item),
            "insufficient_reasons": tuple(str(item) for item in response.get("insufficient_reasons", ()) if item),
        },
        "answer_features": {
            "length": len(str(response.get("answer") or "")),
            "paragraph_count": _paragraph_count(response.get("answer")),
            "contains_footnote_marker": "[1]" in str((public or {}).get("answer") or ""),
        },
    }
    return row


def _refresh_classifications(results: list[dict[str, Any]], fixtures: list[dict[str, Any]]) -> None:
    by_id = {case["fixture_id"]: case for case in fixtures}
    for row in results:
        case = by_id[row["fixture_id"]]
        if case.get("probe") == "recorded_evidence":
            continue
        row["observed_classification"] = _classify(
            case,
            {"status": row.get("actual_status")},
            row["deterministic_failures"],
            row["semantic_failures"],
        )
        row["pre_fix_match"] = row["observed_classification"] == row["pre_fix_expected"]
        row["target_match"] = row["observed_classification"] == row["target_classification"]


def _classify(case: dict[str, Any], response: dict[str, Any], deterministic: list[str], semantic: list[str]) -> str:
    if deterministic or semantic:
        return "FAIL"
    if response.get("status") == "answer_ready" and not _has_substantive_quality_assertion(case):
        return "FAIL"
    if case.get("expected_corpus_gap"):
        if response.get("status") in {"no_results", "insufficient_evidence"}:
            return "EXPECTED_CORPUS_GAP"
        return "FAIL"
    return "PASS"


def _deterministic_failures(
    case: dict[str, Any],
    response: dict[str, Any],
    public: dict[str, Any] | None,
    planner: ProbePlanner | None,
    public_error: str | None,
) -> list[str]:
    spec = case.get("deterministic", {})
    failures: list[str] = []
    _expect_any(failures, "status", response.get("status"), spec.get("status_in"))
    _expect_any(failures, "route", response.get("route"), spec.get("route_in"))
    _expect_any(failures, "operation", response.get("operation"), spec.get("operation_in"))
    if response.get("status") in set(spec.get("forbidden_statuses", ())):
        failures.append(f"forbidden_status:{response.get('status')}")
    citation_count = len(response.get("citations") or response.get("final_citations") or ())
    _expect_min(failures, "citation_count", citation_count, spec.get("minimum_citations"))
    _expect_max(failures, "citation_count", citation_count, spec.get("maximum_citations"))
    if planner is not None:
        _expect_min(failures, "planner_calls", planner.calls, spec.get("planner_calls_min"))
    elif spec.get("planner_calls_min"):
        failures.append("planner_probe_not_configured")
    if spec.get("clarification_dimensions"):
        actual = set(str(value) for value in response.get("missing_dimensions", ()))
        expected = set(str(value) for value in spec["clarification_dimensions"])
        if not expected <= actual:
            failures.append("clarification_dimensions_missing")
    if spec.get("require_clarification_id") and not response.get("clarification_id"):
        failures.append("clarification_id_missing")
    if spec.get("require_no_citations") and citation_count:
        failures.append("citations_present_on_safe_abstention")
    if spec.get("require_no_public_supports") and len((public or {}).get("supports") or ()):
        failures.append("public_support_present_on_safe_abstention")
    if spec.get("maximum_public_supports") is not None:
        if public is None:
            failures.append("public_projection_not_executed")
        elif len(public.get("supports") or ()) > int(spec["maximum_public_supports"]):
            failures.append("public_support_minimality_exceeded")
    if spec.get("public_answer_must_not_contain"):
        if public is None:
            failures.append("public_projection_not_executed")
        else:
            answer = str(public.get("answer") or "")
            for marker in spec["public_answer_must_not_contain"]:
                if str(marker) in answer:
                    failures.append(f"public_answer_contains_forbidden:{marker}")
    if spec.get("public_minimum_supports") is not None:
        if public is None or len(public.get("supports") or ()) < int(spec["public_minimum_supports"]):
            failures.append("public_support_count_below_minimum")
    if spec.get("api_footnote_mode_accepted") and public_error:
        failures.append(f"public_projection_error:{public_error}")
    if response.get("status") == "answer_ready" and not _has_substantive_quality_assertion(case):
        failures.append("answer_ready_without_contract_checks")
    return failures


def _semantic_failures(case: dict[str, Any], response: dict[str, Any]) -> list[str]:
    spec = case.get("semantic", {})
    failures: list[str] = []
    _expect_any(failures, "semantic_operation", response.get("operation"), spec.get("operation_in"))
    answer = str(response.get("answer") or "")
    for marker in spec.get("required_answer_markers", ()):
        if str(marker) not in answer:
            failures.append(f"missing_answer_marker:{marker}")
    for marker in spec.get("forbidden_answer_markers", ()):
        if str(marker) in answer:
            failures.append(f"forbidden_answer_marker:{marker}")
    if spec.get("paragraph_count") is not None and _paragraph_count(answer) != int(spec["paragraph_count"]):
        failures.append("paragraph_count_mismatch")
    claim_count = len(response.get("claim_support") or ())
    _expect_min(failures, "claim_support_count", claim_count, spec.get("minimum_claim_support"))
    if spec.get("requires_non_answer") and response.get("status") in {"answer_ready", "limited_answer"}:
        failures.append("unsupported_claim_published")
    if spec.get("objective_signature"):
        signature = _objective_signature(response)
        if signature != spec["objective_signature"]:
            failures.append(f"objective_signature:{signature}")
    return failures


def _apply_group_checks(results: list[dict[str, Any]], fixtures: list[dict[str, Any]]) -> None:
    groups: dict[str, list[dict[str, Any]]] = {}
    for case, result in zip(fixtures, results, strict=True):
        group = case.get("paraphrase_group") or case.get("canonical_group")
        if group:
            groups.setdefault(str(group), []).append(result)
    for group, rows in groups.items():
        expected_signature = next(
            (
                case.get("semantic", {}).get("objective_signature")
                for case in fixtures
                if (case.get("paraphrase_group") or case.get("canonical_group")) == group
                and case.get("semantic", {}).get("objective_signature")
            ),
            None,
        )
        if expected_signature:
            for row in rows:
                signature = _objective_signature_from_row(row)
                if signature != expected_signature:
                    row["semantic_failures"].append(f"paraphrase_objective_divergence:{signature}")
        if any(case.get("canonical_group") == group for case in fixtures):
            support_sets = [set(row.get("evidence_trace", {}).get("evidence_ids", ())) for row in rows]
            if support_sets and not set.intersection(*support_sets):
                for row in rows:
                    row["deterministic_failures"].append("canonical_variant_support_mismatch")


def _objective_signature(response: dict[str, Any]) -> str:
    operation = str(response.get("operation") or "")
    route = str(response.get("route") or "")
    if operation == "compare":
        return "compare"
    if operation == "analyze":
        return "analyze"
    if route == "legal_relation" or operation in {"trace", "relation"}:
        return "relation"
    return "quote_or_explain"


def _objective_signature_from_row(row: dict[str, Any]) -> str:
    operation = str(row.get("actual_operation") or "")
    route = str(row.get("actual_route") or "")
    if operation == "compare":
        return "compare"
    if operation == "analyze":
        return "analyze"
    if route == "legal_relation" or operation in {"trace", "relation"}:
        return "relation"
    return "quote_or_explain"


def _response_trace(
    response: dict[str, Any],
    public: dict[str, Any] | None,
    planner: ProbePlanner | None,
    deterministic: list[str],
    semantic: list[str],
) -> dict[str, Any]:
    return {
        "response": {
            "status": response.get("status"),
            "route": response.get("route"),
            "operation": response.get("operation"),
        },
        "deterministic_failures": list(deterministic),
        "semantic_failures": list(semantic),
        "planner_calls": planner.calls if planner is not None else 0,
        "evidence_ids": sorted(_support_ids(response)),
    }


def _planner_trace(response: dict[str, Any], planner: ProbePlanner | None) -> dict[str, Any]:
    plan = response.get("research_plan")
    return {
        "calls": planner.calls if planner is not None else 0,
        "provider_status": getattr(plan, "provider_status", None),
        "rejection_reasons": tuple(str(item) for item in (getattr(plan, "rejection_reasons", ()) or ())),
        "variant_count": len(getattr(plan, "variants", ()) or ()) if plan is not None else 0,
        "information_need_count": len(getattr(plan, "information_needs", ()) or ()) if plan is not None else 0,
        "clarification_dimensions": tuple(getattr(plan, "missing_dimensions", ()) or ()) if plan is not None else (),
    }


def _evidence_trace(response: dict[str, Any], public: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "evidence_ids": sorted(_support_ids(response)),
        "citation_count": len(response.get("citations") or response.get("final_citations") or ()),
        "viewer_ref_count": len(response.get("viewer_refs") or ()),
        "public_support_count": len((public or {}).get("supports") or ()) if public is not None else None,
        "public_target_resolvable_count": sum(
            bool(row.get("viewer_target", {}).get("can_resolve"))
            for row in (public or {}).get("supports", ())
            if isinstance(row, dict)
        )
        if public is not None
        else None,
    }


def _support_ids(response: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for field in ("evidence", "citations", "final_citations", "historical_citations", "metadata_support", "relation_support", "trace_support"):
        for row in response.get(field, ()) or ():
            if isinstance(row, dict):
                value = row.get("evidence_id") or row.get("relation_id") or row.get("source_conflict_id")
                if value:
                    ids.add(str(value))
    for row in response.get("claim_support", ()) or ():
        if isinstance(row, dict):
            ids.update(str(value) for value in row.get("support_evidence_ids", ()) or ())
    return ids


def _has_substantive_quality_assertion(case: dict[str, Any]) -> bool:
    if case.get("probe") == "sequence":
        return any(_has_substantive_quality_assertion(turn) for turn in case.get("turns", ()))
    deterministic = case.get("deterministic", {})
    semantic = case.get("semantic", {})
    keys = {
        "route_in",
        "operation_in",
        "minimum_citations",
        "maximum_citations",
        "planner_calls_min",
        "clarification_dimensions",
        "require_clarification_id",
        "require_no_citations",
        "require_no_public_supports",
        "maximum_public_supports",
        "public_answer_must_not_contain",
        "public_minimum_supports",
        "api_footnote_mode_accepted",
        "forbidden_statuses",
    }
    semantic_keys = {
        "operation_in",
        "required_answer_markers",
        "forbidden_answer_markers",
        "paragraph_count",
        "minimum_claim_support",
        "requires_non_answer",
        "objective_signature",
    }
    return any(deterministic.get(key) not in (None, "", [], (), {}) for key in keys) or any(
        semantic.get(key) not in (None, "", [], (), {}) for key in semantic_keys
    )


def _probe_footnote_mode(request_handler, service, case: dict[str, Any]) -> dict[str, Any]:
    payload = {"query": case["query"], "footnote_mode": case["deterministic"]["api_footnote_value"]}
    try:
        result = request_handler("uud", "ask", payload, service=service)
    except Exception as error:
        return {"accepted": False, "error_type": type(error).__name__, "error_reason": getattr(error, "reason", None)}
    return {"accepted": True, "status": result.get("status")}


def _bind_identity(
    repo_root: Path,
    fixtures_path: Path,
    identity_path: Path,
    frozen: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    frozen_identity = frozen["frozen_identity"]
    manifest = repo_root / frozen_identity["manifest_path"]
    validation = repo_root / frozen_identity["validation_path"]
    identity = {
        "identity_mode": "worktree",
        "commit_sha": _git(repo_root, "rev-parse", "HEAD"),
        "tree_sha": _git(repo_root, "rev-parse", "HEAD^{tree}"),
        "branch": _git(repo_root, "branch", "--show-current"),
        "worktree_status": _git_lines(repo_root, "status", "--porcelain=v1", "--untracked-files=all"),
        "worktree_status_sha256": sha256(
            "\n".join(_git_lines(repo_root, "status", "--porcelain=v1", "--untracked-files=all")).encode("utf-8")
        ).hexdigest(),
        "manifest_sha256": _digest(manifest),
        "validation_sha256": _digest(validation),
        "case_set_sha256": _digest(fixtures_path),
        "evaluator_sha256": _digest(Path(__file__).resolve()),
        "identity_file_sha256": _digest(identity_path),
    }
    errors = _frozen_identity_errors(frozen_identity)
    for field in ("branch", "manifest_sha256", "validation_sha256", "case_set_sha256", "evaluator_sha256"):
        if identity[field] != frozen_identity.get(field) and (field != "branch" or identity[field]):
            errors.append(f"{field}:expected={frozen_identity.get(field)}:actual={identity[field]}")
    return identity, errors


def _bind_archive_identity(
    repo_root: Path,
    fixtures_path: Path,
    identity_path: Path,
    frozen: dict[str, Any],
    archive: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    frozen_identity = frozen["frozen_identity"]
    identity = {
        "identity_mode": "archive",
        "archive_sha256": archive.get("verified_archive_sha256"),
        "verified_archive_sha256": archive.get("verified_archive_sha256"),
        "verified_archive_content_sha256": archive.get("verified_archive_content_sha256"),
        "archive_sidecar_sha256": archive.get("sidecar_sha256"),
        "archive_file_count": archive.get("archive_file_count"),
        "claimed_base_commit_sha": archive.get("claimed_base_commit_sha"),
        "claimed_base_tree_sha": archive.get("claimed_base_tree_sha"),
        "claimed_commit_sha": archive.get("claimed_commit_sha"),
        "claimed_tree_sha": archive.get("claimed_tree_sha"),
        "commit_sha": None,
        "tree_sha": None,
        "branch": archive.get("base_branch") or frozen_identity.get("branch"),
        "manifest_sha256": _digest(repo_root / frozen_identity["manifest_path"]),
        "validation_sha256": _digest(repo_root / frozen_identity["validation_path"]),
        "case_set_sha256": _digest(fixtures_path),
        "evaluator_sha256": _digest(repo_root / frozen_identity["evaluator_path"]),
        "identity_file_sha256": _digest(identity_path),
    }
    errors = list(archive.get("identity_errors", ()))
    errors.extend(_frozen_identity_errors(frozen_identity))
    for field in ("branch", "manifest_sha256", "validation_sha256", "case_set_sha256", "evaluator_sha256"):
        if identity[field] != frozen_identity.get(field):
            errors.append(f"{field}:expected={frozen_identity.get(field)}:actual={identity[field]}")
    return identity, errors


def _frozen_identity_errors(frozen_identity: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if frozen_identity.get("commit_sha") != EXPECTED_COMMIT:
        errors.append(f"frozen_baseline_commit_mismatch:{frozen_identity.get('commit_sha')}")
    if frozen_identity.get("tree_sha") != EXPECTED_TREE:
        errors.append(f"frozen_baseline_tree_mismatch:{frozen_identity.get('tree_sha')}")
    return errors


def _verify_historical_baselines(repo_root: Path, frozen: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for baseline in frozen.get("historical_baselines", ()):
        path_value = baseline.get("path")
        path = repo_root / str(path_value) if path_value else None
        if path is None or not path.is_file():
            errors.append(f"historical_baseline_missing:{path_value}")
            continue
        actual_digest = _digest(path)
        if actual_digest != baseline.get("sha256"):
            errors.append(f"historical_baseline_digest_mismatch:{path_value}")
            continue
        if not baseline.get("row_count") and not baseline.get("classification_counts"):
            continue
        try:
            payload = _read_json(path)
        except (OSError, json.JSONDecodeError):
            errors.append(f"historical_baseline_not_machine_readable:{path_value}")
            continue
        if path.name == "p0_historical_150q_baseline.json":
            if not isinstance(payload, dict):
                errors.append(f"historical_baseline_not_machine_readable:{path_value}")
                continue
            if payload.get("schema") != HISTORICAL_CATALOG_SCHEMA:
                errors.append(f"historical_baseline_schema_invalid:{path_value}")
            rows = payload.get("rows")
            if not isinstance(rows, list) or len(rows) != 150:
                errors.append(f"historical_baseline_rows_missing_or_invalid:{path_value}")
                continue
            if any(not isinstance(row, dict) or set(row) != HISTORICAL_CATALOG_ROW_FIELDS for row in rows):
                errors.append(f"historical_baseline_rows_missing_fields:{path_value}")
                continue
            if any(
                type(row["index"]) is not int
                or not isinstance(row["query"], str)
                or not isinstance(row["status"], str)
                or not isinstance(row["route"], str)
                or type(row["supports_count"]) is not int
                or type(row["viewer_count"]) is not int
                or row["supports_count"] < row["viewer_count"]
                or row["supports_count"] < 0
                or row["viewer_count"] < 0
                for row in rows
            ):
                errors.append(f"historical_baseline_support_viewer_semantics_invalid:{path_value}")
            if [row["index"] for row in rows] != list(range(1, 151)):
                errors.append(f"historical_baseline_row_indices_mismatch:{path_value}")
            if any(row["classification"] not in CLASSIFICATIONS for row in rows):
                errors.append(f"historical_baseline_row_classification_invalid:{path_value}")
            derived_count = len(rows)
            derived_counts = dict(sorted(Counter(row["classification"] for row in rows).items()))
            derived_digest = sha256(
                json.dumps(rows, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
            ).hexdigest()
            if payload.get("row_count") != derived_count or baseline.get("row_count") != derived_count:
                errors.append(f"historical_baseline_row_count_mismatch:{path_value}")
            if payload.get("classification_counts") != derived_counts or baseline.get("classification_counts") != derived_counts:
                errors.append(f"historical_baseline_classification_counts_mismatch:{path_value}")
            if payload.get("catalog_digest") != derived_digest or baseline.get("catalog_digest") != derived_digest:
                errors.append(f"historical_baseline_catalog_digest_mismatch:{path_value}")
        else:
            actual_count = _baseline_row_count(payload)
            if actual_count != int(baseline.get("row_count", -1)):
                errors.append(f"historical_baseline_row_count_mismatch:{path_value}")
    return errors


def _baseline_row_count(payload: Any) -> int | None:
    if not isinstance(payload, dict):
        return len(payload) if isinstance(payload, list) else None
    if isinstance(payload.get("rows"), list):
        return len(payload["rows"])
    if isinstance(payload.get("evaluation_identity"), dict) and payload["evaluation_identity"].get("case_count") is not None:
        return int(payload["evaluation_identity"]["case_count"])
    if payload.get("eval_case_count") is not None:
        return int(payload["eval_case_count"])
    counts = payload.get("counts")
    if isinstance(counts, dict) and all(isinstance(value, int) for value in counts.values()):
        return sum(counts.values())
    return None


def _missing_mandatory_families(fixtures: list[dict[str, Any]]) -> set[str]:
    represented = set()
    for case in fixtures:
        represented.add(str(case.get("family")))
        represented.update(str(value) for value in case.get("coverage_families", ()) or ())
    return MANDATORY_FIXTURE_FAMILIES - represented


def _validate_case(case: dict[str, Any]) -> None:
    required = {"fixture_id", "family", "adjudication"}
    missing = required - set(case)
    if missing:
        raise ContractError(f"fixture missing fields: {sorted(missing)}")
    adjudication = case["adjudication"]
    if adjudication.get("target_classification") not in CLASSIFICATIONS:
        raise ContractError(f"invalid target classification: {case['fixture_id']}")
    if adjudication.get("pre_fix_expected") not in CLASSIFICATIONS:
        raise ContractError(f"invalid pre-fix classification: {case['fixture_id']}")
    if not case.get("deterministic") and not case.get("semantic") and case.get("probe") != "recorded_evidence":
        raise ContractError(f"fixture has no deterministic or semantic checks: {case['fixture_id']}")
    if (
        case["adjudication"]["target_classification"] == "PASS"
        and case.get("probe") != "recorded_evidence"
        and not _has_substantive_quality_assertion(case)
    ):
        raise ContractError(f"PASS fixture requires substantive quality assertion: {case['fixture_id']}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [row.get("fixture_id") for row in rows]
    if len(ids) != len(set(ids)):
        raise ContractError("duplicate P0 fixture_id")
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _runtime(repo_root: Path):
    source = str(repo_root / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    from tjipto.runtime.api import handle_request
    from tjipto.runtime.service import LegalRuntimeService

    return LegalRuntimeService, handle_request


def _git(repo_root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=repo_root, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _git_lines(repo_root: Path, *args: str) -> list[str]:
    value = _git(repo_root, *args)
    return value.splitlines() if value != "unavailable" else []


def _digest(path: Path) -> str | None:
    if not path.is_file():
        return None
    return sha256(path.read_bytes()).hexdigest()


def _paragraph_count(value: object) -> int:
    return len([item for item in str(value or "").split("\n\n") if item.strip()])


def _expect_any(failures: list[str], field: str, actual: object, expected: object) -> None:
    if expected and actual not in set(expected):
        failures.append(f"{field}:expected_one_of={tuple(expected)!r}:actual={actual!r}")


def _expect_min(failures: list[str], field: str, actual: int, expected: object) -> None:
    if expected is not None and actual < int(expected):
        failures.append(f"{field}:expected_min={expected}:actual={actual}")


def _expect_max(failures: list[str], field: str, actual: int, expected: object) -> None:
    if expected is not None and actual > int(expected):
        failures.append(f"{field}:expected_max={expected}:actual={actual}")


if __name__ == "__main__":
    raise SystemExit(main())
