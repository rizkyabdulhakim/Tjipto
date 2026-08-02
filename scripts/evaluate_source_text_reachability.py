from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Iterable

from tjipto.runtime.api import handle_request
from tjipto.runtime.service import LegalRuntimeService
from tjipto.runtime.source_text import source_text_health


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "tests" / "fixtures" / "source_text_reachability_cases.json"
ORACLE = ROOT / "tests" / "fixtures" / "source_text_semantic_oracle.json"


def evaluate_projection(raw_rows: Iterable[dict], semantic_rows: list[dict], projected_rows: list[dict]) -> dict:
    oracle = json.loads(ORACLE.read_text(encoding="utf-8"))
    matrix = {
        (row["semantic_classification"], row["legal_force"]): row
        for row in oracle["semantic_matrix"]
    }
    semantic_index: dict[tuple[object, ...], list[dict]] = defaultdict(list)
    for span in semantic_rows:
        semantic_index[(span.get("source_document_id"), span.get("page_number"), span.get("text_start"), span.get("text_end"))].append(span)
    projected_by_id = {str(row.get("raw_source_span_id")): row for row in projected_rows}
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    counts = Counter()
    class_counts = Counter()
    force_counts = Counter()
    raw_nonempty_row_count = 0
    for raw in raw_rows:
        if not str(raw.get("raw_text") or "").strip():
            continue
        raw_nonempty_row_count += 1
        row_id = str(raw.get("raw_source_span_id") or "")
        semantic_text = str(raw.get("semantic_text") or "").strip()
        matches = semantic_index[
            (
                raw.get("source_document_id"),
                raw.get("page_number"),
                raw.get("semantic_text_start"),
                raw.get("semantic_text_end"),
            )
        ] if semantic_text else []
        if semantic_text and len(matches) != 1:
            counts["semantic_join_missing_count" if not matches else "semantic_join_duplicate_count"] += 1
            reason = "semantic_join_missing" if not matches else "semantic_join_duplicate"
            expected = _fail_closed_expected(oracle, raw, reason)
        elif raw.get("classification") == "source_annotation_marker":
            expected = _expected(oracle["source_annotation"], raw, None)
        elif matches:
            semantic = matches[0]
            specification = matrix.get((semantic.get("semantic_classification"), semantic.get("legal_force")))
            expected = (
                _expected(specification, raw, semantic)
                if specification is not None
                else _fail_closed_expected(oracle, raw, "unsupported_semantic_mapping")
            )
        else:
            expected = _fail_closed_expected(
                oracle,
                raw,
                str(raw.get("disposition_reason") or "nonsemantic_source_text"),
            )
        expected["source_role"] = raw.get("source_role")
        actual = projected_by_id.get(row_id)
        if actual is None:
            counts["projected_row_missing_count"] += 1
            continue
        expected_disposition = str(expected["disposition"])
        actual_disposition = str(actual.get("disposition") or "missing")
        confusion[expected_disposition][actual_disposition] += 1
        class_counts[str(actual.get("semantic_classification") or "nonsemantic")] += 1
        force_counts[str(actual.get("legal_force") or "missing")] += 1
        for field in (
            "disposition",
            "legal_force",
            "capabilities",
            "legal_answer_eligible",
            "source_answer_eligible",
            "legal_citation_eligible",
            "source_citation_eligible",
            "default_highlight_eligible",
            "abstention_reason",
            "semantic_join_status",
            "semantic_text_span_id",
            "semantic_classification",
            "source_role",
            "temporal_context",
        ):
            actual_value = tuple(actual.get(field) or ()) if field == "capabilities" else actual.get(field)
            expected_value = tuple(expected.get(field) or ()) if field == "capabilities" else expected.get(field)
            counts[f"{field}_mismatch_count"] += actual_value != expected_value
        noncurrent = expected.get("legal_force") != "canonical_normative"
        promoted = any(actual.get(field) is True for field in (
            "legal_answer_eligible", "legal_citation_eligible", "default_highlight_eligible"
        ))
        counts["legal_force_escalation_count"] += noncurrent and promoted
        counts["nonnormative_legal_answer_count"] += (
            expected.get("semantic_classification") != "normative_constitutional_text"
            and actual.get("legal_answer_eligible") is True
        )
        counts["historical_presented_as_current_count"] += (
            expected.get("legal_force") == "historical_normative" and promoted
        )
        counts["marker_leakage_count"] += (
            raw.get("classification") == "source_annotation_marker" and promoted
        )
        counts["fabricated_annotation_target_count"] += (
            raw.get("classification") == "source_annotation_marker"
            and bool(actual.get("target_legal_unit_id"))
            and actual.get("annotation_target_basis") != "exact_source_selector"
        )
    mismatch_count = sum(value for key, value in counts.items() if key.endswith("_mismatch_count"))
    unsafe_count = sum(
        counts[key]
        for key in (
            "semantic_join_missing_count",
            "semantic_join_duplicate_count",
            "projected_row_missing_count",
            "legal_force_escalation_count",
            "nonnormative_legal_answer_count",
            "historical_presented_as_current_count",
            "marker_leakage_count",
            "fabricated_annotation_target_count",
        )
    )
    total = sum(confusion_row.total() for confusion_row in confusion.values())
    return {
        "status": "PASS" if mismatch_count + unsafe_count == 0 else "FAIL",
        "raw_nonempty_row_count": raw_nonempty_row_count,
        "projected_row_count": len(projected_rows),
        "semantic_disposition_accuracy": 1.0 if total and not counts["disposition_mismatch_count"] else 0.0,
        "legal_force_accuracy": 1.0 if total and not counts["legal_force_mismatch_count"] else 0.0,
        "semantic_disposition_confusion_matrix": {
            expected: dict(sorted(actual.items())) for expected, actual in sorted(confusion.items())
        },
        "semantic_classification_counts": dict(sorted(class_counts.items())),
        "legal_force_counts": dict(sorted(force_counts.items())),
        **dict(sorted(counts.items())),
    }


def _expected(specification: dict, raw: dict, semantic: dict | None) -> dict:
    geometry = all(raw.get(key) is not None for key in ("x0", "y0", "x1", "y1"))
    expected = dict(specification)
    for field in ("source_citation_eligible", "legal_citation_eligible", "default_highlight_eligible"):
        if expected.get(field) == "geometry":
            expected[field] = geometry
        elif expected.get(field) == "viewer_highlightable":
            expected[field] = (semantic or {}).get("viewer_highlightable") is True
    expected.update({
        "semantic_text_span_id": (semantic or {}).get("text_span_id"),
        "semantic_classification": (semantic or {}).get("semantic_classification"),
        "semantic_join_status": expected.get("semantic_join_status", "exact"),
        "temporal_context": str((semantic or {}).get("temporal_context") or raw.get("source_role") or ""),
        "abstention_reason": expected.get("abstention_reason"),
    })
    return expected


def _fail_closed_expected(oracle: dict, raw: dict, reason: str) -> dict:
    expected = _expected(oracle["fail_closed"], raw, None)
    expected["abstention_reason"] = reason
    expected["semantic_join_status"] = reason.removeprefix("semantic_join_") if reason.startswith("semantic_join_") else "not_applicable"
    return expected


def _jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def evaluate(repo_root: Path = ROOT) -> dict:
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    service = LegalRuntimeService(repo_root)
    store = service._store("uud")
    if store is None:
        raise RuntimeError("uud_corpus_unavailable")
    raw_path = store.config.artifact_path("raw_source_spans")
    semantic_rows = [json.loads(line) for line in store.config.artifact_path("page_text_spans").read_text(encoding="utf-8").splitlines() if line.strip()]
    projection = evaluate_projection(_jsonl(raw_path), semantic_rows, store.raw_source_spans)
    results = []
    for case in cases:
        raw = service.ask("uud", case["query"])
        public = handle_request("uud", "ask", {"query": case["query"]}, repo_root, service)
        supports = tuple(public.get("supports") or ())
        authorities = {row.get("authority_kind") for row in supports}
        citations = tuple(row.get("citation") for row in supports if row.get("citation"))
        viewer_targets = tuple(
            row for row in supports if row.get("viewer_target", {}).get("can_resolve") is True
        )
        expected_support = None
        resolved = None
        target_request = None
        if case["expected_exact_support"]:
            for support in viewer_targets:
                target = support.get("viewer_target", {}).get("public_target_id")
                request = service._public_target_request("uud", target)
                if request and request.get("evidence_id") == case["expected_support_identity"]:
                    expected_support = support
                    target_request = request
                    resolved = handle_request("uud", "viewer", {"target": target}, repo_root, service)
                    break
        exact_rectangles = tuple((resolved or {}).get("bbox_rectangles") or ())
        exact_target_valid = bool(
            expected_support
            and target_request
            and resolved
            and target_request.get("source_document_id") == case["expected_source_document_id"]
            and tuple(expected_support.get("page_numbers") or ()) == (case["expected_page_number"],)
            and case["expected_page_number"] in tuple(resolved.get("page_numbers") or ())
            and resolved.get("status") == "viewer_payload_ready"
            and resolved.get("viewer_highlightable") is True
            and exact_rectangles
            and all(row.get("bbox_precision") == "exact" and row.get("viewer_highlightable") is True for row in exact_rectangles)
            and case["expected_page_number"] in {row.get("page_number") for row in exact_rectangles}
            and resolved.get("quoted_text") == expected_support.get("text") == target_request.get("quoted_text")
            and sha256(str(resolved.get("quoted_text") or "").encode("utf-8")).hexdigest() == case["expected_text_sha256"]
            and (
                bool(target_request.get("bbox_refs"))
                if case["expected_lineage"] == "bbox_refs"
                else str(target_request.get("evidence_id") or "").startswith("uud_metadata_field_grounding::")
            )
        )
        nonexact_not_promoted = not case["expected_exact_support"] and all(
            support.get("citation_final") is not True
            and support.get("viewer_target", {}).get("can_resolve") is not True
            for support in supports
        )
        checks = {
            "route": raw.get("route") == case["expected_route"],
            "answerability": (public.get("kind") == "answer" and public.get("status") in {"answer_ready", "limited_answer"})
            is case["expected_answerability"],
            "authority": case["expected_authority"] in authorities,
            "viewer_target_resolution": exact_target_valid if case["expected_exact_support"] else nonexact_not_promoted,
            "citation_presence": bool(citations) is case["expected_citation"],
            "viewer_target_presence": bool(viewer_targets) is case["expected_highlight"],
            "abstention": (public.get("status") == "insufficient_evidence") is case["expected_abstention"],
            "forbidden_authority": case["forbidden_authority_promotion"] not in authorities,
        }
        results.append({"case_id": case["case_id"], "source_class": case["expected_source_class"], "checks": checks})
    health = source_text_health(store)
    failed = sum(not check for result in results for check in result["checks"].values())
    failed += projection["status"] != "PASS"
    failed += sum(
        health[key]
        for key in (
            "meaningful_source_span_without_route_count",
            "unmapped_source_annotation_count",
            "source_annotation_legal_citation_count",
            "source_annotation_default_highlight_count",
            "source_annotation_occurrence_without_selector_or_geometry_count",
            "source_annotation_occurrence_without_target_or_reason_count",
            "fabricated_annotation_target_count",
        )
    )
    return {
        "status": "PASS" if failed == 0 else "FAIL",
        "case_count": len(results),
        "failed_check_count": failed,
        "query_route_accuracy": sum(result["checks"]["route"] for result in results) / len(results),
        "query_authority_accuracy": sum(result["checks"]["authority"] for result in results) / len(results),
        "query_answerability_accuracy": sum(result["checks"]["answerability"] for result in results) / len(results),
        "viewer_target_resolution_accuracy": sum(result["checks"]["viewer_target_resolution"] for result in results) / len(results),
        "citation_presence_accuracy": sum(result["checks"]["citation_presence"] for result in results) / len(results),
        "query_answerability_mismatch_count": sum(not result["checks"]["answerability"] for result in results),
        "query_abstention_accuracy": sum(result["checks"]["abstention"] for result in results) / len(results),
        "meaningful_span_reachability": 1.0
        if health["meaningful_source_span_without_route_count"] == 0
        else 0.0,
        "projection": projection,
        "health": health,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = evaluate()
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
