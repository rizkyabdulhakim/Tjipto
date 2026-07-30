from __future__ import annotations

import argparse
import json
from pathlib import Path

from tjipto.runtime.api import handle_request
from tjipto.runtime.service import LegalRuntimeService
from tjipto.runtime.source_text import source_text_health


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "tests" / "fixtures" / "source_text_reachability_cases.json"


def evaluate(repo_root: Path = ROOT) -> dict:
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    service = LegalRuntimeService(repo_root)
    store = service._store("uud")
    if store is None:
        raise RuntimeError("uud_corpus_unavailable")
    results = []
    for case in cases:
        raw = service.ask("uud", case["query"])
        public = handle_request("uud", "ask", {"query": case["query"]}, repo_root, service)
        supports = tuple(public.get("supports") or ())
        authorities = {row.get("authority_kind") for row in supports}
        citations = tuple(row.get("citation") for row in supports if row.get("citation"))
        highlights = tuple(
            row for row in supports if row.get("viewer_target", {}).get("can_resolve") is True
        )
        checks = {
            "route": raw.get("route") == case["expected_route"],
            "answerability": (public.get("kind") == "answer" and public.get("status") in {"answer_ready", "limited_answer"})
            is case["expected_answerability"],
            "authority": case["expected_authority"] in authorities,
            "exact_support": bool(highlights) is case["expected_exact_support"],
            "citation": bool(citations) is case["expected_citation"],
            "highlight": bool(highlights) is case["expected_highlight"],
            "abstention": (public.get("status") == "insufficient_evidence") is case["expected_abstention"],
            "forbidden_authority": case["forbidden_authority_promotion"] not in authorities,
        }
        results.append({"case_id": case["case_id"], "source_class": case["expected_source_class"], "checks": checks})
    health = source_text_health(store)
    failed = sum(not check for result in results for check in result["checks"].values())
    failed += health["meaningful_source_span_without_route_count"]
    failed += health["unmapped_source_annotation_count"]
    failed += health["source_annotation_legal_citation_count"]
    failed += health["source_annotation_default_highlight_count"]
    return {
        "status": "PASS" if failed == 0 else "FAIL",
        "case_count": len(results),
        "failed_check_count": failed,
        "route_accuracy": sum(result["checks"]["route"] for result in results) / len(results),
        "source_class_accuracy": sum(result["checks"]["authority"] for result in results) / len(results),
        "retrieval_recall": sum(result["checks"]["answerability"] for result in results) / len(results),
        "exact_support_coverage": sum(result["checks"]["exact_support"] for result in results) / len(results),
        "citation_correctness": sum(result["checks"]["citation"] for result in results) / len(results),
        "unsupported_claim_count": sum(not result["checks"]["answerability"] for result in results),
        "legal_force_escalation_count": sum(not result["checks"]["forbidden_authority"] for result in results),
        "abstention_correctness": sum(result["checks"]["abstention"] for result in results) / len(results),
        "marker_leakage_count": health["source_annotation_legal_citation_count"],
        "meaningful_span_reachability": 1.0
        if health["meaningful_source_span_without_route_count"] == 0
        else 0.0,
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
