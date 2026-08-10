from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from tjipto.runtime.service import LegalRuntimeService


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "tests/fixtures/uud/research_retrieval_cases_v0.jsonl"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the frozen UUD legal-research retrieval benchmark v0.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--base-commit", default=None)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    cases = _read_jsonl(args.cases)
    service = LegalRuntimeService(ROOT)
    results = [_evaluate(case, service) for case in cases]
    gaps = sum(row["outcome"] == "RESEARCH_GAP" for row in results)
    report: dict[str, Any] = {
        "status": "pass" if gaps == 0 else "research_gap",
        "counts": {
            "pass": len(results) - gaps,
            "research_gap": gaps,
            "cases": len(results),
        },
        "results": results,
        "evaluation_identity": {
            "base_commit": args.base_commit or _git_head(),
            "case_set_sha256": _sha256(args.cases),
            "evaluator_sha256": _sha256(Path(__file__)),
            "case_count": len(cases),
            "family_counts": {
                family: sum(row["family"] == family for row in cases)
                for family in sorted({row["family"] for row in cases})
            },
            "behavior_counts": {
                behavior: sum(row["expected_behavior"] == behavior for row in cases)
                for behavior in ("retrieve", "abstain", "clarify")
            },
        },
    }
    if args.report:
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"research_retrieval_v0: PASS={len(results) - gaps} RESEARCH_GAP={gaps}")
    for row in results:
        if row["outcome"] != "PASS":
            print(f"RESEARCH_GAP: {row['case_id']} :: {'; '.join(row['errors'])}")
    return 0


def _evaluate(case: dict[str, Any], service: LegalRuntimeService) -> dict[str, Any]:
    response = service.ask(case["corpus_id"], case["query"])
    citations = tuple(response.get("citations", ())) + tuple(response.get("historical_citations", ()))
    actual_ids = [str(row.get("evidence_id")) for row in citations if row.get("evidence_id")]
    actual_citations = [str(row.get("citation")) for row in citations if row.get("citation")]
    errors = []
    if response.get("status") != case["expected_status"]:
        errors.append(f"status:{response.get('status')}!= {case['expected_status']}")
    if response.get("route") != case["expected_route"]:
        errors.append(f"route:{response.get('route')}!= {case['expected_route']}")
    if case.get("expected_clarification_kind") and response.get("clarification_kind") != case["expected_clarification_kind"]:
        errors.append(f"clarification_kind:{response.get('clarification_kind')}")
    if not set(case.get("expected_support_ids", ())).issubset(actual_ids):
        errors.append("support_ids_missing")
    if not set(case.get("expected_citations", ())).issubset(actual_citations):
        errors.append("citations_missing")
    return {
        "case_id": case["case_id"],
        "family": case["family"],
        "query": case["query"],
        "outcome": "PASS" if not errors else "RESEARCH_GAP",
        "errors": errors,
        "actual": {
            "status": response.get("status"),
            "route": response.get("route"),
            "intent": response.get("intent"),
            "support_ids": actual_ids,
            "citations": actual_citations,
            "clarification_kind": response.get("clarification_kind"),
        },
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


if __name__ == "__main__":
    raise SystemExit(main())
