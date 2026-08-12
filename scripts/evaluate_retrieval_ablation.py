"""Measure sparse, dense, hybrid, and bounded-planning retrieval on one frozen case set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "tests/fixtures/uud/research_retrieval_cases.jsonl"
LANES = ("sparse", "dense", "hybrid", "planning")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare bounded retrieval lanes on a frozen research case set.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--runtime-root", type=Path, default=ROOT)
    parser.add_argument("--runtime-commit")
    parser.add_argument("--runtime-tree")
    parser.add_argument("--runtime-digest")
    parser.add_argument("--cutoff", type=int, default=5)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    report = evaluate(args)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        "retrieval_ablation: "
        + " ".join(f"{lane}={report['lanes'][lane]['execution_status']}" for lane in LANES)
    )
    return 0 if report["execution_status"] == "valid" else 1


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    cases = _read_jsonl(args.cases)
    runtime_root = args.runtime_root.resolve()
    sys.path.insert(0, str(runtime_root / "src"))
    from tjipto.runtime.service import LegalRuntimeService
    from tjipto.retrieval.research import ResearchIntent

    service = LegalRuntimeService(runtime_root)
    lanes: dict[str, dict[str, Any]] = {}
    for lane in LANES:
        results: list[dict[str, Any]] = []
        for case in cases:
            started = time.perf_counter()
            try:
                if lane == "planning":
                    intent = ResearchIntent(
                        multiple_supports=case["family"] in {"comparison", "multi_support", "multi_hop_procedure"},
                        comparison=case["family"] == "comparison",
                        decomposition=case["family"] in {"comparison", "multi_support", "multi_hop_procedure"},
                        relation_traversal=case["family"] == "multi_hop_procedure",
                    )
                    response = service.research(case["corpus_id"], case["query"], intent=intent, limit=args.cutoff)
                    rows = tuple(response.get("matches") or ())
                    status = response.get("status")
                else:
                    store = service._store(case["corpus_id"])
                    response = service._route_retrieval(
                        case["corpus_id"], case["query"], store, route=lane, limit=args.cutoff
                    )
                    rows = tuple(response.get("matches") or ())
                    status = response.get("status")
                elapsed = time.perf_counter() - started
                if status in {"dense_unavailable", "unsupported_corpus", "no_results"} and lane == "dense":
                    execution = "unavailable" if status == "dense_unavailable" else "valid"
                else:
                    execution = "valid"
                results.append(
                    {
                        "case_id": case.get("case_id"),
                        "execution_status": execution,
                        "status": status,
                        "candidate_count": len(rows),
                        "latency_seconds": round(elapsed, 6),
                        "support_ids": [str(row.get("evidence_id")) for row in rows if row.get("evidence_id")],
                    }
                )
            except Exception as error:  # report invalid execution, never turn it into a capability result
                results.append(
                    {
                        "case_id": case.get("case_id"),
                        "execution_status": "invalid",
                        "error": f"{type(error).__name__}:{error}",
                        "candidate_count": 0,
                        "latency_seconds": round(time.perf_counter() - started, 6),
                        "support_ids": [],
                    }
                )
        lanes[lane] = _lane_report(cases, results, args.cutoff)
    return {
        "execution_status": "valid" if all(row["execution_status"] != "invalid" for row in lanes.values()) else "invalid",
        "evaluation_identity": {
            "case_set_sha256": _sha256(args.cases),
            "evaluator_sha256": _sha256(Path(__file__)),
            "case_count": len(cases),
            "cutoff": args.cutoff,
            "runtime_commit": args.runtime_commit or _git_optional(runtime_root, "HEAD"),
            "runtime_tree_sha": args.runtime_tree or _git_optional(runtime_root, "HEAD^{tree}"),
            "runtime_snapshot_sha256": args.runtime_digest or "unavailable",
        },
        "lanes": lanes,
    }


def _lane_report(cases: list[dict[str, Any]], results: list[dict[str, Any]], cutoff: int) -> dict[str, Any]:
    valid = [row for row in results if row["execution_status"] == "valid"]
    unavailable = [row for row in results if row["execution_status"] == "unavailable"]
    if not valid:
        metrics = None
    else:
        metrics = _metrics(cases, results, cutoff)
    return {
        "execution_status": "invalid" if any(row["execution_status"] == "invalid" for row in results) else "unavailable" if unavailable else "valid",
        "case_count": len(cases),
        "valid_case_count": len(valid),
        "unavailable_case_count": len(unavailable),
        "metrics": metrics,
        "results": results,
    }


def _metrics(cases: list[dict[str, Any]], results: list[dict[str, Any]], cutoff: int) -> dict[str, Any]:
    hits = 0
    reciprocal = 0.0
    ndcg = 0.0
    group_hits = 0
    group_total = 0
    candidate_counts = [row["candidate_count"] for row in results if row["execution_status"] == "valid"]
    latencies = [row["latency_seconds"] for row in results if row["execution_status"] == "valid"]
    for case, result in zip(cases, results):
        if result["execution_status"] != "valid":
            continue
        ids = result["support_ids"][:cutoff]
        relevant = _relevant_ids(case)
        positions = [index + 1 for index, evidence_id in enumerate(ids) if evidence_id in relevant]
        if positions:
            hits += 1
            reciprocal += 1.0 / min(positions)
        ideal_count = min(len(relevant), cutoff)
        dcg = sum(1.0 / _log2(index + 2) for index, evidence_id in enumerate(ids) if evidence_id in relevant)
        ideal = sum(1.0 / _log2(index + 2) for index in range(ideal_count))
        ndcg += dcg / ideal if ideal else 0.0
        groups = _groups(case)
        if groups:
            group_total += 1
            if any(set(group) <= set(ids) for group in groups):
                group_hits += 1
    denominator = len([row for row in results if row["execution_status"] == "valid"])
    return {
        "cutoff": cutoff,
        "hit_rate_at_k": hits / denominator if denominator else None,
        "mrr": reciprocal / denominator if denominator else None,
        "ndcg_at_k": ndcg / denominator if denominator else None,
        "support_group_recall_at_k": group_hits / group_total if group_total else None,
        "hit_denominator": denominator,
        "support_group_denominator": group_total,
        "mean_candidate_count": sum(candidate_counts) / len(candidate_counts) if candidate_counts else None,
        "mean_latency_seconds": sum(latencies) / len(latencies) if latencies else None,
    }


def _relevant_ids(case: dict[str, Any]) -> set[str]:
    return set(value for group in _groups(case) for value in group)


def _groups(case: dict[str, Any]) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(str(value) for value in group) for group in (*case.get("gold_support_groups", ()), *case.get("alternative_support_groups", ())))


def _log2(value: int) -> float:
    import math

    return math.log(value, 2)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_optional(root: Path, revision: str) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", revision], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


if __name__ == "__main__":
    raise SystemExit(main())
