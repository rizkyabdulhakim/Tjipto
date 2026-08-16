"""Frozen held-out comparison of V0 retrieval and bounded semantic planning."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from math import log2
from pathlib import Path
import subprocess

from tjipto.runtime.service import LegalRuntimeService


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "tests" / "fixtures" / "uud" / "semantic_generalization_cases.jsonl"


class FrozenProvider:
    def __init__(self, proposal: dict):
        self.proposal = proposal
        self.calls = 0

    def propose(self, request: dict) -> dict:
        self.calls += 1
        return self.proposal


def _read_cases() -> list[dict]:
    return [json.loads(line) for line in CASES.read_text(encoding="utf-8").splitlines() if line.strip()]


def _ranking(response: dict, expected: tuple[str, ...]) -> tuple[float, float, float]:
    citations = [str(row.get("citation") or "") for row in response.get("matches", ())]
    ranks = [citations.index(citation) + 1 for citation in expected if citation in citations]
    recall = len(ranks) / len(expected) if expected else 1.0
    mrr = 1.0 / min(ranks) if ranks else 0.0
    ndcg = sum(1.0 / log2(rank + 1) for rank in ranks) / sum(1.0 / log2(index + 2) for index in range(len(expected))) if expected else 1.0
    return recall, mrr, ndcg


def _support_fulfilled(response: dict, expected: tuple[str, ...]) -> bool:
    """Require the server's typed sufficiency result for supported cases."""
    if not expected:
        return True
    sufficiency = response.get("sufficiency") or {}
    if sufficiency.get("status") != "complete":
        return False
    if sufficiency.get("missing_requirement_ids"):
        return False
    citations = {
        str(row.get("citation") or "")
        for row in response.get("matches", ())
        if isinstance(row, dict)
    }
    return set(expected) <= citations


def _runtime_identity() -> dict[str, str]:
    """Bind reports when running from a checkout, but keep archive tests portable."""
    identity: dict[str, str] = {}
    for name, revision in (("commit", "HEAD"), ("tree", "HEAD^{tree}")):
        try:
            identity[name] = subprocess.check_output(
                ["git", "rev-parse", revision], cwd=ROOT, text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            identity[name] = "unavailable"
    return identity


def _measure(cases: list[dict]) -> dict:
    v0 = LegalRuntimeService(ROOT)
    planned = LegalRuntimeService(ROOT)
    rows = []
    for case in cases:
        expected = tuple(str(value) for value in case.get("expected_citations", ()))
        baseline = v0.ask("uud", case["query"])
        provider = FrozenProvider(dict(case["proposal"]))
        planned._planning_provider = provider
        orchestration = planned.ask("uud", case["query"])
        base_metrics = _ranking(baseline, expected)
        planned_metrics = _ranking(orchestration, expected)
        drift = any("scope_invariant_violation" in reason for reason in orchestration.get("research_plan", ()).rejection_reasons) if orchestration.get("research_plan") else False
        allowed_states = tuple(
            str(value)
            for value in case.get("expected_terminal_states", ())
            if isinstance(value, str)
        )
        terminal_state_ok = orchestration["status"] in allowed_states
        support_fulfilled = _support_fulfilled(orchestration, expected)
        rows.append({
            "case_id": case["case_id"],
            "family": case["family"],
            "v0": {"required_support_recall": base_metrics[0], "mrr": base_metrics[1], "ndcg": base_metrics[2]},
            "orchestrated": {"required_support_recall": planned_metrics[0], "mrr": planned_metrics[1], "ndcg": planned_metrics[2]},
            "provider_calls": provider.calls,
            "status": orchestration["status"],
            "allowed_terminal_states": allowed_states,
            "terminal_state_ok": terminal_state_ok,
            "required_support_fulfilled": support_fulfilled,
            "sufficiency": orchestration.get("sufficiency"),
            "query_drift": drift,
            "hard_negative_fp": case["family"] == "out_of_corpus_hard_negative" and orchestration["status"] != "insufficient_evidence",
            "expected_status": case.get("expected_status"),
        })
    def average(path: str, metric: str) -> float:
        return sum(row[path][metric] for row in rows) / len(rows)
    failures = [
        row["case_id"]
        for row, case in zip(rows, cases)
        if row["hard_negative_fp"]
        or not row["terminal_state_ok"]
        or not row["required_support_fulfilled"]
        or case["family"] not in {"out_of_corpus_hard_negative", "negation_modality"} and row["provider_calls"] != 1
    ]
    return {
        "evaluation_identity": {
            "case_set_sha256": sha256(CASES.read_bytes()).hexdigest(),
            "evaluator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
            "case_count": len(cases),
        },
        "runtime_identity": _runtime_identity(),
        "rows": rows,
        "metrics": {
            "v0_required_support_recall": average("v0", "required_support_recall"),
            "orchestrated_required_support_recall": average("orchestrated", "required_support_recall"),
            "v0_mrr": average("v0", "mrr"),
            "orchestrated_mrr": average("orchestrated", "mrr"),
            "v0_ndcg": average("v0", "ndcg"),
            "orchestrated_ndcg": average("orchestrated", "ndcg"),
            "hard_negative_fp": sum(bool(row["hard_negative_fp"]) for row in rows),
            "query_drift_rate": sum(bool(row["query_drift"]) for row in rows) / len(rows),
        },
        "failures": failures,
        "status": "valid" if not failures else "invalid",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    report = _measure(_read_cases())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "metrics": report["metrics"]}, sort_keys=True))
    return 0 if report["status"] == "valid" else 2


if __name__ == "__main__":
    raise SystemExit(main())
