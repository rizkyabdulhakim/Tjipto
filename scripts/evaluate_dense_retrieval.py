from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from tjipto.evidence.store import EvidenceStore
from tjipto.retrieval.dense import DenseUnavailable, LocalDenseProvider, dense_index_for_store
from tjipto.runtime.service import LegalRuntimeService


ROOT = Path(__file__).resolve().parents[1]
RETRIEVAL_CASES = ROOT / "tests/fixtures/uud/retrieval_cases.jsonl"
RESEARCH_CASES = ROOT / "tests/fixtures/uud/research_retrieval_cases.jsonl"
CUTOFF = 10


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare the independent BGE-M3 lane against the frozen retrieval cases.")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args(argv)
    case_paths = (RETRIEVAL_CASES, RESEARCH_CASES)
    all_cases = [row for path in case_paths for row in _read_jsonl(path)]
    cases = _eligible_cases(case_paths)
    identity = {
        "case_set_sha256": _digest_files(case_paths),
        "evaluator_sha256": _sha256(Path(__file__)),
        "case_count": len(all_cases),
        "dense_eligible_case_count": len(cases),
        "cutoff": CUTOFF,
        "base_commit": _git("rev-parse", "HEAD"),
        "base_tree": _git("rev-parse", "HEAD^{tree}"),
    }
    report: dict[str, Any] = {"status": "unavailable", "identity": identity, "case_counts": _case_counts(all_cases)}
    try:
        service = LegalRuntimeService(ROOT)
        store = service._store("uud")
        if store is None:
            raise DenseUnavailable("corpus_not_ready")
        config = replace(store.config, manifest=dict(store.config.manifest) | {"dense_retrieval": True})
        dense_store = EvidenceStore(config)
        provider = LocalDenseProvider(timeout_seconds=args.timeout)
        started = time.perf_counter()
        index = dense_index_for_store(dense_store, provider=provider)
        build_seconds = time.perf_counter() - started
        query_batch = provider.embed(tuple(row["query"] for row in cases))
        metrics = _metrics(cases, [index.search(vector, CUTOFF) for vector in query_batch.vectors])
        report.update(
            {
                "status": "valid",
                "dense": {"metrics": metrics, "index": index.identity_record(), "build_seconds": round(build_seconds, 3)},
                "production_baseline": {"commit": identity["base_commit"], "tree": identity["base_tree"], "behavior": "measured separately by the existing retrieval contract"},
            }
        )
    except DenseUnavailable as error:
        report["reason"] = error.code
    if args.report:
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "case_count": len(all_cases), "reason": report.get("reason")}, sort_keys=True))
    return 0 if report["status"] == "valid" else 2


def _eligible_cases(paths: tuple[Path, ...]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for path in paths:
        for row in _read_jsonl(path):
            if row.get("expected_behavior") != "retrieve":
                continue
            groups = row.get("gold_support_groups") or []
            if not groups and row.get("gold_support_ids"):
                groups = [[support_id] for support_id in row["gold_support_ids"]]
            if groups:
                cases.append({"id": row.get("id") or row.get("case_id"), "query": row["query"], "gold_groups": groups})
    return cases


def _metrics(cases: list[dict[str, Any]], rankings: list[list[dict]]) -> dict[str, float | int | None]:
    relevant = 0
    retrieved = 0
    reciprocal = 0.0
    ndcg = 0.0
    ideal = 0.0
    group_hits = 0
    for case, rows in zip(cases, rankings):
        ids = [str(row.get("evidence_id")) for row in rows]
        gold = {str(item) for group in case["gold_groups"] for item in group}
        hits = [index for index, evidence_id in enumerate(ids, 1) if evidence_id in gold]
        relevant += len(gold)
        retrieved += len(ids)
        if hits:
            reciprocal += 1 / hits[0]
        for rank in hits:
            ndcg += 1 / math.log2(rank + 1)
        ideal += sum(1 / math.log2(rank + 1) for rank in range(1, min(len(gold), CUTOFF) + 1))
        group_hits += int(any(set(group).issubset(ids) for group in case["gold_groups"]))
    count = len(cases)
    return {
        "recall_at_k": _ratio(sum(min(1, len(set(str(row.get("evidence_id")) for row in rows) & {str(item) for group in case["gold_groups"] for item in group})) for case, rows in zip(cases, rankings)), count),
        "precision_at_k": _ratio(sum(len(set(str(row.get("evidence_id")) for row in rows) & {str(item) for group in case["gold_groups"] for item in group}) for case, rows in zip(cases, rankings)), retrieved),
        "mrr": _ratio(reciprocal, count),
        "ndcg": _ratio(ndcg, ideal),
        "support_group_recall": _ratio(group_hits, count),
        "relevant_denominator": relevant,
        "retrieved_denominator": retrieved,
        "case_denominator": count,
    }


def _case_counts(cases: list[dict[str, Any]]) -> dict[str, int]:
    return {behavior: sum(row.get("expected_behavior") == behavior for row in cases) for behavior in ("retrieve", "abstain", "clarify")}


def _ratio(value: float, denominator: int | float) -> float | None:
    return round(value / denominator, 6) if denominator else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest_files(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


if __name__ == "__main__":
    raise SystemExit(main())
