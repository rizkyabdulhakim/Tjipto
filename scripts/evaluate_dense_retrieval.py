from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from tjipto.evidence.store import EvidenceStore
from tjipto.retrieval.dense import DENSE_ALLOWED_MAX_LENGTHS, DenseIndex, DenseUnavailable, LocalDenseProvider, dense_index_for_store
from tjipto.runtime.service import LegalRuntimeService


ROOT = Path(__file__).resolve().parents[1]
RETRIEVAL_CASES = ROOT / "tests/fixtures/uud/retrieval_cases.jsonl"
RESEARCH_CASES = ROOT / "tests/fixtures/uud/research_retrieval_cases.jsonl"
CUTOFF = 10


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare production retrieval and the pinned BGE-M3 lane.")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-length", type=int, choices=DENSE_ALLOWED_MAX_LENGTHS, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--query-batch-size", type=int, default=35)
    parser.add_argument("--runtime-commit")
    parser.add_argument("--runtime-tree")
    parser.add_argument("--identity-sidecar", type=Path)
    parser.add_argument("--persist-index", type=Path)
    parser.add_argument("--load-index", type=Path)
    parser.add_argument("--build-evidence", type=Path)
    args = parser.parse_args(argv)
    case_paths = (RETRIEVAL_CASES, RESEARCH_CASES)
    all_cases = [row for path in case_paths for row in _read_jsonl(path)]
    cases = _eligible_cases(all_cases)
    identity = {
        "case_set_sha256": _digest_files(case_paths),
        "evaluator_sha256": _sha256(Path(__file__)),
        "case_count": len(all_cases),
        "dense_eligible_case_count": len(cases),
        "dense_eligible_case_ids_sha256": _digest_value([case["id"] for case in cases]),
        "dense_eligible_queries_sha256": _digest_value([case["query"] for case in cases]),
        "cutoff": CUTOFF,
        **_runtime_identity(args),
    }
    report: dict[str, Any] = {
        "status": "unavailable",
        "execution_status": "unavailable",
        "identity": identity,
        "case_counts": _case_counts(all_cases),
        "metrics": {"production": None, "dense": None, "delta": None},
    }
    if identity["runtime_commit"] == "unavailable" or identity["runtime_tree_sha"] == "unavailable":
        report["reason"] = "runtime_identity_unavailable"
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"status": report["status"], "case_count": len(cases), "reason": report["reason"]}, sort_keys=True))
        return 2
    try:
        service = LegalRuntimeService(ROOT)
        store = service._store("uud")
        if store is None:
            raise DenseUnavailable("corpus_not_ready")
        # The baseline lane is the current model-free production route.  An
        # explicitly configured dense model is measured below as a separate
        # lane, never accidentally folded into the baseline by environment.
        configured_model_dir = os.environ.pop("TJIPTO_DENSE_MODEL_DIR", None)
        try:
            production_rankings = [
                list(service._route_retrieval("uud", case["query"], store, limit=CUTOFF).get("matches", ()))
                for case in cases
            ]
        finally:
            if configured_model_dir is not None:
                os.environ["TJIPTO_DENSE_MODEL_DIR"] = configured_model_dir
        report["metrics"]["production"] = _metrics(cases, production_rankings)
        config = _copy_dense_config(store)
        dense_store = EvidenceStore(config)
        provider = LocalDenseProvider(
            timeout_seconds=args.timeout,
            max_length=args.max_length,
            batch_size=args.batch_size,
            model_dir=args.model_dir,
        )
        started = time.perf_counter()
        index = DenseIndex.load(args.load_index, dense_store, provider=provider) if args.load_index else dense_index_for_store(dense_store, provider=provider)
        if args.persist_index and not args.load_index:
            index.persist(args.persist_index)
        build_seconds = time.perf_counter() - started
        build_evidence = None
        if args.build_evidence:
            build_evidence = json.loads(args.build_evidence.read_text(encoding="utf-8"))
            build_dense = build_evidence.get("dense") or {}
            build_identity = (build_dense.get("index") or {}).get("identity")
            if build_identity != index.identity or build_evidence.get("status") != "valid":
                raise DenseUnavailable("dense_build_evidence_identity_mismatch")
            build_seconds = float(build_dense.get("build_seconds"))
        query_started = time.perf_counter()
        query_provider = LocalDenseProvider(
            timeout_seconds=args.timeout,
            max_length=args.max_length,
            batch_size=args.query_batch_size,
            model_dir=args.model_dir,
        )
        query_batch = query_provider.embed(tuple(case["query"] for case in cases))
        dense_rankings = [index.search(vector, CUTOFF) for vector in query_batch.vectors]
        dense_metrics = _metrics(cases, dense_rankings)
        from tjipto.retrieval.bm25 import sparse_index_for_store
        from tjipto.retrieval.hybrid import normalize_hits, reciprocal_rank_fusion

        hybrid_rankings = []
        for case, vector in zip(cases, query_batch.vectors):
            sparse = sparse_index_for_store(dense_store).search(case["query"], CUTOFF)
            dense = index.search(vector, CUTOFF)
            fused = reciprocal_rank_fusion(
                {"bm25": normalize_hits(sparse, "bm25"), "dense": normalize_hits(dense, "dense")},
                limit=CUTOFF,
            )
            hybrid_rankings.append([hit.row | {"evidence_id": hit.evidence_id} for hit in fused])
        hybrid_metrics = _metrics(cases, hybrid_rankings)
        report.update(
            {
                "status": "valid",
                "execution_status": "valid",
                "dense": {
                    "metrics": dense_metrics,
                    "hybrid_metrics": hybrid_metrics,
                    "index": index.identity_record(),
                    "build_seconds": round(build_seconds, 3),
                    "build_evidence_path": str(args.build_evidence) if args.build_evidence else None,
                    "query_seconds": round(time.perf_counter() - query_started, 3),
                    "index_size_bytes": (
                        (args.load_index or args.persist_index).stat().st_size
                        if (args.load_index or args.persist_index) and (args.load_index or args.persist_index).is_file()
                        else None
                    ),
                    "truncation_count": index.truncation_count + len(query_batch.truncated_indices),
                    "truncated_retrieval_unit_ids": list(index.truncated_retrieval_unit_ids),
                    "query_truncated_case_ids": [
                        cases[index]["id"] for index in query_batch.truncated_indices if index < len(cases)
                    ],
                    "query_truncation_count": len(query_batch.truncated_indices),
                    "worker_peak_rss_bytes": max(
                        value
                        for value in (index.worker_peak_rss_bytes, query_batch.worker_peak_rss_bytes)
                        if value is not None
                    )
                    if any(value is not None for value in (index.worker_peak_rss_bytes, query_batch.worker_peak_rss_bytes))
                    else None,
                    "worker_peak_rss_scope": "embedding_worker_peak_working_set",
                    "artifact_path": str(args.load_index or args.persist_index) if args.load_index or args.persist_index else None,
                    "artifact_sha256": hashlib.sha256((args.load_index or args.persist_index).read_bytes()).hexdigest() if (args.load_index or args.persist_index) and (args.load_index or args.persist_index).is_file() else None,
                },
                "production_baseline": {
                    "runtime_commit": identity["runtime_commit"],
                    "runtime_tree_sha": identity["runtime_tree_sha"],
                    "case_set_sha256": identity["case_set_sha256"],
                    "cutoff": CUTOFF,
                    "metrics": report["metrics"]["production"],
                },
            }
        )
        report["metrics"]["dense"] = dense_metrics
        report["metrics"]["delta"] = _metric_delta(report["metrics"]["production"], dense_metrics)
    except DenseUnavailable as error:
        report["reason"] = error.code
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        report["status"] = "invalid"
        report["execution_status"] = "invalid"
        report["reason"] = f"{type(error).__name__}:{error}"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "case_count": len(cases),
                "reason": report.get("reason"),
                "production": report["metrics"].get("production"),
                "dense": report["metrics"].get("dense"),
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "valid" else 2


def _copy_dense_config(store):
    from dataclasses import replace

    settings = dict(store.config.settings or {})
    settings.pop("dense_index_path", None)
    settings.pop("dense_promotion_path", None)
    return replace(store.config, manifest=dict(store.config.manifest) | {"dense_retrieval": True}, settings=settings)


def _eligible_cases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for row in rows:
        if row.get("expected_behavior") != "retrieve":
            continue
        groups = row.get("gold_support_groups") or []
        if not groups and row.get("gold_support_ids"):
            groups = [[support_id] for support_id in row["gold_support_ids"]]
        if groups:
            cases.append({"id": row.get("id") or row.get("case_id"), "query": row["query"], "gold_groups": groups})
    return cases


def _metrics(cases: list[dict[str, Any]], rankings: list[list[dict]]) -> dict[str, float | int | None]:
    case_count = len(cases)
    relevant_total = 0
    hit_cases = 0
    reciprocal = 0.0
    ndcg_total = 0.0
    group_hits = 0
    relevant_hits = 0
    retrieved_total = 0
    for case, rows in zip(cases, rankings):
        ids = [str(row.get("evidence_id")) for row in rows[:CUTOFF]]
        gold_groups = [set(str(item) for item in group) for group in case["gold_groups"]]
        gold = set().union(*gold_groups)
        hits = [rank for rank, evidence_id in enumerate(ids, 1) if evidence_id in gold]
        relevant_total += len(gold)
        relevant_hits += len(set(ids) & gold)
        retrieved_total += len(ids)
        hit_cases += int(bool(hits))
        if hits:
            reciprocal += 1 / hits[0]
        group_hits += int(any(group <= set(ids) for group in gold_groups))
        dcg = sum(1 / math.log2(rank + 1) for rank in hits)
        idcg = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(gold), CUTOFF) + 1))
        ndcg_total += dcg / idcg if idcg else 0.0
    return {
        "hit_rate_at_k": _ratio(hit_cases, case_count),
        "mrr": _ratio(reciprocal, case_count),
        "ndcg_at_k": _ratio(ndcg_total, case_count),
        "support_group_recall_at_k": _ratio(group_hits, case_count),
        "recall_at_k": _ratio(relevant_hits, relevant_total),
        "precision_at_k": _ratio(relevant_hits, retrieved_total),
        "case_denominator": case_count,
        "relevant_item_denominator": relevant_total,
        "retrieved_item_denominator": retrieved_total,
        "support_group_denominator": case_count,
    }


def _metric_delta(base: dict[str, Any] | None, dense: dict[str, Any] | None) -> dict[str, float | None] | None:
    if base is None or dense is None:
        return None
    return {
        key: round(float(dense[key]) - float(base[key]), 6)
        if isinstance(base.get(key), (int, float)) and isinstance(dense.get(key), (int, float))
        else None
        for key in ("hit_rate_at_k", "mrr", "ndcg_at_k", "support_group_recall_at_k", "recall_at_k", "precision_at_k")
    }


def _case_counts(cases: list[dict[str, Any]]) -> dict[str, int]:
    return {behavior: sum(row.get("expected_behavior") == behavior for row in cases) for behavior in ("retrieve", "abstain", "clarify")}


def _ratio(value: float, denominator: int | float) -> float | None:
    return round(value / denominator, 6) if denominator else None


def _runtime_identity(args: argparse.Namespace) -> dict[str, str]:
    sidecar: dict[str, Any] = {}
    if args.identity_sidecar and args.identity_sidecar.is_file():
        try:
            loaded = json.loads(args.identity_sidecar.read_text(encoding="utf-8"))
            sidecar = loaded if isinstance(loaded, dict) else {}
        except (OSError, json.JSONDecodeError):
            sidecar = {}
    commit = args.runtime_commit or sidecar.get("commit_sha") or sidecar.get("runtime_commit")
    tree = args.runtime_tree or sidecar.get("tree_sha") or sidecar.get("runtime_tree_sha")
    if not commit:
        commit = _git_optional("rev-parse", "HEAD")
    if not tree and commit not in {None, "unavailable"}:
        tree = _git_optional("rev-parse", f"{commit}^{{tree}}")
    return {"runtime_commit": commit or "unavailable", "runtime_tree_sha": tree or "unavailable"}


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


def _digest_value(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _git_optional(*args: str) -> str:
    try:
        return subprocess.check_output(("git", *args), cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


if __name__ == "__main__":
    raise SystemExit(main())
