from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

from tjipto.retrieval.dense import DENSE_ALLOWED_MAX_LENGTHS, DENSE_BATCH_SIZE, DENSE_MAX_LENGTH, DENSE_POOLING, MODEL_ID, MODEL_REVISION, DenseIndex, LocalDenseProvider, dense_index_for_store
from tjipto.evidence.store import EvidenceStore
from tjipto.runtime.service import LegalRuntimeService
from tjipto.retrieval.dense_worker import _peak_rss_bytes


ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the pinned BGE-M3 promotion probe and dense comparison.")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--max-length", type=int, choices=DENSE_ALLOWED_MAX_LENGTHS, default=DENSE_MAX_LENGTH)
    parser.add_argument("--batch-size", type=int, default=DENSE_BATCH_SIZE)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--ablation-dir", type=Path)
    parser.add_argument("--selected-max-length", type=int, choices=DENSE_ALLOWED_MAX_LENGTHS)
    parser.add_argument("--ablation-report", action="append", type=Path)
    parser.add_argument("--comparison-report", type=Path)
    parser.add_argument("--runtime-commit")
    parser.add_argument("--runtime-tree")
    parser.add_argument("--identity-sidecar", type=Path)
    parser.add_argument("--index-artifact", type=Path)
    parser.add_argument("--promotion-record", type=Path)
    parser.add_argument("--parity-report", type=Path)
    args = parser.parse_args(argv)
    if args.model_dir:
        os.environ["TJIPTO_DENSE_MODEL_DIR"] = str(args.model_dir.resolve())
    selected_length = args.selected_max_length or args.max_length
    report: dict[str, Any] = {
        "status": "unavailable",
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "pooling": DENSE_POOLING, "max_length": args.max_length},
    }
    try:
        completed = None
        probe = (
            json.loads(args.parity_report.read_text(encoding="utf-8"))
            if args.parity_report and args.parity_report.is_file()
            else _parity_probe(args.timeout, selected_length, args.model_dir)
        )
        if not isinstance(probe, dict) or probe.get("passed") is not True:
            raise RuntimeError("dense_parity_probe_failed")
        report["parity_probe"] = probe
        gc.collect()
        comparison = args.comparison_report or args.report.with_name(args.report.stem + ".comparison.json")
        ablations: dict[str, dict[str, Any]] = {}
        if args.ablation_dir:
            args.ablation_dir.mkdir(parents=True, exist_ok=True)
            for length in DENSE_ALLOWED_MAX_LENGTHS:
                candidate_report = args.ablation_dir / f"dense-{length}.json"
                candidate_index = args.ablation_dir / f"dense-{length}.index"
                _run_evaluation(
                    report=candidate_report,
                    index=candidate_index,
                    timeout=args.timeout,
                    max_length=length,
                    batch_size=args.batch_size,
                    model_dir=args.model_dir,
                    runtime_commit=args.runtime_commit,
                    runtime_tree=args.runtime_tree,
                )
                if candidate_report.is_file():
                    ablations[str(length)] = json.loads(candidate_report.read_text(encoding="utf-8"))
            report["ablations"] = ablations
            selected = _select_candidate(ablations)
            if selected is None:
                raise RuntimeError("dense_quality_contract_failed")
            selected_length, selected_data = selected
            selected_index = Path(str((selected_data.get("dense") or {}).get("artifact_path") or ""))
            if not selected_index.is_file():
                raise RuntimeError("selected_dense_index_missing")
            index_artifact = args.index_artifact or selected_index
            if index_artifact.resolve() != selected_index.resolve():
                shutil.copyfile(selected_index, index_artifact)
            comparison_data = selected_data
        else:
            index_artifact = args.index_artifact or args.report.with_suffix(".dense.index")
            command = [
                sys.executable,
                str(ROOT / "scripts/evaluate_dense_retrieval.py"),
                "--report",
                str(comparison),
                "--timeout",
                str(args.timeout),
                "--max-length",
                str(selected_length),
                "--batch-size",
                str(args.batch_size),
                "--persist-index",
                str(index_artifact),
            ]
            if args.model_dir:
                command.extend(("--model-dir", str(args.model_dir)))
            if args.runtime_commit:
                command.extend(("--runtime-commit", args.runtime_commit))
            if args.runtime_tree:
                command.extend(("--runtime-tree", args.runtime_tree))
            if args.identity_sidecar:
                command.extend(("--identity-sidecar", str(args.identity_sidecar)))
            completed = None
            if not (args.comparison_report and comparison.is_file()):
                completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
            comparison_data = json.loads(comparison.read_text(encoding="utf-8")) if comparison.is_file() else {}
        if comparison_data:
            report["comparison"] = comparison_data
            comparison_data = report.get("comparison")
            comparison_data = comparison_data if isinstance(comparison_data, dict) else {}
            dense = comparison_data.get("dense") or {}
            dense = dense if isinstance(dense, dict) else {}
            parity_data = report.get("parity_probe")
            parity_data = parity_data if isinstance(parity_data, dict) else {}
            report["worker_resource"] = {
                "worker_peak_rss_bytes": dense.get("worker_peak_rss_bytes"),
                "worker_peak_rss_scope": dense.get("worker_peak_rss_scope"),
                "promotion_parent_peak_rss_bytes": parity_data.get("promotion_parent_peak_rss_bytes"),
                "promotion_parent_rss_scope": "promotion_parity_parent_peak_working_set",
                "core_ci_rss_bytes": None,
                "core_ci_rss_scope": "normal_model_free_ci_not_measured_by_dense_promotion",
                "build_seconds": dense.get("build_seconds"),
                "query_seconds": dense.get("query_seconds"),
            }
        if args.ablation_report:
            ablations = {}
            for ablation_path in args.ablation_report:
                ablation = json.loads(ablation_path.read_text(encoding="utf-8"))
                model = (ablation.get("dense") or {}).get("index", {}).get("model", {})
                length = model.get("max_length")
                if length not in DENSE_ALLOWED_MAX_LENGTHS:
                    raise RuntimeError("ablation_report_identity_invalid")
                ablations[str(length)] = ablation
            report["ablations"] = dict(sorted(ablations.items(), key=lambda item: int(item[0])))
        if not index_artifact.is_file():
            _persist_index(index_artifact, args.timeout, selected_length, args.batch_size, args.model_dir)
        report["dense_artifact"] = {
            "path": str(index_artifact),
            "sha256": hashlib.sha256(index_artifact.read_bytes()).hexdigest(),
        }
        service = LegalRuntimeService(ROOT)
        store = service._store("uud")
        if store is None:
            raise RuntimeError("corpus_not_ready")
        dense_store = EvidenceStore(_copy_dense_config(store))
        loaded = DenseIndex.load(
            index_artifact,
            dense_store,
            provider=LocalDenseProvider(timeout_seconds=args.timeout, max_length=selected_length, batch_size=args.batch_size, model_dir=args.model_dir),
        )
        report["dense_artifact"]["identity"] = loaded.identity_record()
        report["activation"] = {"status": "identity_validated", "identity": loaded.identity}
        comparison_data = report.get("comparison")
        comparison_data = comparison_data if isinstance(comparison_data, dict) else {}
        if not probe["passed"] or (completed is not None and completed.returncode) or comparison_data.get("status") != "valid":
            report["reason"] = comparison_data.get("reason") or "dense_comparison_failed"
        else:
            report["status"] = "valid"
            if args.promotion_record:
                promotion = {
                    "schema_version": 2,
                    "specification_kind": "tracked_dense_promotion_specification",
                    "runtime_identity_binding": "post_build_attestation",
                    "status": "promoted",
                    "corpus_id": loaded.corpus_id,
                    "artifact_set_digest": loaded.artifact_set_digest,
                    "manifest_digest": loaded.manifest_digest,
                    "index_sha256": hashlib.sha256(index_artifact.read_bytes()).hexdigest(),
                    "index_identity": loaded.identity,
                    "index_identity_record": loaded.identity_record(),
                    "model": loaded.model_identity.as_dict(),
                    "selected_max_length": selected_length,
                    "ablations": _without_runtime_identity(report.get("ablations", {})),
                }
                args.promotion_record.parent.mkdir(parents=True, exist_ok=True)
                args.promotion_record.write_text(json.dumps(promotion, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        report["reason"] = f"{type(error).__name__}:{error}"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "reason": report.get("reason")}, sort_keys=True))
    return 0 if report["status"] == "valid" else 2


def _copy_dense_config(store):
    from dataclasses import replace

    settings = dict(store.config.settings or {})
    settings.pop("dense_index_path", None)
    settings.pop("dense_promotion_path", None)
    return replace(store.config, manifest=dict(store.config.manifest) | {"dense_retrieval": True}, settings=settings)


def _persist_index(path: Path, timeout: float, max_length: int, batch_size: int, model_dir: Path | None) -> None:
    service = LegalRuntimeService(ROOT)
    store = service._store("uud")
    if store is None:
        raise RuntimeError("corpus_not_ready")
    dense_store = EvidenceStore(_copy_dense_config(store))
    provider = LocalDenseProvider(timeout_seconds=timeout, max_length=max_length, batch_size=batch_size, model_dir=model_dir)
    index = dense_index_for_store(dense_store, provider=provider)
    index.persist(path)


def _parity_probe(timeout: float, max_length: int, model_dir: Path | None) -> dict[str, object]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    text = "Ketentuan hukum mengenai kewenangan Presiden."
    provider = LocalDenseProvider(timeout_seconds=timeout, max_length=max_length, model_dir=model_dir)
    observed_batch = provider.embed((text,))
    observed = observed_batch.vectors[0]
    cache_dir = os.environ.get("TJIPTO_DENSE_MODEL_DIR")
    local_snapshot = Path(cache_dir) if cache_dir and (Path(cache_dir) / "config.json").exists() else None
    if local_snapshot is not None and local_snapshot.name != MODEL_REVISION:
        raise ValueError("noncanonical_model_snapshot")
    if local_snapshot is None:
        raise ValueError("model_snapshot_unavailable")
    source = str(local_snapshot)
    kwargs = {"trust_remote_code": False, "local_files_only": True}
    tokenizer = AutoTokenizer.from_pretrained(source, revision=MODEL_REVISION, **kwargs)  # nosec B615 - pinned local probe
    model = AutoModel.from_pretrained(source, revision=MODEL_REVISION, **kwargs)  # nosec B615 - pinned local probe
    model.eval()
    inputs = tokenizer([text], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    with torch.no_grad():
        expected = torch.nn.functional.normalize(model(**inputs).last_hidden_state[:, 0, :], p=2, dim=1)[0]
    error = max(abs(float(left) - float(right)) for left, right in zip(observed, expected.tolist()))
    return {
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "max_abs_error": error,
        "max_length": max_length,
        "passed": error <= 1e-5,
        "vector_sha256": hashlib.sha256(json.dumps(observed, separators=(",", ":")).encode("utf-8")).hexdigest(),
        "worker_peak_rss_bytes": observed_batch.worker_peak_rss_bytes,
        "worker_peak_rss_scope": "embedding_worker_peak_working_set",
        "promotion_parent_peak_rss_bytes": _peak_rss_bytes(),
        "promotion_parent_rss_scope": "promotion_parity_parent_peak_working_set",
    }


def _run_evaluation(*, report: Path, index: Path, timeout: float, max_length: int, batch_size: int, model_dir: Path | None, runtime_commit: str | None, runtime_tree: str | None) -> None:
    command = [
        sys.executable,
        str(ROOT / "scripts/evaluate_dense_retrieval.py"),
        "--report", str(report),
        "--timeout", str(timeout),
        "--max-length", str(max_length),
        "--batch-size", str(batch_size),
        "--persist-index", str(index),
    ]
    if model_dir:
        command.extend(("--model-dir", str(model_dir)))
    if runtime_commit:
        command.extend(("--runtime-commit", runtime_commit))
    if runtime_tree:
        command.extend(("--runtime-tree", runtime_tree))
    subprocess.run(command, cwd=ROOT, check=False)


def _select_candidate(ablations: dict[str, dict[str, Any]]) -> tuple[int, dict[str, Any]] | None:
    # The frozen gate protects recall and complete support groups.  MRR/NDCG
    # remain recorded diagnostics because rank-only fusion intentionally may
    # change ordering while preserving the evidence set.
    required_metrics = ("hit_rate_at_k", "support_group_recall_at_k", "recall_at_k")
    valid: list[tuple[int, dict[str, Any]]] = []
    for raw_length, candidate in ablations.items():
        dense = candidate.get("dense") or {}
        production = (candidate.get("metrics") or {}).get("production") or {}
        hybrid = dense.get("hybrid_metrics") or {}
        if candidate.get("status") != "valid" or not all(
            isinstance(hybrid.get(metric), (int, float))
            and isinstance(production.get(metric), (int, float))
            and hybrid[metric] >= production[metric]
            for metric in required_metrics
        ):
            continue
        valid.append((int(raw_length), candidate))
    return min(valid, key=lambda item: item[0]) if valid else None


def _without_runtime_identity(value: Any) -> Any:
    """Keep Git identity in the post-build attestation, never in the spec."""
    if isinstance(value, dict):
        return {
            key: _without_runtime_identity(item)
            for key, item in value.items()
            if key not in {"runtime_commit", "runtime_tree_sha"}
        }
    if isinstance(value, list):
        return [_without_runtime_identity(item) for item in value]
    return value


if __name__ == "__main__":
    raise SystemExit(main())
