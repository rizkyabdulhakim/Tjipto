from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from tjipto.retrieval.dense import DENSE_MAX_LENGTH, DENSE_POOLING, MODEL_ID, MODEL_REVISION, LocalDenseProvider
from tjipto.retrieval.dense_worker import _peak_rss_bytes


ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the pinned BGE-M3 promotion probe and dense comparison.")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--comparison-report", type=Path)
    parser.add_argument("--runtime-commit")
    parser.add_argument("--runtime-tree")
    parser.add_argument("--identity-sidecar", type=Path)
    args = parser.parse_args(argv)
    report = {
        "status": "unavailable",
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "pooling": DENSE_POOLING, "max_length": DENSE_MAX_LENGTH},
    }
    try:
        probe = _parity_probe(args.timeout)
        report["parity_probe"] = probe
        gc.collect()
        comparison = args.comparison_report or args.report.with_name(args.report.stem + ".comparison.json")
        if args.comparison_report is None:
            command = [
                sys.executable,
                str(ROOT / "scripts/evaluate_dense_retrieval.py"),
                "--report",
                str(comparison),
                "--timeout",
                str(args.timeout),
            ]
            if args.runtime_commit:
                command.extend(("--runtime-commit", args.runtime_commit))
            if args.runtime_tree:
                command.extend(("--runtime-tree", args.runtime_tree))
            if args.identity_sidecar:
                command.extend(("--identity-sidecar", str(args.identity_sidecar)))
            completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
        else:
            completed = None
        if comparison.is_file():
            report["comparison"] = json.loads(comparison.read_text(encoding="utf-8"))
            dense = report.get("comparison", {}).get("dense") or {}
            report["worker_resource"] = {
                "worker_peak_rss_bytes": dense.get("worker_peak_rss_bytes"),
                "worker_peak_rss_scope": dense.get("worker_peak_rss_scope"),
                "promotion_parent_peak_rss_bytes": report.get("parity_probe", {}).get("promotion_parent_peak_rss_bytes"),
                "promotion_parent_rss_scope": "promotion_parity_parent_peak_working_set",
                "core_ci_rss_bytes": None,
                "core_ci_rss_scope": "normal_model_free_ci_not_measured_by_dense_promotion",
                "build_seconds": dense.get("build_seconds"),
                "query_seconds": dense.get("query_seconds"),
            }
        if not probe["passed"] or (completed is not None and completed.returncode) or report.get("comparison", {}).get("status") != "valid":
            report["reason"] = report.get("comparison", {}).get("reason") or "dense_comparison_failed"
        else:
            report["status"] = "valid"
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        report["reason"] = f"{type(error).__name__}:{error}"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "reason": report.get("reason")}, sort_keys=True))
    return 0 if report["status"] == "valid" else 2


def _parity_probe(timeout: float) -> dict[str, object]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    text = "Ketentuan hukum mengenai kewenangan Presiden."
    provider = LocalDenseProvider(timeout_seconds=timeout)
    observed_batch = provider.embed((text,))
    observed = observed_batch.vectors[0]
    cache_dir = os.environ.get("TJIPTO_DENSE_MODEL_DIR")
    local_snapshot = Path(cache_dir) if cache_dir and (Path(cache_dir) / "config.json").exists() else None
    if local_snapshot is not None and local_snapshot.name != MODEL_REVISION:
        raise ValueError("noncanonical_model_snapshot")
    source = str(local_snapshot) if local_snapshot is not None else MODEL_ID
    kwargs = {"trust_remote_code": False, "local_files_only": True}
    tokenizer = AutoTokenizer.from_pretrained(source, revision=MODEL_REVISION, **kwargs)  # nosec B615 - pinned local probe
    model = AutoModel.from_pretrained(source, revision=MODEL_REVISION, **kwargs)  # nosec B615 - pinned local probe
    model.eval()
    inputs = tokenizer([text], padding=True, truncation=True, max_length=DENSE_MAX_LENGTH, return_tensors="pt")
    with torch.no_grad():
        expected = torch.nn.functional.normalize(model(**inputs).last_hidden_state[:, 0, :], p=2, dim=1)[0]
    error = max(abs(float(left) - float(right)) for left, right in zip(observed, expected.tolist()))
    return {
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "max_abs_error": error,
        "passed": error <= 1e-5,
        "vector_sha256": hashlib.sha256(json.dumps(observed, separators=(",", ":")).encode("utf-8")).hexdigest(),
        "worker_peak_rss_bytes": observed_batch.worker_peak_rss_bytes,
        "worker_peak_rss_scope": "embedding_worker_peak_working_set",
        "promotion_parent_peak_rss_bytes": _peak_rss_bytes(),
        "promotion_parent_rss_scope": "promotion_parity_parent_peak_working_set",
    }


if __name__ == "__main__":
    raise SystemExit(main())
