"""Create exact-head evidence for the promoted dense runtime and RRF fusion."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess

from tjipto.retrieval.dense import LocalDenseProvider, dense_configured, dense_index_for_store, dense_runtime_available
from tjipto.runtime.service import LegalRuntimeService


ROOT = Path(__file__).resolve().parents[1]
PROMOTION = ROOT / "data" / "dense" / "uud" / "promotion.json"


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _git_blob_sha256(commit: str, relative_path: str) -> str:
    blob = subprocess.check_output(
        ["git", "show", f"{commit}:{relative_path}"], cwd=ROOT
    )
    return sha256(blob).hexdigest()


def _index_path(store, promotion: dict) -> Path:
    configured = store.config.setting("dense_index_path", None)
    if configured:
        path = Path(configured)
        return path if path.is_absolute() else ROOT / path
    return ROOT / "data" / "dense" / "uud" / "bge-m3-256.index"


def _ablation_digest(promotion: dict) -> str:
    payload = json.dumps(promotion.get("ablations", {}), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


def _hybrid_activation_valid(activation: object) -> bool:
    if not isinstance(activation, dict):
        return False
    fusion = activation.get("fusion")
    if not isinstance(fusion, dict):
        return False
    counts = fusion.get("lane_candidate_counts")
    lanes = activation.get("contributing_lanes")
    return (
        activation.get("dense_configured") is True
        and activation.get("dense_runtime_available") is True
        and activation.get("hybrid_active") is True
        and activation.get("route") == "hybrid"
        and fusion.get("algorithm") == "rrf_rank_only"
        and isinstance(counts, dict)
        and type(counts.get("bm25")) is int
        and counts["bm25"] > 0
        and type(counts.get("dense")) is int
        and counts["dense"] > 0
        and isinstance(lanes, (list, tuple))
        and {"bm25", "dense"} <= set(lanes)
    )


def attest(query: str) -> dict:
    commit = _git("rev-parse", "HEAD")
    tree = _git("rev-parse", "HEAD^{tree}")
    promotion = json.loads(PROMOTION.read_text(encoding="utf-8"))
    service = LegalRuntimeService(ROOT)
    store = service._store("uud")
    model_dir = service._store("uud").config.setting("dense_model_path", None)
    provider = LocalDenseProvider(model_dir=model_dir)
    index = dense_index_for_store(store, provider=provider)
    route = service._route_retrieval(
        "uud",
        query,
        store,
        route="hybrid",
        limit=10,
        dense_provider=LocalDenseProvider(model_dir=model_dir, timeout_seconds=600.0),
    )
    fusion = route.get("fusion") or {}
    lanes = tuple(str(value) for value in fusion.get("contributing_lanes", ()) if isinstance(value, str))
    index_path = _index_path(store, promotion)
    configured = dense_configured(store)
    runtime_available = dense_runtime_available(store)
    activation = {
        "dense_configured": configured,
        "dense_runtime_available": runtime_available,
        "hybrid_active": bool(route.get("hybrid_active")),
        "route": route.get("route"),
        "reason": route.get("reason") or route.get("retrieval_degraded_reason"),
        "fusion": fusion,
        "contributing_lanes": lanes,
    }
    valid = (
        promotion.get("specification_kind") == "tracked_dense_promotion_specification"
        and not any(key in promotion for key in ("runtime_commit", "runtime_tree_sha"))
        and _hybrid_activation_valid(activation)
    )
    return {
        "schema_version": 1,
        "kind": "post_build_dense_runtime_attestation",
        "status": "valid" if valid else "invalid",
        "runtime_identity": {"commit": commit, "tree": tree},
        "promotion_spec": {
            "path": str(PROMOTION.relative_to(ROOT)).replace("\\", "/"),
            "sha256": _git_blob_sha256(commit, str(PROMOTION.relative_to(ROOT)).replace("\\", "/")),
            "schema_version": promotion.get("schema_version"),
            "specification_kind": promotion.get("specification_kind"),
            "ablation_evidence_sha256": _ablation_digest(promotion),
            "ablation_provenance": "digest_equivalent_reuse",
        },
        "index": {
            "path": str(index_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": _sha256(index_path),
            "identity": index.identity,
            "record_count": index.record_count,
            "dimension": index.dimension,
            "normalization": index.model_identity.normalization,
        },
        "model": index.model_identity.as_dict(),
        "activation": activation,
        "query_sha256": sha256(query.encode("utf-8")).hexdigest(),
        "raw_credentials_logged": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--query", default="Saya dilarang sekolah karena agama saya, hak konstitusional apa yang relevan?")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    if args.check_only:
        try:
            report = json.loads(args.report.read_text(encoding="utf-8"))
            if (
                report.get("schema_version") != 1
                or report.get("kind") != "post_build_dense_runtime_attestation"
                or report.get("status") != "valid"
                or report.get("runtime_identity") != {
                    "commit": _git("rev-parse", "HEAD"),
                    "tree": _git("rev-parse", "HEAD^{tree}"),
                }
                or not _hybrid_activation_valid(report.get("activation"))
            ):
                raise ValueError("dense_hybrid_activation_invalid")
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError, subprocess.CalledProcessError) as error:
            report = {"schema_version": 1, "kind": "post_build_dense_runtime_attestation", "status": "invalid", "reason": f"{type(error).__name__}:{error}"}
    else:
        try:
            report = attest(args.query)
        except (OSError, KeyError, TypeError, ValueError, subprocess.CalledProcessError) as error:
            report = {"schema_version": 1, "kind": "post_build_dense_runtime_attestation", "status": "invalid", "reason": f"{type(error).__name__}:{error}"}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "kind": report["kind"]}, sort_keys=True))
    return 0 if report["status"] == "valid" else 2


if __name__ == "__main__":
    raise SystemExit(main())
