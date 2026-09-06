"""Verify the configured deployment planner without recording credentials or prompts."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess

from tjipto.retrieval.research import research_planning_provider_from_environment
from tjipto.runtime.service import LegalRuntimeService


ROOT = Path(__file__).resolve().parents[1]
FAST_QUERY = "Pasal 7A bunyinya apa?"
SEMANTIC_QUERY = "Saya dilarang sekolah karena agama saya, hak konstitusional apa yang relevan?"


def _runtime_identity() -> dict[str, str]:
    return {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "tree": subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=ROOT, text=True).strip(),
    }


class _CountingProvider:
    def __init__(self, delegate):
        self.delegate = delegate
        self.calls = 0
        self.request_sha256: list[str] = []

    def propose(self, request):
        encoded = json.dumps(dict(request), ensure_ascii=False, sort_keys=True).encode("utf-8")
        self.request_sha256.append(sha256(encoded).hexdigest())
        self.calls += 1
        return self.delegate.propose(request)


def _plan_summary(plan) -> dict:
    return {
        "original_query": getattr(plan, "original_query", None),
        "provider_status": getattr(plan, "provider_status", None),
        "variant_count": len(getattr(plan, "variants", ()) or ()),
        "information_need_count": len(getattr(plan, "information_needs", ()) or ()),
        "rejection_reasons": tuple(getattr(plan, "rejection_reasons", ()) or ()),
        "query_variants": tuple(getattr(variant, "query", "") for variant in getattr(plan, "variants", ()) or ()),
    }


def verify() -> dict:
    configured = research_planning_provider_from_environment()
    if configured is None:
        return {
            "schema_version": 1,
            "kind": "live_planner_integration",
            "status": "invalid",
            "reason": "deployment_planner_not_configured",
            "runtime_identity": _runtime_identity(),
            "raw_credentials_logged": False,
        }
    provider = _CountingProvider(configured)
    service = LegalRuntimeService(ROOT, planning_provider=provider)
    fast = service.ask("uud", FAST_QUERY)
    fast_calls = provider.calls
    semantic = service.ask("uud", SEMANTIC_QUERY)
    plan = semantic.get("research_plan")
    summary = _plan_summary(plan) if plan is not None else {}
    forbidden = {
        "evidence_id", "citation", "authority", "authority_kind", "source_role",
        "current_law", "sufficiency", "support_id", "support_ids",
    }
    plan_text = json.dumps(summary, ensure_ascii=False, sort_keys=True).casefold()
    valid = (
        fast_calls == 0
        and provider.calls == 1
        and summary.get("original_query") == SEMANTIC_QUERY
        and 0 < summary.get("variant_count", 0) <= 4
        and summary.get("information_need_count", 0) <= 3
        and summary.get("provider_status") == "accepted"
        and not summary.get("rejection_reasons")
        and not any(token in plan_text for token in forbidden)
        and semantic.get("original_query") == SEMANTIC_QUERY
    )
    return {
        "schema_version": 1,
        "kind": "live_planner_integration",
        "status": "valid" if valid else "invalid",
        "runtime_identity": _runtime_identity(),
        "provider": type(configured).__name__,
        "fast_path": {"query_sha256": sha256(FAST_QUERY.encode()).hexdigest(), "provider_calls": fast_calls, "status": fast.get("status")},
        "semantic_path": {
            "query_sha256": sha256(SEMANTIC_QUERY.encode()).hexdigest(),
            "provider_calls": provider.calls,
            "plan": summary,
            "terminal_status": semantic.get("status"),
        },
        "request_sha256": tuple(provider.request_sha256),
        "raw_credentials_logged": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    report = verify()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "kind": report["kind"]}, sort_keys=True))
    return 0 if report["status"] == "valid" else 2


if __name__ == "__main__":
    raise SystemExit(main())
