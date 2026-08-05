from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import gc
import json
from pathlib import Path

try:
    from scripts.evaluate_meaningful_support import evaluate_rows as evaluate_meaningful_rows
    from scripts.evaluate_meaningful_support import load_artifacts
except ModuleNotFoundError:  # Direct execution places scripts/ on sys.path.
    from evaluate_meaningful_support import evaluate_rows as evaluate_meaningful_rows
    from evaluate_meaningful_support import load_artifacts
from tjipto.runtime.api import handle_request
from tjipto.runtime.service import LegalRuntimeService


ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT / "tests" / "fixtures" / "uud" / "meaningful_support_oracle.json"


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _quote(row: dict, spans: dict[str, dict]) -> str:
    return "\n".join(str(spans[span_id]["exact_quote"]) for span_id in row.get("text_span_ids") or ())


def _expected_rectangles(
    row: dict, characters: dict[str, tuple[dict, dict]]
) -> list[tuple[object, ...]]:
    grouped: dict[str, tuple[dict, list[dict]]] = {}
    for ref in row.get("bbox_refs") or ():
        selected = characters.get(ref)
        if selected is not None:
            word, character = selected
            grouped.setdefault(str(word["word_bbox_id"]), (word, []))[1].append(character)
    return [
        (
            word["page_number"],
            min(character["x0"] for character in selected),
            min(character["y0"] for character in selected),
            max(character["x1"] for character in selected),
            max(character["y1"] for character in selected),
            word.get("page_width"),
            word.get("page_height"),
            "exact",
            True,
        )
        for word, selected in grouped.values()
    ]


def _actual_rectangles(result: dict) -> list[tuple[object, ...]]:
    return [
        (
            rectangle.get("page_number"),
            rectangle.get("x0"),
            rectangle.get("y0"),
            rectangle.get("x1"),
            rectangle.get("y1"),
            rectangle.get("page_width"),
            rectangle.get("page_height"),
            rectangle.get("bbox_precision"),
            rectangle.get("viewer_highlightable"),
        )
        for rectangle in result.get("bbox_rectangles") or ()
    ]


def _reachability_snapshot(rows: list[dict], artifacts: dict[str, list[dict]], service: LegalRuntimeService) -> dict:
    counts: Counter[str] = Counter()
    documents = {row["source_document_id"]: row for row in artifacts["documents"]}
    spans = {row["text_span_id"]: row for row in artifacts["spans"]}
    characters = {
        character["character_bbox_id"]: (word, character)
        for word in artifacts["words"]
        for character in word.get("characters") or ()
    }
    target_ids: list[str] = []
    for row in rows:
        source = documents.get(row.get("source_document_id"))
        counts["unresolved_source_document_count"] += source is None
        if source is not None:
            counts["source_role_mismatch_count"] += source.get("source_role") != row.get("source_role")
            counts["temporal_context_mismatch_count"] += source.get("temporal_context") != row.get("temporal_context")

        if row.get("decision_kind") == "typed_exclusion":
            counts["typed_exclusion_public_leakage_count"] += any(
                row.get(field) is True
                for field in ("answer_eligible", "citation_eligible", "viewer_eligible", "highlight_eligible")
            )
            continue

        target = service.register_public_target("uud", {"support_unit_id": row["support_unit_id"]})
        target_ids.append(target)
        resolved = service.viewer_public("uud", target)
        public = handle_request("uud", "viewer", {"target": target}, service=service)
        counts["unresolved_public_target_count"] += (
            resolved.get("status") != "viewer_payload_ready" or public.get("status") != "viewer_payload_ready"
        )
        counts["wrong_document_count"] += resolved.get("source_document_id") != row["source_document_id"]
        counts["viewer_source_role_mismatch_count"] += resolved.get("source_role") != row["source_role"]
        counts["viewer_temporal_context_mismatch_count"] += resolved.get("temporal_context") != row["temporal_context"]
        counts["wrong_page_count"] += tuple(public.get("page_numbers") or ()) != tuple(row["page_numbers"])
        counts["quote_mismatch_count"] += public.get("quoted_text") != _quote(row, spans)
        if row.get("bbox_precision") == "exact":
            expected = _expected_rectangles(row, characters)
            actual = _actual_rectangles(public)
            counts["unresolved_character_geometry_count"] += any(ref not in characters for ref in row.get("bbox_refs") or ())
            counts["exact_geometry_mismatch_count"] += actual != expected
            counts["exact_highlight_capability_mismatch_count"] += public.get("viewer_highlightable") is not True
        else:
            counts["page_grounded_overlay_leakage_count"] += bool(public.get("bbox_rectangles"))
            counts["page_grounded_highlight_escalation_count"] += public.get("viewer_highlightable") is True
            counts["page_grounded_viewer_loss_count"] += public.get("pdf_access_available") is not True
    counts["duplicate_public_target_count"] = len(target_ids) - len(set(target_ids))
    return {"counters": dict(sorted(counts.items())), "public_target_ids": target_ids}


def evaluate(repo_root: Path = ROOT) -> dict:
    final = repo_root / "data" / "final" / "uud"
    rows = _rows(final / "meaningful_support_units.jsonl")
    artifacts = load_artifacts(final)
    artifacts["documents"] = _rows(final / "source_documents.jsonl")
    oracle = json.loads((repo_root / ORACLE.relative_to(ROOT)).read_text(encoding="utf-8"))
    meaningful = evaluate_meaningful_rows(rows, artifacts, oracle)
    mutation_escapes = 0
    for field, value in (
        ("owner_id", "missing-owner"),
        ("page_numbers", [999]),
        ("quoted_text_sha256", "0" * 64),
        ("authority_kind", "normative_legal_text"),
    ):
        mutated = deepcopy(rows)
        target = next(row for row in mutated if row["decision_kind"] == "canonical_owner_support")
        target[field] = value
        mutation_escapes += evaluate_meaningful_rows(mutated, artifacts, oracle)["status"] != "FAIL"
    mutated = deepcopy(rows)
    target = next(row for row in mutated if row["bbox_precision"] == "exact")
    target["bbox_refs"].append("foreign-character")
    mutation_escapes += evaluate_meaningful_rows(mutated, artifacts, oracle)["status"] != "FAIL"
    reachability_artifacts = {
        name: artifacts[name] for name in ("documents", "spans", "words")
    }
    del artifacts, mutated, oracle
    gc.collect()
    service = LegalRuntimeService(repo_root)
    first = _reachability_snapshot(rows, reachability_artifacts, service)
    second = _reachability_snapshot(rows, reachability_artifacts, service)
    counters = Counter(first["counters"])
    counters["meaningful_support_evaluator_failure_count"] = int(meaningful["status"] != "PASS")
    counters["mutation_escape_count"] = mutation_escapes
    counters["repeat_evaluation_mismatch_count"] = first != second
    failures = sum(counters.values())
    support_rows = [row for row in rows if row.get("decision_kind") != "typed_exclusion"]
    return {
        "status": "PASS" if failures == 0 else "FAIL",
        "decision_count": len(rows),
        "viewer_eligible_count": sum(row.get("viewer_eligible") is True for row in rows),
        "resolved_viewer_count": len(support_rows) - first["counters"].get("unresolved_public_target_count", 0),
        "exact_highlight_count": sum(row.get("bbox_precision") == "exact" for row in support_rows),
        "page_grounded_count": sum(row.get("bbox_precision") == "page_grounded_only" for row in support_rows),
        "typed_exclusion_count": len(rows) - len(support_rows),
        "meaningful_support_status": meaningful["status"],
        "counters": dict(sorted(counters.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate exhaustive meaningful-support reachability.")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = evaluate()
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
