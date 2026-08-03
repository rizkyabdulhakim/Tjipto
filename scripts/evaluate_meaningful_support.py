from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "data" / "final" / "uud"
ORACLE = ROOT / "tests" / "fixtures" / "uud" / "meaningful_support_oracle.json"
REQUIRED_FIELDS = {
    "support_unit_id", "decision_kind", "support_kind", "owner_type", "owner_id",
    "source_document_id", "source_role", "temporal_context", "semantic_classification",
    "legal_force", "authority_kind", "citation_final", "text_span_ids",
    "raw_source_span_ids", "page_numbers", "selector_refs", "bbox_refs", "bbox_precision",
    "quoted_text_sha256", "answer_eligible", "citation_eligible", "viewer_eligible",
    "highlight_eligible", "decision_status", "decision_reason",
}


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_artifacts(final_dir: Path = FINAL) -> dict[str, list[dict]]:
    return {
        name: _rows(final_dir / filename)
        for name, filename in {
            "spans": "page_text_spans.jsonl",
            "raw": "raw_source_spans.jsonl",
            "evidence": "evidence_registry.jsonl",
            "metadata": "metadata_grounding.jsonl",
            "conflicts": "source_conflicts.jsonl",
            "bboxes": "bbox_registry.jsonl",
            "words": "word_bboxes.jsonl",
        }.items()
    }


def evaluate_rows(rows: list[dict], artifacts: dict[str, list[dict]], oracle: dict) -> dict:
    counts: Counter[str] = Counter()
    meaningful_forces = set(oracle["meaningful_legal_forces"])
    spans = {row["text_span_id"]: row for row in artifacts["spans"]}
    meaningful = {key: row for key, row in spans.items() if row["legal_force"] in meaningful_forces}
    raw = {row["raw_source_span_id"]: row for row in artifacts["raw"]}
    evidence = {row["evidence_id"]: row for row in artifacts["evidence"]}
    metadata = {row["metadata_grounding_id"]: row for row in artifacts["metadata"]}
    conflicts = {row["source_conflict_id"]: row for row in artifacts["conflicts"]}
    geometry = {
        row.get("bbox_id") or row.get("word_bbox_id"): row
        for row in (*artifacts["bboxes"], *artifacts["words"])
    }
    seen_units: set[str] = set()
    decisions: Counter[str] = Counter()

    for row in rows:
        missing = REQUIRED_FIELDS - row.keys()
        counts["missing_required_field_count"] += len(missing)
        unit_id = row.get("support_unit_id")
        counts["duplicate_support_unit_id_count"] += unit_id in seen_units
        seen_units.add(unit_id)
        span_rows = [spans.get(span_id) for span_id in row.get("text_span_ids", [])]
        counts["unresolved_span_count"] += sum(span is None for span in span_rows)
        if not span_rows or any(span is None for span in span_rows):
            continue
        resolved_spans = [span for span in span_rows if span is not None]
        for span in resolved_spans:
            decisions[span["text_span_id"]] += 1
        counts["nonmeaningful_span_promoted_count"] += sum(
            span["text_span_id"] not in meaningful for span in resolved_spans
        )
        for field in ("source_document_id", "source_role", "temporal_context", "semantic_classification", "legal_force"):
            values = {span[field] for span in resolved_spans}
            counts[f"cross_{field}_group_count"] += len(values) != 1 or row.get(field) not in values
        pages = sorted({span["page_number"] for span in resolved_spans})
        counts["page_mismatch_count"] += row.get("page_numbers") != pages
        ordered_positions = sorted(
            (span["page_number"], int(span["text_span_id"].rsplit("::", 1)[1])) for span in resolved_spans
        )
        counts["noncontiguous_group_count"] += any(
            current_page != prior_page or current_index != prior_index + 1
            for (prior_page, prior_index), (current_page, current_index) in zip(ordered_positions, ordered_positions[1:])
        )
        expected_hash = sha256("\n".join(span["exact_quote"] for span in resolved_spans).encode()).hexdigest()
        counts["quote_reconstruction_mismatch_count"] += row.get("quoted_text_sha256") != expected_hash

        expected_raw = []
        for span in resolved_spans:
            matches = [
                item["raw_source_span_id"]
                for item in artifacts["raw"]
                if item["source_document_id"] == span["source_document_id"]
                and item["page_number"] == span["page_number"]
                and item.get("semantic_text_start") == span["text_start"]
                and item.get("semantic_text_end") == span["text_end"]
            ]
            counts["raw_span_join_count_mismatch"] += len(matches) != 1
            expected_raw.extend(matches)
        counts["raw_span_reference_mismatch_count"] += row.get("raw_source_span_ids") != expected_raw
        counts["selector_reference_mismatch_count"] += row.get("selector_refs") != expected_raw
        counts["unresolved_selector_count"] += sum(selector not in raw for selector in row.get("selector_refs", []))
        for bbox_id in row.get("bbox_refs", []):
            bbox = geometry.get(bbox_id)
            counts["unresolved_bbox_count"] += bbox is None
            if bbox:
                counts["bbox_source_mismatch_count"] += bbox.get("source_document_id") != row.get("source_document_id")
                counts["bbox_page_mismatch_count"] += bbox.get("page_number") not in pages

        owner_type, owner_id = row.get("owner_type"), row.get("owner_id")
        owner = (
            evidence.get(owner_id) if owner_type == "evidence_registry"
            else metadata.get(owner_id) if owner_type == "metadata_grounding"
            else conflicts.get(owner_id) if owner_type == "source_conflict"
            else spans.get(owner_id) if owner_type == "page_text_span_review"
            else None
        )
        counts["missing_owner_count"] += owner is None
        if owner is None:
            continue
        if owner_type != "page_text_span_review":
            counts["owner_span_mismatch_count"] += any(
                span["text_span_id"] not in (owner.get("text_span_ids") or []) for span in resolved_spans
            )
        expected_authority = owner.get("authority_kind") or "structural_context"
        expected_kind = oracle["authority_support_kinds"].get(expected_authority)
        if owner_type == "page_text_span_review":
            expected_kind = oracle["audited_spans"][owner_id]["support_kind"]
        counts["authority_kind_mismatch_count"] += row.get("authority_kind") != expected_authority
        counts["support_kind_mismatch_count"] += row.get("support_kind") != expected_kind
        owner_final = owner.get("citation_final") is True if owner_type != "page_text_span_review" else False
        counts["fabricated_finality_count"] += row.get("citation_final") is not owner_final
        owner_bbox = set(
            owner.get("raw_provenance_bbox_ids") or owner.get("bbox_refs") or owner.get("bbox_ids") or []
        )
        if owner_type != "page_text_span_review":
            counts["owner_bbox_mismatch_count"] += not set(row.get("bbox_refs", [])) <= owner_bbox
        expected_viewer = bool(row.get("bbox_refs")) and (
            owner_type == "page_text_span_review" or owner.get("viewer_highlightable") is not False
        )
        counts["viewer_eligibility_mismatch_count"] += row.get("viewer_eligible") is not expected_viewer
        counts["highlight_eligibility_mismatch_count"] += row.get("highlight_eligible") is not expected_viewer
        counts["bbox_precision_mismatch_count"] += row.get("bbox_precision") != (
            "exact" if row.get("bbox_refs") else "page_grounded_only"
        )
        expected_answer = expected_authority == "normative_legal_text" and row.get("legal_force") == "canonical_normative"
        expected_citation = owner.get("citable") is True if owner_type == "evidence_registry" else False
        counts["answer_eligibility_mismatch_count"] += row.get("answer_eligible") is not expected_answer
        counts["citation_eligibility_mismatch_count"] += row.get("citation_eligible") is not expected_citation
        counts["legal_force_escalation_count"] += row.get("answer_eligible") is True and row.get("legal_force") != "canonical_normative"
        counts["historical_as_current_count"] += row.get("answer_eligible") is True and row.get("source_role") != "current_consolidated"
        counts["trace_or_metadata_final_count"] += row.get("support_kind") in {"trace", "metadata"} and row.get("citation_final") is True
        counts["furniture_as_legal_count"] += row.get("support_kind") == "document_title" and any(
            row.get(field) is True for field in ("answer_eligible", "citation_eligible", "citation_final")
        )

    missing_decisions = set(meaningful) - set(decisions)
    counts["meaningful_span_without_decision_count"] = len(missing_decisions)
    counts["duplicate_ownership_count"] = sum(value - 1 for value in decisions.values() if value > 1)
    counts["unexpected_meaningful_span_count"] = len(meaningful) != oracle["expected_meaningful_span_count"]
    canonical_owned = sum(
        1 for row in rows if row.get("owner_type") != "page_text_span_review" for _ in row.get("text_span_ids", [])
    )
    counts["canonical_owner_span_count_mismatch"] = canonical_owned != oracle["expected_canonical_owner_span_count"]
    audited: dict[str, bool] = {}
    for span_id, expected in oracle["audited_spans"].items():
        matches = [row for row in rows if span_id in row.get("text_span_ids", [])]
        audited[span_id] = len(matches) == 1 and all(
            matches[0].get(field) == value for field, value in expected.items() if field != "page_number"
        ) and expected["page_number"] in matches[0].get("page_numbers", [])
    counts["audited_decision_mismatch_count"] = sum(not value for value in audited.values())
    failures = sum(counts.values())
    return {
        "status": "PASS" if failures == 0 else "FAIL",
        "support_unit_count": len(rows),
        "meaningful_span_count": len(meaningful),
        "canonical_owner_span_count": canonical_owned,
        "support_kind_unit_counts": dict(sorted(Counter(row.get("support_kind") for row in rows).items())),
        "support_kind_span_counts": dict(sorted(Counter(
            row.get("support_kind") for row in rows for _ in row.get("text_span_ids", [])
        ).items())),
        "counters": dict(sorted(counts.items())),
        "audited_spans": audited,
    }


def evaluate(repo_root: Path = ROOT) -> dict:
    final = repo_root / "data" / "final" / "uud"
    oracle = json.loads((repo_root / ORACLE.relative_to(ROOT)).read_text(encoding="utf-8"))
    return evaluate_rows(_rows(final / "meaningful_support_units.jsonl"), load_artifacts(final), oracle)


def main() -> int:
    parser = argparse.ArgumentParser()
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
