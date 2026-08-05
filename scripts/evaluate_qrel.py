from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path

from tjipto.core.manifest import artifact_set_digest


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QREL = ROOT / "evaluation" / "uud" / "qrel_v0.jsonl"
SCHEMA = ROOT / "evaluation" / "uud" / "qrel_v0.schema.json"
REQUIRED_FIELDS = frozenset((
    "case_id", "case_kind", "query", "intent", "answerability", "gold_support_ids",
    "alternative_valid_support_ids", "forbidden_support_ids", "minimal_relevant_spans",
    "required_claims", "permitted_partial_claims", "source_role", "temporal_scope",
    "authority_kind", "citation_finality", "expected_public_targets",
    "expected_recovery_behavior", "split", "review_status", "reviewer_role", "reviewed_at",
    "corpus_identity", "coverage_tags",
))
LABEL_FIELDS = frozenset((
    "gold_support_ids", "alternative_valid_support_ids", "forbidden_support_ids", "minimal_relevant_spans",
    "required_claims", "permitted_partial_claims", "answerability", "citation_finality", "expected_public_targets",
    "expected_recovery_behavior", "review_status", "reviewer_role", "reviewed_at", "corpus_identity",
))


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _identity(final: Path) -> dict[str, str]:
    manifest_path = final / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    return {
        "corpus_id": str(manifest["corpus_id"]),
        "artifact_set_sha256": artifact_set_digest(manifest),
        "manifest_sha256": sha256(manifest_bytes).hexdigest(),
    }


def _metrics(qrels: list[dict], predictions: list[dict], k: int) -> dict[str, float | None]:
    by_id = {row.get("case_id"): row for row in predictions}
    answerable = [row for row in qrels if row["answerability"] == "answerable"]
    retrieval: list[tuple[dict, dict, list[str], set[str]]] = []
    for row in answerable:
        prediction = by_id.get(row["case_id"], {})
        ranked = [str(value) for value in prediction.get("retrieved_support_ids") or ()][:k]
        relevant = set(row["gold_support_ids"]) | set(row["alternative_valid_support_ids"])
        retrieval.append((row, prediction, ranked, relevant))

    def average(values: list[float]) -> float | None:
        return round(sum(values) / len(values), 6) if values else None

    recall = average([len(set(ranked) & relevant) / len(relevant) for _, _, ranked, relevant in retrieval])
    precision = average([len(set(ranked) & relevant) / max(1, len(ranked)) for _, _, ranked, relevant in retrieval])
    reciprocal = average([
        next((1 / (index + 1) for index, support_id in enumerate(ranked) if support_id in relevant), 0.0)
        for _, _, ranked, relevant in retrieval
    ])
    ndcg = average([
        sum((1 / math.log2(index + 2)) for index, support_id in enumerate(ranked) if support_id in relevant)
        / sum(1 / math.log2(index + 2) for index in range(min(len(relevant), k)))
        for _, _, ranked, relevant in retrieval
    ])

    minimal_expected = {
        (row["case_id"], span_id)
        for row in answerable
        for item in row["minimal_relevant_spans"]
        for span_id in item["text_span_ids"]
    }
    minimal_actual = {
        (row["case_id"], str(span_id))
        for row in answerable
        for span_id in by_id.get(row["case_id"], {}).get("minimal_span_ids") or ()
    }
    minimal_true = len(minimal_expected & minimal_actual)
    highlighted = {
        (row["case_id"], str(span_id))
        for row in answerable
        for span_id in by_id.get(row["case_id"], {}).get("highlighted_span_ids") or ()
    }
    predicted_targets = [
        (row, target)
        for row in answerable
        for target in by_id.get(row["case_id"], {}).get("public_targets") or ()
    ]
    wrong_pages = sum(
        target.get("support_id") not in {item["support_id"] for item in row["expected_public_targets"]}
        or sorted(target.get("page_numbers") or ()) != next(
            (item["page_numbers"] for item in row["expected_public_targets"] if item["support_id"] == target.get("support_id")),
            [],
        )
        for row, target in predicted_targets
    )
    roles = average([
        float(set(prediction.get("source_role") or ()) == set(row["source_role"]))
        for row, prediction, _, _ in retrieval
    ])
    temporal = average([
        float(set(prediction.get("temporal_scope") or ()) == set(row["temporal_scope"]))
        for row, prediction, _, _ in retrieval
    ])
    targets = average([
        float(
            {target.get("support_id") for target in prediction.get("public_targets") or ()}
            == {target["support_id"] for target in row["expected_public_targets"]}
        )
        for row, prediction, _, _ in retrieval
    ])
    hard_negatives = [row for row in qrels if row["case_kind"] == "adversarial"]
    hard_false = sum(
        bool(set(by_id.get(row["case_id"], {}).get("retrieved_support_ids") or ()) & set(row["forbidden_support_ids"]))
        for row in hard_negatives
    )
    expected_abstain = {row["case_id"] for row in qrels if row["expected_recovery_behavior"] == "abstain"}
    predicted_abstain = {row["case_id"] for row in qrels if by_id.get(row["case_id"], {}).get("recovery_behavior") == "abstain"}
    expected_clarify = [row for row in qrels if row["expected_recovery_behavior"] == "clarify"]
    return {
        f"recall@{k}": recall,
        f"precision@{k}": precision,
        "mrr": reciprocal,
        "ndcg": ndcg,
        "minimal_span_precision": round(minimal_true / len(minimal_actual), 6) if minimal_actual else None,
        "minimal_span_recall": round(minimal_true / len(minimal_expected), 6) if minimal_expected else None,
        "over_highlight_ratio": round(len(highlighted - minimal_expected) / len(highlighted), 6) if highlighted else None,
        "wrong_page_rate": round(wrong_pages / len(predicted_targets), 6) if predicted_targets else None,
        "source_role_accuracy": roles,
        "temporal_accuracy": temporal,
        "citation_target_accuracy": targets,
        "hard_negative_false_positive_rate": round(hard_false / len(hard_negatives), 6) if hard_negatives else None,
        "abstention_precision": round(len(expected_abstain & predicted_abstain) / len(predicted_abstain), 6) if predicted_abstain else None,
        "abstention_recall": round(len(expected_abstain & predicted_abstain) / len(expected_abstain), 6) if expected_abstain else None,
        "clarification_accuracy": average([
            float(by_id.get(row["case_id"], {}).get("recovery_behavior") == "clarify") for row in expected_clarify
        ]),
    }


def evaluate_rows(
    qrels: list[dict],
    supports: list[dict],
    identity: dict[str, str],
    predictions: list[dict] | None = None,
    *,
    k: int = 5,
) -> dict:
    counts: Counter[str] = Counter()
    support_by_id = {row["support_unit_id"]: row for row in supports}
    seen: set[str] = set()
    adjudicated: list[dict] = []
    for row in qrels:
        counts["missing_required_field_count"] += len(REQUIRED_FIELDS - row.keys())
        counts["unexpected_field_count"] += len(row.keys() - REQUIRED_FIELDS)
        case_id = str(row.get("case_id") or "")
        counts["duplicate_case_id_count"] += case_id in seen
        seen.add(case_id)
        counts["invalid_case_kind_count"] += row.get("case_kind") not in {"primary", "adversarial"}
        counts["invalid_split_count"] += row.get("split") not in {"dev", "acceptance"}
        counts["invalid_review_status_count"] += row.get("review_status") not in {"candidate", "reviewed", "adjudicated", "rejected"}
        counts["invalid_answerability_count"] += row.get("answerability") not in {
            "answerable", "unanswerable", "clarification_required"
        }
        counts["invalid_recovery_behavior_count"] += row.get("expected_recovery_behavior") not in {
            "retrieve", "abstain", "clarify"
        }
        counts["invalid_citation_finality_count"] += row.get("citation_finality") not in {
            "final", "nonfinal", "mixed", "not_applicable"
        }
        counts["invalid_case_text_count"] += not case_id or not isinstance(row.get("query"), str) or not row.get("query", "").strip()
        counts["corpus_identity_mismatch_count"] += row.get("corpus_identity") != identity
        gold = set(row.get("gold_support_ids") or ())
        alternatives = set(row.get("alternative_valid_support_ids") or ())
        forbidden = set(row.get("forbidden_support_ids") or ())
        counts["overlapping_support_set_count"] += bool((gold & alternatives) | (gold & forbidden) | (alternatives & forbidden))
        counts["unresolved_support_id_count"] += len((gold | alternatives | forbidden) - support_by_id.keys())
        counts["answerable_without_gold_count"] += row.get("answerability") == "answerable" and not gold
        relevant = gold | alternatives
        for item in row.get("minimal_relevant_spans") or ():
            support = support_by_id.get(item.get("support_id"))
            counts["minimal_span_support_mismatch_count"] += item.get("support_id") not in relevant
            counts["unresolved_minimal_span_count"] += support is None or not set(item.get("text_span_ids") or ()) <= set(
                support.get("text_span_ids") or () if support else ()
            )
        for target in row.get("expected_public_targets") or ():
            support = support_by_id.get(target.get("support_id"))
            counts["public_target_support_mismatch_count"] += target.get("support_id") not in relevant
            counts["public_target_contract_mismatch_count"] += support is None or any(
                target.get(field) != support.get(field)
                for field in ("source_document_id", "page_numbers", "viewer_eligible", "highlight_eligible")
            )
        if row.get("review_status") == "adjudicated":
            valid_review = bool(row.get("reviewer_role") and row.get("reviewed_at"))
            counts["invalid_adjudication_record_count"] += not valid_review
            if valid_review:
                adjudicated.append(row)
        elif row.get("review_status") == "candidate":
            counts["candidate_with_review_metadata_count"] += row.get("reviewer_role") is not None or row.get("reviewed_at") is not None

    if predictions is not None:
        prediction_ids: set[str] = set()
        for row in predictions:
            counts["prediction_contains_qrel_label_count"] += bool(row.keys() & LABEL_FIELDS)
            case_id = str(row.get("case_id") or "")
            counts["duplicate_prediction_id_count"] += case_id in prediction_ids
            prediction_ids.add(case_id)
        counts["unknown_prediction_id_count"] = len(prediction_ids - {row["case_id"] for row in adjudicated})
        counts["missing_prediction_id_count"] = len({row["case_id"] for row in adjudicated} - prediction_ids)

    primary_reviewed = sum(row["case_kind"] == "primary" for row in adjudicated)
    adversarial_reviewed = sum(row["case_kind"] == "adversarial" for row in adjudicated)
    failures = sum(counts.values())
    framework_status = "PASS" if failures == 0 else "FAIL"
    acceptance_status = "PASS" if primary_reviewed >= 50 and adversarial_reviewed >= 20 else "BLOCKED"
    metrics = _metrics(adjudicated, predictions, k) if predictions is not None and adjudicated else {
        name: None for name in (
            f"recall@{k}", f"precision@{k}", "mrr", "ndcg", "minimal_span_precision", "minimal_span_recall",
            "over_highlight_ratio", "wrong_page_rate", "source_role_accuracy", "temporal_accuracy",
            "citation_target_accuracy", "hard_negative_false_positive_rate", "abstention_precision",
            "abstention_recall", "clarification_accuracy",
        )
    }
    return {
        "status": framework_status if framework_status == "FAIL" else acceptance_status,
        "framework_status": framework_status,
        "acceptance_status": acceptance_status,
        "case_count": len(qrels),
        "primary_count": sum(row.get("case_kind") == "primary" for row in qrels),
        "adversarial_count": sum(row.get("case_kind") == "adversarial" for row in qrels),
        "candidate_count": sum(row.get("review_status") == "candidate" for row in qrels),
        "reviewed_count": sum(row.get("review_status") == "reviewed" for row in qrels),
        "adjudicated_count": len(adjudicated),
        "adjudicated_primary_count": primary_reviewed,
        "adjudicated_adversarial_count": adversarial_reviewed,
        "unreviewed_counted_toward_acceptance": 0,
        "metrics_status": "available" if predictions is not None and adjudicated else "unavailable",
        "metrics": metrics,
        "counters": dict(sorted(counts.items())),
    }


def evaluate(qrel_path: Path = DEFAULT_QREL, predictions_path: Path | None = None, repo_root: Path = ROOT, *, k: int = 5) -> dict:
    if predictions_path is not None and predictions_path.resolve() == qrel_path.resolve():
        return {
            "status": "FAIL",
            "framework_status": "FAIL",
            "acceptance_status": "BLOCKED",
            "counters": {"self_comparison_count": 1},
        }
    final = repo_root / "data" / "final" / "uud"
    report = evaluate_rows(
        _rows(qrel_path),
        _rows(final / "meaningful_support_units.jsonl"),
        _identity(final),
        _rows(predictions_path) if predictions_path else None,
        k=k,
    )
    report["schema_sha256"] = sha256(SCHEMA.read_bytes()).hexdigest()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and score expert QREL v0.")
    parser.add_argument("--qrel", type=Path, default=DEFAULT_QREL)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    report = evaluate(args.qrel, args.predictions, k=args.k)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0 if report.get("framework_status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
