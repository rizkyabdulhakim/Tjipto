from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re

from tjipto.core.manifest import artifact_set_digest


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "evaluation" / "uud" / "qrel_v0.jsonl"


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _diverse(rows: list[dict], count: int) -> list[dict]:
    by_role: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_role[str(row["source_role"])].append(row)
    selected: list[dict] = []
    while len(selected) < count and any(by_role.values()):
        for role in sorted(by_role):
            if by_role[role] and len(selected) < count:
                selected.append(by_role[role].pop(0))
    return selected


def _identity(final: Path) -> dict[str, str]:
    data = (final / "manifest.json").read_bytes()
    manifest = json.loads(data)
    return {
        "corpus_id": "uud",
        "artifact_set_sha256": artifact_set_digest(manifest),
        "manifest_sha256": sha256(data).hexdigest(),
    }


def _target(row: dict) -> dict:
    return {
        "support_id": row["support_unit_id"],
        "source_document_id": row["source_document_id"],
        "page_numbers": row["page_numbers"],
        "viewer_eligible": row["viewer_eligible"],
        "highlight_eligible": row["highlight_eligible"],
    }


def _citation_finality(rows: list[dict]) -> str:
    values = {row.get("citation_final") is True for row in rows}
    return "mixed" if len(values) > 1 else "final" if values == {True} else "nonfinal"


def _query(row: dict, owners: dict[str, dict], quote: str) -> str:
    owner = owners.get(row["owner_id"], {})
    citation = str(owner.get("citation") or "").strip()
    if citation:
        return f"Apa isi {citation}?"
    if row["support_kind"] == "document_title":
        return "Apa judul resmi dokumen UUD 1945 dalam satu naskah?"
    if row["support_kind"] == "metadata":
        return f"Apa fakta dokumen yang didukung oleh teks {quote[:100]}?"
    if row["support_kind"] == "trace":
        return f"Apa status anomali sumber pada teks {quote[:100]}?"
    return f"Tunjukkan dukungan sumber untuk teks {quote[:120]}"


def _coverage(row: dict, quote: str, alternatives: list[dict]) -> list[str]:
    tags = {row["support_kind"], row["authority_kind"]}
    tags.add("current_text" if row["source_role"] == "current_consolidated" else "historical_text")
    tags.add("exact_reference" if "Pasal" in quote else "paraphrase_seed")
    if len(row["text_span_ids"]) > 1:
        tags.add("multi_support")
    if alternatives:
        tags.update(("alternative_valid_evidence", "temporal_ambiguity"))
    if row["support_kind"] == "instrument":
        tags.update(("amendment_instrument", "decision_or_instrument_clause"))
    if row.get("semantic_classification") == "signatory_block":
        tags.add("signatories")
    if row.get("semantic_classification") == "decision_clause":
        tags.add("decision_clause")
    if row.get("semantic_classification") == "session_institution_metadata":
        tags.add("institutions_and_sessions")
    if row["support_kind"] in {"metadata", "document_title"}:
        tags.add("metadata_or_structure")
    if row["support_kind"] == "trace":
        tags.add("source_annotation_or_anomaly")
    return sorted(tags)


def build(repo_root: Path = ROOT) -> list[dict]:
    final = repo_root / "data" / "final" / "uud"
    supports = _rows(final / "meaningful_support_units.jsonl")
    spans = {row["text_span_id"]: row for row in _rows(final / "page_text_spans.jsonl")}
    owner_rows = [
        *_rows(final / "evidence_registry.jsonl"),
        *_rows(final / "metadata_grounding.jsonl"),
        *_rows(final / "source_conflicts.jsonl"),
    ]
    owners = {
        str(row.get("evidence_id") or row.get("metadata_grounding_id") or row.get("source_conflict_id")): row
        for row in owner_rows
    }
    identity = _identity(final)
    published = [row for row in supports if row["decision_kind"] != "typed_exclusion"]
    selected = [
        *_diverse([row for row in published if row["support_kind"] in {"trace", "metadata", "document_title"}], 7),
        *_diverse([row for row in published if row["support_kind"] == "instrument"], 8),
        *_diverse([row for row in published if row["support_kind"] == "structural"], 8),
        *_diverse([row for row in published if row["support_kind"] == "normative"], 27),
    ]
    by_quote: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in published:
        by_quote[(row["quoted_text_sha256"], row["authority_kind"])].append(row)
    by_page: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in published:
        for page in row["page_numbers"]:
            by_page[(row["source_document_id"], page)].append(row)

    result: list[dict] = []
    for index, row in enumerate(selected, start=1):
        quote = "\n".join(spans[span_id]["exact_quote"] for span_id in row["text_span_ids"])
        alternatives = [
            candidate for candidate in by_quote[(row["quoted_text_sha256"], row["authority_kind"])]
            if candidate["support_unit_id"] != row["support_unit_id"]
        ][:2]
        neighbors = [
            candidate for candidate in by_page[(row["source_document_id"], row["page_numbers"][0])]
            if candidate["support_unit_id"] != row["support_unit_id"]
            and candidate["support_unit_id"] not in {item["support_unit_id"] for item in alternatives}
        ][:2]
        relevant = [row, *alternatives]
        result.append({
            "case_id": f"qrel_v0_primary_{index:03d}",
            "case_kind": "primary",
            "query": _query(row, owners, quote),
            "intent": row["support_kind"],
            "answerability": "answerable",
            "gold_support_ids": [row["support_unit_id"]],
            "alternative_valid_support_ids": [item["support_unit_id"] for item in alternatives],
            "forbidden_support_ids": [item["support_unit_id"] for item in neighbors],
            "minimal_relevant_spans": [
                {"support_id": item["support_unit_id"], "text_span_ids": item["text_span_ids"]} for item in relevant
            ],
            "required_claims": [quote],
            "permitted_partial_claims": [],
            "source_role": sorted({item["source_role"] for item in relevant}),
            "temporal_scope": sorted({item["temporal_context"] for item in relevant}),
            "authority_kind": sorted({item["authority_kind"] for item in relevant}),
            "citation_finality": _citation_finality(relevant),
            "expected_public_targets": [_target(item) for item in relevant],
            "expected_recovery_behavior": "retrieve",
            "split": "acceptance" if index % 4 == 0 else "dev",
            "review_status": "candidate",
            "reviewer_role": None,
            "reviewed_at": None,
            "corpus_identity": identity,
            "coverage_tags": _coverage(row, quote, alternatives),
        })

    regression = _rows(repo_root / "tests" / "fixtures" / "uud" / "retrieval_eval_cases.jsonl")
    unsupported = [
        row for row in regression
        if row.get("expected_support_type") == "insufficient_evidence" or row.get("forbidden_evidence_ids")
    ][:18]
    temporal = [row for row in regression if "temporal" in str(row.get("risk_family") or "") and row not in unsupported][:2]
    candidates = [*unsupported, *temporal]
    owner_supports: dict[str, list[str]] = defaultdict(list)
    for row in published:
        owner_supports[row["owner_id"]].append(row["support_unit_id"])
    searchable = [
        (row, set(re.findall(r"[a-z0-9]+", " ".join(spans[span_id]["exact_quote"] for span_id in row["text_span_ids"]).casefold())))
        for row in published
    ]
    for index, case in enumerate(candidates, start=1):
        query_tokens = set(re.findall(r"[a-z0-9]+", case["query"].casefold()))
        forbidden = list(dict.fromkeys(
            support_id
            for owner_id in case.get("forbidden_evidence_ids") or ()
            for support_id in owner_supports.get(owner_id, ())
        ))
        if not forbidden:
            forbidden = [max(searchable, key=lambda item: (len(query_tokens & item[1]), item[0]["support_unit_id"]))[0]["support_unit_id"]]
        clarify = "temporal" in str(case.get("risk_family") or "")
        result.append({
            "case_id": f"qrel_v0_adversarial_{index:03d}",
            "case_kind": "adversarial",
            "query": case["query"],
            "intent": str(case.get("expected_intent") or case.get("expected_route") or "unsupported"),
            "answerability": "clarification_required" if clarify else "unanswerable",
            "gold_support_ids": [],
            "alternative_valid_support_ids": [],
            "forbidden_support_ids": forbidden[:3],
            "minimal_relevant_spans": [],
            "required_claims": [],
            "permitted_partial_claims": [],
            "source_role": [],
            "temporal_scope": [],
            "authority_kind": [],
            "citation_finality": "not_applicable",
            "expected_public_targets": [],
            "expected_recovery_behavior": "clarify" if clarify else "abstain",
            "split": "acceptance" if index % 2 == 0 else "dev",
            "review_status": "candidate",
            "reviewer_role": None,
            "reviewed_at": None,
            "corpus_identity": identity,
            "coverage_tags": sorted({
                "hard_negative", "unsupported_or_out_of_corpus", "lexical_near_miss",
                str(case.get("risk_family") or "unspecified"),
            } | ({"clarification"} if clarify else set())),
        })
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Build deterministic, unreviewed QREL v0 candidates.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    rows = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({"candidate_count": len(rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
