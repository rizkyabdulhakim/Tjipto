from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import unicodedata


ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT / "tests" / "fixtures" / "uud" / "meaningful_support_oracle.json"
REQUIRED_FIELDS = {
    "support_unit_id", "decision_kind", "support_kind", "owner_type", "owner_id",
    "source_document_id", "source_role", "temporal_context", "semantic_classification",
    "legal_force", "authority_kind", "citation_final", "text_span_ids", "raw_source_span_ids",
    "page_numbers", "selector_refs", "bbox_refs", "bbox_precision", "quoted_text_sha256",
    "answer_eligible", "citation_eligible", "viewer_eligible", "highlight_eligible",
    "decision_status", "decision_reason",
}


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_artifacts(final: Path) -> dict[str, list[dict]]:
    return {
        "spans": _rows(final / "page_text_spans.jsonl"),
        "raw": _rows(final / "raw_source_spans.jsonl"),
        "evidence": _rows(final / "evidence_registry.jsonl"),
        "metadata": _rows(final / "metadata_grounding.jsonl"),
        "conflicts": _rows(final / "source_conflicts.jsonl"),
        "bboxes": _rows(final / "bbox_registry.jsonl"),
        "words": _rows(final / "word_bboxes.jsonl"),
    }


def _canonical_owners(artifacts: dict[str, list[dict]]) -> list[tuple[str, str, dict]]:
    return [
        *(("evidence_registry", row["evidence_id"], row) for row in artifacts["evidence"]),
        *(("metadata_grounding", row["metadata_grounding_id"], row) for row in artifacts["metadata"]),
        *(("source_conflict", row["source_conflict_id"], row) for row in artifacts["conflicts"]),
    ]


def _compatible(span: dict, owner_type: str, owner: dict) -> bool:
    if owner.get("source_document_id") != span["source_document_id"]:
        return False
    role = owner.get("source_role") or (owner.get("source_anomaly_policy") or {}).get("source_role")
    if role != span["source_role"] or (owner.get("temporal_context") or role) != span["temporal_context"]:
        return False
    authority = str(owner.get("authority_kind") or "")
    if owner_type != "source_conflict" and authority != span.get("linked_authority"):
        return False
    return span["legal_force"] in {
        "normative_legal_text": {"canonical_normative", "historical_normative"},
        "structural_context": {"canonical_normative", "historical_normative", "metadata_only"},
        "instrument_provenance": {"amendment_instrument"},
        "metadata": {"metadata_only"},
        "source_anomaly_provenance": {"historical_normative", "amendment_instrument", "metadata_only"},
    }.get(authority, set())


def _expected_owner(span: dict, owners: list[tuple[str, str, dict]]) -> tuple[str, str, dict] | None:
    candidates = [
        item for item in owners
        if span["text_span_id"] in (item[2].get("text_span_ids") or ()) and _compatible(span, item[0], item[2])
    ]
    return min(candidates, key=lambda item: (len(item[2].get("text_span_ids") or ()), item[1], item[0])) if candidates else None


def _visible_text(value: object) -> str:
    return "".join(
        character
        for character in unicodedata.normalize("NFKC", str(value or "")).replace("\u00ad", "")
        if not character.isspace()
    )


def _glyph_key(text: object, bbox: dict) -> tuple[object, ...]:
    return (_visible_text(text), *(round(float(bbox[key]), 2) for key in ("x0", "y0", "x1", "y1")))


def _oracle_character_refs(span: dict, raw: dict, page_glyphs: dict[tuple[object, ...], list[dict]]) -> list[str]:
    if raw.get("semantic_exact_quote") != span.get("exact_quote"):
        return []
    raw_glyphs = [
        (str(text), bbox)
        for text, bbox in zip(raw.get("character_texts") or (), raw.get("character_bboxes") or ())
        if _visible_text(text)
    ]
    result: list[dict] = []
    consumed: set[str] = set()
    for text, bbox in raw_glyphs:
        candidates = [
            glyph for glyph in page_glyphs.get(_glyph_key(text, bbox), ())
            if str(glyph.get("character_bbox_id") or "") not in consumed
        ]
        if len(candidates) != 1:
            return []
        result.append(candidates[0])
        consumed.add(str(candidates[0]["character_bbox_id"]))
    if _visible_text("".join(str(glyph.get("text") or "") for glyph in result)) != _visible_text(span["exact_quote"]):
        return []
    return [str(glyph["character_bbox_id"]) for glyph in result]


def _expected_geometry(
    spans: list[dict], raw_rows: list[dict], glyphs_by_page: dict[tuple[str, int], dict[tuple[object, ...], list[dict]]]
) -> list[str]:
    refs: list[str] = []
    for span, raw in zip(spans, raw_rows):
        selected = _oracle_character_refs(
            span, raw, glyphs_by_page.get((str(span["source_document_id"]), int(span["page_number"])), {})
        )
        if not selected:
            return []
        refs.extend(selected)
    return list(dict.fromkeys(refs))


def evaluate_rows(rows: list[dict], artifacts: dict[str, list[dict]], oracle: dict) -> dict:
    counts: Counter[str] = Counter()
    spans = {row["text_span_id"]: row for row in artifacts["spans"]}
    raw = {row["raw_source_span_id"]: row for row in artifacts["raw"]}
    owners = _canonical_owners(artifacts)
    glyphs_by_page: dict[tuple[str, int], dict[tuple[object, ...], list[dict]]] = {}
    for word in artifacts["words"]:
        index = glyphs_by_page.setdefault((str(word["source_document_id"]), int(word["page_number"])), {})
        for glyph in word.get("characters") or ():
            index.setdefault(_glyph_key(glyph.get("text"), glyph), []).append(glyph)
    owner_lookup = {(owner_type, owner_id): owner for owner_type, owner_id, owner in owners}
    reviews = oracle["reviewed_decisions"]
    expected_spans = {
        span_id for span_id, span in spans.items()
        if span["legal_force"] in oracle["meaningful_legal_forces"] or span_id in reviews
    }
    decisions: Counter[str] = Counter()
    seen_units: set[str] = set()

    for row in rows:
        counts["missing_required_field_count"] += len(REQUIRED_FIELDS - row.keys())
        unit_id = str(row.get("support_unit_id") or "")
        counts["duplicate_support_unit_id_count"] += unit_id in seen_units
        seen_units.add(unit_id)
        span_rows = [spans.get(span_id) for span_id in row.get("text_span_ids", [])]
        counts["unresolved_span_count"] += sum(span is None for span in span_rows)
        if not span_rows or any(span is None for span in span_rows):
            continue
        selected = [span for span in span_rows if span is not None]
        for span in selected:
            decisions[span["text_span_id"]] += 1
        counts["unexpected_span_decision_count"] += sum(span["text_span_id"] not in expected_spans for span in selected)

        for field in ("source_document_id", "source_role", "temporal_context", "semantic_classification", "legal_force"):
            values = {span[field] for span in selected}
            counts[f"cross_{field}_group_count"] += len(values) != 1 or row.get(field) not in values
        pages = sorted({span["page_number"] for span in selected})
        counts["page_mismatch_count"] += row.get("page_numbers") != pages
        positions = sorted((span["page_number"], int(span["text_span_id"].rsplit("::", 1)[1])) for span in selected)
        counts["noncontiguous_group_count"] += any(
            page != prior_page or index != prior_index + 1
            for (prior_page, prior_index), (page, index) in zip(positions, positions[1:])
        )
        expected_hash = sha256("\n".join(span["exact_quote"] for span in selected).encode()).hexdigest()
        counts["quote_reconstruction_mismatch_count"] += row.get("quoted_text_sha256") != expected_hash

        raw_rows: list[dict] = []
        for span in selected:
            matches = [
                item for item in artifacts["raw"]
                if item["source_document_id"] == span["source_document_id"]
                and item["page_number"] == span["page_number"]
                and item.get("semantic_text_start") == span["text_start"]
                and item.get("semantic_text_end") == span["text_end"]
            ]
            counts["raw_span_join_count_mismatch"] += len(matches) != 1
            if len(matches) == 1:
                raw_rows.append(matches[0])
        expected_raw = [item["raw_source_span_id"] for item in raw_rows]
        counts["raw_span_reference_mismatch_count"] += row.get("raw_source_span_ids") != expected_raw
        counts["selector_reference_mismatch_count"] += row.get("selector_refs") != expected_raw
        counts["unresolved_selector_count"] += sum(selector not in raw for selector in row.get("selector_refs", []))
        if len(raw_rows) != len(selected):
            continue

        reviewed = [reviews.get(span["text_span_id"]) for span in selected]
        if any(reviewed):
            counts["mixed_review_group_count"] += any(review is None for review in reviewed)
            expected = reviewed[0]
            if expected is None:
                continue
            counts["reviewed_decision_mismatch_count"] += any(
                row.get(field) != expected.get(field)
                for field in (
                    "decision_kind", "support_kind", "owner_type", "owner_id", "source_document_id",
                    "source_role", "temporal_context", "semantic_classification", "legal_force",
                    "authority_kind", "decision_reason",
                )
            )
            exclusion = expected["decision_kind"] == "typed_exclusion"
            expected_owner = expected
        else:
            selected_owners = [_expected_owner(span, owners) for span in selected]
            counts["missing_owner_count"] += sum(owner is None for owner in selected_owners)
            if any(owner is None for owner in selected_owners):
                continue
            expected_keys = {(owner[0], owner[1]) for owner in selected_owners if owner is not None}
            counts["canonical_owner_arbitration_mismatch_count"] += (
                len(expected_keys) != 1 or (row.get("owner_type"), row.get("owner_id")) not in expected_keys
            )
            owner_key = (str(row.get("owner_type") or ""), str(row.get("owner_id") or ""))
            expected_owner = owner_lookup.get(owner_key)
            counts["missing_owner_count"] += expected_owner is None
            if expected_owner is None:
                continue
            exclusion = False
            authority = expected_owner.get("authority_kind")
            expected_kind = oracle["authority_support_kinds"].get(authority)
            counts["authority_kind_mismatch_count"] += row.get("authority_kind") != authority
            counts["support_kind_mismatch_count"] += row.get("support_kind") != expected_kind
            counts["decision_kind_mismatch_count"] += row.get("decision_kind") != "canonical_owner_support"

        if exclusion:
            counts["exclusion_capability_count"] += sum(
                row.get(field) is not False
                for field in ("answer_eligible", "citation_eligible", "viewer_eligible", "highlight_eligible", "citation_final")
            )
            counts["exclusion_geometry_count"] += bool(row.get("bbox_refs")) or row.get("bbox_precision") != "not_applicable"
            expected_geometry: list[str] = []
        else:
            expected_geometry = _expected_geometry(selected, raw_rows, glyphs_by_page)
            if row.get("bbox_precision") == "exact":
                unexpected_geometry = set(row.get("bbox_refs") or ()) - set(expected_geometry)
                counts["owner_wide_segment_bbox_count"] += bool(unexpected_geometry)
                counts["exact_highlight_outside_selected_span_count"] += bool(unexpected_geometry)
                counts["segment_geometry_mismatch_count"] += row.get("bbox_refs") != expected_geometry or not expected_geometry
            else:
                counts["page_grounding_mismatch_count"] += (
                    row.get("bbox_precision") != "page_grounded_only" or bool(row.get("bbox_refs"))
                )
            owner_highlightable = expected_owner.get("viewer_highlightable") is not False
            expected_highlight = bool(expected_geometry) and owner_highlightable
            counts["viewer_eligibility_mismatch_count"] += row.get("viewer_eligible") is not True
            counts["highlight_eligibility_mismatch_count"] += row.get("highlight_eligible") is not expected_highlight

            authority = expected_owner.get("authority_kind")
            expected_final = expected_owner.get("citation_final") is True if row.get("owner_type") != "review_decision" else False
            expected_answer = authority == "normative_legal_text" and row.get("legal_force") == "canonical_normative"
            expected_citation = row.get("owner_type") == "evidence_registry" and expected_owner.get("citable") is True
            counts["finality_mismatch_count"] += row.get("citation_final") is not expected_final
            counts["answer_eligibility_mismatch_count"] += row.get("answer_eligible") is not expected_answer
            counts["citation_eligibility_mismatch_count"] += row.get("citation_eligible") is not expected_citation
            counts["valid_legal_support_lost_only_for_geometry_count"] += (
                not expected_geometry
                and (
                    (expected_answer and row.get("answer_eligible") is not True)
                    or (expected_citation and row.get("citation_eligible") is not True)
                )
            )

        counts["legal_force_escalation_count"] += row.get("answer_eligible") is True and row.get("legal_force") != "canonical_normative"
        counts["historical_as_current_count"] += row.get("answer_eligible") is True and row.get("source_role") != "current_consolidated"
        counts["trace_or_metadata_final_count"] += row.get("support_kind") in {"trace", "metadata"} and row.get("citation_final") is True
        counts["layout_separator_published_count"] += row.get("support_kind") == "layout_separator" and row.get("decision_kind") != "typed_exclusion"

    counts["meaningful_span_without_decision_count"] = len(expected_spans - set(decisions))
    counts["duplicate_ownership_count"] = sum(value - 1 for value in decisions.values() if value > 1)
    counts["decision_for_unknown_span_count"] = len(set(decisions) - expected_spans)
    failures = sum(counts.values())
    support_rows = [row for row in rows if row.get("decision_kind") != "typed_exclusion"]
    exclusion_rows = [row for row in rows if row.get("decision_kind") == "typed_exclusion"]
    return {
        "status": "PASS" if failures == 0 else "FAIL",
        "decision_unit_count": len(rows),
        "support_unit_count": len(support_rows),
        "exclusion_unit_count": len(exclusion_rows),
        "decision_span_count": sum(len(row.get("text_span_ids", [])) for row in rows),
        "support_span_count": sum(len(row.get("text_span_ids", [])) for row in support_rows),
        "exclusion_span_count": sum(len(row.get("text_span_ids", [])) for row in exclusion_rows),
        "canonical_owner_span_count": sum(
            len(row.get("text_span_ids", [])) for row in support_rows if row.get("owner_type") != "review_decision"
        ),
        "exact_geometry_unit_count": sum(row.get("bbox_precision") == "exact" for row in support_rows),
        "page_grounded_unit_count": sum(row.get("bbox_precision") == "page_grounded_only" for row in support_rows),
        "support_kind_unit_counts": dict(sorted(Counter(row.get("support_kind") for row in rows).items())),
        "support_kind_span_counts": dict(sorted(Counter(
            row.get("support_kind") for row in rows for _ in row.get("text_span_ids", [])
        ).items())),
        "counters": dict(sorted(counts.items())),
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
