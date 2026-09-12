from __future__ import annotations

from collections import Counter, defaultdict
import re

from tjipto.contracts.evidence import normalize_source_text, valid_text_range
from tjipto.corpora.disposition import (
    EXCLUDED_STATUSES,
    LEGAL_FORCES,
    PROMOTED_STATUSES,
    PROMOTION_STATUSES,
    REVIEW_STATUSES,
    SEMANTIC_CLASSIFICATIONS,
    SPAN_DISPOSITION_FIELDS,
    SPAN_ROLES,
)
from tjipto.corpora.uud.proposition_builder import source_marker_character_boxes
from tjipto.evidence.bbox import (
    VIEWER_GEOMETRY_SPACE_FIELDS,
    derive_viewer_overlay,
    geometry_space_key,
    positive_area_intersection,
    viewer_overlay_rectangles,
)
from tjipto.ingestion.pdf.source_objects import TERMINAL_DISPOSITIONS


def _raw_source_geometry_health(rows: list[dict]) -> dict:
    mismatches = 0
    marker_mismatches = 0
    for row in rows:
        boxes = row.get("character_bboxes") or []
        valid = (
            row.get("raw_geometry_method") == "pdf_rawdict_character_bbox"
            and "".join(str(value) for value in row.get("character_texts") or ()) == str(row.get("raw_text") or "")
            and len(boxes) == len(row.get("character_ids") or ())
            and bool(boxes)
        )
        if valid:
            union = (
                min(box["x0"] for box in boxes),
                min(box["y0"] for box in boxes),
                max(box["x1"] for box in boxes),
                max(box["y1"] for box in boxes),
            )
            valid = union == tuple(row.get(field) for field in ("x0", "y0", "x1", "y1"))
        if not valid:
            mismatches += 1
            if row.get("classification") == "source_annotation_marker":
                marker_mismatches += 1
    return {
        "raw_segment_count": len(rows),
        "non_empty_disposition_count": sum(bool(str(row.get("raw_text") or "")) for row in rows),
        "raw_geometry_mismatch_count": mismatches,
        "marker_geometry_mismatch_count": marker_mismatches,
        "whole_line_geometry_fallback_count": sum(row.get("raw_geometry_method") != "pdf_rawdict_character_bbox" for row in rows),
        "status": "pass" if not mismatches and all(str(row.get("raw_text") or "") for row in rows) else "fail",
    }


def _source_object_disposition_health(rows: list[dict], page_text_spans: list[dict]) -> dict:
    ids = [str(row.get("source_object_id") or "") for row in rows]
    known_span_ids = {str(row.get("text_span_id") or "") for row in page_text_spans}
    invalid_dispositions = [row for row in rows if row.get("disposition") not in TERMINAL_DISPOSITIONS]
    unresolved_spans = [span_id for row in rows for span_id in row.get("text_span_ids") or () if str(span_id) not in known_span_ids]
    missing_lineage = [
        row
        for row in rows
        if not row.get("source_document_id") or not row.get("source_sha256") or not isinstance(row.get("page_number"), int)
    ]
    return {
        "source_object_count": len(rows),
        "terminal_disposition_count": sum(row.get("disposition") in TERMINAL_DISPOSITIONS for row in rows),
        "duplicate_source_object_id_count": len(ids) - len(set(ids)),
        "missing_source_object_id_count": sum(not value for value in ids),
        "invalid_disposition_count": len(invalid_dispositions),
        "unresolved_text_span_ref_count": len(unresolved_spans),
        "missing_lineage_count": len(missing_lineage),
        "extraction_failed_count": sum(row.get("disposition") == "extraction_failed" for row in rows),
        "needs_review_count": sum(row.get("disposition") == "needs_review" for row in rows),
        "status": "complete"
        if rows and not invalid_dispositions and not unresolved_spans and not missing_lineage and len(ids) == len(set(ids)) and all(ids)
        else "incomplete",
    }


def _aggregate_compare_text(text: object) -> str:
    normalized = normalize_source_text(re.sub(r"\*+\)", "", str(text or "")))
    return re.sub(r"\W+", "", normalized)


def _validate_relation_support(row: dict, evidence: dict, span_by_id: dict[str, dict], bbox_by_id: dict[str, dict]) -> tuple[str, ...]:
    if row.get("relation_type") not in {"RENAMES", "RENUMBERED_TO"}:
        return ()
    quoted = str(evidence.get("quoted_text") or "")
    old_range = valid_text_range(row.get("old_reference_range"), len(quoted))
    new_range = valid_text_range(row.get("new_reference_range"), len(quoted))
    if old_range is None or new_range is None:
        return ("article_relation_support_missing_mapping_range",)
    expected_ids = tuple(evidence.get("text_span_ids") or ())
    expected_bbox_ids = tuple(evidence.get("bbox_refs") or ())
    support_ids = tuple(row.get("text_span_ids") or ())
    support_bbox_ids = tuple(row.get("bbox_refs") or ())
    source_support_bbox_ids = tuple(bbox_id for bbox_id in support_bbox_ids if bbox_id not in set(row.get("target_bbox_refs") or ()))
    errors: list[str] = []
    if not support_ids or not support_bbox_ids:
        return ("article_relation_support_missing_segment",)
    if not set(support_ids) <= set(expected_ids) or not set(source_support_bbox_ids) <= set(expected_bbox_ids):
        errors.append("article_relation_support_reference_mismatch")
    span_order = _ordered_support_slice(expected_ids, support_ids, span_by_id, quoted, old_range[0], new_range[1])
    bbox_order = _ordered_support_slice(expected_bbox_ids, source_support_bbox_ids, bbox_by_id, quoted, old_range[0], new_range[1])
    if span_order is None or bbox_order is None:
        errors.append("article_relation_support_not_minimal")
    support_text = _support_text(support_ids, span_by_id)
    support_bbox_text = _support_text(source_support_bbox_ids, bbox_by_id)
    old_text = normalize_source_text(quoted[old_range[0] : old_range[1]])
    new_text = normalize_source_text(quoted[new_range[0] : new_range[1]])
    for text in (support_text, support_bbox_text):
        normalized = normalize_source_text(text)
        old_position = normalized.find(old_text)
        transition_position = normalized.find("menjadi", old_position + len(old_text))
        new_position = normalized.find(new_text, transition_position + len("menjadi"))
        if old_position < 0 or transition_position < 0 or new_position < 0:
            errors.append("article_relation_support_missing_mapping_part")
        elif not old_position < transition_position < new_position:
            errors.append("article_relation_support_order")
    return tuple(dict.fromkeys(errors))


def _ordered_support_slice(
    expected_ids: tuple[str, ...], support_ids: tuple[str, ...], rows_by_id: dict[str, dict], quoted: str, start: int, end: int
) -> tuple[str, ...] | None:
    positions: list[tuple[str, int, int]] = []
    cursor = 0
    for row_id in expected_ids:
        text = str(rows_by_id.get(row_id, {}).get("text") or "")
        position = quoted.find(text, cursor) if text else -1
        if position < 0:
            return None
        positions.append((row_id, position, position + len(text)))
        cursor = position + len(text)
    required = tuple(row_id for row_id, row_start, row_end in positions if row_end > start and row_start < end)
    return required if tuple(support_ids) == required else None


def _support_text(ids: tuple[str, ...], rows_by_id: dict[str, dict]) -> str:
    return " ".join(str(rows_by_id.get(row_id, {}).get("text") or "") for row_id in ids)


def _metadata_bbox_registry_health(metadata_grounding_registry: list[dict], known_bbox_ids: set[str]) -> dict:
    bbox_ids = [row.get("bbox_id") for row in metadata_grounding_registry if row.get("bbox_id")]
    ref_ids = [row.get("metadata_grounding_ref_id") for row in metadata_grounding_registry if row.get("metadata_grounding_ref_id")]
    exact_rows = [row for row in metadata_grounding_registry if row.get("bbox_precision") == "exact"]
    page_rows = [row for row in metadata_grounding_registry if row.get("bbox_precision") == "page_grounded_only"]
    unresolved = [row for row in metadata_grounding_registry if row.get("bbox_id") and row.get("bbox_id") not in known_bbox_ids]
    unresolved_exact = [row for row in exact_rows if row.get("bbox_id") not in known_bbox_ids]
    return {
        "metadata_grounding_registry_rows": len(metadata_grounding_registry),
        "metadata_grounding_ref_id_count": len(ref_ids),
        "metadata_grounding_ref_id_unique_count": len(set(ref_ids)),
        "duplicate_bbox_id_reference_count": len(bbox_ids) - len(set(bbox_ids)),
        "unresolved_bbox_id_count": len(unresolved),
        "exact_metadata_bbox_rows": len(exact_rows),
        "exact_metadata_viewer_highlightable_rows": sum(1 for row in exact_rows if row.get("viewer_highlightable") is True),
        "unresolved_exact_metadata_bbox_rows": len(unresolved_exact),
        "page_grounded_only_metadata_rows": len(page_rows),
        "metadata_bbox_false_exact_claims": len(unresolved_exact),
    }


def _word_bbox_registry_health(*, word_bboxes: list[dict], pages: list[dict]) -> dict:
    page_keys = {(row["source_document_id"], row["page_number"]) for row in pages}
    invalid_coords = [
        row
        for row in word_bboxes
        if None in {row.get("x0"), row.get("y0"), row.get("x1"), row.get("y1")}
        or row.get("x1", 0) < row.get("x0", 0)
        or row.get("y1", 0) < row.get("y0", 0)
        or row.get("x0", 0) < 0
        or row.get("y0", 0) < 0
    ]
    missing_page_refs = [row for row in word_bboxes if (row.get("source_document_id"), row.get("page_number")) not in page_keys]
    missing_normalized_text = [row for row in word_bboxes if row.get("text") and not str(row.get("normalized_text") or "").strip()]
    return {
        "word_bbox_rows": len(word_bboxes),
        "source_document_count": len({row.get("source_document_id") for row in word_bboxes}),
        "page_count": len({(row.get("source_document_id"), row.get("page_number")) for row in word_bboxes}),
        "invalid_coordinate_count": len(invalid_coords),
        "missing_page_ref_count": len(missing_page_refs),
        "missing_normalized_text_count": len(missing_normalized_text),
        "status": "complete" if not invalid_coords and not missing_page_refs and not missing_normalized_text else "incomplete",
    }


def _selector_geometry_health(
    *, propositions: list[dict], evidence: list[dict], page_text_spans: list[dict], word_bboxes: list[dict]
) -> dict:
    spans = {row.get("text_span_id"): row for row in page_text_spans}
    referenced_character_ids = {
        str(character_id)
        for proposition in propositions
        for character_id in (
            *(proposition.get("bbox_refs") or ()),
            *(
                character_id
                for selector in proposition.get("source_selectors") or ()
                for character_id in selector.get("character_bbox_ids") or ()
            ),
        )
    }
    character_count, known_character_ids, characters_by_id = _selected_character_geometry(word_bboxes, referenced_character_ids)
    marker_boxes: dict[tuple[object, ...], list[dict]] = defaultdict(list)
    for marker in source_marker_character_boxes(word_bboxes):
        marker_boxes[geometry_space_key(marker)].append(marker)
    evidence_by_id = {row.get("evidence_id"): row for row in evidence}
    marker_pattern = re.compile(r"\*{1,4}(?:/\*{1,4})?\)")
    totals: Counter[str] = Counter()
    for proposition in propositions:
        totals.update(
            _proposition_geometry_counts(
                proposition=proposition,
                spans=spans,
                known_character_ids=known_character_ids,
                characters_by_id=characters_by_id,
                marker_boxes=marker_boxes,
                evidence_by_id=evidence_by_id,
                marker_pattern=marker_pattern,
            )
        )
    counts = {
        "selector_round_trip_mismatch_count": totals["round_trip"],
        "absolute_selector_mismatch_count": totals["absolute_selector"],
        "unknown_selected_character_id_count": totals["unknown_character"],
        "partial_selector_whole_span_bbox_count": totals["whole_span_bbox"],
        "persisted_character_bbox_id_duplicate_count": character_count - len(known_character_ids),
        "marker_text_in_exact_quote_count": totals["marker_text"],
        "marker_viewer_geometry_intersection_count": totals["marker_intersection"],
        "viewer_geometry_without_exact_selector_lineage_count": totals["overlay_lineage"],
        "citable_support_without_valid_geometry_count": totals["citable_without_geometry"],
    }
    return counts | {"status": "complete" if not any(counts.values()) else "incomplete"}


def _selected_character_geometry(word_bboxes: list[dict], referenced_character_ids: set[str]) -> tuple[int, set[str], dict[str, dict]]:
    character_count = 0
    known_character_ids: set[str] = set()
    characters_by_id: dict[str, dict] = {}
    for word in word_bboxes:
        for character in word.get("characters") or ():
            character_id = str(character.get("character_bbox_id") or "")
            if character_id:
                character_count += 1
                known_character_ids.add(character_id)
            if character_id in referenced_character_ids:
                geometry_space = {field: word.get(field) for field in VIEWER_GEOMETRY_SPACE_FIELDS}
                characters_by_id[character_id] = geometry_space | {
                    "word_bbox_id": word.get("word_bbox_id"),
                    "character_bbox_id": character_id,
                    "x0": character["x0"],
                    "y0": character["y0"],
                    "x1": character["x1"],
                    "y1": character["y1"],
                }
    return character_count, known_character_ids, characters_by_id


def _proposition_geometry_counts(
    *,
    proposition: dict,
    spans: dict[object, dict],
    known_character_ids: set[str],
    characters_by_id: dict[str, dict],
    marker_boxes: dict[tuple[object, ...], list[dict]],
    evidence_by_id: dict[object, dict],
    marker_pattern: re.Pattern[str],
) -> Counter[str]:
    counts: Counter[str] = Counter()
    selector_quotes: list[str] = []
    selector_bboxes: list[str] = []
    for selector in proposition.get("source_selectors") or ():
        span = spans.get(selector.get("text_span_id")) or {}
        text = str(span.get("text") or "")
        start, end = selector.get("start"), selector.get("end")
        if not isinstance(start, int) or not isinstance(end, int):
            selector_quotes.append("")
            continue
        selector_quotes.append(text[start:end])
        character_ids = selector.get("character_bbox_ids") or ()
        selector_bboxes.extend(character_ids)
        if _selector_offsets_mismatch(selector, span, start, end):
            counts["absolute_selector"] += 1
        counts["unknown_character"] += len(set(character_ids) - known_character_ids)
        if set(character_ids) & set(span.get("span_bbox_ids") or ()):
            counts["whole_span_bbox"] += 1
    geometry = tuple(dict.fromkeys(selector_bboxes))
    exact_quote = proposition.get("exact_quote")
    if "\n".join(selector_quotes) != exact_quote or geometry != tuple(proposition.get("bbox_refs") or ()):
        counts["round_trip"] += 1
    counts.update(_overlay_geometry_counts(proposition, geometry, characters_by_id, marker_boxes))
    if marker_pattern.search(str(exact_quote or "")):
        counts["marker_text"] += 1
    evidence_row = evidence_by_id.get(proposition.get("evidence_id"), {})
    if evidence_row.get("citation_final") is True and (not geometry or not set(geometry) <= known_character_ids):
        counts["citable_without_geometry"] += 1
    return counts


def _selector_offsets_mismatch(selector: dict, span: dict, start: int, end: int) -> bool:
    return (
        selector.get("stream_id") != span.get("stream_id")
        or selector.get("absolute_start") != span.get("text_start", 0) + start
        or selector.get("absolute_end") != span.get("text_start", 0) + end
    )


def _overlay_geometry_counts(
    proposition: dict,
    geometry: tuple[str, ...],
    characters_by_id: dict[str, dict],
    marker_boxes: dict[tuple[object, ...], list[dict]],
) -> Counter[str]:
    counts: Counter[str] = Counter()
    overlay = proposition.get("viewer_overlay") or {}
    expected_overlay = derive_viewer_overlay(proposition, characters_by_id, marker_boxes)
    rectangles = viewer_overlay_rectangles(proposition, characters_by_id)
    covered_character_ids = {character_id for rectangle in rectangles for character_id in rectangle.get("character_bbox_ids") or ()}
    if expected_overlay.get("status") != "complete" or overlay != expected_overlay or covered_character_ids != set(geometry):
        counts["overlay_lineage"] += 1
    counts["marker_intersection"] += sum(
        1
        for rectangle in rectangles
        for marker in marker_boxes.get(geometry_space_key(rectangle), ())
        if positive_area_intersection(rectangle, marker)
    )
    return counts


def _pdf_health_summary(report: dict) -> dict:
    pages = report.get("pages") or ()
    sources = report.get("source_documents") or ()
    return {
        "status": report.get("status") or "missing",
        "source_count": int(report.get("source_count") or 0),
        "page_count": int(report.get("page_count") or 0),
        "native_text_ok_source_count": int(report.get("native_text_ok_source_count") or 0),
        "native_text_ok_page_count": int(report.get("native_text_ok_page_count") or 0),
        "ocr_required_count": int(report.get("ocr_required_count") or 0),
        "ocr_dependency_status": report.get("ocr_dependency_status") or "unknown",
        "source_unusable_count": sum(1 for row in sources if row.get("health_decision") == "source_unusable"),
        "needs_review_count": sum(1 for row in (*sources, *pages) if row.get("health_decision") == "needs_review"),
        "repair_required_count": sum(1 for row in (*sources, *pages) if row.get("health_decision") == "repair_required"),
    }


def _source_quote_fidelity_health(*, metadata_grounding: list[dict], evidence: list[dict], pages: list[dict]) -> dict:
    page_text = {(row["source_document_id"], row["page_number"]): row.get("text", "") for row in pages}
    metadata_rows = list(metadata_grounding)
    evidence_rows = [row for row in evidence if row.get("bbox_precision") == "page_grounded_only"]
    metadata_mismatches = [row for row in metadata_rows if not _quote_in_pages(row, page_text)]
    evidence_mismatches = [row for row in evidence_rows if not _quote_in_pages(row, page_text)]
    tracked = [row for row in metadata_mismatches if row.get("failure_reason")]
    return {
        "metadata_grounding_checked_count": len(metadata_rows),
        "page_grounded_evidence_checked_count": len(evidence_rows),
        "metadata_source_quote_mismatch_count": len(metadata_mismatches),
        "page_grounded_evidence_source_quote_mismatch_count": len(evidence_mismatches),
        "tracked_exception_count": len(tracked),
        "untracked_mismatch_count": len(metadata_mismatches) - len(tracked) + len(evidence_mismatches),
        "page_grounded_only_metadata_count": sum(1 for row in metadata_rows if row.get("bbox_precision") == "page_grounded_only"),
    }


def _quote_in_pages(row: dict, page_text: dict[tuple[str, int], str]) -> bool:
    quote = normalize_source_text(row.get("quoted_text"))
    source_document_id = row.get("source_document_id")
    if not quote or not isinstance(source_document_id, str):
        return False
    haystack = " ".join(
        normalize_source_text(page_text.get((source_document_id, page_number), "")) for page_number in row.get("page_numbers") or ()
    )
    return quote in haystack


def _span_sequence_grounding_health(
    *,
    metadata_grounding: list[dict],
    evidence: list[dict],
    legal_units: list[dict],
    chunks: list[dict],
    bbox_rows: list[dict],
    word_bboxes: list[dict],
    page_text_spans: list[dict],
) -> dict:
    span_rows = [row for row in page_text_spans if row.get("text_span_id")]
    active_units = [row for row in legal_units if row.get("status") in {"final", "finalizable"}]
    active_chunks = [row for row in chunks if row.get("status") == "active_canonical_record"]
    units_without_spans = [row for row in active_units if not row.get("text_span_ids")]
    chunks_without_spans = [row for row in active_chunks if not row.get("text_span_ids")]
    bbox_by_id = {row["bbox_id"]: row for row in bbox_rows} | {
        row["word_bbox_id"]: {
            "bbox_id": row["word_bbox_id"],
            "bbox_precision": "exact",
            "viewer_highlightable": True,
            **row,
        }
        for row in word_bboxes
    }
    page_metadata = [row for row in metadata_grounding if row.get("bbox_precision") == "page_grounded_only"]
    page_evidence = [row for row in evidence if row.get("bbox_precision") == "page_grounded_only"]
    bbox_ids = set(bbox_by_id)
    invalid_bbox_refs = [
        bbox_id
        for row in (*metadata_grounding, *evidence)
        for bbox_id in row.get("bbox_refs") or ()
        if bbox_id not in bbox_ids and row.get("bbox_precision") == "exact"
    ]
    invalid_coordinates = [
        row["bbox_id"]
        for row in bbox_rows
        if row.get("bbox_precision") == "exact" and not all(row.get(key) is not None for key in ("x0", "y0", "x1", "y1"))
    ]
    return {
        "fixable_page_grounded_metadata_count": sum(
            1 for row in page_metadata if _can_match_span_sequence(row, span_rows) and _has_exact_bbox_refs(row, bbox_by_id)
        ),
        "unresolved_page_grounded_metadata_count": len(page_metadata),
        "active_legal_units_without_span_ids": len(units_without_spans),
        "active_chunks_without_span_ids": len(chunks_without_spans),
        "fixable_legal_units_without_span_ids": sum(1 for row in units_without_spans if _can_match_span_sequence(row, span_rows)),
        "fixable_chunks_without_span_ids": sum(1 for row in chunks_without_spans if _can_match_span_sequence(row, span_rows)),
        "page_grounded_evidence_without_failure_reason": sum(1 for row in page_evidence if not row.get("failure_reason")),
        "false_exact_metadata_claims": sum(
            1
            for row in metadata_grounding
            if row.get("bbox_precision") == "exact" and (not row.get("bbox_refs") or not row.get("viewer_highlightable"))
        ),
        "invalid_bbox_refs": len(invalid_bbox_refs),
        "invalid_bbox_coordinates": len(invalid_coordinates),
        "untracked_grounding_exception_count": sum(
            1
            for row in (*page_metadata, *units_without_spans, *chunks_without_spans, *page_evidence)
            if not (row.get("failure_reason") or row.get("provenance_exception_category") or row.get("grounding_status"))
        ),
    }


def _can_match_span_sequence(row: dict, spans: list[dict]) -> bool:
    text = row.get("quoted_text") or row.get("text")
    target = normalize_source_text(text)
    if not target:
        return False
    pages = set(row.get("page_numbers") or range(int(row.get("page_start") or 0), int(row.get("page_end") or -1) + 1))
    rows = [span for span in spans if span.get("source_document_id") == row.get("source_document_id") and span.get("page_number") in pages]
    for start in range(len(rows)):
        joined = ""
        for span in rows[start:]:
            joined = normalize_source_text(f"{joined} {span.get('text', '')}")
            if joined == target:
                return True
            if len(joined) > len(target) + 80 or not target.startswith(joined):
                break
    return False


def _has_exact_bbox_refs(row: dict, bbox_by_id: dict[str, dict]) -> bool:
    refs = row.get("bbox_refs") or ()
    return bool(refs) and all(
        (bbox := bbox_by_id.get(ref))
        and bbox.get("bbox_precision") == "exact"
        and all(bbox.get(key) is not None for key in ("x0", "y0", "x1", "y1"))
        for ref in refs
    )


def _promotion_engine_health(
    *,
    evidence: list[dict],
    metadata_grounding: list[dict],
    bbox_rows: list[dict],
    word_bboxes: list[dict],
    legal_units: list[dict],
    chunks: list[dict],
    page_text_spans: list[dict],
    pages: list[dict],
) -> dict:
    bbox_by_id = {row["bbox_id"]: row for row in bbox_rows} | {
        row["word_bbox_id"]: {
            "bbox_id": row["word_bbox_id"],
            "bbox_precision": "exact",
            "viewer_highlightable": True,
            **row,
        }
        for row in word_bboxes
    }
    span_rows = [row for row in page_text_spans if row.get("text_span_id")]
    page_text = {(row["source_document_id"], row["page_number"]): row.get("text", "") for row in pages}
    unit_by_id = {row["legal_unit_id"]: row for row in legal_units}
    chunk_by_unit = {row["legal_unit_id"]: row for row in chunks}
    non_exact_evidence = [row for row in evidence if row.get("bbox_precision") != "exact" or row.get("viewer_highlightable") is not True]
    non_exact_metadata = [
        row for row in metadata_grounding if row.get("bbox_precision") != "exact" or row.get("viewer_highlightable") is not True
    ]
    non_exact_bbox = [row for row in bbox_rows if row.get("bbox_precision") != "exact" or row.get("viewer_highlightable") is not True]
    promotable_exact = [
        row
        for row in (*non_exact_evidence, *non_exact_metadata)
        if _quote_in_pages(row, page_text) and _can_match_span_sequence(row, span_rows) and _has_exact_bbox_refs(row, bbox_by_id)
    ]
    exact_evidence = [row for row in evidence if row.get("bbox_precision") == "exact"]
    exact_metadata = [row for row in metadata_grounding if row.get("bbox_precision") == "exact"]
    false_exact = [
        row
        for row in (*exact_evidence, *exact_metadata)
        if not row.get("text_span_ids") or not _has_exact_bbox_refs(row, bbox_by_id) or row.get("viewer_highlightable") is not True
    ]
    exact_bbox_rows = [row for row in bbox_rows if row.get("bbox_precision") == "exact"]
    invalid_exact_bbox_refs = [
        bbox_id for row in (*exact_evidence, *exact_metadata) for bbox_id in row.get("bbox_refs") or () if bbox_id not in bbox_by_id
    ]
    invalid_exact_coordinates = [
        row
        for row in exact_bbox_rows
        if not (
            all(row.get(key) is not None for key in ("x0", "y0", "x1", "y1"))
            and row.get("x1", 0) >= row.get("x0", 0)
            and row.get("y1", 0) >= row.get("y0", 0)
        )
    ]
    missing_reason = [
        row
        for row in (*non_exact_evidence, *non_exact_metadata, *non_exact_bbox)
        if not (row.get("failure_reason") or row.get("rejection_reason") or row.get("grounding_status"))
    ]
    containing_overclaims = []
    for row in exact_evidence:
        unit = unit_by_id.get(row.get("legal_unit_id"), {})
        chunk = chunk_by_unit.get(row.get("legal_unit_id"), {})
        if row.get("viewer_highlightable") is True and "text_span_containing_match" in {
            unit.get("grounding_status"),
            chunk.get("grounding_status"),
        }:
            containing_overclaims.append(row)
    counts = {
        "evidence_exact_count": len(exact_evidence),
        "evidence_page_grounded_only_count": sum(1 for row in evidence if row.get("bbox_precision") == "page_grounded_only"),
        "evidence_trace_only_count": sum(1 for row in evidence if row.get("failure_reason") == "instrument_trace_only_not_public_citation"),
        "bbox_exact_count": len(exact_bbox_rows),
        "bbox_page_grounded_only_count": sum(1 for row in bbox_rows if row.get("bbox_precision") == "page_grounded_only"),
        "bbox_non_highlightable_count": sum(1 for row in bbox_rows if row.get("viewer_highlightable") is not True),
        "metadata_grounding_exact_count": len(exact_metadata),
        "metadata_grounding_page_grounded_only_count": sum(
            1 for row in metadata_grounding if row.get("bbox_precision") == "page_grounded_only"
        ),
        "promotable_exact_count": len(promotable_exact),
        "promotion_blocked_count": len(non_exact_evidence) + len(non_exact_metadata) + len(non_exact_bbox),
        "missing_promotion_reason_count": len(missing_reason),
        "false_exact_claim_count": len(false_exact),
        "invalid_bbox_ref_count": len(invalid_exact_bbox_refs),
        "invalid_bbox_coordinate_count": len(invalid_exact_coordinates),
        "containing_span_exact_overclaim_count": len(containing_overclaims),
    }
    error_keys = {
        "promotable_exact_count",
        "missing_promotion_reason_count",
        "false_exact_claim_count",
        "invalid_bbox_ref_count",
        "invalid_bbox_coordinate_count",
        "containing_span_exact_overclaim_count",
    }
    return {**counts, "status": "complete" if not any(counts[key] for key in error_keys) else "incomplete"}


def _promotion_decision_audit_health(
    *,
    evidence: list[dict],
    metadata_grounding: list[dict],
    bbox_rows: list[dict],
    promotion_decisions: list[dict],
    promotion_engine_health: dict,
    article_relations: list[dict] | tuple[dict, ...] = (),
) -> dict:
    expected = {
        ("evidence", row["evidence_id"])
        for row in evidence
        if row.get("bbox_precision") != "exact"
        or row.get("viewer_highlightable") is not True
        or row.get("failure_reason") == "instrument_trace_only_not_public_citation"
        or row.get("promotion_candidate") is True
    }
    expected |= {
        ("metadata_grounding", row["metadata_grounding_id"])
        for row in metadata_grounding
        if row.get("bbox_precision") != "exact" or row.get("viewer_highlightable") is not True
    }
    expected |= {("article_relation", row["relation_id"]) for row in article_relations}
    expected |= {
        ("bbox", row["bbox_id"])
        for row in bbox_rows
        if row.get("bbox_precision") != "exact" or row.get("viewer_highlightable") is not True or row.get("promotion_candidate") is True
    }
    actual = {(row.get("record_type"), row.get("record_id")) for row in promotion_decisions}
    duplicate_decision_ids = len(promotion_decisions) - len({row.get("decision_id") for row in promotion_decisions})
    blocked = [row for row in promotion_decisions if row.get("decision") == "keep_non_exact"]
    attempted = [row for row in promotion_decisions if row.get("promotion_attempted") is True]
    false_exact = [
        row
        for row in promotion_decisions
        if row.get("decision") == "promote_exact"
        and not (row.get("exact_quote_available") and row.get("exact_span_available") and row.get("exact_bbox_available"))
    ]
    required_fields = {
        "promotion_attempt_method",
        "promotion_attempt_result",
        "quote_match_status",
        "span_match_status",
        "subspan_match_status",
        "bbox_union_status",
        "matched_span_ids",
        "matched_page_numbers",
        "matched_text_excerpt",
        "field_bbox_feasibility",
        "metadata_exact_promotion_feasibility",
        "blocker_evidence",
        "can_be_exact_citation",
        "can_be_exact_highlight",
    }
    generic_reasons = {"non_exact_grounding", "page_grounded_only", "field_level_grounded", "blocked", "insufficient_evidence"}
    counts = {
        "promotion_decision_count": len(promotion_decisions),
        "expected_promotion_decision_count": len(expected),
        "blocked_decision_count": len(blocked),
        "promotion_blocked_count": len(blocked),
        "promotion_attempted_count": len(attempted),
        "promotion_attempt_missing_count": len(promotion_decisions) - len(attempted),
        "exact_quote_match_count": sum(1 for row in promotion_decisions if row.get("quote_match_status") == "exact_full_quote_match"),
        "span_sequence_candidate_count": sum(
            1 for row in promotion_decisions if row.get("span_match_status") == "normalized_span_sequence_match"
        ),
        "subspan_match_candidate_count": sum(1 for row in promotion_decisions if row.get("subspan_match_status") == "matched"),
        "bbox_union_candidate_count": sum(1 for row in promotion_decisions if row.get("bbox_union_status") == "bbox_union_available"),
        "bbox_union_not_supported_count": sum(
            1 for row in promotion_decisions if row.get("bbox_union_status") == "not_supported_by_current_bbox_artifact"
        ),
        "new_exact_promotion_count": sum(1 for row in promotion_decisions if row.get("decision") == "promote_exact"),
        "kept_non_exact_after_attempt_count": len(blocked),
        "generic_blocker_reason_count": sum(1 for row in blocked if row.get("failure_reason") in generic_reasons),
        "false_highlightable_claim_count": sum(
            1
            for row in promotion_decisions
            if row.get("record_type") != "article_relation"
            and row.get("highlightable") is True
            and row.get("can_be_exact_highlight") is not True
        ),
        "missing_feasibility_field_count": sum(1 for row in promotion_decisions if required_fields - set(row)),
        "missing_decision_count": len(expected - actual),
        "unexpected_decision_count": len(actual - expected),
        "duplicate_decision_id_count": duplicate_decision_ids,
        "blocked_decision_missing_reason_count": sum(1 for row in blocked if not row.get("failure_reason")),
        "false_exact_decision_count": len(false_exact),
    }
    return {
        **counts,
        "status": "complete"
        if counts["promotion_decision_count"] == counts["expected_promotion_decision_count"]
        and counts["blocked_decision_count"] == counts["promotion_blocked_count"]
        and not any(
            counts[key]
            for key in (
                "missing_decision_count",
                "unexpected_decision_count",
                "duplicate_decision_id_count",
                "blocked_decision_missing_reason_count",
                "false_exact_decision_count",
                "promotion_attempt_missing_count",
                "generic_blocker_reason_count",
                "false_highlightable_claim_count",
                "missing_feasibility_field_count",
            )
        )
        else "incomplete",
    }


def _metadata_exact_promotion_feasibility_health(*, promotion_decisions: list[dict]) -> dict:
    metadata_rows = [row for row in promotion_decisions if row.get("record_type") == "metadata_grounding"]
    field_feasibility_categories = {
        "exact_safe",
        "line_level_only",
        "sentence_extends_beyond_field",
        "page_level_only",
        "requires_word_level_bbox",
        "blocked_by_layout",
        "blocked_by_text_boundary",
    }
    categories = {
        "promotable_exact",
        "exact_span_found_but_bbox_missing",
        "multi_span_exact_possible",
        "page_level_only_by_policy",
        "blocked_by_text_boundary",
        "blocked_by_no_exact_bbox",
        "blocked_by_layout",
    }
    counts = {
        "audited_metadata_row_count": len(metadata_rows),
        **{
            f"{category}_count": sum(1 for row in metadata_rows if row.get("metadata_exact_promotion_feasibility") == category)
            for category in sorted(categories)
        },
        "metadata_decision_sentence_continues_beyond_field_count": sum(
            1 for row in metadata_rows if row.get("failure_reason") == "metadata_decision_sentence_continues_beyond_field"
        ),
        "metadata_publication_block_requires_page_level_support_count": sum(
            1 for row in metadata_rows if row.get("failure_reason") == "metadata_publication_block_requires_page_level_support"
        ),
        "missing_feasibility_count": sum(1 for row in metadata_rows if row.get("metadata_exact_promotion_feasibility") not in categories),
        "missing_final_reason_count": sum(1 for row in metadata_rows if not row.get("failure_reason")),
        "missing_field_bbox_feasibility_count": sum(
            1 for row in metadata_rows if row.get("field_bbox_feasibility") not in field_feasibility_categories
        ),
        "field_bbox_feasibility_counts": dict(sorted(Counter(row.get("field_bbox_feasibility") for row in metadata_rows).items())),
    }
    return {
        **counts,
        "status": "complete"
        if counts["missing_feasibility_count"] == 0
        and counts["missing_final_reason_count"] == 0
        and counts["missing_field_bbox_feasibility_count"] == 0
        else "incomplete",
    }


def _all_text_disposition_health(
    page_text_spans: list[dict],
    legal_units: list[dict],
    chunks: list[dict],
    metadata_grounding: list[dict],
    source_conflicts: list[dict],
) -> dict:
    referenced_span_ids = {text_span_id for row in (*legal_units, *chunks) for text_span_id in row.get("text_span_ids") or ()}
    span_ids = {row["text_span_id"] for row in page_text_spans}
    legal_targets = {row["legal_unit_id"] for row in legal_units} | {row["chunk_id"] for row in chunks}
    metadata_targets = {row["metadata_grounding_id"] for row in metadata_grounding}
    conflict_targets = {row["source_conflict_id"] for row in source_conflicts}
    missing_fields = [row for row in page_text_spans if any(field not in row for field in SPAN_DISPOSITION_FIELDS)]
    excluded_missing_reason = [
        row for row in page_text_spans if row.get("promotion_status") in EXCLUDED_STATUSES and not row.get("exclusion_reason")
    ]
    fake_grounding_ids = [
        row
        for row in page_text_spans
        if row.get("promotion_status") in PROMOTED_STATUSES and not _target_exists(row, legal_targets, metadata_targets, conflict_targets)
    ]
    needs_review_rows = [row for row in page_text_spans if row.get("promotion_status") == "needs_review"]
    missing_source_refs = [row for row in page_text_spans if not row.get("source_document_id")]
    missing_page_refs = [row for row in page_text_spans if not isinstance(row.get("page_number"), int)]
    missing_bbox_coordinates = [row for row in page_text_spans if any(key not in row for key in ("x0", "y0", "x1", "y1"))]
    invalid_bbox_coordinates = [row for row in page_text_spans if not _valid_coordinates(row)]
    invalid_span_roles = [row for row in page_text_spans if row.get("span_role") not in SPAN_ROLES]
    invalid_semantic_classifications = [
        row for row in page_text_spans if row.get("semantic_classification") not in SEMANTIC_CLASSIFICATIONS
    ]
    invalid_legal_forces = [row for row in page_text_spans if row.get("legal_force") not in LEGAL_FORCES]
    invalid_promotion_statuses = [row for row in page_text_spans if row.get("promotion_status") not in PROMOTION_STATUSES]
    invalid_review_statuses = [row for row in page_text_spans if row.get("review_status") not in REVIEW_STATUSES]
    ambiguous_dispositions = [
        row
        for row in page_text_spans
        if row.get("promotion_status") not in PROMOTED_STATUSES
        and row.get("promotion_target_id")
        and row.get("promotion_target_type") not in {"legal_unit", "chunk"}
    ]
    return {
        "page_text_span_count": len(page_text_spans),
        "classified_span_count": sum(1 for row in page_text_spans if row.get("semantic_classification") in SEMANTIC_CLASSIFICATIONS),
        "span_disposition_present_count": len(page_text_spans) - len(missing_fields),
        "span_disposition_missing_count": len(missing_fields),
        "semantic_classification_present_count": sum(1 for row in page_text_spans if bool(row.get("semantic_classification"))),
        "known_unreferenced_span_count": len(span_ids - referenced_span_ids),
        "promotion_status_present_count": sum(1 for row in page_text_spans if "promotion_status" in row),
        "legal_force_present_count": sum(1 for row in page_text_spans if "legal_force" in row),
        "missing_source_ref_count": len(missing_source_refs),
        "missing_page_ref_count": len(missing_page_refs),
        "missing_bbox_coordinate_count": len(missing_bbox_coordinates),
        "invalid_bbox_coordinate_count": len(invalid_bbox_coordinates),
        "invalid_span_role_count": len(invalid_span_roles),
        "invalid_semantic_classification_count": len(invalid_semantic_classifications),
        "invalid_legal_force_count": len(invalid_legal_forces),
        "invalid_promotion_status_count": len(invalid_promotion_statuses),
        "invalid_review_status_count": len(invalid_review_statuses),
        "ambiguous_disposition_count": len(ambiguous_dispositions),
        "exclusion_reason_missing_for_excluded_count": len(excluded_missing_reason),
        "needs_review_count": len(needs_review_rows),
        "runtime_loadable_needs_review_count": sum(1 for row in needs_review_rows if row.get("runtime_loadable") is True),
        "canonical_use_allowed_needs_review_count": sum(1 for row in needs_review_rows if row.get("canonical_use_allowed") is True),
        "fake_grounding_id_count": len(fake_grounding_ids),
        "status": "complete"
        if page_text_spans
        and not missing_fields
        and not excluded_missing_reason
        and not fake_grounding_ids
        and not missing_source_refs
        and not missing_page_refs
        and not missing_bbox_coordinates
        and not invalid_bbox_coordinates
        and not invalid_span_roles
        and not invalid_semantic_classifications
        and not invalid_legal_forces
        and not invalid_promotion_statuses
        and not invalid_review_statuses
        and not ambiguous_dispositions
        else "incomplete",
    }


def _valid_coordinates(row: dict) -> bool:
    try:
        x0 = float(row["x0"])
        y0 = float(row["y0"])
        x1 = float(row["x1"])
        y1 = float(row["y1"])
    except (KeyError, TypeError, ValueError):
        return False
    return x0 < x1 and y0 < y1


def _target_exists(row: dict, legal_targets: set[str], metadata_targets: set[str], conflict_targets: set[str]) -> bool:
    target_type = row.get("promotion_target_type")
    target_id = row.get("promotion_target_id")
    if target_type in {"legal_unit", "chunk"}:
        return target_id in legal_targets
    if target_type == "metadata_grounding":
        return target_id in metadata_targets
    if target_type == "source_conflict":
        return target_id in conflict_targets
    return False
