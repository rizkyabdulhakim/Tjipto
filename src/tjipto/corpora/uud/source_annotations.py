from __future__ import annotations

from collections import defaultdict, deque
import re
import unicodedata

from tjipto.contracts.source_text import (
    SourceAnnotation,
    SourceAnnotationOccurrence,
    SourceSelector,
    SourceTextQueryResult,
)


_ANNOTATION_QUERY_TERMS = ("arti tanda", "arti marker", "perbedaan *", "diberi tanda", "marker gabungan")
_ORIGIN_QUERY = re.compile(r"\bpasal\s+mana\b.*\bberasal\s+dari\s+(perubahan\s+\w+)\b", re.IGNORECASE)
_PASAL_QUERY = re.compile(r"\bpasal\s+([0-9]+[a-z]?)\b", re.IGNORECASE)


def query_source_annotations(store, query: str) -> SourceTextQueryResult | None:
    normalized = _normalize(query)
    origin = _ORIGIN_QUERY.search(normalized)
    if not origin and not any(term in normalized for term in _ANNOTATION_QUERY_TERMS):
        return None
    legends = _verified_legends(store)
    if not legends:
        return None

    requested = _requested_markers(store, query, legends)
    if origin:
        requested = tuple(marker for marker, annotation in legends.items() if _normalize(annotation.meaning) == origin.group(1))
    if not requested:
        return None

    annotations = tuple(legends[marker] for marker in requested)
    pasal = _PASAL_QUERY.search(normalized)
    supports = tuple(_legend_support(store, annotation) for annotation in annotations)
    meanings = "; ".join(f"{annotation.marker} berarti {annotation.meaning}" for annotation in annotations)
    if origin:
        answer = (
            f"Legenda sumber menjelaskan {meanings}, tetapi target pasal yang tepat untuk "
            f"{origin.group(1).title()} tidak dapat dipastikan dari naskah sumber."
        )
    elif pasal:
        answer = f"Legenda menjelaskan {meanings}, tetapi keterkaitannya dengan {pasal.group(0).title()} tidak dapat dipastikan."
    else:
        answer = f"Legenda sumber menjelaskan: {meanings}."
    return SourceTextQueryResult("source_annotation", answer, annotations, supports)


def annotation_health(store) -> dict[str, int]:
    rows = tuple(_artifact_rows(store))
    markers = [row for row in rows if row.get("classification") == "source_annotation_marker"]
    occurrences = source_annotation_occurrences(store)
    valid_targets = {str(row.get("legal_unit_id")) for row in store.legal_units}
    return {
        "raw_nonempty_source_span_count": sum(bool(str(row.get("raw_text") or "").strip()) for row in rows),
        "source_annotation_occurrence_count": len(markers),
        "unmapped_source_annotation_count": len(markers) - len(occurrences),
        "ordinary_punctuation_annotation_count": sum(
            1 for row in markers if str(row.get("raw_text") or "").strip() == ":"
        ),
        "source_annotation_legal_citation_count": sum(1 for row in markers if row.get("citation_eligible") is True),
        "source_annotation_default_highlight_count": sum(
            1 for row in markers if row.get("default_highlight_eligible") is True
        ),
        "source_annotation_occurrence_without_selector_or_geometry_count": len(markers) - len(occurrences),
        "source_annotation_occurrence_without_target_or_reason_count": sum(
            not occurrence.target_legal_unit_id and not occurrence.target_reason for occurrence in occurrences
        ),
        "fabricated_annotation_target_count": sum(
            bool(row.get("target_legal_unit_id"))
            and (
                row.get("annotation_target_basis") != "exact_source_selector"
                or str(row.get("target_legal_unit_id")) not in valid_targets
            )
            for row in markers
        ),
    }


def document_annotations(store) -> tuple[SourceAnnotation, ...]:
    return tuple(_verified_legends(store).values())


def source_annotation_occurrences(store) -> tuple[SourceAnnotationOccurrence, ...]:
    legends = _verified_legends(store)
    legend_selectors = {
        (annotation.selector.stream_id, annotation.selector.start, annotation.selector.end)
        for annotation in legends.values()
    }
    legal_unit_ids = {str(row.get("legal_unit_id")) for row in store.legal_units}
    occurrences = []
    for row in _artifact_rows(store):
        if row.get("classification") != "source_annotation_marker":
            continue
        marker_parts = _marker_parts(str(row.get("raw_text") or ""), legends)
        if not marker_parts or any(row.get(field) is None for field in ("x0", "y0", "x1", "y1")):
            continue
        selector = SourceSelector(
            str(row.get("raw_stream_id") or ""),
            int(row.get("raw_text_start") or 0),
            int(row.get("raw_text_end") or 0),
        )
        requested_target = str(row.get("target_legal_unit_id") or "")
        exact_target = (
            requested_target
            if row.get("annotation_target_basis") == "exact_source_selector" and requested_target in legal_unit_ids
            else None
        )
        selector_key = (selector.stream_id, selector.start, selector.end)
        occurrences.append(
            SourceAnnotationOccurrence(
                occurrence_id=str(row.get("raw_source_span_id") or ""),
                marker=str(row.get("raw_text") or ""),
                legend_markers=marker_parts,
                source_document_id=str(row.get("source_document_id") or ""),
                source_sha256=str(row.get("source_sha256") or ""),
                page_number=int(row.get("page_number") or 0),
                extraction_order=int(row.get("extraction_order") or 0),
                selector=selector,
                geometry=(float(row["x0"]), float(row["y0"]), float(row["x1"]), float(row["y1"])),
                target_legal_unit_id=exact_target,
                target_reason=None
                if exact_target
                else "legend_definition"
                if selector_key in legend_selectors
                else "needs_review"
                if requested_target
                else "ambiguous_target",
            )
        )
    return tuple(occurrences)


def _verified_legends(store) -> dict[str, SourceAnnotation]:
    candidates: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    window: deque[dict] = deque(maxlen=3)
    identity = None
    for row in _artifact_rows(store):
        row_identity = (row.get("source_document_id"), row.get("page_number"))
        if row_identity != identity:
            window.clear()
            identity = row_identity
        window.append(row)
        if len(window) == 3:
            marker_row, separator, meaning_row = window
            if (
                marker_row.get("classification") == "source_annotation_marker"
                and separator.get("raw_text") == ":"
                and _normalize(str(meaning_row.get("semantic_text") or "")).startswith("perubahan ")
            ):
                candidates[str(marker_row["raw_text"])].append((marker_row, meaning_row))
    result: dict[str, SourceAnnotation] = {}
    for marker_value, rows in candidates.items():
        meanings = {_normalize(str(meaning.get("semantic_text") or "")) for _, meaning in rows}
        if len(meanings) != 1:
            continue
        marker_row, meaning_row = rows[0]
        meaning = str(meaning_row["semantic_text"])
        related_roles = tuple(
            sorted(
                str(document["source_role"])
                for document in store.source_documents
                if meaning.casefold() in str(document.get("document_title") or "").casefold()
            )
        )
        result[marker_value] = SourceAnnotation(
            marker=marker_value,
            meaning=meaning,
            source_document_id=str(marker_row["source_document_id"]),
            source_sha256=str(marker_row["source_sha256"]),
            page_number=int(marker_row["page_number"]),
            selector=SourceSelector(
                str(marker_row["raw_stream_id"]),
                int(marker_row["raw_text_start"]),
                int(marker_row["raw_text_end"]),
            ),
            legend_support_id=str(meaning_row["source_support_id"]),
            related_source_roles=related_roles,
        )
    return result


def _requested_markers(store, query: str, legends: dict[str, SourceAnnotation]) -> tuple[str, ...]:
    if "gabungan" in _normalize(query):
        combined_value = next(
            (
                str(row.get("raw_text") or "")
                for row in _artifact_rows(store)
                if row.get("classification") == "source_annotation_marker" and "/" in str(row.get("raw_text") or "")
            ),
            "",
        )
        combined = _marker_parts(combined_value, legends)
        return combined
    explicit = tuple(f"{stars})" for stars in re.findall(r"\*+", query) if f"{stars})" in legends)
    return explicit or tuple(sorted(legends, key=len))


def _marker_parts(value: str, legends: dict[str, SourceAnnotation]) -> tuple[str, ...]:
    parts = {f"{stars})" for stars in re.findall(r"\*+", value)}
    return tuple(marker for marker in legends if marker in parts)


def _legend_support(store, annotation: SourceAnnotation) -> dict:
    row = store.source_span_for_support(annotation.legend_support_id) or {}
    source: dict = next(
        (
            document
            for document in store.source_documents
            if document.get("source_document_id") == annotation.source_document_id
        ),
        {},
    )
    return {
        "evidence_id": annotation.legend_support_id,
        "authority_kind": "source_annotation",
        "support_kind": "trace_support",
        "fact_kind": "source_annotation",
        "citation_final": False,
        "display_label": f"Catatan sumber {annotation.marker}",
        "display_text": f"{annotation.marker} berarti {annotation.meaning}.",
        "document_title": source.get("document_title"),
        "source_document_id": annotation.source_document_id,
        "source_role": row.get("source_role"),
        "page_numbers": (annotation.page_number,),
        "viewer_highlightable": row.get("source_citation_eligible") is True,
        "viewer_target": {"can_resolve": True},
    }


def _normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _artifact_rows(store):
    yield from store.raw_source_spans
