from __future__ import annotations

from collections import defaultdict, deque
from functools import lru_cache
import json
import re
import unicodedata

from tjipto.contracts.source_text import SourceAnnotation, SourceSelector, SourceTextQueryResult


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
    target_labels: tuple[str, ...] = ()
    pasal = _PASAL_QUERY.search(normalized)
    if pasal:
        label = f"Pasal {pasal.group(1).upper()}"
        target_labels = (label,) if _unit_marker(store, label) in requested else ()
    elif origin:
        target_labels = tuple(
            sorted(
                (
                    str(unit["unit_label"])
                    for unit in store.legal_units
                    if unit.get("unit_type") == "pasal_record"
                    and unit.get("source_role") == "current_consolidated"
                    and _unit_marker(store, str(unit.get("unit_label") or "")) in requested
                ),
                key=_pasal_order,
            )
        )

    supports = tuple(_legend_support(store, annotation) for annotation in annotations)
    meanings = "; ".join(f"{annotation.marker} berarti {annotation.meaning}" for annotation in annotations)
    if origin:
        answer = f"{origin.group(1).title()} ditandai oleh legenda sumber sebagai {meanings}. Pasal terkait: {', '.join(target_labels) or 'tidak dapat dipastikan'}."
    elif pasal:
        answer = (
            f"{pasal.group(0).title()} memakai anotasi sumber {meanings}."
            if target_labels
            else f"Legenda menjelaskan {meanings}, tetapi keterkaitannya dengan {pasal.group(0).title()} tidak dapat dipastikan."
        )
    else:
        answer = f"Legenda sumber menjelaskan: {meanings}."
    return SourceTextQueryResult("source_annotation", answer, annotations, supports, target_labels)


def annotation_health(store) -> dict[str, int]:
    legends = _verified_legends(store)
    rows = tuple(_artifact_rows(store))
    markers = [row for row in rows if row.get("classification") == "source_annotation_marker"]
    mapped = [row for row in markers if _marker_parts(str(row.get("raw_text") or ""), legends)]
    return {
        "raw_nonempty_source_span_count": sum(1 for row in rows if str(row.get("raw_text") or "").strip()),
        "source_annotation_occurrence_count": len(markers),
        "unmapped_source_annotation_count": len(markers) - len(mapped),
        "ordinary_punctuation_annotation_count": sum(
            1 for row in markers if str(row.get("raw_text") or "").strip() == ":"
        ),
        "source_annotation_legal_citation_count": sum(1 for row in markers if row.get("citation_eligible") is True),
        "source_annotation_default_highlight_count": sum(
            1 for row in markers if row.get("default_highlight_eligible") is True
        ),
    }


def document_annotations(store) -> tuple[SourceAnnotation, ...]:
    return tuple(_verified_legends(store).values())


@lru_cache(maxsize=1)
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


def _unit_marker(store, label: str) -> str | None:
    return _unit_markers(store).get(label)


@lru_cache(maxsize=1)
def _unit_markers(store) -> dict[str, str]:
    units = tuple(
        row
        for row in store.legal_units
        if row.get("unit_type") == "pasal_record" and row.get("source_role") == "current_consolidated"
    )
    pages = {int(row["page_start"]) for row in units}
    rows_by_page: dict[int, list[dict]] = defaultdict(list)
    for row in _artifact_rows(store):
        if row.get("source_role") == "current_consolidated" and row.get("page_number") in pages:
            rows_by_page[int(row["page_number"])].append(row)
    result: dict[str, str] = {}
    for unit in units:
        rows = rows_by_page[int(unit["page_start"])]
        unit_text = _normalize(str(unit.get("text") or ""))
        matching = [
            row
            for row in rows
            if row.get("semantic_text") and _normalize(str(row["semantic_text"])) in unit_text
        ]
        if not matching:
            continue
        last_order = max(int(row["extraction_order"]) for row in matching)
        marker = next(
            (
                str(row["raw_text"])
                for row in rows
                if int(row["extraction_order"]) == last_order + 1
                and row.get("classification") == "source_annotation_marker"
            ),
            None,
        )
        if marker:
            result[str(unit["unit_label"])] = marker
    return result


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
        "viewer_highlightable": row.get("default_highlight_eligible") is True,
        "viewer_target": {"can_resolve": True},
    }


def _pasal_order(label: str) -> tuple[int, str]:
    match = re.search(r"(\d+)([A-Z]?)", label)
    return (int(match.group(1)), match.group(2)) if match else (10_000, label)


def _normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _artifact_rows(store):
    with store.config.artifact_path("raw_source_spans").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)
