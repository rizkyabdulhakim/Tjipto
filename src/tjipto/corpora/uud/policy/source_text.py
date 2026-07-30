from __future__ import annotations

from collections import defaultdict, deque
import re


def validate_source_text_closure(rows: list[dict]) -> tuple[str, ...]:
    legends = _legend_markers(rows)
    errors = []
    for row in rows:
        row_id = str(row.get("raw_source_span_id") or "unknown")
        value = str(row.get("raw_text") or "")
        if not value.strip():
            continue
        if not row.get("raw_stream_id") or row.get("raw_text_start") is None or row.get("raw_text_end") is None:
            errors.append(f"source_text_selector_missing:{row_id}")
        if not all(row.get(field) is not None for field in ("x0", "y0", "x1", "y1")) and not row.get("disposition_reason"):
            errors.append(f"source_text_geometry_or_reason_missing:{row_id}")
        marker = row.get("classification") == "source_annotation_marker"
        if marker:
            if value.strip() == ":":
                errors.append(f"ordinary_punctuation_annotation:{row_id}")
            parts = {f"{stars})" for stars in re.findall(r"\*+", value)}
            if not parts or not parts <= legends:
                errors.append(f"source_annotation_unmapped:{row_id}")
            if any(row.get(field) is not False for field in ("legal_text", "citation_eligible", "default_highlight_eligible")):
                errors.append(f"source_annotation_authority_leak:{row_id}")
        elif not str(row.get("semantic_text") or "").strip() and not row.get("disposition_reason"):
            errors.append(f"meaningful_source_text_without_route_or_review:{row_id}")
    return tuple(errors)


def _legend_markers(rows: list[dict]) -> set[str]:
    candidates: dict[str, set[str]] = defaultdict(set)
    window: deque[dict] = deque(maxlen=3)
    identity = None
    for row in rows:
        row_identity = (row.get("source_document_id"), row.get("page_number"))
        if row_identity != identity:
            window.clear()
            identity = row_identity
        window.append(row)
        if len(window) != 3:
            continue
        marker, separator, meaning = window
        if (
            marker.get("classification") == "source_annotation_marker"
            and separator.get("raw_text") == ":"
            and str(meaning.get("semantic_text") or "").casefold().startswith("perubahan ")
        ):
            candidates[str(marker.get("raw_text") or "")].add(str(meaning.get("semantic_text") or "").casefold())
    return {marker for marker, meanings in candidates.items() if marker and len(meanings) == 1}
