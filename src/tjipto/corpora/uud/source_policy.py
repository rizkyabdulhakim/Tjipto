from __future__ import annotations

import re
import unicodedata


SOURCE_MARKER_RE = re.compile(r"\*{1,4}(?:/\*{1,4})?\)")
NORMALIZATION_CONTRACT = "NFKC;NBSP=SPACE;SOFT_HYPHEN=REMOVE;UUD_SOURCE_MARKERS=PROVENANCE_ONLY;LINE_SEPARATOR=LF"


def normalize_semantic_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "").replace("\xa0", " ").replace("\u00ad", "")
    value = SOURCE_MARKER_RE.sub("", value)
    return value if not value.startswith("Dihapus.") else value.split("****", 1)[0].strip()


def segment_source_line(raw_text: str) -> list[dict]:
    matches = list(SOURCE_MARKER_RE.finditer(raw_text))
    boundaries = [0] + [point for match in matches for point in (match.start(), match.end())] + [len(raw_text)]
    segments = []
    for start, end in zip(boundaries, boundaries[1:]):
        if start == end:
            continue
        marker = bool(SOURCE_MARKER_RE.fullmatch(raw_text[start:end]))
        segments.append({
            "start": start,
            "end": end,
            "classification": "source_annotation_marker" if marker else "source_text",
            "legal_text": not marker,
            "citation_eligible": not marker,
            "relevant_quote_eligible": not marker,
            "default_highlight_eligible": not marker,
            "normalization_actions": [{"action": "remove_source_annotation_marker", "start": start, "end": end, "quote": raw_text[start:end]}] if marker else [],
            "disposition_reason": "source_annotation_marker" if marker else ("semantic_projection" if not matches else "source_annotation_marker_removed"),
        })
    return segments
