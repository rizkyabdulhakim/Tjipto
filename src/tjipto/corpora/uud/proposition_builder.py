"""Build conservative, source-grounded textual proposition records."""

from __future__ import annotations

from hashlib import sha256
import re


_BOUNDARY = re.compile(r"[.!?;,:]")
_TOKEN = re.compile(r"[a-z0-9]+")


def build_textual_propositions(*, legal_units: list[dict], evidence: list[dict], page_text_spans: list[dict]) -> list[dict]:
    """Publish only exact textual segments; normative parsing remains fail-closed."""
    spans = {str(row.get("text_span_id")): row for row in page_text_spans}
    evidence_by_unit = {
        str(row.get("legal_unit_id")): row
        for row in evidence
        if row.get("status") == "final" and row.get("legal_unit_id")
    }
    parents = {str(row.get("parent_legal_unit_id")) for row in legal_units if row.get("unit_type") == "ayat_record"}
    rows = []
    for unit in legal_units:
        unit_id = str(unit.get("legal_unit_id") or "")
        if not unit_id or (unit.get("unit_type") != "ayat_record" and unit_id in parents):
            continue
        record = evidence_by_unit.get(unit_id)
        if record is None:
            continue
        for segment_index, segment in enumerate(_segments(unit, spans)):
            quoted = segment["exact_quote"]
            normalized = _normalize(quoted)
            if not normalized:
                continue
            identity = "\x1f".join((unit_id, str(record["evidence_id"]), str(segment_index), *segment["text_span_ids"], quoted)).encode("utf-8")
            rows.append(
                {
                    "proposition_id": f"proposition::{sha256(identity).hexdigest()}",
                    "claim_type": "textual_occurrence",
                    "legal_unit_id": unit_id,
                    "subject": unit.get("canonical_label") or unit.get("unit_label"),
                    "predicate": "mentions",
                    "object": normalized,
                    "polarity": "positive",
                    "modality": "textual",
                    "conditions": [],
                    "exceptions": [],
                    "source_role": record.get("source_role"),
                    "temporal_context": record.get("temporal_context"),
                    "evidence_id": record["evidence_id"],
                    "text_segment_id": f"segment::{sha256(identity).hexdigest()}",
                    "exact_quote": quoted,
                    "text_span_ids": segment["text_span_ids"],
                    "source_selectors": segment["source_selectors"],
                    "bbox_refs": segment["bbox_refs"],
                    "page_numbers": segment["page_numbers"],
                    "source_document_id": record.get("source_document_id"),
                    "source_sha256": record.get("source_sha256"),
                    "terminal_boundary": segment["terminal_boundary"],
                }
            )
    return sorted(rows, key=lambda row: row["proposition_id"])


def _segments(unit: dict, spans: dict[str, dict]) -> tuple[dict, ...]:
    selected = [spans[span_id] for span_id in unit.get("text_span_ids") or () if span_id in spans]
    if not selected:
        return ()
    text = ""
    mapping: list[tuple[str, int] | None] = []
    for index, span in enumerate(selected):
        if index:
            text += " "
            mapping.append(None)
        value = str(span.get("text") or "")
        text += value
        mapping.extend((str(span["text_span_id"]), offset) for offset in range(len(value)))
    rows = []
    start = 0
    for match in _BOUNDARY.finditer(text):
        end = match.end()
        row = _segment(text, mapping, selected, start, end, match.group())
        if row:
            rows.append(row)
        start = end
    row = _segment(text, mapping, selected, start, len(text), "source_end")
    if row:
        rows.append(row)
    return tuple(rows)


def _segment(text: str, mapping: list[tuple[str, int] | None], spans: list[dict], start: int, end: int, boundary: str) -> dict | None:
    quote = text[start:end].strip()
    if not _normalize(quote):
        return None
    ids = []
    selectors: dict[str, list[int]] = {}
    for item in mapping[start:end]:
        if item is None:
            continue
        span_id, offset = item
        if span_id not in selectors:
            selectors[span_id] = [offset, offset + 1]
            ids.append(span_id)
        else:
            selectors[span_id][1] = offset + 1
    if not ids:
        return None
    span_by_id = {str(row["text_span_id"]): row for row in spans}
    return {
        "exact_quote": quote,
        "text_span_ids": ids,
        "source_selectors": [
            {"text_span_id": span_id, "start": offsets[0], "end": offsets[1]}
            for span_id, offsets in selectors.items()
        ],
        "bbox_refs": list(dict.fromkeys(ref for span_id in ids for ref in span_by_id[span_id].get("span_bbox_ids") or ())),
        "page_numbers": sorted({int(span_by_id[span_id]["page_number"]) for span_id in ids}),
        "terminal_boundary": boundary,
    }


def _normalize(text: str) -> str:
    return " ".join(_TOKEN.findall(text.casefold()))
