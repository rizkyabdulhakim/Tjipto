"""Build source-span-grounded textual and explicit normative propositions."""

from __future__ import annotations

from hashlib import sha256
import re


_BOUNDARY = re.compile(r"[.!?;,:]")
_TOKEN = re.compile(r"[a-z0-9]+")
_NORMATIVE = re.compile(r"^(?P<subject>.+?)\s+(?P<operator>wajib|dilarang)\s+(?P<object>.+)$", re.IGNORECASE)
_OPERATORS = {
    "wajib": ("requires", "positive", "obligation"),
    "dilarang": ("prohibits", "positive", "prohibition"),
}


def build_propositions(*, legal_units: list[dict], evidence: list[dict], page_text_spans: list[dict]) -> list[dict]:
    """Publish only complete, BBox-backed source clauses."""
    spans = {str(row.get("text_span_id")): row for row in page_text_spans}
    evidence_by_span: dict[str, list[dict]] = {}
    for evidence_row in evidence:
        if evidence_row.get("status") != "final":
            continue
        for span_id in evidence_row.get("text_span_ids") or ():
            evidence_by_span.setdefault(str(span_id), []).append(evidence_row)
    # A parent legal unit is retrieval context, not another source clause.
    # Emit propositions only from leaf units so the same source sentence
    # cannot be republished through a BAB/Pasal and one of its Ayat.
    parents = {str(row.get("parent_legal_unit_id")) for row in legal_units if row.get("parent_legal_unit_id")}
    rows = []
    for unit in legal_units:
        unit_id = str(unit.get("legal_unit_id") or "")
        if not unit_id or (unit.get("unit_type") != "ayat_record" and unit_id in parents):
            continue
        record: dict | None = _evidence_for_unit(unit, evidence_by_span)
        if record is None:
            continue
        for segment_index, segment in enumerate(_segments(unit, spans)):
            quoted = segment["exact_quote"]
            normalized = _normalize(quoted)
            if not normalized or not segment["bbox_refs"]:
                continue
            common = _common_record(unit_id, record, segment_index, segment)
            rows.append(common | {
                "proposition_id": _proposition_id("textual_occurrence", common),
                "claim_type": "textual_occurrence",
                "subject": unit.get("canonical_label") or unit.get("unit_label"),
                "predicate": "mentions",
                "object": normalized,
                "polarity": "positive",
                "modality": "textual",
                "conditions": [],
                "exceptions": [],
            })
            normative = _normative_proposition(quoted)
            if normative:
                rows.append(common | {
                    "proposition_id": _proposition_id("normative_proposition", common),
                    "claim_type": "normative_proposition",
                    **normative,
                })
    return sorted(rows, key=lambda row: row["proposition_id"])


def _evidence_for_unit(unit: dict, evidence_by_span: dict[str, list[dict]]) -> dict | None:
    """Select the final source record that owns this unit's exact spans."""
    span_ids = tuple(str(span_id) for span_id in unit.get("text_span_ids") or ())
    if not span_ids:
        return None
    candidates = {
        str(record.get("evidence_id")): record
        for span_id in span_ids
        for record in evidence_by_span.get(span_id, ())
        if record.get("source_role") == unit.get("source_role")
        and record.get("source_document_id") == unit.get("source_document_id")
    }
    if not candidates:
        return None
    return min(
        candidates.values(),
        key=lambda record: (-len(set(span_ids) & set(record.get("text_span_ids") or ())), str(record["evidence_id"])),
    )


def _segments(unit: dict, spans: dict[str, dict]) -> tuple[dict, ...]:
    selected = [spans[span_id] for span_id in unit.get("text_span_ids") or () if span_id in spans]
    if not selected:
        return ()
    text_parts: list[str] = []
    mapping: list[tuple[str, int] | None] = []
    for index, span in enumerate(selected):
        if index:
            text_parts.append("\n")
            mapping.append(None)
        text = str(span.get("text") or "")
        text_parts.append(text)
        mapping.extend((str(span["text_span_id"]), offset) for offset in range(len(text)))
    text = "".join(text_parts)
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


def _common_record(unit_id: str, record: dict, segment_index: int, segment: dict) -> dict:
    identity = "\x1f".join((unit_id, str(record["evidence_id"]), str(segment_index), *segment["text_span_ids"], segment["exact_quote"])).encode("utf-8")
    digest = sha256(identity).hexdigest()
    return {
        "legal_unit_id": unit_id,
        "evidence_id": record["evidence_id"],
        "text_segment_id": f"segment::{digest}",
        "exact_quote": segment["exact_quote"],
        "text_span_ids": segment["text_span_ids"],
        "source_selectors": segment["source_selectors"],
        "bbox_refs": segment["bbox_refs"],
        "page_numbers": segment["page_numbers"],
        "source_document_id": record.get("source_document_id"),
        "source_sha256": record.get("source_sha256"),
        "source_role": record.get("source_role"),
        "temporal_context": record.get("temporal_context"),
        "terminal_boundary": segment["terminal_boundary"],
    }


def _proposition_id(claim_type: str, record: dict) -> str:
    identity = "\x1f".join((claim_type, str(record["evidence_id"]), str(record["text_segment_id"]))).encode("utf-8")
    return f"proposition::{sha256(identity).hexdigest()}"


def _normative_proposition(quote: str) -> dict | None:
    match = _NORMATIVE.match(" ".join(quote.split()))
    if not match:
        return None
    predicate, polarity, modality = _OPERATORS[match["operator"].casefold()]
    subject = _normalize(match["subject"])
    object_ = _normalize(match["object"])
    if not subject or not object_:
        return None
    return {
        "subject": subject,
        "predicate": predicate,
        "object": object_,
        "polarity": polarity,
        "modality": modality,
        "conditions": [],
        "exceptions": [],
    }
