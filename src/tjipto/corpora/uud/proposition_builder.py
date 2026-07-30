"""Build source-span-grounded textual and explicit normative propositions."""

from __future__ import annotations

from hashlib import sha256
import re
import unicodedata

from tjipto.evidence.bbox import (
    derive_viewer_overlay,
    geometry_space_key,
)

_BOUNDARY = re.compile(r"[.!?;,:]")
_TOKEN = re.compile(r"[a-z0-9]+")
_NORMATIVE = re.compile(r"^(?P<subject>.+?)\s+(?P<operator>wajib|dilarang)\s+(?P<object>.+)$", re.IGNORECASE)
_SOURCE_MARKER = re.compile(r"\*{1,4}(?:/\*{1,4})?\)")
_OPERATORS = {
    "wajib": ("requires", "positive", "obligation"),
    "dilarang": ("prohibits", "positive", "prohibition"),
}


def build_propositions(
    *, legal_units: list[dict], evidence: list[dict], page_text_spans: list[dict], word_bboxes: list[dict]
) -> list[dict]:
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
    words_by_id = {str(row.get("word_bbox_id")): row for row in word_bboxes}
    characters_by_id = {
        str(character["character_bbox_id"]): word | character
        for word in word_bboxes
        for character in word.get("characters") or ()
    }
    marker_boxes: dict[tuple[object, ...], list[dict]] = {}
    for marker in source_marker_character_boxes(word_bboxes):
        marker_boxes.setdefault(geometry_space_key(marker), []).append(marker)
    rows: list[dict] = []
    for unit in legal_units:
        unit_id = str(unit.get("legal_unit_id") or "")
        if not unit_id or (unit.get("unit_type") != "ayat_record" and unit_id in parents):
            continue
        record: dict | None = _evidence_for_unit(unit, evidence_by_span)
        if record is None:
            continue
        for segment_index, segment in enumerate(_segments(unit, spans, words_by_id)):
            quoted = segment["exact_quote"]
            normalized = _normalize(quoted)
            if not normalized or not segment["bbox_refs"]:
                continue
            common = _common_record(unit_id, record, segment_index, segment)
            rows.append(_with_viewer_overlay(common | {
                "proposition_id": _proposition_id("textual_occurrence", common),
                "claim_type": "textual_occurrence",
                "subject": unit.get("canonical_label") or unit.get("unit_label"),
                "predicate": "mentions",
                "object": normalized,
                "polarity": "positive",
                "modality": "textual",
                "conditions": [],
                "exceptions": [],
            }, characters_by_id, marker_boxes))
            normative = _normative_proposition(quoted)
            if normative:
                rows.append(_with_viewer_overlay(common | {
                    "proposition_id": _proposition_id("normative_proposition", common),
                    "claim_type": "normative_proposition",
                    **normative,
                }, characters_by_id, marker_boxes))
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


def _segments(unit: dict, spans: dict[str, dict], words_by_id: dict[str, dict]) -> tuple[dict, ...]:
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
        row = _segment(text, mapping, selected, words_by_id, start, end, match.group())
        if row:
            rows.append(row)
        start = end
    row = _segment(text, mapping, selected, words_by_id, start, len(text), "source_end")
    if row:
        rows.append(row)
    return tuple(rows)


def _segment(
    text: str,
    mapping: list[tuple[str, int] | None],
    spans: list[dict],
    words_by_id: dict[str, dict],
    start: int,
    end: int,
    boundary: str,
) -> dict | None:
    raw_quote = text[start:end]
    left = len(raw_quote) - len(raw_quote.lstrip())
    right = len(raw_quote.rstrip())
    start, end = start + left, start + right
    quote = text[start:end]
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
    selector_rows = []
    for span_id, offsets in selectors.items():
        span = span_by_id[span_id]
        character_ids = _selected_character_ids(span, offsets[0], offsets[1], words_by_id)
        if not character_ids:
            return None
        selector_rows.append(
            {
                "text_span_id": span_id,
                "start": offsets[0],
                "end": offsets[1],
                "stream_id": span["stream_id"],
                "absolute_start": span["text_start"] + offsets[0],
                "absolute_end": span["text_start"] + offsets[1],
                "prefix": span["text"][max(0, offsets[0] - 32) : offsets[0]],
                "suffix": span["text"][offsets[1] : offsets[1] + 32],
                "source_document_id": span["source_document_id"],
                "source_sha256": span["source_sha256"],
                "page_number": span["page_number"],
                "character_bbox_ids": character_ids,
            }
        )
    return {
        "exact_quote": quote,
        "text_span_ids": ids,
        "source_selectors": selector_rows,
        "bbox_refs": list(dict.fromkeys(char_id for selector in selector_rows for char_id in selector["character_bbox_ids"])),
        "page_numbers": sorted({int(span_by_id[span_id]["page_number"]) for span_id in ids}),
        "terminal_boundary": boundary,
    }


def _selected_character_ids(span: dict, start: int, end: int, words_by_id: dict[str, dict]) -> list[str]:
    """Return one unambiguous visible-character slice; partial quotes never inherit a span box."""
    selected = str(span.get("text") or "")[start:end]
    target = _visible(selected)
    if not target:
        return []
    characters = [
        character
        for bbox_id in span.get("span_bbox_ids") or ()
        for character in (words_by_id.get(str(bbox_id), {}).get("characters") or ())
    ]
    visible = "".join(str(character.get("text") or "") for character in characters)
    match_start = visible.find(target)
    if match_start < 0 or visible.find(target, match_start + 1) >= 0:
        return []
    return [
        str(character["character_bbox_id"])
        for character in characters[match_start : match_start + len(target)]
        if character.get("character_bbox_id")
    ]


def _visible(value: str) -> str:
    return "".join(character for character in unicodedata.normalize("NFKC", value) if not character.isspace())


def source_marker_character_boxes(word_bboxes: list[dict]) -> tuple[dict, ...]:
    """Return immutable source-marker glyph geometry without changing extraction rows."""
    rows: list[dict] = []
    for word in word_bboxes:
        for match in _SOURCE_MARKER.finditer(str(word.get("text") or "")):
            rows.extend(
                word | character
                for character in word.get("characters") or ()
                if character.get("char_start", -1) < match.end() and character.get("char_end", -1) > match.start()
            )
    return tuple(rows)


def _with_viewer_overlay(
    proposition: dict,
    characters_by_id: dict[str, dict],
    marker_boxes: dict[tuple[object, ...], list[dict]],
) -> dict:
    proposition["viewer_overlay"] = derive_viewer_overlay(proposition, characters_by_id, marker_boxes)
    return proposition

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
