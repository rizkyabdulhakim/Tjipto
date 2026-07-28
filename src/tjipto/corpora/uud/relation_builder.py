from __future__ import annotations

from hashlib import sha256
import re

from tjipto.corpora.parser_dispatch import parse_legal_references


_AYAT_RE = re.compile(r"\bayat\s*\(\s*(\d+)\s*\)", re.IGNORECASE)
_AMENDMENT_ROLE_RE = re.compile(r"\bPerubahan\s+(Pertama|Kedua|Ketiga|Keempat)\b", re.IGNORECASE)


def parse_renumbering_mappings(text: str) -> list[dict]:
    """Derive ordered old/new reference pairs from explicit source text."""
    mappings: list[dict] = []
    for transition in re.finditer(r"\bmenjadi\b", text or "", re.IGNORECASE):
        left_start = max(text.rfind(";", 0, transition.start()), text.rfind(".", 0, transition.start())) + 1
        right_end = text.find(";", transition.end())
        if right_end < 0:
            right_end = len(text)
        left_refs = _reference_units(text, left_start, transition.start())
        right_refs = _reference_units(text, transition.end(), right_end)
        if not left_refs or len(left_refs) != len(right_refs):
            continue
        role_match = _AMENDMENT_ROLE_RE.search(text[left_start : transition.start()])
        source_role = _source_role_for_ordinal(role_match.group(1)) if role_match else None
        if not source_role:
            continue
        for old, new in zip(left_refs, right_refs):
            mappings.append(
                {
                    "old_reference": old["reference"],
                    "new_reference": new["reference"],
                    "old_range": (old["start"], old["end"]),
                    "new_range": (new["start"], new["end"]),
                    "old_range_kind": old["range_kind"],
                    "new_range_kind": new["range_kind"],
                    "source_role": source_role,
                }
            )
    return mappings


def _reference_units(text: str, start: int, end: int) -> list[dict]:
    segment = text[start:end]
    references = parse_legal_references("uud", segment)
    rows: list[dict] = []
    for index, reference in enumerate(references):
        local_start = int(reference["start"])
        local_end = int(references[index + 1]["start"]) if index + 1 < len(references) else len(segment)
        ayats = list(_AYAT_RE.finditer(segment[local_start:local_end]))
        if not ayats:
            rows.append(
                {
                    "reference": str(reference["reference"]),
                    "start": start + local_start,
                    "end": start + int(reference["end"]),
                    "range_kind": "literal",
                }
            )
            continue
        for ayat_index, ayat in enumerate(ayats):
            prefix = str(reference["reference"])
            rows.append(
                {
                    "reference": f"{prefix} ayat ({ayat.group(1)})",
                    "start": start + local_start,
                    "end": start + local_start + ayat.end(),
                    "range_kind": "literal" if ayat_index == 0 else "contextual",
                }
            )
    return rows


def _source_role_for_ordinal(ordinal: str) -> str:
    return {
        "Pertama": "amendment_1_historical",
        "Kedua": "amendment_2_historical",
        "Ketiga": "amendment_3_historical",
        "Keempat": "amendment_4_historical",
    }[ordinal.capitalize()]


def build_document_relations(source_documents: list[dict]) -> list[dict]:
    source_by_role = {row["source_role"]: row for row in source_documents}
    original = source_by_role["original_historical"]
    rows = []
    amendment_roles = sorted(role for role in source_by_role if role.startswith("amendment_"))
    for role in amendment_roles:
        source = source_by_role[role]
        rows.append(_document_relation("AMENDS", source, original))
        rows.append(_document_relation("AMENDED_BY", original, source, support_role=role))
    consolidated = source_by_role.get("current_consolidated")
    if consolidated is not None:
        rows.append(_document_relation("DERIVED_FROM", consolidated, original))
        rows.extend(_document_relation("CONSOLIDATES", consolidated, source_by_role[role]) for role in amendment_roles)
    return sorted(rows, key=lambda row: row["relation_id"])


def materialize_document_relation_edges(document_relations: list[dict]) -> list[dict]:
    """Project audited document relations into the single runtime graph."""
    rows = []
    for relation in document_relations:
        relation_id = str(relation["relation_id"])
        rows.append(
            {
                "edge_id": f"edge::{sha256(relation_id.encode('utf-8')).hexdigest()}",
                "source_id": f"source_role::{relation['source_role']}",
                "target_id": f"source_role::{relation['target_source_role']}",
                "edge_type": relation["relation_type"],
                "relation_type": relation["relation_type"],
                "relation_id": relation_id,
                "runtime_loadable": relation.get("runtime_loadable") is True,
                "support_kind": relation.get("support_kind") or "provenance_only",
                "support_relation_ids": [relation_id],
                "support_evidence_ids": list(relation.get("support_evidence_ids") or ()),
                "support_exception_ids": list(relation.get("support_exception_ids") or ()),
            }
        )
    return rows


def build_article_amendment_relations(
    *,
    graph_edges: list[dict],
    legal_units: list[dict],
    evidence: list[dict],
    bbox_rows: list[dict],
    page_text_spans: list[dict],
    word_bboxes: list[dict],
) -> list[dict]:
    units = {row["legal_unit_id"]: row for row in legal_units}
    evidence_by_id = {row["evidence_id"]: row for row in evidence}
    bbox_ids = {row["bbox_id"] for row in bbox_rows} | {row["word_bbox_id"] for row in word_bboxes}
    bbox_ids |= {character["character_bbox_id"] for word in word_bboxes for character in word.get("characters") or ()}
    spans_by_id = {row["text_span_id"]: row for row in page_text_spans}
    bboxes_by_id = {row["bbox_id"]: row for row in bbox_rows}
    bboxes_by_id.update({row["word_bbox_id"]: row | {"bbox_id": row["word_bbox_id"], "bbox_precision": "exact", "viewer_highlightable": True} for row in word_bboxes})
    for word in word_bboxes:
        for character in word.get("characters") or ():
            bboxes_by_id[character["character_bbox_id"]] = {
                **word,
                **character,
                "bbox_id": character["character_bbox_id"],
                "bbox_precision": "exact",
                "viewer_highlightable": True,
            }
    rows = []
    for edge in graph_edges:
        relation_type = edge.get("edge_type")
        if relation_type not in {"MODIFIES", "DELETES", "RENAMES", "RENUMBERED_TO"}:
            continue
        supporting_ids = edge.get("supporting_evidence_ids") or ()
        evidence_row = evidence_by_id.get(supporting_ids[0]) if supporting_ids else None
        source_unit_id = _legal_unit_id(edge.get("source_id"))
        target_unit_id = _legal_unit_id(edge.get("target_id"))
        source_unit = units.get(source_unit_id or "")
        target = units.get(target_unit_id or "")
        target_citation = target.get("unit_label") if target else None
        if not str(target_citation or "").startswith(("Pasal ", "Ayat ")):
            continue
        if not evidence_row or not target or not all(ref in bbox_ids for ref in evidence_row.get("bbox_refs") or ()):
            continue
        mapping = edge.get("reference_mapping") or {}
        target_phrase = str(mapping.get("new_reference") or target_citation or "")
        target_span_ids = _target_span_ids(evidence_row, target_phrase, spans_by_id)
        relation_span_ids, relation_bbox_refs = _relation_support_refs(
            evidence_row,
            mapping,
            spans_by_id,
            bboxes_by_id,
        )
        target_word_bbox_refs = _recover_target_word_bbox_refs(
            target_phrase,
            evidence_row["source_document_id"],
            relation_bbox_refs,
            bboxes_by_id,
            word_bboxes,
            _target_quote(target_span_ids, spans_by_id) if target_span_ids else str(evidence_row.get("quoted_text") or ""),
        )
        target_character_bbox_refs = _recover_target_character_bbox_refs(
            target_phrase,
            evidence_row["source_document_id"],
            relation_bbox_refs,
            bboxes_by_id,
            word_bboxes,
            _target_quote(target_span_ids, spans_by_id) if target_span_ids else str(evidence_row.get("quoted_text") or ""),
        )
        target_bbox_refs = target_character_bbox_refs or target_word_bbox_refs
        bbox_refs = target_bbox_refs
        if not relation_bbox_refs and source_support_available(evidence_row, bboxes_by_id):
            relation_bbox_refs = list(evidence_row.get("bbox_refs") or ())
        relation_bbox_refs = list(dict.fromkeys([*relation_bbox_refs, *bbox_refs]))
        source_support_exact = _complete_mapping_support(evidence_row, mapping)
        target_support_exact = (
            evidence_row.get("bbox_precision") == "exact"
            and evidence_row.get("viewer_highlightable") is True
            and bool(bbox_refs)
            and all(ref in bbox_ids for ref in bbox_refs)
        )
        target_local_geometry = bool(target_word_bbox_refs or target_character_bbox_refs)
        exact_support = source_support_exact and target_support_exact and target_local_geometry
        support_class = "exact_article_relation" if exact_support else "trace_article_relation"
        trace_only_reason = (
            None
            if exact_support
            else _trace_reason(
                evidence_row, target_phrase, target_span_ids, bbox_refs, mapping=mapping, source_support_exact=source_support_exact
            )
        )
        target_reference = target_phrase
        old_reference = str(mapping.get("old_reference") or "")
        if old_reference.startswith("Pasal 25E") and target_reference.startswith("Pasal 25A"):
            relation_type = "RENUMBERED_TO"
        rows.append(
            {
                "relation_id": _relation_id(relation_type, evidence_row["evidence_id"], target_unit_id, mapping),
                "corpus_id": "uud",
                "source_document_id": evidence_row["source_document_id"],
                "source_role": evidence_row["source_role"],
                "relation_type": relation_type,
                **(
                    {"substantive_change": False, "source_conflict": False, "anomaly": False}
                    if relation_type == "RENUMBERED_TO"
                    else {}
                ),
                "target_legal_unit_id": target_unit_id,
                "target_citation": target_citation,
                "source_legal_unit_id": source_unit_id,
                "source_legal_unit_role": source_unit.get("source_role") if source_unit else None,
                "source_label": source_unit.get("unit_label") if source_unit else None,
                "target_label": target_citation,
                "target_source_role": target.get("source_role") if target else None,
                "old_reference": old_reference,
                "new_reference": target_reference,
                "old_reference_range": mapping.get("old_range"),
                "new_reference_range": mapping.get("new_range"),
                "old_reference_range_kind": mapping.get("old_range_kind"),
                "new_reference_range_kind": mapping.get("new_range_kind"),
                "evidence_id": evidence_row["evidence_id"],
                "bbox_refs": relation_bbox_refs,
                "text_span_ids": relation_span_ids,
                "target_text_span_ids": target_span_ids,
                "target_bbox_refs": bbox_refs,
                "page_number": (evidence_row.get("page_numbers") or [None])[0],
                "quoted_text": evidence_row.get("quoted_text")
                if mapping
                else (_target_quote(target_span_ids, spans_by_id) if exact_support and target_span_ids else evidence_row.get("quoted_text")),
                "source_pdf_sha256": evidence_row.get("source_sha256"),
                "grounding_level": (
                    "exact_source_text"
                    if exact_support
                    else "exact_source_text_shared_target"
                    if source_support_exact
                    else "target_reference_only"
                    if mapping and target_support_exact
                    else "page_grounded_trace"
                ),
                "source_support_exact": source_support_exact,
                "target_precision": "target_local" if exact_support else ("shared_span" if mapping else "unresolved"),
                "support_class": support_class,
                "authority_kind": evidence_row.get("authority_kind") or "instrument_provenance",
                "citation_final": evidence_row.get("citation_final") is True,
                "bbox_precision": evidence_row.get("bbox_precision"),
                "target_geometry_method": "character_geometry" if target_character_bbox_refs else ("word_geometry" if target_word_bbox_refs else "unavailable"),
                "target_geometry_source_ids": list(bbox_refs),
                "recovery_capability": "exact_materialized"
                if exact_support
                else "word_geometry"
                if target_word_bbox_refs
                else "character_geometry"
                if target_character_bbox_refs
                else "technically_unrecoverable",
                "recovery_status": "promoted" if exact_support else "semantically_validated",
                "viewer_highlightable": bool(relation_bbox_refs),
                "citation_available": exact_support,
                "trace_only_reason": trace_only_reason,
                "runtime_loadable": True,
                "validator_status": "valid",
            }
        )
    return sorted(rows, key=lambda row: row["relation_id"])


def _target_span_ids(evidence: dict, citation: str, spans_by_id: dict[str, dict]) -> list[str]:
    pattern = _reference_pattern(citation)
    return [
        span_id
        for span_id in evidence.get("text_span_ids") or ()
        if pattern.search(str(spans_by_id.get(span_id, {}).get("text") or ""))
        and _isolates_reference(str(spans_by_id.get(span_id, {}).get("text") or ""), citation)
    ]


def _relation_support_refs(
    evidence: dict,
    mapping: dict,
    spans_by_id: dict[str, dict],
    bboxes_by_id: dict[str, dict],
) -> tuple[list[str], list[str]]:
    if not mapping:
        return list(evidence.get("text_span_ids") or ()), list(evidence.get("bbox_refs") or ())
    old_range = _valid_range(mapping.get("old_range"), len(str(evidence.get("quoted_text") or "")))
    new_range = _valid_range(mapping.get("new_range"), len(str(evidence.get("quoted_text") or "")))
    if old_range is None or new_range is None:
        return [], []
    start, end = old_range[0], new_range[1]
    return (
        _slice_support_ids(evidence.get("text_span_ids") or (), spans_by_id, start, end, str(evidence.get("quoted_text") or "")),
        _slice_support_ids(evidence.get("bbox_refs") or (), bboxes_by_id, start, end, str(evidence.get("quoted_text") or "")),
    )


def _slice_support_ids(ids: list[str] | tuple[str, ...], rows_by_id: dict[str, dict], start: int, end: int, quoted: str) -> list[str]:
    cursor = 0
    ranges: list[tuple[str, int, int]] = []
    for row_id in ids:
        text = str(rows_by_id.get(row_id, {}).get("text") or "")
        if not text:
            continue
        position = quoted.find(text, cursor)
        if position < 0:
            return []
        ranges.append((row_id, position, position + len(text)))
        cursor = position + len(text)
    selected = [row_id for row_id, row_start, row_end in ranges if row_end > start and row_start < end]
    return selected


def _target_bbox_refs(evidence: dict, citation: str, bboxes_by_id: dict[str, dict]) -> list[str]:
    pattern = _reference_pattern(citation)
    return [
        bbox_id
        for bbox_id in evidence.get("bbox_refs") or ()
        if pattern.search(str(bboxes_by_id.get(bbox_id, {}).get("text") or ""))
        and _isolates_reference(str(bboxes_by_id.get(bbox_id, {}).get("text") or ""), citation)
        and bboxes_by_id.get(bbox_id, {}).get("bbox_precision") == "exact"
        and bboxes_by_id.get(bbox_id, {}).get("viewer_highlightable") is True
    ]


def _recover_target_word_bbox_refs(
    citation: str,
    source_document_id: str,
    source_bbox_refs: list[str],
    bboxes_by_id: dict[str, dict],
    word_bboxes: list[dict],
    source_text: str,
) -> list[str]:
    """Return a unique contiguous word match wholly inside canonical source proof."""
    target = _compact(citation)
    source_boxes = [bboxes_by_id[ref] for ref in source_bbox_refs if ref in bboxes_by_id]
    if not target or not source_boxes:
        return []
    words_by_page: dict[int, list[dict]] = {}
    for word in word_bboxes:
        if word.get("source_document_id") == source_document_id:
            words_by_page.setdefault(int(word["page_number"]), []).append(word)
    matches: list[list[str]] = []
    for page_number, words in words_by_page.items():
        target_offsets = _target_offsets(source_text, citation, words)
        offsets: list[int] = []
        cursor = 0
        for word in words:
            offsets.append(cursor)
            cursor += len(_compact(word.get("text", "")))
        page_boxes = [row for row in source_boxes if row.get("page_number") == page_number]
        for start in range(len(words)):
            joined = ""
            matched: list[dict] = []
            for word in words[start:]:
                matched.append(word)
                joined = _compact(f"{joined} {word.get('text', '')}")
                if joined == target:
                    if offsets[start] in target_offsets and _compact("".join(str(item.get("text") or "") for item in matched)) == target and all(
                        _word_inside_any_box(item, page_boxes) for item in matched
                    ):
                        matches.append([item["word_bbox_id"] for item in matched])
                    break
                if len(joined) > len(target) + 24 or not target.startswith(joined):
                    break
    return matches[0] if len(matches) == 1 else []


def _recover_target_character_bbox_refs(
    citation: str,
    source_document_id: str,
    source_bbox_refs: list[str],
    bboxes_by_id: dict[str, dict],
    word_bboxes: list[dict],
    source_text: str,
) -> list[str]:
    target = _compact(citation)
    if not target:
        return []
    source_boxes = [bboxes_by_id[ref] for ref in source_bbox_refs if ref in bboxes_by_id]
    characters: list[dict] = []
    for word in word_bboxes:
        if word.get("source_document_id") != source_document_id:
            continue
        if not _word_inside_any_box(word, source_boxes):
            continue
        characters.extend(
            {
                **character,
                "page_number": word.get("page_number"),
                "source_document_id": word.get("source_document_id"),
                "word_bbox_id": word.get("word_bbox_id"),
            }
            for character in word.get("characters") or ()
            if _word_inside_any_box(character, source_boxes)
        )
    matches: list[list[str]] = []
    target_offsets = _target_offsets(source_text, citation, characters)
    character_offsets: list[int] = []
    cursor = 0
    for character in characters:
        character_offsets.append(cursor)
        cursor += len(_compact(character.get("text") or ""))
    for start in range(len(characters)):
        joined = ""
        matched: list[dict] = []
        for character in characters[start:]:
            matched.append(character)
            joined = _compact(joined + str(character.get("text") or ""))
            if joined == target:
                next_character = characters[start + len(matched)] if start + len(matched) < len(characters) else None
                next_text = str(next_character.get("text") or "") if next_character else ""
                if (
                    character_offsets[start] in target_offsets
                    and
                    (not next_text.isalnum() or (next_character or {}).get("word_bbox_id") != matched[-1].get("word_bbox_id"))
                    and len({item.get("page_number") for item in matched}) == 1
                    and all(item.get("source_document_id") == source_document_id for item in matched)
                    and _compact("".join(str(item.get("text") or "") for item in matched)) == target
                ):
                    matches.append([item["character_bbox_id"] for item in matched])
                break
            if len(joined) > len(target) or not target.startswith(joined):
                break
    return matches[0] if len(matches) == 1 else []


def source_support_available(evidence: dict, bboxes_by_id: dict[str, dict]) -> bool:
    return bool(evidence.get("bbox_refs")) and all(
        bboxes_by_id.get(ref, {}).get("bbox_precision") == "exact" and bboxes_by_id.get(ref, {}).get("viewer_highlightable") is True
        for ref in evidence.get("bbox_refs") or ()
    )


def _word_inside_any_box(word: dict, boxes: list[dict]) -> bool:
    center_x = (float(word["x0"]) + float(word["x1"])) / 2
    center_y = (float(word["y0"]) + float(word["y1"])) / 2
    return any(float(box["x0"]) <= center_x <= float(box["x1"]) and float(box["y0"]) <= center_y <= float(box["y1"]) for box in boxes)


def _compact(value: object) -> str:
    return "".join(re.findall(r"\w+", str(value or "").casefold()))


def _target_offsets(source_text: str, citation: str, tokens: list[dict]) -> set[int]:
    source = _compact(source_text)
    target = _compact(citation)
    if not source or not target:
        return set()
    relative = {len(_compact(source_text[: match.start()])) for match in _reference_pattern(citation).finditer(source_text)}
    if not relative:
        return set()
    page = "".join(_compact(token.get("text") or "") for token in tokens)
    starts: set[int] = set()
    cursor = page.find(source)
    while cursor >= 0:
        starts.update(cursor + offset for offset in relative)
        cursor = page.find(source, cursor + 1)
    return starts


def _word_phrase(value: object) -> str:
    return " ".join(re.sub(r"[^\w]+", " ", str(value or "").casefold()).split())


def _isolates_reference(text: str, citation: str) -> bool:
    pattern = _reference_pattern(citation)
    return bool(pattern.search(text)) and len(parse_legal_references("uud", text)) == 1 and _ayat_count(text) == _ayat_count(citation)


def _reference_pattern(citation: str) -> re.Pattern:
    # The suffix is adjacent to the article number.  Do not let the optional
    # suffix consume the leading ``a`` of ``ayat``.
    parts = re.findall(r"Pasal\s+([0-9]+)([A-Za-z]?)(?:\s+ayat\s+\((\d+)\))?", citation, re.IGNORECASE)
    if not parts:
        return re.compile(r"(?!x)x")
    number, suffix, ayat = parts[0]
    pasal = rf"{number}\s*{suffix}" if suffix else number
    suffix = rf"\s*ayat\s*\(\s*{re.escape(ayat)}\s*\)" if ayat else ""
    return re.compile(rf"(?i)\bpasal\s*{pasal}{suffix}(?!\w)")


def _ayat_count(text: str) -> int:
    return len(_AYAT_RE.findall(text or ""))


def _target_quote(span_ids: list[str], spans_by_id: dict[str, dict]) -> str:
    return " ".join(str(spans_by_id[span_id].get("text") or "").strip() for span_id in span_ids).strip()


def _trace_reason(
    evidence: dict,
    citation: str,
    span_ids: list[str],
    bbox_refs: list[str],
    *,
    mapping: dict | None = None,
    source_support_exact: bool = True,
) -> str:
    if mapping and not source_support_exact:
        return "mapping_support_incomplete"
    if not span_ids:
        if _reference_pattern(citation).search(str(evidence.get("quoted_text") or "")):
            return "shared_source_line_target_not_isolatable"
        return "unresolved_target_mention"
    if not bbox_refs:
        return "missing_relation_local_bbox"
    return evidence.get("failure_reason") or "relation_target_proof_incomplete"


def _complete_mapping_support(evidence: dict, mapping: dict) -> bool:
    if not mapping:
        return evidence.get("bbox_precision") == "exact" and evidence.get("viewer_highlightable") is True
    quoted = str(evidence.get("quoted_text") or "")
    old_range = _valid_range(mapping.get("old_range"), len(quoted))
    new_range = _valid_range(mapping.get("new_range"), len(quoted))
    if old_range is None or new_range is None or old_range[1] > new_range[0]:
        return False
    if not _range_matches_reference(quoted[old_range[0] : old_range[1]], mapping.get("old_reference"), mapping.get("old_range_kind")):
        return False
    if not _range_matches_reference(quoted[new_range[0] : new_range[1]], mapping.get("new_reference"), mapping.get("new_range_kind")):
        return False
    if "menjadi" not in _normalize_reference(quoted[old_range[1] : new_range[0]]):
        return False
    return bool(evidence.get("text_span_ids")) and bool(evidence.get("bbox_refs"))


def _valid_range(value: object, length: int) -> tuple[int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        start, end = int(value[0]), int(value[1])
    except (TypeError, ValueError):
        return None
    return (start, end) if 0 <= start < end <= length else None


def _normalize_reference(value: object) -> str:
    return " ".join(str(value or "").casefold().split())


def _range_matches_reference(text: str, reference: object, kind: object) -> bool:
    expected = _normalize_reference(reference)
    actual = _normalize_reference(text)
    if kind != "contextual":
        return actual == expected
    parsed = parse_legal_references("uud", text)
    if not parsed or not expected.startswith(_normalize_reference(parsed[0]["reference"])):
        return False
    ayat = re.search(r"\bayat\s*\(\s*(\d+)\s*\)", expected, re.IGNORECASE)
    return bool(ayat and any(match.group(1) == ayat.group(1) for match in _AYAT_RE.finditer(text)))


def _relation_id(relation_type: str, evidence_id: str, target_unit_id: str | None, mapping: dict) -> str:
    mapping_key = f"{mapping.get('old_reference', '')}->{mapping.get('new_reference', '')}" if mapping else ""
    suffix = f"::{mapping_key}" if mapping_key else ""
    return f"uud_article_amendment_relation::{relation_type.lower()}::{evidence_id}::{target_unit_id}{suffix}"


def _document_relation(relation_type: str, source: dict, target: dict, *, support_role: str | None = None) -> dict:
    source_role = source["source_role"]
    target_role = target["source_role"]
    support = support_role or source_role
    provenance_only = relation_type in {"DERIVED_FROM", "CONSOLIDATES"}
    return {
        "relation_id": f"uud_document_relation::{source_role.lower()}::{relation_type.lower()}::{target_role.lower()}",
        "corpus_id": "uud",
        "relation_type": relation_type,
        "source_document_id": source["source_document_id"],
        "source_role": source_role,
        "target_document_id": target["source_document_id"],
        "target_source_role": target_role,
        "support_type": "source_role_grounded",
        "object_role": "relation_proof",
        "support_relation_ids": [],
        "support_evidence_ids": [],
        "support_exception_ids": [] if provenance_only else [_support_ref(relation_type, source_role, target_role, support)],
        "support_kind": "provenance_only",
        "runtime_loadable": True,
        "viewer_highlightable": False,
        "citation_available": False,
        "article_level": False,
        "reason": (
            "consolidated_provenance_without_legal_force_claim"
            if provenance_only
            else "source_role_document_level_relation_without_pasal_ayat_evidence_bbox"
        ),
    }


def _legal_unit_id(node_id: object) -> str | None:
    text = str(node_id or "")
    return text.split("legal_unit::", 1)[1] if text.startswith("legal_unit::") else None


def _support_ref(relation_type: str, source_role: str, target_role: str, support_role: str) -> str:
    if relation_type == "AMENDS":
        return f"uud_legal_graph_edge::{source_role}::amends_original_record"
    return f"uud_legal_graph_edge::{source_role}::amended_by_{target_role if target_role.startswith('amendment_') else support_role}_record"
