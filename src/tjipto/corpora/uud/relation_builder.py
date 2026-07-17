from __future__ import annotations

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
                }
            )
            continue
        for ayat in ayats:
            prefix = str(reference["reference"])
            rows.append(
                {
                    "reference": f"{prefix} ayat ({ayat.group(1)})",
                    "start": start + local_start if ayat.start() == 0 else start + local_start + ayat.start(),
                    "end": start + local_start + ayat.end(),
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
    for role in sorted(role for role in source_by_role if role.startswith("amendment_")):
        source = source_by_role[role]
        rows.append(_document_relation("AMENDS", source, original))
        rows.append(_document_relation("AMENDED_BY", original, source, support_role=role))
    return sorted(rows, key=lambda row: row["relation_id"])


def build_article_amendment_relations(
    *,
    graph_edges: list[dict],
    legal_units: list[dict],
    evidence: list[dict],
    bbox_rows: list[dict],
    page_text_spans: list[dict],
) -> list[dict]:
    units = {row["legal_unit_id"]: row for row in legal_units}
    evidence_by_id = {row["evidence_id"]: row for row in evidence}
    bbox_ids = {row["bbox_id"] for row in bbox_rows}
    spans_by_id = {row["text_span_id"]: row for row in page_text_spans}
    bboxes_by_id = {row["bbox_id"]: row for row in bbox_rows}
    rows = []
    for edge in graph_edges:
        relation_type = edge.get("edge_type")
        if relation_type not in {"MODIFIES", "DELETES", "RENAMES"}:
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
        bbox_refs = _target_bbox_refs(evidence_row, target_phrase, bboxes_by_id)
        exact_support = (
            evidence_row.get("bbox_precision") == "exact"
            and evidence_row.get("viewer_highlightable") is True
            and bool(target_span_ids)
            and bool(bbox_refs)
            and all(ref in bbox_ids for ref in bbox_refs)
        )
        support_class = "exact_article_relation" if exact_support else "trace_article_relation"
        trace_only_reason = None if exact_support else _trace_reason(evidence_row, target_phrase, target_span_ids, bbox_refs)
        relation_bbox_refs = bbox_refs if exact_support else list(evidence_row.get("bbox_refs") or ())
        relation_span_ids = target_span_ids if exact_support else list(evidence_row.get("text_span_ids") or ())
        target_reference = target_phrase
        old_reference = str(mapping.get("old_reference") or "")
        rows.append(
            {
                "relation_id": _relation_id(relation_type, evidence_row["evidence_id"], target_unit_id, mapping),
                "corpus_id": "uud",
                "source_document_id": evidence_row["source_document_id"],
                "source_role": evidence_row["source_role"],
                "relation_type": relation_type,
                "target_legal_unit_id": target_unit_id,
                "target_citation": target_citation,
                "source_legal_unit_id": source_unit_id,
                "source_label": source_unit.get("unit_label") if source_unit else None,
                "target_label": target_citation,
                "target_source_role": target.get("source_role") if target else None,
                "old_reference": old_reference,
                "new_reference": target_reference,
                "old_reference_range": mapping.get("old_range"),
                "new_reference_range": mapping.get("new_range"),
                "evidence_id": evidence_row["evidence_id"],
                "bbox_refs": relation_bbox_refs,
                "text_span_ids": relation_span_ids,
                "target_text_span_ids": target_span_ids,
                "target_bbox_refs": bbox_refs,
                "page_number": (evidence_row.get("page_numbers") or [None])[0],
                "quoted_text": _target_quote(target_span_ids, spans_by_id) if exact_support else evidence_row.get("quoted_text"),
                "source_pdf_sha256": evidence_row.get("source_sha256"),
                "grounding_level": (
                    "exact_source_text"
                    if exact_support
                    else "exact_source_text_shared_target"
                    if mapping and evidence_row.get("bbox_precision") == "exact" and evidence_row.get("viewer_highlightable") is True
                    else "page_grounded_trace"
                ),
                "source_support_exact": evidence_row.get("bbox_precision") == "exact" and evidence_row.get("viewer_highlightable") is True,
                "target_precision": "target_local" if exact_support else ("shared_span" if mapping else "unresolved"),
                "support_class": support_class,
                "bbox_precision": evidence_row.get("bbox_precision"),
                "viewer_highlightable": exact_support,
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


def _isolates_reference(text: str, citation: str) -> bool:
    pattern = _reference_pattern(citation)
    return bool(pattern.search(text)) and len(parse_legal_references("uud", text)) == 1 and _ayat_count(text) == _ayat_count(citation)


def _reference_pattern(citation: str) -> re.Pattern:
    parts = re.findall(r"Pasal\s+([0-9]+[A-Za-z]?)(?:\s+ayat\s+\((\d+)\))?", citation, re.IGNORECASE)
    if not parts:
        return re.compile(r"(?!x)x")
    pasal, ayat = parts[0]
    suffix = rf"\s*ayat\s*\(\s*{re.escape(ayat)}\s*\)" if ayat else ""
    return re.compile(rf"(?i)\bpasal\s*{re.escape(pasal)}{suffix}\b")


def _ayat_count(text: str) -> int:
    return len(_AYAT_RE.findall(text or ""))


def _target_quote(span_ids: list[str], spans_by_id: dict[str, dict]) -> str:
    return " ".join(str(spans_by_id[span_id].get("text") or "").strip() for span_id in span_ids).strip()


def _trace_reason(evidence: dict, citation: str, span_ids: list[str], bbox_refs: list[str]) -> str:
    if not span_ids:
        if _reference_pattern(citation).search(str(evidence.get("quoted_text") or "")):
            return "shared_source_line_target_not_isolatable"
        return "unresolved_target_mention"
    if not bbox_refs:
        return "missing_relation_local_bbox"
    return evidence.get("failure_reason") or "relation_target_proof_incomplete"


def _relation_id(relation_type: str, evidence_id: str, target_unit_id: str | None, mapping: dict) -> str:
    mapping_key = f"{mapping.get('old_reference', '')}->{mapping.get('new_reference', '')}" if mapping else ""
    suffix = f"::{mapping_key}" if mapping_key else ""
    return f"uud_article_amendment_relation::{relation_type.lower()}::{evidence_id}::{target_unit_id}{suffix}"


def _document_relation(relation_type: str, source: dict, target: dict, *, support_role: str | None = None) -> dict:
    source_role = source["source_role"]
    target_role = target["source_role"]
    support = support_role or source_role
    return {
        "relation_id": f"uud_document_relation::{source_role.lower()}::{relation_type.lower()}::{target_role.lower()}",
        "corpus_id": "uud",
        "relation_type": relation_type,
        "source_document_id": source["source_document_id"],
        "source_role": source_role,
        "target_document_id": target["source_document_id"],
        "target_source_role": target_role,
        "support_type": "source_role_grounded",
        "support_refs": [_support_ref(relation_type, source_role, target_role, support)],
        "runtime_loadable": True,
        "viewer_highlightable": False,
        "citation_available": False,
        "article_level": False,
        "reason": "source_role_document_level_relation_without_pasal_ayat_evidence_bbox",
    }


def _legal_unit_id(node_id: object) -> str | None:
    text = str(node_id or "")
    return text.split("legal_unit::", 1)[1] if text.startswith("legal_unit::") else None


def _support_ref(relation_type: str, source_role: str, target_role: str, support_role: str) -> str:
    if relation_type == "AMENDS":
        return f"uud_legal_graph_edge::{source_role}::amends_original_record"
    return f"uud_legal_graph_edge::{source_role}::amended_by_{target_role if target_role.startswith('amendment_') else support_role}_record"
