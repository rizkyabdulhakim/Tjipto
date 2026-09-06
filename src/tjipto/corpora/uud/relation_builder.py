from __future__ import annotations

from hashlib import sha256
import re

from tjipto.contracts.evidence import valid_text_range
from tjipto.contracts.relations import descriptor_for
from tjipto.corpora.parser_dispatch import parse_legal_references
from tjipto.corpora.uud.parser import matches_uud_contextual_reference


_AYAT_RE = re.compile(r"\bayat\s*\(\s*(\d+)\s*\)", re.IGNORECASE)
_AMENDMENT_ROLE_RE = re.compile(r"\bPerubahan\s+(Pertama|Kedua|Ketiga|Keempat)\b", re.IGNORECASE)
_AMENDMENT_ROLE_ORDER = {
    "original_historical": 0,
    "amendment_1_historical": 1,
    "amendment_2_historical": 2,
    "amendment_3_historical": 3,
    "amendment_4_historical": 4,
}


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
        for old, new in zip(left_refs, right_refs, strict=True):
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
    seen_articles: set[str] = set()
    for reference in references:
        value = str(reference["reference"])
        article = value.partition(" ayat ")[0]
        has_ayat = " ayat " in value
        range_kind = "contextual" if has_ayat and article in seen_articles else "literal"
        if has_ayat:
            seen_articles.add(article)
        rows.append(
            {
                "reference": value,
                "start": start + int(reference["start"]),
                "end": start + int(reference["end"]),
                "range_kind": range_kind,
            }
        )
    return rows


def resolve_relation_unit(
    legal_units: list[dict],
    reference: str,
    *,
    source_role: str | None = None,
    source_document_id: str | None = None,
) -> dict | None:
    """Resolve a UUD relation reference without collapsing an Ayat to its Pasal."""
    article_label, separator, ayat_label = str(reference).partition(" ayat ")
    candidates = (
        row
        for row in legal_units
        if (source_role is None or row.get("source_role") == source_role)
        and (source_document_id is None or row.get("source_document_id") == source_document_id)
    )
    article = next((row for row in candidates if row.get("unit_label") == article_label), None)
    if article is None or not separator or not re.fullmatch(r"\(\d+\)", ayat_label.strip()):
        return article
    return next(
        (
            row
            for row in legal_units
            if (source_role is None or row.get("source_role") == source_role)
            and (source_document_id is None or row.get("source_document_id") == source_document_id)
            and row.get("unit_label") == ayat_label.strip()
            and article["legal_unit_id"] in (row.get("parent_legal_unit_ids") or ())
        ),
        None,
    )


def legal_unit_reference(unit: dict | None, units_by_id: dict[str, dict]) -> str | None:
    """Return the complete relation reference represented by a legal unit."""
    if unit is None:
        return None
    label = str(unit.get("unit_label") or "")
    if label.startswith("Pasal "):
        return label
    parent = units_by_id.get(str(unit.get("parent_legal_unit_id") or ""))
    if parent and str(parent.get("unit_label") or "").startswith("Pasal ") and re.fullmatch(r"\(\d+\)", label):
        return f"{parent['unit_label']} ayat {label}"
    return label or None


def classify_scope_operation(
    legal_units: list[dict],
    source_role: str,
    target_citation: str,
    *,
    canonical_target: dict | None = None,
) -> dict[str, object]:
    """Classify an ambiguous amendment scope only from adjacent source units.

    A scope sentence is not enough to choose ``ADDS`` or ``MODIFIES``.  The
    historical unit inventory supplies the predecessor/successor comparison;
    absent predecessor means addition, while a changed predecessor text means
    modification.  Missing or unchanged proof stays trace-only.
    """
    source_index = _AMENDMENT_ROLE_ORDER.get(source_role)
    if source_index is None or not target_citation:
        return {"relation_type": "AMBIGUOUS_OPERATION", "operation_candidates": ("MODIFIES", "ADDS")}
    units_by_id = {row["legal_unit_id"]: row for row in legal_units}
    by_reference: dict[str, list[dict]] = {}
    for unit in legal_units:
        reference = legal_unit_reference(unit, units_by_id)
        if reference:
            by_reference.setdefault(reference, []).append(unit)
    successor = next(
        (row for row in by_reference.get(target_citation, ()) if row.get("source_role") == source_role),
        None,
    )
    if successor is None:
        canonical_target = canonical_target or next(
            (
                row
                for row in by_reference.get(target_citation, ())
                if row.get("source_role") == "current_consolidated"
            ),
            None,
        )
        successor = _successor_by_normative_text(
            legal_units,
            source_role,
            canonical_target,
            units_by_id,
        )
    if successor is None:
        return {
            "relation_type": "AMBIGUOUS_OPERATION",
            "operation_candidates": ("MODIFIES", "ADDS"),
        }
    previous = _previous_unit_for_reference(
        by_reference,
        target_citation,
        source_index,
    )
    comparison = {
        "successor_legal_unit_id": successor["legal_unit_id"],
        "successor_source_role": successor.get("source_role"),
        "comparison_basis": "versioned_normative_text",
    }
    if previous is None:
        return {"relation_type": "ADDS", **comparison}
    comparison.update(
        {
            "predecessor_legal_unit_id": previous["legal_unit_id"],
            "predecessor_source_role": previous.get("source_role"),
            "predecessor_reference": legal_unit_reference(previous, units_by_id),
        }
    )
    if _normalized_normative_text(previous.get("text")) != _normalized_normative_text(successor.get("text")):
        return {"relation_type": "MODIFIES", **comparison}
    return {
        "relation_type": "AMBIGUOUS_OPERATION",
        "operation_candidates": ("MODIFIES", "ADDS"),
        **comparison,
    }


def _successor_by_normative_text(
    legal_units: list[dict],
    source_role: str,
    canonical_target: dict | None,
    units_by_id: dict[str, dict],
) -> dict | None:
    """Resolve a printed-numbering mismatch only when the norm is unique.

    Historical amendment PDFs can print a replacement paragraph under a
    different local number than the canonical consolidated text.  Matching
    the complete normalized text within the same article is deterministic;
    zero or multiple candidates remain trace-only.
    """
    if canonical_target is None:
        return None
    parent = units_by_id.get(str(canonical_target.get("parent_legal_unit_id") or ""))
    article_label = str(parent.get("unit_label") or "") if parent else str(canonical_target.get("unit_label") or "")
    normalized = _normalized_normative_text(canonical_target.get("text"))
    if not article_label or not normalized:
        return None
    candidates = []
    for unit in legal_units:
        if unit.get("source_role") != source_role:
            continue
        unit_parent = units_by_id.get(str(unit.get("parent_legal_unit_id") or ""))
        if (str(unit_parent.get("unit_label") or "") if unit_parent else "") != article_label:
            continue
        if _normalized_normative_text(unit.get("text")) == normalized:
            candidates.append(unit)
    return candidates[0] if len(candidates) == 1 else None


def versioned_relation_lineage(
    legal_units: list[dict],
    relation_type: str,
    source_role: str,
    target_citation: str,
    target_unit: dict | None,
) -> dict[str, object] | None:
    """Materialize the version comparison behind an article relation.

    Scope text identifies an operation, but it is not the normative target.
    Only the historical unit inventory can prove the predecessor/successor
    pair.  Deletion edges already point at the predecessor, so they only need
    the explicit deletion clause plus that unit's source lineage.
    """
    if relation_type in {"ADDS", "MODIFIES"}:
        comparison = classify_scope_operation(legal_units, source_role, target_citation)
        if comparison.get("relation_type") != relation_type:
            return None
        units_by_id = {row["legal_unit_id"]: row for row in legal_units}
        successor = units_by_id.get(str(comparison.get("successor_legal_unit_id") or ""))
        predecessor = units_by_id.get(str(comparison.get("predecessor_legal_unit_id") or ""))
        if successor is None or target_unit is None or target_unit.get("legal_unit_id") != successor.get("legal_unit_id"):
            return None
        if relation_type == "ADDS" and predecessor is not None:
            return None
        if relation_type == "MODIFIES" and predecessor is None:
            return None
        return _lineage_payload(predecessor, successor)
    if relation_type == "DELETES" and target_unit is not None:
        return _lineage_payload(target_unit, None)
    return None


def direct_rename_lineage(
    source_unit: dict | None,
    target_unit: dict | None,
    mapping: dict,
    units_by_id: dict[str, dict],
) -> dict[str, object] | None:
    """Prove a rename/renumber edge from its two normative legal units.

    The amendment clause supplies the operation and mapping; the old and new
    unit payloads prove that the same norm survived under a new label. A
    clause-only edge stays trace-only when either side cannot be resolved.
    """
    if source_unit is None or target_unit is None:
        return None
    old_reference = str(mapping.get("old_reference") or "")
    new_reference = str(mapping.get("new_reference") or "")
    if not old_reference or not new_reference:
        return None
    if legal_unit_reference(source_unit, units_by_id) != old_reference:
        return None
    if legal_unit_reference(target_unit, units_by_id) != new_reference:
        return None
    if _normalized_normative_text(source_unit.get("text")) != _normalized_normative_text(target_unit.get("text")):
        return None
    return _lineage_payload(source_unit, target_unit)


def _lineage_payload(predecessor: dict | None, successor: dict | None) -> dict[str, object]:
    """Return the closed, source-unit-owned comparison fields."""
    return {
        "predecessor_legal_unit_id": predecessor.get("legal_unit_id") if predecessor else None,
        "successor_legal_unit_id": successor.get("legal_unit_id") if successor else None,
        "predecessor_text_span_ids": list((predecessor or {}).get("source_text_span_ids") or (predecessor or {}).get("text_span_ids") or ()),
        "successor_text_span_ids": list((successor or {}).get("source_text_span_ids") or (successor or {}).get("text_span_ids") or ()),
        "predecessor_bbox_refs": list((predecessor or {}).get("source_bbox_refs") or (predecessor or {}).get("bbox_ids") or ()),
        "successor_bbox_refs": list((successor or {}).get("source_bbox_refs") or (successor or {}).get("bbox_ids") or ()),
        "predecessor_pdf_sha256": predecessor.get("source_sha256") if predecessor else None,
        "successor_pdf_sha256": successor.get("source_sha256") if successor else None,
        "comparison_basis": "versioned_normative_text",
    }


def _empty_lineage() -> dict[str, object]:
    return _lineage_payload(None, None) | {"comparison_basis": None}


def _string_refs(value: object) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [str(ref) for ref in value if ref]


def predecessor_unit_for_reference(
    legal_units: list[dict], target_citation: str, source_role: str
) -> dict | None:
    """Resolve the immediately preceding historical unit for a relation target."""
    source_index = _AMENDMENT_ROLE_ORDER.get(source_role)
    if source_index is None or not target_citation:
        return None
    units_by_id = {row["legal_unit_id"]: row for row in legal_units}
    by_reference: dict[str, list[dict]] = {}
    for unit in legal_units:
        reference = legal_unit_reference(unit, units_by_id)
        if reference:
            by_reference.setdefault(reference, []).append(unit)
    return _previous_unit_for_reference(by_reference, target_citation, source_index)


def _previous_unit_for_reference(
    by_reference: dict[str, list[dict]], target_citation: str, source_index: int
) -> dict | None:
    candidates = [
        row
        for row in by_reference.get(target_citation, ())
        if _AMENDMENT_ROLE_ORDER.get(str(row.get("source_role")), 99) < source_index
    ]
    if not candidates:
        match = re.fullmatch(r"(Pasal\s+\d+[A-Za-z]?)\s+ayat\s+\(1\)", target_citation, re.IGNORECASE)
        if match:
            candidates = [
                row
                for row in by_reference.get(match.group(1), ())
                if _AMENDMENT_ROLE_ORDER.get(str(row.get("source_role")), 99) < source_index
            ]
    return max(candidates, key=lambda row: _AMENDMENT_ROLE_ORDER.get(str(row.get("source_role")), -1), default=None)


def _normalized_normative_text(value: object) -> str:
    text = str(value or "").casefold()
    text = re.sub(r"\bpasal\s+\d+[a-z]?\b", " ", text)
    text = re.sub(r"\bayat\s*\(\s*\d+\s*\)\b", " ", text)
    text = re.sub(r"\(\s*\d+\s*\)", " ", text)
    return "".join(re.findall(r"[a-z0-9]+", text))


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


def materialize_relation_projections(
    edges: list[dict], document_relations: list[dict], article_relations: list[dict]
) -> None:
    """Embed the audited relation record on its persisted runtime edge."""
    relations = {
        str(row["relation_id"]): row
        for row in (*document_relations, *article_relations)
    }
    for edge in edges:
        relation = relations.get(str(edge.get("relation_id") or ""))
        if relation is None:
            continue
        edge["relation_projection"] = _relation_projection(edge, relation)


def _relation_projection(edge: dict, relation: dict) -> dict:
    projection = dict(relation)
    article_relation = "source_legal_unit_id" in projection
    if article_relation:
        projection["support_document_id"] = projection.get("source_document_id")
        projection["support_source_role"] = projection.get("source_role")
        projection["source_document_id"] = projection.get("source_legal_unit_document_id")
        projection["source_role"] = projection.get("source_legal_unit_role")
    inverse = edge.get("derived_from_edge_id") is not None
    if inverse:
        descriptor = descriptor_for(relation.get("relation_type"))
        if descriptor is None or descriptor.inverse != edge.get("edge_type"):
            raise ValueError(f"invalid_inverse_relation_projection:{edge.get('edge_id')}")
        for source_key, target_key in (
            ("source_document_id", "target_document_id"),
            ("source_role", "target_source_role"),
            ("source_label", "target_label"),
            ("source_legal_unit_id", "target_legal_unit_id"),
        ):
            if source_key in projection or target_key in projection:
                projection[source_key], projection[target_key] = projection.get(target_key), projection.get(source_key)
        if article_relation:
            projection["source_legal_unit_role"] = projection.get("source_role")
            projection["target_citation"] = projection.get("target_label")
    projection.update(
        {
            "relation_type": edge["edge_type"],
            "source_id": edge["source_id"],
            "target_id": edge["target_id"],
            "projection_direction": "inverse" if inverse else "forward",
        }
    )
    return projection


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
        if relation_type not in {"MODIFIES", "DELETES", "ADDS", "AMBIGUOUS_OPERATION", "RENAMES", "RENUMBERED_TO"}:
            continue
        supporting_ids = edge.get("supporting_evidence_ids") or ()
        evidence_row = evidence_by_id.get(supporting_ids[0]) if supporting_ids else None
        source_unit_id = _legal_unit_id(edge.get("source_id"))
        target_unit_id = _legal_unit_id(edge.get("target_id"))
        source_unit = units.get(source_unit_id or "")
        target = units.get(target_unit_id or "")
        mapping = edge.get("reference_mapping") or {}
        target_citation = str(
            mapping.get("new_reference")
            or edge.get("target_citation")
            or legal_unit_reference(target, units)
            or ""
        )
        if not target_citation.startswith("Pasal "):
            continue
        if not evidence_row or not target or not all(ref in bbox_ids for ref in evidence_row.get("bbox_refs") or ()):
            continue
        lineage = versioned_relation_lineage(
            legal_units,
            str(relation_type),
            str(evidence_row.get("source_role") or ""),
            target_citation,
            target,
        )
        if lineage is None and relation_type in {"RENAMES", "RENUMBERED_TO"}:
            lineage = direct_rename_lineage(source_unit, target, mapping, units)
        lineage = lineage or _empty_lineage()
        target_phrase = target_citation
        old_reference = str(mapping.get("old_reference") or "")
        lineage_unit_id = lineage.get("successor_legal_unit_id") or lineage.get("predecessor_legal_unit_id")
        lineage_unit = units.get(str(lineage_unit_id or ""))
        # A child ayat's source span normally contains only ``(n)`` rather
        # than the full ``Pasal n ayat (n)`` label.  The versioned unit is
        # nevertheless the authoritative normative target, so retain its
        # complete spans when the label-only matcher cannot isolate a span.
        target_span_ids = (
            _lineage_target_span_ids(lineage_unit, target_phrase, spans_by_id)
            or _target_span_ids(evidence_row, target_phrase, spans_by_id)
            or _string_refs(lineage.get("successor_text_span_ids"))
        )
        relation_span_ids, relation_bbox_refs = _relation_support_refs(
            evidence_row,
            mapping,
            spans_by_id,
            bboxes_by_id,
        )
        target_document_id = str(
            (lineage_unit or {}).get("source_document_id")
            or target.get("source_document_id")
            or evidence_row["source_document_id"]
        )
        target_lineage_bbox_refs = [
            ref
            for ref in (
                _string_refs(lineage.get("successor_bbox_refs"))
                or _string_refs(lineage.get("predecessor_bbox_refs"))
            )
            if ref in bbox_ids
        ]
        # Target geometry must be searched inside the versioned normative
        # unit.  Relation-clause boxes belong to the instrument scope and can
        # be on a different page; using them first makes an otherwise exact
        # target look unrecoverable.
        target_context_bbox_refs = target_lineage_bbox_refs or relation_bbox_refs
        target_source_text = _target_quote(target_span_ids, spans_by_id)
        target_source_texts = [
            str(spans_by_id[span_id].get("text") or "").strip()
            for span_id in target_span_ids
            if str(spans_by_id.get(span_id, {}).get("text") or "").strip()
        ]
        if target_source_text:
            target_source_texts.append(target_source_text)
        page_context = _target_page_quote(target_span_ids, spans_by_id)
        if page_context and page_context not in target_source_texts:
            target_source_texts.append(page_context)
        if not target_source_texts and target_span_ids:
            target_source_texts.append(str(evidence_row.get("quoted_text") or ""))
        # Character geometry is the most precise grounding available.  Try
        # it first and retain word geometry only as a deterministic fallback
        # for sources without a unique character match.
        target_character_bbox_refs: list[str] = []
        target_word_bbox_refs: list[str] = []
        printed_target_reference = legal_unit_reference(target, units)
        for geometry_citation in filter(None, (target_phrase, old_reference, printed_target_reference)):
            for target_source_text in target_source_texts:
                target_character_bbox_refs = _recover_target_character_bbox_refs(
                    geometry_citation,
                    target_document_id,
                    target_context_bbox_refs,
                    bboxes_by_id,
                    word_bboxes,
                    target_source_text,
                )
                if target_character_bbox_refs:
                    break
                target_word_bbox_refs = _recover_target_word_bbox_refs(
                    geometry_citation,
                    target_document_id,
                    target_context_bbox_refs,
                    bboxes_by_id,
                    word_bboxes,
                    target_source_text,
                )
                if target_word_bbox_refs:
                    break
            if target_character_bbox_refs or target_word_bbox_refs:
                break
        lineage_bbox_refs = target_lineage_bbox_refs
        target_bbox_refs = target_word_bbox_refs or target_character_bbox_refs or lineage_bbox_refs
        if not target_bbox_refs:
            target_bbox_refs = _target_bbox_refs(evidence_row, target_phrase, bboxes_by_id)
        bbox_refs = target_bbox_refs
        if not relation_bbox_refs and source_support_available(evidence_row, bboxes_by_id):
            relation_bbox_refs = list(evidence_row.get("bbox_refs") or ())
        source_support_exact = _complete_mapping_support(evidence_row, mapping)
        target_support_exact = (
            evidence_row.get("bbox_precision") == "exact"
            and evidence_row.get("viewer_highlightable") is True
            and bool(bbox_refs)
            and all(ref in bbox_ids for ref in bbox_refs)
        )
        target_local_geometry = bool(target_bbox_refs)
        exact_support = source_support_exact and target_support_exact and target_local_geometry
        if relation_type == "AMBIGUOUS_OPERATION":
            # The source names a change/addition set but does not choose one
            # operation.  Preserve exact provenance as trace support only.
            exact_support = False
        support_class = "exact_article_relation" if exact_support else "trace_article_relation"
        trace_only_reason = (
            None
            if exact_support
                else (
                    "ambiguous_source_operation"
                    if relation_type == "AMBIGUOUS_OPERATION"
                    else _trace_reason(
                    evidence_row, target_phrase, target_span_ids, bbox_refs, mapping=mapping, source_support_exact=source_support_exact
                    )
                )
        )
        target_reference = target_phrase
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
                    {"operation_candidates": ["MODIFIES", "ADDS"]}
                    if relation_type == "AMBIGUOUS_OPERATION"
                    else {}
                ),
                **(
                    {"substantive_change": False, "source_conflict": False, "anomaly": False}
                    if relation_type == "RENUMBERED_TO"
                    else {"source_conflict": True, "anomaly": True}
                    if edge.get("provenance_ref_kind") == "source_conflict"
                    else {}
                ),
                "target_legal_unit_id": target_unit_id,
                "target_citation": target_citation,
                "source_legal_unit_id": source_unit_id,
                "source_legal_unit_document_id": source_unit.get("source_document_id") if source_unit else None,
                "source_legal_unit_role": source_unit.get("source_role") if source_unit else None,
                "source_label": source_unit.get("unit_label") if source_unit else None,
                "target_document_id": target.get("source_document_id") if target else None,
                "target_label": target.get("unit_label"),
                "target_source_role": target.get("source_role") if target else None,
                "old_reference": (
                    legal_unit_reference(target, units)
                    if edge.get("provenance_ref_kind") == "source_conflict"
                    else old_reference
                ),
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
                # Keep the operation clause as the relation quote.  The
                # versioned normative target is carried separately through
                # target_text_span_ids/target_bbox_refs, avoiding a synthetic
                # quote that would blur operation and target evidence.
                "quoted_text": evidence_row.get("quoted_text") or _target_quote(target_span_ids, spans_by_id),
                "source_pdf_sha256": evidence_row.get("source_sha256"),
                **lineage,
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
                # Relation finality is a claim-level contract.  A direct,
                # isolated source clause with exact target geometry can be
                # cited even when its owner evidence remains instrument
                # provenance.  Ambiguous operation rows never reach this
                # branch because they are downgraded to trace support above.
                "citation_final": exact_support,
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


def _lineage_target_span_ids(unit: dict | None, citation: str, spans_by_id: dict[str, dict]) -> list[str]:
    """Select the printed citation span from the normative versioned unit."""
    if unit is None:
        return []
    source_span_ids = list(unit.get("source_text_span_ids") or unit.get("text_span_ids") or ())
    matched = _target_span_ids({"text_span_ids": source_span_ids}, citation, spans_by_id)
    if matched:
        return matched
    # Ayat units often store only ``(n)`` in their span.  Once the lineage
    # resolver has selected the normative unit, those spans are already the
    # isolated target even when the canonical article label is not repeated.
    article_label = str(citation).partition(" ayat ")[0]
    unit_articles = [str(value) for value in unit.get("hierarchy") or () if str(value).startswith("Pasal ")]
    if source_span_ids and article_label and article_label in unit_articles:
        return source_span_ids
    return []


def _relation_support_refs(
    evidence: dict,
    mapping: dict,
    spans_by_id: dict[str, dict],
    bboxes_by_id: dict[str, dict],
) -> tuple[list[str], list[str]]:
    if not mapping:
        return list(evidence.get("text_span_ids") or ()), list(evidence.get("bbox_refs") or ())
    old_range = valid_text_range(mapping.get("old_range"), len(str(evidence.get("quoted_text") or "")))
    new_range = valid_text_range(mapping.get("new_range"), len(str(evidence.get("quoted_text") or "")))
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
    source_boxes = [bboxes_by_id[ref] for ref in source_bbox_refs if ref in bboxes_by_id]
    target_fragments = _target_fragments(source_text, citation)
    if not target_fragments or not source_boxes:
        return []
    words_by_page: dict[int, list[dict]] = {}
    for word in word_bboxes:
        if word.get("source_document_id") == source_document_id:
            words_by_page.setdefault(int(word["page_number"]), []).append(word)
    matches: list[list[str]] = []
    for page_number, words in words_by_page.items():
        offsets: list[int] = []
        cursor = 0
        for word in words:
            offsets.append(cursor)
            cursor += len(_compact(word.get("text", "")))
        page_boxes = [row for row in source_boxes if row.get("page_number") == page_number]
        for target, relative_offset in target_fragments:
            target_offsets = _page_target_offsets(source_text, relative_offset, words)
            for start in range(len(words)):
                joined = ""
                matched: list[dict] = []
                for word in words[start:]:
                    matched.append(word)
                    joined = _compact(f"{joined} {word.get('text', '')}")
                    if joined == target:
                        if offsets[start] in target_offsets and all(_word_inside_any_box(item, page_boxes) for item in matched):
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
    target_fragments = _target_fragments(source_text, citation)
    if not target_fragments:
        return []
    source_boxes = [bboxes_by_id[ref] for ref in source_bbox_refs if ref in bboxes_by_id]
    characters_by_page: dict[int, list[dict]] = {}
    for word in word_bboxes:
        if word.get("source_document_id") != source_document_id:
            continue
        characters_by_page.setdefault(int(word["page_number"]), []).extend(
            {
                **character,
                "page_number": word.get("page_number"),
                "source_document_id": word.get("source_document_id"),
                "word_bbox_id": word.get("word_bbox_id"),
            }
            for character in word.get("characters") or ()
        )
    matches: list[list[str]] = []
    for page_number, characters in characters_by_page.items():
        page_boxes = [row for row in source_boxes if row.get("page_number") == page_number]
        character_offsets: list[int] = []
        cursor = 0
        for character in characters:
            character_offsets.append(cursor)
            cursor += len(_compact(character.get("text") or ""))
        for target, relative_offset in target_fragments:
            target_offsets = _page_target_offsets(source_text, relative_offset, characters)
            for start in range(len(characters)):
                if not _compact(characters[start].get("text") or ""):
                    continue
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
                            and (not next_text.isalnum() or (next_character or {}).get("word_bbox_id") != matched[-1].get("word_bbox_id"))
                            and all(_word_inside_any_box(item, page_boxes) for item in matched)
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


def _page_target_offsets(source_text: str, relative_offset: int, tokens: list[dict]) -> set[int]:
    source = _compact(source_text)
    if not source:
        return set()
    page = "".join(_compact(token.get("text") or "") for token in tokens)
    starts: set[int] = set()
    cursor = page.find(source)
    while cursor >= 0:
        starts.add(cursor + relative_offset)
        cursor = page.find(source, cursor + 1)
    return starts


def _target_fragments(source_text: str, citation: str) -> tuple[tuple[str, int], ...]:
    """Locate the printed target, including contextual ayat shorthand, once."""
    direct = tuple(
        (_compact(match.group(0)), len(_compact(source_text[: match.start()])))
        for match in _reference_pattern(citation).finditer(source_text)
    )
    if direct:
        return direct
    match = re.fullmatch(r"Pasal\s+([0-9]+)([A-Za-z]?)\s+ayat\s+\((\d+)\)", citation, re.IGNORECASE)
    if match is None:
        return ()
    number, suffix, ayat = match.groups()
    article = re.compile(rf"(?i)\bpasal\s*{number}\s*{re.escape(suffix)}(?!\w)")
    # A PDF page may start with an ayat continuation while the article header
    # is on the preceding page.  The legal-unit span already binds the
    # continuation to its article, so the leading ayat marker is an exact
    # local target even when the header is not repeated on this page.
    leading = re.match(rf"\s*\(\s*{re.escape(ayat)}\s*\)", source_text)
    # Page context can begin with an ayat marker belonging to a different
    # article.  Prefer the target article segment whenever its header is
    # present; use the leading marker only for a continuation page that does
    # not repeat that header.
    if leading and not article.search(source_text):
        return ((_compact(leading.group(0)), len(_compact(source_text[: leading.start()]))),)
    boundary = re.compile(r"(?i)\b(?:pasal|bab)\s*[0-9ivxlcdm]")
    candidates: list[tuple[str, int]] = []
    for article_match in article.finditer(source_text):
        remainder = source_text[article_match.end() :]
        boundary_match = boundary.search(remainder)
        segment_end = article_match.end() + (boundary_match.start() if boundary_match else len(remainder))
        segment = source_text[article_match.end() : segment_end]
        target = re.compile(rf"(?i)(?:\bayat\s*)?\(\s*{re.escape(ayat)}\s*\)")
        for target_match in target.finditer(segment):
            start = article_match.end() + target_match.start()
            candidates.append((_compact(target_match.group(0)), len(_compact(source_text[:start]))))
    return tuple(candidates) if len(candidates) == 1 else ()


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


def _target_page_quote(span_ids: list[str], spans_by_id: dict[str, dict]) -> str:
    """Provide page context when a legal-unit span only contains ``(n)``."""
    if not span_ids:
        return ""
    anchor = spans_by_id.get(span_ids[0], {})
    stream_id = anchor.get("stream_id")
    if not stream_id:
        return ""
    page_spans = sorted(
        (row for row in spans_by_id.values() if row.get("stream_id") == stream_id),
        key=lambda row: int(row.get("text_start") or 0),
    )
    return " ".join(str(row.get("text") or "").strip() for row in page_spans if row.get("text")).strip()


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
    old_range = valid_text_range(mapping.get("old_range"), len(quoted))
    new_range = valid_text_range(mapping.get("new_range"), len(quoted))
    if old_range is None or new_range is None or old_range[1] > new_range[0]:
        return False
    if not _range_matches_reference(quoted[old_range[0] : old_range[1]], mapping.get("old_reference"), mapping.get("old_range_kind")):
        return False
    if not _range_matches_reference(quoted[new_range[0] : new_range[1]], mapping.get("new_reference"), mapping.get("new_range_kind")):
        return False
    if "menjadi" not in _normalize_reference(quoted[old_range[1] : new_range[0]]):
        return False
    return bool(evidence.get("text_span_ids")) and bool(evidence.get("bbox_refs"))


def _normalize_reference(value: object) -> str:
    return " ".join(str(value or "").casefold().split())


def _range_matches_reference(text: str, reference: object, kind: object) -> bool:
    expected = _normalize_reference(reference)
    actual = _normalize_reference(text)
    if kind != "contextual":
        return actual == expected
    return matches_uud_contextual_reference(text, reference)


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
