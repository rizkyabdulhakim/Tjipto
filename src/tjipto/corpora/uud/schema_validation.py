from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from hashlib import sha256
import re
from typing import cast

from tjipto.contracts.artifacts import (
    ARTIFACT_ALLOWED_FIELDS,
    ARTIFACT_OPTIONAL_FIELDS,
    COMMON_ARTIFACT_FIELDS,
    FORBIDDEN_ARTIFACT_FIELDS,
    MINIMUM_ARTIFACT_FIELDS,
)
from tjipto.contracts.evidence import normalize_source_text, valid_text_range
from tjipto.corpora.uud.parser import matches_uud_contextual_reference
from tjipto.corpora.uud.policy.source_text import validate_source_text_closure
from tjipto.ingestion.pdf.source_objects import TERMINAL_DISPOSITIONS


def _artifact_id_field(artifact: str) -> str:
    return {
        "bbox_registry": "bbox_id",
        "evidence_registry": "evidence_id",
        "page_text_spans": "text_span_id",
        "retrieval_units": "retrieval_unit_id",
        "metadata_grounding": "metadata_grounding_id",
        "article_amendment_relations": "relation_id",
        "graph_edges": "edge_id",
        "validation_exceptions": "exception_id",
        "source_documents": "source_document_id",
        "word_bboxes": "word_bbox_id",
        "promotion_decisions": "decision_id",
    }.get(artifact, "id")


def validate_schema_contract(artifacts: Mapping[str, object]) -> tuple[str, ...]:
    """Validate the closed UUD artifact contract by evidence owner."""

    def rows(name: str) -> list[dict]:
        return _artifact_rows(artifacts, name)

    errors = [*_artifact_field_errors(rows)]
    errors.extend(validate_source_text_closure(rows("raw_source_spans")))
    errors.extend(_source_object_errors(rows))
    errors.extend(_raw_source_errors(rows("raw_source_spans")))

    evidence = rows("evidence_registry")
    units = rows("legal_units")
    spans = rows("page_text_spans")
    bboxes = rows("bbox_registry")
    bbox_ids = {row.get("bbox_id") for row in bboxes}
    for word in rows("word_bboxes"):
        bbox_ids.add(word.get("word_bbox_id"))
        bbox_ids.update(character.get("character_bbox_id") for character in word.get("characters") or ())

    errors.extend(_source_marker_errors(spans, evidence))
    errors.extend(_proposition_errors(rows("propositions"), spans, bbox_ids))
    errors.extend(_geometry_errors(bboxes))
    errors.extend(_span_errors(spans, evidence, bbox_ids))
    errors.extend(_evidence_errors(evidence, units, spans, bboxes, bbox_ids, rows("source_documents")))
    errors.extend(_metadata_errors(rows("metadata_grounding"), evidence))
    errors.extend(_relation_errors(rows("article_amendment_relations"), rows("graph_edges"), evidence, units))
    return tuple(dict.fromkeys(errors))


def _artifact_rows(artifacts: Mapping[str, object], name: str) -> list[dict]:
    value = artifacts.get(name) or []
    return value if isinstance(value, list) else []


def _artifact_field_errors(rows) -> list[str]:
    errors: list[str] = []
    for artifact, required in MINIMUM_ARTIFACT_FIELDS.items():
        allowed = set(ARTIFACT_ALLOWED_FIELDS.get(artifact, ())) or (
            set(required) | set(ARTIFACT_OPTIONAL_FIELDS.get(artifact, ())) | set(COMMON_ARTIFACT_FIELDS)
        )
        for row in rows(artifact):
            row_id = str(row.get(_artifact_id_field(artifact)) or "<missing>")
            errors.extend(f"missing_required_field:{artifact}:{row_id}:{field}" for field in required if field not in row)
            errors.extend(
                f"owner_field_violation:{artifact}:{row_id}:{field}"
                for field in FORBIDDEN_ARTIFACT_FIELDS.get(artifact, ())
                if field in row
            )
            errors.extend(f"unknown_field:{artifact}:{row_id}:{field}" for field in row if field not in allowed)
            for field in ("viewer_highlightable", "citation_final", "citable", "evidence_exists", "runtime_loadable"):
                if field in row and not isinstance(row[field], bool):
                    errors.append(f"invalid_type:{artifact}:{row_id}:{field}")
    return errors


def _source_object_errors(rows) -> list[str]:
    errors: list[str] = []
    known_span_ids = {str(row.get("text_span_id") or "") for row in rows("page_text_spans")}
    seen: set[str] = set()
    for row in rows("source_objects"):
        object_id = str(row.get("source_object_id") or "")
        if not object_id:
            errors.append("missing_required_field:source_objects:<missing>:source_object_id")
        elif object_id in seen:
            errors.append(f"duplicate_source_object_id:{object_id}")
        else:
            seen.add(object_id)
        if row.get("disposition") not in TERMINAL_DISPOSITIONS:
            errors.append(f"invalid_source_object_disposition:{object_id}")
        errors.extend(
            f"source_object_span_unresolved:{object_id}:{span_id}"
            for span_id in row.get("text_span_ids") or ()
            if str(span_id) not in known_span_ids
        )
    return errors


def _raw_source_errors(raw_rows: list[dict]) -> list[str]:
    errors: list[str] = []
    support_ids: set[str] = set()
    semantic_streams: dict[str, list[dict]] = defaultdict(list)
    for row in raw_rows:
        row_errors, semantic_text = _raw_source_row_errors(row, support_ids)
        errors.extend(row_errors)
        if semantic_text:
            semantic_streams[str(row.get("semantic_stream_id"))].append(row)
    errors.extend(_semantic_stream_errors(semantic_streams))
    errors.extend(_raw_stream_errors(raw_rows))
    return errors


def _raw_source_row_errors(row: dict, support_ids: set[str]) -> tuple[list[str], str]:
    errors: list[str] = []
    row_id = str(row.get("raw_source_span_id") or "<missing>")
    support_id = row.get("source_support_id")
    if not isinstance(support_id, str) or not support_id:
        errors.append(f"missing_required_field:raw_source_spans:{row_id}:source_support_id")
    elif support_id in support_ids:
        errors.append(f"duplicate_source_support_id:{support_id}")
    else:
        support_ids.add(support_id)
    semantic_text = row.get("semantic_text")
    if not isinstance(semantic_text, str):
        errors.append(f"invalid_type:raw_source_spans:{row_id}:semantic_text")
        semantic_text = ""
    if semantic_text:
        errors.extend(_semantic_selector_errors(row, row_id, semantic_text))
    elif row.get("semantic_text_start") != 0 or row.get("semantic_text_end") != 0:
        errors.append(f"invalid_selector:{row_id}:empty_semantic_offset")
    if not semantic_text and any(
        row.get(field) is not False
        for field in ("legal_text", "citation_eligible", "relevant_quote_eligible", "default_highlight_eligible")
    ):
        errors.append(f"owner_field_violation:raw_source_spans:{row_id}:empty_semantic_policy")
    return errors, semantic_text


def _semantic_selector_errors(row: dict, row_id: str, semantic_text: str) -> list[str]:
    errors = []
    if row.get("semantic_exact_quote") != semantic_text:
        errors.append(f"invalid_selector:{row_id}:semantic_quote")
    start, end = row.get("semantic_text_start"), row.get("semantic_text_end")
    if not isinstance(start, int) or not isinstance(end, int):
        errors.append(f"invalid_selector:{row_id}:semantic_offset_type")
    elif end - start != len(semantic_text):
        errors.append(f"invalid_selector:{row_id}:semantic_offset_length")
    return errors


def _semantic_stream_errors(streams: dict[str, list[dict]]) -> list[str]:
    errors: list[str] = []
    for stream_rows in streams.values():
        stream_rows.sort(key=lambda item: int(item.get("extraction_order") or 0))
        stream_hash = sha256("\n".join(str(item["semantic_text"]) for item in stream_rows).encode()).hexdigest()
        cursor = 0
        for index, row in enumerate(stream_rows):
            row_id = row.get("raw_source_span_id")
            if row.get("semantic_stream_sha256") != stream_hash:
                errors.append(f"invalid_selector:{row_id}:semantic_stream_hash")
            start, end = row.get("semantic_text_start"), row.get("semantic_text_end")
            if not isinstance(start, int) or not isinstance(end, int):
                errors.append(f"invalid_selector:{row_id}:semantic_offset_type")
                continue
            if start != cursor or end != start + len(row["semantic_text"]):
                errors.append(f"invalid_selector:{row_id}:semantic_offset")
            cursor = end + (index < len(stream_rows) - 1)
    return errors


def _raw_stream_errors(raw_rows: list[dict]) -> list[str]:
    errors: list[str] = []
    stream_ids = dict.fromkeys(str(row.get("raw_stream_id")) for row in raw_rows)
    for stream_id in stream_ids:
        rows = sorted(
            (row for row in raw_rows if str(row.get("raw_stream_id")) == stream_id),
            key=lambda item: item.get("raw_text_start", 0),
        )
        max_end = max(int(row.get("raw_text_end") or 0) for row in rows)
        pieces = ["\n"] * max_end
        for row in rows:
            text = str(row.get("raw_text") or "")
            start, end = int(row.get("raw_text_start") or 0), int(row.get("raw_text_end") or 0)
            if end - start == len(text):
                pieces[start:end] = text
        stream = "".join(pieces)
        if sha256(stream.encode()).hexdigest() != str(rows[0].get("raw_stream_sha256")):
            errors.append(f"invalid_selector:{stream_id}:raw_stream_hash")
        for row in rows:
            errors.extend(_raw_row_errors(row, stream))
    return errors


def _raw_row_errors(row: dict, stream: str) -> list[str]:
    row_id = row.get("raw_source_span_id")
    errors: list[str] = []
    start, end, quote = row.get("raw_text_start"), row.get("raw_text_end"), row.get("raw_quote")
    if not isinstance(start, int) or not isinstance(end, int) or not isinstance(quote, str) or stream[start:end] != quote:
        errors.append(f"invalid_selector:{row_id}:raw_offset")
    if row.get("classification") == "source_annotation_marker" and any(
        row.get(field) is not False
        for field in ("legal_text", "citation_eligible", "relevant_quote_eligible", "default_highlight_eligible")
    ):
        errors.append(f"owner_field_violation:raw_source_spans:{row_id}:marker_policy")
    texts = row.get("character_texts") or []
    boxes = row.get("character_bboxes") or []
    if "".join(str(value) for value in texts) != str(row.get("raw_text") or ""):
        errors.append(f"raw_geometry_mismatch:{row_id}:character_text")
    if len(texts) != len(row.get("character_ids") or ()) or len(texts) != len(boxes):
        errors.append(f"raw_geometry_mismatch:{row_id}:character_lineage")
    if boxes:
        union = (
            min(float(b["x0"]) for b in boxes),
            min(float(b["y0"]) for b in boxes),
            max(float(b["x1"]) for b in boxes),
            max(float(b["y1"]) for b in boxes),
        )
        if union != tuple(float(cast(float, row.get(field))) for field in ("x0", "y0", "x1", "y1")):
            errors.append(f"raw_geometry_mismatch:{row_id}:union")
    return errors


def _source_marker_errors(spans: list[dict], evidence: list[dict]) -> list[str]:
    marker = re.compile(r"\*{1,4}(?:/\*{1,4})?\)")
    errors = [f"invalid_selector:{row.get('text_span_id')}:source_marker" for row in spans if marker.search(str(row.get("text") or ""))]
    for row in evidence:
        if marker.search(str(row.get("quoted_text") or "")):
            errors.append(f"owner_field_violation:evidence_registry:{row.get('evidence_id')}:source_marker")
        expected = (
            row.get("citable") is True
            and row.get("authority_kind") == "normative_legal_text"
            and row.get("evidence_owner_kind") == "legal_unit_source"
        )
        if row.get("relevant_quote_eligible") is not expected:
            errors.append(f"owner_field_violation:evidence_registry:{row.get('evidence_id')}:relevant_quote_eligible")
    return errors


def _proposition_errors(propositions: list[dict], spans: list[dict], bbox_ids: set[object]) -> list[str]:
    errors: list[str] = []
    spans_by_id = {row.get("text_span_id"): row for row in spans}
    for row in propositions:
        errors.extend(_proposition_row_errors(row, spans_by_id, bbox_ids))
    return errors


def _proposition_row_errors(row: dict, spans_by_id: dict[object, dict], bbox_ids: set[object]) -> list[str]:
    proposition_id = str(row.get("proposition_id") or "<missing>")
    selectors = row.get("source_selectors")
    if not isinstance(selectors, list) or not selectors:
        return [f"invalid_selector:{proposition_id}:missing"]
    errors: list[str] = []
    quotes: list[str] = []
    bboxes: list[str] = []
    for selector in selectors:
        selector_errors, quote, character_ids = _proposition_selector_errors(selector, spans_by_id, bbox_ids, proposition_id)
        errors.extend(selector_errors)
        if quote is not None:
            quotes.append(quote)
            bboxes.extend(character_ids)
    if "\n".join(quotes) != row.get("exact_quote"):
        errors.append(f"invalid_selector:{proposition_id}:quote")
    if tuple(dict.fromkeys(bboxes)) != tuple(row.get("bbox_refs") or ()):
        errors.append(f"invalid_selector:{proposition_id}:bbox_refs")
    return errors


def _proposition_selector_errors(
    selector: dict, spans_by_id: dict[object, dict], bbox_ids: set[object], proposition_id: str
) -> tuple[list[str], str | None, list[str]]:
    span = spans_by_id.get(selector.get("text_span_id"))
    start, end = selector.get("start"), selector.get("end")
    if not span or not isinstance(start, int) or not isinstance(end, int):
        return [f"invalid_selector:{proposition_id}:span"], None, []
    text_start = span.get("text_start")
    if not isinstance(text_start, int):
        return [f"invalid_selector:{proposition_id}:absolute_offset"], None, []
    errors: list[str] = []
    if any(selector.get(field) != span.get(field) for field in ("stream_id", "source_document_id", "source_sha256", "page_number")):
        errors.append(f"invalid_selector:{proposition_id}:source")
    if selector.get("absolute_start") != text_start + start or selector.get("absolute_end") != text_start + end:
        errors.append(f"invalid_selector:{proposition_id}:absolute_offset")
    text = str(span.get("text") or "")
    if selector.get("prefix") != text[max(0, start - 32) : start] or selector.get("suffix") != text[end : end + 32]:
        errors.append(f"invalid_selector:{proposition_id}:context")
    character_ids = selector.get("character_bbox_ids") or []
    if not character_ids or any(value not in bbox_ids for value in character_ids):
        errors.append(f"invalid_selector:{proposition_id}:geometry")
    return errors, text[start:end], character_ids


def _geometry_errors(bboxes: list[dict]) -> list[str]:
    errors: list[str] = []
    seen: set[tuple[object, ...]] = set()
    fields = (
        "source_document_id",
        "source_sha256",
        "page_number",
        "coordinate_space",
        "coordinate_origin",
        "x0",
        "y0",
        "x1",
        "y1",
        "transform_version",
        "text",
    )
    for row in bboxes:
        identity = tuple(row.get(field) for field in fields)
        if identity in seen:
            errors.append(f"duplicate_geometry:{row.get('bbox_id')}")
        seen.add(identity)
    return errors


def _span_errors(spans: list[dict], evidence: list[dict], bbox_ids: set[object]) -> list[str]:
    errors: list[str] = []
    evidence_ids = {row.get("evidence_id") for row in evidence}
    streams: dict[str, list[dict]] = defaultdict(list)
    for span in spans:
        streams[str(span.get("stream_id"))].append(span)
    stream_values: dict[str, str] = {}
    for stream_id, stream_rows in streams.items():
        stream_rows.sort(key=lambda item: item.get("text_start", 0))
        stream = "\n".join(str(item.get("text") or "") for item in stream_rows)
        stream_values[stream_id] = stream
        digest = sha256(stream.encode()).hexdigest()
        if any(item.get("page_text_hash") != digest for item in stream_rows):
            errors.append(f"invalid_selector:{stream_id}:stream_hash")
    for row in spans:
        errors.extend(_span_row_errors(row, stream_values, evidence_ids, bbox_ids))
    return errors


def _span_row_errors(row: dict, stream_values: dict[str, str], evidence_ids: set[object], bbox_ids: set[object]) -> list[str]:
    row_id = row.get("text_span_id")
    start, end, quote = row.get("text_start"), row.get("text_end"), row.get("exact_quote")
    errors: list[str] = []
    if not isinstance(start, int) or not isinstance(end, int) or not isinstance(quote, str) or end < start or end - start != len(quote):
        errors.append(f"invalid_selector:{row_id}")
    if quote != row.get("text"):
        errors.append(f"invalid_selector:{row_id}:quote")
    if isinstance(start, int) and isinstance(end, int) and stream_values.get(str(row.get("stream_id")), "")[start:end] != quote:
        errors.append(f"invalid_selector:{row_id}:offset")
    if row.get("semantic_classification") == "normative_constitutional_text" and row.get("linked_authority") == "rejected":
        errors.append(f"owner_field_violation:page_text_spans:{row_id}:linked_authority")
    if any(ref not in evidence_ids for ref in row.get("evidence_ids") or ()):
        errors.append(f"invalid_reference:page_text_spans:{row_id}:evidence_ids")
    if any(ref not in bbox_ids for ref in row.get("span_bbox_ids") or ()):
        errors.append(f"invalid_reference:page_text_spans:{row_id}:span_bbox_ids")
    return errors


def _evidence_errors(
    evidence: list[dict], units: list[dict], spans: list[dict], bboxes: list[dict], bbox_ids: set[object], sources: list[dict]
) -> list[str]:
    errors: list[str] = []
    units_by_id = {row.get("legal_unit_id"): row for row in units}
    spans_by_id = {row.get("text_span_id"): row for row in spans}
    sources_by_id = {row.get("source_document_id"): row for row in sources}
    source_ids = set(sources_by_id)
    for row in evidence:
        row_id = str(row.get("evidence_id") or "")
        if row.get("legal_unit_id") is None or row.get("source_document_id") not in source_ids:
            errors.append(f"invalid_reference:evidence_registry:{row_id}")
        source = sources_by_id.get(row.get("source_document_id"))
        if source and row.get("source_sha256") != source.get("sha256"):
            errors.append(f"EVIDENCE_SOURCE_LINEAGE_INVALID:{row_id}")
        if any(ref not in bbox_ids for ref in row.get("bbox_refs") or ()):
            errors.append(f"invalid_reference:evidence_registry:{row_id}:bbox_refs")
        if any(ref not in spans_by_id for ref in row.get("text_span_ids") or ()):
            errors.append(f"invalid_reference:evidence_registry:{row_id}:text_span_ids")
        span_text = " ".join(str(spans_by_id.get(ref, {}).get("text") or "") for ref in row.get("text_span_ids") or ())
        if row.get("quoted_text") and normalize_source_text(row.get("quoted_text")) not in normalize_source_text(span_text):
            errors.append(f"EVIDENCE_QUOTE_SOURCE_MISMATCH:{row_id}")
        if row.get("citation_final") is True and row.get("exactness") != "exact":
            errors.append(f"owner_field_violation:evidence_registry:{row_id}:citation_final")
        unit = units_by_id.get(row.get("legal_unit_id"), {})
        if (
            unit.get("unit_type") in {"bab_record", "pasal_record"}
            and any(unit.get("legal_unit_id") in (candidate.get("parent_legal_unit_ids") or ()) for candidate in units)
            and not set(unit.get("text_span_ids") or ()) <= set(row.get("text_span_ids") or ())
        ):
            errors.append(f"aggregate_text_span_sequence_incomplete:{row_id}")
    referenced = {ref for row in evidence for ref in row.get("bbox_refs") or ()}
    if any(row.get("bbox_id") not in referenced for row in bboxes):
        errors.append("invalid_reference:orphan_geometry")
    return errors


def _metadata_errors(rows: list[dict], evidence: list[dict]) -> list[str]:
    evidence_by_id = {row.get("evidence_id"): row for row in evidence}
    return [
        f"signatory_grounding_unknown_name:{row.get('metadata_grounding_id')}"
        for row in rows
        if "signatories" in str(row.get("metadata_grounding_id") or "")
        and row.get("quoted_text")
        and (
            donor := " ".join(str(evidence_by_id.get(ref, {}).get("quoted_text") or "") for ref in row.get("supporting_evidence_ids") or ())
        )
        and normalize_source_text(row.get("quoted_text")) not in normalize_source_text(donor)
    ]


def _relation_errors(relations: list[dict], graph_edges: list[dict], evidence: list[dict], units: list[dict]) -> list[str]:
    errors: list[str] = []
    evidence_by_id = {row.get("evidence_id"): row for row in evidence}
    units_by_id = {row.get("legal_unit_id"): row for row in units}
    for row in relations:
        errors.extend(_relation_row_errors(row, evidence_by_id, units_by_id))
    errors.extend(_graph_relation_errors(graph_edges, {row.get("relation_id"): row for row in relations}))
    return errors


def _relation_row_errors(row: dict, evidence_by_id: dict[object, dict], units_by_id: dict[object, dict]) -> list[str]:
    relation_id = str(row.get("relation_id") or "<missing>")
    relation_type = row.get("relation_type")
    errors: list[str] = []
    if relation_type not in {"MODIFIES", "DELETES", "ADDS", "AMBIGUOUS_OPERATION", "RENAMES", "RENUMBERED_TO", "SUPPLEMENTS"}:
        errors.append(f"invalid_enum:article_amendment_relations:{relation_id}:relation_type")
    evidence_row = evidence_by_id.get(row.get("evidence_id"))
    if evidence_row is None:
        return [*errors, f"invalid_reference:article_amendment_relations:{relation_id}:evidence_id"]
    if row.get("source_pdf_sha256") != evidence_row.get("source_sha256"):
        errors.append(f"article_relation_source_sha_mismatch:{relation_id}")
    if row.get("bbox_precision") != evidence_row.get("bbox_precision"):
        errors.append(f"article_relation_bbox_precision_mismatch:{relation_id}")
    if set(row.get("bbox_refs") or ()) - set(row.get("target_bbox_refs") or ()) - set(evidence_row.get("bbox_refs") or ()):
        errors.append(f"article_relation_bbox_source_mismatch:{relation_id}")
    if row.get("support_class") == "exact_article_relation" and row.get("target_precision") != "target_local":
        errors.append(f"article_relation_exact_precision_mismatch:{relation_id}")
    errors.extend(_relation_unit_errors(row, units_by_id, relation_id))
    if relation_type in {"RENAMES", "RENUMBERED_TO"}:
        errors.extend(f"{code}:{relation_id}" for code in _validate_mapping_support(row, evidence_row))
    return errors


def _relation_unit_errors(row: dict, units_by_id: dict[object, dict], relation_id: str) -> list[str]:
    errors: list[str] = []
    source_unit = units_by_id.get(row.get("source_legal_unit_id"))
    target_unit = units_by_id.get(row.get("target_legal_unit_id"))
    if source_unit is None:
        errors.append(f"article_relation_unknown_source:{relation_id}")
    elif row.get("source_label") != source_unit.get("unit_label"):
        errors.append(f"article_relation_source_label_mismatch:{relation_id}")
    if target_unit is None:
        errors.append(f"article_relation_unknown_target:{relation_id}")
    elif row.get("target_source_role") != target_unit.get("source_role"):
        errors.append(f"article_relation_target_role_mismatch:{relation_id}")
    return errors


def _graph_relation_errors(graph_edges: list[dict], relations_by_id: dict[object, dict]) -> list[str]:
    errors: list[str] = []
    for row in graph_edges:
        relation = relations_by_id.get(str(row.get("relation_id") or row.get("article_relation_ref") or ""))
        if row.get("edge_type") not in {"RENAMES", "RENUMBERED_TO"} or relation is None:
            continue
        if row.get("source_id") != f"legal_unit::{relation.get('source_legal_unit_id')}":
            errors.append(f"graph_relation_source_mismatch:{row.get('edge_id')}")
        if row.get("target_id") != f"legal_unit::{relation.get('target_legal_unit_id')}":
            errors.append(f"graph_relation_target_mismatch:{row.get('edge_id')}")
    return errors


def _validate_mapping_support(row: dict, evidence: dict) -> tuple[str, ...]:
    quoted = str(evidence.get("quoted_text") or "")
    old_range = valid_text_range(row.get("old_reference_range"), len(quoted))
    new_range = valid_text_range(row.get("new_reference_range"), len(quoted))
    errors: list[str] = []
    if old_range is None or new_range is None:
        return ("article_relation_invalid_reference_range",)
    if old_range[1] > new_range[0]:
        errors.append("article_relation_reference_range_order")
    if not _validator_range_matches_reference(
        quoted[old_range[0] : old_range[1]], row.get("old_reference"), row.get("old_reference_range_kind")
    ):
        errors.append("article_relation_old_range_text_mismatch")
    if not _validator_range_matches_reference(
        quoted[new_range[0] : new_range[1]], row.get("new_reference"), row.get("new_reference_range_kind")
    ):
        errors.append("article_relation_new_range_text_mismatch")
    if "menjadi" not in normalize_source_text(quoted[old_range[1] : new_range[0]]):
        errors.append("article_relation_missing_transition_support")
    if not evidence.get("text_span_ids") or not evidence.get("bbox_refs"):
        errors.append("article_relation_missing_support_segment")
    return tuple(errors)


def _validator_range_matches_reference(text: str, reference: object, kind: object) -> bool:
    expected = normalize_source_text(reference)
    actual = normalize_source_text(text)
    if kind != "contextual":
        return actual == expected
    return matches_uud_contextual_reference(text, reference)
