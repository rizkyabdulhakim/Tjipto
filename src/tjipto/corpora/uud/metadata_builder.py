from __future__ import annotations

from collections import defaultdict
import re
import unicodedata

from tjipto.evidence.store import exact_bboxes_for_text_spans
from tjipto.corpora.uud.specs import METADATA_BLOCK_SPECS
from tjipto.corpora.uud.structure_builder import compact, matching_sequence
from tjipto.ingestion.pdf.words import align_text_to_word_bboxes, word_rows_by_page


AMENDMENT_FIELD_STATUSES = {
    "date": "grounded",
    "institution": "grounded",
    "ln_tln": "not_found_in_source",
    "official_title": "not_found_in_source",
    "penetapan": "grounded",
    "pengundangan": "not_found_in_source",
    "place": "grounded",
    "promulgation": "not_found_in_source",
    "signatories": "grounded",
    "source_publication": "not_applicable",
}

GROUNDING_FIELD_ORDER = (
    "date",
    "institution",
    "penetapan",
    "place",
    "signatories",
    "decision_date",
    "decision_session",
    "effective_rule",
    "official_title",
    "source_publication",
)

# Indonesian legal office labels parsed from source PDFs; these are not credentials.
CHAIR_TOKEN = "Ketua,"  # nosec B105
VICE_CHAIR_TOKEN = "Wakil"  # nosec B105
CHAIR_ROLE = "Ketua"
VICE_CHAIR_ROLE = "Wakil Ketua"


def build_document_metadata(source_documents: dict[str, dict], metadata_grounding: list[dict]) -> list[dict]:
    block_by_role = {row["source_role"]: row for row in metadata_grounding}
    rows = [
        _amendment_document_metadata(source, block_by_role[source["source_role"]])
        for source in source_documents.values()
        if source["source_role"].startswith("amendment_")
    ]
    rows.append(_current_document_metadata(source_documents["uud::current_consolidated"], block_by_role["current_consolidated"]))
    rows.append(_original_document_metadata(source_documents["uud::original_historical"]))
    return sorted(rows, key=lambda row: row["document_metadata_id"])


def build_metadata_assertions(evidence: list[dict], metadata_grounding: list[dict], bbox_rows: list[dict]) -> list[dict]:
    bbox_by_id = {row["bbox_id"]: row for row in bbox_rows}
    rows: list[dict] = []
    for row in sorted(evidence, key=_metadata_evidence_sort_key):
        if row["evidence_id"].startswith("uud_instrument_final_citation_evidence::"):
            continue
        rows.extend(_evidence_metadata_assertions(row, bbox_by_id))
    for row in metadata_grounding:
        if not row["metadata_grounding_id"].startswith("uud_metadata_block_final_evidence::"):
            continue
        rows.append(_block_metadata_assertion(row))
    return rows


def build_metadata_graph_edges(metadata_assertions: list[dict]) -> list[dict]:
    rows: list[dict] = []
    block_rows = []
    for row in metadata_assertions:
        if row["predicate"] == "legal_unit_identity":
            rows.append(
                _metadata_graph_edge(
                    edge_id=f"uud_legal_graph_edge::{_evidence_link_target_id(row)}::has_metadata",
                    edge_type="HAS_METADATA",
                    row=row,
                    source_id=f"legal_unit::{row['subject_id']}",
                )
            )
        elif row["predicate"] in {"instrument_closing_and_issuance_block", "source_publication_metadata_block"}:
            block_rows.append(row)
    for row in block_rows:
        role = row["source_role"]
        slug = _evidence_link_target_id(row).rsplit("::", 1)[-1]
        if row["predicate"] == "source_publication_metadata_block":
            rows.append(
                _metadata_graph_edge(
                    edge_id=f"uud_legal_graph_edge::{role}::{slug}::source_published_by::institution_majelis_permusyawaratan_rakyat_sekretariat_jenderal",
                    edge_type="SOURCE_PUBLISHED_BY",
                    row=row,
                    source_id=f"source_role::{role}",
                )
            )
            continue
        for edge_type, suffix in (
            ("ISSUED_BY", "issued_by::institution_majelis_permusyawaratan_rakyat_republik_indonesia"),
            ("SIGNED_BY", f"signed_by::signature_block_{role}_{slug}"),
        ):
            rows.append(
                _metadata_graph_edge(
                    edge_id=f"uud_legal_graph_edge::{role}::{slug}::{suffix}",
                    edge_type=edge_type,
                    row=row,
                    source_id=f"source_role::{role}",
                )
            )
        if "diputuskan" in row["value"].casefold():
            rows.append(
                _metadata_graph_edge(
                    edge_id=f"uud_legal_graph_edge::{role}::{slug}::decided_by::institution_majelis_permusyawaratan_rakyat_republik_indonesia",
                    edge_type="DECIDED_BY",
                    row=row,
                    source_id=f"source_role::{role}",
                )
            )
    return rows


def _metadata_evidence_sort_key(row: dict) -> tuple[int, str]:
    evidence_id = row["evidence_id"]
    prefix_order = (
        "uud_current_consolidated_final_citation_evidence_",
        "uud_source_role_final_citation_evidence_",
        "uud_source_role_historical_final_citation_evidence_",
        "uud_source_role_additional_final_citation_evidence_",
    )
    for index, prefix in enumerate(prefix_order):
        if evidence_id.startswith(prefix):
            return index, evidence_id
    return len(prefix_order), evidence_id


def _metadata_graph_edge(*, edge_id: str, edge_type: str, row: dict, source_id: str) -> dict:
    return {
        "corpus_id": "uud",
        "edge_id": edge_id,
        "edge_type": edge_type,
        "evidence_link": row["evidence_link"],
        "runtime_loadable": False,
        "source_id": source_id,
        "source_role": row["source_role"],
        "status": "accepted",
        "target_id": row["metadata_id"],
        "temporal_context": row["temporal_context"],
        "validator_status": "valid",
    }


def _evidence_link_target_id(row: dict) -> str:
    link = row["evidence_link"]
    return link.get("metadata_grounding_id") or link["final_evidence_id"]


def build_metadata_block_grounding(
    *,
    pages_by_source: dict[tuple[str, int], str],
    source_documents: dict[str, dict],
) -> list[dict]:
    source_by_role = {row["source_role"]: row for row in source_documents.values()}
    rows = []
    for spec in METADATA_BLOCK_SPECS:
        role = spec["source_role"]
        source = source_by_role[role]
        quoted_text = _metadata_quote(spec, pages_by_source, source["source_document_id"])
        metadata_grounding_id = f"uud_metadata_block_final_evidence::{role}::{spec['slug']}"
        rows.append(
            {
                "bbox_refs": [f"uud_metadata_block_bbox::{role}::{spec['slug']}::0000"],
                "corpus_id": "uud",
                "metadata_grounding_id": metadata_grounding_id,
                "page_numbers": [spec["page_number"]],
                "provenance": {"donor_id": metadata_grounding_id},
                "quoted_text": quoted_text,
                "source_document_id": source["source_document_id"],
                "source_pdf_path": source["path"],
                "source_role": role,
                "source_sha256": source["sha256"],
                "status": "accepted_metadata_grounding",
                "temporal_context": role,
                "viewer_highlightable": False,
            }
        )
    return rows


def rebuild_metadata_grounding(
    *,
    document_metadata: list[dict],
    metadata_grounding: list[dict],
    evidence: list[dict],
    bbox_rows: list[dict],
    word_bboxes: list[dict],
    legal_units: list[dict],
    page_text_spans: list[dict],
    source_conflicts: list[dict],
) -> tuple[list[dict], list[dict], list[dict]]:
    evidence_by_key = {(row.get("source_role"), row.get("citation")): row for row in evidence}
    bboxes_by_evidence: dict[str, list[dict]] = defaultdict(list)
    for row in bbox_rows:
        bboxes_by_evidence[row["evidence_id"]].append(row)
    words_by_page = word_rows_by_page(word_bboxes)
    spans_by_id = {row["text_span_id"]: row for row in page_text_spans if row.get("text_span_id")}
    source_role_by_id = {row["source_document_id"]: row["source_role"] for row in document_metadata}
    units_by_key = {(source_role_by_id[row["source_document_id"]], row.get("unit_label")): row for row in legal_units}
    block_rows: list[dict] = []
    block_registry_rows: list[dict] = []
    for row in metadata_grounding:
        if str(row.get("metadata_grounding_id", "")).startswith("uud_metadata_field_grounding::"):
            continue
        page_numbers = list(row.get("page_numbers") or ())
        text_span_ids = _exact_text_span_ids(
            str(row.get("quoted_text") or ""),
            str(row.get("source_document_id") or ""),
            page_numbers,
            page_text_spans,
        )
        exact_bbox_rows = []
        if text_span_ids and _allow_global_exact_bbox_reuse(row.get("source_role", ""), "block"):
            exact_bbox_rows = exact_bboxes_for_text_spans([spans_by_id.get(text_span_id) for text_span_id in text_span_ids], bbox_rows)
        if not exact_bbox_rows and _allow_word_exact_bbox_promotion(row.get("source_role", ""), row.get("metadata_field", "block")):
            exact_bbox_rows = _word_exact_bbox_rows(
                quoted_text=str(row.get("quoted_text") or ""),
                source_document_id=str(row.get("source_document_id") or ""),
                page_numbers=page_numbers,
                page_text_spans=page_text_spans,
                words_by_page=words_by_page,
            )
        exact_bbox_refs = [item["bbox_id"] for item in exact_bbox_rows]
        block_row = row | {
            "bbox_ids": exact_bbox_refs,
            "bbox_precision": "exact" if exact_bbox_refs and text_span_ids else "page_grounded_only",
            "grounding_status": "text_bbox_exact"
            if exact_bbox_refs and text_span_ids
            else row.get("grounding_status") or "block_level_grounded",
            "text_span_ids": text_span_ids,
            "failure_reason": None
            if exact_bbox_refs and text_span_ids
            else row.get("failure_reason")
            or _metadata_failure_reason(
                row.get("source_role", ""), row.get("metadata_field", "block"), row.get("metadata_grounding_id", "")
            ),
            "viewer_highlightable": bool(exact_bbox_refs and text_span_ids),
        }
        block_row["bbox_refs"] = exact_bbox_refs or block_row["bbox_refs"]
        block_rows.append(block_row)
        block_registry_rows.extend(_block_registry_rows(block_row, exact_bbox_rows))
    source_conflicts_by_role: dict[str, list[dict]] = defaultdict(list)
    for row in source_conflicts:
        source_conflicts_by_role[source_role_by_id[row["source_document_id"]]].append(row)
    field_rows: list[dict] = []
    field_registry_rows: list[dict] = []

    def append_grounding(
        *,
        source_role: str,
        source_document_id: str,
        metadata_field: str,
        quoted_text: str,
        donor_id: str,
        page_numbers: list[int] | tuple[int, ...],
        source_pdf_path: str,
        source_sha256: str,
    ) -> str:
        field_id = f"uud_metadata_field_grounding::{source_role}::{metadata_field}"
        exact_bbox_rows = _exact_bbox_rows(quoted_text, bboxes_by_evidence.get(donor_id, []))
        text_span_ids = _exact_text_span_ids(quoted_text, source_document_id, page_numbers, page_text_spans)
        if not exact_bbox_rows and text_span_ids and _allow_global_exact_bbox_reuse(source_role, metadata_field):
            exact_bbox_rows = exact_bboxes_for_text_spans([spans_by_id.get(text_span_id) for text_span_id in text_span_ids], bbox_rows)
        if not text_span_ids:
            text_span_ids = _supporting_text_span_ids(quoted_text, source_document_id, page_numbers, page_text_spans)
        if not exact_bbox_rows and _allow_word_exact_bbox_promotion(source_role, metadata_field):
            exact_bbox_rows = _word_exact_bbox_rows(
                quoted_text=quoted_text,
                source_document_id=source_document_id,
                page_numbers=page_numbers,
                page_text_spans=page_text_spans,
                words_by_page=words_by_page,
            )
        exact_bbox_refs = [row["bbox_id"] for row in exact_bbox_rows]
        exact_safe = bool(exact_bbox_refs and text_span_ids)
        bbox_refs = exact_bbox_refs or [
            f"uud_metadata_field_bbox::{source_role}::{metadata_field}::{index:04d}" for index, _ in enumerate(page_numbers)
        ]
        bbox_precision = "exact" if exact_safe else "page_grounded_only"
        grounding_status = "text_bbox_exact" if exact_safe else "field_level_grounded"
        failure_reason = None if exact_safe else _metadata_failure_reason(source_role, metadata_field, donor_id)
        registry_rows = exact_bbox_rows or [
            {"bbox_id": bbox_id, "page_number": page_number} for bbox_id, page_number in zip(bbox_refs, page_numbers)
        ]
        viewer_highlightable = exact_safe
        for index, registry_row in enumerate(registry_rows):
            field_registry_rows.append(
                {
                    "bbox_id": registry_row["bbox_id"],
                    "bbox_precision": bbox_precision,
                    "corpus_id": "uud",
                    "failure_reason": failure_reason,
                    "metadata_grounding_id": field_id,
                    "metadata_grounding_ref_id": _metadata_grounding_ref_id(field_id, metadata_field, registry_row["bbox_id"], index),
                    "metadata_field": metadata_field,
                    "page_number": registry_row["page_number"],
                    "quoted_text": quoted_text,
                    "source_document_id": source_document_id,
                    "source_pdf_path": source_pdf_path,
                    "source_sha256": source_sha256,
                    "status": "accepted_metadata_grounding",
                    "text_span_ids": text_span_ids,
                    "viewer_highlightable": viewer_highlightable,
                }
            )
        field_rows.append(
            {
                "bbox_ids": exact_bbox_refs,
                "bbox_precision": bbox_precision,
                "bbox_refs": bbox_refs,
                "corpus_id": "uud",
                "grounding_status": grounding_status,
                "metadata_field": metadata_field,
                "metadata_grounding_id": field_id,
                "page_numbers": list(page_numbers),
                "provenance": {"donor_id": donor_id},
                "quote": quoted_text,
                "quoted_text": quoted_text,
                "runtime_loadable": False,
                "source_document_id": source_document_id,
                "source_pdf_path": source_pdf_path,
                "source_role": source_role,
                "source_sha256": source_sha256,
                "status": "accepted_metadata_grounding",
                "temporal_context": source_role,
                "text_span_ids": text_span_ids,
                "viewer_highlightable": viewer_highlightable,
            }
            | ({"failure_reason": failure_reason} if failure_reason else {})
        )
        return field_id

    for row in document_metadata:
        field_statuses = dict(row.get("field_statuses") or {})
        grounded_fields = {key: list(value) for key, value in (row.get("grounded_fields") or {}).items()}
        role = row["source_role"]
        for key in (
            "decision_date",
            "decision_session",
            "effective_rule",
            "effective_date",
            "promulgation_date",
            "revocation_date",
            "source_anomaly_status",
            "source_publication",
        ):
            field_statuses.setdefault(key, "not_found_in_source")
        if role.startswith("amendment_"):
            source_document_id = row["source_document_id"]
            ordinal = _ordinal_label(role)
            determination = evidence_by_key.get((role, f"Perubahan {ordinal} Determination"))
            decision = evidence_by_key.get((role, f"Perubahan {ordinal} Decision"))
            signatories = evidence_by_key.get((role, f"Perubahan {ordinal} Signatories"))
            effective = units_by_key.get((role, f"Perubahan {ordinal} Effective"))
            if determination:
                grounding_id = append_grounding(
                    source_role=role,
                    source_document_id=source_document_id,
                    metadata_field="penetapan",
                    quoted_text=determination["quoted_text"],
                    donor_id=determination["evidence_id"],
                    page_numbers=determination["page_numbers"],
                    source_pdf_path=determination["source_pdf_path"],
                    source_sha256=determination["source_sha256"],
                )
                grounded_fields["penetapan"] = [grounding_id]
                field_statuses["penetapan"] = "grounded"
                date_text = row.get("penetapan", {}).get("date_text")
                if date_text:
                    date_quote = next(
                        (line for line in determination["quoted_text"].splitlines() if date_text in line), determination["quoted_text"]
                    )
                    grounded_fields["date"] = [
                        append_grounding(
                            source_role=role,
                            source_document_id=source_document_id,
                            metadata_field="date",
                            quoted_text=date_quote,
                            donor_id=determination["evidence_id"],
                            page_numbers=determination["page_numbers"],
                            source_pdf_path=determination["source_pdf_path"],
                            source_sha256=determination["source_sha256"],
                        )
                    ]
                    field_statuses["date"] = "grounded"
                place_text = row.get("place")
                if place_text:
                    place_quote = next(
                        (line for line in determination["quoted_text"].splitlines() if place_text in line), determination["quoted_text"]
                    )
                    grounded_fields["place"] = [
                        append_grounding(
                            source_role=role,
                            source_document_id=source_document_id,
                            metadata_field="place",
                            quoted_text=place_quote,
                            donor_id=determination["evidence_id"],
                            page_numbers=determination["page_numbers"],
                            source_pdf_path=determination["source_pdf_path"],
                            source_sha256=determination["source_sha256"],
                        )
                    ]
                    field_statuses["place"] = "grounded"
            if signatories:
                institution_quote = "\n".join(signatories["quoted_text"].splitlines()[:2]).strip()
                grounded_fields["institution"] = [
                    append_grounding(
                        source_role=role,
                        source_document_id=source_document_id,
                        metadata_field="institution",
                        quoted_text=institution_quote,
                        donor_id=signatories["evidence_id"],
                        page_numbers=signatories["page_numbers"],
                        source_pdf_path=signatories["source_pdf_path"],
                        source_sha256=signatories["source_sha256"],
                    )
                ]
                grounded_fields["signatories"] = [
                    append_grounding(
                        source_role=role,
                        source_document_id=source_document_id,
                        metadata_field="signatories",
                        quoted_text=signatories["quoted_text"],
                        donor_id=signatories["evidence_id"],
                        page_numbers=signatories["page_numbers"],
                        source_pdf_path=signatories["source_pdf_path"],
                        source_sha256=signatories["source_sha256"],
                    )
                ]
                field_statuses["institution"] = "grounded"
                field_statuses["signatories"] = "grounded"
            if decision:
                decision_date = _extract_metadata_date(decision["quoted_text"])
                decision_session = _extract_decision_session(decision["quoted_text"], decision_date)
                if decision_date:
                    row["decision_date"] = decision_date
                    grounded_fields["decision_date"] = [
                        append_grounding(
                            source_role=role,
                            source_document_id=source_document_id,
                            metadata_field="decision_date",
                            quoted_text=decision["quoted_text"],
                            donor_id=decision["evidence_id"],
                            page_numbers=decision["page_numbers"],
                            source_pdf_path=decision["source_pdf_path"],
                            source_sha256=decision["source_sha256"],
                        )
                    ]
                    field_statuses["decision_date"] = "grounded"
                if decision_session:
                    row["decision_session"] = decision_session
                    grounded_fields["decision_session"] = [
                        append_grounding(
                            source_role=role,
                            source_document_id=source_document_id,
                            metadata_field="decision_session",
                            quoted_text=decision["quoted_text"],
                            donor_id=decision["evidence_id"],
                            page_numbers=decision["page_numbers"],
                            source_pdf_path=decision["source_pdf_path"],
                            source_sha256=decision["source_sha256"],
                        )
                    ]
                    field_statuses["decision_session"] = "grounded"
            if effective:
                row["effective_rule"] = effective["text"]
                grounded_fields["effective_rule"] = [
                    append_grounding(
                        source_role=role,
                        source_document_id=source_document_id,
                        metadata_field="effective_rule",
                        quoted_text=effective["text"],
                        donor_id=effective["legal_unit_id"],
                        page_numbers=list(range(effective["page_start"], effective["page_end"] + 1)),
                        source_pdf_path=next(
                            item["source_pdf_path"] for item in evidence if item["source_document_id"] == source_document_id
                        ),
                        source_sha256=effective["source_sha256"],
                    )
                ]
                field_statuses["effective_rule"] = "grounded"
            anomaly_rows = source_conflicts_by_role.get(role) or []
            if anomaly_rows:
                row["source_anomaly_status"] = anomaly_rows[0]["classification"]
                field_statuses["source_anomaly_status"] = "artifact_recorded"
        elif role == "current_consolidated":
            block = next((item for item in block_rows if item["source_role"] == role), None)
            source_document_id = row["source_document_id"]
            source_pdf_path = block["source_pdf_path"] if block else ""
            source_sha256 = block["source_sha256"] if block else ""
            page_numbers = block["page_numbers"] if block else [1]
            if row.get("institution"):
                grounded_fields["institution"] = [
                    append_grounding(
                        source_role=role,
                        source_document_id=source_document_id,
                        metadata_field="institution",
                        quoted_text=row["institution"],
                        donor_id=block["metadata_grounding_id"] if block else row["document_metadata_id"],
                        page_numbers=page_numbers,
                        source_pdf_path=source_pdf_path,
                        source_sha256=source_sha256,
                    )
                ]
                field_statuses["institution"] = "grounded"
            if row.get("official_title"):
                grounded_fields["official_title"] = [
                    append_grounding(
                        source_role=role,
                        source_document_id=source_document_id,
                        metadata_field="official_title",
                        quoted_text=row["official_title"],
                        donor_id=block["metadata_grounding_id"] if block else row["document_metadata_id"],
                        page_numbers=page_numbers,
                        source_pdf_path=source_pdf_path,
                        source_sha256=source_sha256,
                    )
                ]
                field_statuses["official_title"] = "grounded"
            if row.get("source_publication"):
                grounded_fields["source_publication"] = [
                    append_grounding(
                        source_role=role,
                        source_document_id=source_document_id,
                        metadata_field="source_publication",
                        quoted_text=block["quoted_text"] if block else row["official_title"],
                        donor_id=block["metadata_grounding_id"] if block else row["document_metadata_id"],
                        page_numbers=page_numbers,
                        source_pdf_path=source_pdf_path,
                        source_sha256=source_sha256,
                    )
                ]
                field_statuses["source_publication"] = "grounded"
        row["field_statuses"] = field_statuses
        row["grounded_fields"] = {key: tuple(grounded_fields[key]) for key in GROUNDING_FIELD_ORDER if grounded_fields.get(key)}
        row["grounding_refs"] = tuple(dict.fromkeys(ref for refs in row["grounded_fields"].values() for ref in refs))
    all_grounding_rows = block_rows + field_rows
    all_registry_rows = block_registry_rows + field_registry_rows
    return document_metadata, all_grounding_rows, all_registry_rows


def _block_registry_rows(row: dict, exact_bbox_rows: list[dict]) -> list[dict]:
    if row.get("bbox_precision") == "exact":
        return [
            {
                "bbox_id": bbox_row["bbox_id"],
                "bbox_precision": "exact",
                "corpus_id": row["corpus_id"],
                "failure_reason": None,
                "metadata_grounding_id": row["metadata_grounding_id"],
                "metadata_grounding_ref_id": _metadata_grounding_ref_id(row["metadata_grounding_id"], "block", bbox_row["bbox_id"], index),
                "page_number": bbox_row["page_number"],
                "quoted_text": row["quoted_text"],
                "source_document_id": row["source_document_id"],
                "source_pdf_path": row["source_pdf_path"],
                "source_sha256": row["source_sha256"],
                "status": row["status"],
                "text_span_ids": row.get("text_span_ids", []),
                "viewer_highlightable": True,
            }
            for index, bbox_row in enumerate(exact_bbox_rows)
        ]
    bbox_id = row["bbox_refs"][0]
    return [
        {
            "bbox_id": bbox_id,
            "bbox_precision": "page_grounded_only",
            "corpus_id": row["corpus_id"],
            "failure_reason": row["failure_reason"],
            "metadata_grounding_id": row["metadata_grounding_id"],
            "metadata_grounding_ref_id": _metadata_grounding_ref_id(row["metadata_grounding_id"], "block", bbox_id, 0),
            "page_number": row["page_numbers"][0],
            "quoted_text": row["quoted_text"],
            "source_document_id": row["source_document_id"],
            "source_pdf_path": row["source_pdf_path"],
            "source_sha256": row["source_sha256"],
            "status": row["status"],
            "viewer_highlightable": False,
        }
    ]


def _metadata_failure_reason(source_role: str, metadata_field: str, donor_id: str) -> str:
    if metadata_field in {"decision_date", "decision_session", "effective_rule"}:
        return "metadata_decision_sentence_continues_beyond_field"
    if source_role == "current_consolidated":
        return "metadata_publication_block_requires_page_level_support"
    if str(donor_id).startswith("uud_metadata_block_final_evidence::"):
        return "metadata_block_has_no_exact_bbox_span"
    return "metadata_bbox_reference_unresolved"


def _metadata_grounding_ref_id(metadata_grounding_id: str, metadata_field: str, bbox_id: str, index: int) -> str:
    return f"metadata_grounding_ref::{metadata_grounding_id}::{metadata_field}::{bbox_id}::{index:04d}"


def _allow_global_exact_bbox_reuse(source_role: str, metadata_field: str) -> bool:
    if source_role == "current_consolidated" and metadata_field in {"institution", "official_title", "source_publication"}:
        return False
    return True


def _allow_word_exact_bbox_promotion(source_role: str, metadata_field: str) -> bool:
    return True


def _evidence_metadata_assertions(row: dict, bbox_by_id: dict[str, dict]) -> list[dict]:
    evidence_link = _evidence_link(row, bbox_by_id)
    return [
        {
            "corpus_id": "uud",
            "evidence_link": evidence_link,
            "metadata_id": f"uud_metadata_assertion::{row['evidence_id']}::source_role",
            "predicate": "source_role",
            "source_role": row["source_role"],
            "status": "accepted",
            "subject_id": f"source_role::{row['source_role']}",
            "temporal_context": row["temporal_context"],
            "type": "HAS_METADATA",
            "validator_status": "valid",
            "value": row["source_role"],
        },
        {
            "corpus_id": "uud",
            "evidence_link": evidence_link,
            "metadata_id": f"uud_metadata_assertion::{row['evidence_id']}::temporal_context",
            "predicate": "temporal_context",
            "source_role": row["source_role"],
            "status": "accepted",
            "subject_id": f"source_role::{row['source_role']}",
            "temporal_context": row["temporal_context"],
            "type": "HAS_METADATA",
            "validator_status": "valid",
            "value": row["temporal_context"],
        },
        {
            "corpus_id": "uud",
            "evidence_link": evidence_link,
            "metadata_id": f"uud_metadata_assertion::{row['evidence_id']}::legal_unit_identity",
            "predicate": "legal_unit_identity",
            "source_role": row["source_role"],
            "status": "accepted",
            "subject_id": row["legal_unit_id"],
            "temporal_context": row["temporal_context"],
            "type": "HAS_METADATA",
            "validator_status": "valid",
            "value": row["citation"],
        },
    ]


def _block_metadata_assertion(row: dict) -> dict:
    predicate = (
        "source_publication_metadata_block" if row["source_role"] == "current_consolidated" else "instrument_closing_and_issuance_block"
    )
    return {
        "corpus_id": "uud",
        "evidence_link": {
            "bbox_refs": row["bbox_refs"],
            "final_evidence_id": row["metadata_grounding_id"],
            "link_target_type": "metadata_grounding",
            "metadata_grounding_id": row["metadata_grounding_id"],
            "page_number": row["page_numbers"][0],
            "quoted_text": row["quoted_text"],
            "source_pdf_sha256": row["source_sha256"],
        },
        "metadata_id": f"uud_metadata_assertion::{row['source_role']}::{predicate}::page_{row['page_numbers'][0]:04d}",
        "predicate": predicate,
        "source_role": row["source_role"],
        "status": "accepted",
        "subject_id": f"source_document::{row['source_role']}",
        "temporal_context": row["temporal_context"],
        "type": "HAS_METADATA",
        "validator_status": "valid",
        "value": row["quoted_text"],
    }


def _evidence_link(row: dict, bbox_by_id: dict[str, dict]) -> dict:
    bbox_ref = next((ref for ref in row["bbox_refs"] if ref.endswith("::0000")), row["bbox_refs"][0])
    bbox = bbox_by_id.get(bbox_ref, {})
    return {
        "bbox_refs": [bbox_ref],
        "final_evidence_id": row["evidence_id"],
        "page_number": bbox.get("page_number", row["page_numbers"][0]),
        "quoted_text": _first_quoted_line(row["quoted_text"]),
        "source_pdf_sha256": row["source_sha256"],
    }


def _first_quoted_line(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text.splitlines()[0])).strip()


def _metadata_quote(
    spec: dict,
    pages_by_source: dict[tuple[str, int], str],
    source_document_id: str,
) -> str:
    if "title_text" in spec:
        return f"{spec['institution']} {spec['title_text']}"
    lines = pages_by_source[(source_document_id, spec["page_number"])].splitlines()
    start = next(index for index, line in enumerate(lines) if spec["start_marker"] in line)
    return " ".join(line.strip() for line in lines[start:] if line.strip())


def _amendment_document_metadata(source: dict, block: dict) -> dict:
    date = _extract_metadata_date(block["quoted_text"])
    grounding_ref = block["metadata_grounding_id"]
    return {
        "corpus_id": "uud",
        "date": date,
        "document_metadata_id": f"uud_document_metadata::{source['source_role']}",
        "document_type": "uud_amendment_document",
        "enactment": None,
        "evidence_refs": [grounding_ref],
        "field_statuses": dict(AMENDMENT_FIELD_STATUSES),
        "grounded_fields": {},
        "grounding_refs": [],
        "institution": "MAJELIS PERMUSYAWARATAN RAKYAT REPUBLIK INDONESIA",
        "ln_tln": None,
        "official_title": None,
        "officials": [],
        "penetapan": {
            "date_text": date,
            "grounding_refs": [grounding_ref],
            "institution": "MAJELIS PERMUSYAWARATAN RAKYAT REPUBLIK INDONESIA",
            "place": "Jakarta",
        },
        "pengesahan": None,
        "pengundangan": None,
        "place": "Jakarta",
        "promulgation": None,
        "runtime_loadable": False,
        "signatories": _signatories(block["quoted_text"]),
        "source_document_id": source["source_document_id"],
        "source_publication": None,
        "source_role": source["source_role"],
        "status": "evidence_grounded_partial_metadata",
        "temporal_context": source["temporal_context"],
    }


def _current_document_metadata(source: dict, block: dict) -> dict:
    title = "UNDANG\xadUNDANG DASAR NEGARA REPUBLIK INDONESIA TAHUN 1945 DALAM SATU NASKAH"
    institution = block["quoted_text"].removesuffix(f" {title}")
    return {
        "corpus_id": "uud",
        "date": None,
        "document_metadata_id": "uud_document_metadata::current_consolidated",
        "document_type": "uud_consolidated_source_publication",
        "enactment": None,
        "evidence_refs": [block["metadata_grounding_id"]],
        "field_statuses": {
            "date": "not_found_in_source",
            "institution": "grounded",
            "ln_tln": "not_found_in_source",
            "official_title": "grounded",
            "penetapan": "not_applicable",
            "pengundangan": "not_found_in_source",
            "place": "not_found_in_source",
            "promulgation": "not_found_in_source",
            "signatories": "not_applicable",
            "source_publication": "grounded",
            "decision_date": "not_found_in_source",
            "decision_session": "not_found_in_source",
            "effective_rule": "not_found_in_source",
            "effective_date": "not_found_in_source",
            "promulgation_date": "not_found_in_source",
            "revocation_date": "not_found_in_source",
            "source_anomaly_status": "not_found_in_source",
        },
        "grounded_fields": {},
        "grounding_refs": [],
        "institution": institution,
        "ln_tln": None,
        "official_title": title,
        "officials": [],
        "penetapan": None,
        "pengesahan": None,
        "pengundangan": None,
        "place": None,
        "promulgation": None,
        "runtime_loadable": False,
        "signatories": [],
        "source_document_id": source["source_document_id"],
        "source_publication": {
            "grounding_refs": [block["metadata_grounding_id"]],
            "institution": institution,
            "title_text": title,
        },
        "source_role": source["source_role"],
        "status": "evidence_grounded_partial_metadata",
        "temporal_context": source["temporal_context"],
    }


def _original_document_metadata(source: dict) -> dict:
    return {
        "corpus_id": "uud",
        "date": None,
        "document_metadata_id": "uud_document_metadata::original_historical",
        "document_type": "uud_source_document",
        "enactment": None,
        "evidence_refs": [],
        "field_statuses": {
            "metadata": "not_found_in_source",
            "decision_date": "not_found_in_source",
            "decision_session": "not_found_in_source",
            "effective_rule": "not_found_in_source",
            "effective_date": "not_found_in_source",
            "promulgation_date": "not_found_in_source",
            "revocation_date": "not_found_in_source",
            "source_anomaly_status": "not_found_in_source",
            "source_publication": "not_found_in_source",
        },
        "grounded_fields": {},
        "grounding_refs": [],
        "institution": None,
        "ln_tln": None,
        "official_title": None,
        "officials": [],
        "penetapan": None,
        "pengesahan": None,
        "pengundangan": None,
        "place": None,
        "promulgation": None,
        "runtime_loadable": False,
        "signatories": [],
        "source_document_id": source["source_document_id"],
        "source_publication": None,
        "source_role": source["source_role"],
        "status": "not_found_in_source",
        "temporal_context": source["temporal_context"],
    }


def _signatories(text: str) -> list[dict]:
    parts = [part.strip() for part in text.split()]
    rows = []
    role = None
    name_parts: list[str] = []
    index = 0
    while index < len(parts):
        token = parts[index]
        spaced_chair = parts[index : index + 5] == ["K", "e", "t", "u", "a,"]
        marker = (
            token == CHAIR_TOKEN
            or spaced_chair
            or (token == VICE_CHAIR_TOKEN and index + 1 < len(parts) and parts[index + 1] == CHAIR_TOKEN)
        )
        if marker:
            if role and name_parts:
                rows.append({"name_text": " ".join(name_parts), "role_text": role})
            role = VICE_CHAIR_ROLE if token == VICE_CHAIR_TOKEN else CHAIR_ROLE
            name_parts = []
            index += 2 if token == VICE_CHAIR_TOKEN else 5 if spaced_chair else 1
            continue
        if role:
            name_parts.append(token)
        index += 1
    if role and name_parts:
        rows.append({"name_text": " ".join(name_parts), "role_text": role})
    return rows


def _exact_bbox_rows(quoted_text: str, bbox_rows: list[dict]) -> list[dict]:
    if not bbox_rows or any(row.get("bbox_precision") != "exact" for row in bbox_rows):
        return []
    matched_sequence = matching_sequence(bbox_rows, quoted_text)
    if matched_sequence:
        return matched_sequence
    wanted = [compact(line) for line in quoted_text.splitlines() if compact(line)]
    if not wanted:
        return []
    matched = [row for row in bbox_rows if compact(row.get("text")) in wanted]
    matched_text = {compact(row.get("text")) for row in bbox_rows if compact(row.get("text")) in wanted}
    return matched if set(wanted) <= matched_text else []


def _exact_text_span_ids(
    quoted_text: str,
    source_document_id: str,
    page_numbers: list[int] | tuple[int, ...],
    page_text_spans: list[dict],
) -> list[str]:
    wanted = [compact(line) for line in quoted_text.splitlines() if compact(line)]
    if not wanted:
        return []
    pages = set(page_numbers)
    rows = [row for row in page_text_spans if row["source_document_id"] == source_document_id and row["page_number"] in pages]
    matched_sequence = matching_sequence(rows, quoted_text)
    if matched_sequence:
        return [row["text_span_id"] for row in matched_sequence]
    matched = [row["text_span_id"] for row in rows if compact(row.get("text")) in wanted]
    matched_text = {compact(row.get("text")) for row in rows if compact(row.get("text")) in wanted}
    return matched if set(wanted) <= matched_text else []


def _supporting_text_span_ids(
    quoted_text: str,
    source_document_id: str,
    page_numbers: list[int] | tuple[int, ...],
    page_text_spans: list[dict],
) -> list[str]:
    target = compact(quoted_text)
    if not target:
        return []
    rows = [row for row in page_text_spans if row["source_document_id"] == source_document_id and row["page_number"] in set(page_numbers)]
    direct = [row["text_span_id"] for row in rows if target and target in compact(row.get("text"))]
    if direct:
        return direct
    for start in range(len(rows)):
        joined = ""
        matched: list[str] = []
        for row in rows[start:]:
            matched.append(row["text_span_id"])
            joined = compact(f"{joined} {row.get('text')}")
            if target in joined:
                return matched
            if len(joined) > len(target) + 120 and target not in joined:
                break
    return []


def _word_exact_bbox_rows(
    *,
    quoted_text: str,
    source_document_id: str,
    page_numbers: list[int] | tuple[int, ...],
    page_text_spans: list[dict],
    words_by_page: dict[tuple[str, int], list[dict]],
) -> list[dict]:
    reference_span = next(
        (
            row
            for row in page_text_spans
            if row["source_document_id"] == source_document_id
            and row["page_number"] in set(page_numbers)
            and compact(quoted_text) in compact(row.get("text"))
        ),
        None,
    )
    match = align_text_to_word_bboxes(
        text=quoted_text,
        source_document_id=source_document_id,
        page_numbers=list(page_numbers),
        words_by_page=words_by_page,
        reference_bbox=reference_span,
    )
    if not match:
        return []
    bbox_rows: list[dict] = []
    lookup = {row["word_bbox_id"]: row for rows in words_by_page.values() for row in rows}
    for bbox_id in match["matched_word_bbox_ids"]:
        row = lookup[bbox_id]
        bbox_rows.append(
            {
                "bbox_id": row["word_bbox_id"],
                "bbox_precision": "exact",
                "viewer_highlightable": True,
                "page_number": row["page_number"],
                "source_document_id": row["source_document_id"],
                "source_pdf": row["source_pdf"],
                "source_pdf_path": row["source_pdf_path"],
                "source_sha256": row["source_sha256"],
                "page_width": row.get("page_width"),
                "page_height": row.get("page_height"),
                "text": row["text"],
                "x0": row["x0"],
                "y0": row["y0"],
                "x1": row["x1"],
                "y1": row["y1"],
            }
        )
    return bbox_rows


def _ordinal_label(source_role: str) -> str:
    return {
        "amendment_1_historical": "Pertama",
        "amendment_2_historical": "Kedua",
        "amendment_3_historical": "Ketiga",
        "amendment_4_historical": "Keempat",
    }[source_role]


def _extract_metadata_date(text: str) -> str | None:
    match = re.search(r"\b\d{1,2}\s+[A-Z][a-z]+\s+\d{4}\b", text)
    return match.group(0) if match else None


def _extract_decision_session(text: str, decision_date: str | None) -> str | None:
    compact = " ".join(text.split())
    compact = compact.split(", dan mulai berlaku", 1)[0].strip()
    if decision_date:
        compact = compact.replace(f" tanggal {decision_date}", "")
    if compact.startswith("Perubahan tersebut diputuskan dalam "):
        compact = compact.removeprefix("Perubahan tersebut diputuskan dalam ").strip()
    if compact.endswith("."):
        compact = compact[:-1]
    return compact or None
