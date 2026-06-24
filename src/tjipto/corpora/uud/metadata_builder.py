from __future__ import annotations

from collections import defaultdict
import re


def rebuild_metadata_grounding(
    *,
    document_metadata: list[dict],
    metadata_grounding: list[dict],
    metadata_grounding_registry: list[dict],
    evidence: list[dict],
    legal_units: list[dict],
    source_conflicts: list[dict],
) -> tuple[list[dict], list[dict], list[dict]]:
    evidence_by_key = {
        (row.get("source_role"), row.get("citation")): row
        for row in evidence
    }
    units_by_key = {
        (row["source_document_id"].split("::", 1)[1], row.get("unit_label")): row
        for row in legal_units
    }
    block_rows = [
        row | {
            "bbox_precision": "page_grounded_only",
            "grounding_status": row.get("grounding_status") or "block_level_grounded",
            "viewer_highlightable": False,
        }
        for row in metadata_grounding
        if not str(row.get("metadata_grounding_id", "")).startswith("uud_metadata_field_grounding::")
    ]
    block_ids = {row["metadata_grounding_id"] for row in block_rows}
    block_registry_rows = [
        row
        for row in metadata_grounding_registry
        if row.get("metadata_grounding_id") in block_ids
    ]
    source_conflicts_by_role: dict[str, list[dict]] = defaultdict(list)
    for row in source_conflicts:
        source_conflicts_by_role[row["source_document_id"].split("::", 1)[1]].append(row)
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
        bbox_refs = []
        for index, page_number in enumerate(page_numbers):
            bbox_id = f"uud_metadata_field_bbox::{source_role}::{metadata_field}::{index:04d}"
            bbox_refs.append(bbox_id)
            field_registry_rows.append({
                "bbox_id": bbox_id,
                "corpus_id": "uud",
                "metadata_grounding_id": field_id,
                "metadata_field": metadata_field,
                "page_number": page_number,
                "quoted_text": quoted_text,
                "source_document_id": source_document_id,
                "source_pdf_path": source_pdf_path,
                "source_sha256": source_sha256,
                "status": "accepted_metadata_grounding",
            })
        field_rows.append({
            "bbox_precision": "page_grounded_only",
            "bbox_refs": bbox_refs,
            "corpus_id": "uud",
            "grounding_status": "field_level_grounded",
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
            "viewer_highlightable": False,
        })
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
                    date_quote = next((line for line in determination["quoted_text"].splitlines() if date_text in line), determination["quoted_text"])
                    grounded_fields["date"] = [append_grounding(
                        source_role=role,
                        source_document_id=source_document_id,
                        metadata_field="date",
                        quoted_text=date_quote,
                        donor_id=determination["evidence_id"],
                        page_numbers=determination["page_numbers"],
                        source_pdf_path=determination["source_pdf_path"],
                        source_sha256=determination["source_sha256"],
                    )]
                    field_statuses["date"] = "grounded"
                place_text = row.get("place")
                if place_text:
                    place_quote = next((line for line in determination["quoted_text"].splitlines() if place_text in line), determination["quoted_text"])
                    grounded_fields["place"] = [append_grounding(
                        source_role=role,
                        source_document_id=source_document_id,
                        metadata_field="place",
                        quoted_text=place_quote,
                        donor_id=determination["evidence_id"],
                        page_numbers=determination["page_numbers"],
                        source_pdf_path=determination["source_pdf_path"],
                        source_sha256=determination["source_sha256"],
                    )]
                    field_statuses["place"] = "grounded"
            if signatories:
                institution_quote = "\n".join(signatories["quoted_text"].splitlines()[:2]).strip()
                grounded_fields["institution"] = [append_grounding(
                    source_role=role,
                    source_document_id=source_document_id,
                    metadata_field="institution",
                    quoted_text=institution_quote,
                    donor_id=signatories["evidence_id"],
                    page_numbers=signatories["page_numbers"],
                    source_pdf_path=signatories["source_pdf_path"],
                    source_sha256=signatories["source_sha256"],
                )]
                grounded_fields["signatories"] = [append_grounding(
                    source_role=role,
                    source_document_id=source_document_id,
                    metadata_field="signatories",
                    quoted_text=signatories["quoted_text"],
                    donor_id=signatories["evidence_id"],
                    page_numbers=signatories["page_numbers"],
                    source_pdf_path=signatories["source_pdf_path"],
                    source_sha256=signatories["source_sha256"],
                )]
                field_statuses["institution"] = "grounded"
                field_statuses["signatories"] = "grounded"
            if decision:
                decision_date = _extract_metadata_date(decision["quoted_text"])
                decision_session = _extract_decision_session(decision["quoted_text"], decision_date)
                if decision_date:
                    row["decision_date"] = decision_date
                    grounded_fields["decision_date"] = [append_grounding(
                        source_role=role,
                        source_document_id=source_document_id,
                        metadata_field="decision_date",
                        quoted_text=decision["quoted_text"],
                        donor_id=decision["evidence_id"],
                        page_numbers=decision["page_numbers"],
                        source_pdf_path=decision["source_pdf_path"],
                        source_sha256=decision["source_sha256"],
                    )]
                    field_statuses["decision_date"] = "grounded"
                if decision_session:
                    row["decision_session"] = decision_session
                    grounded_fields["decision_session"] = [append_grounding(
                        source_role=role,
                        source_document_id=source_document_id,
                        metadata_field="decision_session",
                        quoted_text=decision["quoted_text"],
                        donor_id=decision["evidence_id"],
                        page_numbers=decision["page_numbers"],
                        source_pdf_path=decision["source_pdf_path"],
                        source_sha256=decision["source_sha256"],
                    )]
                    field_statuses["decision_session"] = "grounded"
            if effective:
                row["effective_rule"] = effective["text"]
                grounded_fields["effective_rule"] = [append_grounding(
                    source_role=role,
                    source_document_id=source_document_id,
                    metadata_field="effective_rule",
                    quoted_text=effective["text"],
                    donor_id=effective["legal_unit_id"],
                    page_numbers=list(range(effective["page_start"], effective["page_end"] + 1)),
                    source_pdf_path=next(item["source_pdf_path"] for item in evidence if item["source_document_id"] == source_document_id),
                    source_sha256=effective["source_sha256"],
                )]
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
                grounded_fields["institution"] = [append_grounding(
                    source_role=role,
                    source_document_id=source_document_id,
                    metadata_field="institution",
                    quoted_text=row["institution"],
                    donor_id=block["metadata_grounding_id"] if block else row["document_metadata_id"],
                    page_numbers=page_numbers,
                    source_pdf_path=source_pdf_path,
                    source_sha256=source_sha256,
                )]
                field_statuses["institution"] = "grounded"
            if row.get("official_title"):
                grounded_fields["official_title"] = [append_grounding(
                    source_role=role,
                    source_document_id=source_document_id,
                    metadata_field="official_title",
                    quoted_text=row["official_title"],
                    donor_id=block["metadata_grounding_id"] if block else row["document_metadata_id"],
                    page_numbers=page_numbers,
                    source_pdf_path=source_pdf_path,
                    source_sha256=source_sha256,
                )]
                field_statuses["official_title"] = "grounded"
            if row.get("source_publication"):
                grounded_fields["source_publication"] = [append_grounding(
                    source_role=role,
                    source_document_id=source_document_id,
                    metadata_field="source_publication",
                    quoted_text=block["quoted_text"] if block else row["official_title"],
                    donor_id=block["metadata_grounding_id"] if block else row["document_metadata_id"],
                    page_numbers=page_numbers,
                    source_pdf_path=source_pdf_path,
                    source_sha256=source_sha256,
                )]
                field_statuses["source_publication"] = "grounded"
        row["field_statuses"] = field_statuses
        row["grounded_fields"] = {key: tuple(value) for key, value in grounded_fields.items() if value}
        row["grounding_refs"] = tuple(dict.fromkeys(ref for refs in row["grounded_fields"].values() for ref in refs))
    all_grounding_rows = block_rows + field_rows
    all_registry_rows = block_registry_rows + field_registry_rows
    return document_metadata, all_grounding_rows, all_registry_rows


def repair_metadata_graph_edges(edges: list[dict], metadata_assertions: list[dict]) -> list[dict]:
    metadata_id_by_key = {
        (row["evidence_link"]["final_evidence_id"], row["predicate"]): row["metadata_id"]
        for row in metadata_assertions
    }
    for edge in edges:
        evidence_id = edge.get("evidence_link", {}).get("final_evidence_id")
        predicate = str(edge.get("target_id") or "").rsplit("::", 1)[-1]
        metadata_id = metadata_id_by_key.get((evidence_id, predicate))
        if metadata_id:
            edge["target_id"] = metadata_id
    return edges


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
    if decision_date:
        compact = compact.replace(f" tanggal {decision_date}", "")
    if compact.startswith("Perubahan tersebut diputuskan dalam "):
        compact = compact.removeprefix("Perubahan tersebut diputuskan dalam ").strip()
    if compact.endswith("."):
        compact = compact[:-1]
    return compact or None
