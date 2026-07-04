from __future__ import annotations

import re

from tjipto.corpora.uud.bbox_builder import aggregate_bbox_precision, apply_inserted_bab_heading_bbox_policy, build_bbox_rows
from tjipto.corpora.uud.specs import INSERTED_BAB_SPECS
from tjipto.corpora.uud.structure_builder import compact, slug


def build_evidence_and_bboxes(
    *,
    legal_units: list[dict],
    chunks: list[dict],
    source_documents: dict[str, dict],
    pdf_lines_by_source: dict[str, dict[int, list[dict]]],
) -> tuple[list[dict], list[dict]]:
    units_by_id = {row["legal_unit_id"]: row for row in legal_units}
    evidence: list[dict] = []
    bbox_rows: list[dict] = []
    next_instrument_id = 1
    for chunk in sorted(chunks, key=lambda row: row["chunk_id"]):
        if chunk["status"] != "active_canonical_record":
            continue
        unit = units_by_id[chunk["legal_unit_id"]]
        if unit["unit_type"] == "effective_clause_record":
            continue
        source_id = unit["source_document_id"]
        source_meta = source_documents[source_id]
        source_role = source_meta["source_role"]
        evidence_id, next_instrument_id = _evidence_id(chunk, unit, source_role, next_instrument_id)
        bbox_records = build_bbox_rows(
            evidence_id=evidence_id,
            source_meta=source_meta,
            source_id=source_id,
            text=_bbox_text(chunk["text"], unit),
            page_start=chunk["page_range"]["start_page_number"],
            page_end=chunk["page_range"]["end_page_number"],
            line_entries=pdf_lines_by_source[source_id],
        )
        quoted_text = "\n".join(row["text"] for row in bbox_records)
        evidence.append(
            {
                "bbox_refs": [row["bbox_id"] for row in bbox_records],
                "bbox_precision": aggregate_bbox_precision(bbox_records),
                "citation": unit["unit_label"],
                "corpus_id": "uud",
                "evidence_id": evidence_id,
                "hierarchy": chunk["hierarchy"],
                "legal_unit_id": unit["legal_unit_id"],
                "page_numbers": sorted({row["page_number"] for row in bbox_records}),
                "quoted_text": quoted_text,
                "source_document_id": source_id,
                "source_pdf": source_meta["filename"],
                "source_pdf_path": source_meta["path"],
                "source_role": source_role,
                "source_sha256": source_meta["sha256"],
                "status": "final",
                "temporal_context": source_meta.get("temporal_context", source_role),
                "viewer_highlightable": any(row["viewer_highlightable"] for row in bbox_records),
            }
        )
        bbox_rows.extend(bbox_records)
    _append_inserted_bab_bbox_refs(
        evidence=evidence,
        bbox_rows=bbox_rows,
        chunks=chunks,
        legal_units=legal_units,
        source_documents=source_documents,
        pdf_lines_by_source=pdf_lines_by_source,
    )
    apply_inserted_bab_heading_bbox_policy(bbox_rows, evidence)
    evidence.sort(key=lambda row: row["evidence_id"])
    bbox_rows.sort(key=lambda row: (row["source_document_id"], row["page_number"], row["bbox_id"]))
    return evidence, bbox_rows


def _evidence_id(chunk: dict, unit: dict, source_role: str, next_instrument_id: int) -> tuple[str, int]:
    if unit["unit_type"] in {
        "amendment_recital_record",
        "amendment_scope_record",
        "instrument_clause_record",
        "instrument_closing_record",
        "decision_clause_record",
        "determination_clause_record",
        "signatory_block_record",
    }:
        return (
            f"uud_instrument_final_citation_evidence::{source_role}::{next_instrument_id:05d}::{slug(unit['unit_label'])}",
            next_instrument_id + 1,
        )
    chunk_number = int(chunk["chunk_id"].rsplit("_", 1)[1])
    prefix = _evidence_prefix(source_role, chunk_number)
    return f"{prefix}_{chunk_number:05d}", next_instrument_id


def _bbox_text(text: str, unit: dict) -> str:
    if unit["source_document_id"] != "uud::current_consolidated":
        return text
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "_________" or re.fullmatch(r"\*+\)", stripped):
            continue
        lines.append(line)
    return "\n".join(lines)


def _append_inserted_bab_bbox_refs(
    *,
    evidence: list[dict],
    bbox_rows: list[dict],
    chunks: list[dict],
    legal_units: list[dict],
    source_documents: dict[str, dict],
    pdf_lines_by_source: dict[str, dict[int, list[dict]]],
) -> None:
    chunks_by_unit = {row["legal_unit_id"]: row for row in chunks}
    units_by_id = {row["legal_unit_id"]: row for row in legal_units}
    source_evidence = sorted(
        evidence,
        key=lambda row: (row["source_document_id"], chunks_by_unit[row["legal_unit_id"]]["chunk_id"]),
    )
    for spec in INSERTED_BAB_SPECS:
        source_id = spec["source_document_id"]
        if source_id == "uud::current_consolidated":
            continue
        child = _first_evidence_at_or_after_child(spec, source_evidence, chunks_by_unit, units_by_id)
        if not child:
            continue
        previous = _previous_source_evidence(child, source_evidence)
        if previous:
            title_row = _inserted_bab_row(
                evidence_id=previous["evidence_id"],
                bbox_id=f"uud_unified_bbox::{previous['evidence_id']}::{len(previous['bbox_refs']) + 1:04d}",
                text=spec["title"],
                page_number=spec["page_number"],
                source_id=source_id,
                source_meta=source_documents[source_id],
                pdf_lines_by_source=pdf_lines_by_source,
                viewer_highlightable=True,
            )
            bbox_rows.append(title_row)
            previous["bbox_refs"].append(title_row["bbox_id"])
        heading_row = _inserted_bab_row(
            evidence_id=child["evidence_id"],
            bbox_id=f"uud_unified_bbox::{child['evidence_id']}::heading_{slug(spec['label'])}",
            text=spec["label"],
            page_number=spec["page_number"],
            source_id=source_id,
            source_meta=source_documents[source_id],
            pdf_lines_by_source=pdf_lines_by_source,
            viewer_highlightable=False,
        )
        bbox_rows.append(heading_row)
        child["bbox_refs"] = [heading_row["bbox_id"], *child["bbox_refs"]]
    by_evidence = {row["evidence_id"]: [] for row in evidence}
    for row in bbox_rows:
        by_evidence.setdefault(row["evidence_id"], []).append(row)
    for row in evidence:
        rows = by_evidence.get(row["evidence_id"], [])
        row["bbox_precision"] = aggregate_bbox_precision(rows)
        row["viewer_highlightable"] = any(item["viewer_highlightable"] for item in rows)
        row["page_numbers"] = sorted({item["page_number"] for item in rows})


def _first_evidence_at_or_after_child(
    spec: dict, source_evidence: list[dict], chunks_by_unit: dict[str, dict], units_by_id: dict[str, dict]
) -> dict | None:
    child_ids = [
        row["legal_unit_id"]
        for row in units_by_id.values()
        if row["source_document_id"] == spec["source_document_id"] and row.get("unit_label") == spec["child_labels"][0]
    ]
    if not child_ids:
        return None
    child_chunk = min(chunks_by_unit[unit_id]["chunk_id"] for unit_id in child_ids if unit_id in chunks_by_unit)
    return next(
        (
            row
            for row in source_evidence
            if row["source_document_id"] == spec["source_document_id"] and chunks_by_unit[row["legal_unit_id"]]["chunk_id"] >= child_chunk
        ),
        None,
    )


def _previous_source_evidence(child: dict, source_evidence: list[dict]) -> dict | None:
    previous = None
    for row in source_evidence:
        if row["source_document_id"] != child["source_document_id"]:
            continue
        if row["evidence_id"] == child["evidence_id"]:
            return previous
        previous = row
    return None


def _inserted_bab_row(
    *,
    evidence_id: str,
    bbox_id: str,
    text: str,
    page_number: int,
    source_id: str,
    source_meta: dict,
    pdf_lines_by_source: dict[str, dict[int, list[dict]]],
    viewer_highlightable: bool,
) -> dict:
    line = next(row for row in pdf_lines_by_source[source_id][page_number] if compact(row["text"]) == compact(text))
    return {
        "bbox_id": bbox_id,
        "bbox_precision": "exact",
        "corpus_id": "uud",
        "evidence_id": evidence_id,
        "page_number": page_number,
        "source_document_id": source_id,
        "source_pdf": source_meta["filename"],
        "source_pdf_path": source_meta["path"],
        "source_sha256": source_meta["sha256"],
        "status": "accepted",
        "text": line["text"],
        "viewer_highlightable": viewer_highlightable,
        "x0": line["x0"],
        "x1": line["x1"],
        "y0": line["y0"],
        "y1": line["y1"],
    }


def _evidence_prefix(source_role: str, chunk_number: int) -> str:
    if source_role == "current_consolidated":
        return "uud_current_consolidated_final_citation_evidence"
    if source_role == "original_historical" and chunk_number in {499, 522}:
        return "uud_source_role_additional_final_citation_evidence"
    if source_role in {"amendment_1_historical", "amendment_4_historical"}:
        return "uud_source_role_final_citation_evidence"
    return "uud_source_role_historical_final_citation_evidence"
