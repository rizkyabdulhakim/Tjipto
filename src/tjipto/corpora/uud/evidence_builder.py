from __future__ import annotations

from collections.abc import Callable

from tjipto.corpora.uud.bbox_builder import aggregate_bbox_precision, build_bbox_rows
from tjipto.corpora.uud.retrieval_builder import retrieval_text
from tjipto.corpora.uud.structure_builder import slug


def append_instrument_unit(
    *,
    source_id: str,
    unit_type: str,
    unit_label: str,
    text: str,
    page_start: int,
    page_end: int,
    source_documents: dict[str, dict],
    pdf_lines_by_source: dict[str, dict[int, list[dict]]],
    legal_units: list[dict],
    chunks: list[dict],
    evidence: list[dict],
    bbox_rows: list[dict],
    bbox_by_evidence: dict[str, list[dict]],
    retrieval_units: list[dict],
    allocate_legal_id: Callable[[], str],
    allocate_chunk_id: Callable[[], str],
    allocate_evidence_id: Callable[[str, str], str],
    hierarchy: list[str] | None = None,
    parent_legal_unit_ids: list[str] | None = None,
    chunk_type: str | None = None,
    canonical_use_allowed: bool = True,
    chunk_status: str = "active_canonical_record",
    runtime_loadable: bool | None = None,
    exclusion_ref: str | None = None,
    build_evidence: bool = True,
) -> str:
    existing = next(
        (
            row
            for row in legal_units
            if row["source_document_id"] == source_id
            and row.get("unit_type") == unit_type
            and row.get("unit_label") == unit_label
            and row.get("hierarchy") == (hierarchy or [])
        ),
        None,
    )
    if existing:
        return existing["legal_unit_id"]
    legal_unit_id = allocate_legal_id()
    chunk_id = allocate_chunk_id()
    source_meta = source_documents[source_id]
    source_role = source_meta["source_role"]
    temporal_context = source_meta.get("temporal_context", source_role)
    unit = {
        "corpus_id": "uud",
        "canonical_use_allowed": canonical_use_allowed,
        "hierarchy": hierarchy or [],
        "legal_unit_id": legal_unit_id,
        "page_end": page_end,
        "page_start": page_start,
        "parent_legal_unit_ids": parent_legal_unit_ids or [],
        "provenance": {"donor_id": legal_unit_id},
        "source_document_id": source_id,
        "source_sha256": source_meta["sha256"],
        "status": chunk_status if runtime_loadable is False else "finalizable",
        "text": text,
        "unit_label": unit_label,
        "unit_type": unit_type,
    }
    if runtime_loadable is False:
        unit["runtime_loadable"] = False
    if exclusion_ref:
        unit["exclusion_ref"] = exclusion_ref
    legal_units.append(unit)
    chunk = {
        "canonical_use_allowed": canonical_use_allowed,
        "chunk_id": chunk_id,
        "chunk_type": chunk_type or f"{unit_type.replace('_record', '')}_chunk_record",
        "corpus_id": "uud",
        "hierarchy": hierarchy or ([unit_label] if unit_label else []),
        "legal_unit_id": legal_unit_id,
        "page_range": {"start_page_number": page_start, "end_page_number": page_end},
        "provenance": {"donor_id": chunk_id},
        "source_sha256": source_meta["sha256"],
        "status": chunk_status,
        "text": text,
    }
    if runtime_loadable is False:
        chunk["runtime_loadable"] = False
    if exclusion_ref:
        chunk["exclusion_ref"] = exclusion_ref
    chunks.append(chunk)
    if not build_evidence:
        return legal_unit_id
    evidence_id = allocate_evidence_id(source_role, slug(unit_label or unit_type))
    bbox_records = build_bbox_rows(
        evidence_id=evidence_id,
        source_meta=source_meta,
        source_id=source_id,
        text=text,
        page_start=page_start,
        page_end=page_end,
        line_entries=pdf_lines_by_source[source_id],
    )
    quoted_text = "\n".join(row["text"] for row in bbox_records)
    evidence_row = {
        "bbox_refs": [row["bbox_id"] for row in bbox_records],
        "bbox_precision": aggregate_bbox_precision(bbox_records),
        "citation": unit_label,
        "corpus_id": "uud",
        "evidence_id": evidence_id,
        "hierarchy": hierarchy or ([unit_label] if unit_label else []),
        "legal_unit_id": legal_unit_id,
        "page_numbers": sorted({row["page_number"] for row in bbox_records}),
        "quoted_text": quoted_text,
        "source_document_id": source_id,
        "source_url": source_meta["source_page_url"],
        "source_pdf": source_meta["filename"],
        "source_pdf_path": source_meta["path"],
        "source_role": source_role,
        "source_sha256": source_meta["sha256"],
        "status": "final",
        "temporal_context": temporal_context,
        "runtime_loadable": runtime_loadable is not False,
        "evidence_owner_kind": "legal_unit_source",
        "viewer_highlightable": any(row["viewer_highlightable"] for row in bbox_records),
    }
    evidence.append(evidence_row)
    bbox_rows.extend(bbox_records)
    bbox_by_evidence[evidence_id] = bbox_records
    retrieval_units.append(
        {
            "bbox_sample_refs": [bbox_records[0]["bbox_id"]] if bbox_records else [],
            "bbox_total_count": len(bbox_records),
            "chunk_id": chunk_id,
            "corpus_id": "uud",
            "evidence_id": evidence_id,
            "legal_unit_id": legal_unit_id,
            "page_numbers": evidence_row["page_numbers"],
            "retrieval_unit_id": f"uud_retrieval_unit::{evidence_id}",
            "source_pdf_path": source_meta["path"],
            "source_role": source_role,
            "source_sha256": source_meta["sha256"],
            "status": "accepted",
            "temporal_context": temporal_context,
            "text": retrieval_text(unit_label, hierarchy or [], quoted_text),
        }
    )
    return legal_unit_id


def rebuild_evidence(
    existing: dict,
    text: str,
    line_entries: dict[int, list[dict]],
    source_meta: dict,
    bbox_by_evidence: dict[str, list[dict]],
) -> None:
    bbox_records = build_bbox_rows(
        evidence_id=existing["evidence_id"],
        source_meta=source_meta,
        source_id=existing["source_document_id"],
        text=text,
        page_start=min(existing["page_numbers"]),
        page_end=max(existing["page_numbers"]),
        line_entries=line_entries,
    )
    existing["quoted_text"] = "\n".join(row["text"] for row in bbox_records)
    existing["page_numbers"] = sorted({row["page_number"] for row in bbox_records})
    existing["bbox_refs"] = [row["bbox_id"] for row in bbox_records]
    existing["bbox_precision"] = aggregate_bbox_precision(bbox_records)
    existing["viewer_highlightable"] = any(row["viewer_highlightable"] for row in bbox_records)
    bbox_by_evidence[existing["evidence_id"]] = bbox_records
