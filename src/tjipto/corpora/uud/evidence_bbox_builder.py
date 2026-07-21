from __future__ import annotations

import re

from tjipto.contracts.coordinates import coordinate_metadata
from tjipto.corpora.uud.bbox_builder import aggregate_bbox_precision, build_bbox_rows
from tjipto.ingestion.pdf.bbox import _geometry_id
from tjipto.corpora.uud.provenance_exceptions import RECOVERABLE_GROUNDING_LABELS, SEGMENTATION_BOUNDARY_LABELS
from tjipto.corpora.uud.structure_builder import compact, slug
from tjipto.ingestion.pdf.words import align_text_to_word_bboxes, compact_text, word_rows_by_page


def _admit_evidence(unit: dict, chunk: dict, *, has_descendants: bool = False) -> bool:
    if unit.get("unit_type") == "effective_clause_record":
        return False
    if unit.get("unit_type") == "bab_record" and not has_descendants:
        return "dihapus" in compact(unit.get("text")) and unit.get("source_role") in {
            "current_consolidated",
            "original_historical",
            "amendment_4_historical",
        }
    if has_descendants or unit.get("unit_type") == "pasal_record" and _has_ayat(unit.get("text")):
        return True
    return chunk.get("status") in {"active_canonical_record", "active_historical_record"}


def build_evidence_and_bboxes(
    *,
    legal_units: list[dict],
    chunks: list[dict],
    source_documents: dict[str, dict],
    pdf_lines_by_source: dict[str, dict[int, list[dict]]],
    word_bboxes: list[dict] | None = None,
) -> tuple[list[dict], list[dict]]:
    units_by_id = {row["legal_unit_id"]: row for row in legal_units}
    descendants_by_parent = _descendants_by_parent(legal_units)
    evidence: list[dict] = []
    bbox_rows: list[dict] = []
    next_instrument_id = 1
    for chunk in sorted(chunks, key=lambda row: row["chunk_id"]):
        unit = units_by_id[chunk["legal_unit_id"]]
        descendants = _descendants(unit["legal_unit_id"], descendants_by_parent)
        is_aggregate = bool(descendants)
        if not _admit_evidence(unit, chunk, has_descendants=is_aggregate):
            continue
        source_id = unit["source_document_id"]
        source_meta = source_documents[source_id]
        source_role = source_meta["source_role"]
        evidence_id, next_instrument_id = _evidence_id(chunk, unit, source_role, next_instrument_id)
        aggregate_text = _aggregate_text(unit, descendants)
        aggregate_parts = _aggregate_recovery_parts(unit, descendants)
        if is_aggregate:
            page_start = min(
                [
                    chunk["page_range"]["start_page_number"],
                    *(child.get("page_start") or chunk["page_range"]["start_page_number"] for child in descendants),
                ]
            )
            page_end = max(
                [
                    chunk["page_range"]["end_page_number"],
                    *(child.get("page_end") or chunk["page_range"]["end_page_number"] for child in descendants),
                ]
            )
            chunk["page_range"] = {"start_page_number": page_start, "end_page_number": page_end}
            unit["page_start"], unit["page_end"] = page_start, page_end
        bbox_records = build_bbox_rows(
            evidence_id=evidence_id,
            source_meta=source_meta,
            source_id=source_id,
            text=aggregate_text if is_aggregate else _bbox_text(chunk["text"], unit),
            page_start=chunk["page_range"]["start_page_number"],
            page_end=chunk["page_range"]["end_page_number"],
            line_entries=pdf_lines_by_source[source_id],
        )
        if is_aggregate:
            failure = _aggregate_failure_reason(bbox_records, source_id, aggregate_text)
            if failure:
                recovered = _recover_aggregate_word_bboxes(
                    evidence_id=evidence_id,
                    text=aggregate_text,
                    text_parts=aggregate_parts,
                    source_meta=source_meta,
                    source_id=source_id,
                    page_numbers=list(range(chunk["page_range"]["start_page_number"], chunk["page_range"]["end_page_number"] + 1)),
                    word_bboxes=word_bboxes or [],
                )
                if recovered:
                    bbox_records = recovered
                    failure = _aggregate_failure_reason(bbox_records, source_id, aggregate_text)
                if failure:
                    unit["aggregate_failure_reason"] = failure
                    chunk["aggregate_failure_reason"] = failure
                    continue
            unit["canonical_use_allowed"] = True
            chunk["canonical_use_allowed"] = True
            chunk["status"] = (
                "active_canonical_record"
                if unit.get("source_role") == "current_consolidated"
                else "active_historical_record"
            )
        recoverable = unit["unit_label"] in RECOVERABLE_GROUNDING_LABELS
        if recoverable and aggregate_bbox_precision(bbox_records) != "exact" and word_bboxes:
            recovered = _recover_word_bbox(
                evidence_id=evidence_id,
                text=chunk["text"],
                source_meta=source_meta,
                source_id=source_id,
                page_numbers=sorted({row["page_number"] for row in bbox_records}),
                word_bboxes=word_bboxes,
            )
            if recovered:
                bbox_records = recovered
        trace_only = unit["unit_label"] in SEGMENTATION_BOUNDARY_LABELS
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
                "source_url": source_meta["source_page_url"],
                "source_pdf": source_meta["filename"],
                "source_pdf_path": source_meta["path"],
                "source_role": source_role,
                "source_sha256": source_meta["sha256"],
                "status": "final",
                "temporal_context": source_meta.get("temporal_context", source_role),
                "runtime_loadable": unit.get("runtime_loadable") is not False,
                "evidence_owner_kind": "legal_unit_source",
                "viewer_highlightable": any(row["viewer_highlightable"] for row in bbox_records),
            }
            | ({"failure_reason": "instrument_trace_only_not_public_citation"} if trace_only else {})
            | ({"promotion_candidate": True} if recoverable else {})
        )
        bbox_rows.extend(bbox_records)
    for bbox in bbox_rows:
        if any(row.get("evidence_id") == bbox.get("evidence_id") and row.get("promotion_candidate") is True for row in evidence):
            bbox["promotion_candidate"] = True
        if bbox.get("bbox_precision") != "exact" or bbox.get("viewer_highlightable") is not True:
            bbox.setdefault("failure_reason", "bbox_geometry_unavailable")
    evidence.sort(key=lambda row: row["evidence_id"])
    bbox_rows = list({row["bbox_id"]: row for row in bbox_rows}.values())
    bbox_rows.sort(key=lambda row: (row["source_document_id"], row["page_number"], row["bbox_id"]))
    return evidence, bbox_rows


def _recover_word_bbox(
    *,
    evidence_id: str,
    text: str,
    source_meta: dict,
    source_id: str,
    page_numbers: list[int],
    word_bboxes: list[dict],
) -> list[dict]:
    words_by_page = word_rows_by_page(word_bboxes)
    match = align_text_to_word_bboxes(
        text=text,
        source_document_id=source_id,
        page_numbers=page_numbers,
        words_by_page=words_by_page,
    )
    if not match:
        return []
    union = match["union_bbox"]
    return [
        {
            "bbox_id": _geometry_id(
                source_id=source_id,
                source_sha256=source_meta["sha256"],
                page_number=union["page_number"],
                text=text,
                coordinates=union,
            ),
            "bbox_precision": "exact",
            "corpus_id": "uud",
            "page_number": union["page_number"],
            "source_document_id": source_id,
            "source_pdf": source_meta["filename"],
            "source_pdf_path": source_meta["path"],
            "source_sha256": source_meta["sha256"],
            "status": "accepted",
            "text": text,
            "viewer_highlightable": True,
            "x0": union["x0"],
            "x1": union["x1"],
            "y0": union["y0"],
            "y1": union["y1"],
            **coordinate_metadata(
                {
                    "width": union["page_width"],
                    "height": union["page_height"],
                },
                highlightable=True,
            ),
        }
    ]


def _recover_aggregate_word_bboxes(
    *,
    evidence_id: str,
    text: str,
    text_parts: list[str] | None = None,
    source_meta: dict,
    source_id: str,
    page_numbers: list[int],
    word_bboxes: list[dict],
) -> list[dict]:
    words_by_page = word_rows_by_page(word_bboxes)
    matches = []
    for part in text_parts or [text]:
        match = align_text_to_word_bboxes(
            text=part,
            source_document_id=source_id,
            page_numbers=page_numbers,
            words_by_page=words_by_page,
            allow_cross_page=True,
        )
        if not match:
            prefix = next((line.strip() for line in part.splitlines() if line.strip()), "")
            if prefix and prefix != part:
                match = align_text_to_word_bboxes(
                    text=prefix,
                    source_document_id=source_id,
                    page_numbers=page_numbers,
                    words_by_page=words_by_page,
                    allow_cross_page=True,
                )
            if not match:
                continue
        matches.extend(match.get("matched_word_bboxes") or ())
    matched = list(dict.fromkeys(word.get("word_bbox_id") for word in matches))
    if not matched:
        return []
    matches = sorted(
        {word["word_bbox_id"]: word for word in matches}.values(),
        key=lambda word: (word["page_number"], word.get("y0", 0), word.get("x0", 0), word.get("word_index", 0)),
    )
    rows: list[dict] = []
    for index, page_number in enumerate(dict.fromkeys(word["page_number"] for word in matches)):
        words = [word for word in matches if word["word_bbox_id"] in matched and word["page_number"] == page_number]
        if not words:
            continue
        first = words[0]
        rows.append(
            {
                "bbox_id": _geometry_id(
                    source_id=source_id,
                    source_sha256=source_meta["sha256"],
                    page_number=page_number,
                    text=" ".join(word["text"] for word in words),
                    coordinates={
                        "x0": min(word["x0"] for word in words),
                        "y0": min(word["y0"] for word in words),
                        "x1": max(word["x1"] for word in words),
                        "y1": max(word["y1"] for word in words),
                    },
                ),
                "bbox_precision": "exact",
                "corpus_id": "uud",
                "page_number": page_number,
                "source_document_id": source_id,
                "source_pdf": source_meta["filename"],
                "source_pdf_path": source_meta["path"],
                "source_sha256": source_meta["sha256"],
                "status": "accepted",
                "text": " ".join(word["text"] for word in words),
                "viewer_highlightable": True,
                "x0": min(word["x0"] for word in words),
                "x1": max(word["x1"] for word in words),
                "y0": min(word["y0"] for word in words),
                "y1": max(word["y1"] for word in words),
                **coordinate_metadata(
                    {"width": first.get("page_width"), "height": first.get("page_height")},
                    highlightable=True,
                ),
            }
        )
    return rows


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
    prefix = _evidence_prefix(source_role)
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


def _has_ayat(text: object) -> bool:
    return bool(re.search(r"(?m)^\([0-9]+\)", str(text or "")))


def _aggregate_failure_reason(bbox_records: list[dict], source_id: str, expected_text: str) -> str | None:
    if not bbox_records:
        return "pasal_aggregate_source_missing"
    if any(
        row.get("source_document_id") != source_id
        or row.get("bbox_precision") != "exact"
        or row.get("viewer_highlightable") is not True
        or not all(row.get(field) is not None for field in ("x0", "y0", "x1", "y1"))
        for row in bbox_records
    ):
        return "pasal_aggregate_geometry_unavailable"
    if _aggregate_compare_text(" ".join(row.get("text") or "" for row in bbox_records)) != _aggregate_compare_text(expected_text):
        return "pasal_aggregate_geometry_unavailable"
    return None


def _aggregate_compare_text(text: str) -> str:
    return compact_text(re.sub(r"\*+\)", "", text))


def _descendants_by_parent(legal_units: list[dict]) -> dict[str, list[dict]]:
    rows: dict[str, list[dict]] = {}
    for unit in legal_units:
        for parent_id in unit.get("parent_legal_unit_ids") or ():
            rows.setdefault(parent_id, []).append(unit)
    return rows


def _descendants(unit_id: str, descendants_by_parent: dict[str, list[dict]]) -> list[dict]:
    found: list[dict] = []
    pending = list(descendants_by_parent.get(unit_id, ()))
    while pending:
        current = pending.pop(0)
        found.append(current)
        pending.extend(descendants_by_parent.get(current["legal_unit_id"], ()))
    return found


def _aggregate_text(unit: dict, descendants: list[dict]) -> str:
    return "\n".join(_aggregate_parts(unit, descendants)).strip()


def _aggregate_parts(unit: dict, descendants: list[dict]) -> list[str]:
    text = str(unit.get("text") or "")
    parts = [text]
    accumulated = compact(text)
    for child in sorted(
        _direct_descendants(unit, descendants),
        key=lambda row: (row.get("page_start", 0), row.get("sibling_order", 0), row["legal_unit_id"]),
    ):
        child_text = str(child.get("text") or "")
        child_compact = compact(child_text)
        if not child_compact or child_compact in accumulated:
            continue
        parts.append(child_text)
        accumulated = compact(f"{accumulated} {child_compact}")
    return parts


def _aggregate_recovery_parts(unit: dict, descendants: list[dict]) -> list[str]:
    return [
        text
        for row in [
            unit,
            *sorted(
                descendants,
                key=lambda item: (item.get("page_start", 0), item.get("sibling_order", 0), item["legal_unit_id"]),
            ),
        ]
        if (text := str(row.get("text") or "").strip())
    ]


def _direct_descendants(unit: dict, descendants: list[dict]) -> list[dict]:
    unit_id = unit.get("legal_unit_id")
    return [row for row in descendants if unit_id in (row.get("parent_legal_unit_ids") or ())]


def _evidence_prefix(source_role: str) -> str:
    if source_role == "current_consolidated":
        return "uud_current_consolidated_final_citation_evidence"
    if source_role in {"amendment_1_historical", "amendment_4_historical"}:
        return "uud_source_role_final_citation_evidence"
    return "uud_source_role_historical_final_citation_evidence"
