from __future__ import annotations

import re

from tjipto.corpora.uud.specs import UUD_CHUNK_ID_STARTS, UUD_LEGAL_UNIT_SOURCE_ORDER
from tjipto.corpora.uud.span_disposition_policy import substantive_structural_unit


def build_chunks_from_legal_units(legal_units: list[dict]) -> list[dict]:
    by_source = {source_id: 0 for source_id in UUD_LEGAL_UNIT_SOURCE_ORDER}
    rows = []
    for unit in _chunk_ordered_units(legal_units):
        source_id = unit["source_document_id"]
        chunk_number = _chunk_number(source_id, by_source[source_id], unit)
        by_source[source_id] += 1
        chunk_id = f"uud_chunk_{chunk_number:05d}"
        row = {
            "canonical_use_allowed": _canonical_use_allowed(unit),
            "chunk_id": chunk_id,
            "chunk_type": _chunk_type(unit),
            "corpus_id": "uud",
            "hierarchy": _chunk_hierarchy(unit),
            "legal_unit_id": unit["legal_unit_id"],
            "page_range": {"start_page_number": unit["page_start"], "end_page_number": unit["page_end"]},
            "provenance": {"donor_id": chunk_id},
            "source_sha256": unit["source_sha256"],
            "status": _chunk_status(unit),
            "text": unit["text"],
        }
        if unit.get("runtime_loadable") is False:
            row["runtime_loadable"] = False
        if unit.get("exclusion_ref"):
            row["exclusion_ref"] = unit["exclusion_ref"]
        rows.append(row)
    rows.sort(key=lambda row: row["chunk_id"])
    return rows


def _chunk_ordered_units(legal_units: list[dict]) -> list[dict]:
    source_rank = {source_id: index for index, source_id in enumerate(UUD_LEGAL_UNIT_SOURCE_ORDER)}
    return sorted(
        legal_units,
        key=lambda row: (
            source_rank[row["source_document_id"]],
            int(row["legal_unit_id"].rsplit("_", 1)[1]),
        ),
    )


def _chunk_number(source_id: str, source_index: int, unit: dict) -> int:
    legal_number = int(unit["legal_unit_id"].rsplit("_", 1)[1])
    if legal_number >= 610:
        return legal_number
    return UUD_CHUNK_ID_STARTS[source_id] + source_index


def _chunk_type(unit: dict) -> str:
    unit_type = unit["unit_type"]
    if unit_type == "pembukaan_record":
        return "pembukaan_special_chunk_record"
    if unit_type == "bab_record":
        return "bab_structural_context_record"
    if unit_type in {"aturan_peralihan_record", "aturan_tambahan_record"}:
        return "aturan_section_context_record"
    if unit_type == "ayat_record":
        return "ayat_chunk_record"
    if unit_type == "pasal_record":
        return "pasal_parent_context_record" if _has_ayat(unit["text"]) else "pasal_chunk_record"
    return f"{unit_type.replace('_record', '')}_chunk_record"


def _chunk_status(unit: dict) -> str:
    if unit.get("runtime_loadable") is False:
        return unit["status"]
    if unit.get("status") == "active_historical_record":
        return "active_historical_record"
    if substantive_structural_unit(unit):
        return "active_canonical_record"
    if _chunk_type(unit) in {"bab_structural_context_record", "aturan_section_context_record", "pasal_parent_context_record"}:
        return "parent_context_only"
    return "active_canonical_record"


def _canonical_use_allowed(unit: dict) -> bool:
    if "canonical_use_allowed" in unit:
        return unit["canonical_use_allowed"] is True
    return _chunk_status(unit) == "active_canonical_record"


def _chunk_hierarchy(unit: dict) -> list[str]:
    hierarchy = list(unit.get("hierarchy") or [])
    label = unit.get("unit_label")
    if label and (not hierarchy or hierarchy[-1] != label):
        hierarchy.append(label)
    return hierarchy


def _has_ayat(text: str) -> bool:
    return bool(re.search(r"(?m)^\([0-9]+\)", text or ""))
