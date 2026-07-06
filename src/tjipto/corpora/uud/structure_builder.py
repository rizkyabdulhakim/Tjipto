from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable


def apply_inserted_bab_specs(
    *,
    specs: tuple[dict, ...],
    pages_by_source: dict[tuple[str, int], str],
    source_documents: dict[str, dict],
    legal_units: list[dict],
    chunks: list[dict],
    units_by_source_label: dict[tuple[str, str | None], dict],
    trim_unit: Callable[..., None],
    trim_bab: Callable[[str, str, str], None],
    allocate_legal_id: Callable[[], str],
    allocate_chunk_id: Callable[[], str],
) -> None:
    for spec in specs:
        source_id = spec["source_document_id"]
        page_text = pages_by_source[(source_id, spec["page_number"])]
        bab_text = slice_between(page_text, spec["start"], spec["end"])
        for target in spec["trim_targets"]:
            if isinstance(target, tuple):
                trim_unit(source_id, target[-1], spec["label"], hierarchy_suffix=target)
            else:
                trim_unit(source_id, target, spec["label"])
        if spec["trim_bab"]:
            trim_bab(source_id, spec["trim_bab"], spec["label"])
        existing = units_by_source_label.get((source_id, spec["label"]))
        if existing:
            legal_unit_id = existing["legal_unit_id"]
            for child_label in spec["child_labels"]:
                child = units_by_source_label[(source_id, child_label)]
                if legal_unit_id not in child["parent_legal_unit_ids"]:
                    child["parent_legal_unit_ids"] = [legal_unit_id, *child["parent_legal_unit_ids"]]
            continue
        parent_ids = []
        if spec["parent_label"]:
            parent = units_by_source_label[(source_id, spec["parent_label"])]
            parent_ids.append(parent["legal_unit_id"])
        legal_unit_id = allocate_legal_id()
        chunk_id = allocate_chunk_id()
        source_meta = source_documents[source_id]
        legal_units.append(
            {
                "corpus_id": "uud",
                "hierarchy": [],
                "legal_unit_id": legal_unit_id,
                "page_end": spec["page_number"],
                "page_start": spec["page_number"],
                "parent_legal_unit_ids": parent_ids,
                "provenance": {"donor_id": legal_unit_id},
                "source_document_id": source_id,
                "source_sha256": source_meta["sha256"],
                "status": "finalizable",
                "text": bab_text,
                "unit_label": spec["label"],
                "unit_type": "bab_record",
            }
        )
        chunks.append(
            {
                "canonical_use_allowed": False,
                "chunk_id": chunk_id,
                "chunk_type": "bab_structural_context_record",
                "corpus_id": "uud",
                "hierarchy": [spec["label"]],
                "legal_unit_id": legal_unit_id,
                "page_range": {"start_page_number": spec["page_number"], "end_page_number": spec["page_number"]},
                "provenance": {"donor_id": chunk_id},
                "source_sha256": source_meta["sha256"],
                "status": "parent_context_only",
                "text": bab_text,
            }
        )
        units_by_source_label[(source_id, spec["label"])] = legal_units[-1]
        for child_label in spec["child_labels"]:
            child = units_by_source_label[(source_id, child_label)]
            if legal_unit_id not in child["parent_legal_unit_ids"]:
                child["parent_legal_unit_ids"] = [legal_unit_id, *child["parent_legal_unit_ids"]]
            for unit in legal_units:
                if unit["source_document_id"] != source_id or unit["unit_type"] != "ayat_record":
                    continue
                if (
                    child["legal_unit_id"] in (unit.get("parent_legal_unit_ids") or ())
                    and legal_unit_id not in unit["parent_legal_unit_ids"]
                ):
                    unit["parent_legal_unit_ids"] = [legal_unit_id, *unit["parent_legal_unit_ids"]]


def next_numeric_id(rows: list[dict], key: str) -> int:
    max_value = 0
    for row in rows:
        value = numeric_suffix(str(row.get(key, "")))
        if value:
            max_value = max(max_value, value)
    return max_value + 1


def numeric_suffix(value: str) -> int:
    match = re.search(r"(\d+)$", value)
    return int(match.group(1)) if match else 0


def find_unit(
    legal_units: list[dict],
    source_document_id: str,
    unit_label: str,
    *,
    hierarchy_suffix: tuple[str, ...] | None = None,
) -> dict:
    candidates = [row for row in legal_units if row["source_document_id"] == source_document_id and row.get("unit_label") == unit_label]
    if hierarchy_suffix is not None:
        compact_suffix = tuple(compact(part) for part in hierarchy_suffix)
        candidates = [
            row
            for row in candidates
            if tuple(compact(part) for part in [*(row.get("hierarchy") or ()), row.get("unit_label")])[-len(compact_suffix) :]
            == compact_suffix
        ]
    if len(candidates) != 1:
        raise KeyError(f"unable_to_resolve_unit:{source_document_id}:{unit_label}:{hierarchy_suffix}")
    return candidates[0]


def slice_before(text: str, marker: str) -> str:
    return text[: text.index(marker)]


def slice_between(text: str, start: str, end: str) -> str:
    start_index = text.index(start)
    end_index = text.index(end, start_index + len(start))
    return text[start_index:end_index].strip()


def trim_before(text: str, marker: str) -> str:
    return text[: text.index(marker)].rstrip() + "\n"


def split_effective_clause(text: str) -> tuple[str, str]:
    marker = ", dan mulai berlaku"
    if marker not in text:
        return text, "dan mulai berlaku pada tanggal ditetapkan."
    head, tail = text.split(marker, 1)
    effective = ("dan mulai berlaku" + tail).strip()
    return f"{head.strip()}, {effective}", effective


def page_span_for_text(
    pages_by_source: dict[tuple[str, int], str],
    source_id: str,
    text: str,
    page_start: int,
    page_end: int,
) -> tuple[int, int]:
    for page_number in range(page_start, page_end + 1):
        if compact(text) in compact(pages_by_source[(source_id, page_number)]):
            return page_number, page_number
    return page_start, page_end


def compact(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").replace("\u00ad", "").replace("\u00c2", "")
    return re.sub(r"\s+", " ", text).strip().casefold()


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (text or "").casefold()).strip("_")
