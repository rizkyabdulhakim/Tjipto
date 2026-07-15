from __future__ import annotations

import re
import unicodedata

from tjipto.corpora.uud.anomaly_builder import append_amendment_instrument_units
from tjipto.corpora.uud.parser import UUD_LEGAL_TOKEN_RE
from tjipto.corpora.uud.specs import (
    EXCLUDED_RECORD_SPECS,
    INSERTED_BAB_SPECS,
    UUD_LEGAL_UNIT_ID_STARTS,
    UUD_LEGAL_UNIT_SOURCE_ORDER,
    UUD_CHUNK_ID_STARTS,
)
from tjipto.corpora.uud.structure_builder import page_span_for_text, slice_between, trim_before
from tjipto.corpora.structure import apply_structural_contract


UUD_STRUCTURAL_ROLE_BY_UNIT_TYPE = {
    "bab_record": "division",
    "pasal_record": "provision",
    "ayat_record": "subprovision",
    "instrument_clause_record": "item",
    "pembukaan_record": "document",
    "aturan_peralihan_record": "division",
    "aturan_tambahan_record": "division",
}


def build_legal_units_from_sources(
    *,
    pages_by_source: dict[tuple[str, int], str],
    source_documents: dict[str, dict],
) -> list[dict]:
    rows: list[dict] = []
    chunk_by_unit: dict[str, str] = {}
    for source_id in UUD_LEGAL_UNIT_SOURCE_ORDER:
        _append_source_units(source_id, pages_by_source, source_documents, rows, chunk_by_unit)
    _append_inserted_bab_units(pages_by_source, source_documents, rows)
    _append_instrument_units(pages_by_source, source_documents, rows)
    apply_structural_contract(rows, role_by_unit_type=UUD_STRUCTURAL_ROLE_BY_UNIT_TYPE)
    rows.sort(key=lambda row: row["legal_unit_id"])
    return rows


def _append_source_units(
    source_id: str,
    pages_by_source: dict[tuple[str, int], str],
    source_documents: dict[str, dict],
    rows: list[dict],
    chunk_by_unit: dict[str, str],
) -> None:
    source_meta = source_documents[source_id]
    source_text, page_ranges = _source_text(source_id, pages_by_source)
    tokens = list(UUD_LEGAL_TOKEN_RE.finditer(source_text))
    skipped_babs = {spec["label"] for spec in INSERTED_BAB_SPECS if spec["source_document_id"] == source_id}
    inserted_children = {spec["label"]: set(spec["child_labels"]) for spec in INSERTED_BAB_SPECS if spec["source_document_id"] == source_id}
    current_bab: dict | None = None
    current_pasal: dict | None = None
    source_count = 0
    excluded = {row["legacy_chunk_id"]: row for row in EXCLUDED_RECORD_SPECS}
    for index, token in enumerate(tokens):
        label = token.group(1)
        if _token_kind(label) == "Stop":
            continue
        if label in skipped_babs:
            current_bab = {"id": None, "label": label}
            current_pasal = None
            continue
        if source_id == "uud::amendment_4_historical" and label == "ATURAN TAMBAHAN":
            break
        unit_type = _unit_type(label)
        if (
            unit_type == "pasal_record"
            and current_bab
            and current_bab["id"] is None
            and label not in inserted_children.get(current_bab["label"], set())
        ):
            current_bab = None
        if source_id == "uud::amendment_4_historical" and unit_type == "pasal_record" and label == "Pasal 37":
            current_bab = None
        end = _unit_end(source_text, tokens, index, unit_type)
        text = source_text[token.start() : end]
        page_start, page_end = _span_pages(token.start(), end, page_ranges)
        legal_id = f"uud_legal_unit_{UUD_LEGAL_UNIT_ID_STARTS[source_id] + source_count:05d}"
        chunk_id = f"uud_chunk_{UUD_CHUNK_ID_STARTS[source_id] + source_count:05d}"
        source_count += 1
        if unit_type == "pembukaan_record":
            label = "PEMBUKAAN/Preambule"
            current_bab = None
            current_pasal = None
        elif unit_type in {"bab_record", "aturan_peralihan_record", "aturan_tambahan_record"}:
            current_bab = {"id": legal_id, "label": label}
            current_pasal = None
        elif unit_type == "pasal_record":
            current_pasal = {"id": legal_id, "label": label}
        row = {
            "corpus_id": "uud",
            "hierarchy": _hierarchy(unit_type, current_bab, current_pasal),
            "legal_unit_id": legal_id,
            "page_end": page_end,
            "page_start": page_start,
            "parent_legal_unit_ids": _parent_ids(unit_type, current_bab, current_pasal),
            "provenance": {"donor_id": legal_id},
            "source_document_id": source_id,
            "source_role": source_meta["source_role"],
            "source_sha256": source_meta["sha256"],
            "temporal_context": source_meta.get("temporal_context", source_meta["source_role"]),
            "status": "finalizable",
            "text": text,
            "unit_label": label,
            "unit_type": unit_type,
        }
        if chunk_id in excluded:
            row["status"] = excluded[chunk_id]["status"]
            row["runtime_loadable"] = False
            row["exclusion_ref"] = excluded[chunk_id]["excluded_record_id"]
        rows.append(row)
        chunk_by_unit[legal_id] = chunk_id
        if source_id == "uud::amendment_4_historical" and unit_type == "bab_record" and label == "BAB IV":
            current_bab = None


def _append_inserted_bab_units(
    pages_by_source: dict[tuple[str, int], str],
    source_documents: dict[str, dict],
    rows: list[dict],
) -> None:
    next_id = 610
    by_key = {(row["source_document_id"], row["unit_label"]): row for row in rows}
    for spec in INSERTED_BAB_SPECS:
        source_id = spec["source_document_id"]
        source_meta = source_documents[source_id]
        legal_id = f"uud_legal_unit_{next_id:05d}"
        next_id += 1
        text = slice_between(pages_by_source[(source_id, spec["page_number"])], spec["start"], spec["end"])
        parent_ids = []
        if spec["parent_label"]:
            parent_ids.append(by_key[(source_id, spec["parent_label"])]["legal_unit_id"])
        row = {
            "corpus_id": "uud",
            "hierarchy": [],
            "legal_unit_id": legal_id,
            "page_end": spec["page_number"],
            "page_start": spec["page_number"],
            "parent_legal_unit_ids": parent_ids,
            "provenance": {"donor_id": legal_id},
            "source_document_id": source_id,
            "source_role": source_meta["source_role"],
            "source_sha256": source_meta["sha256"],
            "temporal_context": source_meta.get("temporal_context", source_meta["source_role"]),
            "status": "finalizable",
            "text": text,
            "unit_label": spec["label"],
            "unit_type": "bab_record",
        }
        rows.append(row)
        by_key[(source_id, spec["label"])] = row
        for child_label in spec["child_labels"]:
            child = by_key[(source_id, child_label)]
            if legal_id not in child["parent_legal_unit_ids"]:
                child["parent_legal_unit_ids"] = [legal_id, *child["parent_legal_unit_ids"]]
            for unit in rows:
                if unit["source_document_id"] != source_id or unit["unit_type"] != "ayat_record":
                    continue
                if child["legal_unit_id"] in unit.get("parent_legal_unit_ids", []) and legal_id not in unit["parent_legal_unit_ids"]:
                    unit["parent_legal_unit_ids"] = [legal_id, *unit["parent_legal_unit_ids"]]


def _append_instrument_units(
    pages_by_source: dict[tuple[str, int], str],
    source_documents: dict[str, dict],
    rows: list[dict],
) -> None:
    next_id = 620

    def allocate_id() -> str:
        nonlocal next_id
        value = f"uud_legal_unit_{next_id:05d}"
        next_id += 1
        return value

    def append_unit(
        source_id: str,
        unit_type: str,
        unit_label: str,
        text: str,
        page_start: int,
        page_end: int,
        *,
        hierarchy: list[str] | None = None,
        parent_legal_unit_ids: list[str] | None = None,
        chunk_status: str = "active_canonical_record",
        runtime_loadable: bool | None = None,
        exclusion_ref: str | None = None,
        **_: object,
    ) -> str:
        legal_id = allocate_id()
        row = {
            "corpus_id": "uud",
            "hierarchy": hierarchy or [],
            "legal_unit_id": legal_id,
            "page_end": page_end,
            "page_start": page_start,
            "parent_legal_unit_ids": parent_legal_unit_ids or [],
            "provenance": {"donor_id": legal_id},
            "source_document_id": source_id,
            "source_role": source_documents[source_id]["source_role"],
            "source_sha256": source_documents[source_id]["sha256"],
            "temporal_context": source_documents[source_id].get("temporal_context", source_documents[source_id]["source_role"]),
            "status": chunk_status if runtime_loadable is False else "finalizable",
            "text": text,
            "unit_label": unit_label,
            "unit_type": unit_type,
        }
        if runtime_loadable is False:
            row["runtime_loadable"] = False
        if exclusion_ref:
            row["exclusion_ref"] = exclusion_ref
        rows.append(row)
        return legal_id

    def trim_unit(source_document_id: str, unit_label: str, marker: str, *, hierarchy_suffix: tuple[str, ...] | None = None) -> None:
        unit = _find_unit(rows, source_document_id, unit_label, hierarchy_suffix)
        if marker not in unit["text"]:
            return
        trimmed = trim_before(unit["text"], marker)
        unit["text"] = trimmed
        unit["page_start"], unit["page_end"] = page_span_for_text(
            pages_by_source,
            source_document_id,
            trimmed,
            unit["page_start"],
            unit["page_end"],
        )

    def trim_bab(source_document_id: str, unit_label: str, marker: str) -> None:
        trim_unit(source_document_id, unit_label, marker)

    append_amendment_instrument_units(
        pages_by_source=pages_by_source,
        append_instrument_unit=append_unit,
        trim_unit=trim_unit,
        trim_bab=trim_bab,
    )


def _source_text(source_id: str, pages_by_source: dict[tuple[str, int], str]) -> tuple[str, list[tuple[int, int, int]]]:
    parts: list[str] = []
    ranges: list[tuple[int, int, int]] = []
    cursor = 0
    for page_number in sorted(page for doc_id, page in pages_by_source if doc_id == source_id):
        text = _strip_page_header(source_id, pages_by_source[(source_id, page_number)])
        parts.append(text)
        ranges.append((cursor, cursor + len(text), page_number))
        cursor += len(text) + 1
    text = "\n".join(parts)
    text, offset = _normative_slice(source_id, text)
    ranges = [(max(0, start - offset), max(0, end - offset), page) for start, end, page in ranges if end > offset]
    return text, ranges


def _normative_slice(source_id: str, text: str) -> tuple[str, int]:
    if source_id == "uud::current_consolidated":
        start = text.index("PEMBUKAAN")
        return text[start:], start
    if source_id == "uud::original_historical":
        start = text.index("PEMBUKAAN")
        return text[start:], start
    marker = {
        "uud::amendment_1_historical": "selengkapnya menjadi berbunyi sebagai berikut :",
        "uud::amendment_2_historical": "sehingga selengkapnya berbunyi sebagai berikut :",
        "uud::amendment_3_historical": "sehingga selengkapnya menjadi berbunyi sebagai berikut:",
        "uud::amendment_4_historical": "sehingga selengkapnya berbunyi sebagai \nberikut.",
    }[source_id]
    start = text.index(marker) + len(marker)
    return text[start:], start


def _strip_page_header(source_id: str, text: str) -> str:
    if source_id != "uud::current_consolidated":
        return text
    lines = text.split("\n")
    for index, line in enumerate(lines):
        if line.strip() == "Perubahan Keempat":
            return "\n".join(lines[index + 1 :]).lstrip("\n")
    return text


def _unit_end(text: str, tokens: list[re.Match[str]], index: int, unit_type: str) -> int:
    stops = {"BAB", "ATURAN", "PEMBUKAAN", "Stop"}
    if unit_type == "pasal_record":
        stops.add("Pasal")
    elif unit_type == "ayat_record":
        stops |= {"Pasal", "Ayat"}
    for token in tokens[index + 1 :]:
        if _token_kind(token.group(1)) in stops:
            return token.start()
    return len(text)


def _span_pages(start: int, end: int, ranges: list[tuple[int, int, int]]) -> tuple[int, int]:
    pages = [
        page
        for range_start, range_end, page in ranges
        if range_start <= start < range_end or range_start < max(start, end - 1) <= range_end
    ]
    if pages:
        return min(pages), max(pages)
    return ranges[0][2], ranges[-1][2]


def _token_kind(label: str) -> str:
    if label.startswith("BAB"):
        return "BAB"
    if label.startswith("ATURAN"):
        return "ATURAN"
    if label.startswith("Pasal"):
        return "Pasal"
    if label.startswith("("):
        return "Ayat"
    if label.startswith("UNDANG"):
        return "Stop"
    return label


def _unit_type(label: str) -> str:
    if label == "PEMBUKAAN":
        return "pembukaan_record"
    if label.startswith("BAB"):
        return "bab_record"
    if label == "ATURAN PERALIHAN":
        return "aturan_peralihan_record"
    if label == "ATURAN TAMBAHAN":
        return "aturan_tambahan_record"
    if label.startswith("Pasal"):
        return "pasal_record"
    return "ayat_record"


def _hierarchy(unit_type: str, current_bab: dict | None, current_pasal: dict | None) -> list[str]:
    if unit_type in {"pembukaan_record", "bab_record", "aturan_peralihan_record", "aturan_tambahan_record"}:
        return []
    if unit_type == "pasal_record":
        return [current_bab["label"]] if current_bab else []
    return [value for value in [current_bab["label"] if current_bab else None, current_pasal["label"] if current_pasal else None] if value]


def _parent_ids(unit_type: str, current_bab: dict | None, current_pasal: dict | None) -> list[str]:
    if unit_type == "pasal_record" and current_bab and current_bab["id"]:
        return [current_bab["id"]]
    if unit_type == "ayat_record":
        return [value for value in [current_bab["id"] if current_bab else None, current_pasal["id"] if current_pasal else None] if value]
    return []


def _find_unit(rows: list[dict], source_document_id: str, unit_label: str, hierarchy_suffix: tuple[str, ...] | None) -> dict:
    candidates = [row for row in rows if row["source_document_id"] == source_document_id and row.get("unit_label") == unit_label]
    if hierarchy_suffix:
        suffix = tuple(_compact(part) for part in hierarchy_suffix)
        candidates = [
            row
            for row in candidates
            if tuple(_compact(part) for part in [*(row.get("hierarchy") or []), row.get("unit_label")])[-len(suffix) :] == suffix
        ]
    if len(candidates) != 1:
        raise KeyError(f"unable_to_resolve_unit:{source_document_id}:{unit_label}:{hierarchy_suffix}")
    return candidates[0]


def _compact(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").replace("\u00ad", "").replace("\u00c2", "")
    return re.sub(r"\s+", " ", text).strip().casefold()
