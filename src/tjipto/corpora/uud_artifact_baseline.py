from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
import re
import unicodedata

from tjipto.core.manifest import file_sha256, read_json, read_jsonl


FINAL_DIR = Path("data/final/uud")
DECISION_LABELS = {
    "Perubahan Pertama Decision",
    "Perubahan Ketiga Decision",
    "Perubahan Keempat Decision",
}
INSERTED_BAB_HEADING_BBOX_MARKER = "::heading_bab_"
STRUCTURAL_FORBIDDEN_MARKERS = (
    "Ditetapkan di Jakarta",
)


INSERTED_BAB_SPECS = (
    {
        "source_document_id": "uud::current_consolidated",
        "label": "BAB VIIA",
        "title": "DEWAN PERWAKILAN DAERAH",
        "page_number": 14,
        "start": "BAB VIIA",
        "end": "Pasal 22C",
        "parent_label": "BAB VII",
        "child_labels": ("Pasal 22C", "Pasal 22D"),
        "trim_targets": ("Pasal 22B",),
        "trim_bab": "BAB VII",
    },
    {
        "source_document_id": "uud::current_consolidated",
        "label": "BAB VIIB",
        "title": "PEMILIHAN UMUM",
        "page_number": 15,
        "start": "BAB VIIB",
        "end": "Pasal 22E",
        "parent_label": "BAB VII",
        "child_labels": ("Pasal 22E",),
        "trim_targets": ("Pasal 22D", ("BAB VIIA", "Pasal 22D", "(4)")),
        "trim_bab": "BAB VII",
    },
    {
        "source_document_id": "uud::current_consolidated",
        "label": "BAB VIIIA",
        "title": "BADAN PEMERIKSA KEUANGAN",
        "page_number": 16,
        "start": "BAB VIIIA",
        "end": "Pasal 23E",
        "parent_label": "BAB VIII",
        "child_labels": ("Pasal 23E", "Pasal 23F", "Pasal 23G"),
        "trim_targets": ("Pasal 23D",),
        "trim_bab": "BAB VIII",
    },
    {
        "source_document_id": "uud::current_consolidated",
        "label": "BAB IXA",
        "title": "WILAYAH NEGARA",
        "page_number": 19,
        "start": "BAB IXA",
        "end": "Pasal 25A",
        "parent_label": "BAB IX",
        "child_labels": ("Pasal 25A",),
        "trim_targets": ("Pasal 25",),
        "trim_bab": "BAB IX",
    },
    {
        "source_document_id": "uud::current_consolidated",
        "label": "BAB XA",
        "title": "HAK ASASI MANUSIA",
        "page_number": 20,
        "start": "BAB XA",
        "end": "Pasal 28A",
        "parent_label": "BAB X",
        "child_labels": ("Pasal 28A", "Pasal 28B", "Pasal 28C", "Pasal 28D", "Pasal 28E", "Pasal 28F", "Pasal 28G", "Pasal 28H", "Pasal 28I", "Pasal 28J"),
        "trim_targets": ("Pasal 28",),
        "trim_bab": "BAB X",
    },
    {
        "source_document_id": "uud::amendment_2_historical",
        "label": "BAB IXA",
        "title": "WILAYAH NEGARA",
        "page_number": 3,
        "start": "BAB IXA",
        "end": "Pasal 25E",
        "parent_label": None,
        "child_labels": ("Pasal 25E",),
        "trim_targets": (),
        "trim_bab": None,
    },
    {
        "source_document_id": "uud::amendment_2_historical",
        "label": "BAB XA",
        "title": "HAK ASASI MANUSIA",
        "page_number": 4,
        "start": "BAB XA",
        "end": "Pasal 28A",
        "parent_label": "BAB X",
        "child_labels": ("Pasal 28A", "Pasal 28B", "Pasal 28C", "Pasal 28D", "Pasal 28E", "Pasal 28F", "Pasal 28G", "Pasal 28H", "Pasal 28I", "Pasal 28J"),
        "trim_targets": ("Pasal 27",),
        "trim_bab": "BAB X",
    },
    {
        "source_document_id": "uud::amendment_3_historical",
        "label": "BAB VIIA",
        "title": "DEWAN PERWAKILAN DAERAH",
        "page_number": 4,
        "start": "BAB VIIA",
        "end": "Pasal 22C",
        "parent_label": None,
        "child_labels": ("Pasal 22C", "Pasal 22D"),
        "trim_targets": ("Pasal 17",),
        "trim_bab": None,
    },
    {
        "source_document_id": "uud::amendment_3_historical",
        "label": "BAB VIIB",
        "title": "PEMILIHAN UMUM",
        "page_number": 5,
        "start": "BAB VIIB",
        "end": "Pasal 22E",
        "parent_label": None,
        "child_labels": ("Pasal 22E",),
        "trim_targets": ("Pasal 22D",),
        "trim_bab": None,
    },
    {
        "source_document_id": "uud::amendment_3_historical",
        "label": "BAB VIIIA",
        "title": "BADAN PEMERIKSA KEUANGAN",
        "page_number": 6,
        "start": "BAB VIIIA",
        "end": "Pasal 23E",
        "parent_label": None,
        "child_labels": ("Pasal 23E", "Pasal 23F", "Pasal 23G"),
        "trim_targets": (),
        "trim_bab": None,
    },
)


def rebuild_uud_artifact_baseline(repo_root: Path) -> dict:
    try:
        import fitz
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("PyMuPDF is required to rebuild UUD artifacts") from error

    final_dir = (repo_root / FINAL_DIR).resolve()
    manifest = read_json(final_dir / "manifest.json")
    pages = read_jsonl(final_dir / "pages.jsonl")
    legal_units = read_jsonl(final_dir / "legal_units.jsonl")
    chunks = read_jsonl(final_dir / "chunks.jsonl")
    evidence = read_jsonl(final_dir / "evidence_registry.jsonl")
    bbox_rows = read_jsonl(final_dir / "bbox_registry.jsonl")
    retrieval_units = read_jsonl(final_dir / "retrieval_units.jsonl")
    document_metadata = read_jsonl(final_dir / "document_metadata.jsonl")
    metadata_grounding = read_jsonl(final_dir / "metadata_grounding.jsonl")
    metadata_grounding_registry = read_jsonl(final_dir / "metadata_grounding_registry.jsonl")
    source_conflicts = read_jsonl(final_dir / "source_conflicts.jsonl")
    validation_report = read_json(final_dir / "validation_report.json")

    source_documents = {row["source_document_id"]: row for row in read_jsonl(final_dir / "source_documents.jsonl")}
    pages_by_source = {(row["source_document_id"], row["page_number"]): row["text"] for row in pages}
    legal_units = [row for row in legal_units if _numeric_suffix(row["legal_unit_id"]) <= 609]
    chunks = [row for row in chunks if _numeric_suffix(row["chunk_id"]) <= 609]
    evidence = [row for row in evidence if not row["evidence_id"].startswith("uud_instrument_final_citation_evidence::")]
    bbox_rows = [row for row in bbox_rows if not row["evidence_id"].startswith("uud_instrument_final_citation_evidence::")]
    retrieval_units = [row for row in retrieval_units if not row["retrieval_unit_id"].startswith("uud_retrieval_unit::uud_instrument_final_citation_evidence::")]
    units_by_source_label = {(row["source_document_id"], row.get("unit_label")): row for row in legal_units}
    chunks_by_unit = {row["legal_unit_id"]: row for row in chunks}
    evidence_by_unit = {row["legal_unit_id"]: row for row in evidence}
    bbox_by_evidence: dict[str, list[dict]] = defaultdict(list)
    for row in bbox_rows:
        row.setdefault("bbox_precision", "exact")
        row.setdefault("viewer_highlightable", True)
        bbox_by_evidence[row["evidence_id"]].append(row)
    for row in evidence:
        if row["evidence_id"] in bbox_by_evidence:
            row["bbox_precision"] = _aggregate_bbox_precision(bbox_by_evidence[row["evidence_id"]])
            row["viewer_highlightable"] = any(item["viewer_highlightable"] for item in bbox_by_evidence[row["evidence_id"]])

    docs = {
        source_id: fitz.open(repo_root / meta["path"])
        for source_id, meta in source_documents.items()
    }
    pdf_lines = {
        source_id: _pdf_lines(doc)
        for source_id, doc in docs.items()
    }

    next_legal_id = _next_numeric_id(legal_units, "legal_unit_id")
    next_chunk_id = _next_numeric_id(chunks, "chunk_id")
    next_evidence_id = 1

    def allocate_legal_id() -> str:
        nonlocal next_legal_id
        value = f"uud_legal_unit_{next_legal_id:05d}"
        next_legal_id += 1
        return value

    def allocate_chunk_id() -> str:
        nonlocal next_chunk_id
        value = f"uud_chunk_{next_chunk_id:05d}"
        next_chunk_id += 1
        return value

    def allocate_evidence_id(source_role: str, slug: str) -> str:
        nonlocal next_evidence_id
        value = f"uud_instrument_final_citation_evidence::{source_role}::{next_evidence_id:05d}::{slug}"
        next_evidence_id += 1
        return value

    def trim_unit(source_document_id: str, unit_label: str, marker: str, *, hierarchy_suffix: tuple[str, ...] | None = None) -> None:
        unit = _find_unit(legal_units, source_document_id, unit_label, hierarchy_suffix=hierarchy_suffix)
        chunk = chunks_by_unit[unit["legal_unit_id"]]
        if marker not in unit["text"]:
            return
        trimmed = _trim_before(unit["text"], marker)
        unit["text"] = trimmed
        unit["page_start"], unit["page_end"] = _page_span_for_text(pages_by_source, source_document_id, trimmed, unit["page_start"], unit["page_end"])
        chunk["text"] = trimmed
        chunk["page_range"] = {"start_page_number": unit["page_start"], "end_page_number": unit["page_end"]}
        existing = evidence_by_unit.get(unit["legal_unit_id"])
        if existing:
            _rebuild_evidence(existing, trimmed, pdf_lines[source_document_id], source_documents[source_document_id], bbox_by_evidence)
            _rebuild_retrieval(existing, chunk, retrieval_units)

    def trim_bab(source_document_id: str, unit_label: str, marker: str) -> None:
        unit = units_by_source_label[(source_document_id, unit_label)]
        chunk = chunks_by_unit[unit["legal_unit_id"]]
        if marker not in unit["text"]:
            return
        trimmed = _trim_before(unit["text"], marker)
        unit["text"] = trimmed
        unit["page_start"], unit["page_end"] = _page_span_for_text(pages_by_source, source_document_id, trimmed, unit["page_start"], unit["page_end"])
        chunk["text"] = trimmed
        chunk["page_range"] = {"start_page_number": unit["page_start"], "end_page_number": unit["page_end"]}

    for spec in INSERTED_BAB_SPECS:
        source_id = spec["source_document_id"]
        page_text = pages_by_source[(source_id, spec["page_number"])]
        bab_text = _slice_between(page_text, spec["start"], spec["end"])
        for target in spec["trim_targets"]:
            if isinstance(target, tuple):
                trim_unit(source_id, target[-1], spec["label"], hierarchy_suffix=target)
            else:
                trim_unit(source_id, target, spec["label"])
        if spec["trim_bab"]:
            trim_bab(source_id, spec["trim_bab"], spec["label"])
        parent_ids = []
        if spec["parent_label"]:
            parent = units_by_source_label[(source_id, spec["parent_label"])]
            parent_ids.append(parent["legal_unit_id"])
        legal_unit_id = allocate_legal_id()
        chunk_id = allocate_chunk_id()
        source_meta = source_documents[source_id]
        legal_units.append({
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
        })
        chunks.append({
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
        })
        units_by_source_label[(source_id, spec["label"])] = legal_units[-1]
        for child_label in spec["child_labels"]:
            child = units_by_source_label[(source_id, child_label)]
            if legal_unit_id not in child["parent_legal_unit_ids"]:
                child["parent_legal_unit_ids"] = [legal_unit_id, *child["parent_legal_unit_ids"]]
            for unit in legal_units:
                if unit["source_document_id"] != source_id or unit["unit_type"] != "ayat_record":
                    continue
                if child["legal_unit_id"] in (unit.get("parent_legal_unit_ids") or ()) and legal_unit_id not in unit["parent_legal_unit_ids"]:
                    unit["parent_legal_unit_ids"] = [legal_unit_id, *unit["parent_legal_unit_ids"]]

    source_role = lambda source_id: source_id.split("::", 1)[1]

    def append_instrument_unit(
        source_id: str,
        unit_type: str,
        unit_label: str,
        text: str,
        page_start: int,
        page_end: int,
        *,
        hierarchy: list[str] | None = None,
        parent_legal_unit_ids: list[str] | None = None,
        chunk_type: str | None = None,
        canonical_use_allowed: bool = True,
        chunk_status: str = "active_canonical_record",
        runtime_loadable: bool | None = None,
        exclusion_ref: str | None = None,
        build_evidence: bool = True,
    ) -> str:
        legal_unit_id = allocate_legal_id()
        chunk_id = allocate_chunk_id()
        source_meta = source_documents[source_id]
        unit = {
            "corpus_id": "uud",
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
        evidence_id = allocate_evidence_id(source_role(source_id), _slug(unit_label or unit_type))
        bbox_records = _build_bbox_rows(
            evidence_id=evidence_id,
            source_meta=source_meta,
            source_id=source_id,
            text=text,
            page_start=page_start,
            page_end=page_end,
            line_entries=pdf_lines[source_id],
        )
        quoted_text = "\n".join(row["text"] for row in bbox_records)
        evidence_row = {
            "bbox_refs": [row["bbox_id"] for row in bbox_records],
            "bbox_precision": _aggregate_bbox_precision(bbox_records),
            "citation": unit_label,
            "corpus_id": "uud",
            "evidence_id": evidence_id,
            "hierarchy": hierarchy or ([unit_label] if unit_label else []),
            "legal_unit_id": legal_unit_id,
            "page_numbers": sorted({row["page_number"] for row in bbox_records}),
            "quoted_text": quoted_text,
            "source_document_id": source_id,
            "source_pdf": source_meta["filename"],
            "source_pdf_path": source_meta["path"],
            "source_role": source_role(source_id),
            "source_sha256": source_meta["sha256"],
            "status": "final",
            "temporal_context": source_role(source_id),
            "viewer_highlightable": any(row["viewer_highlightable"] for row in bbox_records),
        }
        evidence.append(evidence_row)
        bbox_rows.extend(bbox_records)
        bbox_by_evidence[evidence_id] = bbox_records
        retrieval_units.append({
            "bbox_sample_refs": [bbox_records[0]["bbox_id"]] if bbox_records else [],
            "bbox_total_count": len(bbox_records),
            "chunk_id": chunk_id,
            "corpus_id": "uud",
            "evidence_id": evidence_id,
            "legal_unit_id": legal_unit_id,
            "page_numbers": evidence_row["page_numbers"],
            "retrieval_unit_id": f"uud_retrieval_unit::{evidence_id}",
            "source_pdf_path": source_meta["path"],
            "source_role": source_role(source_id),
            "source_sha256": source_meta["sha256"],
            "status": "accepted",
            "temporal_context": source_role(source_id),
            "text": _retrieval_text(unit_label, hierarchy or [], quoted_text),
        })
        return legal_unit_id

    # Amendment 1
    source_id = "uud::amendment_1_historical"
    page1 = pages_by_source[(source_id, 1)]
    page3 = pages_by_source[(source_id, 3)]
    recital_text = _slice_before(page1, "Setelah mempelajari").strip()
    scope_text = _slice_between(page1, "Setelah mempelajari", "selengkapnya menjadi berbunyi sebagai berikut :").strip() + " selengkapnya menjadi berbunyi sebagai berikut :"
    append_instrument_unit(source_id, "amendment_recital_record", "Perubahan Pertama Recital", recital_text, 1, 1)
    append_instrument_unit(source_id, "amendment_scope_record", "Perubahan Pertama Scope", scope_text, 1, 1)
    trim_unit(source_id, "Pasal 21", "Naskah perubahan ini merupakan bagian tak terpisahkan")
    closing = _slice_between(page3, "Naskah perubahan ini merupakan", "Perubahan tersebut diputuskan").strip()
    decision, effective = _split_effective_clause(_slice_between(page3, "Perubahan tersebut diputuskan", "Ditetapkan di Jakarta").strip())
    determination = _slice_between(page3, "Ditetapkan di Jakarta", "MAJELIS PERMUSYAWARATAN RAKYAT").strip()
    signatory = page3[page3.index("MAJELIS PERMUSYAWARATAN RAKYAT"):].strip()
    append_instrument_unit(source_id, "instrument_closing_record", "Perubahan Pertama Closing", closing, 3, 3)
    append_instrument_unit(source_id, "decision_clause_record", "Perubahan Pertama Decision", decision, 3, 3)
    append_instrument_unit(source_id, "effective_clause_record", "Perubahan Pertama Effective", effective, 3, 3, build_evidence=False)
    append_instrument_unit(source_id, "determination_clause_record", "Perubahan Pertama Determination", determination, 3, 3)
    append_instrument_unit(source_id, "signatory_block_record", "Perubahan Pertama Signatories", signatory, 3, 3)

    # Amendment 2
    source_id = "uud::amendment_2_historical"
    page1 = pages_by_source[(source_id, 1)]
    page8 = pages_by_source[(source_id, 8)]
    recital_text = _slice_before(page1, "Setelah Mempelajari").strip()
    scope_text = _slice_between(page1, "Setelah Mempelajari", "sehingga selengkapnya berbunyi sebagai berikut :").strip() + " sehingga selengkapnya berbunyi sebagai berikut :"
    append_instrument_unit(source_id, "amendment_recital_record", "Perubahan Kedua Recital", recital_text, 1, 1)
    append_instrument_unit(source_id, "amendment_scope_record", "Perubahan Kedua Scope", scope_text, 1, 1)
    trim_bab(source_id, "BAB XV", "Ditetapkan di Jakarta")
    trim_unit(source_id, "Pasal 36C", "Ditetapkan di Jakarta")
    determination = _slice_between(page8, "Ditetapkan di Jakarta", "MAJELIS PERMUSYAWARATAN RAKYAT").strip()
    signatory = page8[page8.index("MAJELIS PERMUSYAWARATAN RAKYAT"):].strip()
    append_instrument_unit(source_id, "determination_clause_record", "Perubahan Kedua Determination", determination, 8, 8)
    append_instrument_unit(source_id, "signatory_block_record", "Perubahan Kedua Signatories", signatory, 8, 8)

    # Amendment 3
    source_id = "uud::amendment_3_historical"
    page1 = pages_by_source[(source_id, 1)]
    page8 = pages_by_source[(source_id, 8)]
    page9 = pages_by_source[(source_id, 9)]
    recital_text = _slice_before(page1, "Setelah mempelajari").strip()
    scope_text = _slice_between(page1, "Setelah mempelajari", "sehingga selengkapnya menjadi berbunyi sebagai berikut:").strip() + " sehingga selengkapnya menjadi berbunyi sebagai berikut:"
    append_instrument_unit(source_id, "amendment_recital_record", "Perubahan Ketiga Recital", recital_text, 1, 1)
    append_instrument_unit(source_id, "amendment_scope_record", "Perubahan Ketiga Scope", scope_text, 1, 1)
    trim_unit(source_id, "Pasal 24C", "Naskah perubahan ini merupakan bagian tak terpisahkan")
    trim_unit(source_id, "(6)", "Naskah perubahan ini merupakan bagian tak terpisahkan", hierarchy_suffix=("Pasal 24C", "(6)"))
    closing = _slice_between(page8, "Naskah perubahan ini merupakan", "Perubahan tersebut diputuskan").strip()
    decision, effective = _split_effective_clause(page8[page8.index("Perubahan tersebut diputuskan"):].strip())
    determination = _slice_between(page9, "Ditetapkan di Jakarta", "MAJELIS PERMUSYAWARATAN RAKYAT").strip()
    signatory = page9[page9.index("MAJELIS PERMUSYAWARATAN RAKYAT"):].strip()
    append_instrument_unit(source_id, "instrument_closing_record", "Perubahan Ketiga Closing", closing, 8, 8)
    append_instrument_unit(source_id, "decision_clause_record", "Perubahan Ketiga Decision", decision, 8, 8)
    append_instrument_unit(source_id, "effective_clause_record", "Perubahan Ketiga Effective", effective, 8, 8, build_evidence=False)
    append_instrument_unit(source_id, "determination_clause_record", "Perubahan Ketiga Determination", determination, 9, 9)
    append_instrument_unit(source_id, "signatory_block_record", "Perubahan Ketiga Signatories", signatory, 9, 9)

    # Amendment 4
    source_id = "uud::amendment_4_historical"
    page1 = pages_by_source[(source_id, 1)]
    page5 = pages_by_source[(source_id, 5)]
    page6 = pages_by_source[(source_id, 6)]
    recital_text = _slice_before(page1, "(a)").strip()
    scope_text = _slice_between(page1, "(a)", "berikut.").strip() + " berikut."
    append_instrument_unit(source_id, "amendment_recital_record", "Perubahan Keempat Recital", recital_text, 1, 1)
    scope_unit_id = append_instrument_unit(source_id, "amendment_scope_record", "Perubahan Keempat Scope", scope_text, 1, 1)
    for clause in ("(a)", "(b)", "(c)", "(d)", "(e)"):
        next_clause = {"(a)": "(b)", "(b)": "(c)", "(c)": "(d)", "(d)": "(e)", "(e)": "berikut."}[clause]
        clause_text = _slice_between(page1, clause, next_clause).strip()
        append_instrument_unit(
            source_id,
            "instrument_clause_record",
            f"Perubahan Keempat Clause {clause}",
            clause_text,
            1,
            1,
            hierarchy=["Perubahan Keempat Scope", clause],
            parent_legal_unit_ids=[scope_unit_id],
        )
    aturan_text = _slice_between(page5 + "\n" + page6, "ATURAN TAMBAHAN", "Perubahan tersebut diputuskan").strip()
    pasal_i_text = _slice_between(page5, "Pasal I", "Pasal III").strip()
    pasal_iii_text = _slice_between(page6, "Pasal III", "Perubahan tersebut diputuskan").strip()
    anomaly_ref = "source_typo_reference::uud_source_typo_reference_00001"
    aturan_unit_id = append_instrument_unit(
        source_id,
        "aturan_tambahan_record",
        "ATURAN TAMBAHAN source typo reference",
        "ATURAN TAMBAHAN\n" + aturan_text,
        5,
        6,
        hierarchy=["ATURAN TAMBAHAN"],
        chunk_type="aturan_section_context_record",
        canonical_use_allowed=False,
        chunk_status="inactive_source_typo_reference",
        runtime_loadable=False,
        exclusion_ref=anomaly_ref,
        build_evidence=False,
    )
    append_instrument_unit(
        source_id,
        "pasal_record",
        "Pasal I",
        "Pasal I\n" + pasal_i_text,
        5,
        5,
        hierarchy=["ATURAN TAMBAHAN", "Pasal I"],
        parent_legal_unit_ids=[aturan_unit_id],
        chunk_type="pasal_chunk_record",
        canonical_use_allowed=False,
        chunk_status="inactive_source_typo_reference",
        runtime_loadable=False,
        exclusion_ref=anomaly_ref,
        build_evidence=False,
    )
    append_instrument_unit(
        source_id,
        "pasal_record",
        "Pasal III",
        "Pasal III\n" + pasal_iii_text,
        6,
        6,
        hierarchy=["ATURAN TAMBAHAN", "Pasal III"],
        parent_legal_unit_ids=[aturan_unit_id],
        chunk_type="pasal_chunk_record",
        canonical_use_allowed=False,
        chunk_status="inactive_source_typo_reference",
        runtime_loadable=False,
        exclusion_ref=anomaly_ref,
        build_evidence=False,
    )
    decision, effective = _split_effective_clause(_slice_between(page6, "Perubahan tersebut diputuskan", "Ditetapkan di Jakarta").strip())
    determination = _slice_between(page6, "Ditetapkan di Jakarta", "MAJELIS PERMUSYAWARATAN RAKYAT").strip()
    signatory = page6[page6.index("MAJELIS PERMUSYAWARATAN RAKYAT"):].strip()
    append_instrument_unit(source_id, "decision_clause_record", "Perubahan Keempat Decision", decision, 6, 6)
    append_instrument_unit(source_id, "effective_clause_record", "Perubahan Keempat Effective", effective, 6, 6, build_evidence=False)
    append_instrument_unit(source_id, "determination_clause_record", "Perubahan Keempat Determination", determination, 6, 6)
    append_instrument_unit(source_id, "signatory_block_record", "Perubahan Keempat Signatories", signatory, 6, 6)

    document_metadata, metadata_grounding, metadata_grounding_registry = _rebuild_metadata_grounding(
        document_metadata=document_metadata,
        metadata_grounding=metadata_grounding,
        metadata_grounding_registry=metadata_grounding_registry,
        evidence=evidence,
        legal_units=legal_units,
        source_conflicts=source_conflicts,
    )

    bbox_rows = [
        row
        for evidence_id in sorted(bbox_by_evidence)
        for row in bbox_by_evidence[evidence_id]
    ]
    _apply_inserted_bab_heading_bbox_policy(bbox_rows, evidence)
    bbox_rows.sort(key=lambda row: (row["source_document_id"], row["page_number"], row["bbox_id"]))
    legal_units.sort(key=lambda row: row["legal_unit_id"])
    chunks.sort(key=lambda row: row["chunk_id"])
    evidence.sort(key=lambda row: row["evidence_id"])
    retrieval_units.sort(key=lambda row: row["retrieval_unit_id"])

    _write_jsonl(final_dir / "legal_units.jsonl", legal_units)
    _write_jsonl(final_dir / "chunks.jsonl", chunks)
    _write_jsonl(final_dir / "evidence_registry.jsonl", evidence)
    _write_jsonl(final_dir / "bbox_registry.jsonl", bbox_rows)
    _write_jsonl(final_dir / "retrieval_units.jsonl", retrieval_units)
    _write_jsonl(final_dir / "document_metadata.jsonl", document_metadata)
    _write_jsonl(final_dir / "metadata_grounding.jsonl", metadata_grounding)
    _write_jsonl(final_dir / "metadata_grounding_registry.jsonl", metadata_grounding_registry)

    validation_report["final_artifact_counts"] = {
        "chunks": len(chunks),
        "legal_units": len(legal_units),
        "excluded_records": len(read_jsonl(final_dir / "excluded_records.jsonl")),
        "evidence_records": len(evidence),
        "bbox_records": len(bbox_rows),
        "retrieval_units": len(retrieval_units),
    }
    validation_report["bbox_precision_counts"] = _bbox_precision_counts(bbox_rows)
    validation_report["bbox_highlightability_counts"] = {
        "viewer_highlightable": sum(1 for row in bbox_rows if row.get("viewer_highlightable") is True),
        "non_highlightable": sum(1 for row in bbox_rows if row.get("viewer_highlightable") is not True),
    }
    validation_report.setdefault("instrument_baseline", {})
    validation_report["instrument_baseline"] = {
        "status": "corrected",
        "instrument_unit_types": [
            "amendment_recital_record",
            "amendment_scope_record",
            "instrument_clause_record",
            "instrument_closing_record",
            "decision_clause_record",
            "effective_clause_record",
            "determination_clause_record",
            "signatory_block_record",
        ],
        "metadata_viewer_highlightable": False,
    }
    validation_report["bbox_precision_policy"] = {
        "status": "corrected",
        "exact_policy": "bbox_precision=exact rows may remain viewer_highlightable",
        "fallback_policy": "bbox_precision=page_grounded_only rows are not viewer_highlightable",
        "coarse_policy": "bbox_precision=coarse rows are not viewer_highlightable",
    }
    validation_report["metadata_grounding_contract"] = {
        "status": "field_grounded",
        "note": "field-level metadata grounding preserves block-level rows and keeps metadata viewer highlights fail-closed unless exact accepted support exists",
    }
    validation_report.setdefault("structure_fidelity", {})
    validation_report["structure_fidelity"]["inserted_bab_heading_owner_policy"] = (
        "inserted heading bboxes may stay exact, but they are viewer_highlightable only when owned by bab_record evidence"
    )
    (final_dir / "validation_report.json").write_text(json.dumps(validation_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    _refresh_manifest(final_dir, manifest)
    return {
        "legal_units": len(legal_units),
        "chunks": len(chunks),
        "evidence": len(evidence),
        "bbox": len(bbox_rows),
        "retrieval_units": len(retrieval_units),
    }


def validate_uud_artifact_baseline(repo_root: Path) -> tuple[str, ...]:
    final_dir = (repo_root / FINAL_DIR).resolve()
    legal_units = read_jsonl(final_dir / "legal_units.jsonl")
    chunks = read_jsonl(final_dir / "chunks.jsonl")
    evidence = read_jsonl(final_dir / "evidence_registry.jsonl")
    bbox_rows = read_jsonl(final_dir / "bbox_registry.jsonl")
    retrieval_units = read_jsonl(final_dir / "retrieval_units.jsonl")
    metadata_grounding = read_jsonl(final_dir / "metadata_grounding.jsonl")

    errors: list[str] = []
    seen_ids: dict[str, set[str]] = defaultdict(set)
    units_by_id = {row["legal_unit_id"]: row for row in legal_units}
    chunks_by_id = {row["chunk_id"]: row for row in chunks}
    evidence_by_id = {row["evidence_id"]: row for row in evidence}
    bbox_by_id = {row["bbox_id"]: row for row in bbox_rows}
    bbox_by_evidence: dict[str, list[dict]] = defaultdict(list)
    for row in bbox_rows:
        bbox_by_evidence[row["evidence_id"]].append(row)

    expected_instrument_labels = {
        "Perubahan Pertama Recital",
        "Perubahan Pertama Scope",
        "Perubahan Kedua Recital",
        "Perubahan Kedua Scope",
        "Perubahan Ketiga Recital",
        "Perubahan Ketiga Scope",
        "Perubahan Keempat Recital",
        "Perubahan Keempat Scope",
        "Perubahan Keempat Clause (a)",
        "Perubahan Keempat Clause (b)",
        "Perubahan Keempat Clause (c)",
        "Perubahan Keempat Clause (d)",
        "Perubahan Keempat Clause (e)",
    }
    actual_labels = {row.get("unit_label") for row in legal_units}
    for label in expected_instrument_labels:
        if label not in actual_labels:
            errors.append(f"missing_instrument_unit:{label}")

    forbidden_markers = ("Naskah perubahan ini merupakan", "Perubahan tersebut diputuskan", "Ditetapkan di Jakarta")
    for row in legal_units:
        if row["legal_unit_id"] in seen_ids["legal_unit_id"]:
            errors.append(f"duplicate_legal_unit_id:{row['legal_unit_id']}")
        seen_ids["legal_unit_id"].add(row["legal_unit_id"])
        if row["unit_type"] in {"pasal_record", "ayat_record"}:
            if any(marker in row["text"] for marker in forbidden_markers):
                errors.append(f"closing_clause_attached_to_normative_unit:{row['legal_unit_id']}")
            if any(marker in row["text"] for marker in ("BAB VIIA", "BAB VIIB", "BAB VIIIA", "BAB IXA", "BAB XA")):
                errors.append(f"inserted_bab_inside_normative_unit:{row['legal_unit_id']}")
        if row["unit_type"] == "pasal_record" and row.get("hierarchy") and str(row["hierarchy"][0]).startswith("BAB"):
            if not any(units_by_id[parent]["unit_type"] == "bab_record" for parent in row.get("parent_legal_unit_ids") or () if parent in units_by_id):
                errors.append(f"pasal_missing_bab_parent:{row['legal_unit_id']}")
        if row["unit_type"] == "bab_record" and any(marker in row["text"] for marker in STRUCTURAL_FORBIDDEN_MARKERS):
            errors.append(f"structural_bab_contains_instrument_text:{row['legal_unit_id']}")

    for row in chunks:
        if row["chunk_id"] in seen_ids["chunk_id"]:
            errors.append(f"duplicate_chunk_id:{row['chunk_id']}")
        seen_ids["chunk_id"].add(row["chunk_id"])
        if row["legal_unit_id"] not in units_by_id:
            errors.append(f"orphan_chunk:{row['chunk_id']}")
        if row["chunk_type"] == "bab_structural_context_record" and any(marker in row["text"] for marker in STRUCTURAL_FORBIDDEN_MARKERS):
            errors.append(f"structural_chunk_contains_instrument_text:{row['chunk_id']}")
    for row in evidence:
        if row["evidence_id"] in seen_ids["evidence_id"]:
            errors.append(f"duplicate_evidence_id:{row['evidence_id']}")
        seen_ids["evidence_id"].add(row["evidence_id"])
        if row["legal_unit_id"] not in units_by_id:
            errors.append(f"orphan_evidence:{row['evidence_id']}")
        for bbox_id in row.get("bbox_refs") or ():
            if bbox_id not in bbox_by_id:
                errors.append(f"orphan_bbox_ref:{row['evidence_id']}:{bbox_id}")
            elif bbox_by_id[bbox_id]["page_number"] not in row["page_numbers"]:
                errors.append(f"bbox_page_mismatch:{row['evidence_id']}:{bbox_id}")
        if row.get("citation") in DECISION_LABELS:
            for bbox_id in row.get("bbox_refs") or ():
                bbox_row = bbox_by_id.get(bbox_id)
                if not bbox_row:
                    continue
                if bbox_row.get("viewer_highlightable") and bbox_row.get("bbox_precision") != "exact":
                    errors.append(f"decision_bbox_not_exact:{row['evidence_id']}:{bbox_id}")
                if bbox_row.get("viewer_highlightable") and "Pasal " in bbox_row.get("text", ""):
                    errors.append(f"decision_bbox_contains_normative_text:{row['evidence_id']}:{bbox_id}")
                if bbox_row.get("bbox_precision") in {"coarse", "page_grounded_only"} and bbox_row.get("viewer_highlightable"):
                    errors.append(f"coarse_bbox_marked_highlightable:{bbox_id}")
    for row in bbox_rows:
        if row["bbox_id"] in seen_ids["bbox_id"]:
            errors.append(f"duplicate_bbox_id:{row['bbox_id']}")
        seen_ids["bbox_id"].add(row["bbox_id"])
        evidence_row = evidence_by_id.get(row["evidence_id"])
        owner = units_by_id.get(evidence_row["legal_unit_id"]) if evidence_row else None
        if (
            INSERTED_BAB_HEADING_BBOX_MARKER in row["bbox_id"]
            and owner is not None
            and owner.get("unit_type") != "bab_record"
            and row.get("viewer_highlightable")
        ):
            errors.append(f"inserted_bab_heading_highlightable_without_bab_owner:{row['bbox_id']}")
        if row.get("bbox_precision") not in {"exact", "coarse", "page_grounded_only"}:
            errors.append(f"invalid_bbox_precision:{row['bbox_id']}")
        if row.get("bbox_precision") in {"coarse", "page_grounded_only"} and row.get("viewer_highlightable"):
            errors.append(f"coarse_bbox_marked_highlightable:{row['bbox_id']}")
    for row in retrieval_units:
        if row["retrieval_unit_id"] in seen_ids["retrieval_unit_id"]:
            errors.append(f"duplicate_retrieval_unit_id:{row['retrieval_unit_id']}")
        seen_ids["retrieval_unit_id"].add(row["retrieval_unit_id"])
        if row["chunk_id"] not in chunks_by_id:
            errors.append(f"orphan_retrieval_chunk:{row['retrieval_unit_id']}")
        if row["evidence_id"] not in evidence_by_id:
            errors.append(f"orphan_retrieval_evidence:{row['retrieval_unit_id']}")
    for row in metadata_grounding:
        if row.get("viewer_highlightable") is not False:
            errors.append(f"metadata_grounding_highlightable_not_clarified:{row['metadata_grounding_id']}")
        if row.get("bbox_precision") not in {None, "coarse", "exact", "page_grounded_only"}:
            errors.append(f"invalid_metadata_bbox_precision:{row['metadata_grounding_id']}")

    return tuple(sorted(set(errors)))


def _refresh_manifest(final_dir: Path, manifest: dict) -> None:
    counts = manifest.setdefault("counts", {})
    counts["document_metadata"] = _count_jsonl(final_dir / "document_metadata.jsonl")
    counts["legal_units"] = _count_jsonl(final_dir / "legal_units.jsonl")
    counts["chunks"] = _count_jsonl(final_dir / "chunks.jsonl")
    counts["evidence_records"] = _count_jsonl(final_dir / "evidence_registry.jsonl")
    counts["bbox_records"] = _count_jsonl(final_dir / "bbox_registry.jsonl")
    counts["metadata_grounding"] = _count_jsonl(final_dir / "metadata_grounding.jsonl")
    counts["metadata_grounding_records"] = _count_jsonl(final_dir / "metadata_grounding_registry.jsonl")
    counts["retrieval_units"] = _count_jsonl(final_dir / "retrieval_units.jsonl")
    for rel in manifest["files"]:
        path = final_dir / rel
        if path.exists():
            manifest["files"][rel]["bytes"] = path.stat().st_size
            manifest["files"][rel]["sha256"] = file_sha256(path)
    (final_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _count_jsonl(path: Path) -> int:
    return len(read_jsonl(path))


def _next_numeric_id(rows: list[dict], key: str) -> int:
    max_value = 0
    for row in rows:
        value = _numeric_suffix(str(row.get(key, "")))
        if value:
            max_value = max(max_value, value)
    return max_value + 1


def _numeric_suffix(value: str) -> int:
    match = re.search(r"(\d+)$", value)
    return int(match.group(1)) if match else 0


def _find_unit(
    legal_units: list[dict],
    source_document_id: str,
    unit_label: str,
    *,
    hierarchy_suffix: tuple[str, ...] | None = None,
) -> dict:
    candidates = [
        row
        for row in legal_units
        if row["source_document_id"] == source_document_id and row.get("unit_label") == unit_label
    ]
    if hierarchy_suffix is not None:
        compact_suffix = tuple(_compact(part) for part in hierarchy_suffix)
        candidates = [
            row
            for row in candidates
            if tuple(_compact(part) for part in [*(row.get("hierarchy") or ()), row.get("unit_label")])[-len(compact_suffix):] == compact_suffix
        ]
    if len(candidates) != 1:
        raise KeyError(f"unable_to_resolve_unit:{source_document_id}:{unit_label}:{hierarchy_suffix}")
    return candidates[0]


def _slice_before(text: str, marker: str) -> str:
    return text[:text.index(marker)]


def _slice_between(text: str, start: str, end: str) -> str:
    start_index = text.index(start)
    end_index = text.index(end, start_index + len(start))
    return text[start_index:end_index].strip()


def _trim_before(text: str, marker: str) -> str:
    return text[:text.index(marker)].rstrip() + "\n"


def _split_effective_clause(text: str) -> tuple[str, str]:
    marker = ", dan mulai berlaku"
    if marker not in text:
        return text, "dan mulai berlaku pada tanggal ditetapkan."
    head, tail = text.split(marker, 1)
    return head.strip() + ".", ("dan mulai berlaku" + tail).strip()


def _page_span_for_text(pages_by_source: dict[tuple[str, int], str], source_id: str, text: str, page_start: int, page_end: int) -> tuple[int, int]:
    for page_number in range(page_start, page_end + 1):
        if _compact(text) in _compact(pages_by_source[(source_id, page_number)]):
            return page_number, page_number
    return page_start, page_end


def _rebuild_evidence(existing: dict, text: str, line_entries: dict[int, list[dict]], source_meta: dict, bbox_by_evidence: dict[str, list[dict]]) -> None:
    bbox_records = _build_bbox_rows(
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
    existing["bbox_precision"] = _aggregate_bbox_precision(bbox_records)
    existing["viewer_highlightable"] = any(row["viewer_highlightable"] for row in bbox_records)
    bbox_by_evidence[existing["evidence_id"]] = bbox_records


def _rebuild_retrieval(existing: dict, chunk: dict, retrieval_units: list[dict]) -> None:
    quoted = existing["quoted_text"]
    for row in retrieval_units:
        if row["evidence_id"] == existing["evidence_id"]:
            row["page_numbers"] = existing["page_numbers"]
            row["bbox_sample_refs"] = existing["bbox_refs"][:1]
            row["bbox_total_count"] = len(existing["bbox_refs"])
            row["text"] = _retrieval_text(existing["citation"], existing.get("hierarchy") or [], quoted)
            chunk["text"] = quoted if chunk["status"] == "active_canonical_record" else chunk["text"]
            break


def _retrieval_text(citation: str | None, hierarchy: list[str] | tuple[str, ...], quoted_text: str) -> str:
    prefix = " ".join([item for item in [citation, *hierarchy] if item])
    return f"{prefix}\n{quoted_text}".strip()


def _build_bbox_rows(
    *,
    evidence_id: str,
    source_meta: dict,
    source_id: str,
    text: str,
    page_start: int,
    page_end: int,
    line_entries: dict[int, list[dict]],
) -> list[dict]:
    expected = [_compact(line) for line in text.splitlines() if line.strip()]
    matched = []
    target_index = 0
    for page_number in range(page_start, page_end + 1):
        candidates = line_entries.get(page_number, [])
        for candidate in candidates:
            if target_index >= len(expected):
                break
            if _compact(candidate["text"]) != expected[target_index]:
                continue
            matched.append({
                "page_number": page_number,
                **candidate,
            })
            target_index += 1
        if target_index >= len(expected):
            break
    if target_index < len(expected):
        return _fallback_bbox_rows(
            evidence_id=evidence_id,
            source_meta=source_meta,
            source_id=source_id,
            text=text,
            page_start=page_start,
            page_end=page_end,
            line_entries=line_entries,
        )
    rows = []
    for index, row in enumerate(matched):
        rows.append({
            "bbox_id": f"uud_unified_bbox::{evidence_id}::{index:04d}",
            "bbox_precision": "exact",
            "corpus_id": "uud",
            "evidence_id": evidence_id,
            "page_number": row["page_number"],
            "source_document_id": source_id,
            "source_pdf": source_meta["filename"],
            "source_pdf_path": source_meta["path"],
            "source_sha256": source_meta["sha256"],
            "status": "accepted",
            "text": row["text"],
            "viewer_highlightable": True,
            "x0": row["x0"],
            "x1": row["x1"],
            "y0": row["y0"],
            "y1": row["y1"],
        })
    return rows


def _pdf_lines(doc) -> dict[int, list[dict]]:
    pages: dict[int, list[dict]] = {}
    for page_number in range(1, doc.page_count + 1):
        page = doc[page_number - 1]
        entries = []
        for block in page.get_text("dict").get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                text = "".join(span.get("text", "") for span in line.get("spans", [])).strip()
                if not text:
                    continue
                x0 = min(span["bbox"][0] for span in line.get("spans", []))
                y0 = min(span["bbox"][1] for span in line.get("spans", []))
                x1 = max(span["bbox"][2] for span in line.get("spans", []))
                y1 = max(span["bbox"][3] for span in line.get("spans", []))
                entries.append({"text": text, "x0": x0, "y0": y0, "x1": x1, "y1": y1})
        pages[page_number] = entries
    return pages


def _compact(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").replace("\u00ad", "").replace("Â", "")
    return re.sub(r"\s+", " ", text).strip().casefold()


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (text or "").casefold()).strip("_")


def _fallback_bbox_rows(
    *,
    evidence_id: str,
    source_meta: dict,
    source_id: str,
    text: str,
    page_start: int,
    page_end: int,
    line_entries: dict[int, list[dict]],
) -> list[dict]:
    rows = []
    for index, page_number in enumerate(range(page_start, page_end + 1)):
        candidates = line_entries.get(page_number, [])
        if not candidates:
            continue
        rows.append({
            "bbox_id": f"uud_unified_bbox::{evidence_id}::{index:04d}",
            "bbox_precision": "page_grounded_only",
            "corpus_id": "uud",
            "evidence_id": evidence_id,
            "page_number": page_number,
            "source_document_id": source_id,
            "source_pdf": source_meta["filename"],
            "source_pdf_path": source_meta["path"],
            "source_sha256": source_meta["sha256"],
            "status": "accepted",
            "text": text.strip() if index == 0 else "",
            "viewer_highlightable": False,
            "x0": min(row["x0"] for row in candidates),
            "x1": max(row["x1"] for row in candidates),
            "y0": min(row["y0"] for row in candidates),
            "y1": max(row["y1"] for row in candidates),
        })
    if not rows:
        raise ValueError(f"unable_to_build_bbox_rows:{source_id}:{text[:80]}")
    return rows


def _aggregate_bbox_precision(rows: list[dict]) -> str:
    precisions = {row.get("bbox_precision") for row in rows}
    if precisions == {"exact"}:
        return "exact"
    if "page_grounded_only" in precisions:
        return "page_grounded_only"
    return "coarse"


def _apply_inserted_bab_heading_bbox_policy(bbox_rows: list[dict], evidence: list[dict]) -> None:
    evidence_by_id = {row["evidence_id"]: row for row in evidence}
    for row in bbox_rows:
        if INSERTED_BAB_HEADING_BBOX_MARKER not in row["bbox_id"]:
            continue
        evidence_row = evidence_by_id.get(row["evidence_id"])
        if evidence_row and evidence_row.get("citation") == row.get("text"):
            continue
        row["viewer_highlightable"] = False
    by_evidence: dict[str, list[dict]] = defaultdict(list)
    for row in bbox_rows:
        by_evidence[row["evidence_id"]].append(row)
    for row in evidence:
        bbox_records = by_evidence.get(row["evidence_id"], [])
        if not bbox_records:
            continue
        row["bbox_precision"] = _aggregate_bbox_precision(bbox_records)
        row["viewer_highlightable"] = any(item.get("viewer_highlightable") is True for item in bbox_records)


def _bbox_precision_counts(bbox_rows: list[dict]) -> dict[str, int]:
    return {
        "exact": sum(1 for row in bbox_rows if row.get("bbox_precision") == "exact"),
        "coarse": sum(1 for row in bbox_rows if row.get("bbox_precision") == "coarse"),
        "page_grounded_only": sum(1 for row in bbox_rows if row.get("bbox_precision") == "page_grounded_only"),
    }


def _rebuild_metadata_grounding(
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
    block_rows = [row | {"bbox_precision": "page_grounded_only", "viewer_highlightable": False} for row in metadata_grounding]
    block_registry_rows = list(metadata_grounding_registry)
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
            determination = evidence_by_key.get((role, f"Perubahan {_ordinal_label(role)} Determination"))
            decision = evidence_by_key.get((role, f"Perubahan {_ordinal_label(role)} Decision"))
            signatories = evidence_by_key.get((role, f"Perubahan {_ordinal_label(role)} Signatories"))
            effective = units_by_key.get((role, f"Perubahan {_ordinal_label(role)} Effective"))
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
