from __future__ import annotations

import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

from tjipto.corpora.uud.anomaly_builder import append_amendment_instrument_units
from tjipto.corpora.uud.bbox_builder import (
    aggregate_bbox_precision,
    apply_inserted_bab_heading_bbox_policy,
    pdf_lines,
)
from tjipto.corpora.uud.compatibility_seed import load_compatibility_seed
from tjipto.corpora.uud.evidence_builder import append_instrument_unit as append_instrument_record, rebuild_evidence
from tjipto.corpora.uud.graph_builder import build_graph_artifacts
from tjipto.corpora.uud.manifest import build_manifest, refresh_manifest, write_json, write_jsonl
from tjipto.corpora.uud.metadata_builder import build_document_metadata, build_metadata_assertions, build_metadata_block_grounding, build_metadata_graph_edges, rebuild_metadata_grounding
from tjipto.corpora.uud.pages_builder import build_pages
from tjipto.corpora.uud.pipeline import run_staged_uud_pipeline
from tjipto.corpora.uud.retrieval_builder import apply_chunk_grounding, rebuild_retrieval
from tjipto.corpora.uud.specs import EXCLUDED_RECORD_SPECS, FINAL_DIR, INSERTED_BAB_SPECS
from tjipto.corpora.uud.source_conflict_builder import apply_source_conflict_grounding, build_source_conflicts
from tjipto.corpora.uud.source_documents_builder import build_source_documents
from tjipto.corpora.uud.structure_builder import (
    apply_inserted_bab_specs,
    find_unit,
    next_numeric_id,
    page_span_for_text,
    slice_before,
    slice_between,
    split_effective_clause,
    trim_before,
)
from tjipto.corpora.uud.text_span_builder import build_page_text_spans
from tjipto.corpora.uud.validation import update_validation_report, validate_uud_artifact_dir


def rebuild_uud_artifact_baseline(repo_root: Path) -> dict:
    final_dir = (repo_root / FINAL_DIR).resolve()
    result: dict = {}

    def build(stage_dir: Path) -> None:
        nonlocal result
        result = _rebuild_uud_artifact_baseline_at(repo_root, stage_dir)

    run_staged_uud_pipeline(
        final_dir,
        build,
        validate_uud_artifact_dir,
    )
    return result


def _rebuild_uud_artifact_baseline_at(repo_root: Path, final_dir: Path) -> dict:
    try:
        import fitz
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("PyMuPDF is required to rebuild UUD artifacts") from error

    seed = load_compatibility_seed(final_dir)
    legal_units = seed["legal_units"]
    chunks = seed["chunks"]
    evidence = seed["evidence"]
    bbox_rows = seed["bbox_rows"]
    retrieval_units = seed["retrieval_units"]
    validation_report = seed["validation_report"]

    excluded_records = deepcopy(list(EXCLUDED_RECORD_SPECS))
    source_documents = {row["source_document_id"]: row for row in build_source_documents(repo_root)}
    manifest = build_manifest(source_documents)
    source_conflicts = build_source_conflicts()
    pages = build_pages(repo_root, source_documents)
    pages_by_source = {(row["source_document_id"], row["page_number"]): row["text"] for row in pages}
    metadata_grounding = build_metadata_block_grounding(
        pages_by_source=pages_by_source,
        source_documents=source_documents,
    )
    document_metadata = build_document_metadata(source_documents, metadata_grounding)
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
            row["bbox_precision"] = aggregate_bbox_precision(bbox_by_evidence[row["evidence_id"]])
            row["viewer_highlightable"] = any(item["viewer_highlightable"] for item in bbox_by_evidence[row["evidence_id"]])

    docs = {
        source_id: fitz.open(repo_root / meta["path"])
        for source_id, meta in source_documents.items()
    }
    pdf_lines_by_source = {
        source_id: pdf_lines(doc)
        for source_id, doc in docs.items()
    }
    page_text_spans = build_page_text_spans(source_documents=source_documents, pdf_lines=pdf_lines_by_source)

    next_legal_id = next_numeric_id(legal_units, "legal_unit_id")
    next_chunk_id = next_numeric_id(chunks, "chunk_id")
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
        unit = find_unit(legal_units, source_document_id, unit_label, hierarchy_suffix=hierarchy_suffix)
        chunk = chunks_by_unit[unit["legal_unit_id"]]
        if marker not in unit["text"]:
            return
        trimmed = trim_before(unit["text"], marker)
        unit["text"] = trimmed
        unit["page_start"], unit["page_end"] = page_span_for_text(pages_by_source, source_document_id, trimmed, unit["page_start"], unit["page_end"])
        chunk["text"] = trimmed
        chunk["page_range"] = {"start_page_number": unit["page_start"], "end_page_number": unit["page_end"]}
        existing = evidence_by_unit.get(unit["legal_unit_id"])
        if existing:
            rebuild_evidence(existing, trimmed, pdf_lines_by_source[source_document_id], source_documents[source_document_id], bbox_by_evidence)
            rebuild_retrieval(existing, chunk, retrieval_units)

    def trim_bab(source_document_id: str, unit_label: str, marker: str) -> None:
        unit = units_by_source_label[(source_document_id, unit_label)]
        chunk = chunks_by_unit[unit["legal_unit_id"]]
        if marker not in unit["text"]:
            return
        trimmed = trim_before(unit["text"], marker)
        unit["text"] = trimmed
        unit["page_start"], unit["page_end"] = page_span_for_text(pages_by_source, source_document_id, trimmed, unit["page_start"], unit["page_end"])
        chunk["text"] = trimmed
        chunk["page_range"] = {"start_page_number": unit["page_start"], "end_page_number": unit["page_end"]}

    apply_inserted_bab_specs(
        specs=INSERTED_BAB_SPECS,
        pages_by_source=pages_by_source,
        source_documents=source_documents,
        legal_units=legal_units,
        chunks=chunks,
        units_by_source_label=units_by_source_label,
        trim_unit=trim_unit,
        trim_bab=trim_bab,
        allocate_legal_id=allocate_legal_id,
        allocate_chunk_id=allocate_chunk_id,
    )

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
        return append_instrument_record(
            source_id=source_id,
            unit_type=unit_type,
            unit_label=unit_label,
            text=text,
            page_start=page_start,
            page_end=page_end,
            source_documents=source_documents,
            pdf_lines_by_source=pdf_lines_by_source,
            legal_units=legal_units,
            chunks=chunks,
            evidence=evidence,
            bbox_rows=bbox_rows,
            bbox_by_evidence=bbox_by_evidence,
            retrieval_units=retrieval_units,
            allocate_legal_id=allocate_legal_id,
            allocate_chunk_id=allocate_chunk_id,
            allocate_evidence_id=allocate_evidence_id,
            hierarchy=hierarchy,
            parent_legal_unit_ids=parent_legal_unit_ids,
            chunk_type=chunk_type,
            canonical_use_allowed=canonical_use_allowed,
            chunk_status=chunk_status,
            runtime_loadable=runtime_loadable,
            exclusion_ref=exclusion_ref,
            build_evidence=build_evidence,
        )

    append_amendment_instrument_units(
        pages_by_source=pages_by_source,
        append_instrument_unit=append_instrument_unit,
        trim_unit=trim_unit,
        trim_bab=trim_bab,
    )

    document_metadata, metadata_grounding, metadata_grounding_registry = rebuild_metadata_grounding(
        document_metadata=document_metadata,
        metadata_grounding=metadata_grounding,
        evidence=evidence,
        bbox_rows=bbox_rows,
        legal_units=legal_units,
        page_text_spans=page_text_spans,
        source_conflicts=source_conflicts,
    )

    bbox_rows = [
        row
        for evidence_id in sorted(bbox_by_evidence)
        for row in bbox_by_evidence[evidence_id]
    ]
    apply_inserted_bab_heading_bbox_policy(bbox_rows, evidence)
    apply_source_conflict_grounding(source_conflicts, evidence, bbox_rows, page_text_spans)
    bbox_rows.sort(key=lambda row: (row["source_document_id"], row["page_number"], row["bbox_id"]))
    metadata_assertions = build_metadata_assertions(evidence, metadata_grounding, bbox_rows)
    legal_units.sort(key=lambda row: row["legal_unit_id"])
    chunks.sort(key=lambda row: row["chunk_id"])
    evidence.sort(key=lambda row: row["evidence_id"])
    retrieval_units.sort(key=lambda row: row["retrieval_unit_id"])
    apply_chunk_grounding(chunks, legal_units, evidence, page_text_spans)
    metadata_graph_edges = build_metadata_graph_edges(metadata_assertions)
    graph_nodes, graph_edges = build_graph_artifacts(
        source_documents=list(source_documents.values()),
        pages=pages,
        legal_units=legal_units,
        evidence=evidence,
        bbox_rows=bbox_rows,
        excluded_records=excluded_records,
        source_conflicts=source_conflicts,
        metadata_grounding=metadata_grounding,
    )
    write_jsonl(final_dir / "legal_units.jsonl", legal_units)
    write_jsonl(final_dir / "chunks.jsonl", chunks)
    write_jsonl(final_dir / "evidence_registry.jsonl", evidence)
    write_jsonl(final_dir / "bbox_registry.jsonl", bbox_rows)
    write_jsonl(final_dir / "retrieval_units.jsonl", retrieval_units)
    write_jsonl(final_dir / "document_metadata.jsonl", document_metadata)
    write_jsonl(final_dir / "metadata_grounding.jsonl", metadata_grounding)
    write_jsonl(final_dir / "metadata_grounding_registry.jsonl", metadata_grounding_registry)
    write_jsonl(final_dir / "metadata.jsonl", metadata_assertions)
    write_jsonl(final_dir / "metadata_graph_edges.jsonl", metadata_graph_edges)
    write_jsonl(final_dir / "source_conflicts.jsonl", source_conflicts)
    write_jsonl(final_dir / "source_documents.jsonl", list(source_documents.values()))
    write_jsonl(final_dir / "pages.jsonl", pages)
    write_jsonl(final_dir / "graph_nodes.jsonl", graph_nodes)
    write_jsonl(final_dir / "graph_edges.jsonl", graph_edges)
    write_jsonl(final_dir / "page_text_spans.jsonl", page_text_spans)

    update_validation_report(
        validation_report,
        chunks=chunks,
        legal_units=legal_units,
        excluded_records=excluded_records,
        evidence=evidence,
        bbox_rows=bbox_rows,
        retrieval_units=retrieval_units,
        graph_nodes=graph_nodes,
        graph_edges=graph_edges,
        page_text_spans=page_text_spans,
    )
    write_json(final_dir / "validation_report.json", validation_report)

    refresh_manifest(final_dir, manifest)
    return {
        "legal_units": len(legal_units),
        "chunks": len(chunks),
        "evidence": len(evidence),
        "bbox": len(bbox_rows),
        "retrieval_units": len(retrieval_units),
        "graph_nodes": len(graph_nodes),
        "graph_edges": len(graph_edges),
    }


def validate_uud_artifact_baseline(repo_root: Path) -> tuple[str, ...]:
    final_dir = (repo_root / FINAL_DIR).resolve()
    return validate_uud_artifact_dir(final_dir)
