from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from tjipto.corpora.uud.anomaly_builder import append_amendment_instrument_units
from tjipto.corpora.uud.bbox_builder import (
    aggregate_bbox_precision,
    apply_inserted_bab_heading_bbox_policy,
    pdf_lines,
)
from tjipto.corpora.uud.evidence_builder import append_instrument_unit as append_instrument_record, rebuild_evidence
from tjipto.corpora.uud.graph_builder import build_graph_artifacts
from tjipto.corpora.uud.manifest import refresh_manifest, write_json, write_jsonl
from tjipto.corpora.uud.metadata_builder import rebuild_metadata_grounding, repair_metadata_graph_edges
from tjipto.corpora.uud.pipeline import run_staged_uud_pipeline
from tjipto.corpora.uud.retrieval_builder import apply_chunk_grounding, rebuild_retrieval
from tjipto.corpora.uud.specs import FINAL_DIR, INSERTED_BAB_SPECS
from tjipto.corpora.uud.source_conflict_builder import apply_source_conflict_grounding
from tjipto.corpora.uud.structure_builder import (
    apply_inserted_bab_specs,
    find_unit,
    next_numeric_id,
    numeric_suffix,
    page_span_for_text,
    slice_before,
    slice_between,
    split_effective_clause,
    trim_before,
)
from tjipto.corpora.uud.text_span_builder import build_page_text_spans
from tjipto.corpora.uud.validation import update_validation_report, validate_uud_artifact_dir
from tjipto.core.manifest import read_json, read_jsonl


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
    metadata_assertions = read_jsonl(final_dir / "metadata.jsonl")
    metadata_graph_edges = read_jsonl(final_dir / "metadata_graph_edges.jsonl")
    excluded_records = read_jsonl(final_dir / "excluded_records.jsonl")
    source_conflicts = read_jsonl(final_dir / "source_conflicts.jsonl")
    validation_report = read_json(final_dir / "validation_report.json")

    source_documents = {row["source_document_id"]: row for row in read_jsonl(final_dir / "source_documents.jsonl")}
    pages_by_source = {(row["source_document_id"], row["page_number"]): row["text"] for row in pages}
    legal_units = [row for row in legal_units if numeric_suffix(row["legal_unit_id"]) <= 609]
    chunks = [row for row in chunks if numeric_suffix(row["chunk_id"]) <= 609]
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
        metadata_grounding_registry=metadata_grounding_registry,
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
    legal_units.sort(key=lambda row: row["legal_unit_id"])
    chunks.sort(key=lambda row: row["chunk_id"])
    evidence.sort(key=lambda row: row["evidence_id"])
    retrieval_units.sort(key=lambda row: row["retrieval_unit_id"])
    apply_chunk_grounding(chunks, legal_units, evidence, page_text_spans)
    metadata_graph_edges = repair_metadata_graph_edges(metadata_graph_edges, metadata_assertions)
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
    manifest["page_text_spans"] = "page_text_spans.jsonl"
    manifest.setdefault("files", {}).setdefault("page_text_spans.jsonl", {})

    write_jsonl(final_dir / "legal_units.jsonl", legal_units)
    write_jsonl(final_dir / "chunks.jsonl", chunks)
    write_jsonl(final_dir / "evidence_registry.jsonl", evidence)
    write_jsonl(final_dir / "bbox_registry.jsonl", bbox_rows)
    write_jsonl(final_dir / "retrieval_units.jsonl", retrieval_units)
    write_jsonl(final_dir / "document_metadata.jsonl", document_metadata)
    write_jsonl(final_dir / "metadata_grounding.jsonl", metadata_grounding)
    write_jsonl(final_dir / "metadata_grounding_registry.jsonl", metadata_grounding_registry)
    write_jsonl(final_dir / "metadata_graph_edges.jsonl", metadata_graph_edges)
    write_jsonl(final_dir / "source_conflicts.jsonl", source_conflicts)
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
