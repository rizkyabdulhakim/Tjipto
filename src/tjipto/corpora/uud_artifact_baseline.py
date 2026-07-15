from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

from tjipto.corpora.uud.bbox_builder import pdf_lines
from tjipto.corpora.uud.chunk_builder import build_chunks_from_legal_units
from tjipto.corpora.uud.evidence_bbox_builder import build_evidence_and_bboxes
from tjipto.corpora.uud.graph_builder import build_graph_artifacts
from tjipto.artifacts.writer import write_json, write_jsonl
from tjipto.corpora.uud.legal_unit_builder import build_legal_units_from_sources
from tjipto.contracts.structure import apply_chunk_structural_contract
from tjipto.corpora.uud.manifest import build_manifest, refresh_manifest
from tjipto.corpora.uud.metadata_builder import (
    build_document_metadata,
    build_metadata_assertions,
    build_metadata_block_grounding,
    build_metadata_graph_edges,
    rebuild_metadata_grounding,
)
from tjipto.corpora.uud.pages_builder import build_pages
from tjipto.corpora.uud.pipeline import run_staged_uud_pipeline
from tjipto.corpora.uud.relation_builder import build_article_amendment_relations, build_document_relations
from tjipto.corpora.uud.retrieval_builder import apply_chunk_grounding, build_retrieval_units
from tjipto.corpora.uud.specs import EXCLUDED_RECORD_SPECS, FINAL_DIR
from tjipto.corpora.uud.span_disposition_builder import apply_page_text_span_dispositions
from tjipto.corpora.uud.source_conflict_builder import apply_source_conflict_grounding, build_source_conflicts
from tjipto.corpora.uud.source_documents_builder import build_source_documents
from tjipto.corpora.uud.text_span_builder import build_page_text_spans
from tjipto.corpora.uud.validation import build_validation_report, validate_uud_artifact_dir
from tjipto.corpora.uud.policy.authority import apply_authority_contract
from tjipto.corpora.uud.policy.relations import apply_graph_relation_policy
from tjipto.corpora.intent_config import intent_config_for
from tjipto.corpora.registry import CorpusRegistry
from tjipto.grounding.promotion import build_promotion_decisions
from tjipto.ingestion.pdf.health import build_pdf_health_report
from tjipto.ingestion.pdf.words import build_word_bbox_rows


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

    excluded_records = deepcopy(list(EXCLUDED_RECORD_SPECS))
    source_documents = {row["source_document_id"]: row for row in build_source_documents(repo_root)}
    manifest = build_manifest(source_documents)
    source_conflicts = build_source_conflicts()
    pages = build_pages(repo_root, source_documents)
    pages_by_source = {(row["source_document_id"], row["page_number"]): row["text"] for row in pages}
    docs = {source_id: fitz.open(repo_root / meta["path"]) for source_id, meta in source_documents.items()}
    try:
        pdf_lines_by_source = {source_id: pdf_lines(doc) for source_id, doc in docs.items()}
        word_bboxes = [
            row
            for source_id, doc in docs.items()
            for row in build_word_bbox_rows(
                doc=doc,
                corpus_id="uud",
                source_document_id=source_id,
                source_meta=source_documents[source_id],
                bbox_id_prefix="uud_word_bbox",
            )
        ]
    finally:
        for doc in docs.values():
            doc.close()
    page_text_spans = build_page_text_spans(source_documents=source_documents, pdf_lines=pdf_lines_by_source)
    legal_units = build_legal_units_from_sources(
        pages_by_source=pages_by_source,
        source_documents=source_documents,
    )
    chunks = build_chunks_from_legal_units(legal_units)
    evidence, bbox_rows = build_evidence_and_bboxes(
        legal_units=legal_units,
        chunks=chunks,
        source_documents=source_documents,
        pdf_lines_by_source=pdf_lines_by_source,
    )
    metadata_grounding = build_metadata_block_grounding(
        pages_by_source=pages_by_source,
        source_documents=source_documents,
    )
    document_metadata = build_document_metadata(source_documents, metadata_grounding)

    document_metadata, metadata_grounding, metadata_grounding_registry = rebuild_metadata_grounding(
        document_metadata=document_metadata,
        metadata_grounding=metadata_grounding,
        evidence=evidence,
        bbox_rows=bbox_rows,
        word_bboxes=word_bboxes,
        legal_units=legal_units,
        page_text_spans=page_text_spans,
        source_conflicts=source_conflicts,
    )

    apply_source_conflict_grounding(source_conflicts, evidence, bbox_rows, word_bboxes, page_text_spans)
    bbox_rows.sort(key=lambda row: (row["source_document_id"], row["page_number"], row["bbox_id"]))
    metadata_assertions = build_metadata_assertions(evidence, metadata_grounding, bbox_rows)
    legal_units.sort(key=lambda row: row["legal_unit_id"])
    chunks.sort(key=lambda row: row["chunk_id"])
    evidence.sort(key=lambda row: row["evidence_id"])
    apply_chunk_grounding(chunks, legal_units, evidence, page_text_spans)
    apply_chunk_structural_contract(chunks, legal_units)
    retrieval_units = build_retrieval_units(evidence, chunks)
    retrieval_units.sort(key=lambda row: row["retrieval_unit_id"])
    apply_page_text_span_dispositions(
        page_text_spans=page_text_spans,
        bbox_rows=bbox_rows,
        word_bboxes=word_bboxes,
        legal_units=legal_units,
        chunks=chunks,
        metadata_grounding=metadata_grounding,
        source_conflicts=source_conflicts,
    )
    pdf_health_report = build_pdf_health_report(
        repo_root=repo_root,
        corpus_id="uud",
        source_documents=source_documents,
        pages=pages,
        page_text_spans=page_text_spans,
    )
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
    apply_graph_relation_policy(edges=graph_edges, nodes=graph_nodes, evidence=evidence)
    apply_authority_contract(
        spans=page_text_spans,
        evidence=evidence,
        bboxes=bbox_rows,
        units=legal_units,
        chunks=chunks,
        nodes=graph_nodes,
        edges=graph_edges,
    )
    document_relations = build_document_relations(list(source_documents.values()))
    article_amendment_relations = build_article_amendment_relations(
        graph_edges=graph_edges,
        legal_units=legal_units,
        evidence=evidence,
        bbox_rows=bbox_rows,
    )
    promotion_decisions = build_promotion_decisions(
        evidence=evidence,
        metadata_grounding=metadata_grounding,
        bbox_rows=bbox_rows,
        page_text_spans=page_text_spans,
        pages=pages,
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
    write_jsonl(final_dir / "excluded_records.jsonl", excluded_records)
    write_jsonl(final_dir / "source_documents.jsonl", list(source_documents.values()))
    write_jsonl(final_dir / "pages.jsonl", pages)
    write_jsonl(final_dir / "graph_nodes.jsonl", graph_nodes)
    write_jsonl(final_dir / "graph_edges.jsonl", graph_edges)
    write_jsonl(final_dir / "document_relations.jsonl", document_relations)
    write_jsonl(final_dir / "article_amendment_relations.jsonl", article_amendment_relations)
    write_jsonl(final_dir / "page_text_spans.jsonl", page_text_spans)
    write_jsonl(final_dir / "word_bboxes.jsonl", word_bboxes)
    write_jsonl(final_dir / "promotion_decisions.jsonl", promotion_decisions)
    write_json(final_dir / "pdf_health_report.json", pdf_health_report)

    corpus_config = CorpusRegistry(repo_root).resolve("uud")
    validation_report = build_validation_report(
        chunks=chunks,
        legal_units=legal_units,
        excluded_records=excluded_records,
        source_conflicts=source_conflicts,
        evidence=evidence,
        bbox_rows=bbox_rows,
        retrieval_units=retrieval_units,
        metadata_grounding=metadata_grounding,
        metadata_grounding_registry=metadata_grounding_registry,
        manifest_files=manifest["files"],
        graph_nodes=graph_nodes,
        graph_edges=graph_edges,
        document_relations=document_relations,
        article_amendment_relations=article_amendment_relations,
        promotion_decisions=promotion_decisions,
        page_text_spans=page_text_spans,
        word_bboxes=word_bboxes,
        pdf_health_report=pdf_health_report,
        pages=pages,
        intent_config=intent_config_for(getattr(corpus_config, "structured_strategy", "generic"), corpus_config),
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


def main(argv: list[str] | None = None) -> int:
    command = (argv or sys.argv[1:] or ["validate"])[0]
    if command == "validate":
        errors = validate_uud_artifact_baseline(Path.cwd())
        if errors:
            print("\n".join(errors))
            return 1
        print("PASS")
        return 0
    if command == "rebuild":
        print(rebuild_uud_artifact_baseline(Path.cwd()))
        return 0
    print("usage: python -m tjipto.corpora.uud_artifact_baseline [validate|rebuild]")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
