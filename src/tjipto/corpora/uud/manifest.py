from __future__ import annotations

from pathlib import Path

from tjipto.artifacts.manifest import refresh_manifest as refresh_artifact_manifest
from tjipto.corpora.uud.artifact_policy import UUD_ARTIFACT_ORIGIN_POLICY


ARTIFACT_FILES = (
    ("article_amendment_relations", "article_amendment_relations.jsonl"),
    ("article_versions", "article_versions.jsonl"),
    ("bbox_registry", "bbox_registry.jsonl"),
    ("chunks", "chunks.jsonl"),
    ("document_metadata", "document_metadata.jsonl"),
    ("document_relations", "document_relations.jsonl"),
    ("evidence_registry", "evidence_registry.jsonl"),
    ("excluded_records", "excluded_records.jsonl"),
    ("graph_edges", "graph_edges.jsonl"),
    ("graph_nodes", "graph_nodes.jsonl"),
    ("legal_units", "legal_units.jsonl"),
    ("metadata", "metadata.jsonl"),
    ("metadata_graph_edges", "metadata_graph_edges.jsonl"),
    ("metadata_grounding", "metadata_grounding.jsonl"),
    ("metadata_grounding_registry", "metadata_grounding_registry.jsonl"),
    ("pages", "pages.jsonl"),
    ("retrieval_units", "retrieval_units.jsonl"),
    ("source_conflicts", "source_conflicts.jsonl"),
    ("source_documents", "source_documents.jsonl"),
    ("source_integrity", "source_integrity.json"),
    ("validation_alignment_results", "validation_alignment_results.jsonl"),
    ("validation_exception_review_labels", "validation_exception_review_labels.jsonl"),
    ("validation_exceptions", "validation_exceptions.jsonl"),
    ("validation_report", "validation_report.json"),
    ("page_text_spans", "page_text_spans.jsonl"),
)

COUNT_FILES = (
    ("article_amendment_relations", "article_amendment_relations.jsonl"),
    ("article_versions", "article_versions.jsonl"),
    ("bbox_records", "bbox_registry.jsonl"),
    ("chunks", "chunks.jsonl"),
    ("document_metadata", "document_metadata.jsonl"),
    ("document_relations", "document_relations.jsonl"),
    ("evidence_records", "evidence_registry.jsonl"),
    ("excluded_records", "excluded_records.jsonl"),
    ("graph_edges", "graph_edges.jsonl"),
    ("graph_nodes", "graph_nodes.jsonl"),
    ("legal_units", "legal_units.jsonl"),
    ("metadata_assertions", "metadata.jsonl"),
    ("metadata_graph_edges", "metadata_graph_edges.jsonl"),
    ("metadata_grounding", "metadata_grounding.jsonl"),
    ("metadata_grounding_records", "metadata_grounding_registry.jsonl"),
    ("not_promoted_amends_edges", "not_promoted_amends_edges.jsonl"),
    ("pages", "pages.jsonl"),
    ("retrieval_units", "retrieval_units.jsonl"),
    ("source_conflicts", "source_conflicts.jsonl"),
    ("source_documents", "source_documents.jsonl"),
    ("validation_alignment_results", "validation_alignment_results.jsonl"),
    ("validation_exception_review_labels", "validation_exception_review_labels.jsonl"),
    ("validation_exceptions", "validation_exceptions.jsonl"),
    ("page_text_spans", "page_text_spans.jsonl"),
)

LEGACY_COUNTS = {
    "not_promoted_amends_edges": 8,
}

FIXTURES = {
    "eval_fixtures": "tests/fixtures/uud/eval_fixtures.jsonl",
    "graph_retrieval_eval_cases": "tests/fixtures/uud/graph_retrieval_eval_cases.jsonl",
    "graph_retrieval_eval_results": "tests/fixtures/uud/graph_retrieval_eval_results.jsonl",
    "graph_retrieval_traces": "tests/fixtures/uud/graph_retrieval_traces.jsonl",
    "orchestrator_eval_results": "tests/fixtures/uud/orchestrator_eval_results.jsonl",
    "orchestrator_eval_summary": "tests/fixtures/uud/orchestrator_eval_summary.json",
    "retrieval_metrics_baseline": "tests/fixtures/uud/retrieval_metrics_baseline.json",
    "retrieval_summary_baseline": "tests/fixtures/uud/retrieval_summary_baseline.json",
}


def build_manifest(source_documents: dict[str, dict]) -> dict:
    manifest = {
        "article_amendment_relations": "article_amendment_relations.jsonl",
        "article_versions": "article_versions.jsonl",
        "bbox_registry": "bbox_registry.jsonl",
        "chunks": "chunks.jsonl",
        "corpus_id": "uud",
        "counts": {},
        "document_metadata": "document_metadata.jsonl",
        "document_relations": "document_relations.jsonl",
        "evidence_registry": "evidence_registry.jsonl",
        "excluded_records": "excluded_records.jsonl",
        "files": {filename: dict(UUD_ARTIFACT_ORIGIN_POLICY[filename]) for _, filename in ARTIFACT_FILES},
        "fixtures": FIXTURES,
        "graph_edges": "graph_edges.jsonl",
        "graph_nodes": "graph_nodes.jsonl",
        "legal_units": "legal_units.jsonl",
        "metadata": "metadata.jsonl",
        "metadata_graph_edges": "metadata_graph_edges.jsonl",
        "metadata_grounding": "metadata_grounding.jsonl",
        "metadata_grounding_registry": "metadata_grounding_registry.jsonl",
        "pages": "pages.jsonl",
        "retrieval_units": "retrieval_units.jsonl",
        "schema_version": 1,
        "source_conflicts": "source_conflicts.jsonl",
        "source_documents": "source_documents.jsonl",
        "source_files": {row["path"]: row["sha256"] for row in sorted(source_documents.values(), key=lambda item: item["path"])},
        "source_integrity": "source_integrity.json",
        "status": "final",
        "validation_alignment_results": "validation_alignment_results.jsonl",
        "validation_exception_review_labels": "validation_exception_review_labels.jsonl",
        "validation_exceptions": "validation_exceptions.jsonl",
        "validation_report": "validation_report.json",
        "page_text_spans": "page_text_spans.jsonl",
    }
    return manifest


def refresh_manifest(final_dir: Path, manifest: dict) -> None:
    refresh_artifact_manifest(
        final_dir,
        manifest,
        count_files=COUNT_FILES,
        legacy_counts=LEGACY_COUNTS,
    )
