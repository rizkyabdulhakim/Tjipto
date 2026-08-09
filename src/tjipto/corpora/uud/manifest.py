from __future__ import annotations

from pathlib import Path

from tjipto.corpora.uud.artifact_policy import UUD_ARTIFACT_SCHEMA
from tjipto.artifacts.manifest import refresh_manifest as refresh_artifact_manifest
from tjipto.contracts.artifacts import MINIMUM_ARTIFACT_FIELDS
from tjipto.corpora.uud.artifact_policy import UUD_ARTIFACT_ORIGIN_POLICY
from tjipto.corpora.uud.contract import CONTRACT_FINGERPRINT, CONTRACT_ID, CONTRACT_VERSION
from tjipto.ingestion.pdf.fingerprint import extractor_fingerprint


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
    ("meaningful_support_units", "meaningful_support_units.jsonl"),
    ("pages", "pages.jsonl"),
    ("pdf_health_report", "pdf_health_report.json"),
    ("promotion_decisions", "promotion_decisions.jsonl"),
    ("propositions", "propositions.jsonl"),
    ("retrieval_units", "retrieval_units.jsonl"),
    ("runtime_projection", "runtime_projection.json"),
    ("source_conflicts", "source_conflicts.jsonl"),
    ("source_documents", "source_documents.jsonl"),
    ("source_objects", "source_objects.jsonl"),
    ("source_integrity", "source_integrity.json"),
    ("validation_alignment_results", "validation_alignment_results.jsonl"),
    ("validation_exception_review_labels", "validation_exception_review_labels.jsonl"),
    ("validation_exceptions", "validation_exceptions.jsonl"),
    ("validation_report", "validation_report.json"),
    ("page_text_spans", "page_text_spans.jsonl"),
    ("raw_source_spans", "raw_source_spans.jsonl"),
    ("word_bboxes", "word_bboxes.jsonl"),
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
    ("meaningful_support_units", "meaningful_support_units.jsonl"),
    ("not_promoted_amends_edges", "not_promoted_amends_edges.jsonl"),
    ("pages", "pages.jsonl"),
    ("promotion_decisions", "promotion_decisions.jsonl"),
    ("propositions", "propositions.jsonl"),
    ("retrieval_units", "retrieval_units.jsonl"),
    ("source_conflicts", "source_conflicts.jsonl"),
    ("source_documents", "source_documents.jsonl"),
    ("source_objects", "source_objects.jsonl"),
    ("validation_alignment_results", "validation_alignment_results.jsonl"),
    ("validation_exception_review_labels", "validation_exception_review_labels.jsonl"),
    ("validation_exceptions", "validation_exceptions.jsonl"),
    ("page_text_spans", "page_text_spans.jsonl"),
    ("raw_source_spans", "raw_source_spans.jsonl"),
    ("word_bboxes", "word_bboxes.jsonl"),
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
}

PRIMARY_IDS = {
    "article_amendment_relations": "relation_id",
    "article_versions": "article_version_id",
    "bbox_registry": "bbox_id",
    "chunks": "chunk_id",
    "document_metadata": "document_metadata_id",
    "document_relations": "relation_id",
    "evidence_registry": "evidence_id",
    "excluded_records": "excluded_record_id",
    "graph_edges": "edge_id",
    "graph_nodes": "node_id",
    "legal_units": "legal_unit_id",
    "metadata": "metadata_id",
    "metadata_graph_edges": "edge_id",
    "metadata_grounding": "metadata_grounding_id",
    "metadata_grounding_registry": "metadata_grounding_ref_id",
    "meaningful_support_units": "support_unit_id",
    "pages": "page_id",
    "page_text_spans": "text_span_id",
    "raw_source_spans": "raw_source_span_id",
    "promotion_decisions": "decision_id",
    "propositions": "proposition_id",
    "retrieval_units": "retrieval_unit_id",
    "source_documents": "source_document_id",
    "source_objects": "source_object_id",
    "source_conflicts": "source_conflict_id",
    "validation_alignment_results": "alignment_result_id",
    "validation_exception_review_labels": "exception_review_id",
    "validation_exceptions": "exception_id",
    "word_bboxes": "word_bbox_id",
}

REQUIRED_FIELDS = MINIMUM_ARTIFACT_FIELDS


def build_manifest(source_documents: dict[str, dict]) -> dict:
    manifest = {
        "article_amendment_relations": "article_amendment_relations.jsonl",
        "article_versions": "article_versions.jsonl",
        "bbox_registry": "bbox_registry.jsonl",
        "chunks": "chunks.jsonl",
        "corpus_id": "uud",
        "contract_id": CONTRACT_ID,
        "contract_version": CONTRACT_VERSION,
        "contract_fingerprint": CONTRACT_FINGERPRINT,
        "extractor_fingerprint": extractor_fingerprint(),
        "counts": {},
        "document_metadata": "document_metadata.jsonl",
        "document_relations": "document_relations.jsonl",
        "evidence_registry": "evidence_registry.jsonl",
        "excluded_records": "excluded_records.jsonl",
        "files": {
            filename: {
                **UUD_ARTIFACT_ORIGIN_POLICY[filename],
                "logical_key": logical_key,
                "artifact_kind": logical_key,
                "format": "json" if filename.endswith(".json") else "jsonl",
                "artifact_schema": UUD_ARTIFACT_SCHEMA,
                **({"primary_id": PRIMARY_IDS[logical_key]} if logical_key in PRIMARY_IDS else {}),
                **({"required_fields": REQUIRED_FIELDS[logical_key]} if logical_key in REQUIRED_FIELDS else {}),
            }
            for logical_key, filename in ARTIFACT_FILES
        },
        "fixtures": FIXTURES,
        "graph_edges": "graph_edges.jsonl",
        "graph_nodes": "graph_nodes.jsonl",
        "legal_units": "legal_units.jsonl",
        "metadata": "metadata.jsonl",
        "metadata_graph_edges": "metadata_graph_edges.jsonl",
        "metadata_grounding": "metadata_grounding.jsonl",
        "metadata_grounding_registry": "metadata_grounding_registry.jsonl",
        "meaningful_support_units": "meaningful_support_units.jsonl",
        "pages": "pages.jsonl",
        "pdf_health_report": "pdf_health_report.json",
        "promotion_decisions": "promotion_decisions.jsonl",
        "propositions": "propositions.jsonl",
        "retrieval_units": "retrieval_units.jsonl",
        "runtime_projection": "runtime_projection.json",
        "schema_version": UUD_ARTIFACT_SCHEMA,
        "source_conflicts": "source_conflicts.jsonl",
        "source_documents": "source_documents.jsonl",
        "source_objects": "source_objects.jsonl",
        "source_files": {row["path"]: row["sha256"] for row in sorted(source_documents.values(), key=lambda item: item["path"])},
        "source_integrity": "source_integrity.json",
        "status": "final",
        "validation_alignment_results": "validation_alignment_results.jsonl",
        "validation_exception_review_labels": "validation_exception_review_labels.jsonl",
        "validation_exceptions": "validation_exceptions.jsonl",
        "validation_report": "validation_report.json",
        "page_text_spans": "page_text_spans.jsonl",
        "raw_source_spans": "raw_source_spans.jsonl",
        "word_bboxes": "word_bboxes.jsonl",
    }
    return manifest


def refresh_manifest(final_dir: Path, manifest: dict) -> None:
    refresh_artifact_manifest(
        final_dir,
        manifest,
        count_files=COUNT_FILES,
        legacy_counts=LEGACY_COUNTS,
    )
