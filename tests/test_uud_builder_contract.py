from __future__ import annotations

import json
from pathlib import Path
import unittest

from tjipto.core.manifest import read_json, read_jsonl
from tjipto.corpora.intent_config import intent_config_for
from tjipto.corpora.registry import CorpusRegistry
from tjipto.corpora.uud.bbox_builder import pdf_lines
from tjipto.corpora.uud.chunk_builder import build_chunks_from_legal_units
from tjipto.corpora.uud.evidence_bbox_builder import build_evidence_and_bboxes
from tjipto.corpora.uud.legal_unit_builder import build_legal_units_from_sources
from tjipto.corpora.uud.manifest import build_manifest
from tjipto.corpora.uud.metadata_builder import (
    build_document_metadata,
    build_metadata_assertions,
    build_metadata_block_grounding,
    build_metadata_graph_edges,
    rebuild_metadata_grounding,
)
from tjipto.corpora.uud.pages_builder import build_pages
from tjipto.corpora.uud.retrieval_builder import build_retrieval_units
from tjipto.corpora.uud.specs import EXCLUDED_RECORD_SPECS
from tjipto.corpora.uud.source_conflict_builder import apply_source_conflict_grounding, build_source_conflicts
from tjipto.corpora.uud.source_documents_builder import build_source_documents
from tjipto.corpora.uud.validation import build_validation_report


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "data/final/uud"


class UudBuilderContractTest(unittest.TestCase):
    def test_builder_does_not_strip_old_final_artifact_rows(self) -> None:
        source = (ROOT / "src/tjipto/corpora/uud_artifact_baseline.py").read_text(encoding="utf-8")
        self.assertNotIn("<= 609", source)
        self.assertNotIn('startswith("uud_instrument_final_citation_evidence::")', source)
        self.assertNotIn('startswith("uud_retrieval_unit::uud_instrument_final_citation_evidence::")', source)

    def test_generic_artifact_writer_is_not_owned_by_uud_adapter(self) -> None:
        manifest_source = (ROOT / "src/tjipto/corpora/uud/manifest.py").read_text(encoding="utf-8")
        pipeline_source = (ROOT / "src/tjipto/corpora/uud/pipeline.py").read_text(encoding="utf-8")
        self.assertNotIn("def write_json", manifest_source)
        self.assertNotIn("def write_jsonl", manifest_source)
        self.assertIn("tjipto.artifacts.manifest", manifest_source)
        self.assertIn("tjipto.artifacts.pipeline", pipeline_source)

    def test_generic_pdf_builders_are_not_owned_by_uud_adapter(self) -> None:
        source_documents_source = (ROOT / "src/tjipto/corpora/uud/source_documents_builder.py").read_text(encoding="utf-8")
        pages_source = (ROOT / "src/tjipto/corpora/uud/pages_builder.py").read_text(encoding="utf-8")
        spans_source = (ROOT / "src/tjipto/corpora/uud/text_span_builder.py").read_text(encoding="utf-8")
        bbox_source = (ROOT / "src/tjipto/corpora/uud/bbox_builder.py").read_text(encoding="utf-8")
        self.assertNotIn("import fitz", source_documents_source)
        self.assertNotIn('get_text("text")', pages_source)
        self.assertNotIn("accepted_text_span", spans_source)
        self.assertNotIn("def pdf_lines", bbox_source)
        self.assertIn("tjipto.ingestion.pdf.source_documents", source_documents_source)
        self.assertIn("tjipto.ingestion.pdf.pages", pages_source)
        self.assertIn("tjipto.ingestion.pdf.text_spans", spans_source)
        self.assertIn("tjipto.ingestion.pdf.bbox", bbox_source)

    def test_uud_policy_lives_in_specs_and_parser(self) -> None:
        legal_unit_source = (ROOT / "src/tjipto/corpora/uud/legal_unit_builder.py").read_text(encoding="utf-8")
        specs_source = (ROOT / "src/tjipto/corpora/uud/specs.py").read_text(encoding="utf-8")
        parser_source = (ROOT / "src/tjipto/corpora/uud/parser.py").read_text(encoding="utf-8")
        for name in ("SOURCE_ORDER", "LEGAL_STARTS", "CHUNK_STARTS", "TOKEN_RE"):
            self.assertNotIn(f"{name} =", legal_unit_source)
        self.assertIn("UUD_LEGAL_UNIT_SOURCE_ORDER", specs_source)
        self.assertIn("UUD_LEGAL_UNIT_ID_STARTS", specs_source)
        self.assertIn("UUD_CHUNK_ID_STARTS", specs_source)
        self.assertIn("UUD_LEGAL_TOKEN_RE", parser_source)

    def test_uud_builders_do_not_derive_source_role_from_source_id_shape(self) -> None:
        for relative_path in (
            "src/tjipto/corpora/uud/evidence_bbox_builder.py",
            "src/tjipto/corpora/uud/evidence_builder.py",
            "src/tjipto/corpora/uud/graph_builder.py",
            "src/tjipto/corpora/uud/metadata_builder.py",
            "src/tjipto/corpora/uud/retrieval_builder.py",
            "src/tjipto/ingestion/pdf/text_spans.py",
        ):
            source = (ROOT / relative_path).read_text(encoding="utf-8")
            self.assertNotIn('.split("::"', source, relative_path)
            self.assertNotIn(".split('::'", source, relative_path)

    def test_builder_does_not_use_compatibility_seed_as_active_input(self) -> None:
        source = (ROOT / "src/tjipto/corpora/uud_artifact_baseline.py").read_text(encoding="utf-8")
        seed = (ROOT / "src/tjipto/corpora/uud/compatibility_seed.py").read_text(encoding="utf-8")
        self.assertNotIn("load_compatibility_seed", source)
        self.assertNotIn("read_jsonl(final_dir", source)
        self.assertNotIn('read_jsonl(stage_dir / "legal_units.jsonl")', source)
        self.assertNotIn('read_jsonl(stage_dir / "chunks.jsonl")', source)
        self.assertNotIn('read_jsonl(stage_dir / "evidence_registry.jsonl")', source)
        self.assertNotIn('read_jsonl(stage_dir / "bbox_registry.jsonl")', source)
        self.assertIn("compatibility bridge", seed)

    def test_validation_report_does_not_report_active_compatibility_seed_bridge(self) -> None:
        bridge = read_json(FINAL / "validation_report.json")["artifact_governance"]["compatibility_seed_bridge"]
        self.assertNotEqual(bridge.get("status"), "temporary_limitation")
        self.assertNotEqual(bridge.get("source"), "existing_stage_artifacts")
        self.assertFalse(bridge.get("seeded_artifacts"))

    def test_compatibility_seed_file_remains_narrow_if_present(self) -> None:
        seed = (ROOT / "src/tjipto/corpora/uud/compatibility_seed.py").read_text(encoding="utf-8")
        self.assertNotIn("manifest.json", seed)
        self.assertNotIn("source_documents.jsonl", seed)
        self.assertNotIn("pages.jsonl", seed)
        self.assertNotIn("source_conflicts.jsonl", seed)
        self.assertNotIn("excluded_records.jsonl", seed)
        self.assertNotIn("document_metadata.jsonl", seed)
        self.assertNotIn("metadata_grounding.jsonl", seed)
        self.assertNotIn("metadata_grounding_registry.jsonl", seed)
        self.assertNotIn("metadata.jsonl", seed)
        self.assertNotIn("metadata_graph_edges.jsonl", seed)
        self.assertNotIn("retrieval_units.jsonl", seed)
        self.assertNotIn("validation_report.json", seed)

    def test_source_documents_rebuild_from_specs_and_pdfs(self) -> None:
        self.assertEqual(
            build_source_documents(ROOT),
            read_jsonl(FINAL / "source_documents.jsonl"),
        )

    def test_manifest_rebuilds_from_artifact_contract_and_source_specs(self) -> None:
        source_documents = {row["source_document_id"]: row for row in build_source_documents(ROOT)}
        expected = build_manifest(source_documents)
        actual = read_json(FINAL / "manifest.json")
        self.assertEqual(expected["source_files"], actual["source_files"])
        self.assertEqual(expected["fixtures"], actual["fixtures"])
        self.assertEqual(list(expected["files"]), list(actual["files"]))
        self.assertEqual(
            {key: value for key, value in expected.items() if key not in {"counts", "files"}},
            {key: value for key, value in actual.items() if key not in {"counts", "files"}},
        )

    def test_pages_rebuild_from_specs_and_pdfs(self) -> None:
        source_documents = {row["source_document_id"]: row for row in build_source_documents(ROOT)}
        self.assertEqual(
            build_pages(ROOT, source_documents),
            read_jsonl(FINAL / "pages.jsonl"),
        )

    def test_legal_units_rebuild_identity_from_specs_and_pages_without_seed(self) -> None:
        builder_source = (ROOT / "src/tjipto/corpora/uud/legal_unit_builder.py").read_text(encoding="utf-8")
        self.assertNotIn("compatibility_seed", builder_source)
        self.assertNotIn("read_jsonl", builder_source)
        source_documents = {row["source_document_id"]: row for row in build_source_documents(ROOT)}
        pages_by_source = {(row["source_document_id"], row["page_number"]): row["text"] for row in build_pages(ROOT, source_documents)}
        rebuilt = build_legal_units_from_sources(
            pages_by_source=pages_by_source,
            source_documents=source_documents,
        )
        expected = read_jsonl(FINAL / "legal_units.jsonl")
        self.assertEqual(len(rebuilt), len(expected))
        self.assertEqual(
            [(row["legal_unit_id"], row["source_document_id"], row["unit_label"], row["unit_type"]) for row in rebuilt],
            [(row["legal_unit_id"], row["source_document_id"], row["unit_label"], row["unit_type"]) for row in expected],
        )

    def test_chunks_rebuild_identity_from_rebuilt_legal_units_without_seed(self) -> None:
        builder_source = (ROOT / "src/tjipto/corpora/uud/chunk_builder.py").read_text(encoding="utf-8")
        self.assertNotIn("compatibility_seed", builder_source)
        self.assertNotIn("read_jsonl", builder_source)
        source_documents = {row["source_document_id"]: row for row in build_source_documents(ROOT)}
        pages_by_source = {(row["source_document_id"], row["page_number"]): row["text"] for row in build_pages(ROOT, source_documents)}
        legal_units = build_legal_units_from_sources(
            pages_by_source=pages_by_source,
            source_documents=source_documents,
        )
        rebuilt = build_chunks_from_legal_units(legal_units)
        expected = read_jsonl(FINAL / "chunks.jsonl")
        self.assertEqual(len(rebuilt), len(expected))
        self.assertEqual(
            [(row["chunk_id"], row["legal_unit_id"], row["chunk_type"]) for row in rebuilt],
            [(row["chunk_id"], row["legal_unit_id"], row["chunk_type"]) for row in expected],
        )

    def test_evidence_rebuild_identity_from_rebuilt_chunks_without_seed(self) -> None:
        import fitz

        builder_source = (ROOT / "src/tjipto/corpora/uud/evidence_bbox_builder.py").read_text(encoding="utf-8")
        self.assertNotIn("compatibility_seed", builder_source)
        self.assertNotIn("read_jsonl", builder_source)
        source_documents = {row["source_document_id"]: row for row in build_source_documents(ROOT)}
        pages_by_source = {(row["source_document_id"], row["page_number"]): row["text"] for row in build_pages(ROOT, source_documents)}
        legal_units = build_legal_units_from_sources(
            pages_by_source=pages_by_source,
            source_documents=source_documents,
        )
        chunks = build_chunks_from_legal_units(legal_units)
        docs = {source_id: fitz.open(ROOT / meta["path"]) for source_id, meta in source_documents.items()}
        try:
            evidence, bbox_rows = build_evidence_and_bboxes(
                legal_units=legal_units,
                chunks=chunks,
                source_documents=source_documents,
                pdf_lines_by_source={source_id: pdf_lines(doc) for source_id, doc in docs.items()},
            )
        finally:
            for doc in docs.values():
                doc.close()
        expected = read_jsonl(FINAL / "evidence_registry.jsonl")
        self.assertEqual(len(evidence), len(expected))
        self.assertEqual(
            [(row["evidence_id"], row["legal_unit_id"], row["citation"], row["source_document_id"]) for row in evidence],
            [(row["evidence_id"], row["legal_unit_id"], row["citation"], row["source_document_id"]) for row in expected],
        )
        expected_bbox = read_jsonl(FINAL / "bbox_registry.jsonl")
        self.assertEqual(len(bbox_rows), len(expected_bbox))
        self.assertEqual(
            [row["bbox_id"] for row in bbox_rows],
            [row["bbox_id"] for row in expected_bbox],
        )

    def test_metadata_block_grounding_rebuilds_from_specs_and_pages(self) -> None:
        source_documents = {row["source_document_id"]: row for row in build_source_documents(ROOT)}
        pages_by_source = {(row["source_document_id"], row["page_number"]): row["text"] for row in build_pages(ROOT, source_documents)}
        expected = [
            row
            for row in read_jsonl(FINAL / "metadata_grounding.jsonl")
            if not row["metadata_grounding_id"].startswith("uud_metadata_field_grounding::")
        ]
        self.assertEqual(
            build_metadata_block_grounding(
                pages_by_source=pages_by_source,
                source_documents=source_documents,
            ),
            [{key: row[key] for key in row if key not in {"bbox_precision", "grounding_status", "failure_reason"}} for row in expected],
        )

    def test_document_metadata_rebuilds_from_specs_and_grounding(self) -> None:
        source_documents = {row["source_document_id"]: row for row in build_source_documents(ROOT)}
        pages_by_source = {(row["source_document_id"], row["page_number"]): row["text"] for row in build_pages(ROOT, source_documents)}
        metadata_grounding = build_metadata_block_grounding(
            pages_by_source=pages_by_source,
            source_documents=source_documents,
        )
        document_metadata, _, _ = rebuild_metadata_grounding(
            document_metadata=build_document_metadata(source_documents, metadata_grounding),
            metadata_grounding=metadata_grounding,
            evidence=read_jsonl(FINAL / "evidence_registry.jsonl"),
            bbox_rows=read_jsonl(FINAL / "bbox_registry.jsonl"),
            legal_units=read_jsonl(FINAL / "legal_units.jsonl"),
            page_text_spans=read_jsonl(FINAL / "page_text_spans.jsonl"),
            source_conflicts=read_jsonl(FINAL / "source_conflicts.jsonl"),
        )
        self.assertEqual(json.loads(json.dumps(document_metadata)), read_jsonl(FINAL / "document_metadata.jsonl"))

    def test_metadata_assertions_rebuild_from_evidence_and_grounding(self) -> None:
        source_documents = {row["source_document_id"]: row for row in build_source_documents(ROOT)}
        pages_by_source = {(row["source_document_id"], row["page_number"]): row["text"] for row in build_pages(ROOT, source_documents)}
        _, metadata_grounding, _ = rebuild_metadata_grounding(
            document_metadata=build_document_metadata(
                source_documents,
                build_metadata_block_grounding(
                    pages_by_source=pages_by_source,
                    source_documents=source_documents,
                ),
            ),
            metadata_grounding=build_metadata_block_grounding(
                pages_by_source=pages_by_source,
                source_documents=source_documents,
            ),
            evidence=read_jsonl(FINAL / "evidence_registry.jsonl"),
            bbox_rows=read_jsonl(FINAL / "bbox_registry.jsonl"),
            legal_units=read_jsonl(FINAL / "legal_units.jsonl"),
            page_text_spans=read_jsonl(FINAL / "page_text_spans.jsonl"),
            source_conflicts=read_jsonl(FINAL / "source_conflicts.jsonl"),
        )
        self.assertEqual(
            build_metadata_assertions(
                read_jsonl(FINAL / "evidence_registry.jsonl"),
                metadata_grounding,
                read_jsonl(FINAL / "bbox_registry.jsonl"),
            ),
            read_jsonl(FINAL / "metadata.jsonl"),
        )

    def test_metadata_graph_edges_rebuild_from_metadata_assertions(self) -> None:
        self.assertEqual(
            build_metadata_graph_edges(read_jsonl(FINAL / "metadata.jsonl")),
            read_jsonl(FINAL / "metadata_graph_edges.jsonl"),
        )

    def test_retrieval_units_rebuild_from_evidence_and_chunks(self) -> None:
        self.assertEqual(
            build_retrieval_units(
                read_jsonl(FINAL / "evidence_registry.jsonl"),
                read_jsonl(FINAL / "chunks.jsonl"),
            ),
            read_jsonl(FINAL / "retrieval_units.jsonl"),
        )

    def test_validation_report_rebuilds_from_artifact_rows(self) -> None:
        self.assertEqual(
            build_validation_report(
                chunks=read_jsonl(FINAL / "chunks.jsonl"),
                legal_units=read_jsonl(FINAL / "legal_units.jsonl"),
                excluded_records=read_jsonl(FINAL / "excluded_records.jsonl"),
                source_conflicts=read_jsonl(FINAL / "source_conflicts.jsonl"),
                evidence=read_jsonl(FINAL / "evidence_registry.jsonl"),
                bbox_rows=read_jsonl(FINAL / "bbox_registry.jsonl"),
                retrieval_units=read_jsonl(FINAL / "retrieval_units.jsonl"),
                metadata_grounding=read_jsonl(FINAL / "metadata_grounding.jsonl"),
                metadata_grounding_registry=read_jsonl(FINAL / "metadata_grounding_registry.jsonl"),
                manifest_files=read_json(FINAL / "manifest.json")["files"],
                graph_nodes=read_jsonl(FINAL / "graph_nodes.jsonl"),
                graph_edges=read_jsonl(FINAL / "graph_edges.jsonl"),
                page_text_spans=read_jsonl(FINAL / "page_text_spans.jsonl"),
                intent_config=intent_config_for("uud_1945", CorpusRegistry(ROOT).resolve("uud")),
            ),
            read_json(FINAL / "validation_report.json"),
        )

    def test_source_conflicts_rebuild_from_specs_and_grounding(self) -> None:
        source_conflicts = build_source_conflicts()
        apply_source_conflict_grounding(
            source_conflicts,
            read_jsonl(FINAL / "evidence_registry.jsonl"),
            read_jsonl(FINAL / "bbox_registry.jsonl"),
            read_jsonl(FINAL / "page_text_spans.jsonl"),
        )
        self.assertEqual(source_conflicts, read_jsonl(FINAL / "source_conflicts.jsonl"))

    def test_excluded_records_rebuild_from_specs(self) -> None:
        self.assertEqual(list(EXCLUDED_RECORD_SPECS), read_jsonl(FINAL / "excluded_records.jsonl"))


if __name__ == "__main__":
    unittest.main()
