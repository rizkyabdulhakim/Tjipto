from __future__ import annotations

import json
from pathlib import Path
import unittest

from tjipto.core.manifest import read_json, read_jsonl
from tjipto.corpora.uud.manifest import build_manifest
from tjipto.corpora.uud.metadata_builder import build_document_metadata, build_metadata_assertions, build_metadata_block_grounding, build_metadata_graph_edges, rebuild_metadata_grounding
from tjipto.corpora.uud.pages_builder import build_pages
from tjipto.corpora.uud.specs import EXCLUDED_RECORD_SPECS
from tjipto.corpora.uud.source_conflict_builder import apply_source_conflict_grounding, build_source_conflicts
from tjipto.corpora.uud.source_documents_builder import build_source_documents


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "data/final/uud"


class UudBuilderContractTest(unittest.TestCase):
    def test_builder_does_not_strip_old_final_artifact_rows(self) -> None:
        source = (ROOT / "src/tjipto/corpora/uud_artifact_baseline.py").read_text(encoding="utf-8")
        self.assertNotIn("<= 609", source)
        self.assertNotIn('startswith("uud_instrument_final_citation_evidence::")', source)
        self.assertNotIn('startswith("uud_retrieval_unit::uud_instrument_final_citation_evidence::")', source)

    def test_builder_seed_dependency_is_isolated_as_compatibility_bridge(self) -> None:
        source = (ROOT / "src/tjipto/corpora/uud_artifact_baseline.py").read_text(encoding="utf-8")
        seed = (ROOT / "src/tjipto/corpora/uud/compatibility_seed.py").read_text(encoding="utf-8")
        self.assertIn("load_compatibility_seed(final_dir)", source)
        self.assertNotIn("read_jsonl(final_dir", source)
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
        self.assertIn("compatibility bridge", seed)

    def test_source_documents_rebuild_from_specs_and_pdfs(self) -> None:
        self.assertEqual(
            build_source_documents(ROOT),
            read_jsonl(FINAL / "source_documents.jsonl"),
        )

    def test_manifest_rebuilds_from_artifact_contract_and_source_specs(self) -> None:
        source_documents = {
            row["source_document_id"]: row
            for row in build_source_documents(ROOT)
        }
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
        source_documents = {
            row["source_document_id"]: row
            for row in build_source_documents(ROOT)
        }
        self.assertEqual(
            build_pages(ROOT, source_documents),
            read_jsonl(FINAL / "pages.jsonl"),
        )

    def test_metadata_block_grounding_rebuilds_from_specs_and_pages(self) -> None:
        source_documents = {
            row["source_document_id"]: row
            for row in build_source_documents(ROOT)
        }
        pages_by_source = {
            (row["source_document_id"], row["page_number"]): row["text"]
            for row in build_pages(ROOT, source_documents)
        }
        expected = [
            row for row in read_jsonl(FINAL / "metadata_grounding.jsonl")
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
        source_documents = {
            row["source_document_id"]: row
            for row in build_source_documents(ROOT)
        }
        pages_by_source = {
            (row["source_document_id"], row["page_number"]): row["text"]
            for row in build_pages(ROOT, source_documents)
        }
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
        source_documents = {
            row["source_document_id"]: row
            for row in build_source_documents(ROOT)
        }
        pages_by_source = {
            (row["source_document_id"], row["page_number"]): row["text"]
            for row in build_pages(ROOT, source_documents)
        }
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
