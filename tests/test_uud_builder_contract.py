from __future__ import annotations

from copy import deepcopy
import ctypes
import gc
import json
from collections import Counter
import hashlib
from pathlib import Path
import shutil
import tempfile
import unittest

from tjipto.core.manifest import read_json, read_jsonl
from tjipto.corpora.intent_config import intent_config_for
from tjipto.corpora.registry import CorpusRegistry
from tjipto.corpora.uud.bbox_builder import extract_pdf
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
    ensure_metadata_source_evidence,
)
from tjipto.corpora.uud.pages_builder import build_pages
from tjipto.corpora.uud.retrieval_builder import build_retrieval_units
from tjipto.corpora.uud.source_conflict_builder import apply_source_conflict_grounding, build_source_conflicts
from tjipto.corpora.uud.source_documents_builder import build_source_documents
from tjipto.corpora.uud.validation import (
    _REQUIRED_HEALTH_STATUSES,
    _derive_validation_status,
    _selector_geometry_health,
    build_validation_report,
)
from tjipto.corpora.uud_artifact_baseline import rebuild_uud_artifact_baseline
from tjipto.ingestion.pdf.words import build_word_bbox_rows


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "data/final/uud"


class UudBuilderContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        source_documents = build_source_documents(ROOT)
        cls._source_documents = {row["source_document_id"]: row for row in source_documents}
        cls._pages = build_pages(ROOT, cls._source_documents)
        cls._pages_by_source = {
            (row["source_document_id"], row["page_number"]): row["text"]
            for row in cls._pages
        }
        cls._cached_word_bboxes: list[dict] | None = None

    @classmethod
    def tearDownClass(cls) -> None:
        cls._release_word_bboxes()

    @classmethod
    def _release_word_bboxes(cls) -> None:
        cls._cached_word_bboxes = None
        gc.collect()
        try:
            ctypes.CDLL(None).malloc_trim(0)
        except (AttributeError, OSError, TypeError):
            pass

    def _source_documents_for_test(self) -> dict[str, dict]:
        return dict(self._source_documents)

    def _pages_by_source_for_test(self) -> dict[tuple[str, int], str]:
        return dict(self._pages_by_source)

    def _pages_for_test(self) -> list[dict]:
        return list(self._pages)

    def test_rebuild_executes_and_is_byte_identical(self) -> None:
        type(self)._cached_word_bboxes = None
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copytree(ROOT / "data", root / "data")
            rebuild_uud_artifact_baseline(root)
            first = _artifact_hashes(root / "data/final/uud")
            rebuild_uud_artifact_baseline(root)
            self.assertEqual(first, _artifact_hashes(root / "data/final/uud"))

    def _word_bboxes(self) -> list[dict]:
        cls = type(self)
        if cls._cached_word_bboxes is not None:
            return cls._cached_word_bboxes
        import fitz

        source_documents = self._source_documents_for_test()
        docs = {source_id: fitz.open(ROOT / meta["path"]) for source_id, meta in source_documents.items()}
        try:
            cls._cached_word_bboxes = [
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
            return cls._cached_word_bboxes
        finally:
            for doc in docs.values():
                doc.close()

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
            list(self._source_documents_for_test().values()),
            read_jsonl(FINAL / "source_documents.jsonl"),
        )

    def test_manifest_rebuilds_from_artifact_contract_and_source_specs(self) -> None:
        source_documents = self._source_documents_for_test()
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
        self.assertEqual(
            self._pages_for_test(),
            read_jsonl(FINAL / "pages.jsonl"),
        )

    def test_legal_units_rebuild_identity_from_specs_and_pages_without_seed(self) -> None:
        builder_source = (ROOT / "src/tjipto/corpora/uud/legal_unit_builder.py").read_text(encoding="utf-8")
        self.assertNotIn("compatibility_seed", builder_source)
        self.assertNotIn("read_jsonl", builder_source)
        source_documents = self._source_documents_for_test()
        pages_by_source = self._pages_by_source_for_test()
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
        source_documents = self._source_documents_for_test()
        pages_by_source = self._pages_by_source_for_test()
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
        source_documents = self._source_documents_for_test()
        pages_by_source = self._pages_by_source_for_test()
        legal_units = build_legal_units_from_sources(
            pages_by_source=pages_by_source,
            source_documents=source_documents,
        )
        chunks = build_chunks_from_legal_units(legal_units)
        word_bboxes = self._word_bboxes()
        docs = {source_id: fitz.open(ROOT / meta["path"]) for source_id, meta in source_documents.items()}
        try:
            evidence, bbox_rows = build_evidence_and_bboxes(
                legal_units=legal_units,
                chunks=chunks,
                source_documents=source_documents,
                pdf_lines_by_source={source_id: extract_pdf(doc, source_id).lines for source_id, doc in docs.items()},
                word_bboxes=word_bboxes,
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
        source_documents = self._source_documents_for_test()
        pages_by_source = self._pages_by_source_for_test()
        expected = [
            row
            for row in read_jsonl(FINAL / "metadata_grounding.jsonl")
            if not row["metadata_grounding_id"].startswith("uud_metadata_field_grounding::")
        ]
        built = build_metadata_block_grounding(
            pages_by_source=pages_by_source,
            source_documents=source_documents,
        )
        ensure_metadata_source_evidence(evidence=read_jsonl(FINAL / "evidence_registry.jsonl"), metadata_grounding=built)
        self.assertEqual(
            [
                _raw_metadata_block_contract(row)
                for row in built
            ],
            [_raw_metadata_block_contract(row) for row in expected],
        )

    def test_document_metadata_rebuilds_from_specs_and_grounding(self) -> None:
        source_documents = self._source_documents_for_test()
        pages_by_source = self._pages_by_source_for_test()
        metadata_grounding = build_metadata_block_grounding(
            pages_by_source=pages_by_source,
            source_documents=source_documents,
        )
        document_metadata, _, _ = rebuild_metadata_grounding(
            document_metadata=build_document_metadata(source_documents, metadata_grounding),
            metadata_grounding=metadata_grounding,
            evidence=read_jsonl(FINAL / "evidence_registry.jsonl"),
            bbox_rows=read_jsonl(FINAL / "bbox_registry.jsonl"),
            word_bboxes=read_jsonl(FINAL / "word_bboxes.jsonl"),
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
            word_bboxes=read_jsonl(FINAL / "word_bboxes.jsonl"),
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
        type(self)._release_word_bboxes()
        self.assertEqual(
            build_validation_report(
                chunks=read_jsonl(FINAL / "chunks.jsonl"),
                legal_units=read_jsonl(FINAL / "legal_units.jsonl"),
                excluded_records=read_jsonl(FINAL / "excluded_records.jsonl"),
                source_conflicts=read_jsonl(FINAL / "source_conflicts.jsonl"),
                evidence=read_jsonl(FINAL / "evidence_registry.jsonl"),
                bbox_rows=read_jsonl(FINAL / "bbox_registry.jsonl"),
                retrieval_units=read_jsonl(FINAL / "retrieval_units.jsonl"),
                promotion_decisions=read_jsonl(FINAL / "promotion_decisions.jsonl"),
                propositions=read_jsonl(FINAL / "propositions.jsonl"),
                metadata_grounding=read_jsonl(FINAL / "metadata_grounding.jsonl"),
                metadata_grounding_registry=read_jsonl(FINAL / "metadata_grounding_registry.jsonl"),
                source_objects=read_jsonl(FINAL / "source_objects.jsonl"),
                word_bboxes=read_jsonl(FINAL / "word_bboxes.jsonl"),
                manifest_files=read_json(FINAL / "manifest.json")["files"],
                graph_nodes=read_jsonl(FINAL / "graph_nodes.jsonl"),
                graph_edges=read_jsonl(FINAL / "graph_edges.jsonl"),
                document_relations=read_jsonl(FINAL / "document_relations.jsonl"),
                article_amendment_relations=read_jsonl(FINAL / "article_amendment_relations.jsonl"),
                page_text_spans=read_jsonl(FINAL / "page_text_spans.jsonl"),
                pdf_health_report=read_json(FINAL / "pdf_health_report.json"),
                pages=read_jsonl(FINAL / "pages.jsonl"),
                intent_config=intent_config_for("uud_1945", CorpusRegistry(ROOT).resolve("uud")),
            ),
            read_json(FINAL / "validation_report.json"),
        )

    def test_validation_report_status_is_derived_from_required_health(self) -> None:
        report = {
            key: {"status": expected}
            for key, expected in _REQUIRED_HEALTH_STATUSES.items()
        }
        _derive_validation_status(report)
        self.assertEqual((report["status"], report["required_health"]["unsatisfied_section_count"]), ("valid", 0))
        for observed in ("incomplete", "unknown", None):
            with self.subTest(observed=observed):
                mutated = dict(report)
                if observed is None:
                    mutated.pop("selector_geometry_health")
                else:
                    mutated["selector_geometry_health"] = {"status": observed}
                _derive_validation_status(mutated)
                self.assertEqual(
                    (mutated["status"], mutated["required_health"]["unsatisfied_section_count"]),
                    ("invalid", 1),
                )
                self.assertIn("selector_geometry_health", mutated["required_health"]["unsatisfied_sections"])

    def test_overlay_lineage_mutations_fail_selector_and_top_level_health(self) -> None:
        propositions = read_jsonl(FINAL / "propositions.jsonl")
        word_bboxes = read_jsonl(FINAL / "word_bboxes.jsonl")
        evidence = read_jsonl(FINAL / "evidence_registry.jsonl")
        page_text_spans = read_jsonl(FINAL / "page_text_spans.jsonl")
        base = next(
            row
            for row in propositions
            if len(row["viewer_overlay"]["rectangles"]) > 1
            and any(
                rectangle["selected_character_end"] - rectangle["selected_character_start"] > 1
                for rectangle in row["viewer_overlay"]["rectangles"]
            )
        )
        clipped_base = next(row for row in propositions if row["viewer_overlay"]["clipped_rectangle_indexes"])

        mutations: dict[str, dict] = {}
        for name in (
            "shifted", "shrunk", "expanded", "missing", "extra", "duplicate",
            "reordered", "wrong_slice", "wrong_space",
        ):
            mutations[name] = deepcopy(base)
        mutations["shifted"]["viewer_overlay"]["rectangles"][0]["x0"] += 0.1
        mutations["shifted"]["viewer_overlay"]["rectangles"][0]["y0"] += 0.1
        mutations["shrunk"]["viewer_overlay"]["rectangles"][0]["x1"] -= 0.1
        mutations["expanded"]["viewer_overlay"]["rectangles"][0]["x1"] += 0.1
        mutations["missing"]["viewer_overlay"]["rectangles"].pop()
        mutations["extra"]["viewer_overlay"]["rectangles"].append(
            deepcopy(mutations["extra"]["viewer_overlay"]["rectangles"][-1])
        )
        mutations["duplicate"]["viewer_overlay"]["rectangles"].insert(
            0, deepcopy(mutations["duplicate"]["viewer_overlay"]["rectangles"][0])
        )
        mutations["reordered"]["viewer_overlay"]["rectangles"].reverse()
        sliced = next(
            rectangle
            for rectangle in mutations["wrong_slice"]["viewer_overlay"]["rectangles"]
            if rectangle["selected_character_end"] - rectangle["selected_character_start"] > 1
        )
        sliced["selected_character_start"] += 1
        mutations["wrong_space"]["viewer_overlay"]["rectangles"][0]["geometry_space_index"] = len(
            mutations["wrong_space"]["viewer_overlay"]["geometry_spaces"]
        )

        mutations["wrong_clipping"] = deepcopy(clipped_base)
        mutations["wrong_clipping"]["viewer_overlay"]["clipped_rectangle_indexes"] = []

        mutations["unrelated_character"] = deepcopy(base)
        original_id = mutations["unrelated_character"]["bbox_refs"][0]
        first_character = next(
            word | character
            for word in word_bboxes
            for character in word.get("characters") or ()
            if character["character_bbox_id"] == original_id
        )
        unrelated_id = next(
            merged["character_bbox_id"]
            for word in word_bboxes
            for character in word.get("characters") or ()
            if (merged := word | character)["character_bbox_id"] not in base["bbox_refs"]
            and merged["source_document_id"] == first_character["source_document_id"]
            and merged["page_number"] == first_character["page_number"]
        )
        mutations["unrelated_character"]["bbox_refs"][0] = unrelated_id
        for selector in mutations["unrelated_character"]["source_selectors"]:
            selector["character_bbox_ids"] = [
                unrelated_id if character_id == original_id else character_id
                for character_id in selector["character_bbox_ids"]
            ]

        for name, proposition in mutations.items():
            with self.subTest(mutation=name):
                health = _selector_geometry_health(
                    propositions=[proposition],
                    evidence=evidence,
                    page_text_spans=page_text_spans,
                    word_bboxes=word_bboxes,
                )
                self.assertGreater(health["viewer_geometry_without_exact_selector_lineage_count"], 0)
                self.assertEqual(health["status"], "incomplete")
                report = {
                    key: {"status": expected}
                    for key, expected in _REQUIRED_HEALTH_STATUSES.items()
                }
                report["selector_geometry_health"] = health
                _derive_validation_status(report)
                self.assertEqual(report["status"], "invalid")

    def test_source_conflicts_rebuild_from_specs_and_grounding(self) -> None:
        source_conflicts = build_source_conflicts()
        apply_source_conflict_grounding(
            source_conflicts,
            read_jsonl(FINAL / "evidence_registry.jsonl"),
            read_jsonl(FINAL / "bbox_registry.jsonl"),
            read_jsonl(FINAL / "word_bboxes.jsonl"),
            read_jsonl(FINAL / "page_text_spans.jsonl"),
        )
        self.assertEqual(source_conflicts, read_jsonl(FINAL / "source_conflicts.jsonl"))

    def test_word_bbox_registry_exists_and_is_deterministic(self) -> None:
        # The artifact writer uses the release-wide six-decimal float contract;
        # compare the builder's in-memory values under that same representation.
        self.assertEqual(
            _canonicalize_numeric_rows(self._word_bboxes()),
            _canonicalize_numeric_rows(read_jsonl(FINAL / "word_bboxes.jsonl")),
        )
        type(self)._release_word_bboxes()

    def test_absent_span_keys_are_fully_classified_with_specific_reasons(self) -> None:
        spans = read_jsonl(FINAL / "page_text_spans.jsonl")
        bbox_rows = read_jsonl(FINAL / "bbox_registry.jsonl")
        bbox_keys = {_span_bbox_key(row) for row in bbox_rows}
        missing = [row for row in spans if _span_bbox_key(row) not in bbox_keys]
        self.assertEqual(len(missing), sum(1 for row in spans if row.get("bbox_registry_coverage_status") == "bbox_key_absent"))
        bucket_counts = Counter(row["bbox_registry_coverage_bucket"] for row in missing)
        self.assertEqual(sum(bucket_counts.values()), len(missing))
        self.assertTrue(
            set(bucket_counts)
            <= {
                "legal_citation_candidate",
                "metadata_provenance_candidate",
                "nonlegal_excluded_provenance",
                "source_anomaly_provenance_candidate",
                "structural_provenance_only",
            }
        )
        for row in missing:
            self.assertEqual(row["bbox_registry_coverage_status"], "bbox_key_absent")
            self.assertTrue(row["bbox_registry_coverage_bucket"])
            self.assertTrue(row["bbox_registry_coverage_reason"])
            if row["bbox_registry_coverage_reason"] == "exact_word_bbox_available":
                self.assertTrue(row["span_bbox_ids"])

    def test_metadata_block_reuses_existing_exact_bbox_rows_when_safe(self) -> None:
        rows = {row["metadata_grounding_id"]: row for row in read_jsonl(FINAL / "metadata_grounding.jsonl")}
        for metadata_grounding_id in (
            "uud_metadata_block_final_evidence::amendment_1_historical::instrument_closing_issuance_page_0003",
            "uud_metadata_block_final_evidence::amendment_2_historical::instrument_closing_issuance_page_0008",
            "uud_metadata_block_final_evidence::amendment_3_historical::instrument_closing_issuance_page_0009",
            "uud_metadata_block_final_evidence::amendment_4_historical::instrument_closing_issuance_page_0006",
        ):
            row = rows[metadata_grounding_id]
            self.assertEqual(row["bbox_precision"], "exact")
            self.assertTrue(row["viewer_highlightable"])
            self.assertTrue(row["bbox_ids"])
            self.assertTrue(row["text_span_ids"])
        consolidated = rows["uud_metadata_block_final_evidence::current_consolidated::source_publication_page_0001"]
        self.assertEqual(consolidated["bbox_precision"], "exact")
        self.assertTrue(consolidated["viewer_highlightable"])
        self.assertTrue(consolidated["bbox_ids"])

    def test_ordinal_exclusion_artifact_is_empty_after_grounding_admission(self) -> None:
        self.assertEqual(read_jsonl(FINAL / "excluded_records.jsonl"), [])


def _span_bbox_key(row: dict) -> tuple[object, ...]:
    return (
        row.get("source_document_id"),
        row.get("source_sha256"),
        row.get("page_number"),
        row.get("text"),
        row.get("x0"),
        row.get("y0"),
        row.get("x1"),
        row.get("y1"),
    )


def _raw_metadata_block_contract(row: dict) -> dict:
    return {
        "corpus_id": row["corpus_id"],
        "metadata_grounding_id": row["metadata_grounding_id"],
        "page_numbers": row["page_numbers"],
        "provenance": row["provenance"],
        "quoted_text": row["quoted_text"],
        "source_document_id": row["source_document_id"],
        "source_pdf_path": row["source_pdf_path"],
        "source_role": row["source_role"],
        "source_sha256": row["source_sha256"],
        "status": row["status"],
        "temporal_context": row["temporal_context"],
    }


def _artifact_hashes(final: Path) -> dict[str, str]:
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(final.iterdir()) if path.is_file()}


def _canonicalize_numeric_rows(rows: list[dict]) -> list[dict]:
    return [_canonicalize_numeric(row) for row in rows]


def _canonicalize_numeric(value):
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {key: _canonicalize_numeric(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_canonicalize_numeric(item) for item in value]
    return value


if __name__ == "__main__":
    unittest.main()
