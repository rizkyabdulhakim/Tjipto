from __future__ import annotations

from pathlib import Path
from hashlib import sha256
import json
import re
import tempfile
import unittest

from tjipto.corpora import parser_dispatch
from tjipto.corpora.strategy import (
    CorpusStrategy,
    NavigationResolver,
    QueryNormalizer,
    ReferenceParser,
    StrategyRegistry,
)
from tjipto.corpora.uud import parser as uud_parser
from tjipto.corpora.uud.relation_builder import parse_renumbering_mappings
from tjipto.contracts.artifacts import CURRENT_ARTIFACT_SCHEMA
from tjipto.core.manifest import artifact_set_digest
from tjipto.runtime.service import LegalRuntimeService


ROOT = Path(__file__).resolve().parents[1]


class CorpusBoundaryContractTest(unittest.TestCase):
    def test_generic_layers_do_not_define_uud_legal_regex(self) -> None:
        for rel_path, names in {
            "src/tjipto/retrieval/query.py": ("PASAL_RE", "AYAT_RE", "PASAL_SHORTHAND_AYAT_RE"),
            "src/tjipto/retrieval/structured.py": ("BAB_RE", "PASAL_RE", "AYAT_RE"),
            "src/tjipto/retrieval/relations.py": ("BAB_RE", "PASAL_RE"),
            "src/tjipto/evidence/citation.py": ("PASAL_RE", "AYAT_RE"),
        }.items():
            source = (ROOT / rel_path).read_text(encoding="utf-8")
            for name in names:
                self.assertIsNone(re.search(rf"(?m)^{name}\s*=", source), rel_path)

    def test_uud_parser_owns_legal_reference_helpers(self) -> None:
        self.assertEqual(uud_parser.parse_uud_bab_reference("BAB XA"), "BAB XA")
        self.assertEqual(uud_parser.parse_uud_pasal_reference("ayat (1) Pasal 28"), "Pasal 28")
        self.assertEqual(uud_parser.parse_uud_ayat_reference("Pasal 28 ayat (1)"), "(1)")
        self.assertEqual(
            uud_parser.normalize_uud_query_reference("pasal 28 (1)"),
            "Pasal 28 ayat (1)",
        )

    def test_parser_dispatch_resolves_uud_and_fails_safely(self) -> None:
        strategy = parser_dispatch.get_strategy("uud")
        self.assertIs(strategy.normalizer.normalize_query_reference, uud_parser.normalize_uud_query_reference)
        self.assertEqual(parser_dispatch.normalize_query_reference("uud", "pasal 28 (1)"), "Pasal 28 ayat (1)")
        self.assertEqual(
            parser_dispatch.parse_legal_reference("uud", "BAB XA Pasal 28 ayat (1)"),
            {"bab": "BAB XA", "pasal": "Pasal 28", "ayat": "(1)"},
        )
        self.assertIn("bab xa", parser_dispatch.label_keys("uud", "BAB X A"))
        with self.assertRaisesRegex(ValueError, "unsupported_corpus_parser:unknown"):
            parser_dispatch.get_strategy("unknown")

    def test_second_strategy_uses_its_own_reference_syntax(self) -> None:
        normalizer = QueryNormalizer(
            normalize_query_reference=lambda text: text.strip(),
            normalize_metadata_intent=lambda text: text.casefold(),
        )
        references = ReferenceParser(
            parse_legal_reference=lambda text, **_: {"rule": text if text.startswith("Rule ") else None},
            parse_legal_references=lambda text: [{"reference": text}] if text.startswith("Rule ") else [],
            parse_bab_reference=lambda _: None,
            parse_pasal_reference=lambda *_args, **_kwargs: None,
            parse_ayat_reference=lambda _: None,
            label_keys=lambda value: {str(value).casefold()},
        )
        navigation = NavigationResolver(
            resolve_navigation=lambda _: None,
        )
        strategy = StrategyRegistry({"demo": CorpusStrategy("demo", normalizer, references, navigation, lambda _: None)}).require("demo")
        self.assertEqual(strategy.references.parse_legal_reference("Rule 1"), {"rule": "Rule 1"})
        self.assertIsNone(strategy.proposition_operator("Rule 1 requires payment"))

    def test_second_strategy_executes_verified_runtime_retrieval(self) -> None:
        normalizer = QueryNormalizer(
            normalize_query_reference=lambda text: text.replace("RULE-", "Rule "),
            normalize_metadata_intent=lambda text: text.casefold(),
        )
        references = ReferenceParser(
            parse_legal_reference=lambda text, **_: {"rule": text if text.startswith("Rule ") else None},
            parse_legal_references=lambda text: [{"reference": text}] if text.startswith("Rule ") else [],
            parse_bab_reference=lambda _: None,
            parse_pasal_reference=lambda *_args, **_kwargs: None,
            parse_ayat_reference=lambda _: None,
            label_keys=lambda value: {str(value).casefold()},
        )
        navigation = NavigationResolver(
            resolve_navigation=lambda _: None,
        )
        strategies = StrategyRegistry({"demo": CorpusStrategy("demo", normalizer, references, navigation, lambda _: None)})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            final = root / "data/final/demo"
            final.mkdir(parents=True)
            source_id, span_id, bbox_id = "demo::published", "demo_span_1", "demo_bbox_1"
            evidence = {
                "evidence_id": "demo_evidence_1", "corpus_id": "demo", "legal_unit_id": "demo_rule_1",
                "citation": "Rule 1", "quoted_text": "Rule 1 requires payment.", "hierarchy": ["Rule 1"],
                "source_document_id": source_id, "source_pdf_path": "data/sources/demo.pdf", "source_sha256": "demo-sha",
                "source_role": "published", "temporal_context": "current", "page_numbers": [1], "status": "final",
                "runtime_loadable": True, "bbox_precision": "exact", "viewer_highlightable": True,
                "bbox_refs": [bbox_id], "bbox_ids": [bbox_id], "text_span_ids": [span_id],
                "authority_kind": "normative_legal_text", "citation_final": True, "citable": True,
                "evidence_owner_kind": "legal_unit_source", "relevant_quote_eligible": True,
            }
            source = {"source_document_id": source_id, "sha256": "demo-sha", "path": "data/sources/demo.pdf", "source_role": "published", "temporal_context": "current"}
            span = {"text_span_id": span_id, "source_document_id": source_id, "source_sha256": "demo-sha", "page_number": 1, "text": evidence["quoted_text"]}
            bbox = {"bbox_id": bbox_id, "source_document_id": source_id, "source_sha256": "demo-sha", "page_number": 1, "text": evidence["quoted_text"]}
            projection = {"schema": 1, "artifacts": {
                "evidence_registry": [evidence], "bbox_registry": [bbox], "legal_units": [], "chunks": [], "retrieval_units": [],
                "graph_edges": [], "source_documents": [source], "page_text_spans": [span], "document_metadata": [],
                "metadata_grounding": [], "metadata_grounding_registry": [], "document_relations": [],
                "article_amendment_relations": [], "source_conflicts": [],
            }}
            payloads = {"runtime_projection.json": json.dumps(projection).encode(), "propositions.jsonl": b""}
            for name, data in payloads.items():
                (final / name).write_bytes(data)
            files = {
                name: {
                    "logical_key": key, "artifact_kind": key, "format": "json" if name.endswith(".json") else "jsonl",
                    "artifact_schema": CURRENT_ARTIFACT_SCHEMA, "origin": "generated", "producer": "test", "build_stage": "test",
                    "sha256": sha256(data).hexdigest(), "bytes": len(data), "primary_id": "proposition_id" if key == "propositions" else None,
                }
                for name, key, data in (
                    ("runtime_projection.json", "runtime_projection", payloads["runtime_projection.json"]),
                    ("propositions.jsonl", "propositions", payloads["propositions.jsonl"]),
                    ("validation_report.json", "validation_report", b""),
                )
            }
            manifest = {
                "corpus_id": "demo", "schema_version": CURRENT_ARTIFACT_SCHEMA,
                "runtime_projection": "runtime_projection.json", "propositions": "propositions.jsonl", "validation_report": "validation_report.json", "files": files,
            }
            report = {"status": "pass", "validated_artifact_set_digest": artifact_set_digest(manifest, exclude=("validation_report.json",))}
            report_data = json.dumps(report).encode()
            (final / "validation_report.json").write_bytes(report_data)
            files["validation_report.json"].update({"sha256": sha256(report_data).hexdigest(), "bytes": len(report_data)})
            manifest_data = json.dumps(manifest).encode()
            (final / "manifest.json").write_bytes(manifest_data)
            (root / "data/corpus_registry.json").write_text(json.dumps({"demo": {
                "manifest": "data/final/demo/manifest.json", "manifest_sha256": sha256(manifest_data).hexdigest(),
                "runtime_required_artifacts": ["runtime_projection", "validation_report", "propositions"],
                "query_normalization_enabled": True, "exact_citation_intent_enabled": False,
                "source_roles": ["published"], "temporal_contexts": ["current"], "preferred_source_role": "published",
            }}), encoding="utf-8")
            result = LegalRuntimeService(root, strategy_registry=strategies).ask("demo", "RULE-1 payment")
            self.assertEqual(result["status"], "limited_answer")
            self.assertEqual(result["normalized_query"], "Rule 1 payment")
            self.assertEqual(result["evidence"][0]["citation"], "Rule 1")
            service = LegalRuntimeService(root, strategy_registry=strategies)
            for query in ("RULE-1 requires payment", "RULE-1 amended by Rule 2"):
                with self.subTest(query=query):
                    self.assertNotEqual(service.ask("demo", query)["status"], "answer_ready")

    def test_parser_dispatch_preserves_all_reference_ranges(self) -> None:
        text = "Pasal19, Pasal\n28C, dan pasal 28G."
        rows = parser_dispatch.parse_legal_references("uud", text)
        self.assertEqual([row["reference"] for row in rows], ["Pasal 19", "Pasal 28C", "Pasal 28G"])
        self.assertEqual([text[int(row["start"]) : int(row["end"])] for row in rows], ["Pasal19", "Pasal\n28C", "pasal 28G"])

    def test_parser_dispatch_retains_paragraph_reference(self) -> None:
        text = "Pasal 3 ayat (3) menjadi Pasal 3 ayat (2)"
        rows = parser_dispatch.parse_legal_references("uud", text)
        self.assertEqual([row["reference"] for row in rows], ["Pasal 3 ayat (3)", "Pasal 3 ayat (2)"])
        self.assertEqual([text[int(row["start"]) : int(row["end"])] for row in rows], ["Pasal 3 ayat (3)", "Pasal 3 ayat (2)"])

    def test_renumbering_parser_preserves_paragraph_level_pairs(self) -> None:
        text = (
            "Pengubahan penomoran Pasal 3 ayat (3) dan ayat (4) Perubahan Ketiga "
            "Undang-Undang Dasar Negara Republik Indonesia Tahun 1945 menjadi "
            "Pasal 3 ayat (2) dan ayat (3); Pasal 25E Perubahan Kedua "
            "Undang-Undang Dasar Negara Republik Indonesia Tahun 1945 menjadi Pasal 25A."
        )
        rows = parse_renumbering_mappings(text)
        self.assertEqual(
            [(row["old_reference"], row["new_reference"]) for row in rows],
            [
                ("Pasal 3 ayat (3)", "Pasal 3 ayat (2)"),
                ("Pasal 3 ayat (4)", "Pasal 3 ayat (3)"),
                ("Pasal 25E", "Pasal 25A"),
            ],
        )
        self.assertTrue(all(text[row["old_range"][0] : row["old_range"][1]] for row in rows))
        self.assertEqual(text[rows[1]["old_range"][0] : rows[1]["old_range"][1]], "Pasal 3 ayat (3) dan ayat (4)")
        self.assertEqual(rows[1]["old_range_kind"], "contextual")
        self.assertEqual(rows[2]["new_range_kind"], "literal")

    def test_coordinated_article_scope_inherits_parent_for_bare_paragraphs(self) -> None:
        text = (
            "Perubahan Pertama mengubah Pasal 13 ayat (2) dan (3), "
            "Pasal 17 ayat (2) dan (3) menjadi Pasal 13 ayat (2) dan (3), "
            "Pasal 17 ayat (2) dan (3);"
        )
        rows = parse_renumbering_mappings(text)
        self.assertEqual(
            [(row["old_reference"], row["new_reference"]) for row in rows],
            [
                ("Pasal 13 ayat (2)", "Pasal 13 ayat (2)"),
                ("Pasal 13 ayat (3)", "Pasal 13 ayat (3)"),
                ("Pasal 17 ayat (2)", "Pasal 17 ayat (2)"),
                ("Pasal 17 ayat (3)", "Pasal 17 ayat (3)"),
            ],
        )

    def test_generic_layers_use_parser_dispatch_not_uud_parser(self) -> None:
        for rel_path in (
            "src/tjipto/retrieval/query.py",
            "src/tjipto/retrieval/structured.py",
            "src/tjipto/retrieval/relations.py",
            "src/tjipto/evidence/citation.py",
        ):
            source = (ROOT / rel_path).read_text(encoding="utf-8")
            self.assertNotIn("tjipto.corpora.uud.parser", source, rel_path)

    def test_parser_dispatch_uses_adapter_not_uud_named_calls(self) -> None:
        source = (ROOT / "src/tjipto/corpora/parser_dispatch.py").read_text(encoding="utf-8")
        for leak in ("normalize_uud", "parse_uud", "uud_label"):
            self.assertNotIn(leak, source)

    def test_generic_provenance_has_no_uud_header_logic(self) -> None:
        source = (ROOT / "src/tjipto/corpora/provenance.py").read_text(encoding="utf-8")
        self.assertNotIn("UUD_SATU_NASKAH_HEADER_RE", source)
        self.assertNotIn("_strip_uud_header", source)
        self.assertNotIn("Perubahan Pertama", source)

    def test_runtime_does_not_parse_source_metadata_from_id_shape(self) -> None:
        source = (ROOT / "src/tjipto/runtime/service.py").read_text(encoding="utf-8")
        self.assertNotIn('source_document_id") or "").split("::")', source)
        self.assertNotIn("source_document_id.split", source)


if __name__ == "__main__":
    unittest.main()
