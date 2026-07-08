from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import unicodedata
import unittest

from tjipto.corpora.adapter import config_for
from tjipto.corpora.uud_artifact_baseline import validate_uud_artifact_baseline
from tjipto.corpora.uud.specs import UUD_INSERTED_BAB_PREDECESSORS, UUD_LEGAL_GRAPH_EDGE_SCHEMA
from tjipto.core.manifest import file_sha256
from tjipto.graph.store import GraphStore
from tjipto.core.manifest import read_jsonl
from tjipto.corpora.reproducibility import validate_corpus_ingestion_artifacts
from tjipto.corpora.uud_reproducibility import validate_uud_ingestion_artifacts


ROOT = Path(__file__).resolve().parents[1]
INSERTED_BAB_PREDECESSORS = {
    "BAB VIIA": ("BAB VII",),
    "BAB VIIB": ("BAB VIIA", "BAB VII"),
    "BAB VIIIA": ("BAB VIII",),
    "BAB IXA": ("BAB IX",),
    "BAB XA": ("BAB X",),
}
SEQUENCE_EDGE_TYPES = {"INSERTED_AFTER", "PRECEDES", "FOLLOWS"}


def _article_for(row: dict) -> str | None:
    values = [
        str(row.get("unit_label") or ""),
        str(row.get("citation") or ""),
        " ".join(row.get("hierarchy") or ()),
        str(row.get("text") or ""),
        str(row.get("quoted_text") or ""),
    ]
    match = re.search(
        r"\bPasal\s+(22C|22D|22E|23E|23F|23G|25A|25E|28[A-J]|23B|23D|24|37)\b",
        " ".join(values),
    )
    return f"Pasal {match.group(1)}" if match else None


class GraphContractTest(unittest.TestCase):
    def test_graph_counts_match_manifest(self) -> None:
        graph = GraphStore(config_for("uud", ROOT))
        manifest = json.loads((ROOT / "data/final/uud/manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(
            graph.counts(),
            {"nodes": manifest["counts"]["graph_nodes"], "edges": manifest["counts"]["graph_edges"]},
        )

    def test_article_versions_are_source_scoped_not_equivalence_claims(self) -> None:
        rows = read_jsonl(ROOT / "data/final/uud/article_versions.jsonl")
        self.assertEqual(len(rows), 218)
        for row in rows:
            self.assertIn("not_accepted_legal_equivalence", row["cross_source_equivalence_status"])
            self.assertTrue(row["members"])
            for member in row["members"]:
                self.assertIn("chunk_id", member)
                self.assertNotIn("chunk_candidate_id", member)

    def test_source_conflicts_reference_source_documents(self) -> None:
        source_ids = {row["source_document_id"] for row in read_jsonl(ROOT / "data/final/uud/source_documents.jsonl")}
        rows = read_jsonl(ROOT / "data/final/uud/source_conflicts.jsonl")
        self.assertEqual(len(rows), 2)
        conflict_ids = {row["source_conflict_id"] for row in rows}
        self.assertIn(
            "uud_1945_amendment_2_pasal_25e_current_pasal_25a_renumbering_conflict",
            conflict_ids,
        )
        for row in rows:
            self.assertIn(row["source_document_id"], source_ids)
            self.assertNotEqual(row["status"], "unresolved_review_required_required")

    def test_inserted_bab_policy_and_schema_live_in_specs(self) -> None:
        graph_builder_source = (ROOT / "src/tjipto/corpora/uud/graph_builder.py").read_text(encoding="utf-8")
        self.assertNotIn("INSERTED_BAB_PREDECESSORS = {", graph_builder_source)
        self.assertEqual(UUD_INSERTED_BAB_PREDECESSORS, INSERTED_BAB_PREDECESSORS)
        for edge_type in SEQUENCE_EDGE_TYPES:
            schema = UUD_LEGAL_GRAPH_EDGE_SCHEMA[edge_type]
            self.assertEqual(schema["category"], "structural_sequence")
            self.assertFalse(schema["hierarchy_edge"])
            self.assertFalse(schema["runtime_loadable"])
            self.assertEqual(schema["validation_status"], "accepted_structural_sequence")
            self.assertEqual(schema["derivation_basis"], "structural_order")

    def test_inserted_bab_hierarchy_is_consistent(self) -> None:
        final = ROOT / "data/final/uud"
        units = {row["legal_unit_id"]: row for row in read_jsonl(final / "legal_units.jsonl")}
        evidence = read_jsonl(final / "evidence_registry.jsonl")
        chunks = {row["chunk_id"]: row for row in read_jsonl(final / "chunks.jsonl")}
        retrieval = read_jsonl(final / "retrieval_units.jsonl")
        nodes_by_unit = {
            row["legal_unit_id"]: row for row in read_jsonl(final / "graph_nodes.jsonl") if row.get("node_type") == "legal_unit"
        }

        expected = {
            ("current_consolidated", "Pasal 22C"): "BAB VIIA",
            ("current_consolidated", "Pasal 22D"): "BAB VIIA",
            ("current_consolidated", "Pasal 22E"): "BAB VIIB",
            ("current_consolidated", "Pasal 23E"): "BAB VIIIA",
            ("current_consolidated", "Pasal 23F"): "BAB VIIIA",
            ("current_consolidated", "Pasal 23G"): "BAB VIIIA",
            ("current_consolidated", "Pasal 25A"): "BAB IXA",
            ("amendment_2_historical", "Pasal 25E"): "BAB IXA",
            ("amendment_3_historical", "Pasal 22C"): "BAB VIIA",
            ("amendment_3_historical", "Pasal 22D"): "BAB VIIA",
            ("amendment_3_historical", "Pasal 22E"): "BAB VIIB",
            ("amendment_3_historical", "Pasal 23E"): "BAB VIIIA",
            ("amendment_3_historical", "Pasal 23F"): "BAB VIIIA",
            ("amendment_3_historical", "Pasal 23G"): "BAB VIIIA",
        }
        for letter in "ABCDEFGHIJ":
            expected[("current_consolidated", f"Pasal 28{letter}")] = "BAB XA"
            expected[("amendment_2_historical", f"Pasal 28{letter}")] = "BAB XA"

        for row in units.values():
            if row["unit_type"] == "bab_record":
                continue
            role = str(row["source_document_id"].split("::", 1)[1])
            article = _article_for(row)
            if article is not None and (role, article) in expected:
                self.assertEqual(row["hierarchy"][0], expected[(role, article)], row["legal_unit_id"])
                self.assertNotIn("BAB VII", row["hierarchy"][:1], row["legal_unit_id"])
                self.assertNotIn("BAB X", row["hierarchy"][:1], row["legal_unit_id"])
                node = nodes_by_unit.get(row["legal_unit_id"])
                if node:
                    self.assertEqual(node["hierarchy_path"][0], expected[(role, article)], row["legal_unit_id"])

        for row in evidence:
            role = str(row["source_role"])
            article = _article_for(row)
            if article is not None and (role, article) in expected:
                self.assertEqual(row["hierarchy"][0], expected[(role, article)], row["evidence_id"])
        for row in retrieval:
            evidence_row = next(item for item in evidence if item["evidence_id"] == row["evidence_id"])
            article = _article_for(evidence_row)
            key = (str(row["source_role"]), article)
            if article is not None and key in expected:
                self.assertIn(expected[key], row["text"], row["retrieval_unit_id"])
                self.assertEqual(chunks[row["chunk_id"]]["hierarchy"][0], expected[key], row["chunk_id"])

    def test_inserted_bab_graph_uses_sibling_sequence_not_parent_child(self) -> None:
        final = ROOT / "data/final/uud"
        nodes = {row["node_id"]: row for row in read_jsonl(final / "graph_nodes.jsonl")}
        bab_nodes = {
            (row["source_document_id"], row["unit_label"]): row["node_id"]
            for row in nodes.values()
            if row.get("node_type") == "legal_unit" and row.get("unit_type") == "bab_record"
        }
        edges = read_jsonl(final / "graph_edges.jsonl")
        edge_ids = [row["edge_id"] for row in edges]
        self.assertEqual(len(edge_ids), len(set(edge_ids)))
        for edge in edges:
            self.assertIn(edge["source_id"], nodes, edge["edge_id"])
            self.assertIn(edge["target_id"], nodes, edge["edge_id"])

        false_contains = []
        false_part_of = []
        for edge in edges:
            source = nodes[edge["source_id"]]
            target = nodes[edge["target_id"]]
            if source.get("source_document_id") != target.get("source_document_id"):
                continue
            source_label = source.get("unit_label")
            target_label = target.get("unit_label")
            if edge["edge_type"] == "CONTAINS" and source_label in INSERTED_BAB_PREDECESSORS.get(target_label, ()):
                false_contains.append(edge)
            if edge["edge_type"] == "PART_OF" and target_label in INSERTED_BAB_PREDECESSORS.get(source_label, ()):
                false_part_of.append(edge)
        self.assertEqual(false_contains, [])
        self.assertEqual(false_part_of, [])

        edge_keys = {(row["edge_type"], row["source_id"], row["target_id"], row.get("source_document_id")) for row in edges}
        for (source_document_id, inserted_label), inserted_node in bab_nodes.items():
            predecessors = INSERTED_BAB_PREDECESSORS.get(inserted_label)
            if not predecessors:
                continue
            predecessor_node = next(
                (bab_nodes[(source_document_id, label)] for label in predecessors if (source_document_id, label) in bab_nodes),
                None,
            )
            if not predecessor_node:
                continue
            self.assertIn(("PRECEDES", predecessor_node, inserted_node, source_document_id), edge_keys)
            self.assertIn(("FOLLOWS", inserted_node, predecessor_node, source_document_id), edge_keys)
            self.assertIn(("INSERTED_AFTER", inserted_node, predecessor_node, source_document_id), edge_keys)

        self._assert_contains(final, "current_consolidated", "BAB VIIA", "Pasal 22C")
        self._assert_contains(final, "current_consolidated", "BAB VIIA", "Pasal 22D")

    def test_inserted_bab_sequence_edges_follow_schema_and_source_scope(self) -> None:
        final = ROOT / "data/final/uud"
        nodes = {row["node_id"]: row for row in read_jsonl(final / "graph_nodes.jsonl")}
        edges = [row for row in read_jsonl(final / "graph_edges.jsonl") if row["edge_type"] in SEQUENCE_EDGE_TYPES]
        self.assertEqual(len(edges), 21)
        edge_keys = {(row["edge_type"], row["source_id"], row["target_id"]) for row in edges}
        for edge in edges:
            source = nodes[edge["source_id"]]
            target = nodes[edge["target_id"]]
            self.assertEqual(edge["relation_type"], edge["edge_type"])
            self.assertEqual(edge["source_document_id"], source["source_document_id"])
            self.assertEqual(edge["source_document_id"], target["source_document_id"])
            self.assertEqual(edge["source_role"], source["source_role"])
            self.assertEqual(edge["temporal_context"], source["source_role"])
            self.assertFalse(edge["runtime_loadable"])
            self.assertEqual(edge["validation_status"], "accepted_structural_sequence")
            self.assertEqual(edge["derivation_basis"], "structural_order")
            self.assertEqual(edge["confidence_policy"], "inserted_bab_sibling_sequence_artifact")
            if edge["edge_type"] == "INSERTED_AFTER":
                self.assertIn(("FOLLOWS", edge["source_id"], edge["target_id"]), edge_keys)
                self.assertIn(("PRECEDES", edge["target_id"], edge["source_id"]), edge_keys)
            if edge["edge_type"] == "PRECEDES":
                self.assertIn(("FOLLOWS", edge["target_id"], edge["source_id"]), edge_keys)

    def test_amendment_4_does_not_invent_missing_bab_headings(self) -> None:
        for filename in ("legal_units.jsonl", "chunks.jsonl", "evidence_registry.jsonl"):
            for row in read_jsonl(ROOT / "data/final/uud" / filename):
                if row.get("chunk_type") == "bab_structural_context_record":
                    continue
                role = row.get("source_role") or row.get("temporal_context") or row.get("source_document_id", "").split("::")[-1]
                article = _article_for(row)
                if role == "amendment_4_historical" and article in {"Pasal 23B", "Pasal 23D", "Pasal 24", "Pasal 37"}:
                    self.assertFalse((row.get("hierarchy") or [""])[0].startswith("BAB"), (filename, row))

    def test_validation_artifacts_resolve_refs(self) -> None:
        source_ids = {row["source_document_id"] for row in read_jsonl(ROOT / "data/final/uud/source_documents.jsonl")}
        chunk_ids = {row["chunk_id"] for row in read_jsonl(ROOT / "data/final/uud/chunks.jsonl")}
        unit_ids = {row["legal_unit_id"] for row in read_jsonl(ROOT / "data/final/uud/legal_units.jsonl")}
        alignment = read_jsonl(ROOT / "data/final/uud/validation_alignment_results.jsonl")
        self.assertEqual(len(alignment), 610)
        for row in alignment:
            if row.get("legal_unit_id"):
                self.assertIsNotNone(row.get("source_document_id"))
            if row.get("chunk_id"):
                self.assertIn(row["chunk_id"], chunk_ids)
            if row.get("legal_unit_id"):
                self.assertIn(row["legal_unit_id"], unit_ids)
            if row.get("source_document_id"):
                self.assertIn(row["source_document_id"], source_ids)
        exceptions = read_jsonl(ROOT / "data/final/uud/validation_exceptions.jsonl")
        self.assertEqual(len(exceptions), 19)
        for row in exceptions:
            if row.get("source_document_id"):
                self.assertIn(row["source_document_id"], source_ids)
            if row.get("chunk_id"):
                self.assertIn(row["chunk_id"], chunk_ids)

    def test_uud_baseline_validator_passes(self) -> None:
        self.assertEqual(validate_uud_artifact_baseline(ROOT), ())

    def test_metadata_graph_edges_exclude_source_role_level_amends(self) -> None:
        edges = read_jsonl(ROOT / "data/final/uud/metadata_graph_edges.jsonl")
        self.assertEqual(len(edges), 449)
        self.assertFalse([edge for edge in edges if edge["edge_type"] in {"AMENDS", "AMENDED_BY"}])
        self.assertTrue(all(edge["status"] == "accepted" for edge in edges))
        self.assertTrue(all(edge["runtime_loadable"] is False for edge in edges))

    def test_amends_edges_are_preserved_as_not_promoted_exceptions(self) -> None:
        graph_edges = read_jsonl(ROOT / "data/final/uud/graph_edges.jsonl")
        self.assertFalse([row for row in graph_edges if row["edge_type"] in {"AMENDS", "AMENDED_BY"}])
        exceptions = read_jsonl(ROOT / "data/final/uud/validation_exceptions.jsonl")
        amends = [row for row in exceptions if row.get("edge_type") in {"AMENDS", "AMENDED_BY"}]
        self.assertEqual(len(amends), 8)
        for row in amends:
            self.assertEqual(row["status"], "not_promoted_source_role_level_only")
            self.assertFalse(row["runtime_loadable"])

    def test_document_relation_artifact_preserves_not_promoted_document_edges(self) -> None:
        source_ids = {row["source_document_id"] for row in read_jsonl(ROOT / "data/final/uud/source_documents.jsonl")}
        exception_ids = {row["exception_id"] for row in read_jsonl(ROOT / "data/final/uud/validation_exceptions.jsonl")}
        rows = read_jsonl(ROOT / "data/final/uud/document_relations.jsonl")
        self.assertEqual(len(rows), 8)
        self.assertEqual(len({row["relation_id"] for row in rows}), len(rows))
        self.assertEqual(sum(1 for row in rows if row["relation_type"] == "AMENDS"), 4)
        self.assertEqual(sum(1 for row in rows if row["relation_type"] == "AMENDED_BY"), 4)
        for row in rows:
            self.assertIn(row["source_document_id"], source_ids)
            self.assertIn(row["target_document_id"], source_ids)
            self.assertFalse(row["article_level"])
            self.assertFalse(row["viewer_highlightable"])
            self.assertFalse(row["citation_available"])
            for ref in row["support_refs"]:
                self.assertIn(ref, exception_ids)

    def test_article_amendment_relation_artifact_separates_exact_and_trace_support(self) -> None:
        evidence = {row["evidence_id"]: row for row in read_jsonl(ROOT / "data/final/uud/evidence_registry.jsonl")}
        bbox_ids = {row["bbox_id"] for row in read_jsonl(ROOT / "data/final/uud/bbox_registry.jsonl")}
        units = {row["legal_unit_id"]: row for row in read_jsonl(ROOT / "data/final/uud/legal_units.jsonl")}
        rows = read_jsonl(ROOT / "data/final/uud/article_amendment_relations.jsonl")
        health = json.loads((ROOT / "data/final/uud/validation_report.json").read_text(encoding="utf-8"))[
            "article_relation_runtime_policy_health"
        ]
        self.assertTrue(rows)
        self.assertEqual(len({row["relation_id"] for row in rows}), len(rows))
        self.assertFalse([row for row in rows if row["relation_type"] in {"ADDS", "RENAMES", "SUPPLEMENTS"}])
        exact_rows = [row for row in rows if row["support_class"] == "exact_article_relation"]
        trace_rows = [row for row in rows if row["support_class"] == "trace_article_relation"]
        self.assertEqual(len(exact_rows), 32)
        self.assertEqual(len(trace_rows), 31)
        self.assertEqual(health["article_relation_total_count"], len(rows))
        self.assertEqual(health["article_relation_exact_support_count"], len(exact_rows))
        self.assertEqual(health["article_relation_trace_only_count"], len(trace_rows))
        self.assertEqual(health["article_relation_trace_missing_reason_count"], 0)
        self.assertEqual(
            health["article_relation_trace_reason_counts"],
            {"instrument_trace_only_not_public_citation": 31},
        )
        self.assertEqual(health["article_relation_invalid_bbox_refs"], 0)
        self.assertEqual(health["article_relation_invalid_coordinates"], 0)
        self.assertEqual(health["article_relation_partial_answer_risk_count"], 1)
        self.assertEqual(health["relation_runtime_policy_slow_gate_status"], "covered_by_runtime_policy_test")
        for row in rows:
            source = evidence[row["evidence_id"]]
            self.assertIn(row["target_legal_unit_id"], units)
            self.assertEqual(row["quoted_text"], source["quoted_text"])
            self.assertTrue(row["runtime_loadable"])
            self.assertTrue(row["bbox_refs"])
            for bbox_id in row["bbox_refs"]:
                self.assertIn(bbox_id, bbox_ids)
            if row["support_class"] == "exact_article_relation":
                self.assertEqual(row["grounding_level"], "exact_source_text")
                self.assertEqual(row["bbox_precision"], "exact")
                self.assertTrue(row["viewer_highlightable"])
                self.assertTrue(row["citation_available"])
                self.assertEqual(source["bbox_precision"], "exact")
                self.assertTrue(source["viewer_highlightable"])
            else:
                self.assertEqual(row["grounding_level"], "page_grounded_trace")
                self.assertEqual(row["trace_only_reason"], "instrument_trace_only_not_public_citation")
                self.assertFalse(row["viewer_highlightable"])
                self.assertFalse(row["citation_available"])

    def test_graph_edges_include_evidence_backed_legal_baseline(self) -> None:
        edges = read_jsonl(ROOT / "data/final/uud/graph_edges.jsonl")
        report = json.loads((ROOT / "data/final/uud/validation_report.json").read_text(encoding="utf-8"))["legal_graph_baseline"]
        nodes = {row["node_id"] for row in read_jsonl(ROOT / "data/final/uud/graph_nodes.jsonl")}
        actual_counts: dict[str, int] = {}
        for row in edges:
            actual_counts[row["edge_type"]] = actual_counts.get(row["edge_type"], 0) + 1
        self.assertEqual(report["status"], "evidence_backed_minimal_baseline")
        self.assertEqual(report["actual_edge_type_counts"], dict(sorted(actual_counts.items())))
        self.assertNotIn("legal_edge_types", report)
        self.assertEqual(set(report["not_promoted_edge_types"]), {"ADDS", "AMENDED_BY", "AMENDS", "RENAMES", "SUPPLEMENTS"})
        for edge_type in report["not_promoted_edge_types"]:
            self.assertNotIn(edge_type, report["actual_edge_type_counts"])
        for edge_type in {"ADDS", "AMENDED_BY", "AMENDS", "RENAMES", "SUPPLEMENTS"}:
            self.assertNotIn(edge_type, report["actual_promoted_legal_edge_type_counts"])
            self.assertIn(edge_type, report["schema_edge_types"])
        legal_edges = [
            row
            for row in edges
            if row.get("edge_type")
            in {
                "CONTAINS",
                "PART_OF",
                "MODIFIES",
                "DELETES",
                "HAS_EFFECTIVE_RULE",
                "HAS_SIGNATORY",
                "HAS_DECISION_SESSION",
                "HAS_SOURCE_ANOMALY",
            }
        ]
        self.assertTrue(legal_edges)
        self.assertTrue(any(row["edge_type"] == "DELETES" for row in legal_edges))
        self.assertTrue(any(row["edge_type"] == "HAS_SIGNATORY" for row in legal_edges))
        self.assertTrue(any(row["edge_type"] == "HAS_DECISION_SESSION" for row in legal_edges))
        self.assertTrue(any(row["edge_type"] == "HAS_EFFECTIVE_RULE" for row in legal_edges))
        self.assertTrue(any(row["edge_type"] == "HAS_SOURCE_ANOMALY" for row in legal_edges))
        self.assertFalse(
            [row for row in legal_edges if row["edge_type"] in {"AMENDS", "AMENDED_BY"} and row["source_id"].startswith("source_role::")]
        )
        for row in legal_edges:
            self.assertIn(row["source_id"], nodes)
            self.assertIn(row["target_id"], nodes)
            self.assertEqual(row["relation_type"], row["edge_type"])
            self.assertTrue(row.get("source_document_id"))
            if row.get("runtime_loadable") is True:
                self.assertTrue(row.get("evidence_ref"))
                self.assertTrue(row.get("validation_status"))
                self.assertTrue(row.get("confidence_policy"))

    def test_uud_ingestion_artifacts_are_consistent(self) -> None:
        import fitz

        final = ROOT / "data/final/uud"
        stale_path = "data/sources/constitutional/uud/bpk"

        for path in final.iterdir():
            if path.is_file():
                self.assertNotIn(stale_path, path.read_text(encoding="utf-8"), path.name)

        manifest = json.loads((final / "manifest.json").read_text(encoding="utf-8"))
        for rel, expected in manifest["files"].items():
            path = final / rel
            self.assertEqual(path.stat().st_size, expected["bytes"], rel)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected["sha256"], rel)

        source_docs = {row["source_document_id"]: row for row in read_jsonl(final / "source_documents.jsonl")}
        pdfs = {}
        pdf_text = {}
        for source_id, row in source_docs.items():
            pdf_path = ROOT / row["path"]
            self.assertTrue(pdf_path.exists(), row["path"])
            self.assertEqual(file_sha256(pdf_path), row["sha256"])
            self.assertEqual(pdf_path.stat().st_size, row["file_size"])
            pdf = fitz.open(pdf_path)
            pdfs[source_id] = pdf
            self.assertEqual(pdf.page_count, row["page_count"])
            pdf_text[source_id] = {
                page_number: self._compact_text(pdf[page_number - 1].get_text()) for page_number in range(1, pdf.page_count + 1)
            }

        pages = read_jsonl(final / "pages.jsonl")
        page_keys = {(row["source_document_id"], row["page_number"]) for row in pages}
        page_text = {}
        for row in pages:
            key = (row["source_document_id"], row["page_number"])
            self.assertIn(row["source_document_id"], source_docs)
            self.assertLessEqual(row["page_number"], pdfs[row["source_document_id"]].page_count)
            page_text[key] = self._compact_text(row["text"])
            self.assertIn(page_text[key], pdf_text[row["source_document_id"]][row["page_number"]])

        legal_units = {row["legal_unit_id"]: row for row in read_jsonl(final / "legal_units.jsonl")}
        for row in legal_units.values():
            self.assertIn(row["source_document_id"], source_docs)
            for parent_id in row["parent_legal_unit_ids"]:
                self.assertIn(parent_id, legal_units)

        chunks = read_jsonl(final / "chunks.jsonl")
        exception_chunk_ids = {
            row["chunk_id"] for row in read_jsonl(final / "validation_exceptions.jsonl") if row.get("type") == "pdf_text_layer_noise_review"
        }
        chunk_by_legal_unit: dict[str, set[str]] = {}
        for row in chunks:
            self.assertIn(row["legal_unit_id"], legal_units)
            unit_text = self._compact_text(legal_units[row["legal_unit_id"]]["text"])
            chunk_text = self._compact_text(row["text"])
            self.assertTrue(chunk_text in unit_text or row["status"] == "parent_context_only")
            chunk_by_legal_unit.setdefault(row["legal_unit_id"], set()).add(row["chunk_id"])

        bboxes = {row["bbox_id"]: row for row in read_jsonl(final / "bbox_registry.jsonl")}
        evidence = {row["evidence_id"]: row for row in read_jsonl(final / "evidence_registry.jsonl")}
        for row in bboxes.values():
            self.assertEqual(row["status"], "accepted")
            self.assertIn(row["evidence_id"], evidence)
            self.assertIn(row["source_document_id"], source_docs)
            self.assertIn((row["source_document_id"], row["page_number"]), page_keys)
            rect = pdfs[row["source_document_id"]][row["page_number"] - 1].rect
            self.assertLessEqual(0, row["x0"])
            self.assertLessEqual(0, row["y0"])
            self.assertLessEqual(row["x1"], rect.width)
            self.assertLessEqual(row["y1"], rect.height)

        for row in evidence.values():
            self.assertEqual(row["status"], "final")
            self.assertIn(row["source_document_id"], source_docs)
            self.assertIn(row["legal_unit_id"], legal_units)
            for page_number in row["page_numbers"]:
                self.assertIn((row["source_document_id"], page_number), page_keys)
            self.assertTrue(set(row["bbox_refs"]) <= bboxes.keys())
            quote = self._compact_text(row["quoted_text"])
            pages_text = "".join(page_text[(row["source_document_id"], page_number)] for page_number in row["page_numbers"])
            exception_ids = chunk_by_legal_unit.get(row["legal_unit_id"], set()) & exception_chunk_ids
            self.assertTrue(quote in pages_text or exception_ids, row["evidence_id"])

        metadata_bbox_ids = {row["bbox_id"] for row in read_jsonl(final / "metadata_grounding_registry.jsonl")}
        for row in read_jsonl(final / "metadata_grounding.jsonl"):
            self.assertIn(row["source_document_id"], source_docs)
            for page_number in row["page_numbers"]:
                self.assertIn((row["source_document_id"], page_number), page_keys)
            self.assertTrue(set(row["bbox_refs"]) <= metadata_bbox_ids)
            quote = self._compact_text(row["quoted_text"])
            pages_text = "".join(page_text[(row["source_document_id"], page_number)] for page_number in row["page_numbers"])
            self.assertIn(quote, pages_text)

        graph_nodes = read_jsonl(final / "graph_nodes.jsonl")
        graph_node_ids = {row["node_id"] for row in graph_nodes}
        for row in graph_nodes:
            if row.get("source_pdf_path"):
                self.assertTrue((ROOT / row["source_pdf_path"]).exists(), row["node_id"])
                self.assertEqual(file_sha256(ROOT / row["source_pdf_path"]), row.get("source_sha256"), row["node_id"])
        for edge in read_jsonl(final / "graph_edges.jsonl"):
            self.assertIn(edge["source_id"], graph_node_ids, edge["edge_id"])
            self.assertIn(edge["target_id"], graph_node_ids, edge["edge_id"])

    def test_uud_ingestion_reproducibility_runner_passes(self) -> None:
        result = validate_corpus_ingestion_artifacts("uud", ROOT)
        self.assertEqual(result["status"], "valid", result["errors"][:5])
        self.assertEqual(result["counts"]["evidence_records"], 464)
        self.assertEqual(result["counts"]["bbox_records"], 1584)
        self.assertEqual(validate_uud_ingestion_artifacts(ROOT), result)

    def _compact_text(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text or "").replace("\u00ad", "")
        return "".join(re.findall(r"\w+", text.casefold()))

    def _assert_contains(self, final: Path, source_role: str, parent_label: str, child_label: str) -> None:
        units = {row["legal_unit_id"]: row for row in read_jsonl(final / "legal_units.jsonl")}
        nodes = {
            row["legal_unit_id"]: row["node_id"] for row in read_jsonl(final / "graph_nodes.jsonl") if row.get("node_type") == "legal_unit"
        }
        parent = next(row for row in units.values() if row.get("source_role") == source_role and row.get("unit_label") == parent_label)
        child = next(row for row in units.values() if row.get("source_role") == source_role and row.get("unit_label") == child_label)
        edge_keys = {(row["edge_type"], row["source_id"], row["target_id"]) for row in read_jsonl(final / "graph_edges.jsonl")}
        self.assertIn(("CONTAINS", nodes[parent["legal_unit_id"]], nodes[child["legal_unit_id"]]), edge_keys)
        self.assertIn(("PART_OF", nodes[child["legal_unit_id"]], nodes[parent["legal_unit_id"]]), edge_keys)


if __name__ == "__main__":
    unittest.main()
