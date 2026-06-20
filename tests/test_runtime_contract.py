from __future__ import annotations

import json
from pathlib import Path
import os
import tempfile
import unittest

from tjipto.corpora.registry import CorpusRegistry
from tjipto.evidence.store import EvidenceStore
from tjipto.graph.store import GraphStore
from tjipto.runtime.service import LegalRuntimeService


ROOT = Path(__file__).resolve().parents[1]


class RuntimeContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = LegalRuntimeService(ROOT)

    def test_search_citation_and_viewer_work(self) -> None:
        search = self.service.search("uud", "negara hukum", limit=3)
        self.assertEqual(search["status"], "found")
        self.assertTrue(search["matches"])

        citation = self.service.citation("uud", "Pasal 1 ayat (3)")
        self.assertEqual(citation["status"], "found")
        evidence = citation["matches"][0]
        self.assertEqual(evidence["source_role"], "current_consolidated")

        viewer = self.service.viewer("uud", evidence["evidence_id"])
        self.assertEqual(viewer["status"], "viewer_payload_ready")
        self.assertTrue(viewer["page_numbers"])
        self.assertGreater(viewer["bbox_count"], 0)
        self.assertTrue(viewer["bbox_rectangles"])

    def test_retrieval_units_reference_final_evidence(self) -> None:
        from tjipto.core.manifest import read_jsonl

        evidence_ids = {
            row["evidence_id"]
            for row in read_jsonl(ROOT / "data/final/uud/evidence_registry.jsonl")
        }
        bbox_ids = {
            row["bbox_id"]
            for row in read_jsonl(ROOT / "data/final/uud/bbox_registry.jsonl")
        }
        rows = read_jsonl(ROOT / "data/final/uud/retrieval_units.jsonl")
        self.assertEqual(len(rows), 438)
        for row in rows:
            self.assertIn(row["evidence_id"], evidence_ids)
            self.assertGreaterEqual(row["bbox_total_count"], len(row["bbox_sample_refs"]))
            self.assertTrue(set(row["bbox_sample_refs"]) <= bbox_ids)
            self.assertTrue((ROOT / row["source_pdf_path"]).exists())

    def test_graph_retrieval_eval_fixtures_resolve_refs(self) -> None:
        from tjipto.core.manifest import read_jsonl

        evidence_ids = {
            row["evidence_id"]
            for row in read_jsonl(ROOT / "data/final/uud/evidence_registry.jsonl")
        }
        chunk_ids = {
            row["chunk_id"]
            for row in read_jsonl(ROOT / "data/final/uud/chunks.jsonl")
        }
        cases = read_jsonl(ROOT / "tests/fixtures/uud/graph_retrieval_eval_cases.jsonl")
        self.assertEqual(len(cases), 76)
        for row in cases:
            self.assertTrue(set(row.get("expected_final_evidence_ids") or []) <= evidence_ids)
        orchestrator = read_jsonl(ROOT / "tests/fixtures/uud/orchestrator_eval_results.jsonl")
        self.assertEqual(len(orchestrator), 175)
        for row in orchestrator:
            for chunk_id in row.get("observed_chunk_ids") or ():
                self.assertIn(chunk_id, chunk_ids)
        traces = read_jsonl(ROOT / "tests/fixtures/uud/graph_retrieval_traces.jsonl")
        self.assertEqual(len(traces), 76)
        for row in traces:
            for evidence_id in row.get("outputs", {}).get("ranked_final_evidence_ids") or ():
                self.assertIn(evidence_id, evidence_ids)

    def test_bm25_prioritizes_term_frequency_without_breaking_exact_citation(self) -> None:
        citation = self.service.citation("uud", "Pasal 1 ayat (3)")
        search = self.service.search("uud", "Pasal 1 ayat (3)", limit=1)
        self.assertEqual(search["matches"][0]["evidence_id"], citation["matches"][0]["evidence_id"])

        results = self.service.search("uud", "negara negara negara hukum", limit=3)
        self.assertEqual(results["status"], "found")
        self.assertTrue(any("negara" in row["quoted_text"].casefold() for row in results["matches"]))

    def test_ask_contract_is_evidence_bounded(self) -> None:
        answer = self.service.ask("uud", "Pasal 1 ayat (3)")
        self.assertEqual(answer["status"], "answer_ready")
        self.assertTrue(answer["evidence"])
        first = answer["evidence"][0]
        self.assertTrue(first["evidence_id"])
        self.assertGreater(first["bbox_count"], 0)
        self.assertTrue(first["viewer_ref"])

        limited = self.service.ask("uud", "negara hukum")
        self.assertIn(limited["status"], {"answer_ready", "limited_answer"})
        self.assertTrue(limited["evidence"])

        for query in ("Pasal 999", "Pasal 1 ayat 999", "Pasal 28E ayat (999)"):
            result = self.service.ask("uud", query)
            self.assertEqual(result["status"], "citation_not_found")
            self.assertEqual(result["reason"], "citation_not_found")
            self.assertFalse(result["evidence"])
        self.assertEqual(self.service.ask("uud", "aturan KUHP tentang pencurian")["status"], "insufficient_corpus")
        self.assertEqual(self.service.ask("unknown", "Pasal 1")["status"], "unsupported_corpus")

    def test_unsupported_corpus_fails_safely(self) -> None:
        self.assertEqual(
            self.service.search("unknown", "Pasal 1")["status"],
            "unsupported_corpus",
        )

    def test_corpus_id_resolves_through_registry(self) -> None:
        config = CorpusRegistry(ROOT).resolve("uud")
        self.assertIsNotNone(config)
        self.assertEqual(config.corpus_id, "uud")

    def test_artifact_paths_resolve_through_manifest_keys(self) -> None:
        config = CorpusRegistry(ROOT).resolve("uud")
        self.assertEqual(len(config.jsonl("evidence")), 438)
        self.assertEqual(len(config.jsonl("bbox")), 1388)
        self.assertEqual(len(config.jsonl("graph_nodes")), 2339)

    def test_runtime_services_and_stores_do_not_hardcode_artifacts(self) -> None:
        checked_roots = (
            ROOT / "src/tjipto/runtime",
            ROOT / "src/tjipto/retrieval",
            ROOT / "src/tjipto/evidence",
            ROOT / "src/tjipto/graph",
        )
        text = "\n".join(
            path.read_text(encoding="utf-8").casefold()
            for root in checked_roots
            for path in root.rglob("*.py")
        )
        for forbidden in (
            "data/final/uud",
            "data\\final\\uud",
            "data/final/<corpus_id>",
            "evidence_registry.jsonl",
            "bbox_registry.jsonl",
            "graph_nodes.jsonl",
            "graph_edges.jsonl",
            '"uud"',
        ):
            self.assertNotIn(forbidden, text)

    def test_non_uud_corpus_uses_registry_and_renamed_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            corpus = root / "corpus"
            data.mkdir()
            corpus.mkdir()
            (data / "corpus_registry.json").write_text(
                json.dumps({"demo": "corpus/manifest.json"}),
                encoding="utf-8",
            )
            (corpus / "proof.rows").write_text(
                json.dumps({
                    "evidence_id": "demo_evidence_1",
                    "citation": "Rule 1",
                    "quoted_text": "generic corpus resolution",
                    "hierarchy": [],
                })
                + "\n",
                encoding="utf-8",
            )
            (corpus / "boxes.rows").write_text(
                json.dumps({"evidence_id": "demo_evidence_1", "bbox_id": "box_1"}) + "\n",
                encoding="utf-8",
            )
            (corpus / "nodes.rows").write_text("{}\n{}\n", encoding="utf-8")
            (corpus / "edges.rows").write_text("{}\n", encoding="utf-8")
            (corpus / "manifest.json").write_text(
                json.dumps({
                    "corpus_id": "demo",
                    "evidence_registry": "proof.rows",
                    "bbox_registry": "boxes.rows",
                    "graph_nodes": "nodes.rows",
                    "graph_edges": "edges.rows",
                }),
                encoding="utf-8",
            )

            config = CorpusRegistry(root).resolve("demo")
            self.assertIsNotNone(config)
            store = EvidenceStore(config)
            self.assertEqual(store.evidence[0]["evidence_id"], "demo_evidence_1")
            self.assertEqual(store.bboxes_for("demo_evidence_1")[0]["bbox_id"], "box_1")
            self.assertEqual(GraphStore(config).counts(), {"nodes": 2, "edges": 1})
            self.assertEqual(LegalRuntimeService(root).ask("demo", "generic corpus resolution")["status"], "limited_answer")

    def test_missing_or_invalid_registry_fails_safely(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(
                LegalRuntimeService(root).search("demo", "x")["status"],
                "unsupported_corpus",
            )
            (root / "data").mkdir()
            (root / "data/corpus_registry.json").write_text("{", encoding="utf-8")
            self.assertIsNone(CorpusRegistry(root).resolve("demo"))

    def test_registry_uses_env_repo_root_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data/final/demo").mkdir(parents=True)
            (root / "data/corpus_registry.json").write_text(json.dumps({"demo": "data/final/demo/manifest.json"}), encoding="utf-8")
            (root / "data/final/demo/manifest.json").write_text(json.dumps({"corpus_id": "demo"}), encoding="utf-8")
            old = os.environ.get("TJIPTO_REPO_ROOT")
            os.environ["TJIPTO_REPO_ROOT"] = str(root)
            try:
                self.assertEqual(CorpusRegistry().resolve("demo").corpus_id, "demo")
            finally:
                if old is None:
                    os.environ.pop("TJIPTO_REPO_ROOT", None)
                else:
                    os.environ["TJIPTO_REPO_ROOT"] = old

    def test_registry_rejects_absolute_and_parent_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data").mkdir()
            registry = root / "data/corpus_registry.json"

            registry.write_text(
                json.dumps({"demo": str((root / "manifest.json").resolve())}),
                encoding="utf-8",
            )
            self.assertIsNone(CorpusRegistry(root).resolve("demo"))

            registry.write_text(
                json.dumps({"demo": "../manifest.json"}),
                encoding="utf-8",
            )
            self.assertIsNone(CorpusRegistry(root).resolve("demo"))

    def test_manifest_rejects_absolute_and_parent_artifact_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            corpus = root / "corpus"
            data.mkdir()
            corpus.mkdir()
            (data / "corpus_registry.json").write_text(
                json.dumps({"demo": "corpus/manifest.json"}),
                encoding="utf-8",
            )

            for artifact_path in (str((root / "outside.rows").resolve()), "../outside.rows"):
                (corpus / "manifest.json").write_text(
                    json.dumps({"corpus_id": "demo", "evidence_registry": artifact_path}),
                    encoding="utf-8",
                )
                config = CorpusRegistry(root).resolve("demo")
                self.assertIsNotNone(config)
                with self.assertRaises(ValueError):
                    config.artifact_path("evidence")


if __name__ == "__main__":
    unittest.main()
