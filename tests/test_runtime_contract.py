from __future__ import annotations

import json
from pathlib import Path
import os
import tempfile
import unittest

from tjipto.corpora.registry import CorpusRegistry
from tjipto.corpora.provenance import validate_corpus_provenance
from tjipto.evidence.store import EvidenceStore
from tjipto.graph.store import GraphStore
from tjipto.retrieval.candidates import merge_ranked
from tjipto.retrieval.dense import dense_search
from tjipto.retrieval.metadata import filter_evidence, normalize_filters
from tjipto.retrieval.query import classify_intent, normalize_query
from tjipto.retrieval.router import route_retrieval
from tjipto.retrieval.structured import structured_lookup
from tjipto.retrieval.answer import assemble_context_pack, validate_answer_candidate
from tjipto.runtime.api import handle_request
from tjipto.runtime.service import LegalRuntimeService


ROOT = Path(__file__).resolve().parents[1]


class RuntimeContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = LegalRuntimeService(ROOT)

    def test_search_citation_and_viewer_work(self) -> None:
        search = self.service.search("uud", "negara hukum", limit=3)
        self.assertEqual(search["status"], "found")
        self.assertEqual(search["route"], "bm25")
        self.assertTrue(search["matches"])

        citation = self.service.citation("uud", "Pasal 1 ayat (3)")
        self.assertEqual(citation["status"], "found")
        self.assertEqual(citation["route"], "exact")
        evidence = citation["matches"][0]
        self.assertEqual(evidence["source_role"], "current_consolidated")

        non_citation = self.service.citation("uud", "negara hukum")
        self.assertEqual(non_citation["status"], "citation_not_found")
        self.assertEqual(non_citation["route"], "citation_not_found")
        self.assertFalse(non_citation["matches"])
        self.assertEqual(non_citation["citation_payloads"], ())
        self.assertEqual(non_citation["viewer_refs"], ())
        self.assertEqual(non_citation["validation_reasons"], {})

        missing_citation = self.service.citation("uud", "Pasal 999")
        self.assertEqual(missing_citation["status"], "citation_not_found")
        self.assertEqual(missing_citation["citation_payloads"], ())
        self.assertEqual(missing_citation["viewer_refs"], ())
        self.assertEqual(missing_citation["validation_reasons"], {})

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
        self.assertEqual(answer["route"], "exact")
        self.assertEqual(answer["intent"], "exact_citation")
        self.assertEqual(answer["normalized_query"], "Pasal 1 ayat (3)")
        self.assertTrue(answer["evidence"])
        first = answer["evidence"][0]
        self.assertTrue(first["evidence_id"])
        self.assertGreater(first["bbox_count"], 0)
        self.assertTrue(first["viewer_ref"])

        limited = self.service.ask("uud", "negara hukum")
        self.assertEqual(limited["status"], "limited_answer")
        self.assertEqual(limited["route"], "bm25")
        self.assertEqual(limited["intent"], "natural_language")
        self.assertTrue(limited["evidence"])

        for query in ("Pasal 999", "Pasal 1 ayat 999", "Pasal 28E ayat (999)"):
            result = self.service.ask("uud", query)
            self.assertEqual(result["status"], "citation_not_found")
            self.assertEqual(result["route"], "citation_not_found")
            self.assertEqual(result["reason"], "citation_not_found")
            self.assertFalse(result["evidence"])

        domain_query = self.service.ask("uud", "aturan KUHP tentang pencurian")
        self.assertEqual(domain_query["status"], "insufficient_evidence")
        self.assertEqual(domain_query["route"], "bm25")
        self.assertIsNone(domain_query["required_corpus"])
        self.assertFalse(domain_query["evidence"])
        self.assertFalse(domain_query["citations"])
        self.assertFalse(domain_query["viewer_refs"])

        unsupported = self.service.ask("unknown", "Pasal 1")
        self.assertEqual(unsupported["status"], "unsupported_corpus")
        self.assertEqual(unsupported["intent"], "unsupported_corpus")

    def test_query_normalization_and_intent_classification(self) -> None:
        self.assertEqual(normalize_query("pasal 28 e")["normalized_query"], "Pasal 28E")
        self.assertEqual(normalize_query("pasal 1 ayat 3")["normalized_query"], "Pasal 1 ayat (3)")
        self.assertEqual(normalize_query("uud 45")["normalized_query"], "UUD 1945")

        self.assertEqual(classify_intent("uud", "Pasal 1 ayat (3)")["intent"], "exact_citation")
        self.assertEqual(classify_intent("uud", "negara hukum")["intent"], "natural_language")
        self.assertEqual(classify_intent("uud", "aturan KUHP tentang pencurian")["intent"], "natural_language")
        self.assertEqual(
            classify_intent("unknown", "Pasal 1", corpus_supported=False)["intent"],
            "unsupported_corpus",
        )
        self.assertNotIn("required_corpus", classify_intent("uud", "aturan KUHP tentang pencurian"))

    def test_bm25_relevance_gate_keeps_core_uud_queries_answerable(self) -> None:
        for query in (
            "hukum adat",
            "hak pendidikan",
            "lingkungan hidup",
            "persatuan Indonesia",
            "hak asasi manusia",
            "makna negara hukum",
            "Majelis Permusyawaratan Rakyat",
            "hak untuk bekerja",
            "pekerjaan dan penghidupan yang layak",
        ):
            result = self.service.ask("uud", query)
            self.assertEqual(result["status"], "limited_answer", query)
            self.assertEqual(result["route"], "bm25", query)
            self.assertIsNone(result["required_corpus"], query)
            self.assertTrue(result["evidence"], query)

    def test_weak_bm25_matches_do_not_become_final_payloads(self) -> None:
        for query in (
            "aturan KUHP tentang pencurian",
            "UU Pers tentang wartawan",
            "berapa lama polisi boleh menahan tersangka",
            "hukuman pembunuhan berapa",
            "Permen teknis izin usaha",
            "sanksi adat menggantikan pidana",
        ):
            result = self.service.ask("uud", query)
            self.assertIn(result["status"], {"insufficient_evidence", "no_results"}, query)
            self.assertIsNone(result["required_corpus"], query)
            self.assertFalse(result["evidence"], query)
            self.assertFalse(result["citations"], query)
            self.assertFalse(result["viewer_refs"], query)
            self.assertFalse(result["context_pack"]["answer_evidence"], query)
            self.assertFalse(result["context_pack"]["citation_payloads"], query)
            self.assertFalse(result["context_pack"]["viewer_refs"], query)
            reasons = set(result["context_pack"]["validation_reasons"].values())
            if result["route"] == "bm25":
                self.assertIn("insufficient_query_support", reasons, query)

    def test_retrieval_router_envelope_routes(self) -> None:
        config = CorpusRegistry(ROOT).resolve("uud")
        store = EvidenceStore(config)

        exact = route_retrieval("uud", "pasal 1 ayat 3", store)
        self.assertEqual(exact["status"], "found")
        self.assertEqual(exact["route"], "exact")
        self.assertEqual(exact["normalized_query"], "Pasal 1 ayat (3)")
        self.assertTrue(exact["matches"])

        bm25 = route_retrieval("uud", "negara hukum", store, limit=2)
        self.assertEqual(bm25["status"], "found")
        self.assertEqual(bm25["route"], "bm25")
        self.assertEqual(bm25["reason"], None)
        self.assertLessEqual(len(bm25["matches"]), 2)

        dense = route_retrieval("uud", "negara hukum", store, route="dense")
        self.assertEqual(dense["status"], "dense_unavailable")
        self.assertEqual(dense["route"], "dense_unavailable")
        self.assertEqual(dense["reason"], "not_configured")
        self.assertEqual(dense["matches"], ())

        no_results = route_retrieval("uud", "zyxqv unsupported legal relation", store)
        self.assertEqual(no_results["status"], "no_results")
        self.assertEqual(no_results["intent"], "no_results")

        domain_query = route_retrieval("uud", "aturan KUHP tentang pencurian", store)
        self.assertEqual(domain_query["status"], "found")
        self.assertEqual(domain_query["route"], "bm25")
        self.assertIsNone(domain_query["required_corpus"])
        self.assertTrue(domain_query["matches"])
        self.assertTrue(all(row["lexical_relevance_ok"] is False for row in domain_query["matches"]))

        unsupported = route_retrieval("unknown", "Pasal 1", None)
        self.assertEqual(unsupported["status"], "unsupported_corpus")
        self.assertEqual(unsupported["route"], "unsupported_corpus")

    def test_metadata_filtering_limits_retrieval_safely(self) -> None:
        config = CorpusRegistry(ROOT).resolve("uud")
        store = EvidenceStore(config)

        filters = normalize_filters({"source_role": "current_consolidated"})
        self.assertEqual(filters, {"source_role": "current_consolidated", "status": "final"})
        rows = filter_evidence(tuple(store.evidence), filters)
        self.assertTrue(rows)
        self.assertTrue(all(row["status"] == "final" for row in rows))
        self.assertTrue(all(row["source_role"] == "current_consolidated" for row in rows))

        current = self.service.ask(
            "uud",
            "Pasal 1 ayat (3)",
            filters={"source_role": "current_consolidated"},
        )
        self.assertEqual(current["status"], "answer_ready")
        self.assertEqual(current["applied_filters"]["source_role"], "current_consolidated")
        self.assertTrue(all(row["bbox_count"] > 0 for row in current["evidence"]))

        removed = self.service.ask(
            "uud",
            "Pasal 1 ayat (3)",
            filters={"source_role": "amendment_1_historical"},
        )
        self.assertEqual(removed["status"], "no_results")
        self.assertEqual(removed["reason"], "filters_removed_all")
        self.assertFalse(removed["matches"])
        self.assertFalse(removed["evidence"])

        historical = self.service.citation(
            "uud",
            "Pasal 5 ayat (1)",
            source_role="amendment_1_historical",
        )
        self.assertEqual(historical["status"], "found")
        self.assertEqual(historical["route"], "exact")
        self.assertTrue(historical["matches"])
        self.assertTrue(
            all(row["source_role"] == "amendment_1_historical" for row in historical["matches"])
        )

        historical_search = self.service.search(
            "uud",
            "negara hukum",
            filters={"source_role": "amendment_1_historical"},
        )
        self.assertEqual(historical_search["status"], "found")
        self.assertTrue(
            all(row["source_role"] == "amendment_1_historical" for row in historical_search["matches"])
        )

        temporal = self.service.search(
            "uud",
            "presiden",
            limit=1,
            filters={"temporal_context": "amendment_1_historical"},
        )
        self.assertEqual(temporal["status"], "found")
        self.assertEqual(temporal["applied_filters"]["temporal_context"], "amendment_1_historical")
        self.assertEqual(len(temporal["matches"]), 1)
        self.assertEqual(temporal["matches"][0]["temporal_context"], "amendment_1_historical")

        conflicting = self.service.search(
            "uud",
            "presiden",
            filters={
                "source_role": "current_consolidated",
                "temporal_context": "amendment_1_historical",
            },
        )
        self.assertEqual(conflicting["status"], "invalid_filter")
        self.assertEqual(conflicting["reason"], "conflicting_filters")
        self.assertFalse(conflicting["matches"])

        invalid = self.service.search(
            "uud",
            "presiden",
            filters={"source_role": "not_a_source_role"},
        )
        self.assertEqual(invalid["status"], "invalid_filter")
        self.assertEqual(invalid["reason"], "invalid_filter")
        self.assertEqual(invalid["invalid_filters"], ("source_role",))

        api_result = handle_request(
            "uud",
            "ask",
            {
                "query": "Pasal 5 ayat (1)",
                "filters": {
                    "source_role": "amendment_1_historical",
                    "temporal_context": "amendment_1_historical",
                },
            },
            ROOT,
        )
        self.assertEqual(api_result["status"], "answer_ready")
        self.assertEqual(api_result["applied_filters"]["temporal_context"], "amendment_1_historical")

    def test_dense_readiness_does_not_fake_matches(self) -> None:
        config = CorpusRegistry(ROOT).resolve("uud")
        store = EvidenceStore(config)
        result = dense_search(store, "negara hukum")
        self.assertEqual(result["status"], "dense_unavailable")
        self.assertEqual(result["route"], "dense_unavailable")
        self.assertEqual(result["reason"], "not_configured")
        self.assertEqual(result["matches"], ())

    def test_provenance_validation_reports_header_stripped_uud_matches(self) -> None:
        config = CorpusRegistry(ROOT).resolve("uud")
        report = validate_corpus_provenance(config)
        self.assertEqual(report["status"], "pass")
        for key in ("legal_units", "chunks"):
            self.assertEqual(report[key]["total"], 609)
            self.assertEqual(report[key]["raw_pdf_match"], 584)
            self.assertEqual(report[key]["normalized_pdf_match"], 584)
            self.assertEqual(report[key]["header_stripped_pdf_match"], 609)
            self.assertEqual(report[key]["needs_review"], 0)

    def test_structured_lookup_is_evidence_and_bbox_backed(self) -> None:
        config = CorpusRegistry(ROOT).resolve("uud")
        store = EvidenceStore(config)
        cases = (
            ("Pembukaan", "PEMBUKAAN"),
            ("BAB I", "BAB I"),
            ("Pasal 1 ayat (3)", "Pasal 1"),
            ("Aturan Peralihan", "ATURAN PERALIHAN"),
            ("Aturan Tambahan", "ATURAN TAMBAHAN"),
        )
        for query, expected in cases:
            rows = structured_lookup(store, query, limit=3)
            self.assertTrue(rows, query)
            self.assertTrue(all(row["status"] == "final" for row in rows))
            self.assertTrue(all(store.bboxes_for(row["evidence_id"]) for row in rows))
            self.assertTrue(any(expected in " ".join(row.get("hierarchy") or row.get("citation", "")) for row in rows))

        filtered = self.service.search(
            "uud",
            "Pembukaan",
            filters={"source_role": "current_consolidated", "temporal_context": "current_consolidated"},
        )
        self.assertEqual(filtered["status"], "found")
        self.assertEqual(filtered["route"], "structured")
        self.assertTrue(all(row["source_role"] == "current_consolidated" for row in filtered["matches"]))

        class NoBBoxStore:
            evidence = (
                {
                    "evidence_id": "e1",
                    "status": "final",
                    "citation": "(1)",
                    "hierarchy": ["BAB I", "Pasal 1", "(1)"],
                },
            )

            def bboxes_for(self, evidence_id):
                return []

        self.assertEqual(structured_lookup(NoBBoxStore(), "Pasal 1 ayat (1)"), ())

        class UnitBackedStore:
            evidence = ({"evidence_id": "e2", "legal_unit_id": "lu2", "status": "final", "hierarchy": []},)
            legal_units = ({"legal_unit_id": "lu2", "unit_label": "Pasal 9", "hierarchy": []},)
            chunks = ()

            def bboxes_for(self, evidence_id):
                return [{"bbox_id": "b2"}]

        self.assertEqual(structured_lookup(UnitBackedStore(), "Pasal 9")[0]["evidence_id"], "e2")

    def test_graph_candidate_pipeline_merges_dedups_and_ranks_stably(self) -> None:
        class Store:
            evidence = [
                {"evidence_id": "e1", "legal_unit_id": "lu1", "status": "final", "source_role": "current_consolidated"},
                {"evidence_id": "e2", "legal_unit_id": "lu2", "status": "final", "source_role": "current_consolidated"},
                {"evidence_id": "e3", "legal_unit_id": "lu3", "status": "final", "source_role": "amendment_1_historical"},
                {"evidence_id": "e4", "legal_unit_id": "lu1", "status": "final", "source_role": "current_consolidated"},
                {"evidence_id": "no_box", "legal_unit_id": "lu4", "status": "final", "source_role": "current_consolidated"},
            ]
            graph_edges = [
                {"edge_type": "HAS_FINAL_EVIDENCE", "source_id": "legal_unit::lu1", "target_id": "final_evidence::e1"},
                {"edge_type": "HAS_FINAL_EVIDENCE", "source_id": "legal_unit::lu1", "target_id": "final_evidence::e2"},
                {"edge_type": "HAS_FINAL_EVIDENCE", "source_id": "legal_unit::lu1", "target_id": "final_evidence::e4"},
                {"edge_type": "HAS_FINAL_EVIDENCE", "source_id": "legal_unit::lu1", "target_id": "final_evidence::no_box"},
                {"edge_type": "HAS_FINAL_EVIDENCE", "source_id": "legal_unit::lu3", "target_id": "final_evidence::e3"},
            ]

            def get(self, evidence_id):
                return next((row for row in self.evidence if row["evidence_id"] == evidence_id), None)

            def bboxes_for(self, evidence_id):
                return [] if evidence_id == "no_box" else [{"bbox_id": evidence_id}]

        store = Store()
        ranked, trace = merge_ranked(
            store,
            {
                "bm25": (store.evidence[1], store.evidence[0]),
                "structured": (store.evidence[0],),
            },
            {"status": "final", "source_role": "current_consolidated"},
        )
        self.assertEqual([row["evidence_id"] for row in ranked], ["e1", "e2", "e4"])
        self.assertEqual(ranked[0]["route_sources"], ("bm25", "structured"))
        self.assertIn("pass", ranked[0]["rank_reasons"])
        self.assertTrue(trace)
        self.assertTrue(all(item["evidence_id"] != "no_box" for item in trace))

        empty, empty_trace = merge_ranked(store, {}, {"status": "final"})
        self.assertEqual(empty, ())
        self.assertEqual(empty_trace, ())

    def test_runtime_bm25_exposes_ranked_route_signals(self) -> None:
        result = self.service.search("uud", "negara hukum", limit=3)
        self.assertEqual(result["status"], "found")
        self.assertEqual(result["route"], "bm25")
        self.assertTrue(result["expansion_trace"])
        for row in result["matches"]:
            self.assertTrue(row["bbox_count"])
            self.assertIn("route_sources", row)
            self.assertIn("rank_reasons", row)
            self.assertIn("route_score", row)

    def test_ask_excludes_graph_only_answer_evidence(self) -> None:
        direct_routes = {"exact", "structured", "bm25"}

        exact = self.service.ask("uud", "Pasal 1 ayat (3)", limit=5)
        self.assertEqual(exact["status"], "answer_ready")
        self.assertEqual(exact["answer_type"], "quoted_evidence")
        self.assertIn("context_pack", exact)
        self.assertEqual(exact["context_pack"]["answer_evidence"], exact["evidence"])
        self.assertTrue(exact["citations"])
        self.assertTrue(exact["viewer_refs"])
        citation_payload = exact["citations"][0]
        for field in (
            "evidence_id",
            "citation",
            "quoted_text",
            "source_role",
            "temporal_context",
            "source_pdf_path",
            "source_sha256",
            "page_numbers",
            "bbox_count",
            "viewer_ref",
            "evidence_status",
        ):
            self.assertIn(field, citation_payload)
        viewer_ref = exact["viewer_refs"][0]
        self.assertTrue(viewer_ref["can_resolve"])
        self.assertGreater(viewer_ref["bbox_count"], 0)
        self.assertTrue(viewer_ref["page_numbers"])
        self.assertTrue(self.service.viewer("uud", viewer_ref["evidence_id"])["bbox_rectangles"])
        self.assertTrue(any(row["route_sources"] == ("graph",) for row in exact["matches"]))
        self.assertTrue(any(row["route_sources"] == ("graph",) for row in exact["context_pack"]["supporting_context"]))
        self.assertTrue(any(row["reason"] == "graph_only" for row in exact["context_pack"]["excluded_results"]))
        self.assertTrue(
            all(direct_routes & set(row["route_sources"]) for row in exact["matches"] if row["evidence_id"] in {
                evidence["evidence_id"] for evidence in exact["evidence"]
            })
        )
        self.assertFalse(any(evidence["citation"] == "PEMBUKAAN/Preambule" for evidence in exact["evidence"]))

        structured = self.service.ask("uud", "Pembukaan", limit=5)
        self.assertEqual(structured["status"], "limited_answer")
        self.assertEqual(structured["answer_type"], "limited_evidence_summary")
        self.assertTrue(any(row["route_sources"] == ("graph",) for row in structured["matches"]))
        evidence_ids = {evidence["evidence_id"] for evidence in structured["evidence"]}
        self.assertTrue(all(direct_routes & set(row["route_sources"]) for row in structured["matches"] if row["evidence_id"] in evidence_ids))

        search = self.service.search("uud", "Pasal 1 ayat (3)", limit=5)
        self.assertTrue(any(row["route_sources"] == ("graph",) for row in search["matches"]))

    def test_citation_response_exposes_public_payloads(self) -> None:
        result = self.service.citation("uud", "Pasal 1 ayat (3)")
        self.assertEqual(result["status"], "found")
        self.assertTrue(result["matches"])
        self.assertTrue(result["citation_payloads"])
        self.assertTrue(result["viewer_refs"])
        self.assertTrue(result["validation_reasons"])
        self.assertEqual(result["citation_payloads"][0]["evidence_status"], "final")
        self.assertTrue(result["viewer_refs"][0]["can_resolve"])
        self.assertEqual(result["matches"][0]["evidence_id"], result["citation_payloads"][0]["evidence_id"])

    def test_policy_docs_match_runtime_truth(self) -> None:
        policy_paths = (
            ROOT / "docs/policies/corpus_coverage_policy.json",
            ROOT / "docs/policies/legal_research_orchestrator_policy.json",
        )
        for path in policy_paths:
            data = json.loads(path.read_text(encoding="utf-8"))
            text = json.dumps(data).casefold()
            self.assertNotIn("candidate only", text)
            self.assertNotIn("policy_only", text)
            self.assertEqual(data["runtime_policy_status"], "implemented_for_uud_runtime_baseline")
            self.assertFalse(data["non_uud_corpora_available"])
            self.assertIn("not claim full legal-grade production", data["not_legal_grade_note"])

    def test_no_coverage_classifier_slop_remains(self) -> None:
        self.assertFalse((ROOT / "src/tjipto/corpora/coverage.py").exists())
        source = "\n".join(
            path.read_text(encoding="utf-8")
            for base in ("src/tjipto/corpora", "src/tjipto/retrieval", "src/tjipto/runtime")
            for path in (ROOT / base).rglob("*.py")
        )
        self.assertNotIn("classify_coverage", source)
        self.assertNotIn("required_missing_corpus", source)
        self.assertNotIn("out_of_corpus", source)

    def test_answer_context_validator_explains_inclusion_and_exclusion(self) -> None:
        class Store:
            def bboxes_for(self, evidence_id):
                return [] if evidence_id == "missing_bbox" else [{"bbox_id": evidence_id}]

        store = Store()
        base = {
            "evidence_id": "direct",
            "status": "final",
            "citation": "Pasal 1",
            "quoted_text": "quoted",
            "source_pdf_path": "source.pdf",
            "source_sha256": "sha",
            "source_role": "current_consolidated",
            "temporal_context": "current_consolidated",
            "page_numbers": (1,),
            "route_sources": ("exact",),
        }
        graph = base | {"evidence_id": "graph", "route_sources": ("graph",)}
        missing_bbox = base | {"evidence_id": "missing_bbox"}
        not_loadable = base | {"evidence_id": "not_loadable", "runtime_loadable": False}

        self.assertEqual(validate_answer_candidate(store, base), (True, "answer_evidence"))
        self.assertEqual(validate_answer_candidate(store, graph), (False, "graph_only"))
        self.assertEqual(validate_answer_candidate(store, missing_bbox), (False, "missing_bbox"))
        self.assertEqual(validate_answer_candidate(store, not_loadable), (False, "runtime_not_loadable"))

        pack = assemble_context_pack(store, (base, graph, missing_bbox, not_loadable))
        self.assertEqual([row["evidence_id"] for row in pack["answer_evidence"]], ["direct"])
        self.assertEqual([row["evidence_id"] for row in pack["supporting_context"]], ["graph"])
        self.assertEqual(pack["validation_reasons"]["graph"], "graph_only")
        self.assertEqual(pack["validation_reasons"]["missing_bbox"], "missing_bbox")
        self.assertEqual(pack["validation_reasons"]["not_loadable"], "runtime_not_loadable")
        self.assertFalse(any(row["evidence_id"] == "not_loadable" for row in pack["citation_payloads"]))

    def test_failure_ask_context_pack_has_no_final_payloads(self) -> None:
        for result in (
            self.service.ask("uud", "Pasal 999"),
            self.service.ask("uud", "wartawan meliput demonstrasi"),
            self.service.ask("unknown", "Pasal 1"),
        ):
            self.assertEqual(result["answer_type"], "none")
            self.assertEqual(result["evidence"], ())
            self.assertEqual(result["citations"], ())
            self.assertEqual(result["viewer_refs"], ())
            self.assertEqual(result["context_pack"]["answer_evidence"], ())
            self.assertEqual(result["context_pack"]["citation_payloads"], ())
            self.assertEqual(result["context_pack"]["viewer_refs"], ())
            self.assertIn("request", result["context_pack"]["validation_reasons"])

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
                    "source_pdf_path": "source.pdf",
                    "source_sha256": "sha",
                    "source_role": "current_consolidated",
                    "temporal_context": "current_consolidated",
                    "page_numbers": [1],
                    "status": "final",
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
