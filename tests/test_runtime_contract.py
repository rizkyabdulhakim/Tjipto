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
from tjipto.runtime.api import _public_bbox, handle_request
from tjipto.runtime.service import LegalRuntimeService
from tests.test_source_conflict_runtime_contract import _source_conflict_cases


ROOT = Path(__file__).resolve().parents[1]


def _exact_highlightable_evidence_id() -> str:
    for row in CorpusRegistry(ROOT).resolve("uud").jsonl("evidence"):
        if row.get("viewer_highlightable") is True:
            return row["evidence_id"]
    raise AssertionError("missing exact highlightable evidence")


def _page_grounded_decision_evidence_id() -> str:
    for row in CorpusRegistry(ROOT).resolve("uud").jsonl("evidence"):
        if row.get("citation") in {
            "Perubahan Pertama Decision",
            "Perubahan Ketiga Decision",
            "Perubahan Keempat Decision",
        } and row.get("viewer_highlightable") is False:
            return row["evidence_id"]
    raise AssertionError("missing page-grounded decision evidence")


UNSAFE_INSTRUMENT_EVIDENCE = {
    "00623": "uud_instrument_final_citation_evidence::amendment_1_historical::00004::perubahan_pertama_decision",
    "00632": "uud_instrument_final_citation_evidence::amendment_3_historical::00012::perubahan_ketiga_scope",
    "00634": "uud_instrument_final_citation_evidence::amendment_3_historical::00014::perubahan_ketiga_decision",
    "00639": "uud_instrument_final_citation_evidence::amendment_4_historical::00018::perubahan_keempat_scope",
    "00648": "uud_instrument_final_citation_evidence::amendment_4_historical::00024::perubahan_keempat_decision",
}

SAFE_INSTRUMENT_EVIDENCE = {
    "00621": "uud_instrument_final_citation_evidence::amendment_1_historical::00002::perubahan_pertama_scope",
    "00628": "uud_instrument_final_citation_evidence::amendment_2_historical::00008::perubahan_kedua_scope",
    "00638": "uud_instrument_final_citation_evidence::amendment_4_historical::00017::perubahan_keempat_recital",
}


def _relation_cases() -> tuple[dict, ...]:
    path = ROOT / "tests/fixtures/uud/relation_cases.jsonl"
    return tuple(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _unsupported_relation_cases() -> tuple[dict, ...]:
    path = ROOT / "tests/fixtures/uud/unsupported_relation_cases.jsonl"
    return tuple(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _metadata_runtime_cases() -> tuple[dict, ...]:
    path = ROOT / "tests/fixtures/uud/metadata_runtime_cases.jsonl"
    return tuple(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _instrument_runtime_cases() -> tuple[dict, ...]:
    path = ROOT / "tests/fixtures/uud/instrument_runtime_cases.jsonl"
    return tuple(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _query_intent_cases() -> tuple[dict, ...]:
    path = ROOT / "tests/fixtures/uud/query_intent_cases.jsonl"
    return tuple(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _bm25_relevance_cases() -> tuple[dict, ...]:
    path = ROOT / "tests/fixtures/uud/bm25_relevance_cases.jsonl"
    return tuple(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _weak_bm25_cases() -> tuple[dict, ...]:
    path = ROOT / "tests/fixtures/uud/weak_bm25_cases.jsonl"
    return tuple(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _retrieval_router_cases() -> tuple[dict, ...]:
    path = ROOT / "tests/fixtures/uud/retrieval_router_cases.jsonl"
    return tuple(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


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
        self.assertEqual(viewer["corpus_id"], "uud")
        self.assertEqual(viewer["legal_unit_id"], evidence["legal_unit_id"])
        self.assertEqual(viewer["source_document_id"], evidence["source_document_id"])
        self.assertTrue(viewer["pdf_access_available"])
        self.assertEqual(viewer["render_status"], "pdf_access_available")
        self.assertEqual(viewer["pdf"]["mime_type"], "application/pdf")
        self.assertTrue(viewer["pdf"]["access_url"].startswith("/legal/uud/pdf?"))
        self.assertNotIn("data_url", viewer["pdf"])
        self.assertEqual(viewer["source_role"], "current_consolidated")
        self.assertEqual(viewer["source_status_label"], "Berlaku (konsolidasi saat ini)")
        self.assertTrue(viewer["page_numbers"])
        self.assertGreater(viewer["bbox_count"], 0)
        self.assertTrue(viewer["bbox_rectangles"])
        for box in viewer["bbox_rectangles"]:
            self.assertGreaterEqual(box["x0"], 0)
            self.assertGreaterEqual(box["y0"], 0)
            self.assertGreater(box["x1"], box["x0"])
            self.assertGreater(box["y1"], box["y0"])

    def test_viewer_rejects_invalid_render_inputs_safely(self) -> None:
        evidence = self.service.citation("uud", "Pasal 1 ayat (3)")["matches"][0]
        evidence_id = evidence["evidence_id"]
        cases = (
            ({"source_document_id": "uud::missing"}, "invalid_source"),
            ({"page_number": 999}, "invalid_page"),
            ({"bbox_id": "missing_bbox"}, "invalid_bbox"),
            ({"source_pdf_path": "../secret.pdf"}, "invalid_source"),
        )
        for kwargs, reason in cases:
            result = self.service.viewer("uud", evidence_id, **kwargs)
            self.assertEqual(result["status"], "viewer_payload_ready", kwargs)
            self.assertFalse(result["rendering_available"], kwargs)
            self.assertIn(result["render_status"], {"render_unavailable", "render_failed_safe"}, kwargs)
            self.assertEqual(result["reason"], reason, kwargs)
            text = json.dumps(result)
            self.assertNotIn(str(ROOT), text)
            self.assertNotIn("Traceback", text)

        self.assertEqual(self.service.viewer("unknown", evidence_id)["status"], "unsupported_corpus")
        self.assertEqual(self.service.viewer("uud", "missing")["reason"], "invalid_evidence")

    def test_search_results_are_public_evidence_payloads(self) -> None:
        result = self.service.search("uud", "negara hukum", limit=2)
        self.assertEqual(result["status"], "found")
        self.assertTrue(result["results"])
        for row in result["results"]:
            for field in (
                "corpus_id",
                "legal_unit_id",
                "evidence_id",
                "citation_id",
                "viewer_ref_id",
                "title",
                "snippet",
                "retrieval_method",
                "reasons",
                "status",
            ):
                self.assertIn(field, row)
            self.assertEqual(row["status"], "evidence")
            self.assertNotRegex(row["title"], r"^\([0-9]+\)$")

        weak = self.service.search("uud", "aturan KUHP tentang pencurian")
        self.assertEqual(weak["status"], "found")
        self.assertEqual(weak["public_status"], "no_results")
        self.assertFalse(weak["results"])

    def test_bookmarks_store_pointers_only(self) -> None:
        evidence_id = self.service.citation("uud", "Pasal 1 ayat (3)")["matches"][0]["evidence_id"]
        saved = self.service.bookmark("uud", evidence_id, note="cek lagi")
        self.assertEqual(saved["status"], "saved")
        bookmark = saved["bookmark"]
        self.assertEqual(bookmark["evidence_id"], evidence_id)
        self.assertEqual(bookmark["status"], "active")
        self.assertNotIn("quoted_text", bookmark)
        self.assertNotIn("source_sha256", bookmark)
        self.assertTrue(self.service.bookmarks("uud")["bookmarks"])
        self.assertEqual(self.service.bookmarks("uud")["persistence"], "memory")
        self.assertEqual(self.service.bookmarks("uud")["persistence_label"], "temporary_process_memory")
        self.assertEqual(self.service.bookmark("uud", "missing")["status"], "unavailable")

    def test_retrieval_units_reference_final_evidence(self) -> None:
        from tjipto.core.manifest import read_jsonl

        config = CorpusRegistry(ROOT).resolve("uud")
        evidence_ids = {
            row["evidence_id"]
            for row in read_jsonl(ROOT / "data/final/uud/evidence_registry.jsonl")
        }
        bbox_ids = {
            row["bbox_id"]
            for row in read_jsonl(ROOT / "data/final/uud/bbox_registry.jsonl")
        }
        rows = read_jsonl(ROOT / "data/final/uud/retrieval_units.jsonl")
        self.assertEqual(len(rows), config.manifest["counts"]["retrieval_units"])
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
        self.assertEqual(answer["route"], "legal_reference")
        self.assertEqual(answer["intent"], "exact_citation")
        self.assertEqual(answer["normalized_query"], "Pasal 1 ayat (3)")
        self.assertIn("Dukungan sitasi berbasis bukti", answer["answer"])
        self.assertNotIn("Evidence-grounded", answer["answer"])
        self.assertTrue(answer["evidence"])
        first = answer["evidence"][0]
        self.assertTrue(first["evidence_id"])
        self.assertGreater(first["bbox_count"], 0)
        self.assertTrue(first["viewer_ref"])

        limited = self.service.ask("uud", "negara hukum")
        self.assertEqual(limited["status"], "limited_answer")
        self.assertEqual(limited["route"], "lexical_fallback")
        self.assertEqual(limited["intent"], "natural_language")
        self.assertTrue(limited["evidence"])

        for query in ("Pasal 999", "Pasal 1 ayat 999", "Pasal 28E ayat (999)"):
            result = self.service.ask("uud", query)
            self.assertEqual(result["status"], "citation_not_found")
            self.assertEqual(result["route"], "legal_reference")
            self.assertEqual(result["reason"], "citation_not_found")
            self.assertFalse(result["evidence"])

        domain_query = self.service.ask("uud", "aturan KUHP tentang pencurian")
        self.assertEqual(domain_query["status"], "insufficient_evidence")
        self.assertEqual(domain_query["route"], "lexical_fallback")
        self.assertIsNone(domain_query["required_corpus"])
        self.assertFalse(domain_query["evidence"])
        self.assertFalse(domain_query["citations"])
        self.assertFalse(domain_query["viewer_refs"])

        unsupported = self.service.ask("unknown", "Pasal 1")
        self.assertEqual(unsupported["status"], "unsupported_corpus")
        self.assertEqual(unsupported["intent"], "unsupported_corpus")

    def test_ask_answers_grounded_document_metadata(self) -> None:
        for case in _metadata_runtime_cases():
            result = self.service.ask("uud", case["query"])
            self.assertEqual(result["status"], case["status"], case["query"])
            self.assertEqual(result["route"], case["route"], case["query"])
            self.assertEqual(result["intent"], case["intent"], case["query"])
            if case["status"] == "answer_ready":
                self.assertEqual(result["answer_type"], "metadata_fact", case["query"])
                self.assertEqual(result["metadata_facts"][0]["field"], case["field"], case["query"])
                self.assertEqual(result["metadata_facts"][0]["answer"], case["answer"], case["query"])
                self.assertEqual(result["citations"][0]["evidence_id"], case["evidence_id"], case["query"])
                self.assertEqual(result["citations"][0]["metadata_field"], case["field"], case["query"])
                self.assertEqual(result["citations"][0]["metadata_answer"], case["answer"], case["query"])
                self.assertFalse(result["viewer_refs"][0]["can_resolve"], case["query"])
            else:
                self.assertFalse(result["metadata_facts"], case["query"])
                self.assertFalse(result["citations"], case["query"])
            for unexpected in case["not_contains"]:
                self.assertNotIn(unexpected, result["answer"], case["query"])

    def test_ask_answers_grounded_legal_unit_relations(self) -> None:
        for case in _relation_cases():
            result = self.service.ask("uud", case["query"], limit=5)
            self.assertEqual(result["status"], "answer_ready", case["query"])
            self.assertEqual(result["route"], "legal_relation", case["query"])
            self.assertEqual(result["intent"], "legal_relation_lookup", case["query"])
            self.assertEqual(result["answer_type"], "legal_relation", case["query"])
            self.assertEqual(
                {row["target_label"] for row in result["legal_relations"]},
                set(case["expected_target_labels"]),
                case["query"],
            )
            self.assertEqual({row["candidate_type"] for row in result["evidence"]}, {"relation_candidate"}, case["query"])
            self.assertTrue(result["citations"], case["query"])
            self.assertTrue(result["viewer_refs"], case["query"])

    def test_relation_fixture_asserts_stable_runtime_and_graph_ids(self) -> None:
        from tjipto.core.manifest import read_jsonl

        graph_edges = {
            row["edge_id"]: row
            for row in read_jsonl(ROOT / "data/final/uud/graph_edges.jsonl")
        }
        for case in _relation_cases():
            result = self.service.ask("uud", case["query"])
            self.assertEqual(result["status"], "answer_ready", case["query"])
            self.assertEqual(result["route"], "legal_relation", case["query"])
            relation = result["legal_relations"][0]
            self.assertEqual(relation["relation_type"], case["runtime_relation_type"], case["query"])
            self.assertEqual(relation["source_legal_unit_id"], case["source_legal_unit_id"], case["query"])
            self.assertEqual(relation["target_legal_unit_id"], case["target_legal_unit_id"], case["query"])
            self.assertEqual(relation["source_label"], case["source_label"], case["query"])
            self.assertEqual(relation["target_label"], case["target_label"], case["query"])
            edge = graph_edges[case["graph_edge_id"]]
            self.assertEqual(edge["relation_type"], case["graph_relation_type"], case["query"])
            self.assertEqual(edge["source_id"], f"legal_unit::{case['graph_source_legal_unit_id']}", case["query"])
            self.assertEqual(edge["target_id"], f"legal_unit::{case['graph_target_legal_unit_id']}", case["query"])

    def test_ask_does_not_promote_ungrounded_legal_relations(self) -> None:
        for case in _unsupported_relation_cases():
            result = self.service.ask("uud", case["query"])
            self.assertEqual(result["status"], case["status"], case["query"])
            self.assertEqual(result["route"], case["route"], case["query"])
            self.assertEqual(result["intent"], case["intent"], case["query"])
            for field in case["expect_empty"]:
                self.assertFalse(result[field], case["query"])

    def test_ask_answers_instrument_scope_and_clause_queries(self) -> None:
        for case in _instrument_runtime_cases():
            result = self.service.ask("uud", case["query"])
            self.assertEqual(result["status"], case["status"], case["query"])
            self.assertEqual(result["route"], case["route"], case["query"])
            self.assertEqual(result["intent"], case["intent"], case["query"])
            self.assertEqual(result["evidence"][0]["candidate_type"], case["candidate_type"], case["query"])
            self.assertEqual(result["evidence"][0]["evidence_id"], case["evidence_id"], case["query"])
            self.assertEqual(result["citations"][0]["citation"], case["citation"], case["query"])
            self.assertEqual(result["citations"][0]["evidence_id"], case["evidence_id"], case["query"])

    def test_ask_explains_known_source_anomalies_safely(self) -> None:
        for case in _source_conflict_cases():
            result = self.service.ask("uud", case["query"])
            self.assertEqual(result["status"], "insufficient_evidence", case["query"])
            self.assertEqual(result["route"], "source_anomaly_explanation", case["query"])
            self.assertEqual(result["source_conflict"]["source_conflict_id"], case["source_conflict_id"], case["query"])
            for reason in case["expected_insufficient_reasons"]:
                self.assertIn(reason, result["insufficient_reasons"], case["query"])
            self.assertFalse(result["citations"], case["query"])
            self.assertFalse(result["viewer_refs"], case["query"])

    def test_viewer_payload_is_multi_page_and_public_safe(self) -> None:
        evidence_id = "uud_current_consolidated_final_citation_evidence_00232"
        viewer = self.service.viewer("uud", evidence_id)
        self.assertEqual(viewer["status"], "viewer_payload_ready")
        self.assertEqual(set(viewer["page_numbers"]), {2, 3})
        self.assertTrue({2, 3} <= {row["page_number"] for row in viewer["bbox_rectangles"]})
        self.assertNotIn("source_pdf_path", viewer)
        self.assertNotIn("source_sha256", viewer)
        self.assertNotIn("source_sha256", viewer["pdf"]["access_url"])
        self.assertNotIn("source_pdf_path", viewer["pdf"]["access_url"])

    def test_public_viewer_payload_exposes_highlightability_contract(self) -> None:
        exact = handle_request(
            "uud",
            "viewer",
            {"evidence_id": _exact_highlightable_evidence_id()},
            ROOT,
        )
        self.assertTrue(exact["bbox_rectangles"])
        self.assertEqual(exact["bbox_rectangles"][0]["bbox_precision"], "exact")
        self.assertTrue(exact["bbox_rectangles"][0]["viewer_highlightable"])
        self.assertNotIn("source_pdf_path", exact["bbox_rectangles"][0])
        self.assertNotIn("source_sha256", exact["bbox_rectangles"][0])
        self.assertNotIn("evidence_id", exact["bbox_rectangles"][0])
        self.assertNotIn("source_document_id", exact["bbox_rectangles"][0])

        decision = handle_request(
            "uud",
            "viewer",
            {"evidence_id": _page_grounded_decision_evidence_id()},
            ROOT,
        )
        self.assertEqual(decision["status"], "source_page_trace_only")
        self.assertTrue(decision["pdf_access_available"])
        self.assertFalse(decision["rendering_available"])
        self.assertFalse(decision["bbox_rectangles"])
        self.assertEqual(decision["bbox_precision"], "page_grounded_only")
        self.assertFalse(decision["viewer_highlightable"])

    def test_public_bbox_defaults_fail_closed(self) -> None:
        for value in (None, "oops"):
            row = _public_bbox({"bbox_precision": value, "viewer_highlightable": True})
            self.assertEqual(row["bbox_precision"], "page_grounded_only")
            self.assertFalse(row["viewer_highlightable"])
        self.assertFalse(_public_bbox({"bbox_precision": "exact", "viewer_highlightable": None})["viewer_highlightable"])

    def test_query_normalization_and_intent_classification(self) -> None:
        for case in _query_intent_cases():
            if "normalized_query" in case:
                config = CorpusRegistry(ROOT).resolve("uud") if case.get("use_config") else None
                result = normalize_query(case["query"], strategy=case.get("strategy", "generic"), config=config)
                self.assertEqual(result["normalized_query"], case["normalized_query"], case["query"])
            if "intent" in case:
                config = CorpusRegistry(ROOT).resolve("uud") if case.get("use_config") else None
                result = classify_intent(
                    case["corpus_id"],
                    case["query"],
                    strategy=case.get("strategy", "generic"),
                    corpus_supported=case.get("corpus_supported", True),
                    config=config,
                )
                self.assertEqual(result["intent"], case["intent"], case["query"])
                for field in case.get("not_in") or ():
                    self.assertNotIn(field, result, case["query"])

    def test_bm25_relevance_gate_keeps_core_uud_queries_answerable(self) -> None:
        for case in _bm25_relevance_cases():
            result = self.service.ask("uud", case["query"])
            self.assertEqual(result["status"], case["status"], case["query"])
            self.assertEqual(result["route"], case["route"], case["query"])
            self.assertIsNone(result["required_corpus"], case["query"])
            self.assertEqual(bool(result["evidence"]), case["has_evidence"], case["query"])

    def test_weak_bm25_matches_do_not_become_final_payloads(self) -> None:
        for case in _weak_bm25_cases():
            result = self.service.ask("uud", case["query"])
            self.assertIn(result["status"], set(case["statuses"]), case["query"])
            self.assertIsNone(result["required_corpus"], case["query"])
            self.assertFalse(result["evidence"], case["query"])
            self.assertFalse(result["citations"], case["query"])
            self.assertFalse(result["viewer_refs"], case["query"])
            self.assertFalse(result["context_pack"]["answer_evidence"], case["query"])
            self.assertFalse(result["context_pack"]["citation_payloads"], case["query"])
            self.assertFalse(result["context_pack"]["viewer_refs"], case["query"])
            reasons = set(result["context_pack"]["validation_reasons"].values())
            if result["route"] == "lexical_fallback":
                self.assertIn("insufficient_query_support", reasons, case["query"])

    def test_retrieval_router_envelope_routes(self) -> None:
        config = CorpusRegistry(ROOT).resolve("uud")
        store = EvidenceStore(config)
        self.assertEqual(config.query_strategy, "uud_1945")
        self.assertEqual(config.structured_strategy, "uud_1945")
        self.assertIn("current_consolidated", config.source_roles)
        self.assertIn("amendment_1_historical", config.temporal_contexts)
        self.assertEqual(config.preferred_source_role, "current_consolidated")

        for case in _retrieval_router_cases():
            case_store = None if case.get("store", "default") is None else store
            result = route_retrieval(
                case["corpus_id"],
                case["query"],
                case_store,
                limit=case.get("limit", 5),
                route=case.get("requested_route"),
            )
            for field in ("status", "route", "reason", "intent", "normalized_query", "required_corpus"):
                if field in case:
                    self.assertEqual(result[field], case[field], case["query"])
            if "has_matches" in case:
                self.assertEqual(bool(result["matches"]), case["has_matches"], case["query"])
            if "max_matches" in case:
                self.assertLessEqual(len(result["matches"]), case["max_matches"], case["query"])
            if "lexical_relevance_ok" in case:
                self.assertTrue(all(row["lexical_relevance_ok"] is case["lexical_relevance_ok"] for row in result["matches"]))

    def test_metadata_filtering_limits_retrieval_safely(self) -> None:
        config = CorpusRegistry(ROOT).resolve("uud")
        store = EvidenceStore(config)

        filters = normalize_filters({"source_role": "current_consolidated"}, config=config)
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
        self.assertEqual(api_result["route"], "legal_reference")
        self.assertEqual(api_result["citations"][0]["temporal_context"], "amendment_1_historical")

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
            self.assertEqual(report[key]["total"], 651)
            self.assertEqual(report[key]["raw_pdf_match"], 624)
            self.assertEqual(report[key]["normalized_pdf_match"], 624)
            self.assertEqual(report[key]["header_stripped_pdf_match"], 648)
            self.assertEqual(report[key]["evidence_grounded_match"], 464)
            self.assertEqual(report[key]["needs_review"], 3)
            self.assertEqual(report[key]["status"], "pass_with_reviewed_exceptions")
        self.assertEqual(report["provenance_exception_health"]["unresolved_needs_review_count"], 0)

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

        bab_xa = self.service.search("uud", "BAB XA", limit=3)
        self.assertEqual(bab_xa["status"], "found")
        self.assertEqual(bab_xa["route"], "structured")
        self.assertEqual(bab_xa["intent"], "structured_lookup")
        self.assertTrue(bab_xa["results"])
        self.assertTrue(all(row["title"].startswith("BAB XA") for row in bab_xa["results"]))

        pasal_28a = self.service.search("uud", "Pasal 28A", limit=1)
        self.assertEqual(pasal_28a["status"], "found")
        self.assertTrue(pasal_28a["results"][0]["title"].startswith("BAB XA"))

        pasal_28 = self.service.search("uud", "Pasal 28", limit=1)
        self.assertEqual(pasal_28["status"], "found")
        self.assertTrue(pasal_28["results"][0]["title"].startswith("BAB X / Pasal 28"))

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

        NoBBoxStore.config = config
        self.assertEqual(structured_lookup(NoBBoxStore(), "Pasal 1 ayat (1)"), ())

        class UnitBackedStore:
            evidence = (
                {"evidence_id": "e2", "legal_unit_id": "lu2", "status": "final", "hierarchy": []},
                {"evidence_id": "e3", "legal_unit_id": "lu3", "status": "final", "hierarchy": ["BAB X A"]},
            )
            legal_units = (
                {"legal_unit_id": "lu2", "unit_label": "Pasal 9", "hierarchy": []},
                {"legal_unit_id": "lu3", "unit_label": "BAB X A", "hierarchy": []},
            )
            chunks = ()

            def bboxes_for(self, evidence_id):
                return [{"bbox_id": "b2"}]

        UnitBackedStore.config = config
        self.assertEqual(structured_lookup(UnitBackedStore(), "Pasal 9")[0]["evidence_id"], "e2")
        self.assertEqual(structured_lookup(UnitBackedStore(), "BAB XA")[0]["evidence_id"], "e3")

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
            "label",
            "hierarchy",
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
        self.assertEqual(structured["status"], "answer_ready")
        self.assertEqual(structured["route"], "legal_reference")
        self.assertEqual(structured["answer_type"], "quoted_evidence")
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
            legal_units = ({"legal_unit_id": "lu", "runtime_loadable": True, "text_span_ids": ("s1",)},)
            chunks = ({"legal_unit_id": "lu", "runtime_loadable": True, "text_span_ids": ("s1",)},)
            retrieval_units = ()

            def bboxes_for(self, evidence_id):
                return [] if evidence_id == "missing_bbox" else [{"bbox_id": evidence_id}]

        store = Store()
        base = {
            "evidence_id": "direct",
            "legal_unit_id": "lu",
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

    def test_public_answerability_rejects_unsafe_instrument_records(self) -> None:
        class Store:
            legal_units = (
                {"legal_unit_id": "lu_runtime", "runtime_loadable": True, "text_span_ids": ("s1",)},
                {"legal_unit_id": "lu_blocked", "runtime_loadable": False, "text_span_ids": ("s1",)},
            )
            chunks = (
                {"legal_unit_id": "lu_runtime", "runtime_loadable": True, "text_span_ids": ("s1",)},
                {"legal_unit_id": "lu_blocked", "runtime_loadable": False, "text_span_ids": ("s1",)},
            )
            retrieval_units = (
                {"evidence_id": "not_accepted", "status": "excluded_public_answer"},
            )

            def bboxes_for(self, evidence_id):
                return [{"bbox_id": evidence_id}]

        store = Store()
        base = {
            "evidence_id": "safe",
            "legal_unit_id": "lu_runtime",
            "status": "final",
            "citation": "Perubahan",
            "quoted_text": "quoted",
            "source_pdf_path": "source.pdf",
            "source_sha256": "sha",
            "source_role": "amendment_1_historical",
            "temporal_context": "amendment_1_historical",
            "page_numbers": (1,),
            "route_sources": ("bm25",),
            "bbox_precision": "exact",
            "viewer_highlightable": True,
        }
        self.assertEqual(validate_answer_candidate(store, base), (True, "answer_evidence"))
        self.assertEqual(
            validate_answer_candidate(store, base | {"evidence_id": "page", "bbox_precision": "page_grounded_only"}),
            (False, "page_grounded_only_not_answerable"),
        )
        self.assertEqual(
            validate_answer_candidate(store, base | {"evidence_id": "viewer", "viewer_highlightable": False}),
            (False, "viewer_not_highlightable"),
        )
        self.assertEqual(
            validate_answer_candidate(store, base | {"evidence_id": "blocked_unit", "legal_unit_id": "lu_blocked"}),
            (False, "linked_legal_unit_not_runtime_loadable"),
        )
        self.assertEqual(
            validate_answer_candidate(store, base | {"evidence_id": "not_accepted"}),
            (False, "retrieval_unit_backing_record_not_answerable"),
        )

    def test_instrument_page_grounded_records_do_not_leak_publicly(self) -> None:
        cases = (
            ("Perubahan Pertama Decision", "00623"),
            ("Perubahan Ketiga Decision", "00634"),
            ("Perubahan Keempat Decision", "00648"),
            ("Perubahan Ketiga Scope", "00632"),
            ("Perubahan Keempat Scope", "00639"),
        )
        for query, key in cases:
            evidence_id = UNSAFE_INSTRUMENT_EVIDENCE[key]
            ask = self.service.ask("uud", query, limit=10)
            search = self.service.search("uud", query, limit=10)
            self.assertNotIn(evidence_id, {row["evidence_id"] for row in ask["evidence"]}, query)
            self.assertNotIn(evidence_id, {row["evidence_id"] for row in search["results"]}, query)
            excluded = {
                row["evidence_id"]: row
                for row in (*ask["context_pack"]["excluded_results"], *search["context_pack"]["excluded_results"])
            }
            if evidence_id in excluded:
                self.assertIn(
                    excluded[evidence_id]["reason"],
                    {"page_grounded_only_not_answerable", "exact_instrument_unit_fail_closed"},
                    query,
                )
                self.assertFalse(excluded[evidence_id]["viewer_ref"]["can_resolve"], query)

    def test_exact_fail_closed_instrument_queries_do_not_substitute_neighbors(self) -> None:
        forbidden = (
            "Recital",
            "Determination",
            "Closing",
            "Signatories",
            "Clause (a)",
            "Clause (b)",
            "Clause (c)",
            "Clause (d)",
        )
        for query in (
            "Perubahan Pertama Decision",
            "Perubahan Ketiga Decision",
            "Perubahan Keempat Decision",
            "Perubahan Ketiga Scope",
            "Perubahan Keempat Scope",
        ):
            ask = self.service.ask("uud", query, limit=10)
            search = self.service.search("uud", query, limit=10)
            self.assertEqual(ask["status"], "insufficient_evidence", query)
            self.assertEqual(ask["route"], "exact_instrument_fail_closed", query)
            self.assertFalse(ask["evidence"], query)
            self.assertFalse(search["results"], query)
            self.assertEqual(search["public_status"], "no_results", query)
            self.assertIn("exact_instrument_unit_fail_closed", ask["insufficient_reasons"], query)
            self.assertTrue(ask["context_pack"]["excluded_results"], query)
            self.assertEqual(ask["context_pack"]["excluded_results"][0]["reason"], "exact_instrument_unit_fail_closed", query)
            for row in (*ask["evidence"], *search["results"]):
                self.assertFalse(any(token in (row.get("citation") or "") for token in forbidden), query)

    def test_natural_instrument_queries_do_not_substitute_neighbors(self) -> None:
        cases = (
            ("decision perubahan ketiga", "Perubahan Ketiga Decision"),
            ("Perubahan Ketiga Decision dalam UUD", "Perubahan Ketiga Decision"),
            ("decision perubahan keempat", "Perubahan Keempat Decision"),
            ("scope perubahan ketiga", "Perubahan Ketiga Scope"),
            ("scope perubahan keempat", "Perubahan Keempat Scope"),
            ("amandemen keempat scope", "Perubahan Keempat Scope"),
            ("apa scope amandemen keempat", "Perubahan Keempat Scope"),
            ("perubahan ke-4 scope", "Perubahan Keempat Scope"),
            ("perubahan ke 4 scope", "Perubahan Keempat Scope"),
            ("amandemen IV scope", "Perubahan Keempat Scope"),
            ("scope: perubahan keempat", "Perubahan Keempat Scope"),
            ("lingkup perubahan keempat", "Perubahan Keempat Scope"),
            ("cakupan amandemen keempat", "Perubahan Keempat Scope"),
            ("apa yang diubah perubahan keempat", "Perubahan Keempat Scope"),
            ("perubahan keempat mengubah apa", "Perubahan Keempat Scope"),
            ("materi perubahan keempat", "Perubahan Keempat Scope"),
            ("substansi amandemen keempat", "Perubahan Keempat Scope"),
            ("decision, perubahan ketiga", "Perubahan Ketiga Decision"),
            ("decision: perubahan ketiga", "Perubahan Ketiga Decision"),
        )
        forbidden = ("Determination", "Recital", "Closing", "Signatories", "Clause")
        for query, citation in cases:
            ask = self.service.ask("uud", query, limit=10)
            search = self.service.search("uud", query, limit=10)
            self.assertEqual(ask["status"], "insufficient_evidence", query)
            self.assertEqual(ask["route"], "instrument_resolved_fail_closed", query)
            self.assertFalse(ask["evidence"], query)
            self.assertFalse(search["results"], query)
            self.assertEqual(search["public_status"], "no_results", query)
            self.assertEqual(ask["context_pack"]["excluded_results"][0]["citation"], citation, query)
            self.assertEqual(ask["context_pack"]["excluded_results"][0]["reason"], "instrument_resolved_fail_closed", query)
            for row in (*ask["evidence"], *search["results"]):
                self.assertFalse(any(token in (row.get("citation") or "") for token in forbidden), query)

    def test_instrument_intent_matrix_blocks_neighbor_fallback(self) -> None:
        intent = CorpusRegistry(ROOT).resolve("uud").setting("intent_config")
        matrix = intent["instrument_intent_matrix"]
        queries = [
            template.format(role=role, amendment=amendment)
            for role in matrix["role_family_terms"]
            for amendment in matrix["amendment_terms"]
            for template in matrix["word_orders"]
        ]
        self.assertGreater(len(queries), 0)
        forbidden = ("Determination", "Recital", "Closing", "Signatories", "Clause")
        for query in queries:
            ask = self.service.ask("uud", query, limit=10)
            search = self.service.search("uud", query, limit=10)
            self.assertFalse(ask["route"] == "lexical_fallback" and ask["evidence"], query)
            self.assertFalse(search["route"] == "bm25" and search["results"], query)
            for row in (*ask["evidence"], *search["results"]):
                citation = row.get("citation") or ""
                self.assertIn("Scope", citation, query)
                self.assertFalse(any(token in citation for token in forbidden), query)

        for query in (
            "daftar pasal perubahan pertama",
            "perubahan pertama daftar pasal",
            "daftar pasal amandemen keempat",
            "ruanglingkup perubahan pertama",
        ):
            ask = self.service.ask("uud", query, limit=10)
            search = self.service.search("uud", query, limit=10)
            self.assertFalse(ask["route"] == "lexical_fallback" and ask["evidence"], query)
            self.assertFalse(search["route"] == "bm25" and search["results"], query)
            for row in (*ask["evidence"], *search["results"]):
                self.assertIn("Scope", row.get("citation") or "", query)

        ask = self.service.ask("uud", "huruf perubahan keempat", limit=10)
        search = self.service.search("uud", "huruf perubahan keempat", limit=10)
        self.assertEqual(ask["route"], "instrument_unresolved")
        self.assertFalse(ask["evidence"])
        self.assertEqual(search["public_status"], "no_results")
        self.assertFalse(search["results"])

    def test_partial_signal_instrument_queries_fail_closed_before_bm25(self) -> None:
        queries = (
            "ketentuan yang berubah perubahan pertama",
            "ketentuan yang berubah perubahan keempat",
            "ubah pasal apa perubahan keempat",
            "pasal apa yang diubah amandemen keempat",
            "ketentuan apa yang diubah perubahan ketiga",
            "objek perubahan keempat",
            "sasaran perubahan keempat",
            "isi perubahan pertama",
            "norma yang berubah amandemen keempat",
            "bagian yang diganti amandemen keempat",
            "pasal terdampak perubahan pertama",
        )
        forbidden = ("Determination", "Recital", "Closing", "Signatories", "Clause")
        for query in queries:
            ask = self.service.ask("uud", query, limit=10)
            search = self.service.search("uud", query, limit=10)
            self.assertEqual(ask["status"], "insufficient_evidence", query)
            self.assertIn(ask["route"], {"instrument_unresolved", "instrument_resolved_fail_closed"}, query)
            self.assertFalse(ask["evidence"], query)
            self.assertFalse(search["results"], query)
            self.assertEqual(search["public_status"], "no_results", query)
            for row in (*ask["evidence"], *search["results"]):
                self.assertFalse(any(token in (row.get("citation") or "") for token in forbidden), query)

    def test_partial_signal_boundary_does_not_overblock_noninstrument_routes(self) -> None:
        education = self.service.ask("uud", "pasal apa yang mengatur pendidikan", limit=10)
        self.assertNotIn(education["route"], {"instrument_unresolved", "instrument_resolved_fail_closed"})
        self.assertNotEqual(education["status"], "insufficient_evidence")

        pasal = self.service.ask("uud", "apa isi Pasal 31", limit=10)
        self.assertEqual(pasal["route"], "legal_reference")
        self.assertTrue(pasal["evidence"])

        date = self.service.ask("uud", "kapan perubahan keempat ditetapkan", limit=10)
        self.assertEqual(date["route"], "metadata_fact")

        institution = self.service.ask("uud", "lembaga yang menetapkan perubahan keempat", limit=10)
        self.assertEqual(institution["route"], "metadata_fact")

        relation = self.service.ask("uud", "relasi Pasal 31 dengan pendidikan", limit=10)
        self.assertIn(relation["route"], {"legal_relation", "legal_reference", "lexical_fallback"})

    def test_instrument_like_content_and_effect_queries_fail_closed_before_bm25(self) -> None:
        queries = (
            "muatan perubahan keempat",
            "rincian perubahan keempat",
            "apa dampak perubahan keempat",
            "dampak amandemen keempat",
            "implikasi perubahan keempat",
            "akibat amandemen keempat",
            "konsekuensi perubahan keempat",
            "objek perubahan keempat",
            "sasaran perubahan keempat",
            "hal yang berubah perubahan pertama",
            "bagian terdampak amandemen keempat",
        )
        forbidden = ("Determination", "Recital", "Closing", "Clause", "Signatories")
        for query in queries:
            ask = self.service.ask("uud", query, limit=10)
            search = self.service.search("uud", query, limit=10)
            self.assertEqual(ask["status"], "insufficient_evidence", query)
            self.assertEqual(ask["route"], "instrument_unresolved", query)
            self.assertNotEqual(ask["reason"], "instrument_unresolved", query)
            self.assertFalse(ask["evidence"], query)
            self.assertFalse(search["results"], query)
            self.assertEqual(search["public_status"], "no_results", query)
            self.assertNotEqual(search["route"], "bm25", query)
            for row in (*ask["evidence"], *search["results"]):
                self.assertFalse(any(token in (row.get("citation") or "") for token in forbidden), query)

    def test_unsupported_analysis_intent_fails_closed_before_bm25(self) -> None:
        queries = (
            "risiko perubahan keempat",
            "tujuan perubahan keempat",
            "alasan perubahan keempat",
            "latar belakang perubahan keempat",
            "maksud perubahan keempat",
            "makna perubahan keempat",
            "urgensi perubahan keempat",
            "dasar perubahan keempat",
            "rasional amandemen keempat",
            "pertimbangan perubahan ketiga",
            "motif amandemen pertama",
        )
        forbidden = ("Determination", "Recital", "Closing", "Clause", "Signatories")
        for query in queries:
            ask = self.service.ask("uud", query, limit=10)
            search = self.service.search("uud", query, limit=10)
            self.assertEqual(ask["status"], "insufficient_evidence", query)
            self.assertEqual(ask["route"], "instrument_unresolved", query)
            self.assertEqual(ask["reason"], "unsupported_analysis_intent", query)
            self.assertFalse(ask["evidence"], query)
            self.assertFalse(search["results"], query)
            self.assertEqual(search["public_status"], "no_results", query)
            self.assertNotEqual(search["route"], "bm25", query)
            for row in (*ask["evidence"], *search["results"]):
                self.assertFalse(any(token in (row.get("citation") or "") for token in forbidden), query)

    def test_contextual_numbers_do_not_create_amendment_instrument_intent(self) -> None:
        impact = self.service.ask("uud", "apa dampak Pasal 31 ayat 1", limit=10)
        self.assertNotIn(impact["route"], {"instrument_unresolved", "instrument_resolved_fail_closed"})

        ayat = self.service.ask("uud", "apa isi Pasal 31 ayat 2", limit=10)
        self.assertEqual(ayat["route"], "legal_reference")
        self.assertTrue(ayat["evidence"])

        roman = self.service.ask("uud", "Pasal IV", limit=10)
        self.assertNotIn(roman["route"], {"instrument_unresolved", "instrument_resolved_fail_closed"})

    def test_general_perubahan_topics_are_not_instrument_overblocked(self) -> None:
        for query in (
            "pasal apa yang mengatur perubahan iklim",
            "apa isi pasal tentang perubahan iklim",
            "perubahan sosial dalam UUD",
            "pasal yang mengatur perubahan masyarakat",
        ):
            result = self.service.ask("uud", query, limit=10)
            self.assertNotIn(result["route"], {"instrument_unresolved", "instrument_resolved_fail_closed"}, query)

    def test_natural_instrument_exact_labels_rank_first(self) -> None:
        for query, evidence_id in (
            ("Perubahan Pertama Scope", SAFE_INSTRUMENT_EVIDENCE["00621"]),
            ("Perubahan Kedua Scope", SAFE_INSTRUMENT_EVIDENCE["00628"]),
            ("Perubahan Keempat Recital", SAFE_INSTRUMENT_EVIDENCE["00638"]),
            ("Perubahan Pertama Scope?", SAFE_INSTRUMENT_EVIDENCE["00621"]),
            ("Perubahan Kedua Scope?", SAFE_INSTRUMENT_EVIDENCE["00628"]),
            ("Perubahan Keempat Recital?", SAFE_INSTRUMENT_EVIDENCE["00638"]),
        ):
            result = self.service.search("uud", query, limit=10)
            self.assertEqual(result["route"], "instrument_resolved_answerable", query)
            self.assertTrue(result["results"], query)
            self.assertEqual(result["results"][0]["evidence_id"], evidence_id, query)

    def test_page_grounded_instrument_viewer_is_trace_only(self) -> None:
        for evidence_id in UNSAFE_INSTRUMENT_EVIDENCE.values():
            viewer = self.service.viewer("uud", evidence_id)
            self.assertIn(viewer["status"], {"source_page_trace_only", "non_highlightable_trace"}, evidence_id)
            self.assertTrue(viewer["pdf_access_available"], evidence_id)
            self.assertFalse(viewer["rendering_available"], evidence_id)
            self.assertFalse(viewer["bbox_rectangles"], evidence_id)
            self.assertEqual(viewer["bbox_precision"], "page_grounded_only", evidence_id)
            self.assertFalse(viewer["viewer_highlightable"], evidence_id)

    def test_safe_and_fail_closed_instrument_records_keep_policy(self) -> None:
        safety = json.loads((ROOT / "data/final/uud/validation_report.json").read_text(encoding="utf-8"))[
            "instrument_runtime_safety_health"
        ]
        precision = json.loads((ROOT / "data/final/uud/validation_report.json").read_text(encoding="utf-8"))[
            "instrument_query_precision_health"
        ]
        natural_precision = json.loads((ROOT / "data/final/uud/validation_report.json").read_text(encoding="utf-8"))[
            "instrument_natural_query_precision_health"
        ]
        matrix = json.loads((ROOT / "data/final/uud/validation_report.json").read_text(encoding="utf-8"))[
            "instrument_intent_matrix_health"
        ]
        partial = json.loads((ROOT / "data/final/uud/validation_report.json").read_text(encoding="utf-8"))[
            "partial_signal_instrument_boundary_health"
        ]
        general = json.loads((ROOT / "data/final/uud/validation_report.json").read_text(encoding="utf-8"))[
            "instrument_like_boundary_generalization_health"
        ]
        invariant = json.loads((ROOT / "data/final/uud/validation_report.json").read_text(encoding="utf-8"))[
            "instrument_intent_invariant_router_health"
        ]
        self.assertEqual(safety["status"], "complete")
        self.assertEqual(precision["status"], "complete")
        self.assertEqual(natural_precision["status"], "complete")
        self.assertEqual(matrix["status"], "complete")
        self.assertEqual(partial["status"], "complete")
        self.assertEqual(general["status"], "complete")
        self.assertEqual(invariant["status"], "complete")
        self.assertGreater(matrix["matrix_query_count"], 0)
        self.assertGreater(partial["partial_signal_runtime_matrix_count"], 0)
        self.assertGreater(general["runtime_matrix_count"], 0)
        self.assertGreater(invariant["runtime_matrix_count"], 0)
        self.assertGreater(invariant["heldout_analysis_probe_count"], 0)
        for health in (safety, precision, natural_precision, matrix, partial, general, invariant):
            for key, value in health.items():
                if key.endswith("_count") and key not in {"matrix_query_count", "partial_signal_runtime_matrix_count", "runtime_matrix_count", "heldout_analysis_probe_count"}:
                    self.assertEqual(value, 0, key)
        for query, evidence_id in (
            ("Perubahan Pertama Scope", SAFE_INSTRUMENT_EVIDENCE["00621"]),
            ("Perubahan Kedua Scope", SAFE_INSTRUMENT_EVIDENCE["00628"]),
            ("Perubahan Keempat Recital", SAFE_INSTRUMENT_EVIDENCE["00638"]),
        ):
            result = self.service.search("uud", query, limit=10)
            self.assertIn(evidence_id, {row["evidence_id"] for row in result["results"]}, query)
        chunks = {
            row["chunk_id"]: row
            for row in (json.loads(line) for line in (ROOT / "data/final/uud/chunks.jsonl").read_text(encoding="utf-8").splitlines())
        }
        retrieval_chunk_ids = {
            row["chunk_id"]
            for row in (json.loads(line) for line in (ROOT / "data/final/uud/retrieval_units.jsonl").read_text(encoding="utf-8").splitlines())
        }
        for chunk_id in ("uud_chunk_00646", "uud_chunk_00647"):
            self.assertFalse(chunks[chunk_id]["runtime_loadable"])
            self.assertFalse(chunks[chunk_id]["canonical_use_allowed"])
            self.assertNotIn(chunk_id, retrieval_chunk_ids)

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
        self.assertEqual(len(config.jsonl("evidence")), config.manifest["counts"]["evidence_records"])
        self.assertEqual(len(config.jsonl("bbox")), config.manifest["counts"]["bbox_records"])
        self.assertEqual(len(config.jsonl("graph_nodes")), config.manifest["counts"]["graph_nodes"])

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
            if path.name != "http.py"
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
            self.assertEqual(config.query_strategy, "generic")
            self.assertEqual(route_retrieval("demo", "Pasal 1", store)["intent"], "natural_language")
            self.assertNotEqual(route_retrieval("demo", "Jakarta", store)["intent"], "metadata_lookup")
            self.assertNotEqual(route_retrieval("demo", "Amien Rais", store)["intent"], "metadata_lookup")
            self.assertEqual(structured_lookup(store, "Pasal 1", strategy=config.structured_strategy), ())
            filtered = route_retrieval(
                "demo",
                "generic corpus resolution",
                store,
                metadata_filters={"source_role": "current_consolidated"},
            )
            self.assertEqual(filtered["status"], "invalid_filter")
            self.assertEqual(filtered["reason"], "invalid_filter")

    def test_registry_supports_corpus_policy_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data").mkdir()
            (root / "corpus").mkdir()
            (root / "corpus/manifest.json").write_text(json.dumps({"corpus_id": "demo"}), encoding="utf-8")
            (root / "data/corpus_registry.json").write_text(
                json.dumps({
                    "demo": {
                        "manifest": "corpus/manifest.json",
                        "query_strategy": "demo_strategy",
                        "source_roles": ["published"],
                        "temporal_contexts": ["current"],
                        "preferred_source_role": "published",
                    }
                }),
                encoding="utf-8",
            )

            config = CorpusRegistry(root).resolve("demo")
            self.assertIsNotNone(config)
            self.assertEqual(config.query_strategy, "demo_strategy")
            self.assertEqual(config.source_roles, ("published",))
            self.assertEqual(config.temporal_contexts, ("current",))
            self.assertEqual(config.preferred_source_role, "published")

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
