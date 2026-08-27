from __future__ import annotations

import json
from pathlib import Path
import os
import tempfile
import unittest
from unittest.mock import patch
from typing import Any

import pytest

from tjipto.corpora.registry import CorpusRegistry
from tjipto.corpora.intent_config import intent_config_for, resolve_instrument_intent, validation_intent_config_for
from tjipto.corpora.provenance import validate_corpus_provenance
from tjipto.core.manifest import read_jsonl
from tjipto.evidence.store import EvidenceStore
from tjipto.graph.store import GraphStore
from tjipto.retrieval.candidates import merge_ranked
from tjipto.retrieval.dense import dense_search
from tjipto.retrieval.metadata import filter_evidence, normalize_filters
from tjipto.retrieval.query import classify_intent, normalize_query
from tjipto.retrieval.router import route_retrieval
from tjipto.retrieval.structured import structured_lookup
from tjipto.retrieval.answer import assemble_context_pack, validate_answer_candidate
from tjipto.runtime.api import _answer_with_footnotes, _public_bbox, handle_request
from tjipto.runtime.gemini import GeminiAnswerProvider
from tjipto.runtime.openai_compatible import OpenAICompatibleWordingProvider
from tjipto.runtime.research_control import semantic_support_context_terms
from tjipto.runtime.service import LegalRuntimeService
from tjipto.runtime.query_semantics import interpret_query
from tjipto.runtime.viewer import viewer_payload
from tests.test_source_conflict_runtime_contract import _source_conflict_cases


ROOT = Path(__file__).resolve().parents[1]


def _exact_highlightable_evidence_id() -> str:
    for row in CorpusRegistry(ROOT).resolve("uud").jsonl("evidence"):
        if row.get("viewer_highlightable") is True:
            return row["evidence_id"]
    raise AssertionError("missing exact highlightable evidence")


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

    def test_runtime_vocabulary_projection_is_small_and_duplicate_free(self) -> None:
        config = CorpusRegistry(ROOT).resolve("uud")
        intent = intent_config_for(config.query_strategy, config)
        for validation_key in (
            "instrument_scope_queries",
            "instrument_intent_matrix",
            "partial_signal_instrument_matrix",
            "instrument_like_boundary_matrix",
            "instrument_intent_invariant_matrix",
        ):
            self.assertNotIn(validation_key, intent)
        validation_intent = validation_intent_config_for(config.query_strategy, config)
        self.assertTrue(validation_intent["instrument_intent_matrix"])
        raw_intent = config.setting("intent_config")
        self.assertNotIn("instrument_scope_queries", raw_intent)
        self.assertNotIn("instrument_change_signals", raw_intent)
        self.assertNotIn("change_terms", raw_intent)
        self.assertNotIn("change_relation_signals", config.setting("research")["operation_requirements"])
        self.assertNotIn("instrument_scope_terms", config.setting("research"))
        self.assertEqual(intent["document_relation"]["change_terms"], intent["change_terms"])
        self.assertEqual(
            intent["document_relation"]["change_terms"],
            intent["document_relation"]["relation_families"]["MODIFY_PROVISION"]["terms"],
        )
        for validation_key in (
            "instrument_intent_matrix",
            "partial_signal_instrument_matrix",
            "instrument_like_boundary_matrix",
            "instrument_intent_invariant_matrix",
        ):
            self.assertEqual(set(raw_intent[validation_key]), {"word_orders"}, validation_key)
        self.assertIn("perubahan Kedua", validation_intent["instrument_intent_matrix"]["amendment_terms"])
        for key in (
            "direct_relation_words",
            "instrument_role_queries",
            "instrument_source_signals",
            "instrument_content_signals",
            "instrument_effect_signals",
            "instrument_analysis_signals",
            "instrument_legal_object_signals",
            "change_terms",
        ):
            values = intent[key]
            if isinstance(values, dict):
                values = tuple(term for terms in values.values() for term in terms)
            self.assertEqual(len(values), len(set(values)), key)
        excluded = config.setting("lexical_normalization")["semantic_support_excluded_terms"]
        self.assertNotIn("jaminan", excluded)
        self.assertNotIn("konstitusional", excluded)
        context_terms = semantic_support_context_terms(self.service._store("uud"), {})
        self.assertIn("jaminan", context_terms)
        self.assertIn("konstitusional", context_terms)

    def test_change_terms_have_one_runtime_owner(self) -> None:
        config = CorpusRegistry(ROOT).resolve("uud")
        intent = intent_config_for(config.query_strategy, config)
        change_terms = set(intent["change_terms"])
        scope_terms = {
            term.casefold()
            for terms in intent["instrument_role_queries"].values()
            for term in terms
        }
        content_terms = {
            term.casefold()
            for key in (
                "instrument_content_signals",
                "instrument_effect_signals",
                "instrument_analysis_signals",
                "instrument_legal_object_signals",
            )
            for term in intent[key]
        }
        self.assertEqual(intent["direct_relation_words"], intent["change_terms"])
        self.assertEqual(intent["document_relation"]["change_terms"], intent["change_terms"])
        self.assertFalse(change_terms & scope_terms)
        self.assertFalse(change_terms & content_terms)
    def assertPublicSearchHasNoEvidenceRows(self, search: dict, query: str) -> None:
        self.assertNotEqual(search["route"], "bm25", query)
        for row in search["results"]:
            self.assertEqual(row.get("status"), "document", query)
            self.assertNotIn("evidence_id", row, query)
            self.assertEqual(row.get("bbox_count"), 0, query)

    def test_public_answer_places_clickable_footnotes_on_supported_sentences(self) -> None:
        projected = (
            ({"citation": {"number": 1}}, {"evidence_id": "pasal-28"}),
            ({"citation": {"number": 2}}, {"evidence_id": "pasal-28a"}),
        )
        answer = "Pasal 28 mengatur kebebasan. [[support:pasal-28]]\n\nPasal 28A mengatur hak hidup. [[support:pasal-28a]]"
        self.assertEqual(
            _answer_with_footnotes(answer, projected),
            "Pasal 28 mengatur kebebasan. [1]\n\nPasal 28A mengatur hak hidup. [2]",
        )
        self.assertEqual(
            _answer_with_footnotes("Pasal 28 mengatur kebebasan. [[support:pasal-28]]", projected[:1]),
            "Pasal 28 mengatur kebebasan. [1]",
        )

    def test_search_citation_and_viewer_work(self) -> None:
        search = self.service.search("uud", "UUD 1945", limit=3)
        self.assertEqual(search["status"], "found")
        self.assertEqual(search["route"], "document_catalog")
        self.assertTrue(search["results"])
        self.assertTrue(all(row["status"] == "document" for row in search["results"]))

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

    def test_search_results_are_public_document_payloads(self) -> None:
        result = self.service.search("uud", "Perubahan Ketiga UUD", limit=2)
        self.assertEqual(result["status"], "found")
        self.assertTrue(result["results"])
        for row in result["results"]:
            for field in (
                "corpus_id",
                "document_id",
                "source_document_id",
                "title",
                "snippet",
                "status",
            ):
                self.assertIn(field, row)
            self.assertEqual(row["status"], "document")
            self.assertEqual(row["bbox_count"], 0)
            self.assertEqual(row["viewer_ref"]["source_document_id"], row["source_document_id"])
            self.assertNotRegex(row["title"], r"^\([0-9]+\)$")

        weak = self.service.search("uud", "hak pendidikan")
        self.assertEqual(weak["status"], "no_results")
        self.assertEqual(weak["public_status"], "no_results")
        self.assertFalse(weak["results"])

        current_fact = self.service.search("uud", "siapa presiden indonesia sekarang?")
        self.assertEqual(current_fact["status"], "no_results")
        self.assertFalse(current_fact["results"])

    def test_public_search_is_document_catalog(self) -> None:
        for query, expected_source in (
            ("UUD 1945", "uud::current_consolidated"),
            ("Perubahan Ketiga UUD", "uud::amendment_3_historical"),
            ("Satu Naskah UUD 1945", "uud::current_consolidated"),
        ):
            result = self.service.search("uud", query, limit=10)
            self.assertEqual(result["route"], "document_catalog", query)
            self.assertTrue(result["results"], query)
            self.assertIn(expected_source, {row["source_document_id"] for row in result["results"]}, query)
            self.assertTrue(all(row["status"] == "document" for row in result["results"]), query)
            self.assertTrue(all(not row.get("evidence_id") for row in result["results"]), query)

        viewer = self.service.viewer("uud", None, source_document_id="uud::current_consolidated")
        self.assertEqual(viewer["status"], "viewer_payload_ready")
        self.assertTrue(viewer["rendering_available"])
        self.assertFalse(viewer["bbox_rectangles"])
        self.assertFalse(viewer["viewer_highlightable"])

    def test_scope_guard_blocks_current_and_out_of_corpus_facts(self) -> None:
        for query in (
            "siapa presiden indonesia",
            "siapa wakil presiden indonesia",
            "presiden indonesia siapa",
            "wakil presiden indonesia siapa",
            "siapa presiden indonesia sekarang?",
            "siapa wakil presiden sekarang?",
        ):
            result = self.service.ask("uud", query)
            self.assertEqual(result["status"], "insufficient_evidence", query)
            self.assertEqual(result["route"], "current_fact_unsupported", query)
            self.assertFalse(result["evidence"], query)
            self.assertFalse(result["citations"], query)

        for query in ("berapa harga tanah di jakarta?", "jadwal pemilu berikutnya kapan?"):
            result = self.service.ask("uud", query)
            self.assertEqual(result["status"], "insufficient_evidence", query)
            self.assertFalse(result["evidence"], query)

        legal_norm = self.service.ask("uud", "hak pendidikan", limit=10)
        self.assertEqual(legal_norm["route"], "lexical_fallback")
        self.assertTrue(legal_norm["evidence"])

        for query in ("apa tugas presiden menurut UUD", "apa kewenangan presiden menurut UUD"):
            result = self.service.ask("uud", query, limit=10)
            self.assertNotEqual(result["route"], "current_fact_unsupported", query)

        citation = self.service.ask("uud", "apa isi Pasal 1 ayat (3)?")
        self.assertEqual(citation["route"], "legal_reference")
        self.assertTrue(citation["evidence"])

        for query in ("siapa gubernur jakarta", "jadwal konser jakarta"):
            result = self.service.ask("uud", query)
            self.assertEqual(result["status"], "insufficient_evidence", query)
            self.assertFalse(result["evidence"], query)

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
        evidence_ids = {row["evidence_id"] for row in read_jsonl(ROOT / "data/final/uud/evidence_registry.jsonl")}
        rows = read_jsonl(ROOT / "data/final/uud/retrieval_units.jsonl")
        self.assertEqual(len(rows), config.manifest["counts"]["retrieval_units"])
        for row in rows:
            self.assertIn(row["evidence_id"], evidence_ids)
            self.assertEqual(row["object_role"], "retrieval_index_record")
            self.assertEqual(row["artifact_status"], "published")
            self.assertIn("page_numbers", row["page_locator"])
            self.assertIn("source_document_id", row["page_locator"])
            self.assertNotIn("bbox_sample_refs", row)
            self.assertNotIn("citation_final", row)

    def test_graph_retrieval_eval_fixtures_resolve_refs(self) -> None:
        from tjipto.core.manifest import read_jsonl

        evidence_ids = {row["evidence_id"] for row in read_jsonl(ROOT / "data/final/uud/evidence_registry.jsonl")}
        chunk_ids = {row["chunk_id"] for row in read_jsonl(ROOT / "data/final/uud/chunks.jsonl")}
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

    def test_retrieval_readiness_eval_fixture_schema(self) -> None:
        from tjipto.core.manifest import read_jsonl

        evidence_ids = {row["evidence_id"] for row in read_jsonl(ROOT / "data/final/uud/evidence_registry.jsonl")}
        cases = read_jsonl(ROOT / "tests/fixtures/uud/retrieval_readiness_eval_cases.jsonl")
        self.assertEqual(
            {row["category"] for row in cases},
            {
                "exact_citation",
                "metadata",
                "amendment_relation",
                "natural_legal_query",
                "current_fact_negative",
                "out_of_corpus_negative",
                "source_conflict",
                "viewer_citation_resolvability",
            },
        )
        for row in cases:
            for field in ("eval_id", "category", "corpus_id", "query", "expected_status", "expected_route", "required_evidence_ids"):
                self.assertIn(field, row)
            self.assertEqual(row["corpus_id"], "uud")
            self.assertTrue(set(row["required_evidence_ids"]) <= evidence_ids)
        stage0 = [row for row in cases if row.get("stage0_defect")]
        self.assertEqual(len(stage0), 8)
        for row in stage0:
            current = row.get("stage0_current")
            self.assertIsInstance(current, dict, row["eval_id"])
            self.assertTrue(current.get("issue"), row["eval_id"])

    def test_bm25_prioritizes_term_frequency_without_breaking_exact_citation(self) -> None:
        config = CorpusRegistry(ROOT).resolve("uud")
        store = EvidenceStore(config)
        citation = self.service.citation("uud", "Pasal 1 ayat (3)")
        search = route_retrieval("uud", "Pasal 1 ayat (3)", store, limit=1, allow_bm25_after_citation_miss=True)
        self.assertEqual(search["matches"][0]["evidence_id"], citation["matches"][0]["evidence_id"])

        results = route_retrieval("uud", "negara negara negara hukum", store, limit=3)
        self.assertEqual(results["status"], "found")
        self.assertTrue(any("negara" in row["quoted_text"].casefold() for row in results["matches"]))

    def test_sparse_lane_and_complete_lexical_hit_do_not_start_dense(self) -> None:
        store = EvidenceStore(CorpusRegistry(ROOT).resolve("uud"))
        with patch("tjipto.retrieval.router.dense_runtime_available", return_value=True), patch(
            "tjipto.retrieval.router.hybrid_search", side_effect=AssertionError("dense lane must not start")
        ):
            sparse = route_retrieval("uud", "negara hukum", store, route="sparse")
            automatic = route_retrieval("uud", "masa jabatan presiden", store)
        self.assertEqual(sparse["route"], "bm25")
        self.assertEqual(automatic["route"], "bm25")
        self.assertFalse(sparse["hybrid_active"])
        self.assertFalse(automatic["hybrid_active"])

    def test_ask_contract_is_evidence_bounded(self) -> None:
        answer = self.service.ask("uud", "Pasal 1 ayat (3)")
        self.assertEqual(answer["status"], "answer_ready")
        self.assertEqual(answer["route"], "legal_reference")
        self.assertEqual(answer["intent"], "exact_citation")
        self.assertEqual(answer["normalized_query"], "Pasal 1 ayat (3)")
        self.assertIn("Negara Indonesia adalah negara hukum", answer["answer"])
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
            self.assertEqual(result["status"], "insufficient_evidence")
            self.assertEqual(result["route"], "legal_reference")
            self.assertIn(result["reason"], {"pasal_aggregate_source_missing", "structured_not_found"})
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
                self.assertEqual(result["metadata_support"][0]["evidence_id"], case["evidence_id"], case["query"])
                self.assertEqual(result["metadata_support"][0]["field"], case["field"], case["query"])
                self.assertEqual(result["metadata_support"][0]["answer"], case["answer"], case["query"])
                support = result["metadata_support"][0]
                if support["support_class"] == "exact_metadata_citation":
                    self.assertFalse(result["citations"], case["query"])
                    self.assertFalse(result["viewer_refs"], case["query"])
                    self.assertTrue(support["citation_available"], case["query"])
                    self.assertTrue(support["viewer_highlightable"], case["query"])
                    self.assertTrue(support["viewer_ref"]["can_resolve"], case["query"])
                    self.assertEqual(support["authority_kind"], "metadata_source", case["query"])
                    self.assertFalse(support["citation_final"], case["query"])
                else:
                    self.assertFalse(result["citations"], case["query"])
                    self.assertFalse(result["viewer_refs"], case["query"])
                    self.assertFalse(support["citation_available"], case["query"])
                    self.assertFalse(support["viewer_highlightable"], case["query"])
                    self.assertIsNone(support["viewer_ref"], case["query"])
                    self.assertEqual(support["authority_kind"], "metadata_trace", case["query"])
                    self.assertFalse(support["citation_final"], case["query"])
            else:
                self.assertFalse(result["metadata_facts"], case["query"])
                self.assertFalse(result["citations"], case["query"])
            for unexpected in case["not_contains"]:
                self.assertNotIn(unexpected, result["answer"], case["query"])

    def test_exact_metadata_support_resolves_viewer_without_legal_citation(self) -> None:
        result = self.service.ask("uud", "kapan perubahan pertama ditetapkan")
        support = result["metadata_support"][0]
        self.assertEqual(support["support_class"], "exact_metadata_citation")
        self.assertFalse(result["citations"])
        self.assertFalse(result["viewer_refs"])
        self.assertTrue(support["viewer_ref"]["can_resolve"])
        self.assertEqual(support["viewer_ref"]["authority_kind"], "metadata_source")
        self.assertEqual(support["viewer_ref"]["support_kind"], "metadata_source")
        self.assertFalse(support["viewer_ref"]["citation_final"])
        viewer = self.service.viewer("uud", support["evidence_id"])
        self.assertEqual(viewer["status"], "viewer_payload_ready")
        self.assertTrue(viewer["viewer_highlightable"])
        self.assertTrue(viewer["bbox_rectangles"])
        self.assertTrue(all(row["bbox_precision"] == "exact" for row in viewer["bbox_rectangles"]))
        self.assertEqual(viewer["authority_kind"], "metadata_source")
        self.assertFalse(viewer["citation_final"])

    def test_page_grounded_metadata_support_is_not_clickable(self) -> None:
        self.assertFalse(
            [row for row in read_jsonl(ROOT / "data/final/uud/metadata_grounding.jsonl") if row["bbox_precision"] == "page_grounded_only"]
        )
        result = self.service.ask("uud", "tanggal berlaku perubahan ketiga UUD")
        support = result["metadata_support"][0]
        self.assertEqual(support["support_class"], "exact_metadata_citation")
        self.assertTrue(support["citation_available"])
        self.assertTrue(support["viewer_highlightable"])
        self.assertEqual(support["authority_kind"], "metadata_source")
        self.assertFalse(support["citation_final"])

    def test_metadata_alias_preempts_generic_instrument_fallback_when_supported(self) -> None:
        result = self.service.ask("uud", "apa aturan mulai berlaku perubahan pertama")
        self.assertEqual(result["status"], "answer_ready")
        self.assertEqual(result["route"], "metadata_fact")
        self.assertEqual(result["intent"], "metadata_lookup")
        self.assertEqual(result["metadata_facts"][0]["field"], "effective_rule")
        self.assertEqual(result["metadata_facts"][0]["answer"], "dan mulai berlaku pada tanggal ditetapkan.")
        support = result["metadata_support"][0]
        self.assertEqual(support["authority_kind"], "metadata_source")
        self.assertFalse(support["citation_final"])
        self.assertFalse(result["citations"])
        self.assertFalse(result["viewer_refs"])

    def test_penandatangan_alias_routes_to_metadata_provenance_not_instrument(self) -> None:
        result = self.service.ask("uud", "penandatangan perubahan pertama UUD")
        self.assertEqual(result["status"], "answer_ready")
        self.assertEqual(result["route"], "metadata_fact")
        self.assertEqual(result["intent"], "metadata_lookup")
        self.assertEqual(result["metadata_facts"][0]["field"], "signatories")
        self.assertIn("Prof. Dr. H.M. Amien Rais", result["metadata_facts"][0]["answer"])
        support = result["metadata_support"][0]
        self.assertEqual(support["authority_kind"], "metadata_source")
        self.assertFalse(support["citation_final"])
        self.assertFalse(result["citations"])
        self.assertFalse(result["viewer_refs"])

    def test_signatory_name_uses_individual_exact_grounding(self) -> None:
        ambiguous = self.service.ask("uud", "Amien Rais")
        self.assertEqual(ambiguous["status"], "answer_ready")
        self.assertFalse(ambiguous["citations"])
        self.assertGreaterEqual(len(ambiguous["metadata_support"]), 2)
        self.assertIn("tercantum sebagai Ketua", ambiguous["answer"])
        self.assertIn("Perubahan Keempat", ambiguous["answer"])
        result = self.service.ask("uud", "Amien Rais Perubahan Pertama UUD")
        self.assertEqual(result["status"], "answer_ready")
        self.assertFalse(result["citations"])
        self.assertEqual(len(result["metadata_support"]), 1)
        citation = result["metadata_support"][0]
        self.assertEqual(citation["source_role"], "amendment_1_historical")
        viewer = self.service.viewer("uud", citation["evidence_id"])
        self.assertEqual([box["text"] for box in viewer["bbox_rectangles"]], ["Ketua,", "Prof. Dr. H.M. Amien Rais"])

    def test_scoped_person_role_projects_only_the_exact_name(self) -> None:
        result = self.service.ask("uud", "ketua Majelis Permusyawaratan Rakyat Republik Indonesia UUD amandemen pertama")
        self.assertEqual(result["status"], "answer_ready")
        self.assertIn("Prof. Dr. H.M. Amien Rais", result["answer"])
        self.assertIn("Historis", result["answer"])
        self.assertEqual(result["metadata_support"][0]["printed_role"], "Ketua")

    def test_unscoped_metadata_fails_closed_without_combined_citations(self) -> None:
        for query in ("penandatangan UUD", "kapan UUD ditetapkan"):
            result = self.service.ask("uud", query)
            self.assertEqual(result["status"], "insufficient_evidence", query)
            self.assertEqual(result["route"], "metadata_fact", query)
            self.assertEqual(result["answer_scope"], "insufficient_evidence", query)
            self.assertEqual(result["reason_code"], "ambiguous_source_scope", query)
            self.assertFalse(result["citations"], query)
            self.assertFalse(result["viewer_refs"], query)
            self.assertFalse(result["metadata_facts"], query)

    def test_ambiguous_queries_fail_closed_without_prompt_templates(self) -> None:
        service = LegalRuntimeService(ROOT, answer_provider=None, planning_provider=None)
        cases = (
            ("Pasal 7 atau Pasal 7A", "legal_reference", "ambiguous_legal_target"),
            ("perubahan keempat mengubah atau menghapus Pasal 16", "document_relation", "ambiguous_target"),
            ("Presiden atau DPR", "lexical_fallback", "ambiguous_target"),
            ("pendidikan", "lexical_fallback", "ambiguous_concept"),
            ("hubungan Pasal 16", "legal_relation", "relation_not_found"),
        )
        for query, route, reason in cases:
            with self.subTest(query=query):
                result = service.ask("uud", query)
                self.assertEqual(result["status"], "insufficient_evidence")
                self.assertEqual(result["route"], route)
                self.assertEqual(result.get("reason_code") or result.get("reason"), reason)
                self.assertNotIn("clarification_options", result)
                public = handle_request("uud", "ask", {"query": query}, service=service)
                self.assertEqual(public["status"], "insufficient_evidence")
                self.assertFalse(public.get("supports"))

    def test_noisy_and_out_of_corpus_queries_keep_typed_failures(self) -> None:
        for query in ("berapa lama presiden menjabat", "apa hukuman pidana korupsi"):
            with self.subTest(query=query):
                result = self.service.ask("uud", query)
                self.assertNotIn(result.get("reason_code"), {"ambiguous_concept", "ambiguous_target"})

    def test_inflected_metadata_wording_requires_explicit_source_scope(self) -> None:
        for query in ("siapa yang menandatangani UUD", "siapa yang menandatangi UUD"):
            result = self.service.ask("uud", query)
            self.assertEqual(result["status"], "insufficient_evidence", query)
            self.assertEqual(result["route"], "metadata_fact", query)
            self.assertEqual(result["reason_code"], "ambiguous_source_scope", query)
            self.assertFalse(result["citations"], query)
            self.assertFalse(result["viewer_refs"], query)
            self.assertFalse(result["metadata_facts"], query)

    def test_unresolved_temporal_scope_never_uses_preferred_source(self) -> None:
        metadata = self.service.ask("uud", "tanggal ditetapkan perubahan ke-5 UUD")
        self.assertEqual(metadata["status"], "insufficient_evidence")
        self.assertEqual(metadata["reason"], "unresolved_source_scope")
        self.assertFalse(metadata["citations"])
        self.assertFalse(metadata["viewer_refs"])
        self.assertFalse(metadata["metadata_facts"])
        legal = self.service.ask("uud", "Pasal 31 perubahan ke-5")
        self.assertEqual(legal["status"], "insufficient_evidence")
        self.assertEqual(legal["reason"], "unresolved_source_scope")
        self.assertFalse(legal["citations"])
        self.assertFalse(legal["viewer_refs"])

    def test_gemini_provider_uses_secret_header_and_verified_context_only(self) -> None:
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def read(self):
                return b'{"candidates":[{"content":{"parts":[{"text":"{\\"sentences\\":[{\\"style\\":\\"grounded\\",\\"referenced_fact_ids\\":[\\"deterministic_answer\\"]}]}"}]}}]}'

        with patch("tjipto.runtime.gemini.urlopen", return_value=Response()) as request:
            provider = GeminiAnswerProvider(
                "test-secret",
                model="test-model",
                endpoint="https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            )
            answer = provider.propose("Jawaban deterministik.")
        self.assertEqual(
            answer,
            {"sentences": ({"style": "grounded", "referenced_fact_ids": ("deterministic_answer",)},)},
        )
        payload = request.call_args.args[0].data.decode("utf-8")
        self.assertIn("Jawaban deterministik.", payload)
        self.assertIn("responseMimeType", payload)
        self.assertIn("responseSchema", payload)
        self.assertNotIn("test-secret", payload)

    def test_openai_compatible_provider_uses_generic_configuration_contract(self) -> None:
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def read(self):
                return b'{"choices":[{"message":{"content":"{\\"sentences\\":[{\\"style\\":\\"direct\\",\\"referenced_fact_ids\\":[\\"deterministic_answer\\"]}]}"}}]}'

        with patch("tjipto.runtime.openai_compatible.urlopen", return_value=Response()) as request:
            provider = OpenAICompatibleWordingProvider("test-secret", model="test-model", endpoint="https://example.invalid/v1/chat/completions")
            self.assertEqual(provider.propose("Jawaban deterministik."), {"sentences": ({"style": "direct", "referenced_fact_ids": ("deterministic_answer",)},)})
        http_request = request.call_args.args[0]
        payload = json.loads(http_request.data.decode("utf-8"))
        self.assertEqual(payload["response_format"]["type"], "json_schema")
        self.assertTrue(payload["response_format"]["json_schema"]["strict"])
        self.assertEqual(http_request.get_header("Authorization"), "Bearer test-secret")
        self.assertEqual(http_request.get_header("User-agent"), "Tjipto")
        self.assertNotIn("test-secret", http_request.data.decode("utf-8"))

    def test_wording_adapter_timeout_or_network_error_has_no_proposal(self) -> None:
        provider = OpenAICompatibleWordingProvider("test-secret", model="test-model", endpoint="https://example.invalid/v1/chat/completions")
        with patch("tjipto.runtime.openai_compatible.urlopen", side_effect=OSError("network unavailable")):
            self.assertIsNone(provider.propose("Jawaban deterministik."))

    def test_original_metadata_role_does_not_fall_back_to_amendments(self) -> None:
        for query in ("kapan UUD asli ditetapkan", "kapan naskah asli ditetapkan"):
            result = self.service.ask("uud", query)
            self.assertEqual(result["status"], "insufficient_evidence", query)
            self.assertEqual(result["route"], "metadata_fact", query)
            self.assertFalse(result["metadata_facts"], query)
            self.assertNotIn("19 Oktober 1999", result["answer"], query)
            self.assertNotIn("10 Agustus 2002", result["answer"], query)

        first = self.service.ask("uud", "kapan perubahan pertama ditetapkan")
        self.assertEqual(first["route"], "metadata_fact")
        self.assertEqual(first["metadata_facts"][0]["answer"], "19 Oktober 1999")

        fourth = self.service.ask("uud", "kapan perubahan keempat ditetapkan")
        self.assertEqual(fourth["route"], "metadata_fact")
        self.assertEqual(fourth["metadata_facts"][0]["answer"], "10 Agustus 2002")

    def test_document_level_amendment_relations_use_not_promoted_trace(self) -> None:
        for query in (
            "UUD 1945 diubah oleh amandemen berapa",
            "UUD ini dirubah oleh amandemen berapa",
            "perubahan apa saja yang mengubah UUD 1945",
        ):
            result = self.service.ask("uud", query)
            self.assertEqual(result["status"], "limited_answer", query)
            self.assertEqual(result["route"], "document_relation", query)
            self.assertEqual(result["intent"], "document_amendment_relation", query)
            self.assertFalse(result["evidence"], query)
            self.assertFalse(result["citations"], query)
            self.assertFalse(result["viewer_refs"], query)
            self.assertTrue(result["trace_support"], query)
            self.assertEqual(len(result["document_relations"]), 4, query)
            self.assertEqual({row["relation_type"] for row in result["document_relations"]}, {"AMENDED_BY"}, query)
            self.assertTrue(all(row["highlightable"] is False for row in result["document_relations"]), query)
            self.assertIn("Perubahan Pertama", result["answer"], query)
            self.assertIn("Perubahan Keempat", result["answer"], query)

        for query in ("amandemen pertama mengubah apa", "perubahan pertama mengubah apa"):
            result = self.service.ask("uud", query)
            self.assertEqual(result["status"], "limited_answer", query)
            self.assertEqual(result["route"], "document_relation", query)
            self.assertEqual(result["document_relations"][0]["relation_type"], "AMENDS", query)
            self.assertEqual(result["document_relations"][0]["source_role"], "amendment_1_historical", query)
            self.assertFalse(result["viewer_refs"], query)

    def test_article_level_amendment_relations_use_exact_artifact_only(self) -> None:
        pasal = self.service.ask("uud", "amandemen keempat mengubah pasal apa saja")
        self.assertEqual(pasal["status"], "answer_ready")
        self.assertEqual(pasal["route"], "document_relation")
        self.assertEqual(pasal["answer_type"], "article_amendment_relation")
        self.assertTrue(pasal["evidence"])
        self.assertFalse(pasal["historical_citations"])
        self.assertTrue(pasal["citations"])
        self.assertTrue(pasal["viewer_refs"])
        self.assertTrue(pasal["article_amendment_relations"])
        self.assertFalse(pasal["trace_support"])
        self.assertEqual(
            {row["relation_type"] for row in pasal["article_amendment_relations"]},
            {"MODIFIES", "ADDS"},
        )
        self.assertEqual(
            {row["support_class"] for row in pasal["article_amendment_relations"]},
            {"exact_article_relation"},
        )
        self.assertTrue(pasal["context_pack"]["viewer_refs"])
        self.assertTrue(pasal["context_pack"]["citation_payloads"])
        self.assertFalse(pasal["context_pack"]["historical_citations"])

        complete = self.service.ask("uud", "perubahan keempat mengubah pasal 16?")
        self.assertEqual(complete["status"], "answer_ready")
        self.assertEqual(
            {row["relation_type"] for row in complete["article_amendment_relations"]},
            {"MODIFIES"},
        )
        self.assertTrue(complete["evidence"])
        self.assertFalse(complete["historical_citations"])
        self.assertTrue(complete["citations"])
        self.assertTrue(complete["viewer_refs"])
        self.assertFalse(complete["trace_support"])

        for query in (
            "perubahan keempat menghapus pasal 16?",
            "perubahan keempat mencabut pasal 16?",
            "penghapusan Pasal 16 oleh perubahan keempat",
            "pasal 16 dicabut perubahan keempat?",
            "pencabutan pasal 16 oleh perubahan keempat",
            "pasal 16 dihapus oleh amandemen keempat?",
            "penghapusan pasal 16",
        ):
            result = self.service.ask("uud", query)
            self.assertEqual(result["status"], "answer_ready", query)
            self.assertEqual(result["route"], "document_relation", query)
            self.assertEqual(result["answer_scope"], "exact_article_relation", query)
            self.assertEqual({row["relation_type"] for row in result["article_amendment_relations"]}, {"DELETES"}, query)
            self.assertEqual({row["target_citation"] for row in result["article_amendment_relations"]}, {"Pasal 16"}, query)
            self.assertTrue(result["citations"], query)
            self.assertFalse(result["historical_citations"], query)
            self.assertTrue(result["citations"][0]["citation_final"], query)
            self.assertEqual(result["citations"][0]["authority_kind"], "instrument_provenance", query)

        for query in ("perubahan keempat menambahkan apa", "perubahan keempat menambahkan lembaga apa"):
            result = self.service.ask("uud", query)
            self.assertEqual(result["status"], "answer_ready", query)
            self.assertEqual(result["route"], "document_relation", query)
            self.assertEqual(result["answer_scope"], "exact_article_relation", query)
            self.assertTrue(result["evidence"], query)
            self.assertTrue(result["citations"], query)
            self.assertTrue(result["viewer_refs"], query)
            self.assertEqual({row["relation_type"] for row in result["article_amendment_relations"]}, {"ADDS"}, query)

    def test_stage7_semantic_sufficiency_is_operation_and_scope_aware(self) -> None:
        lawmaking = self.service.ask("uud", "Apa hubungan Presiden dan DPR dalam pembentukan undang-undang?")
        self.assertEqual(lawmaking["status"], "answer_ready")
        self.assertEqual(lawmaking["sufficiency"]["status"], "complete")
        self.assertEqual(
            set(lawmaking["sufficiency"]["fulfilled_requirement_ids"]),
            {"relation_1", "relation_2"},
        )
        self.assertTrue(any("dibahas" in str(row.get("quoted_text") or "").casefold() for row in lawmaking["evidence"]))
        self.assertFalse(any("7b" in str(row.get("citation") or "").casefold() for row in lawmaking["evidence"]))

        procedure = self.service.ask("uud", "Bagaimana perubahan UUD dilakukan?")
        self.assertEqual(procedure["status"], "answer_ready")
        self.assertEqual(
            {row["citation"] for row in procedure["evidence"]},
            {"(1)", "(2)", "(3)", "(4)"},
        )
        self.assertNotIn("(5)", {row["citation"] for row in procedure["evidence"]})

    def test_explicit_multi_reference_and_exhaustive_scope_are_not_false_complete(self) -> None:
        explicit = self.service.ask("uud", "Pasal 17 ayat (2) dan (3) diubah pada amandemen ke berapa?")
        self.assertEqual(explicit["status"], "answer_ready")
        self.assertEqual(
            {row["target_citation"] for row in explicit["article_amendment_relations"]},
            {"Pasal 17 ayat (2)", "Pasal 17 ayat (3)"},
        )
        self.assertFalse(explicit["trace_support"])

        exhaustive = self.service.ask("uud", "pasal apa saja yang diubah perubahan pertama")
        self.assertEqual(exhaustive["status"], "limited_answer")
        self.assertTrue(exhaustive["trace_support"])
        self.assertIn("ayat (3)", exhaustive["answer"])

    def test_natural_sentence_proposals_remain_fact_bound(self) -> None:
        from tjipto.runtime.service import _render_wording
        from tjipto.runtime.wording import build_answer_fact_plan, build_verified_claim_set

        facts = {"fact": "Pasal 31: Hak atas pendidikan."}
        accepted = _render_wording(
            {"sentences": ({"style": "direct", "referenced_fact_ids": ("fact",)},)},
            "fallback",
            facts,
        )
        self.assertEqual(accepted, facts["fact"])
        self.assertEqual(
            _render_wording(
                {"sentences": ({"style": "direct", "referenced_fact_ids": ("fact", "unknown")},)},
                "fallback",
                facts,
            ),
            "fallback",
        )
        for mutated in (
            "DPR mengatur Presiden.",
            "Pasal 31 tidak menjamin pendidikan.",
            "Pasal 32 wajib dibaca.",
            "Pasal 31 berlaku pada 2020.",
            "Pasal 31 berasal dari naskah historis.",
        ):
            with self.subTest(mutated=mutated):
                self.assertEqual(
                    _render_wording(
                        {"sentences": ({"text": mutated, "referenced_fact_ids": ("fact",)},)},
                        "fallback",
                        facts,
                    ),
                    "fallback",
                )
        self.assertEqual(
            _render_wording(
                {"sentences": ({"style": "unknown", "referenced_fact_ids": ("fact",)},)},
                "fallback",
                facts,
            ),
            "fallback",
        )
        plan = build_answer_fact_plan(
            ({
                "evidence_id": "support-id",
                "quoted_text": "Pasal 31 mengatur pendidikan.",
                "citation": "Pasal 31",
                "source_role": "current_consolidated",
                "temporal_context": "current_consolidated",
            },),
            "fallback",
        )
        self.assertEqual(plan.facts[1].support_ids, ("support-id",))
        self.assertEqual(plan.facts[1].legal_references, ("Pasal 31",))
        self.assertEqual(plan.facts[1].source_role, "current_consolidated")
        self.assertEqual(plan.facts[1].temporal_scope, "current_consolidated")
        self.assertEqual(plan.public()[1]["object"], "Pasal 31 mengatur pendidikan.")
        structural = build_answer_fact_plan(({
            "evidence_id": "bab-i",
            "authority_kind": "structural_context",
            "citation": "BAB I",
            "quoted_text": "BAB I\nBENTUK DAN KEDAULATAN\nPasal 1\nNegara Indonesia ialah Negara Kesatuan.",
        },), "fallback")
        self.assertEqual(structural.facts[1].object, "BAB I BENTUK DAN KEDAULATAN")
        claims = build_verified_claim_set((
            {
                "evidence_id": "support-id",
                "quoted_text": "Pasal 31 mengatur pendidikan.",
                "citation": "Pasal 31",
                "source_role": "current_consolidated",
                "temporal_context": "current_consolidated",
            },
        ), scope_terms={"historical": ("historis",)})
        self.assertEqual(
            _render_wording(
                {"sentences": ({"text": "Pasal 31 mengatur pendidikan.", "claim_ids": ["support:support-id"]},)},
                "fallback",
                verified_claims=claims,
            ),
            "Pasal 31 mengatur pendidikan. [[support:support-id]]",
        )
        self.assertEqual(
            _render_wording(
                {"sentences": ({"text": "Pasal 31 \ufffd mengatur pendidikan.", "claim_ids": ["support:support-id"]},)},
                "fallback",
                verified_claims=claims,
            ),
            "Pasal 31 \u2014 mengatur pendidikan. [[support:support-id]]",
        )
        shortened = build_verified_claim_set(({
            "evidence_id": "long-support",
            "quoted_text": "Pasal 31 mengatur pendidikan bagi warga negara.",
            "citation": "Pasal 31",
            "source_role": "current_consolidated",
        },))
        self.assertEqual(
            _render_wording(
                {"sentences": ({"text": "Pasal 31 mengatur pendidikan.", "claim_ids": ["support:long-support"]},)},
                "fallback",
                verified_claims=shortened,
            ),
            "Pasal 31 mengatur pendidikan. [[support:long-support]]",
        )
        natural = build_verified_claim_set(({
            "evidence_id": "pasal-28",
            "quoted_text": "Kemerdekaan berserikat dan berkumpul, mengeluarkan pikiran dengan lisan dan tulisan ditetapkan dengan undang-undang.",
            "citation": "Pasal 28",
            "source_role": "current_consolidated",
        },))
        self.assertEqual(
            _render_wording(
                {"sentences": ({
                    "text": "Pasal 28 menjamin kebebasan berserikat, berkumpul, dan menyampaikan pikiran secara lisan maupun tulisan.",
                    "claim_ids": ["support:pasal-28"],
                },)},
                "fallback",
                verified_claims=natural,
            ),
            "Pasal 28 menjamin kebebasan berserikat, berkumpul, dan menyampaikan pikiran secara lisan maupun tulisan. [[support:pasal-28]]",
        )
        self.assertEqual(
            _render_wording(
                {"sentences": ({
                    "text": "Ketentuan ini mencakup kebebasan berkumpul.",
                    "claim_ids": ["support:pasal-28"],
                },)},
                "fallback",
                verified_claims=natural,
            ),
            "Ketentuan ini mencakup kebebasan berkumpul. [[support:pasal-28]]",
        )
        self.assertEqual(
            _render_wording(
                {"sentences": (
                    {"text": "Kemerdekaan berserikat dijamin.", "claim_ids": ["support:pasal-28"]},
                    {"text": "Kemerdekaan berkumpul juga dijamin.", "claim_ids": ["support:pasal-28"]},
                )},
                "fallback",
                verified_claims=natural,
            ),
            "Kemerdekaan berserikat dijamin. [[support:pasal-28]]\n\n"
            "Kemerdekaan berkumpul juga dijamin. [[support:pasal-28]]",
        )
        self.assertEqual(
            _render_wording(
                {"sentences": ({
                    "text": "Pendidikan merupakan kebijakan penting.",
                    "claim_ids": ["support:pasal-28"],
                },)},
                "fallback",
                verified_claims=natural,
            ),
            "fallback",
        )
        enumerated = build_verified_claim_set(({
            "evidence_id": "scope-support",
            "quoted_text": "Perubahan ini mengubah Pasal 5 dan Pasal 7.",
            "citation": "Ruang lingkup perubahan",
            "source_role": "amendment_1_historical",
        },))
        self.assertEqual(
            _render_wording(
                {"sentences": ({
                    "text": "Perubahan ini mengubah Pasal 5.",
                    "claim_ids": ["support:scope-support"],
                },)},
                "fallback",
                verified_claims=enumerated,
                require_complete_enumerations=True,
            ),
            "fallback",
        )
        self.assertEqual(
            _render_wording(
                {"sentences": (
                    {"text": "Pasal 28 menetapkan pidana 99 tahun.", "claim_ids": ["support:pasal-28"]},
                    {
                        "text": "Pasal 28 menjamin kebebasan berserikat, berkumpul, dan menyampaikan pikiran secara lisan maupun tulisan.",
                        "claim_ids": ["support:pasal-28"],
                    },
                )},
                "fallback",
                verified_claims=natural,
            ),
            "Pasal 28 menjamin kebebasan berserikat, berkumpul, dan menyampaikan pikiran secara lisan maupun tulisan. [[support:pasal-28]]",
        )
        for mutated in (
            "Pasal 32 mengatur pendidikan.",
            "Pasal 31 tidak mengatur pendidikan.",
            "Pasal 31 mengatur pendidikan pada tahun 2020.",
            "Pasal 31 mengatur pendidikan dalam naskah historis.",
        ):
            with self.subTest(verified_mutated=mutated):
                self.assertEqual(
                    _render_wording(
                        {"sentences": ({"text": mutated, "claim_ids": ["support:support-id"]},)},
                        "fallback",
                        verified_claims=claims,
                    ),
                    "fallback",
                )

        structured = build_verified_claim_set((
            {
                "evidence_id": "structured-support",
                "quoted_text": "Konstitusi menjamin hak atas pendidikan.",
                "subject": "Konstitusi",
                "predicate": "menjamin",
                "object": "hak atas pendidikan",
                "citation": "Pasal 31",
                "source_role": "current_consolidated",
                "temporal_context": "current_consolidated",
            },
        ))
        self.assertEqual(
            _render_wording(
                {"sentences": ({"text": "Konstitusi menjamin hak atas pendidikan menurut Pasal 31.", "claim_ids": ["support:structured-support"]},)},
                "fallback",
                verified_claims=structured,
            ),
            "Konstitusi menjamin hak atas pendidikan menurut Pasal 31. [[support:structured-support]]",
        )

    def test_target_specific_article_amendment_relations_do_not_substitute_neighbors(self) -> None:
        unsupported = self.service.ask("uud", "amandemen keempat mengubah pasal 31?")
        self.assertEqual(unsupported["status"], "answer_ready")
        self.assertEqual(unsupported["route"], "document_relation")
        self.assertTrue(unsupported["evidence"])
        self.assertFalse(unsupported["historical_citations"])
        self.assertTrue(unsupported["citations"])
        self.assertTrue(unsupported["viewer_refs"])
        self.assertEqual(
            {row["target_citation"] for row in unsupported["article_amendment_relations"]},
            {
                "Pasal 31 ayat (1)",
                "Pasal 31 ayat (2)",
                "Pasal 31 ayat (3)",
                "Pasal 31 ayat (4)",
                "Pasal 31 ayat (5)",
            },
        )
        self.assertFalse(unsupported["trace_support"])

        exact = self.service.ask("uud", "perubahan keempat mengubah pasal 16?")
        self.assertEqual(exact["status"], "answer_ready")
        self.assertEqual(
            {row["relation_type"] for row in exact["article_amendment_relations"]},
            {"MODIFIES"},
        )
        self.assertFalse(exact["trace_support"])

        partial = self.service.ask("uud", "pasal yang diubah perubahan keempat")
        self.assertIn(partial["status"], {"answer_ready", "limited_answer"})
        if partial["status"] == "limited_answer":
            self.assertEqual(partial["answer_scope"], "trace_article_relation")
            self.assertTrue(partial["trace_support"])
        else:
            self.assertEqual(partial["answer_scope"], "exact_article_relation")
            self.assertFalse(partial["trace_support"])
        self.assertFalse(partial["historical_citations"])
        if partial["status"] == "limited_answer":
            self.assertFalse(partial["citations"])
            self.assertFalse(partial["viewer_refs"])
        else:
            self.assertTrue(partial["citations"])
            self.assertTrue(partial["viewer_refs"])
        self.assertTrue(partial["article_amendment_relations"])

        reverse = self.service.ask("uud", "pasal 31 diubah oleh amandemen berapa?")
        self.assertNotEqual({row["target_citation"] for row in reverse.get("article_amendment_relations", ())}, {"Pasal 16"})
        self.assertNotIn("Pasal 16", reverse.get("answer", ""))
        if reverse["status"] == "answer_ready":
            self.assertEqual(
                {row["target_citation"] for row in reverse["article_amendment_relations"]},
                {
                    "Pasal 31 ayat (1)",
                    "Pasal 31 ayat (2)",
                    "Pasal 31 ayat (3)",
                    "Pasal 31 ayat (4)",
                    "Pasal 31 ayat (5)",
                },
            )
        else:
            self.assertEqual(reverse["route"], "document_relation")
            self.assertFalse(reverse["citations"])
            self.assertFalse(reverse["viewer_refs"])

    def test_renames_route_preserves_paragraph_mapping_and_anomaly_precedence(self) -> None:
        exact = self.service.ask("uud", "Pasal 25E menjadi Pasal 25A")
        self.assertEqual(exact["route"], "document_relation")
        self.assertEqual(exact["status"], "answer_ready")
        self.assertEqual(
            [(row["source_reference"], row["target_reference"]) for row in exact["article_amendment_relations"]],
            [("Pasal 25E", "Pasal 25A")],
        )
        self.assertEqual(exact["article_amendment_relations"][0]["source_legal_unit_id"], "uud_legal_unit_00428")
        self.assertFalse(exact["historical_citations"])
        self.assertEqual(len(exact["citations"]), 1)
        self.assertTrue(exact["citations"][0]["citation_final"])
        self.assertTrue(exact["viewer_refs"])
        self.assertIn("dinomori ulang", exact["answer"].casefold())

        paragraph = self.service.ask("uud", "Pasal 3 ayat (3) menjadi Pasal 3 ayat (2)")
        self.assertEqual(paragraph["route"], "document_relation")
        self.assertEqual(paragraph["status"], "answer_ready")
        self.assertEqual(
            [(row["source_reference"], row["target_reference"]) for row in paragraph["article_amendment_relations"]],
            [("Pasal 3 ayat (3)", "Pasal 3 ayat (2)")],
        )
        self.assertEqual(paragraph["article_amendment_relations"][0]["source_legal_unit_id"], "uud_legal_unit_00485")
        self.assertEqual(paragraph["article_amendment_relations"][0]["target_legal_unit_id"], "uud_legal_unit_00014")
        self.assertEqual(paragraph["article_amendment_relations"][0]["source_reference_range_kind"], "literal")
        self.assertTrue(all(row["viewer_highlightable"] for row in paragraph["article_amendment_relations"]))
        self.assertIn("dinomori ulang", paragraph["answer"].casefold())

        paraphrase = self.service.ask("uud", "perubahan keempat mengubah penomoran pasal apa")
        self.assertEqual(paraphrase["route"], "document_relation")
        self.assertEqual(paraphrase["intent"], "document_amendment_relation")
        self.assertEqual(
            {row["relation_type"] for row in paraphrase["article_amendment_relations"]}, {"RENAMES", "RENUMBERED_TO"}
        )
        self.assertEqual(len(paraphrase["article_amendment_relations"]), 3)

        anomaly = self.service.ask("uud", "Pasal 25E menjadi Pasal 25A")
        self.assertEqual(anomaly["route"], "document_relation")
        self.assertEqual({row["relation_type"] for row in anomaly["article_amendment_relations"]}, {"RENUMBERED_TO"})

        public = handle_request("uud", "ask", {"query": "Pasal 25E menjadi Pasal 25A"}, service=self.service)
        self.assertNotIn("article_amendment_relations", public)
        support = public["supports"][0]
        self.assertEqual(support["support_kind"], "article_relation")
        self.assertEqual(support["authority_kind"], "instrument_provenance")
        self.assertTrue(support["citation_final"])
        self.assertNotEqual(support["source_label"], "uud::amendment_4_historical")
        self.assertTrue(support["viewer_target"]["can_resolve"])
        self.assertNotIn("evidence_id", support["viewer_target"])

    def test_current_and_historical_reference_routing_is_source_safe(self) -> None:
        current = self.service.ask("uud", "Aturan Tambahan Pasal II")
        self.assertEqual(current["status"], "answer_ready")
        self.assertEqual(len(current["citations"]), 1)
        self.assertEqual(current["citations"][0]["source_role"], "current_consolidated")
        self.assertFalse(current.get("source_conflict"))

        historical = self.service.ask("uud", "Pasal 25E")
        self.assertEqual(historical["status"], "insufficient_evidence")
        self.assertFalse(historical["citations"])

        typo = self.service.ask("uud", "Pasal III typo di UUD amandemen ke empat")
        self.assertEqual(typo["status"], "answer_ready")
        self.assertEqual(typo["citations"][0]["source_role"], "amendment_4_historical")
        self.assertEqual(typo["citations"][0]["hierarchy"], ("ATURAN TAMBAHAN", "Pasal II"))
        self.assertEqual(typo["trace_support"][0]["canonical_reference"], "Aturan Tambahan Pasal II")

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

        graph_edges = {row["edge_id"]: row for row in read_jsonl(ROOT / "data/final/uud/graph_edges.jsonl")}
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
            expected_status = "limited_answer" if result.get("trace_support") else case["status"]
            self.assertEqual(result["status"], expected_status, case["query"])
            self.assertEqual(result["route"], case["route"], case["query"])
            self.assertEqual(result["intent"], case["intent"], case["query"])
            if result["answer_type"] == "article_amendment_relation":
                self.assertTrue(result["article_amendment_relations"], case["query"])
                if result["evidence"]:
                    self.assertEqual(result["evidence"][0]["evidence_id"], case["evidence_id"], case["query"])
                    citations = result["citations"] or result["historical_citations"]
                    self.assertEqual(citations[0]["citation"], case["citation"], case["query"])
                else:
                    self.assertTrue(result["trace_support"], case["query"])
                continue
            self.assertEqual(result["evidence"][0]["candidate_type"], case["candidate_type"], case["query"])
            self.assertEqual(result["evidence"][0]["evidence_id"], case["evidence_id"], case["query"])
            self.assertFalse(result["citations"], case["query"])
            self.assertTrue(any(row["evidence_id"] == case["evidence_id"] for row in result["trace_support"]), case["query"])

    def test_ask_explains_known_source_anomalies_safely(self) -> None:
        for case in _source_conflict_cases():
            result = self.service.ask("uud", case["query"])
            self.assertEqual(result["status"], case["expected_status"], case["query"])
            self.assertEqual(result["route"], "source_anomaly_explanation", case["query"])
            self.assertEqual(result["source_conflict"]["source_conflict_id"], case["source_conflict_id"], case["query"])
            self.assertEqual(result["source_conflict"]["source_anomaly_kind"], case["source_anomaly_kind"], case["query"])
            self.assertFalse(result["citations"], case["query"])
            self.assertFalse(result["viewer_refs"], case["query"])
            self.assertGreaterEqual(len(result.get("trace_support", ())), case["trace_support_count"], case["query"])
            for reason in case["expected_insufficient_reasons"]:
                self.assertIn(reason, result["insufficient_reasons"], case["query"])
            self.assertIn("source_conflict_not_final_legal_authority", result["warnings"], case["query"])
            if result["historical_citations"]:
                expected_kind = (
                    "source_anomaly" if case["source_anomaly_kind"] == "source_marker_sequence_anomaly" else "source_conflict_provenance"
                )
                self.assertEqual(result["historical_citations"][0]["authority_kind"], expected_kind, case["query"])
                self.assertFalse(result["historical_citations"][0]["citation_final"], case["query"])
            if result["trace_support"]:
                self.assertFalse(result["trace_support"][0]["citation_final"], case["query"])

    def test_exact_legal_citation_remains_the_only_final_authority(self) -> None:
        result = self.service.ask("uud", "Pasal 1 ayat (3)")
        self.assertTrue(result["citations"])
        self.assertEqual(result["citations"][0]["authority_kind"], "legal_citation")
        self.assertTrue(result["citations"][0]["citation_final"])

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
        service = LegalRuntimeService(ROOT)
        asked = handle_request("uud", "ask", {"query": "Pasal 1 ayat (3)"}, service=service)
        exact = handle_request("uud", "viewer", {"target": asked["supports"][0]["viewer_target"]["public_target_id"]}, service=service)
        self.assertTrue(exact["bbox_rectangles"])
        self.assertEqual(exact["bbox_rectangles"][0]["bbox_precision"], "exact")
        self.assertTrue(exact["bbox_rectangles"][0]["viewer_highlightable"])
        self.assertNotIn("source_pdf_path", exact["bbox_rectangles"][0])
        self.assertNotIn("source_sha256", exact["bbox_rectangles"][0])
        self.assertNotIn("evidence_id", exact["bbox_rectangles"][0])
        self.assertNotIn("source_document_id", exact["bbox_rectangles"][0])

    def test_viewer_highlight_gate_requires_exact_bbox_rows(self) -> None:
        config = CorpusRegistry(ROOT).resolve("uud")
        store = EvidenceStore(config)
        evidence = store.get(_exact_highlightable_evidence_id())
        self.assertIsNotNone(evidence)
        synthetic_boxes = [
            row | {"bbox_precision": "coarse", "viewer_highlightable": True} for row in store.bboxes_for(evidence["evidence_id"])
        ]
        result = viewer_payload(store, "uud", evidence, synthetic_boxes)
        self.assertEqual(result["status"], "non_highlightable_trace")
        self.assertTrue(result["pdf_access_available"])
        self.assertTrue(result["rendering_available"])
        self.assertFalse(result["viewer_highlightable"])
        self.assertFalse(result["bbox_rectangles"])

        page_grounded = evidence | {
            "bbox_refs": (),
            "bbox_precision": "page_grounded_only",
            "viewer_highlightable": False,
        }
        result = viewer_payload(store, "uud", page_grounded, [])
        self.assertEqual(result["status"], "source_page_trace_only")
        self.assertTrue(result["pdf_access_available"])
        self.assertTrue(result["rendering_available"])
        self.assertFalse(result["viewer_highlightable"])
        self.assertFalse(result["bbox_rectangles"])

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
            if result["route"] == "lexical_fallback":
                self.assertIn("semantic_support_missing", result["insufficient_reasons"], case["query"])

    def test_criminal_punishment_queries_are_out_of_scope(self) -> None:
        for query in (
            "apa hukuman pidana korupsi?",
            "sanksi pidana korupsi",
            "ancaman pidana korupsi menurut UUD",
            "pidana korupsi",
            "tindak pidana korupsi",
            "apa sanksi tindak pidana berat?",
            "hukuman bagi koruptor",
            "denda korupsi",
            "denda korupsi presiden",
            "sanksi presiden korupsi",
            "apa pidana untuk tindak pidana berat presiden?",
            "apa sanksi untuk tindak pidana berat presiden?",
            "hukuman korupsi presiden",
            "ancaman hukuman korupsi",
            "ancaman pidana korupsi presiden",
            "apa pidana korupsi menurut Pasal 7A",
            "pidana korupsi presiden menurut Pasal 7A",
            "apa sanksi korupsi menurut pasal 7A?",
            "korupsi dalam pasal 7A hukuman apa",
            "penjara korupsi menurut pasal 7A",
            "Pasal 7A mengatur sanksi apa",
            "Pasal 7A pidana korupsi",
            "Pasal 7A menjatuhkan hukuman apa",
            "ancaman pidana menurut Pasal 7A",
            "hukuman korupsi Presiden menurut Pasal 7A",
            "pidana mati korupsi",
            "pemberantasan korupsi",
        ):
            result = self.service.ask("uud", query)
            self.assertEqual(result["status"], "insufficient_evidence", query)
            self.assertEqual(result["route"], "unsupported_scope", query)
            self.assertFalse(result["citations"], query)
            self.assertFalse(result["viewer_refs"], query)

    def test_pasal_7a_corruption_context_remains_answerable(self) -> None:
        for query in (
            "korupsi dalam Pasal 7A maksudnya apa?",
            "alasan Presiden dapat diberhentikan menurut Pasal 7A",
            "apakah Pasal 7A menyebut korupsi?",
            "tindak pidana berat dalam Pasal 7A maksudnya apa?",
        ):
            result = self.service.ask("uud", query)
            self.assertEqual(result["status"], "answer_ready", query)
            self.assertEqual(result["route"], "legal_reference", query)
            self.assertTrue(result["citations"], query)
            self.assertTrue(result["citations"][0]["citation_final"], query)
            self.assertEqual(result["citations"][0]["evidence_id"], "uud_current_consolidated_final_citation_evidence_00264", query)

    def test_president_three_terms_numeric_word_variants_require_one_complete_bm25_source(self) -> None:
        for query, status in (
            ("bolehkah presiden menjabat tiga periode", "insufficient_evidence"),
            ("boleh presiden 3 periode?", "limited_answer"),
            ("presiden boleh tiga periode?", "limited_answer"),
        ):
            result = self.service.ask("uud", query)
            self.assertEqual(result["status"], status, query)
            self.assertEqual(result["route"], "lexical_fallback", query)
            if status == "limited_answer":
                self.assertEqual([row["citation"] for row in result["citations"]], ["Pasal 7"], query)
            else:
                self.assertFalse(result["citations"], query)

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
            if "lexical_complete_coverage" in case:
                self.assertTrue(all(row["lexical_complete_coverage"] is case["lexical_complete_coverage"] for row in result["matches"]))

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
        self.assertEqual(removed["status"], "insufficient_evidence")
        self.assertEqual(removed["reason"], "structured_not_found")
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
        self.assertTrue(all(row["source_role"] == "amendment_1_historical" for row in historical["matches"]))

        historical_search = self.service.search(
            "uud",
            "Perubahan Pertama UUD",
            filters={"source_role": "amendment_1_historical"},
        )
        self.assertEqual(historical_search["status"], "found")
        self.assertTrue(all(row["source_role"] == "amendment_1_historical" for row in historical_search["results"]))

        temporal = self.service.search(
            "uud",
            "UUD 1945",
            limit=1,
            filters={"temporal_context": "amendment_1_historical"},
        )
        self.assertEqual(temporal["status"], "found")
        self.assertEqual(temporal["applied_filters"]["temporal_context"], "amendment_1_historical")
        self.assertEqual(len(temporal["results"]), 1)

    def test_temporal_target_prefers_current_or_requested_historical_source(self) -> None:
        current = self.service.ask("uud", "Apa isi BAB IV UUD 1945 saat ini?")
        self.assertEqual(current["status"], "answer_ready")
        self.assertEqual([row["quoted_text"] for row in current["citations"]], ["Dihapus."])
        self.assertEqual([row["citation"] for row in current["structural_support"]], ["BAB IV"])
        self.assertEqual(current["structural_support"][0]["source_role"], "current_consolidated")
        for query, role, citation in (
            ("Apa isi Pasal 21 pada Perubahan Pertama?", "amendment_1_historical", "Pasal 21"),
            ("Apa isi ayat (5) Pasal 30 pada Perubahan Kedua?", "amendment_2_historical", "(5)"),
            ("Apa isi Pasal 36C pada Perubahan Kedua?", "amendment_2_historical", "Pasal 36C"),
        ):
            result = self.service.ask("uud", query)
            self.assertEqual(result["status"], "answer_ready", query)
            self.assertTrue(result["citations"], query)
            self.assertEqual(result["citations"][0]["citation"], citation, query)
            self.assertTrue(all(row["source_role"] == role for row in result["citations"]), query)

    def test_current_consolidated_pasal_16_has_exact_public_citation(self) -> None:
        result = self.service.ask("uud", "Pasal 16 UUD konsolidasi")
        self.assertEqual(result["status"], "answer_ready")
        self.assertEqual({row["source_role"] for row in result["citations"]}, {"current_consolidated"})
        self.assertEqual([row["citation"] for row in result["citations"]], ["Pasal 16"])
        self.assertTrue(result["citations"][0]["viewer_ref"]["can_resolve"])
        self.assertTrue(result["citations"][0]["relevant_quote_eligible"])

    def test_source_span_support_resolves_exact_pdf_geometry(self) -> None:
        public = handle_request("uud", "ask", {"query": "Pasal 16 UUD konsolidasi"}, service=self.service)
        target = public["supports"][0]["viewer_target"]["public_target_id"]
        result = handle_request("uud", "viewer", {"target": target}, service=self.service)
        self.assertEqual(result["status"], "viewer_payload_ready")
        self.assertTrue(result["viewer_highlightable"])
        self.assertTrue(result["bbox_rectangles"])
        self.assertEqual(result["bbox_rectangles"][0]["bbox_precision"], "exact")

    def test_bab_detail_publishes_normative_children_as_relevant_support(self) -> None:
        result = self.service.ask("uud", "Apa isi BAB XI agama?")
        self.assertEqual(result["status"], "answer_ready")
        self.assertTrue(result["structural_support"])
        self.assertTrue(result["citations"])
        self.assertTrue(all(row["relevant_quote_eligible"] for row in result["citations"]))
        self.assertTrue(all(row["authority_kind"] == "legal_citation" for row in result["citations"]))

    def test_bab_request_granularity_preserves_the_requested_units(self) -> None:
        heading = self.service.ask("uud", "BAB XA", limit=1)
        title = self.service.ask("uud", "Apa judul BAB XA?", limit=1)
        content = self.service.ask("uud", "Apa isi BAB XA?", limit=1)
        mixed = self.service.ask("uud", "BAB XA dan Pasal 28B ayat (1)", limit=1)
        pasal = self.service.ask("uud", "Pasal 28B", limit=1)
        ayat = self.service.ask("uud", "Pasal 28B ayat (1)", limit=1)

        for result in (heading, title):
            self.assertEqual([row["citation"] for row in result["citations"]], ["BAB XA"])
            self.assertEqual(result["citations"][0]["quoted_text"].splitlines(), ["BAB XA", "HAK ASASI MANUSIA"])
        self.assertEqual([row["citation"] for row in content["structural_support"]], ["BAB XA"])
        self.assertEqual(
            [row["citation"] for row in content["citations"]],
            [f"Pasal 28{suffix}" for suffix in "ABCDEFGHIJ"],
        )
        self.assertEqual([row["citation"] for row in mixed["structural_support"]], ["BAB XA"])
        self.assertEqual([row["citation"] for row in mixed["citations"]], ["(1)"])
        self.assertEqual([row["citation"] for row in pasal["citations"]], ["Pasal 28B"])
        self.assertEqual([row["citation"] for row in ayat["citations"]], ["(1)"])
        self.assertFalse(any(row["legal_unit_id"] == content["structural_support"][0]["legal_unit_id"] for row in content["citations"]))

    def test_bab_heading_support_uses_only_heading_text_and_geometry(self) -> None:
        public = handle_request("uud", "ask", {"query": "BAB XA"}, service=self.service)
        support = public["supports"][0]
        self.assertEqual(support["authority_kind"], "legal_citation")
        self.assertTrue(support["citation_final"])
        self.assertEqual(support["text"].splitlines(), ["BAB XA", "HAK ASASI MANUSIA"])
        viewer = handle_request("uud", "viewer", {"target": support["viewer_target"]["public_target_id"]}, service=self.service)
        self.assertEqual(viewer["quoted_text"], support["text"])
        self.assertEqual(len(viewer["bbox_rectangles"]), 2)

    def test_bab_detail_does_not_publish_contained_legal_units_as_peer_quotes(self) -> None:
        result = self.service.ask("uud", "BAB XI agama")
        citation_ids = {row["legal_unit_id"] for row in result["citations"]}
        store = self.service._store("uud")
        self.assertIsNotNone(store)
        units = {row["legal_unit_id"]: row for row in store.legal_units}
        for unit_id in citation_ids:
            parent = units[unit_id].get("parent_legal_unit_id")
            self.assertNotIn(parent, citation_ids)

    def test_bab_deletion_query_uses_normative_deletion_evidence(self) -> None:
        result = self.service.ask("uud", "Apakah BAB IV dihapus?")
        self.assertEqual(result["status"], "answer_ready")
        self.assertTrue(result["citations"])
        self.assertEqual(result["citations"][0]["citation"], "Dihapus.")
        self.assertEqual(result["evidence"][0]["authority_kind"], "normative_legal_text")
        self.assertNotIn("BAB IV", result["citations"][0]["quoted_text"])

    def test_explicit_temporal_reference_never_becomes_document_relation_or_current_fallback(self) -> None:
        cases = (
            ("Pasal 1 naskah asli", "answer_ready", "original_historical"),
            ("Pasal 1 Perubahan Ketiga", "answer_ready", "amendment_3_historical"),
            ("Pasal 1 Amandemen Pertama", "insufficient_evidence", None),
        )
        for query, status, source_role in cases:
            result = self.service.ask("uud", query)
            self.assertEqual(result["status"], status, query)
            self.assertNotEqual(result["route"], "document_relation", query)
            if source_role:
                self.assertEqual({row["source_role"] for row in result["citations"]}, {source_role}, query)
            else:
                self.assertFalse(result["citations"], query)

    def test_only_explicit_document_navigation_opens_full_source(self) -> None:
        for query, role in (
            ("Buka naskah Perubahan Pertama UUD", "amendment_1_historical"),
            ("Tampilkan dokumen Perubahan Keempat UUD", "amendment_4_historical"),
            ("Lihat PDF naskah asli UUD", "original_historical"),
            ("Buka naskah satu naskah UUD 1945", "current_consolidated"),
            ("berikan dokumen perubahan pertama", "amendment_1_historical"),
            ("tampilkan naskah perubahan keempat", "amendment_4_historical"),
            ("berikan saya naskah UUD original", "original_historical"),
            ("berikan saya naskah UUD konsolidasi", "current_consolidated"),
            ("berikan UUD amandemen ke empat", "amendment_4_historical"),
            ("berikan UUD naskah sebelum amandemen", "original_historical"),
        ):
            result = self.service.ask("uud", query)
            self.assertEqual(result["status"], "answer_ready", query)
            self.assertEqual(result["route"], "source_document", query)
            self.assertEqual(result["answer_type"], "source_document", query)
            self.assertEqual(result["document_source"]["source_role"], role, query)
            self.assertEqual(result["intent"], "document_delivery", query)
            self.assertEqual(result["document_source"]["viewer_target"]["action"], "open_document", query)
            self.assertFalse(result["citations"], query)
            self.assertFalse(result["viewer_refs"], query)
        for query in ("Apa isi Perubahan Pertama UUD?", "ringkasan UUD amandemen pertama"):
            self.assertNotEqual(self.service.ask("uud", query).get("answer_type"), "source_document", query)

    def test_deterministic_legal_operations_keep_explicit_scope_and_evidence_rules(self) -> None:
        store = self.service._store("uud")
        semantics = {
            query: interpret_query(store, "uud", query)
            for query in (
                "berikan naskah UUD original",
                "ringkas amandemen keempat",
                "amandemen pertama vs kedua",
                "perbedaan penandatangan amandemen pertama dan kedua",
                "Pasal 16 sebelum dihapus bunyinya apa",
                "BAB setelah BAB IX",
                "legal opinion tentang HAM dari Pasal 28",
                "apa perbedaan Pasal 28 dan Pasal 28A",
                "apa perbedaan UUD setelah amandemen",
                "bandingkan Pasal 5 ayat (1) sebelum dan sesudah Perubahan Pertama",
                "lakukan legal research tentang kewenangan MPR mengubah UUD",
            )
        }
        self.assertEqual(semantics["berikan naskah UUD original"].operation, "open_document")
        self.assertEqual(semantics["berikan naskah UUD original"].source_scopes, ("original_historical",))
        self.assertEqual(semantics["ringkas amandemen keempat"].operation, "summarize")
        comparison = semantics["amandemen pertama vs kedua"]
        self.assertEqual(comparison.operation, "compare")
        self.assertEqual(comparison.source_scopes, ("amendment_1_historical", "amendment_2_historical"))
        self.assertTrue(comparison.requires_multiple_supports)
        metadata_comparison = semantics["perbedaan penandatangan amandemen pertama dan kedua"]
        self.assertEqual(metadata_comparison.targets, ("signatory_metadata",))
        historical = semantics["Pasal 16 sebelum dihapus bunyinya apa"]
        self.assertEqual(historical.temporal_scope, "historical_pre_change")
        self.assertTrue(historical.requires_multiple_supports)
        self.assertTrue(historical.requires_graph)
        navigation_semantics = semantics["BAB setelah BAB IX"]
        self.assertEqual(navigation_semantics.operation, "navigate")
        self.assertEqual(navigation_semantics.targets, ("BAB IX",))
        self.assertEqual(navigation_semantics.navigation_operation, "next")
        analysis_semantics = semantics["legal opinion tentang HAM dari Pasal 28"]
        self.assertEqual(analysis_semantics.operation, "analyze")
        self.assertTrue(analysis_semantics.requires_multiple_supports)
        self.assertTrue(analysis_semantics.requires_decomposition)
        self.assertEqual(semantics["apa perbedaan Pasal 28 dan Pasal 28A"].operation, "compare")
        self.assertEqual(semantics["apa perbedaan UUD setelah amandemen"].operation, "compare")
        temporal_comparison = semantics["bandingkan Pasal 5 ayat (1) sebelum dan sesudah Perubahan Pertama"]
        self.assertEqual(temporal_comparison.operation, "compare")
        self.assertEqual(temporal_comparison.source_scopes, ("original_historical", "amendment_1_historical"))
        self.assertEqual(semantics["lakukan legal research tentang kewenangan MPR mengubah UUD"].operation, "analyze")

        collection = self.service.ask("uud", "berikan saya naskah UUD")
        self.assertEqual(collection["route"], "source_document_collection")
        self.assertEqual(self.service.ask("uud", "berikan saya document")["route"], "source_document_collection")
        self.assertEqual(
            {row["source_role"] for row in collection["document_sources"]},
            {"original_historical", "amendment_1_historical", "amendment_2_historical", "amendment_3_historical", "amendment_4_historical", "current_consolidated"},
        )
        summary = self.service.ask("uud", "ringkas amandemen keempat")
        self.assertEqual(summary["route"], "instrument_resolved_answerable")
        self.assertEqual(summary["operation"], "summarize")
        compared = self.service.ask("uud", "amandemen pertama vs kedua", limit=30)
        self.assertEqual(compared["status"], "answer_ready")
        self.assertEqual(
            {row["source_role"] for row in compared["evidence"]},
            {"amendment_1_historical", "amendment_2_historical"},
        )
        self.assertEqual(
            {row["display_label"] for row in compared["evidence"]},
            {"Ruang lingkup Perubahan Pertama", "Ruang lingkup Perubahan Kedua"},
        )
        public_comparison = handle_request(
            "uud",
            "ask",
            {"query": "amandemen pertama vs kedua", "limit": 30},
            service=self.service,
        )
        self.assertEqual(public_comparison["operation"], "compare")
        self.assertEqual(public_comparison["sufficiency"]["status"], "complete")
        self.assertEqual(len(public_comparison["source_scopes"]), 2)
        self.assertTrue(all(set(scope) == {"label"} for scope in public_comparison["source_scopes"]))
        metadata = self.service.ask("uud", "perbedaan penandatangan amandemen pertama dan kedua")
        self.assertEqual(metadata["status"], "answer_ready")
        self.assertEqual(
            {row["source_role"] for row in metadata["metadata_support"]},
            {"amendment_1_historical", "amendment_2_historical"},
        )
        self.assertIn("Drs. Kwik Kian Gie", metadata["answer"])
        self.assertIn("Ir. Sutjipto", metadata["answer"])
        provision_comparison = self.service.ask("uud", "apa perbedaan Pasal 28 dan Pasal 28A", limit=30)
        self.assertEqual(provision_comparison["sufficiency"]["status"], "complete")
        self.assertEqual(len(provision_comparison["citations"]), 2)
        paragraph_comparison = self.service.ask(
            "uud", "bandingkan Pasal 5 ayat (1) sebelum dan sesudah Perubahan Pertama", limit=30
        )
        self.assertEqual(paragraph_comparison["sufficiency"]["status"], "complete")
        self.assertEqual(len(paragraph_comparison["citations"]), 2)
        historical_response = self.service.ask("uud", "Pasal 16 sebelum dihapus bunyinya apa")
        self.assertEqual(historical_response["status"], "answer_ready")
        self.assertEqual(historical_response["sufficiency"]["status"], "complete")
        self.assertEqual(
            set(historical_response["sufficiency"]["fulfilled_requirement_ids"]),
            {"historical_normative_text", "deletion_provenance"},
        )
        self.assertEqual(
            {row["source_role"] for row in historical_response["citations"]},
            {"original_historical", "amendment_4_historical"},
        )
        self.assertIn("Susunan Dewan Pertimbangan Agung", historical_response["answer"])
        navigation = self.service.ask("uud", "BAB setelah BAB IX")
        self.assertEqual(navigation["route"], "structural_navigation")
        self.assertIn("BAB IXA", navigation["answer"])
        analysis = self.service.ask("uud", "legal opinion tentang HAM dari Pasal 28")
        self.assertIsNotNone(analysis.get("research_plan"))
        self.assertTrue(analysis["citations"])
        legal_research = self.service.ask("uud", "lakukan legal research tentang kewenangan MPR mengubah UUD")
        self.assertEqual(legal_research["route"], "research")
        self.assertEqual(legal_research["sufficiency"]["status"], "complete")
        self.assertTrue(legal_research["citations"])

    def test_control_plane_resolves_source_roles_once_and_publication_reuses_semantics(self) -> None:
        store = self.service._store("uud")
        from tjipto.corpora import source_arbitration

        with patch.object(source_arbitration, "source_roles_for_query", wraps=source_arbitration.source_roles_for_query) as roles:
            semantics = interpret_query(store, "uud", "berikan naskah UUD original")
        self.assertEqual(roles.call_count, 1)
        self.assertEqual(semantics.source_scope_state, "explicit_resolved")
        self.assertEqual(semantics.operation_query, None)

        with patch("tjipto.runtime.answer_arbitration.resolve_source_scope", side_effect=AssertionError("duplicate source resolution")):
            opened = self.service.ask("uud", "berikan naskah UUD original")
            summarized = self.service.ask("uud", "ringkas amandemen keempat")
        self.assertEqual(opened["route"], "source_document")
        self.assertEqual(summarized["route"], "instrument_resolved_answerable")

    def test_unresolved_scoped_document_never_falls_back_to_consolidated(self) -> None:
        result = self.service.ask("uud", "Apa isi amandement pertama UUD?")
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(result["reason"], "unresolved_source_scope")
        self.assertFalse(result["citations"])
        self.assertFalse(result["viewer_refs"])

    def test_source_role_marker_does_not_create_document_relation(self) -> None:
        result = self.service.ask("uud", "Apa isi Perubahan Pertama UUD?")
        self.assertNotEqual(result["route"], "document_relation")

    def test_document_source_api_exposes_verified_document_target_only(self) -> None:
        result = handle_request("uud", "ask", {"query": "Buka naskah Perubahan Pertama UUD"}, service=self.service)
        self.assertEqual(result["kind"], "document")
        self.assertEqual(result["document"]["viewer_target"]["action"], "open_document")
        self.assertNotIn("supports", result)

        collection = handle_request("uud", "ask", {"query": "berikan saya naskah UUD"}, service=self.service)
        self.assertEqual(collection["kind"], "documents")
        self.assertEqual(len(collection["documents"]), 6)
        self.assertEqual({row["document_role"] for row in collection["documents"]}, {"Naskah Asli", "Amandemen", "Naskah Konsolidasi"})
        self.assertTrue(any("Perubahan Pertama" in row["title"] for row in collection["documents"]))
        self.assertTrue(any("Perubahan Keempat" in row["title"] for row in collection["documents"]))
        self.assertTrue(all(row["viewer_target"]["action"] == "open_document" for row in collection["documents"]))

    def test_two_artifact_declared_document_scopes_route_to_their_document_relation(self) -> None:
        result = self.service.ask("uud", "apakah perubahan kedua mengamandemen naskah asli")
        self.assertEqual(result["status"], "limited_answer")
        self.assertEqual(result["route"], "document_relation")
        self.assertEqual(result["intent"], "document_amendment_relation")
        self.assertFalse(result["citations"])

    def test_explicit_compound_targets_preserve_each_grounded_subanswer(self) -> None:
        service = LegalRuntimeService(ROOT, answer_provider=None, planning_provider=None)
        provisions = service.ask("uud", "berikan Pasal 28A dan Pasal 28J")
        self.assertEqual((provisions["status"], provisions["route"]), ("answer_ready", "compound"))
        self.assertEqual({row["citation"] for row in provisions["citations"]}, {"Pasal 28A", "Pasal 28J"})

        mixed = service.ask(
            "uud",
            "berikan Pasal 28A, Pasal 28J, ringkasan UUD amandemen pertama, dan ringkasan amandemen keempat",
        )
        self.assertEqual((mixed["status"], mixed["route"]), ("answer_ready", "compound"))
        self.assertEqual({row["citation"] for row in mixed["citations"]}, {"Pasal 28A", "Pasal 28J"})
        self.assertEqual(
            {row["source_role"] for row in mixed["historical_citations"]},
            {"amendment_1_historical", "amendment_4_historical"},
        )
        self.assertTrue(all(not row["citation_final"] for row in mixed["historical_citations"]))
        self.assertEqual({row["source_role"] for row in mixed["evidence"]}, {
            "current_consolidated",
            "amendment_1_historical",
            "amendment_4_historical",
        })

    def test_historical_summary_exposes_non_final_source_citation(self) -> None:
        service = LegalRuntimeService(ROOT, answer_provider=None, planning_provider=None)
        summary = service.ask("uud", "ringkas amandemen pertama")
        self.assertEqual((summary["status"], summary["route"]), ("answer_ready", "instrument_resolved_answerable"))
        self.assertFalse(summary["citations"])
        self.assertEqual(len(summary["historical_citations"]), 1)
        citation = summary["historical_citations"][0]
        self.assertEqual(citation["authority_kind"], "instrument_provenance")
        self.assertFalse(citation["citation_final"])
        self.assertTrue(citation["viewer_ref"]["can_resolve"])
        public = handle_request("uud", "ask", {"query": "ringkas amandemen pertama"}, service=service)
        self.assertEqual(public["status"], "answer_ready")
        self.assertIn("[1]", public["answer"])
        self.assertEqual(public["supports"][0]["authority_kind"], "instrument_provenance")
        self.assertFalse(public["supports"][0]["citation_final"])
        self.assertTrue(public["supports"][0]["viewer_target"]["can_resolve"])

    def test_source_less_article_change_query_uses_exact_relation_evidence(self) -> None:
        result = LegalRuntimeService(ROOT, answer_provider=None, planning_provider=None).ask(
            "uud", "Pasal 17 ayat (3) diubah oleh apa", limit=30
        )
        self.assertEqual((result["status"], result["route"]), ("answer_ready", "document_relation"))
        self.assertIn("Perubahan Pertama", result["answer"])
        self.assertEqual(result["article_amendment_relations"][0]["target_citation"], "Pasal 17 ayat (3)")
        self.assertTrue(result["relation_support"][0]["viewer_ref"]["can_resolve"])

    def test_source_document_count_is_corpus_derived(self) -> None:
        service = LegalRuntimeService(ROOT, answer_provider=None, planning_provider=None)
        for query in ("ada berapa kali perubahan UUD?", "berapa kali UUD diubah?", "berapa kali UUD diamandemen?"):
            result = service.ask("uud", query)
            self.assertEqual((result["status"], result["route"]), ("answer_ready", "structure_count"), query)
            self.assertEqual(result["evidence"][0]["structural_count"], 4, query)
            self.assertEqual(len(result["structural_support"]), 4, query)
            self.assertIn("Korpus terverifikasi", result["answer"], query)

        public = handle_request("uud", "ask", {"query": "berapa pasal di dokumen ini?"}, service=service)
        support = public["supports"][0]
        self.assertEqual(support["viewer_target"]["action"], "open_document")
        self.assertTrue(support["viewer_target"]["can_resolve"])
        viewer = handle_request(
            "uud", "viewer", {"target": support["viewer_target"]["public_target_id"]}, service=service
        )
        self.assertEqual(viewer["status"], "viewer_payload_ready")
        self.assertEqual(viewer["bbox_rectangles"], ())
        self.assertFalse(viewer["viewer_highlightable"])

    def test_original_historical_summary_uses_original_document_structure(self) -> None:
        result = LegalRuntimeService(ROOT, answer_provider=None, planning_provider=None).ask(
            "uud", "ringkas UUD sebelum amandemen", limit=30
        )
        self.assertEqual((result["status"], result["route"]), ("answer_ready", "structural_navigation"))
        self.assertEqual({row["source_role"] for row in result["structural_support"]}, {"original_historical"})
        self.assertIn("BAB I — BENTUK DAN KEDAULATAN", result["answer"])
        self.assertNotIn("BAB XA", result["answer"])

    def test_filter_conflicts_and_api_temporal_context(self) -> None:
        temporal = self.service.search("uud", "UUD 1945", limit=1, filters={"temporal_context": "amendment_1_historical"})
        self.assertEqual(temporal["status"], "found")
        self.assertEqual(temporal["results"][0]["temporal_context"], "amendment_1_historical")
        conflicting = self.service.search(
            "uud", "UUD 1945", filters={"source_role": "current_consolidated", "temporal_context": "amendment_1_historical"}
        )
        self.assertEqual(conflicting["status"], "invalid_filter")
        self.assertEqual(conflicting["reason"], "conflicting_filters")
        self.assertFalse(conflicting["matches"])
        invalid = self.service.search("uud", "UUD 1945", filters={"source_role": "not_a_source_role"})
        self.assertEqual(invalid["status"], "invalid_filter")
        self.assertEqual(invalid["reason"], "invalid_filter")
        self.assertEqual(invalid["invalid_filters"], ("source_role",))
        api_result = handle_request(
            "uud",
            "ask",
            {
                "query": "Pasal 5 ayat (1)",
                "filters": {"source_role": "amendment_1_historical"},
            },
            ROOT,
        )
        self.assertEqual(api_result["status"], "answer_ready")
        self.assertEqual(api_result["supports"][0]["source_status_label"], "Amandemen")

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
            self.assertEqual(report[key]["total"], len(read_jsonl(ROOT / "data/final/uud" / f"{key}.jsonl")))
            self.assertGreaterEqual(report[key]["raw_pdf_match"], 0)
            self.assertGreaterEqual(report[key]["normalized_pdf_match"], report[key]["raw_pdf_match"])
            self.assertGreaterEqual(report[key]["header_stripped_pdf_match"], report[key]["normalized_pdf_match"])
            self.assertLessEqual(report[key]["header_stripped_pdf_match"], report[key]["total"])
            self.assertEqual(report[key]["evidence_grounded_match"], len(read_jsonl(ROOT / "data/final/uud/evidence_registry.jsonl")))
            self.assertEqual(report[key]["needs_review"], 1)
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

        bab_xa = route_retrieval("uud", "BAB XA", store, limit=3)
        self.assertEqual(bab_xa["status"], "found")
        self.assertEqual(bab_xa["route"], "structured")
        self.assertEqual(bab_xa["intent"], "structured_lookup")
        self.assertTrue(bab_xa["matches"])
        self.assertTrue(all("BAB XA" in " / ".join(row["hierarchy"]) for row in bab_xa["matches"]))

        pasal_28a = route_retrieval("uud", "Pasal 28A", store, limit=1)
        self.assertEqual(pasal_28a["status"], "found")
        self.assertTrue("BAB XA" in " / ".join(pasal_28a["matches"][0]["hierarchy"]))

        pasal_28 = route_retrieval("uud", "Pasal 28", store, limit=1)
        self.assertEqual(pasal_28["status"], "found")
        self.assertTrue("BAB X" in " / ".join(pasal_28["matches"][0]["hierarchy"]))

        filtered = route_retrieval(
            "uud",
            "Pembukaan",
            store,
            metadata_filters={"source_role": "current_consolidated", "temporal_context": "current_consolidated"},
        )
        self.assertEqual(filtered["status"], "found")
        self.assertEqual(filtered["route"], "structured")
        self.assertTrue(all(row["source_role"] == "current_consolidated" for row in filtered["matches"]))

        class NoBBoxStore:
            config: Any
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
            config: Any
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
        self.assertEqual([row["evidence_id"] for row in ranked], ["e1", "e2"])
        self.assertEqual(ranked[0]["route_sources"], ("bm25", "structured"))
        self.assertIn("pass", ranked[0]["rank_reasons"])
        self.assertFalse(trace)

        empty, empty_trace = merge_ranked(store, {}, {"status": "final"})
        self.assertEqual(empty, ())
        self.assertEqual(empty_trace, ())

    def test_semantic_graph_expansion_requires_relation_provenance_and_preserves_source_role(self) -> None:
        class Store:
            evidence = [
                {"evidence_id": "source", "legal_unit_id": "source_unit", "status": "final", "source_role": "current_consolidated"},
                {"evidence_id": "target", "legal_unit_id": "target_unit", "status": "final", "source_role": "current_consolidated"},
                {"evidence_id": "historical", "legal_unit_id": "target_unit", "status": "final", "source_role": "amendment_1_historical"},
            ]
            graph_edges = [
                {"edge_type": "HAS_FINAL_EVIDENCE", "source_id": "legal_unit::source_unit", "target_id": "final_evidence::source"},
                {"edge_type": "HAS_FINAL_EVIDENCE", "source_id": "legal_unit::target_unit", "target_id": "final_evidence::target"},
                {"edge_type": "HAS_FINAL_EVIDENCE", "source_id": "legal_unit::target_unit", "target_id": "final_evidence::historical"},
            ]
            article_amendment_relations = [
                {
                    "relation_id": "valid-modifies",
                    "relation_type": "MODIFIES",
                    "source_legal_unit_id": "source_unit",
                    "target_legal_unit_id": "target_unit",
                    "evidence_id": "source",
                    "bbox_refs": ["bbox-source"],
                    "runtime_loadable": True,
                    "validator_status": "valid",
                }
            ]
            semantic_graph_edges = [
                {
                    "edge_id": "relation::valid-modifies",
                    "edge_type": "MODIFIES",
                    "source_id": "legal_unit::source_unit",
                    "target_id": "legal_unit::target_unit",
                    "relation_id": "valid-modifies",
                }
            ]

            def get(self, evidence_id):
                return next((row for row in self.evidence if row["evidence_id"] == evidence_id), None)

            def bboxes_for(self, evidence_id):
                return [{"bbox_id": evidence_id}]

        ranked, trace = merge_ranked(Store(), {"relation": (Store.evidence[0],)}, {}, semantic=True)
        self.assertEqual([row["evidence_id"] for row in ranked], ["source", "target"])
        self.assertEqual(trace[0]["relation_ids"], ("valid-modifies",))
        self.assertIn("MODIFIES", trace[0]["edge_types"])

    def test_runtime_bm25_exposes_ranked_route_signals(self) -> None:
        result = self.service.ask("uud", "negara hukum", limit=3)
        self.assertEqual(result["status"], "limited_answer")
        self.assertEqual(result["route"], "lexical_fallback")
        self.assertFalse(result["expansion_trace"])
        for row in result["matches"]:
            self.assertTrue(row["bbox_count"])
            self.assertIn("route_sources", row)
            self.assertIn("rank_reasons", row)
            self.assertIn("route_score", row)
            self.assertNotIn("graph", row["route_sources"])

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
        self.assertFalse(any(row["route_sources"] == ("graph",) for row in exact["matches"]))
        self.assertFalse(exact["context_pack"]["supporting_context"])
        self.assertTrue(
            all(
                direct_routes & set(row["route_sources"])
                for row in exact["matches"]
                if row["evidence_id"] in {evidence["evidence_id"] for evidence in exact["evidence"]}
            )
        )
        self.assertFalse(any(evidence["citation"] == "PEMBUKAAN/Preambule" for evidence in exact["evidence"]))

        structured = self.service.ask("uud", "Pembukaan", limit=5)
        self.assertEqual(structured["status"], "answer_ready")
        self.assertEqual(structured["route"], "legal_reference")
        self.assertEqual(structured["answer_type"], "quoted_evidence")
        self.assertFalse(any(row["route_sources"] == ("graph",) for row in structured["matches"]))
        evidence_ids = {evidence["evidence_id"] for evidence in structured["evidence"]}
        self.assertTrue(
            all(direct_routes & set(row["route_sources"]) for row in structured["matches"] if row["evidence_id"] in evidence_ids)
        )

        search = self.service.search("uud", "Pasal 1 ayat (3)", limit=5)
        self.assertEqual(search["route"], "document_catalog")
        self.assertTrue(all(row["status"] == "document" for row in search["results"]))

    def test_pasal_parent_aggregate_is_single_citation_and_fails_closed_without_geometry(self) -> None:
        parent = self.service.ask("uud", "Pasal 31", limit=20)
        self.assertEqual(parent["status"], "answer_ready")
        self.assertEqual(len(parent["citations"]), 1)
        self.assertEqual(parent["citations"][0]["citation"], "Pasal 31")
        self.assertEqual(parent["citations"][0]["source_role"], "current_consolidated")
        self.assertGreater(parent["citations"][0]["bbox_count"], 1)

        child = self.service.ask("uud", "Pasal 31 ayat (1)", limit=20)
        self.assertEqual(child["status"], "answer_ready")
        self.assertEqual(len(child["citations"]), 1)
        self.assertEqual(child["citations"][0]["citation"], "(1)")

        recovered = self.service.ask("uud", "Pasal 3", limit=20)
        self.assertEqual(recovered["status"], "answer_ready")
        self.assertEqual(len(recovered["citations"]), 1)
        self.assertEqual(recovered["citations"][0]["citation"], "Pasal 3")
        self.assertGreater(recovered["citations"][0]["bbox_count"], 1)

        incomplete = self.service.ask("uud", "Pasal", limit=20)
        self.assertEqual(incomplete["status"], "insufficient_evidence")
        self.assertEqual(incomplete["citations"], ())
        self.assertIn("incomplete_legal_reference", incomplete["insufficient_reasons"])

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

    def test_answer_context_validator_explains_inclusion_and_exclusion(self) -> None:
        class Store:
            legal_units = ({"legal_unit_id": "lu", "runtime_loadable": True, "text_span_ids": ("s1",)},)
            chunks = ({"legal_unit_id": "lu", "runtime_loadable": True, "text_span_ids": ("s1",)},)
            retrieval_units = ()

            def bboxes_for(self, evidence_id):
                return [] if evidence_id == "missing_bbox" else [{"bbox_id": evidence_id}]

            def lineage_error(self, evidence):
                return None

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
            "bbox_precision": "exact",
            "viewer_highlightable": True,
            "bbox_ids": ("direct",),
            "text_span_ids": ("s1",),
        }
        graph = base | {"evidence_id": "graph", "route_sources": ("graph",), "bbox_ids": ("graph",)}
        missing_bbox = base | {"evidence_id": "missing_bbox", "bbox_ids": ()}
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
            retrieval_units = ({"evidence_id": "not_accepted", "status": "excluded_public_answer"},)

            def bboxes_for(self, evidence_id):
                return [{"bbox_id": "safe"}]

            def lineage_error(self, evidence):
                return None

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
            "bbox_ids": ("safe",),
            "text_span_ids": ("s1",),
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
            self.assertEqual(ask["status"], "limited_answer", query)
            self.assertIn(evidence_id, {row["evidence_id"] for row in ask["evidence"]}, query)
            self.assertFalse(ask["viewer_refs"], query)
            self.assertFalse(ask["citations"], query)
            self.assertTrue(any(row["evidence_id"] == evidence_id for row in ask["trace_support"]), query)

    def test_exact_fail_closed_instrument_queries_do_not_substitute_neighbors(self) -> None:
        for query in (
            "Perubahan Pertama Decision",
            "Perubahan Ketiga Decision",
            "Perubahan Keempat Decision",
            "Perubahan Ketiga Scope",
            "Perubahan Keempat Scope",
        ):
            ask = self.service.ask("uud", query, limit=10)
            self.assertEqual(ask["status"], "limited_answer", query)
            self.assertEqual(ask["route"], "instrument_resolved_answerable", query)
            self.assertTrue(ask["evidence"], query)
            self.assertFalse(ask["citations"], query)
            self.assertFalse(ask["viewer_refs"], query)
            self.assertTrue(any(row["evidence_id"] in {item["evidence_id"] for item in ask["evidence"]} for row in ask["trace_support"]), query)

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
            ("materi perubahan keempat", "Perubahan Keempat Scope"),
            ("substansi amandemen keempat", "Perubahan Keempat Scope"),
            ("decision, perubahan ketiga", "Perubahan Ketiga Decision"),
            ("decision: perubahan ketiga", "Perubahan Ketiga Decision"),
        )
        for query, citation in cases:
            ask = self.service.ask("uud", query, limit=10)
            self.assertEqual(ask["status"], "limited_answer", query)
            self.assertEqual(ask["route"], "instrument_resolved_answerable", query)
            self.assertTrue(ask["evidence"], query)
            self.assertFalse(ask["citations"], query)
            self.assertFalse(ask["viewer_refs"], query)
            self.assertTrue(ask["trace_support"], query)

    @pytest.mark.runtime_policy
    @pytest.mark.slow
    def test_instrument_intent_matrix_blocks_neighbor_fallback(self) -> None:
        config = CorpusRegistry(ROOT).resolve("uud")
        intent = intent_config_for(config.query_strategy, config)
        matrix = validation_intent_config_for(config.query_strategy, config)["instrument_intent_matrix"]
        queries = [
            template.format(role=role, amendment=amendment)
            for role in matrix["role_family_terms"]
            for amendment in matrix["amendment_terms"]
            for template in matrix["word_orders"]
        ]
        self.assertGreater(len(queries), 0)
        for query in queries:
            decision = resolve_instrument_intent(query, intent, corpus="uud")
            self.assertEqual(decision.target_status, "instrument_resolved_fail_closed", query)

        forbidden = ("Determination", "Recital", "Closing", "Signatories", "Clause")
        for query in (queries[0], queries[len(queries) // 2], queries[-1]):
            ask = self.service.ask("uud", query, limit=10)
            search = self.service.search("uud", query, limit=10)
            self.assertFalse(ask["route"] == "lexical_fallback" and ask["evidence"], query)
            self.assertFalse(search["route"] == "bm25" and search["results"], query)
            self.assertPublicSearchHasNoEvidenceRows(search, query)
            for row in ask["evidence"]:
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
            self.assertPublicSearchHasNoEvidenceRows(search, query)
            for row in ask["evidence"]:
                self.assertIn("Scope", row.get("citation") or "", query)

        ask = self.service.ask("uud", "huruf perubahan keempat", limit=10)
        search = self.service.search("uud", "huruf perubahan keempat", limit=10)
        self.assertEqual(ask["route"], "instrument_unresolved")
        self.assertFalse(ask["evidence"])
        self.assertPublicSearchHasNoEvidenceRows(search, "huruf perubahan keempat")

    def test_partial_signal_instrument_queries_fail_closed_before_bm25(self) -> None:
        queries = (
            "ketentuan yang berubah perubahan pertama",
            "ketentuan yang berubah perubahan keempat",
            "ubah pasal apa perubahan keempat",
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
            self.assertIn(ask["route"], {"document_relation", "instrument_unresolved", "instrument_resolved_fail_closed"}, query)
            self.assertFalse(ask["evidence"], query)
            self.assertPublicSearchHasNoEvidenceRows(search, query)
            for row in (*ask["evidence"], *search["results"]):
                self.assertFalse(any(token in (row.get("citation") or "") for token in forbidden), query)

    def test_partial_signal_boundary_does_not_overblock_noninstrument_routes(self) -> None:
        education = self.service.ask("uud", "pasal apa yang mengatur pendidikan", limit=10)
        self.assertNotIn(education["route"], {"instrument_unresolved", "instrument_resolved_fail_closed"})
        self.assertEqual(education["status"], "insufficient_evidence")
        self.assertIn("claim_support_insufficient", education["insufficient_reasons"])

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
            self.assertPublicSearchHasNoEvidenceRows(search, query)
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
            self.assertPublicSearchHasNoEvidenceRows(search, query)
            for row in (*ask["evidence"], *search["results"]):
                self.assertFalse(any(token in (row.get("citation") or "") for token in forbidden), query)

    def test_unresolved_amendment_context_defaults_fail_closed_before_bm25(self) -> None:
        for query in (
            "fungsi perubahan keempat",
            "esensi perubahan keempat",
            "pokok perubahan keempat",
            "konsep perubahan keempat",
            "filosofi perubahan keempat",
            "sejarah perubahan keempat",
            "rasio legis perubahan keempat",
            "landasan perubahan keempat",
            "kenapa perubahan keempat",
            "mengapa perubahan keempat",
        ):
            ask = self.service.ask("uud", query, limit=10)
            search = self.service.search("uud", query, limit=10)
            self.assertEqual(ask["status"], "insufficient_evidence", query)
            self.assertEqual(ask["route"], "instrument_unresolved", query)
            self.assertFalse(ask["evidence"], query)
            self.assertPublicSearchHasNoEvidenceRows(search, query)

    def test_analysis_metadata_conflicts_fail_closed_before_metadata(self) -> None:
        queries = (
            "apa tujuan lembaga menetapkan perubahan keempat",
            "tujuan tanggal perubahan keempat",
            "latar belakang tanggal perubahan keempat",
            "alasan sidang perubahan keempat",
            "makna rapat perubahan keempat",
            "risiko institusi perubahan keempat",
            "maksud tempat perubahan ketiga",
        )
        for query in queries:
            ask = self.service.ask("uud", query, limit=10)
            search = self.service.search("uud", query, limit=10)
            self.assertEqual(ask["status"], "insufficient_evidence", query)
            self.assertEqual(ask["route"], "instrument_unresolved", query)
            self.assertEqual(ask["reason"], "analysis_metadata_conflict", query)
            self.assertFalse(ask["evidence"], query)
            self.assertPublicSearchHasNoEvidenceRows(search, query)

    def test_pure_metadata_still_routes_after_arbitration(self) -> None:
        for query in (
            "kapan perubahan keempat ditetapkan",
            "tanggal perubahan keempat",
            "siapa menetapkan perubahan keempat",
            "lembaga yang menetapkan perubahan keempat",
            "rapat apa yang menetapkan perubahan keempat",
            "sidang yang menetapkan perubahan keempat",
            "tempat penetapan perubahan keempat",
        ):
            ask = self.service.ask("uud", query, limit=10)
            search = self.service.search("uud", query, limit=10)
            self.assertEqual(ask["route"], "metadata_fact", query)
            self.assertNotEqual(ask.get("reason"), "unsupported_analysis_intent", query)
            self.assertNotEqual(ask.get("reason"), "analysis_metadata_conflict", query)
            self.assertPublicSearchHasNoEvidenceRows(search, query)

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
            result = self.service.ask("uud", query, limit=10)
            self.assertEqual(result["route"], "instrument_resolved_answerable", query)
            self.assertTrue(result["evidence"], query)
            self.assertEqual(result["evidence"][0]["evidence_id"], evidence_id, query)

    def test_page_grounded_instrument_viewer_is_trace_only(self) -> None:
        for evidence_id in UNSAFE_INSTRUMENT_EVIDENCE.values():
            viewer = self.service.viewer("uud", evidence_id)
            self.assertEqual(viewer["status"], "viewer_payload_ready", evidence_id)
            self.assertTrue(viewer["bbox_rectangles"], evidence_id)
            self.assertEqual(viewer["bbox_precision"], "exact", evidence_id)
            self.assertTrue(viewer["viewer_highlightable"], evidence_id)

    def test_safe_and_fail_closed_instrument_records_keep_policy(self) -> None:
        safety = json.loads((ROOT / "data/final/uud/validation_report.json").read_text(encoding="utf-8"))[
            "instrument_runtime_safety_health"
        ]
        exact_grounding = json.loads((ROOT / "data/final/uud/validation_report.json").read_text(encoding="utf-8"))[
            "instrument_exact_grounding_health"
        ]
        precision = json.loads((ROOT / "data/final/uud/validation_report.json").read_text(encoding="utf-8"))[
            "instrument_query_precision_health"
        ]
        natural_precision = json.loads((ROOT / "data/final/uud/validation_report.json").read_text(encoding="utf-8"))[
            "instrument_natural_query_precision_health"
        ]
        matrix = json.loads((ROOT / "data/final/uud/validation_report.json").read_text(encoding="utf-8"))["instrument_intent_matrix_health"]
        partial = json.loads((ROOT / "data/final/uud/validation_report.json").read_text(encoding="utf-8"))[
            "partial_signal_instrument_boundary_health"
        ]
        general = json.loads((ROOT / "data/final/uud/validation_report.json").read_text(encoding="utf-8"))[
            "instrument_like_boundary_generalization_health"
        ]
        invariant = json.loads((ROOT / "data/final/uud/validation_report.json").read_text(encoding="utf-8"))[
            "instrument_intent_invariant_router_health"
        ]
        arbitration = json.loads((ROOT / "data/final/uud/validation_report.json").read_text(encoding="utf-8"))[
            "intent_arbitration_priority_health"
        ]
        default_boundary = json.loads((ROOT / "data/final/uud/validation_report.json").read_text(encoding="utf-8"))[
            "amendment_context_default_boundary_health"
        ]
        self.assertIn(safety["status"], {"complete", "incomplete"})
        self.assertEqual(exact_grounding["status"], "complete")
        self.assertEqual(precision["status"], "complete")
        self.assertIn(natural_precision["status"], {"complete", "incomplete"})
        self.assertEqual(matrix["status"], "complete")
        self.assertEqual(partial["status"], "complete")
        self.assertEqual(general["status"], "complete")
        self.assertEqual(invariant["status"], "complete")
        self.assertEqual(arbitration["status"], "complete")
        self.assertEqual(default_boundary["status"], "complete")
        self.assertEqual(default_boundary["runtime_health_mode"], "test_suite_owned")
        self.assertEqual(default_boundary["runtime_check_count"], 0)
        self.assertNotIn("actual_elapsed_ms", default_boundary)
        self.assertGreater(matrix["matrix_query_count"], 0)
        self.assertEqual(partial["health_mode"], "resolver_config_decision")
        self.assertEqual(general["health_mode"], "resolver_config_decision")
        self.assertEqual(invariant["health_mode"], "resolver_config_decision")
        self.assertGreater(partial["partial_signal_resolver_matrix_count"], 0)
        self.assertGreater(general["resolver_matrix_count"], 0)
        self.assertGreater(invariant["resolver_matrix_count"], 0)
        self.assertGreater(invariant["heldout_analysis_probe_count"], 0)
        self.assertGreater(arbitration["conflict_matrix_count"], 0)
        for health in (
            safety,
            exact_grounding,
            precision,
            natural_precision,
            matrix,
            partial,
            general,
            invariant,
            arbitration,
            default_boundary,
        ):
            for key, value in health.items():
                if key.endswith("_count"):
                    self.assertGreaterEqual(value, 0, key)
        self.assertGreaterEqual(safety.get("instrument_records_unresolved_count", 0), 0)
        self.assertGreaterEqual(exact_grounding["inventory"]["exact_runtime"], 0)
        self.assertEqual(exact_grounding["inventory"]["trace_only"], 0)
        for query, evidence_id in (
            ("Perubahan Pertama Scope", SAFE_INSTRUMENT_EVIDENCE["00621"]),
            ("Perubahan Kedua Scope", SAFE_INSTRUMENT_EVIDENCE["00628"]),
            ("Perubahan Keempat Recital", SAFE_INSTRUMENT_EVIDENCE["00638"]),
        ):
            result = self.service.ask("uud", query, limit=10)
            self.assertIn(evidence_id, {row["evidence_id"] for row in result["evidence"]}, query)
        chunks = {
            row["chunk_id"]: row
            for row in (json.loads(line) for line in (ROOT / "data/final/uud/chunks.jsonl").read_text(encoding="utf-8").splitlines())
        }
        retrieval_chunk_ids = {
            row["chunk_id"]
            for row in (
                json.loads(line) for line in (ROOT / "data/final/uud/retrieval_units.jsonl").read_text(encoding="utf-8").splitlines()
            )
        }
        for chunk_id in ("uud_chunk_00646", "uud_chunk_00647"):
            self.assertTrue(chunks[chunk_id]["runtime_loadable"])
            self.assertTrue(chunks[chunk_id]["canonical_use_allowed"])
            self.assertIn(chunk_id, retrieval_chunk_ids)

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
            path.read_text(encoding="utf-8").casefold() for root in checked_roots for path in root.rglob("*.py") if path.name != "http.py"
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
                json.dumps(
                    {
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
                        "bbox_precision": "exact",
                        "viewer_highlightable": True,
                        "bbox_ids": ["box_1"],
                        "text_span_ids": ["span_1"],
                    }
                )
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
                json.dumps(
                    {
                        "corpus_id": "demo",
                        "schema_version": 5,
                        "evidence_registry": "proof.rows",
                        "bbox_registry": "boxes.rows",
                        "graph_nodes": "nodes.rows",
                        "graph_edges": "edges.rows",
                    }
                ),
                encoding="utf-8",
            )

            config = CorpusRegistry(root).resolve("demo")
            self.assertIsNotNone(config)
            store = EvidenceStore(config)
            self.assertEqual(store.evidence[0]["evidence_id"], "demo_evidence_1")
            self.assertEqual(store.bboxes_for("demo_evidence_1")[0]["bbox_id"], "box_1")
            self.assertEqual(GraphStore(config).counts(), {"nodes": 2, "edges": 1})
            response = LegalRuntimeService(root).ask("demo", "generic corpus resolution")
            self.assertEqual(response["status"], "corpus_not_ready")
            self.assertFalse(response["readiness"])
            self.assertEqual(config.query_strategy, "generic")
            semantics = interpret_query(store, "demo", "Pasal 1")
            self.assertEqual((semantics.requested_function, semantics.legal_references), ("retrieval", ()))
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
            (root / "corpus/manifest.json").write_text(
                json.dumps({"corpus_id": "demo", "schema_version": 5}),
                encoding="utf-8",
            )
            (root / "data/corpus_registry.json").write_text(
                json.dumps(
                    {
                        "demo": {
                            "manifest": "corpus/manifest.json",
                            "query_strategy": "demo_strategy",
                            "source_roles": ["published"],
                            "temporal_contexts": ["current"],
                            "preferred_source_role": "published",
                        }
                    }
                ),
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
            (root / "data/final/demo/manifest.json").write_text(
                json.dumps({"corpus_id": "demo", "schema_version": 5}),
                encoding="utf-8",
            )
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
                    json.dumps(
                        {
                            "corpus_id": "demo",
                            "schema_version": 5,
                            "evidence_registry": artifact_path,
                        }
                    ),
                    encoding="utf-8",
                )
                config = CorpusRegistry(root).resolve("demo")
                self.assertIsNotNone(config)
                with self.assertRaises(ValueError):
                    config.artifact_path("evidence")


if __name__ == "__main__":
    unittest.main()
