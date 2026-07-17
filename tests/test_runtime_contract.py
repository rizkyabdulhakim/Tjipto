from __future__ import annotations

import json
from pathlib import Path
import os
import tempfile
import unittest
from typing import Any

import pytest

from tjipto.corpora.registry import CorpusRegistry
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
from tjipto.runtime.api import _public_bbox, handle_request
from tjipto.runtime.service import LegalRuntimeService
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

    def assertPublicSearchHasNoEvidenceRows(self, search: dict, query: str) -> None:
        self.assertNotEqual(search["route"], "bm25", query)
        for row in search["results"]:
            self.assertEqual(row.get("status"), "document", query)
            self.assertNotIn("evidence_id", row, query)
            self.assertEqual(row.get("bbox_count"), 0, query)

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
        bbox_ids = {row["bbox_id"] for row in read_jsonl(ROOT / "data/final/uud/bbox_registry.jsonl")}
        rows = read_jsonl(ROOT / "data/final/uud/retrieval_units.jsonl")
        self.assertEqual(len(rows), config.manifest["counts"]["retrieval_units"])
        for row in rows:
            self.assertIn(row["evidence_id"], evidence_ids)
            self.assertGreaterEqual(row["bbox_total_count"], len(row["bbox_sample_refs"]))
            self.assertTrue(set(row["bbox_sample_refs"]) <= bbox_ids)
            self.assertTrue((ROOT / row["source_pdf_path"]).exists())

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

    def test_bm25_prioritizes_term_frequency_without_breaking_exact_citation(self) -> None:
        config = CorpusRegistry(ROOT).resolve("uud")
        store = EvidenceStore(config)
        citation = self.service.citation("uud", "Pasal 1 ayat (3)")
        search = route_retrieval("uud", "Pasal 1 ayat (3)", store, limit=1, allow_bm25_after_citation_miss=True)
        self.assertEqual(search["matches"][0]["evidence_id"], citation["matches"][0]["evidence_id"])

        results = route_retrieval("uud", "negara negara negara hukum", store, limit=3)
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
                self.assertEqual(result["metadata_support"][0]["evidence_id"], case["evidence_id"], case["query"])
                self.assertEqual(result["metadata_support"][0]["field"], case["field"], case["query"])
                self.assertEqual(result["metadata_support"][0]["answer"], case["answer"], case["query"])
                support = result["metadata_support"][0]
                if support["support_class"] == "exact_metadata_citation":
                    self.assertTrue(result["citations"], case["query"])
                    self.assertTrue(result["viewer_refs"], case["query"])
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
        self.assertTrue(result["citations"])
        self.assertTrue(result["viewer_refs"])
        self.assertEqual(result["citations"][0]["authority_kind"], "metadata_source")
        self.assertEqual(result["citations"][0]["support_kind"], "metadata_source")
        self.assertFalse(result["citations"][0]["citation_final"])
        self.assertEqual(result["viewer_refs"][0]["authority_kind"], "metadata_source")
        self.assertEqual(result["viewer_refs"][0]["support_kind"], "metadata_source")
        self.assertFalse(result["viewer_refs"][0]["citation_final"])
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
        self.assertTrue(result["citations"])
        self.assertTrue(result["viewer_refs"])

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
        self.assertTrue(result["citations"])
        self.assertEqual(result["citations"][0]["authority_kind"], "metadata_source")
        self.assertFalse(result["citations"][0]["citation_final"])

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
            self.assertEqual(result["status"], "answer_ready", query)
            self.assertEqual(result["route"], "document_relation", query)
            self.assertEqual(result["intent"], "document_amendment_relation", query)
            self.assertFalse(result["evidence"], query)
            self.assertFalse(result["citations"], query)
            self.assertFalse(result["viewer_refs"], query)
            self.assertEqual(len(result["document_relations"]), 4, query)
            self.assertEqual({row["relation_type"] for row in result["document_relations"]}, {"AMENDED_BY"}, query)
            self.assertTrue(all(row["highlightable"] is False for row in result["document_relations"]), query)
            self.assertIn("Perubahan Pertama", result["answer"], query)
            self.assertIn("Perubahan Keempat", result["answer"], query)

        for query in ("amandemen pertama mengubah apa", "perubahan pertama mengubah apa"):
            result = self.service.ask("uud", query)
            self.assertEqual(result["status"], "answer_ready", query)
            self.assertEqual(result["route"], "document_relation", query)
            self.assertEqual(result["document_relations"][0]["relation_type"], "AMENDS", query)
            self.assertEqual(result["document_relations"][0]["source_role"], "amendment_1_historical", query)
            self.assertFalse(result["viewer_refs"], query)

    def test_article_level_amendment_relations_use_exact_artifact_only(self) -> None:
        pasal = self.service.ask("uud", "amandemen keempat mengubah pasal apa saja")
        self.assertEqual(pasal["status"], "limited_answer")
        self.assertEqual(pasal["route"], "document_relation")
        self.assertEqual(pasal["answer_type"], "article_amendment_relation")
        self.assertTrue(pasal["evidence"])
        self.assertTrue(pasal["citations"])
        self.assertTrue(pasal["viewer_refs"])
        self.assertTrue(pasal["article_amendment_relations"])
        self.assertTrue(pasal["trace_support"])
        self.assertEqual({row["relation_type"] for row in pasal["article_amendment_relations"]}, {"MODIFIES"})
        self.assertIn("trace_article_relation", {row["support_class"] for row in pasal["article_amendment_relations"]})
        self.assertTrue(pasal["context_pack"]["viewer_refs"])
        self.assertTrue(pasal["context_pack"]["citation_payloads"])

        complete = self.service.ask("uud", "perubahan keempat mengubah pasal 16?")
        self.assertEqual(complete["status"], "answer_ready")
        self.assertTrue(complete["evidence"])
        self.assertTrue(complete["citations"])
        self.assertTrue(complete["viewer_refs"])
        self.assertFalse(complete["trace_support"])
        self.assertTrue(all(row["can_resolve"] for row in complete["viewer_refs"]))
        self.assertFalse(complete["citations"][0]["citation_final"])
        self.assertEqual(complete["citations"][0]["authority_kind"], "instrument_provenance")

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
            self.assertEqual(result["citations"][0]["authority_kind"], "instrument_provenance", query)
            self.assertFalse(result["citations"][0]["citation_final"], query)

        for query in ("perubahan keempat menambahkan apa", "perubahan keempat menambahkan lembaga apa"):
            result = self.service.ask("uud", query)
            self.assertEqual(result["status"], "insufficient_evidence", query)
            self.assertEqual(result["route"], "document_relation", query)
            self.assertEqual(result["reason"], "relation_not_promoted", query)
            self.assertFalse(result["evidence"], query)
            self.assertFalse(result["citations"], query)
            self.assertFalse(result["viewer_refs"], query)
            self.assertFalse(result["document_relations"], query)

    def test_target_specific_article_amendment_relations_do_not_substitute_neighbors(self) -> None:
        unsupported = self.service.ask("uud", "amandemen keempat mengubah pasal 31?")
        self.assertEqual(unsupported["status"], "limited_answer")
        self.assertEqual(unsupported["route"], "document_relation")
        self.assertFalse(unsupported["evidence"])
        self.assertFalse(unsupported["citations"])
        self.assertFalse(unsupported["viewer_refs"])
        self.assertEqual({row["target_citation"] for row in unsupported["article_amendment_relations"]}, {"Pasal 31"})
        self.assertTrue(unsupported["trace_support"])

        exact = self.service.ask("uud", "perubahan keempat mengubah pasal 16?")
        self.assertEqual(exact["status"], "answer_ready")
        self.assertEqual({row["target_citation"] for row in exact["article_amendment_relations"]}, {"Pasal 16"})
        self.assertFalse(exact["trace_support"])
        self.assertTrue(all(row["can_resolve"] for row in exact["viewer_refs"]))
        self.assertFalse({row["target_citation"] for row in exact["article_amendment_relations"]} - {"Pasal 16"})

        partial = self.service.ask("uud", "pasal yang diubah perubahan keempat")
        self.assertEqual(partial["status"], "limited_answer")
        self.assertEqual(partial["answer_scope"], "partial_exact_article_relation")
        self.assertTrue(partial["citations"])
        self.assertTrue(partial["viewer_refs"])
        self.assertTrue(partial["article_amendment_relations"])
        self.assertTrue(partial["trace_support"])

        reverse = self.service.ask("uud", "pasal 31 diubah oleh amandemen berapa?")
        self.assertNotEqual({row["target_citation"] for row in reverse.get("article_amendment_relations", ())}, {"Pasal 16"})
        self.assertNotIn("Pasal 16", reverse.get("answer", ""))
        if reverse["status"] == "answer_ready":
            self.assertEqual({row["target_citation"] for row in reverse["article_amendment_relations"]}, {"Pasal 31"})
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
        self.assertEqual(len(exact["citations"]), 1)
        self.assertIn("dukungan sumber exact", exact["answer"].casefold())

        paragraph = self.service.ask("uud", "Pasal 3 ayat (3) menjadi Pasal 3 ayat (2)")
        self.assertEqual(paragraph["route"], "document_relation")
        self.assertEqual(paragraph["status"], "limited_answer")
        self.assertEqual(
            [(row["source_reference"], row["target_reference"]) for row in paragraph["article_amendment_relations"]],
            [("Pasal 3 ayat (3)", "Pasal 3 ayat (2)")],
        )
        self.assertEqual(paragraph["article_amendment_relations"][0]["source_legal_unit_id"], "uud_legal_unit_00483")
        self.assertEqual(paragraph["article_amendment_relations"][0]["source_reference_range_kind"], "literal")
        self.assertTrue(all(not row["viewer_highlightable"] for row in paragraph["article_amendment_relations"]))
        self.assertNotIn("didukung bukti exact:", paragraph["answer"].casefold())

        anomaly = self.service.ask("uud", "Apa konflik sumber Pasal 25E dan Pasal 25A Perubahan Kedua?")
        self.assertEqual(anomaly["route"], "source_anomaly_explanation")
        self.assertFalse(anomaly.get("article_amendment_relations"))

        public = handle_request("uud", "ask", {"query": "Pasal 25E menjadi Pasal 25A"}, service=self.service)
        self.assertEqual(public["article_amendment_relations"][0]["target_reference"], "Pasal 25A")
        self.assertEqual(public["article_amendment_relations"][0]["source_legal_unit_id"], "uud_legal_unit_00428")

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
                    self.assertEqual(result["citations"][0]["citation"], case["citation"], case["query"])
                else:
                    self.assertTrue(result["trace_support"], case["query"])
                continue
            self.assertEqual(result["evidence"][0]["candidate_type"], case["candidate_type"], case["query"])
            self.assertEqual(result["evidence"][0]["evidence_id"], case["evidence_id"], case["query"])
            self.assertEqual(result["citations"][0]["citation"], case["citation"], case["query"])
            self.assertEqual(result["citations"][0]["evidence_id"], case["evidence_id"], case["query"])
            self.assertEqual(result["citations"][0]["authority_kind"], "instrument_provenance", case["query"])
            self.assertFalse(result["citations"][0]["citation_final"], case["query"])

    def test_ask_explains_known_source_anomalies_safely(self) -> None:
        for case in _source_conflict_cases():
            result = self.service.ask("uud", case["query"])
            self.assertEqual(result["status"], case["expected_status"], case["query"])
            self.assertEqual(result["route"], "source_anomaly_explanation", case["query"])
            self.assertEqual(result["source_conflict"]["source_conflict_id"], case["source_conflict_id"], case["query"])
            self.assertEqual(result["source_conflict"]["source_anomaly_kind"], case["source_anomaly_kind"], case["query"])
            self.assertEqual(bool(result["citations"]), case["has_citations"], case["query"])
            self.assertEqual(bool(result["viewer_refs"]), case["has_viewer_refs"], case["query"])
            self.assertEqual(len(result.get("trace_support", ())), case["trace_support_count"], case["query"])
            for reason in case["expected_insufficient_reasons"]:
                self.assertIn(reason, result["insufficient_reasons"], case["query"])
            self.assertIn("source_conflict_not_final_legal_authority", result["warnings"], case["query"])
            if result["citations"]:
                expected_kind = (
                    "source_anomaly" if case["source_anomaly_kind"] == "source_marker_sequence_anomaly" else "source_conflict_provenance"
                )
                self.assertEqual(result["citations"][0]["authority_kind"], expected_kind, case["query"])
                self.assertFalse(result["citations"][0]["citation_final"], case["query"])
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
        self.assertFalse(result["rendering_available"])
        self.assertFalse(result["bbox_rectangles"])

        page_grounded = evidence | {"bbox_precision": "page_grounded_only", "viewer_highlightable": True}
        result = viewer_payload(store, "uud", page_grounded, store.bboxes_for(evidence["evidence_id"]))
        self.assertEqual(result["status"], "source_page_trace_only")
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
            reasons = set(result["context_pack"]["validation_reasons"].values())
            if result["route"] == "lexical_fallback":
                self.assertIn("insufficient_query_support", reasons, case["query"])

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

    def test_president_three_terms_numeric_word_variants_are_consistent(self) -> None:
        for query in (
            "bolehkah presiden menjabat tiga periode",
            "boleh presiden 3 periode?",
            "presiden boleh tiga periode?",
        ):
            result = self.service.ask("uud", query)
            self.assertEqual(result["status"], "limited_answer", query)
            self.assertEqual(result["route"], "lexical_fallback", query)
            self.assertTrue(result["citations"], query)
            self.assertEqual(result["citations"][0]["evidence_id"], "uud_current_consolidated_final_citation_evidence_00263", query)

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
        self.assertEqual([row["citation"] for row in current["citations"]], ["BAB IV"])
        self.assertEqual(current["citations"][0]["source_role"], "current_consolidated")
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
                "filters": {"source_role": "amendment_1_historical", "temporal_context": "amendment_1_historical"},
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
            self.assertEqual(report[key]["raw_pdf_match"], 626)
            self.assertEqual(report[key]["normalized_pdf_match"], 626)
            self.assertEqual(report[key]["header_stripped_pdf_match"], 650)
            self.assertEqual(report[key]["evidence_grounded_match"], 472)
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
        self.assertEqual([row["evidence_id"] for row in ranked], ["e1", "e2", "e4"])
        self.assertEqual(ranked[0]["route_sources"], ("bm25", "structured"))
        self.assertIn("pass", ranked[0]["rank_reasons"])
        self.assertTrue(trace)
        self.assertTrue(all(item["evidence_id"] != "no_box" for item in trace))

        empty, empty_trace = merge_ranked(store, {}, {"status": "final"})
        self.assertEqual(empty, ())
        self.assertEqual(empty_trace, ())

    def test_runtime_bm25_exposes_ranked_route_signals(self) -> None:
        result = self.service.ask("uud", "negara hukum", limit=3)
        self.assertEqual(result["status"], "limited_answer")
        self.assertEqual(result["route"], "lexical_fallback")
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
        self.assertTrue(any(row["route_sources"] == ("graph",) for row in structured["matches"]))
        evidence_ids = {evidence["evidence_id"] for evidence in structured["evidence"]}
        self.assertTrue(
            all(direct_routes & set(row["route_sources"]) for row in structured["matches"] if row["evidence_id"] in evidence_ids)
        )

        search = self.service.search("uud", "Pasal 1 ayat (3)", limit=5)
        self.assertEqual(search["route"], "document_catalog")
        self.assertTrue(all(row["status"] == "document" for row in search["results"]))

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
            self.assertEqual(ask["status"], "answer_ready", query)
            self.assertIn(evidence_id, {row["evidence_id"] for row in ask["evidence"]}, query)
            self.assertTrue(ask["viewer_refs"], query)
            self.assertFalse(ask["citations"][0]["citation_final"], query)

    def test_exact_fail_closed_instrument_queries_do_not_substitute_neighbors(self) -> None:
        for query in (
            "Perubahan Pertama Decision",
            "Perubahan Ketiga Decision",
            "Perubahan Keempat Decision",
            "Perubahan Ketiga Scope",
            "Perubahan Keempat Scope",
        ):
            ask = self.service.ask("uud", query, limit=10)
            self.assertEqual(ask["status"], "answer_ready", query)
            self.assertEqual(ask["route"], "instrument_resolved_answerable", query)
            self.assertTrue(ask["evidence"], query)
            self.assertTrue(ask["citations"], query)
            self.assertTrue(ask["viewer_refs"], query)
            self.assertFalse(ask["citations"][0]["citation_final"], query)
            self.assertEqual(ask["citations"][0]["authority_kind"], "instrument_provenance", query)

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
            self.assertEqual(ask["status"], "answer_ready", query)
            self.assertEqual(ask["route"], "instrument_resolved_answerable", query)
            self.assertTrue(ask["evidence"], query)
            self.assertTrue(ask["citations"], query)
            self.assertEqual(ask["citations"][0]["authority_kind"], "instrument_provenance", query)
            self.assertFalse(ask["citations"][0]["citation_final"], query)

    @pytest.mark.runtime_policy
    @pytest.mark.slow
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
        self.assertEqual(safety["status"], "complete")
        self.assertEqual(exact_grounding["status"], "complete")
        self.assertEqual(precision["status"], "complete")
        self.assertEqual(natural_precision["status"], "complete")
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
                if key.endswith("_count") and key not in {
                    "matrix_query_count",
                    "partial_signal_resolver_matrix_count",
                    "resolver_matrix_count",
                    "heldout_analysis_probe_count",
                    "conflict_matrix_count",
                    "runtime_check_count",
                }:
                    self.assertEqual(value, 0, key)
        self.assertGreater(exact_grounding["inventory"]["exact_runtime"], 0)
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
                        "schema_version": 4,
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
                json.dumps({"corpus_id": "demo", "schema_version": 4}),
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
                json.dumps({"corpus_id": "demo", "schema_version": 4}),
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
                            "schema_version": 4,
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
