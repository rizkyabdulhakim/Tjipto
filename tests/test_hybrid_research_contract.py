from __future__ import annotations

from types import SimpleNamespace
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from tjipto.retrieval.hybrid import RetrievalHit, hybrid_search, reciprocal_rank_fusion
from tjipto.retrieval.research import (
    OpenAICompatibleResearchPlanningProvider,
    ResearchIntent,
    execute_research,
    plan_research,
)
from tjipto.retrieval.sufficiency import EvidenceRequirement, EvidenceSet, SufficiencyAssessment, assess_sufficiency, collect_evidence_set
from tjipto.runtime.service import LegalRuntimeService
from tjipto.runtime.query_semantics import interpret_query
from tjipto.runtime.service import _research_requirements_for_ask


ROOT = Path(__file__).resolve().parents[1]


def _hit(evidence_id: str, lane: str, rank: int, score: float) -> RetrievalHit:
    return RetrievalHit(evidence_id, {"evidence_id": evidence_id}, lane, rank, score, lane)


class HybridResearchContractTest(unittest.TestCase):
    def test_semantic_research_reuses_original_route(self) -> None:
        service = LegalRuntimeService()
        routed = {
            "status": "no_results",
            "route": "hybrid",
            "intent": "natural_language",
            "matches": (),
            "reason": "no_results",
        }
        with patch.object(service, "_route_retrieval", return_value=routed) as retrieval:
            service.ask("uud", "hak konstitusional ketika sekolah melarang agama")
        self.assertEqual(retrieval.call_count, 1)

    def test_planner_sparse_and_dense_lanes_execute_as_true_hybrid(self) -> None:
        class Provider:
            def propose(self, _request):
                return {
                    "variants": [],
                    "retrieval_lanes": ["sparse", "dense"],
                    "task_kind": "comparison",
                    "information_needs": [],
                }

        observed = []

        def retrieve(_query, variant):
            observed.append(variant.retrieval_lane)
            return {"matches": ()}

        plan, _ = execute_research(
            "amandemen pertama vs kedua",
            retrieve,
            intent=ResearchIntent(comparison=True, max_rounds=1),
            provider=Provider(),
        )
        self.assertEqual(plan.retrieval_lanes, ("hybrid",))
        self.assertEqual(observed, ["hybrid"])

    def test_planner_variants_pay_dense_worker_startup_once_per_request(self) -> None:
        class Provider:
            def propose(self, _request):
                return {
                    "variants": [{"query": "varian satu"}, {"query": "varian dua"}],
                    "retrieval_lanes": ["hybrid"],
                    "task_kind": "comparison",
                    "information_needs": [],
                }

        observed = []
        execute_research(
            "query asli",
            lambda _query, variant: observed.append(variant.retrieval_lane) or {"matches": ()},
            intent=ResearchIntent(comparison=True, max_rounds=1),
            provider=Provider(),
        )
        self.assertEqual(observed, ["hybrid", "sparse", "sparse"])

    def test_stage8_9_operations_bind_requirements_to_shared_sufficiency(self) -> None:
        service = LegalRuntimeService()
        store = service._store("uud")
        cases = (
            (
                "amandemen pertama vs kedua",
                {"instrument_amendment_1_historical", "instrument_amendment_2_historical"},
                "complete",
            ),
            (
                "perbedaan penandatangan amandemen pertama dan kedua",
                {"signatory_amendment_1_historical", "signatory_amendment_2_historical"},
                "complete",
            ),
            (
                "Pasal 16 sebelum dihapus bunyinya apa",
                {"historical_normative_text", "deletion_provenance"},
                "insufficient",
            ),
            (
                "legal opinion tentang HAM dari Pasal 28",
                {"analysis_issue_provisions", "analysis_limitations_exceptions"},
                "complete",
            ),
        )
        for query, expected_ids, expected_status in cases:
            semantics = interpret_query(store, "uud", query)
            requirements = _research_requirements_for_ask(store, semantics, query)
            self.assertEqual({item.requirement_id for item in requirements}, expected_ids, query)
            response = service.ask("uud", query, limit=30)
            self.assertEqual(response["sufficiency"]["status"], expected_status, query)
            if expected_status == "complete":
                self.assertEqual(set(response["sufficiency"]["fulfilled_requirement_ids"]), expected_ids, query)
            else:
                self.assertEqual(response["sufficiency"]["missing_requirement_ids"], ("deletion_provenance",), query)
        metadata_requirements = _research_requirements_for_ask(
            store,
            interpret_query(store, "uud", cases[1][0]),
            cases[1][0],
        )
        self.assertEqual({item.metadata_field for item in metadata_requirements}, {"signatories"})
        evidence = EvidenceSet(
            ({"evidence_id": "mandatory"},),
            (("mandatory", ("mandatory",)),),
            ("allowed",),
            (("allowed", "verified_support_missing"),),
        )
        self.assertEqual(
            assess_sufficiency(
                evidence,
                (
                    EvidenceRequirement("mandatory", evidence_ids=("mandatory",)),
                    EvidenceRequirement("allowed", evidence_ids=("allowed",), allow_partial=True),
                ),
            ).status,
            "partial",
        )

    def test_heldout_semantics_derive_typed_requirements_without_runtime_fixture_data(self) -> None:
        service = LegalRuntimeService()
        store = service._store("uud")
        fixture = ROOT / "tests/fixtures/uud/research_semantic_cases.jsonl"
        for line in fixture.read_text(encoding="utf-8").splitlines():
            case = json.loads(line)
            semantics = interpret_query(store, "uud", case["query"])
            requirements = _research_requirements_for_ask(store, semantics, case["query"])
            self.assertEqual([row.requirement_id for row in requirements], case["required_ids"], case["case_id"])
            response = service.ask("uud", case["query"])
            self.assertIn(response["status"], {"answer_ready", "limited_answer"}, case["case_id"])
            self.assertEqual(response["sufficiency"]["status"], "complete", case["case_id"])
            self.assertEqual(
                set(response["sufficiency"]["fulfilled_requirement_ids"]),
                set(case["required_ids"]),
                case["case_id"],
            )
        runtime_config = json.dumps(store.config.setting("research", {}), ensure_ascii=False)
        self.assertNotIn("heldout_analysis_probes", runtime_config)
    def test_rrf_deduplicates_and_never_adds_raw_scores(self) -> None:
        sparse = (_hit("b", "bm25", 1, 100.0), _hit("a", "bm25", 2, 1.0))
        dense = (_hit("a", "dense", 1, 0.99), _hit("b", "dense", 2, 0.1))
        fused = reciprocal_rank_fusion({"bm25": sparse, "dense": dense}, k=1)
        self.assertEqual([hit.evidence_id for hit in fused], ["a", "b"])
        self.assertEqual(fused[0].raw_score, None)
        self.assertEqual(fused[0].lane_provenance, (("bm25", 2, 1.0, "bm25"), ("dense", 1, 0.99, "dense")))
        self.assertEqual(fused[0].fused_score, fused[1].fused_score)

    def test_rrf_deduplicates_repeated_evidence_within_one_lane(self) -> None:
        fused = reciprocal_rank_fusion({"bm25": (_hit("same", "bm25", 3, 1.0), _hit("same", "bm25", 1, 2.0))})
        self.assertEqual(fused[0].lane_provenance, (("bm25", 1, 2.0, "bm25"),))

    def test_hybrid_falls_back_to_sparse_with_typed_reason(self) -> None:
        class Index:
            def search(self, query, limit):
                return [{"evidence_id": "sparse", "quoted_text": query, "_bm25_provenance": {"raw_score": 2.0, "rank": 1}}]

        with patch("tjipto.retrieval.hybrid.sparse_index_for_store", return_value=Index()), patch(
            "tjipto.retrieval.hybrid.dense_search",
            return_value={"status": "dense_unavailable", "reason": "worker_timeout", "matches": ()},
        ):
            result = hybrid_search(SimpleNamespace(), "query", 2)
        self.assertEqual(result["status"], "found")
        self.assertEqual(result["retrieval_degraded_reason"], "worker_timeout")
        self.assertEqual(result["matches"][0]["evidence_id"], "sparse")
        self.assertNotIn("raw_score", repr(result))

    def test_hybrid_filters_scope_before_final_cutoff(self) -> None:
        class Index:
            def search(self, query, limit):
                return [
                    {"evidence_id": "wrong", "source_role": "historical", "status": "final", "_bm25_provenance": {"rank": 1}},
                    {"evidence_id": "right", "source_role": "current", "status": "final", "_bm25_provenance": {"rank": 2}},
                ][:limit]

        with patch("tjipto.retrieval.hybrid.sparse_index_for_store", return_value=Index()), patch(
            "tjipto.retrieval.hybrid.dense_search",
            return_value={"status": "dense_unavailable", "reason": "not_configured", "matches": ()},
        ):
            result = hybrid_search(SimpleNamespace(evidence=({}, {})), "query", 1, filters={"status": "final", "source_role": "current"})
        self.assertEqual([row["evidence_id"] for row in result["matches"]], ["right"])

    def test_hybrid_filters_status_before_final_cutoff(self) -> None:
        class Index:
            def search(self, query, limit):
                return [
                    {"evidence_id": "draft", "status": "draft", "_bm25_provenance": {"rank": 1}},
                    {"evidence_id": "final", "status": "final", "_bm25_provenance": {"rank": 2}},
                ][:limit]

        with patch("tjipto.retrieval.hybrid.sparse_index_for_store", return_value=Index()), patch(
            "tjipto.retrieval.hybrid.dense_search",
            return_value={"status": "dense_unavailable", "reason": "not_configured", "matches": ()},
        ):
            result = hybrid_search(SimpleNamespace(evidence=({}, {})), "query", 1, filters={"status": "final"})
        self.assertEqual([row["evidence_id"] for row in result["matches"]], ["final"])

    def test_hybrid_evidence_is_not_vetoed_by_incomplete_bm25_coverage(self) -> None:
        from tjipto.retrieval.answer import validate_answer_candidate

        row = {"evidence_id": "dense", "route_sources": ("hybrid", "dense"), "lexical_complete_coverage": False}
        with patch("tjipto.retrieval.answer._optional_rows", return_value=()):
            # Route provenance must not be the reason for rejection; the
            # remaining grounding checks may still fail on this synthetic row.
            accepted, reason = validate_answer_candidate(SimpleNamespace(lineage_error=lambda row: None, bboxes_for=lambda _id: ()), row)
        self.assertNotEqual(reason, "insufficient_query_support")

    def test_authoritative_requirement_discovery_is_not_vetoed_by_bm25_coverage(self) -> None:
        from tjipto.retrieval.answer import validate_answer_candidate

        row = {"evidence_id": "exact", "route_sources": ("bm25", "structured"), "lexical_complete_coverage": False}
        accepted, reason = validate_answer_candidate(SimpleNamespace(lineage_error=lambda row: None, bboxes_for=lambda _id: ()), row)
        self.assertNotEqual(reason, "insufficient_query_support")

    def test_requirement_assignment_is_verified_and_non_overlapping(self) -> None:
        rows = (
            {"evidence_id": "current", "source_role": "current", "temporal_context": "current"},
            {"evidence_id": "history", "source_role": "historical", "temporal_context": "historical"},
        )
        store = SimpleNamespace()
        with patch("tjipto.retrieval.sufficiency.validate_answer_candidate", return_value=(True, "answer_evidence")):
            evidence = collect_evidence_set(
                store,
                rows,
                (
                    EvidenceRequirement("current", source_role="current"),
                    EvidenceRequirement("history", temporal_context="historical"),
                ),
            )
        assessment = assess_sufficiency(evidence, (EvidenceRequirement("current"), EvidenceRequirement("history")))
        self.assertTrue(evidence.complete)
        self.assertEqual(assessment.status, "complete")
        self.assertEqual(evidence.assignment_map(), {"current": ("current",), "history": ("history",)})

    def test_sufficiency_reports_partial_and_retry_without_fabrication(self) -> None:
        requirement = EvidenceRequirement("missing", allow_partial=True)
        evidence = collect_evidence_set(SimpleNamespace(), (), (requirement,))
        assessment = assess_sufficiency(evidence, (requirement,), retry_budget=1)
        self.assertEqual(assessment.status, "insufficient")
        self.assertEqual(assessment.missing_requirement_ids, ("missing",))
        self.assertTrue(assessment.retry_allowed)

    def test_mandatory_final_support_rejects_trace_only_provenance(self) -> None:
        requirement = EvidenceRequirement(
            "provenance",
            authority_kinds=("instrument_provenance",),
            requires_final_citation=True,
        )
        row = {
            "evidence_id": "trace",
            "authority_kind": "instrument_provenance",
            "citation_final": False,
        }
        with patch("tjipto.retrieval.sufficiency.validate_answer_candidate", return_value=(True, "answer_evidence")):
            evidence = collect_evidence_set(SimpleNamespace(), (row,), (requirement,))

        assessment = assess_sufficiency(evidence, (requirement,))

        self.assertEqual(assessment.status, "insufficient")
        self.assertEqual(assessment.missing_requirement_ids, ("provenance",))

    def test_description_alone_cannot_assign_arbitrary_verified_row(self) -> None:
        row = {"evidence_id": "unrelated", "source_role": "current"}
        with patch("tjipto.retrieval.sufficiency.validate_answer_candidate", return_value=(True, "answer_evidence")):
            evidence = collect_evidence_set(SimpleNamespace(), (row,), (EvidenceRequirement("requirement", description="anything"),))
        self.assertEqual(evidence.missing_requirement_ids, ("requirement",))

    def test_retrieval_query_alone_cannot_assign_arbitrary_verified_row(self) -> None:
        row = {"evidence_id": "unrelated", "quoted_text": "Mahkamah Konstitusi berwenang mengadili."}
        requirement = EvidenceRequirement("authority", retrieval_query="kewenangan presiden")
        with patch("tjipto.retrieval.sufficiency.validate_answer_candidate", return_value=(True, "answer_evidence")):
            evidence = collect_evidence_set(SimpleNamespace(), (row,), (requirement,))
        self.assertEqual(evidence.missing_requirement_ids, ("authority",))

    def test_requirement_semantics_reject_unrelated_verified_authority(self) -> None:
        rows = (
            {"evidence_id": "mk", "quoted_text": "Mahkamah Konstitusi berwenang mengadili.", "_requirement_ids": ("authority",)},
            {"evidence_id": "president", "quoted_text": "Presiden memegang kekuasaan pemerintahan.", "_requirement_ids": ("authority",)},
        )
        requirement = EvidenceRequirement(
            "authority",
            retrieval_query="kewenangan presiden",
            required_entities=("Presiden",),
            support_terms=("kekuasaan", "wewenang"),
        )
        with patch("tjipto.retrieval.sufficiency.validate_answer_candidate", return_value=(True, "answer_evidence")):
            evidence = collect_evidence_set(SimpleNamespace(), rows, (requirement,))
        self.assertEqual(evidence.assignment_map(), {"authority": ("president",)})

    def test_requirement_retry_is_scoped_and_carries_forward_support(self) -> None:
        from tjipto.retrieval.research import QueryVariant, ResearchPlan, execute_research_rounds

        calls = []
        requirement = EvidenceRequirement("missing", retrieval_query="missing query", evidence_ids=("target",))

        def retrieve(query, variant):
            calls.append((query, variant.requirement_id, variant.retrieval_lane))
            row = {"evidence_id": "target", "status": "final"} if variant.requirement_id == "missing" else {}
            return {"status": "found" if row else "no_results", "matches": (row,) if row else ()}

        store = SimpleNamespace()
        with patch("tjipto.retrieval.research.collect_evidence_set", side_effect=lambda _store, rows, reqs: EvidenceSet((rows[0],) if rows else (), (("missing", ("target",)),) if rows else (), () if rows else ("missing",))), patch(
            "tjipto.retrieval.research.assess_sufficiency", side_effect=lambda evidence, reqs, **kwargs: SufficiencyAssessment("complete" if evidence.complete else "insufficient", ("missing",) if evidence.complete else (), () if evidence.complete else ("missing",), (), False),
        ):
            result = execute_research_rounds(
                "original",
                retrieve,
                store=store,
                requirements=(requirement,),
                max_rounds=2,
                plan=ResearchPlan(
                    "original",
                    ResearchIntent(max_rounds=2),
                    (QueryVariant("original"),),
                    retrieval_lanes=("hybrid",),
                    requirements=(requirement,),
                ),
            )
        self.assertEqual(calls, [("original", None, "hybrid"), ("missing query", "missing", "sparse")])
        self.assertEqual(result["stop_reason"], "complete")

    def test_requirement_marker_survives_carry_forward_deduplication(self) -> None:
        from tjipto.retrieval.research import execute_research_rounds

        requirement = EvidenceRequirement("target", retrieval_query="target query", evidence_ids=("same",))

        def retrieve(query, variant):
            return {"status": "found", "matches": ({"evidence_id": "same", "status": "final"},)}

        with patch("tjipto.retrieval.research.collect_evidence_set", return_value=EvidenceSet(({"evidence_id": "same"},), (("target", ("same",)),), ())):
            with patch(
                "tjipto.retrieval.research.assess_sufficiency",
                side_effect=(
                    SufficiencyAssessment("insufficient", (), ("target",), (), True),
                    SufficiencyAssessment("complete", ("target",), (), (), False),
                ),
                ):
                result = execute_research_rounds("original", retrieve, requirements=(requirement,), max_rounds=2)
        self.assertEqual(result["matches"][0]["_requirement_ids"], ("target",))

    def test_requirement_rediscovery_refreshes_neutral_coverage(self) -> None:
        from tjipto.retrieval.research import execute_research_rounds

        requirement = EvidenceRequirement("target", retrieval_query="target query", evidence_ids=("same",))

        def retrieve(query, variant):
            coverage = query == "target query"
            return {"status": "found", "matches": ({
                "evidence_id": "same",
                "lexical_complete_coverage": coverage,
                "route_sources": ("bm25",),
            },)}

        with patch("tjipto.retrieval.research.collect_evidence_set", return_value=EvidenceSet(({"evidence_id": "same"},), (("target", ("same",)),), ())):
            with patch(
                "tjipto.retrieval.research.assess_sufficiency",
                side_effect=(
                    SufficiencyAssessment("insufficient", (), ("target",), (), True),
                    SufficiencyAssessment("complete", ("target",), (), (), False),
                ),
            ):
                result = execute_research_rounds("original", retrieve, requirements=(requirement,), max_rounds=2)
        self.assertTrue(result["matches"][0]["lexical_complete_coverage"])

    def test_ask_missing_requirement_is_fail_closed_without_attribute_error(self) -> None:
        response = LegalRuntimeService().ask(
            "uud",
            "apa isi negara hukum",
            evidence_requirements=(EvidenceRequirement("missing", retrieval_query="query-not-in-corpus"),),
        )
        self.assertEqual(response["status"], "insufficient_evidence")
        self.assertEqual(response["insufficient_reasons"], ("missing",))

    def test_ask_complex_query_publishes_only_when_requirements_are_complete(self) -> None:
        service = LegalRuntimeService()
        response = service.ask("uud", "lingkungan hidup dan pendidikan")
        self.assertEqual(response["status"], "answer_ready")
        self.assertEqual(response["sufficiency"]["status"], "complete")
        self.assertGreaterEqual(len(response["citations"]), 2)

    def test_broad_presidential_authority_uses_verified_multi_support(self) -> None:
        response = LegalRuntimeService().ask("uud", "apa kewenangan presiden menurut UUD")
        self.assertEqual(response["status"], "limited_answer")
        self.assertEqual(response["sufficiency"]["status"], "complete")
        self.assertGreaterEqual(len(response["sufficiency"]["fulfilled_requirement_ids"]), 1)
        self.assertTrue(response["citations"])
        self.assertNotIn("Mahkamah Konstitusi", response["answer"])

    def test_unseen_comparison_wording_keeps_both_entities_as_requirements(self) -> None:
        service = LegalRuntimeService()
        store = service._store("uud")
        for query in ("kontraskan kewenangan DPR dengan DPD", "DPR dibandingkan dengan DPD dalam kewenangan"):
            requirements = _research_requirements_for_ask(store, interpret_query(store, "uud", query), query)
            self.assertEqual({item.required_entities[0] for item in requirements}, {
                "Dewan Perwakilan Rakyat",
                "Dewan Perwakilan Daerah",
            })
            response = service.ask("uud", query)
            self.assertEqual(response["sufficiency"]["status"], "complete")
            self.assertEqual(set(response["sufficiency"]["fulfilled_requirement_ids"]), {"entity_1", "entity_2"})

    def test_generic_procedure_wording_does_not_activate_impeachment_template(self) -> None:
        service = LegalRuntimeService()
        for query in ("mekanisme perubahan UUD", "proses pembentukan undang-undang menurut UUD"):
            response = service.ask("uud", query)
            fulfilled = response.get("sufficiency", {}).get("fulfilled_requirement_ids", ())
            self.assertNotIn("grounds", fulfilled)
            self.assertNotIn("procedure_basis", fulfilled)
            self.assertNotIn("constitutional_review", fulfilled)
            self.assertNotIn("assembly_decision", fulfilled)

    def test_impeachment_relation_query_collects_both_typed_dimensions(self) -> None:
        response = LegalRuntimeService().ask("uud", "hubungan DPR dan MK dalam pemakzulan")
        self.assertEqual(response["status"], "answer_ready")
        self.assertEqual(response["sufficiency"]["status"], "complete")
        self.assertEqual(
            set(response["sufficiency"]["fulfilled_requirement_ids"]),
            {"relation_1", "relation_2"},
        )
        self.assertGreaterEqual(len(response["citations"]), 2)

    def test_impeachment_requirement_query_collects_multiple_supports(self) -> None:
        response = LegalRuntimeService().ask("uud", "apa syarat pemakzulan Presiden")
        self.assertEqual(response["status"], "answer_ready")
        self.assertEqual(response["sufficiency"]["status"], "complete")
        self.assertTrue(response["citations"])

    def test_cross_instrument_comparison_preserves_each_historical_source_role(self) -> None:
        for query in (
            "apa perbedaan Perubahan Pertama dan Perubahan Kedua UUD 1945?",
            "perbedaan uud amandemen pertama dan amandemen kedua",
        ):
            response = LegalRuntimeService().ask("uud", query)
            self.assertEqual(response["sufficiency"]["status"], "complete")
            self.assertEqual(
                {row["source_role"] for row in response["evidence"]},
                {"amendment_1_historical", "amendment_2_historical"},
            )
            self.assertTrue(all("Scope" in str(row.get("citation")) for row in response["evidence"]))

    def test_decomposition_retrieves_structural_procedure_neighbors(self) -> None:
        response = LegalRuntimeService().ask("uud", "bagaimana prosedur pemberhentian Presiden?")
        retrieved = {row.get("evidence_id") for row in response.get("matches", ())}
        self.assertTrue(
            {
                "uud_current_consolidated_final_citation_evidence_00269",
                "uud_current_consolidated_final_citation_evidence_00271",
            } <= retrieved
        )
        self.assertEqual(response["sufficiency"]["status"], "complete")

    def test_education_paraphrase_retains_article_support_in_candidate_set(self) -> None:
        service = LegalRuntimeService()
        store = service._store("uud")
        routed = service._route_retrieval("uud", "hak atas pendidikan diatur dimana", store, limit=10)
        self.assertTrue(any(row.get("citation") == "Pasal 31" for row in routed["matches"]))

    def test_verified_single_support_is_not_vetoed_by_question_surface_tokens(self) -> None:
        for query, citation in (
            ("Hak atas pendidikan diatur di mana?", "Pasal 31"),
            ("Bagaimana perubahan UUD dilakukan?", "Pasal 37"),
        ):
            service = LegalRuntimeService()
            response = service.ask("uud", query)
            self.assertNotEqual(response["status"], "insufficient_evidence", query)
            self.assertTrue(any(citation in tuple(row.get("hierarchy") or ()) or row.get("citation") == citation for row in response["matches"]), query)

    def test_navigation_returns_the_adjacent_unit_not_the_referenced_unit(self) -> None:
        for query, expected in (
            ("Apa ketentuan setelah Pasal 28?", "Pasal 28A"),
            ("Sebelum Pasal 27 apa?", "Pasal 26"),
            ("Setelah Pasal 31 ayat (3) apa?", "(4)"),
        ):
            response = LegalRuntimeService().ask("uud", query)
            self.assertEqual(response["route"], "structural_navigation", query)
            self.assertEqual(response["citations"][0]["citation"], expected, query)

    def test_lawmaking_relation_does_not_admit_impeachment_support(self) -> None:
        response = LegalRuntimeService().ask("uud", "Apa hubungan Presiden dan DPR dalam pembentukan undang-undang?")
        self.assertNotIn("Pasal 7B", {row.get("citation") for row in response["citations"]})

    def test_trace_only_document_relation_is_not_answer_ready(self) -> None:
        response = LegalRuntimeService().ask("uud", "amandemen keempat mengubah apa")
        self.assertEqual(response["route"], "document_relation")
        self.assertEqual(response["status"], "limited_answer")
        self.assertTrue(response["trace_support"])
        self.assertFalse(response["citations"])

    def test_external_tax_scope_remains_fail_closed_without_clarification(self) -> None:
        response = LegalRuntimeService().ask("uud", "undang undang tentang pajak")
        self.assertEqual(response["status"], "insufficient_evidence")
        self.assertEqual(response["route"], "unsupported_scope")
        self.assertNotEqual(response.get("clarification_kind"), "concept_facet")

    def test_simple_lexical_query_does_not_gain_research_answerability(self) -> None:
        response = LegalRuntimeService().ask("uud", "pekerjaan dan penghidupan yang layak")
        self.assertEqual(response["status"], "limited_answer")
        self.assertIsNone(response.get("sufficiency"))

    def test_planner_retains_original_and_rejects_scope_changes(self) -> None:
        class Provider:
            def propose(self, request):
                return {"variants": [
                    {"query": "rewritten", "explicit_references": ("Pasal 1",)},
                    {"query": "same"},
                ]}

        intent = ResearchIntent(comparison=True, decomposition=True, max_variants=3)
        plan = plan_research("original", intent, provider=Provider(), explicit_references=("Pasal 2",))
        self.assertEqual(plan.variants[0].query, "original")
        self.assertEqual(plan.provider_status, "degraded")
        self.assertIn("scope_invariant_violation", plan.rejection_reasons)

    def test_planner_rejects_requirement_scope_drift(self) -> None:
        class Provider:
            def propose(self, request):
                return {"requirements": [{"requirement_id": "r", "source_role": "historical"}]}

        plan = plan_research(
            "original",
            ResearchIntent(comparison=True),
            provider=Provider(),
            source_role="current",
        )
        self.assertFalse(plan.requirements)
        self.assertIn("requirement_scope_invariant_violation", plan.rejection_reasons)
        self.assertIn("provider_requirements_forbidden", plan.rejection_reasons)

    def test_planner_request_is_json_and_provider_cannot_author_support_policy(self) -> None:
        seen = {}

        class Provider:
            def propose(self, request):
                seen.update(request)
                return {
                    "requirements": [{
                        "requirement_id": "forged",
                        "evidence_ids": ["unknown"],
                        "min_supports": 99,
                    }],
                }

        plan = plan_research(
            "DPR dan DPD",
            ResearchIntent(comparison=True),
            provider=Provider(),
            required_entities=("DPR", "DPD"),
            explicit_references=("Pasal 20",),
            source_role="current_consolidated",
            temporal_scope="current_consolidated",
        )
        json.dumps(seen, ensure_ascii=False, sort_keys=True)
        self.assertIsInstance(seen["intent"], dict)
        self.assertEqual(seen["constraints"]["required_entities"], ("DPR", "DPD"))
        self.assertNotIn("evidence_ids", seen["constraints"])
        self.assertFalse(plan.requirements)
        self.assertIn("provider_requirements_forbidden", plan.rejection_reasons)

    def test_openai_compatible_planner_prompt_declares_validated_json_contract(self) -> None:
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return None

            def read(self):
                return json.dumps({
                    "choices": [{
                        "message": {
                            "content": json.dumps({
                                "variants": [{"query": "hak pendidikan"}],
                                "retrieval_lanes": ["hybrid"],
                                "task_kind": "multiple_supports",
                                "information_needs": [{
                                    "description": "hak atas pendidikan",
                                    "query": "hak pendidikan",
                                    "concepts": ["pendidikan"],
                                    "kind": "concept",
                                    "relation_traversal": False,
                                }],
                            }),
                        },
                    }],
                }).encode("utf-8")

        endpoint = "https://planner.example/v1/chat/completions"
        with patch("tjipto.retrieval.research.urlopen", return_value=Response()) as opener:
            proposal = OpenAICompatibleResearchPlanningProvider(
                "secret",
                model="gemini-model",
                endpoint=endpoint,
            ).propose({"query": "hak pendidikan", "intent": {"max_variants": 4}})

        payload = json.loads(opener.call_args.args[0].data.decode("utf-8"))
        content = payload["messages"][0]["content"]
        self.assertEqual(payload["response_format"]["type"], "json_schema")
        schema = payload["response_format"]["json_schema"]
        self.assertTrue(schema["strict"])
        self.assertEqual(schema["schema"]["properties"]["variants"]["maxItems"], 3)
        self.assertFalse(schema["schema"]["additionalProperties"])
        for fragment in (
            '"variants":',
            '"retrieval_lanes":',
            '"task_kind":',
            '"information_needs":',
            '"relation_traversal"',
            "at most 3 provider variants",
            "Never return requirements",
        ):
            self.assertIn(fragment, content)
        self.assertEqual(proposal["retrieval_lanes"], ["hybrid"])

    def test_openai_compatible_planner_retries_one_transient_http_failure(self) -> None:
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return None

            def read(self):
                return b'{"choices":[{"message":{"content":"{}"}}]}'

        endpoint = "https://planner.example/v1/chat/completions"
        transient = HTTPError(endpoint, 503, "unavailable", None, None)
        with patch("tjipto.retrieval.research.urlopen", side_effect=[transient, Response()]) as opener:
            proposal = OpenAICompatibleResearchPlanningProvider("secret", model="gemini-model", endpoint=endpoint).propose({})

        self.assertEqual(proposal, {})
        self.assertEqual(opener.call_count, 2)

    def test_coordinated_ordinals_preserve_reordered_instrument_scope(self) -> None:
        from tjipto.corpora.source_arbitration import source_roles_for_query

        for query in (
            "Perbedaan Perubahan Pertama dan Kedua UUD 1945",
            "Perbedaan Perubahan I dan II UUD 1945",
            "Kedua dan Perubahan Pertama UUD 1945",
        ):
            self.assertEqual(
                source_roles_for_query(query, strategy="uud", config=LegalRuntimeService()._store("uud").config),
                ("amendment_1_historical", "amendment_2_historical"),
                query,
            )

    def test_planner_rejects_malformed_requirement_values_without_authority(self) -> None:
        class Provider:
            def propose(self, request):
                return {
                    "task_kind": [],
                    "requirements": [
                        {"requirement_id": "bad", "retrieval_query": ["not", "text"]},
                        {"requirement_id": "also-bad", "required_entities": "not-a-sequence"},
                    ],
                }

        plan = plan_research("original", ResearchIntent(comparison=True), provider=Provider())
        self.assertFalse(plan.requirements)
        self.assertIn("task_kind_invalid", plan.rejection_reasons)
        self.assertIn("requirement_text_field_invalid", plan.rejection_reasons)
        self.assertIn("requirement_field_type_invalid", plan.rejection_reasons)

    def test_simple_queries_do_not_invoke_provider_or_decompose(self) -> None:
        class Provider:
            def propose(self, request):
                raise AssertionError("simple query must not call planner")

        plan = plan_research("exact query", ResearchIntent(), provider=Provider())
        self.assertEqual(tuple(variant.query for variant in plan.variants), ("exact query",))
        seen = []
        _, rows = execute_research("original", lambda query, variant: seen.append(query) or (), intent=ResearchIntent())
        self.assertEqual(seen, ["original"])
        self.assertEqual(rows, ())

    def test_exact_structured_query_bypasses_semantic_orchestrator(self) -> None:
        class Provider:
            calls = 0

            def propose(self, request):
                self.calls += 1
                raise AssertionError("exact legal target must bypass planner")

        provider = Provider()
        response = LegalRuntimeService(planning_provider=provider).ask("uud", "Pasal 7A bunyinya apa?")
        self.assertEqual(provider.calls, 0)
        self.assertNotEqual(response["status"], "insufficient_evidence")
        self.assertTrue(response["citations"])

    def test_service_binds_deployment_planner_when_no_test_provider_is_injected(self) -> None:
        provider = object()
        with patch("tjipto.runtime.service.research_planning_provider_from_environment", return_value=provider):
            service = LegalRuntimeService()
        self.assertIs(service._planning_provider, provider)

    def test_metadata_query_bypasses_semantic_orchestrator(self) -> None:
        class Provider:
            calls = 0

            def propose(self, request):
                self.calls += 1
                raise AssertionError("metadata lookup must bypass planner")

        provider = Provider()
        response = LegalRuntimeService(planning_provider=provider).ask("uud", "kapan perubahan pertama ditetapkan")
        self.assertEqual(provider.calls, 0)
        self.assertEqual(response["route"], "metadata_fact")

    def test_freeform_semantic_query_invokes_planner_and_keeps_verified_support_server_owned(self) -> None:
        class Provider:
            calls = 0

            def propose(self, request):
                self.calls += 1
                return {
                    "variants": [{"query": "hak pendidikan"}],
                    "information_needs": [{
                        "description": "hak atas pendidikan",
                        "query": "hak pendidikan",
                        "concepts": ["pendidikan"],
                    }],
                }

        provider = Provider()
        response = LegalRuntimeService(planning_provider=provider).ask(
            "uud", "Saya dilarang sekolah karena agama saya, hak konstitusional apa yang relevan?"
        )
        self.assertEqual(provider.calls, 1)
        self.assertEqual(response["research_plan"].provider_status, "accepted")
        self.assertTrue(any(row.get("citation") == "Pasal 31" for row in response["matches"]))
        self.assertEqual(response["status"], "clarification_required")
        self.assertEqual(response["sufficiency"]["status"], "complete")
        self.assertTrue(response["evidence_set"]["support_ids"])

    def test_planner_can_name_procedure_need_but_server_binds_corpus_requirements(self) -> None:
        class Provider:
            calls = 0

            def propose(self, request):
                self.calls += 1
                return {
                    "variants": [{"query": "prosedur pemberhentian Presiden"}],
                    "information_needs": [{
                        "description": "tahapan pemberhentian Presiden",
                        "query": "prosedur pemberhentian Presiden",
                        "concepts": ["pemberhentian"],
                        "kind": "procedure",
                    }],
                }

        provider = Provider()
        response = LegalRuntimeService(planning_provider=provider).ask("uud", "Bagaimana Presiden bisa diberhentikan?")
        self.assertEqual(provider.calls, 1)
        self.assertEqual(response["sufficiency"]["status"], "complete")
        self.assertEqual(
            set(response["sufficiency"]["fulfilled_requirement_ids"]),
            {"grounds", "procedure_basis", "constitutional_review", "assembly_decision"},
        )

    def test_planner_information_need_cannot_inject_authority_scope_or_evidence(self) -> None:
        class Provider:
            def propose(self, request):
                return {"information_needs": [{
                    "description": "forged",
                    "query": "forged",
                    "concepts": ["forged"],
                    "source_role": "historical",
                    "evidence_id": "forged",
                }]}

        plan = plan_research("pertanyaan bebas", ResearchIntent(orchestrate=True), provider=Provider())
        self.assertFalse(plan.information_needs)
        self.assertIn("information_need_invalid", plan.rejection_reasons)


if __name__ == "__main__":
    unittest.main()
