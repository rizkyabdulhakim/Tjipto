from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tjipto.retrieval.answer import candidate_eligibility, publication_eligibility, validate_answer_candidate
from tjipto.retrieval.research import ResearchPlan, TaskPlan
from tjipto.retrieval.router import route_retrieval
from tjipto.runtime.orchestration import OrchestrationController
from tjipto.runtime.service import LegalRuntimeService
from tjipto.runtime.taxonomy import RESULT_REASON_CODES, ResultReason, reason_for_status
from tjipto.runtime.tools import ToolRegistry, ToolRegistryError
from tjipto.telemetry import Telemetry


class ControlPlaneContractTests(unittest.TestCase):
    def test_task_plan_is_the_single_plan_contract(self):
        self.assertIs(TaskPlan, ResearchPlan)

    def test_server_tool_registry_rejects_unknown_and_invalid_requests(self):
        registry = ToolRegistry.default()
        self.assertEqual(registry.names, ("retrieval_dispatch",))
        self.assertIn("retrieval_dispatch", registry.names)
        self.assertNotIn("corpus_inspector", registry.names)
        self.assertTrue(all(callable(item.handler) for item in registry.definitions))
        self.assertEqual(registry.validate(("retrieval_dispatch",)), ("retrieval_dispatch",))
        with self.assertRaises(ToolRegistryError):
            registry.validate(("made_up_tool",))
        with self.assertRaises(ToolRegistryError):
            registry.validate(("",))
        with self.assertRaises(ToolRegistryError):
            registry.validate(None)
        with self.assertRaises(ToolRegistryError):
            registry.validate(())
        with self.assertRaises(ToolRegistryError):
            registry.validate(("retrieval_dispatch", "retrieval_dispatch"))
        self.assertIs(registry.require("retrieval_dispatch").handler, route_retrieval)

    def test_production_retrieval_dispatch_uses_registry_validation(self):
        service = LegalRuntimeService(answer_provider=None, planning_provider=None)
        store = SimpleNamespace()
        routed = {"route": "sparse", "status": "no_results", "matches": (), "dense_configured": False}
        with patch.object(ToolRegistry, "invoke", return_value=routed) as invoke:
            self.assertEqual(service._orchestrator.route_retrieval("uud", "query", store), routed)
        invoke.assert_called_once_with("retrieval_dispatch", "uud", "query", store)

    def test_result_taxonomy_is_complete_without_changing_public_statuses(self):
        required = {
            "answer_ready",
            "clarification_required",
            "partial_answer",
            "insufficient_source_evidence",
            "representation_failure",
            "corpus_gap",
            "capability_unavailable",
            "provider_unavailable",
            "publication_blocked",
        }
        self.assertTrue(required <= RESULT_REASON_CODES)
        self.assertEqual(reason_for_status("limited_answer"), ResultReason.PARTIAL_ANSWER)
        self.assertEqual(reason_for_status("unknown_status"), ResultReason.REPRESENTATION_FAILURE)

    def test_orchestrator_has_no_service_private_control_dependency(self):
        source = (Path(__file__).parents[1] / "src/tjipto/runtime/orchestration.py").read_text(encoding="utf-8")
        self.assertNotIn("LegalRuntimeService", source)
        self.assertNotIn("service._", source)
        service = LegalRuntimeService(answer_provider=None, planning_provider=None)
        controller = service._orchestrator
        self.assertIsInstance(controller, OrchestrationController)
        for name in (
            "store",
            "route_retrieval",
            "research",
            "ask_raw",
            "ask",
            "clarification_response",
            "resume_clarification",
        ):
            self.assertIsNot(getattr(controller, name).__self__, service)
        self.assertFalse(any(value is service for value in controller.__dict__.values()))
        self.assertIn("reason_for_status", source)

    def test_planning_provider_compatibility_seam_proxies_controller_owner(self):
        first = object()
        second = object()
        service = LegalRuntimeService(answer_provider=None, planning_provider=first)
        self.assertIs(service._planning_provider, first)
        self.assertIs(service._orchestrator.planning_provider, first)
        service._planning_provider = second
        self.assertIs(service._planning_provider, second)
        self.assertIs(service._orchestrator.planning_provider, second)

    def test_telemetry_compatibility_seam_proxies_controller_owner(self):
        service = LegalRuntimeService(answer_provider=None, planning_provider=None)
        first = service.telemetry
        self.assertIs(first, service._orchestrator.telemetry)
        replacement = Telemetry()
        service.telemetry = replacement
        self.assertIs(service.telemetry, replacement)
        self.assertIs(service.telemetry, service._orchestrator.telemetry)
        self.assertIs(replacement._registry, service.registry)

    def test_candidate_and_publication_gates_preserve_combined_decision(self):
        store = SimpleNamespace(
            lineage_error=lambda _row: None,
            bboxes_for=lambda _evidence_id: ({"bbox_id": "bbox-1"},),
            legal_units=(),
            chunks=(),
            retrieval_units=(),
        )
        row = {
            "evidence_id": "evidence-1",
            "runtime_loadable": True,
            "bbox_precision": "exact",
            "viewer_highlightable": True,
            "text_span_ids": ("span-1",),
            "bbox_ids": ("bbox-1",),
            "route_sources": ("bm25",),
            "status": "final",
            "page_numbers": (1,),
            "citation": "Pasal 1",
            "quoted_text": "teks",
            "source_pdf_path": "source.pdf",
            "source_sha256": "sha",
            "source_role": "normative",
            "temporal_context": "current",
        }
        candidate = candidate_eligibility(store, row)
        publication = publication_eligibility(store, row)
        self.assertEqual(candidate, (True, "candidate"))
        self.assertEqual(publication, (True, "answer_evidence"))
        self.assertEqual(validate_answer_candidate(store, row), publication)
        self.assertEqual(
            validate_answer_candidate(store, row | {"status": "draft"}),
            publication_eligibility(store, row | {"status": "draft"}),
        )


if __name__ == "__main__":
    unittest.main()
