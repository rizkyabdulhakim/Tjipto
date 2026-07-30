from __future__ import annotations

import json
from pathlib import Path
import inspect
import unittest

from tjipto.catalog import CatalogService
from tjipto.corpora.regulations.catalog import documents as regulation_documents
from tjipto.runtime.api import BadRequest, handle_catalog_request
from tjipto.runtime.service import LegalRuntimeService


ROOT = Path(__file__).resolve().parents[1]


class RegulationCatalogContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.service = LegalRuntimeService(ROOT)

    def test_current_constitution_is_not_truncated_behind_historical_ties(self) -> None:
        result = self.service.catalog_search("undang undang dasar negara republik indonesia tahun 1945", 5)
        self.assertEqual(result["total"], 6)
        self.assertEqual(result["results"][0]["document_role"], "Naskah Berlaku")
        self.assertEqual(len({row["public_target_id"] for row in result["results"]}), len(result["results"]))

    def test_explicit_historical_query_prioritizes_requested_source(self) -> None:
        result = self.service.catalog_search("perubahan pertama uud 1945", 5)
        self.assertEqual(result["results"][0]["short_title"], "Perubahan Pertama UUD 1945")

    def test_pilot_is_search_and_view_only_with_verified_inverse_and_pdf_effect(self) -> None:
        pilot = regulation_documents(ROOT)
        self.assertEqual(len(pilot), 2)
        self.assertTrue(all(document.permissions == frozenset({"catalog", "view"}) for document in pilot))
        relations = {relation for document in pilot for relation in document.relations}
        descriptors = {(relation.relation, relation.source_document_id, relation.target_document_id) for relation in relations}
        self.assertTrue(relations)
        self.assertTrue(
            all((relation.relation.inverse, relation.target_document_id, relation.source_document_id) in descriptors for relation in relations)
        )
        effects = tuple(effect for document in pilot for effect in document.provision_effects)
        self.assertEqual(len(effects), 1)
        self.assertEqual(effects[0].provenance.selector, effects[0].exact_source_text)

    def test_facets_are_server_derived_and_invalid_facets_fail_closed(self) -> None:
        result = handle_catalog_request("search", {"query": "", "filters": {"legal_status": "applicable"}}, service=self.service)
        self.assertTrue(result["results"])
        self.assertEqual({facet["name"] for facet in result["facets"]}, {"document_role", "legal_status", "establishment_period"})
        with self.assertRaises(BadRequest):
            handle_catalog_request("search", {"query": "", "filters": {"corpus": "uud"}}, service=self.service)

    def test_public_catalog_response_is_allowlisted(self) -> None:
        result = handle_catalog_request("search", {"query": "gaji pppk"}, service=self.service)
        forbidden = {
            "evidence_id", "legal_unit_id", "source_document_id", "bbox_refs", "source_role",
            "route", "intent", "reason", "reason_code", "manifest_digest", "artifact_set_digest",
        }
        self.assertFalse(forbidden & _keys(result))
        self.assertNotIn(str(ROOT), json.dumps(result))

    def test_acquisition_record_has_required_audit_fields(self) -> None:
        records = json.loads((ROOT / "data/catalog/regulations.json").read_text(encoding="utf-8"))["documents"]
        required = {
            "path", "retrieval_time", "redirect_chain", "mime_type", "file_size", "sha256",
            "page_count", "source_authority", "reviewer_decision",
        }
        self.assertTrue(all(required <= set(record["acquisition"]) for record in records))
        self.assertTrue(all(record["cross_check"]["reference"] and "discrepancies" in record["cross_check"] for record in records))

    def test_catalog_path_has_no_answer_retrieval_or_graph_dependency(self) -> None:
        source = inspect.getsource(CatalogService).casefold()
        for forbidden in ("llm", "dense", "rerank", "retrieval", "graph"):
            self.assertNotIn(forbidden, source)

    def test_generic_frontend_contains_no_corpus_name(self) -> None:
        source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (ROOT / "apps/web/src").rglob("*")
            if path.suffix in {".ts", ".tsx"}
        ).casefold()
        self.assertNotIn("uud", source)


def _keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | set().union(*(_keys(item) for item in value.values()), set())
    if isinstance(value, (list, tuple)):
        return set().union(*(_keys(item) for item in value), set())
    return set()


if __name__ == "__main__":
    unittest.main()
