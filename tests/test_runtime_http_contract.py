from __future__ import annotations

import json
import os
from pathlib import Path
import threading
import unittest
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from tjipto.corpora.verified import VerifiedCorpusRepository
from tjipto.evidence.store import EvidenceStore
from tjipto.runtime.http import make_server
from tjipto.telemetry import Telemetry


ROOT = Path(__file__).resolve().parents[1]
_FORBIDDEN = (
    "evidence_id", "legal_unit_id", "source_document_id", "bbox_id", "source_bbox_refs",
    "manifest_digest", "artifact_set_digest", "context_pack", "source_role", '"route"',
    '"intent"', '"reason"', "reason_code", "pymupdf", "transform_version",
)


class RuntimeHttpContractTest(unittest.TestCase):
    server: Any
    thread: threading.Thread
    base_url: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = make_server(port=0, repo_root=ROOT)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        host, port = cls.server.server_address
        cls.base_url = f"http://{host}:{port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)
        cls.server = None
        cls.thread = None
        EvidenceStore.clear_shared_cache()
        VerifiedCorpusRepository.clear_shared_cache()

    def test_public_capabilities_and_search_are_closed(self) -> None:
        capabilities = self._get("/legal/uud/capabilities")
        self.assertEqual(capabilities, {"status": "ok", "capabilities": ["search", "ask", "citation", "viewer", "bookmarks"]})
        result = self._post("/legal/uud/search", {"query": "UUD 1945", "limit": 2})
        self.assertEqual(result["kind"], "document")
        self.assertEqual(result["status"], "found")
        self.assertTrue(result["results"])
        self._assert_public(result)
        self.assertEqual(result["results"][0]["document_role"], "Naskah Konsolidasi")
        self.assertNotIn("snippet", result["results"][0])
        self.assertIn("legal_identity", result["results"][0])
        self.assertTrue(result["results"][0]["viewer_target"]["public_target_id"])

    def test_regulation_catalog_filters_and_viewer_use_closed_public_payloads(self) -> None:
        facets = self._post("/legal/catalog/facets", {})
        self.assertEqual(facets["default_corpus"], "uud")
        self.assertEqual({facet["name"] for facet in facets["facets"]}, {"document_role", "legal_status", "establishment_period"})
        result = self._post("/legal/catalog/search", {"query": "gaji pppk", "filters": {"legal_status": "applicable"}})
        self.assertEqual(result["kind"], "catalog")
        self.assertEqual(result["total"], 2)
        self._assert_public(result)
        target = result["results"][0]["viewer_target"]["public_target_id"]
        viewer = self._post("/legal/catalog/viewer", {"target": target})
        self.assertEqual(viewer["kind"], "document")
        self.assertEqual(viewer["legal_status"], "Berlaku")
        self.assertEqual(viewer["legal_status_scope"], "Record peraturan")
        self.assertRegex(viewer["publication"], r"Lembaran Negara Republik Indonesia Tahun \d{4} Nomor \d+")
        self._assert_public(viewer)
        body, headers = self._get_bytes(viewer["pdf"]["access_url"])
        self.assertEqual(headers["Content-Type"], "application/pdf")
        self.assertTrue(body.startswith(b"%PDF"))

    def test_supported_overview_remains_an_answer_until_citation_click(self) -> None:
        for query in ("ringkasan UUD", "ringkasan UUD amandemen pertama", "Apa ringkasan BAB XA?"):
            result = self._post("/legal/uud/ask", {"query": query})
            self.assertEqual(result["kind"], "answer", query)
            self.assertTrue(result["supports"], query)
            self.assertTrue(result["supports"][0]["citation"]["text"], query)
            self.assertNotIn("document", result, query)

    def test_research_context_is_public_without_internal_scope_identifiers(self) -> None:
        result = self._post("/legal/uud/ask", {"query": "amandemen pertama vs kedua", "limit": 30})
        self.assertEqual(result["operation"], "compare")
        self.assertEqual(result["sufficiency"]["status"], "complete")
        self.assertEqual(len(result["source_scopes"]), 2)
        self.assertTrue(all(set(scope) == {"label"} for scope in result["source_scopes"]))
        self._assert_public(result)

    def test_rc2_scenario_manifest_is_complete_and_versioned(self) -> None:
        manifest = json.loads((ROOT / "tests/scenarios/public_evidence_rc2.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["version"], 3)
        for scenario in manifest["scenarios"]:
            self.assertEqual(set(scenario), {"id", "owner", "test", "assertions", "command"})
            self.assertTrue(scenario["id"] and scenario["assertions"] and scenario["command"])

    def test_ask_citation_viewer_pdf_and_bookmark_use_only_public_targets(self) -> None:
        asked = self._post("/legal/uud/ask", {"query": "Pasal 16 UUD konsolidasi"})
        self.assertEqual(asked["kind"], "answer")
        self.assertEqual(asked["status"], "answer_ready")
        self._assert_public(asked)
        support = asked["supports"][0]
        self.assertEqual(set(support), {
            "public_support_id", "authority_kind", "citation_final", "support_kind", "fact_kind", "label", "role_label",
            "text", "source_label", "source_status_label", "page_numbers", "viewer_target", "citation", "document",
        })
        self.assertEqual(support["authority_kind"], "legal_citation")
        self.assertTrue(support["citation_final"])
        self.assertNotIn("source_bbox_refs", json.dumps(support))
        target = support["viewer_target"]["public_target_id"]
        viewer = self._post("/legal/uud/viewer", {"target": target})
        self._assert_public(viewer)
        self.assertTrue(viewer["pdf_access_available"])
        self.assertTrue(viewer["bbox_rectangles"])
        self.assertTrue(viewer["pdf"]["access_url"].startswith("/legal/uud/pdf?target="))
        pdf, headers = self._get_bytes(viewer["pdf"]["access_url"])
        self.assertTrue(pdf.startswith(b"%PDF"))
        self.assertEqual(headers["Content-Type"], "application/pdf")

        saved = self._post("/legal/uud/bookmarks", {"target": target, "note": "cek lagi"})
        self.assertEqual(saved["status"], "saved")
        self._assert_public(saved)
        self.assertEqual(
            set(saved["bookmark"]),
            {"public_bookmark_id", "public_target_id", "note", "created_at", "status", "document"},
        )
        bookmarks = self._get("/legal/uud/bookmarks")
        self._assert_public(bookmarks)
        self.assertTrue(bookmarks["bookmarks"])
        public_bookmark_id = saved["bookmark"]["public_bookmark_id"]
        self.assertEqual(
            self._delete("/legal/uud/bookmarks", {"bookmark": public_bookmark_id}),
            {"status": "deleted", "public_bookmark_id": public_bookmark_id},
        )
        self.assertNotIn(public_bookmark_id, {
            row["public_bookmark_id"] for row in self._get("/legal/uud/bookmarks")["bookmarks"]
        })
        self.assertEqual(
            self._delete("/legal/uud/bookmarks", {"bookmark": public_bookmark_id}),
            {"status": "unavailable"},
        )

    def test_ask_search_bookmark_and_viewer_share_one_document_projection(self) -> None:
        asked = self._post("/legal/uud/ask", {"query": "Pasal 16 UUD konsolidasi"})
        support = asked["supports"][0]
        target = support["viewer_target"]["public_target_id"]
        viewer = self._post("/legal/uud/viewer", {"target": target})
        search = self._post(
            "/legal/catalog/search",
            {"query": "undang undang dasar negara republik indonesia tahun 1945", "limit": 1},
        )
        bookmark = self._post("/legal/uud/bookmarks", {"target": target})["bookmark"]
        projections = (support["document"], viewer["document"], search["results"][0], bookmark["document"])
        keys = ("legal_identity", "legal_status", "document_role", "issuer", "official_url")
        self.assertEqual(
            {tuple(projection[key] for key in keys) for projection in projections},
            {tuple(projections[0][key] for key in keys)},
        )
        self.assertEqual({len(projection["source_annotations"]) for projection in projections}, {4})
        self.assertTrue(
            all(
                set(annotation) == {"label", "text", "source_reference", "page_number"}
                for annotation in projections[0]["source_annotations"]
            )
        )

    def test_document_relations_and_provision_effects_remain_separate(self) -> None:
        result = self._post("/legal/catalog/search", {"query": "perpres 11 tahun 2024"})
        document = result["results"][0]
        self.assertTrue(document["relations"])
        self.assertTrue(document["provision_effects"])
        self.assertEqual(document["relations"][0]["label"], "Mengubah")
        self.assertEqual(document["provision_effects"][0]["label"], "Ketentuan yang diubah")
        self.assertNotEqual(document["relations"][0]["target"], document["provision_effects"][0]["target"])
        conflict = document["official_title_conflict"]
        self.assertEqual(conflict["state"], "Terselesaikan")
        self.assertEqual(len(conflict["values"]), 3)
        self.assertTrue(all(row["source_authority"] and row["source_reference"] for row in conflict["values"]))
        self._assert_public(document)

        amendment = self._post("/legal/catalog/search", {"query": "perubahan pertama UUD 1945"})["results"][0]
        self.assertEqual(amendment["establishment_place"], "Jakarta")
        self.assertEqual(amendment["establishment_date"], "19 Oktober 1999")
        self.assertEqual(amendment["issuer"], "Majelis Permusyawaratan Rakyat Republik Indonesia")
        self.assertEqual(amendment["legal_status"], "Berlaku")
        self.assertEqual(amendment["legal_status_scope"], "Record induk sumber")
        self.assertTrue(amendment["official_url"].endswith(".pdf"))
        self.assertTrue(amendment["signatories"])
        self.assertTrue(amendment["relations"])
        consolidated_relation = next(row for row in amendment["relations"] if row["label"] == "Digabungkan dalam")
        self.assertTrue(consolidated_relation["source_reference"].endswith("UUD45_SatuNaskah.pdf"))
        self.assertTrue(amendment["provision_effects"])
        self.assertTrue(
            all(not any(marker in effect["target"] for marker in ("Scope", "Clause")) for effect in amendment["provision_effects"])
        )

        consolidated = self._post("/legal/catalog/search", {"query": "naskah konsolidasi UUD 1945"})["results"][0]
        self.assertEqual(
            consolidated["title"],
            "Undang-Undang Dasar Negara Republik Indonesia Tahun 1945 dalam Satu Naskah",
        )
        self.assertEqual(
            consolidated["issuer"],
            "Sekretariat Jenderal Majelis Permusyawaratan Rakyat Republik Indonesia",
        )
        self.assertIn("dalam Satu Naskah", consolidated["publication"])
        self.assertIsNone(consolidated["establishment_date"])
        self.assertIsNone(consolidated["effective_date"])
        self.assertEqual(
            {relation["label"] for relation in consolidated["relations"]},
            {"Menggabungkan", "Berasal dari"},
        )
        self.assertFalse(consolidated["provision_effects"])

        original = self._post("/legal/catalog/search", {"query": "naskah asli UUD 1945"})["results"][0]
        self.assertEqual(original["document_role"], "Naskah Asli")
        self.assertIsNone(original["issuer"])
        self.assertIsNone(original["establishment_date"])
        self.assertIsNone(original["establishment_place"])
        self.assertIsNone(original["signatories"])
        amended_by = [row for row in original["relations"] if row["label"] == "Diubah oleh"]
        self.assertEqual(len(amended_by), 1)
        self.assertEqual(
            amended_by[0]["target"],
            "Perubahan Pertama Undang-Undang Dasar Negara Republik Indonesia Tahun 1945",
        )

        expected_chain = (
            ("perubahan pertama UUD 1945", "Perubahan Kedua Undang-Undang Dasar Negara Republik Indonesia Tahun 1945", 10),
            ("perubahan kedua UUD 1945", "Perubahan Ketiga Undang-Undang Dasar Negara Republik Indonesia Tahun 1945", 26),
            ("perubahan ketiga UUD 1945", "Perubahan Keempat Undang-Undang Dasar Negara Republik Indonesia Tahun 1945", 68),
        )
        for query, expected_target, expected_effects in expected_chain:
            chained = self._post("/legal/catalog/search", {"query": query})["results"][0]
            next_relation = next(row for row in chained["relations"] if row["label"] == "Diubah oleh")
            self.assertEqual(next_relation["target"], expected_target)
            self.assertEqual(len(chained["provision_effects"]), expected_effects)

        renumbered = self._post("/legal/catalog/search", {"query": "perubahan keempat UUD 1945"})["results"][0]
        self.assertIn(
            {"label": "Penomoran menjadi", "target": "Pasal 25A"},
            [{"label": effect["label"], "target": effect["target"]} for effect in renumbered["provision_effects"]],
        )

        deleted = renumbered
        self.assertEqual(len(deleted["provision_effects"]), 26)
        self.assertIn(
            {"label": "Ketentuan yang ditambahkan", "target": "Pasal 37 ayat (5)"},
            [{"label": effect["label"], "target": effect["target"]} for effect in deleted["provision_effects"]],
        )
        self.assertTrue(
            all(
                effect["verification_state"] == "Ruang lingkup naskah"
                for effect in deleted["provision_effects"]
                if effect["label"] == "Ketentuan yang diubah dan/atau ditambahkan"
            )
        )
        self.assertIn(
            {"label": "Ketentuan yang dihapus", "target": "Pasal 16"},
            [{"label": effect["label"], "target": effect["target"]} for effect in deleted["provision_effects"]],
        )

    def test_catalog_document_bookmark_uses_the_same_opaque_target_contract(self) -> None:
        result = self._post("/legal/catalog/search", {"query": "perpres 11 tahun 2024"})
        target = result["results"][0]["viewer_target"]["public_target_id"]
        saved = self._post("/legal/uud/bookmarks", {"target": target})
        self.assertEqual(saved["status"], "saved")
        self.assertEqual(saved["bookmark"]["public_target_id"], target)
        self.assertEqual(saved["bookmark"]["document"]["legal_identity"], result["results"][0]["legal_identity"])
        bookmark_id = saved["bookmark"]["public_bookmark_id"]
        self.assertEqual(
            self._delete("/legal/uud/bookmarks", {"bookmark": bookmark_id})["status"],
            "deleted",
        )

    def test_citation_shares_the_support_contract(self) -> None:
        result = self._post("/legal/uud/citation", {"query": "Pasal 1 ayat (3)"})
        self.assertEqual(result["status"], "found")
        self.assertTrue(result["supports"])
        self.assertEqual(result["supports"][0]["authority_kind"], "legal_citation")
        self.assertTrue(result["supports"][0]["citation_final"])
        self._assert_public(result)

    def test_public_support_omits_quote_presentation_fields(self) -> None:
        result = self._post("/legal/uud/ask", {"query": "Pasal 28A"})
        self.assertEqual(result["status"], "answer_ready")
        support = result["supports"][0]
        self.assertFalse({
            "copy_text", "layout_lines", "legal_citation_available", "relevant_quote_eligible", "panel_section",
        } & set(support))

    def test_public_support_exposes_typed_authority_and_finality(self) -> None:
        cases = (
            ("Pasal 16 UUD konsolidasi", "legal_citation", True),
            ("kapan perubahan pertama ditetapkan", "metadata_source", False),
            ("Apa isi BAB XI agama?", "structural_context", False),
            ("pasal yang dihapus", "instrument_provenance", True),
            ("Kenapa Amandemen 4 Aturan Tambahan ada Pasal III, tapi Satu Naskah Pasal II?", "source_anomaly", False),
        )
        for query, authority_kind, citation_final in cases:
            with self.subTest(query=query):
                supports = self._post("/legal/uud/ask", {"query": query})["supports"]
                support = next(row for row in supports if row["authority_kind"] == authority_kind)
                self.assertIs(support["citation_final"], citation_final)

    def test_clause_claim_viewer_uses_opaque_exact_overlay_target(self) -> None:
        result = self._post(
            "/legal/uud/ask",
            {"query": "Pasal 7C menyebut Presiden tidak dapat membekukan dan/atau membubarkan Dewan Perwakilan Rakyat?"},
        )
        support = result["supports"][0]
        viewer = self._post("/legal/uud/viewer", {"target": support["viewer_target"]["public_target_id"]})
        self.assertEqual(viewer["status"], "viewer_payload_ready")
        self.assertEqual(viewer["quoted_text"], support["text"])
        self.assertTrue(viewer["bbox_rectangles"])
        self._assert_public(viewer)

    def test_every_first_amendment_relation_support_has_its_own_pdf_target(self) -> None:
        result = self._post("/legal/uud/ask", {"query": "pasal apa saja yang diubah perubahan pertama"})
        relation_supports = [row for row in result["supports"] if row["support_kind"] == "article_relation"]
        self.assertGreater(len(relation_supports), 1)
        targets = []
        for support in relation_supports:
            target = support["viewer_target"]["public_target_id"]
            self.assertTrue(support["viewer_target"]["can_resolve"], support["label"])
            self.assertIsNotNone(target, support["label"])
            viewer = self._post("/legal/uud/viewer", {"target": target})
            self.assertTrue(viewer["pdf_access_available"], support["label"])
            self.assertTrue(viewer["rendering_available"], support["label"])
            self.assertEqual(viewer["page_numbers"], support["page_numbers"], support["label"])
            self.assertTrue(viewer["bbox_rectangles"], support["label"])
            targets.append(target)
        self.assertEqual(len(targets), len(set(targets)))
        groups = [row for row in result["support_groups"] if row["group_kind"] == "article_relation_members"]
        self.assertEqual(sum(row["member_count"] for row in groups), len(relation_supports))

    def test_groups_preserve_members_and_keep_nonlegal_sections_out_of_quotes(self) -> None:
        result = self._post("/legal/uud/ask", {"query": "siapa wakil ketua yang tercantum dalam Perubahan Pertama?"})
        self.assertTrue(result["support_groups"])
        self._assert_public(result)
        for group in result["support_groups"]:
            self.assertEqual(group["member_count"], len(group["members"]))
            self.assertTrue(group["public_group_id"])
            for member in group["members"]:
                self.assertIn(member, result["supports"])
                self.assertEqual(member["authority_kind"], "metadata_source")
                self.assertFalse(member["citation_final"])
        group = result["support_groups"][0]
        self.assertEqual(group["group_kind"], "role_members")
        self.assertEqual(group["member_count"], 7)

    def test_entity_occurrences_use_one_group_without_losing_exact_members(self) -> None:
        result = self._post("/legal/uud/ask", {"query": "Amien Rais"})
        self.assertEqual(result["kind"], "answer")
        self.assertEqual(len(result["support_groups"]), 1)
        group = result["support_groups"][0]
        self.assertEqual(group["group_kind"], "entity_occurrences")
        self.assertEqual(group["member_count"], 4)
        self.assertEqual(len({row["viewer_target"]["public_target_id"] for row in group["members"]}), 4)

    def test_document_result_is_distinct_from_evidence_supports(self) -> None:
        result = self._post("/legal/uud/ask", {"query": "Buka naskah Perubahan Pertama UUD"})
        self.assertEqual(set(result), {"kind", "status", "document"})
        self.assertEqual(result["kind"], "document")
        self.assertTrue(result["document"]["viewer_target"]["can_resolve"])

    def test_public_payload_rejects_unknown_and_legacy_fields(self) -> None:
        for path, payload in (
            ("/legal/uud/viewer", {"target": "bad", "evidence_id": "forged"}),
            ("/legal/uud/bookmarks", {"evidence_id": "forged"}),
            ("/legal/uud/ask", {"query": "Pasal 1", "context_pack": {}}),
            ("/legal/uud/ask", {"query": "Pasal 1", "clarification_context": "retired"}),
        ):
            with self.assertRaises(HTTPError) as error:
                self._post(path, payload)
            self.assertEqual(error.exception.code, 400)
            self.assertEqual(json.loads(error.exception.read().decode("utf-8")), {"status": "bad_request"})

    def test_invalid_target_and_pdf_query_do_not_leak(self) -> None:
        viewer = self._post("/legal/uud/viewer", {"target": "not-a-target"})
        self.assertEqual(viewer["status"], "not_found")
        self._assert_public(viewer)
        with self.assertRaises(HTTPError) as error:
            self._get("/legal/uud/pdf?evidence_id=forged")
        self.assertEqual(error.exception.code, 400)
        self._assert_public(json.loads(error.exception.read().decode("utf-8")))

    def test_transport_rejects_invalid_json_oversized_body_and_unknown_routes(self) -> None:
        for data, headers, expected in (
            (b"{", {"Content-Type": "application/json"}, (400, "bad_request")),
            (b"x" * (64 * 1024 + 1), {"Content-Type": "application/json", "Content-Length": str(64 * 1024 + 1)}, (413, "payload_too_large")),
        ):
            request = Request(self.base_url + "/legal/uud/ask", data=data, headers=headers, method="POST")
            with self.assertRaises(HTTPError) as error:
                urlopen(request, timeout=10)  # nosec B310
            self.assertEqual(error.exception.code, expected[0])
            self.assertEqual(json.loads(error.exception.read().decode("utf-8")), {"status": expected[1]})
        with self.assertRaises(HTTPError) as error:
            self._get("/not-real")
        self.assertEqual(error.exception.code, 404)

    def test_telemetry_uses_closed_routes_and_unknown_corpus_sentinel(self) -> None:
        records: list[dict] = []
        telemetry = Telemetry(records.append)
        handler = self.server.RequestHandlerClass
        previous_handler_telemetry = handler.telemetry
        previous_service_telemetry = handler.runtime_service.telemetry
        handler.telemetry = telemetry
        handler.runtime_service.telemetry = telemetry
        try:
            self._post("/legal/external-corpus/ask", {"query": "Pasal 1"})
            with self.assertRaises(HTTPError):
                self._get("/legal/external-corpus/arbitrary-route")
        finally:
            handler.telemetry = previous_handler_telemetry
            handler.runtime_service.telemetry = previous_service_telemetry
        serialized = json.dumps(records)
        self.assertNotIn("external-corpus", serialized)
        self.assertNotIn("arbitrary-route", serialized)
        self.assertIn({"event": "integrity_failure", "attributes": {"corpus_id": "unknown", "reason_code": "unknown_corpus"}}, records)
        self.assertTrue(any(row["event"] == "http_request" and row["attributes"]["route"] == "legal.ask" for row in records))
        self.assertTrue(any(row["event"] == "http_request" and row["attributes"]["route"] == "not_found" for row in records))

    def test_local_cors_and_development_alias_are_explicit(self) -> None:
        request = Request(self.base_url + "/legal/uud/ask", data=b'{"query":"Pasal 1"}', headers={"Content-Type": "application/json", "Origin": "http://localhost:5173"}, method="POST")
        with urlopen(request, timeout=10) as response:  # nosec B310
            self.assertEqual(response.headers["Access-Control-Allow-Origin"], "http://localhost:5173")
        self.assertEqual(self._get("/uud/capabilities")["status"], "ok")
        original = os.environ.get("TJIPTO_MODE")
        os.environ["TJIPTO_MODE"] = "staging"
        try:
            with self.assertRaises(HTTPError) as error:
                self._get("/uud/capabilities")
            self.assertEqual(error.exception.code, 404)
        finally:
            if original is None:
                os.environ.pop("TJIPTO_MODE", None)
            else:
                os.environ["TJIPTO_MODE"] = original

    def _assert_public(self, payload: object) -> None:
        body = json.dumps(payload)
        for forbidden in _FORBIDDEN:
            self.assertNotIn(forbidden, body)
        self.assertNotIn(str(ROOT), body)
        self.assertNotIn("Traceback", body)

    def _get(self, path: str) -> dict:
        with urlopen(self.base_url + path, timeout=10) as response:  # nosec B310
            return json.loads(response.read().decode("utf-8"))

    def _get_bytes(self, path: str) -> tuple[bytes, Any]:
        with urlopen(self.base_url + path, timeout=10) as response:  # nosec B310
            return response.read(), response.headers

    def _post(self, path: str, payload: dict) -> dict:
        request = Request(self.base_url + path, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(request, timeout=10) as response:  # nosec B310
            return json.loads(response.read().decode("utf-8"))

    def _delete(self, path: str, payload: dict) -> dict:
        request = Request(
            self.base_url + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="DELETE",
        )
        with urlopen(request, timeout=10) as response:  # nosec B310
            return json.loads(response.read().decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
