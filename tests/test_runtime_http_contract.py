from __future__ import annotations

import json
import os
import threading
import unittest
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from tjipto.runtime.http import make_server


ROOT = Path(__file__).resolve().parents[1]


def _http_ask_cases() -> tuple[dict, ...]:
    path = ROOT / "tests/fixtures/uud/http_ask_cases.jsonl"
    return tuple(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


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

    def test_health(self) -> None:
        self.assertEqual(self._get("/health")["status"], "ok")

    def test_exposed_runtime_endpoints(self) -> None:
        capabilities = self._get("/legal/uud/capabilities")
        self.assertEqual(capabilities["status"], "ok")
        self.assertIn("search", capabilities["capabilities"])

        search = self._post("/legal/uud/search", {"query": "UUD 1945", "limit": 2})
        self.assertEqual(search["status"], "found")
        self.assertTrue(search["results"])
        for internal in ("matches", "context_pack", "route", "intent", "ranked_final_evidence_ids"):
            self.assertNotIn(internal, search)
        first = search["results"][0]
        self.assertEqual(first["status"], "document")
        self.assertTrue(first["document_id"])
        self.assertTrue(first["source_document_id"])
        self.assertIn("viewer_ref", first)
        self.assertIn("source_role", first)
        self.assertFalse(first["viewer_ref"]["bbox_count"])
        self.assertNotIn("route_score", first)
        self.assertNotIn("source_sha256", first)
        self.assertNotIn("source_pdf_path", first)
        self.assertNotIn("source_sha256", first["viewer_ref"])
        self.assertNotIn("source_pdf_path", first["viewer_ref"])

        weak = self._post("/legal/uud/search", {"query": "hak pendidikan"})
        self.assertEqual(weak["public_status"], "no_results")
        self.assertEqual(weak["results"], [])

        citation = self._post("/legal/uud/citation", {"query": "Pasal 1 ayat (3)"})
        self.assertEqual(citation["status"], "found")
        self.assertNotIn("matches", citation)
        self.assertNotIn("context_pack", citation)
        self.assertEqual(citation["citation_payloads"][0]["authority_kind"], "legal_citation")
        self.assertTrue(citation["citation_payloads"][0]["citation_final"])
        self.assertNotIn("source_sha256", citation["citation_payloads"][0])
        self.assertNotIn("source_pdf_path", citation["citation_payloads"][0])
        self.assertNotIn("source_sha256", citation["viewer_refs"][0])
        self.assertNotIn("source_pdf_path", citation["viewer_refs"][0])
        evidence_id = citation["citation_payloads"][0]["evidence_id"]

        viewer = self._post("/legal/uud/viewer", {"evidence_id": evidence_id})
        self.assertEqual(viewer["status"], "viewer_payload_ready")
        self.assertEqual(viewer["evidence_id"], evidence_id)
        self.assertTrue(viewer["pdf_access_available"])
        self.assertEqual(viewer["render_status"], "pdf_access_available")
        self.assertTrue(viewer["pdf"]["access_url"].startswith("/legal/uud/pdf?"))
        self.assertNotIn("source_sha256", viewer)
        self.assertNotIn("source_pdf_path", viewer)
        self.assertNotIn("source_sha256", viewer["pdf"]["access_url"])
        self.assertNotIn("source_pdf_path", viewer["pdf"]["access_url"])
        self.assertNotIn("data_url", viewer["pdf"])
        self.assertTrue(viewer["bbox_rectangles"])
        self.assertEqual(viewer["bbox_rectangles"][0]["bbox_precision"], "exact")
        self.assertTrue(viewer["bbox_rectangles"][0]["viewer_highlightable"])
        self.assertNotIn("source_pdf_path", viewer["bbox_rectangles"][0])
        self.assertNotIn("source_sha256", viewer["bbox_rectangles"][0])
        self.assertNotIn("source_document_id", viewer["bbox_rectangles"][0])
        self.assertNotIn("evidence_id", viewer["bbox_rectangles"][0])
        pdf_body, pdf_headers = self._get_bytes(viewer["pdf"]["access_url"])
        self.assertEqual(pdf_headers["Content-Type"], "application/pdf")
        self.assertTrue(pdf_body.startswith(b"%PDF"))

        document_viewer = self._post("/legal/uud/viewer", {"source_document_id": first["source_document_id"]})
        self.assertEqual(document_viewer["status"], "viewer_payload_ready")
        self.assertTrue(document_viewer["pdf_access_available"])
        self.assertFalse(document_viewer["bbox_rectangles"])
        self.assertFalse(document_viewer["viewer_highlightable"])

        saved = self._post("/legal/uud/bookmarks", {"evidence_id": evidence_id, "note": "cek lagi"})
        self.assertEqual(saved["status"], "saved")
        self.assertNotIn("quoted_text", saved["bookmark"])
        bookmarks = self._get("/legal/uud/bookmarks")
        self.assertEqual(bookmarks["persistence"], "memory")
        self.assertEqual(bookmarks["persistence_label"], "temporary_process_memory")
        self.assertTrue(bookmarks["bookmarks"])

        self.assertEqual(self._post("/legal/unknown/search", {"query": "Pasal 1"})["status"], "unsupported_corpus")

    def test_uud_routes_are_dev_aliases_not_canonical_contract(self) -> None:
        canonical = self._get("/legal/uud/capabilities")
        alias = self._get("/uud/capabilities")
        self.assertEqual(alias["status"], canonical["status"])
        self.assertEqual(alias["capabilities"], canonical["capabilities"])

    def test_uud_alias_is_disabled_outside_development(self) -> None:
        old = os.environ.get("TJIPTO_MODE")
        os.environ["TJIPTO_MODE"] = "staging"
        try:
            with self.assertRaises(HTTPError) as error:
                self._get("/uud/capabilities")
            self.assertEqual(error.exception.code, 404)
        finally:
            if old is None:
                os.environ.pop("TJIPTO_MODE", None)
            else:
                os.environ["TJIPTO_MODE"] = old

    def test_uud_ask_examples(self) -> None:
        for case in _http_ask_cases():
            result = self._post("/legal/uud/ask", {"query": case["query"]})
            self.assertEqual(result["status"], case["status"], case["query"])
            self.assertEqual(result["route"], case["route"], case["query"])
            if "intent" in case:
                self.assertEqual(result["intent"], case["intent"], case["query"])
            if "has_citations" in case:
                if case["has_citations"] and not result["citations"]:
                    self.assertTrue(
                        result.get("metadata_support") or result.get("historical_citations") or result.get("structural_support"),
                        case["query"],
                    )
                else:
                    self.assertEqual(bool(result["citations"]), case["has_citations"], case["query"])
            if "has_viewer_refs" in case:
                if case["has_viewer_refs"] and not result["viewer_refs"]:
                    self.assertTrue(
                        any(row.get("viewer_ref", {}).get("can_resolve") for row in result.get("metadata_support", ())),
                        case["query"],
                    )
                else:
                    self.assertEqual(bool(result["viewer_refs"]), case["has_viewer_refs"], case["query"])
            if "public_keys" in case:
                self.assertTrue(set(case["public_keys"]) <= set(result), case["query"])
                self.assertTrue({"final_citations", "historical_citations", "structural_support"} <= set(result), case["query"])
            for field in case.get("absent_top_level", ()):
                self.assertNotIn(field, result, case["query"])
            for field in case.get("absent_citation_fields", ()):
                self.assertNotIn(field, result["citations"][0], case["query"])
            for field in case.get("absent_viewer_ref_fields", ()):
                self.assertNotIn(field, result["viewer_refs"][0], case["query"])
            if "metadata_field" in case:
                self.assertEqual(result["metadata_facts"][0]["field"], case["metadata_field"], case["query"])
            if "metadata_answer" in case:
                self.assertEqual(result["metadata_facts"][0]["answer"], case["metadata_answer"], case["query"])
            if case["route"] == "metadata_fact" and case["status"] == "answer_ready":
                self.assertTrue(result["metadata_support"], case["query"])
                support = result["metadata_support"][0]
                if support["support_class"] == "exact_metadata_citation":
                    self.assertFalse(result["citations"], case["query"])
                    self.assertFalse(result["viewer_refs"], case["query"])
                    self.assertTrue(support["citation_available"], case["query"])
                    self.assertTrue(support["viewer_highlightable"], case["query"])
                    self.assertTrue(support["viewer_ref"]["can_resolve"], case["query"])
                    self.assertEqual(support["authority_kind"], "metadata_source", case["query"])
                    self.assertFalse(support["citation_final"], case["query"])
                    self.assertNotIn("source_sha256", json.dumps(support), case["query"])
                    self.assertNotIn("source_pdf_path", json.dumps(support), case["query"])
                else:
                    self.assertFalse(support["citation_available"], case["query"])
                    self.assertFalse(support["viewer_highlightable"], case["query"])
                    self.assertIsNone(support["viewer_ref"], case["query"])
                    self.assertEqual(support["authority_kind"], "metadata_trace", case["query"])
            if "relation_target_labels" in case:
                self.assertEqual(
                    {row["target_label"] for row in result["legal_relations"]},
                    set(case["relation_target_labels"]),
                    case["query"],
                )
            for field in ("citations", "viewer_refs", "insufficient_reasons"):
                if field in case:
                    self.assertEqual(result[field], case[field], case["query"])

    def test_penandatangan_alias_uses_metadata_public_contract(self) -> None:
        result = self._post("/legal/uud/ask", {"query": "penandatangan perubahan pertama UUD"})
        self.assertEqual(result["status"], "answer_ready")
        self.assertEqual(result["route"], "metadata_fact")
        self.assertEqual(result["intent"], "metadata_lookup")
        self.assertEqual(result["metadata_facts"][0]["field"], "signatories")
        self.assertFalse(result["citations"])
        self.assertFalse(result["viewer_refs"])
        self.assertTrue(result["metadata_support"])
        self.assertEqual(result["metadata_support"][0]["authority_kind"], "metadata_source")
        self.assertFalse(result["metadata_support"][0]["citation_final"])

    def test_unscoped_metadata_public_contract_requires_clarification(self) -> None:
        result = self._post("/legal/uud/ask", {"query": "penandatangan UUD"})
        self.assertEqual(result["status"], "clarification_required")
        self.assertEqual(result["route"], "metadata_fact")
        self.assertTrue(result["clarification_options"])
        self.assertFalse(result["citations"])
        self.assertFalse(result["viewer_refs"])
        self.assertFalse(result["metadata_facts"])

    def test_unresolved_temporal_scope_public_contract_is_fail_closed(self) -> None:
        result = self._post("/legal/uud/ask", {"query": "Pasal 31 perubahan ke-5"})
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertIn("unresolved_source_scope", result["insufficient_reasons"])
        self.assertFalse(result["citations"])
        self.assertFalse(result["viewer_refs"])

    def test_local_dev_cors_only(self) -> None:
        request = Request(
            self.base_url + "/legal/uud/ask",
            data=json.dumps({"query": "Pasal 1 ayat (3)"}).encode("utf-8"),
            headers={"Content-Type": "application/json", "Origin": "http://localhost:5173"},
            method="POST",
        )
        with self._open_local(request) as response:
            self.assertEqual(response.headers["Access-Control-Allow-Origin"], "http://localhost:5173")
            self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")
            self.assertEqual(response.headers["Referrer-Policy"], "no-referrer")

        request = Request(
            self.base_url + "/legal/uud/ask",
            data=json.dumps({"query": "Pasal 1 ayat (3)"}).encode("utf-8"),
            headers={"Content-Type": "application/json", "Origin": "https://example.com"},
            method="POST",
        )
        with self._open_local(request) as response:
            self.assertIsNone(response.headers.get("Access-Control-Allow-Origin"))

    def test_unknown_path(self) -> None:
        with self.assertRaises(HTTPError) as error:
            self._get("/not-real")
        self.assertEqual(error.exception.code, 404)

    def test_invalid_json_and_oversized_payloads_fail_safely(self) -> None:
        request = Request(
            self.base_url + "/legal/uud/ask",
            data=b"{",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.assertRaises(HTTPError) as error:
            self._open_local(request)
        self.assertEqual(error.exception.code, 400)
        self.assertEqual(json.loads(error.exception.read().decode("utf-8"))["reason"], "invalid_json")

        request = Request(
            self.base_url + "/legal/uud/ask",
            data=b"x" * (64 * 1024 + 1),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.assertRaises(HTTPError) as error:
            self._open_local(request)
        self.assertEqual(error.exception.code, 413)
        self.assertEqual(json.loads(error.exception.read().decode("utf-8"))["reason"], "request_body_too_large")

        request = Request(
            self.base_url + "/legal/uud/search",
            data=json.dumps({"query": "x", "limit": 0}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.assertRaises(HTTPError) as error:
            self._open_local(request)
        self.assertEqual(error.exception.code, 400)
        self.assertEqual(json.loads(error.exception.read().decode("utf-8"))["reason"], "invalid_limit")

    def test_viewer_invalid_inputs_do_not_leak_paths_or_traces(self) -> None:
        citation = self._post("/legal/uud/citation", {"query": "Pasal 1 ayat (3)"})
        evidence_id = citation["citation_payloads"][0]["evidence_id"]
        for payload in (
            {"evidence_id": evidence_id, "source_document_id": "uud::missing"},
            {"evidence_id": evidence_id, "page_number": 999},
            {"evidence_id": evidence_id, "bbox_id": "missing_bbox"},
            {"evidence_id": evidence_id, "source_pdf_path": "../secret.pdf"},
        ):
            viewer = self._post("/legal/uud/viewer", payload)
            self.assertEqual(viewer["status"], "viewer_payload_ready")
            self.assertFalse(viewer["rendering_available"])
            body = json.dumps(viewer)
            self.assertNotIn(str(ROOT), body)
            self.assertNotIn("Traceback", body)
        self.assertEqual(
            self._post("/legal/unknown/viewer", {"evidence_id": evidence_id})["status"],
            "unsupported_corpus",
        )

        viewer = self._post("/legal/uud/viewer", {"evidence_id": evidence_id})
        forged = viewer["pdf"]["access_url"] + "&source_sha256=" + ("0" * 64)
        with self.assertRaises(HTTPError) as error:
            self._get(forged)
        self.assertEqual(error.exception.code, 404)
        body = error.exception.read().decode("utf-8")
        self.assertNotIn(str(ROOT), body)
        self.assertNotIn("Traceback", body)

        with self.assertRaises(HTTPError) as error:
            self._get(viewer["pdf"]["access_url"] + "&source_pdf_path=../secret.pdf")
        self.assertEqual(error.exception.code, 404)

    def _get(self, path: str) -> dict:
        with self._open_local(self.base_url + path) as response:
            return json.loads(response.read().decode("utf-8"))

    def _get_bytes(self, path: str) -> tuple[bytes, Any]:
        with self._open_local(self.base_url + path) as response:
            return response.read(), response.headers

    def _post(self, path: str, payload: dict) -> dict:
        request = Request(
            self.base_url + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self._open_local(request) as response:
            return json.loads(response.read().decode("utf-8"))

    def _open_local(self, request_or_url):
        return urlopen(request_or_url, timeout=10)  # nosec B310


if __name__ == "__main__":
    unittest.main()
