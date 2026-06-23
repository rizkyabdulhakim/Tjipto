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


class RuntimeHttpContractTest(unittest.TestCase):
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

        search = self._post("/legal/uud/search", {"query": "negara hukum", "limit": 2})
        self.assertEqual(search["status"], "found")
        self.assertTrue(search["results"])
        for internal in ("matches", "context_pack", "route", "intent", "ranked_final_evidence_ids"):
            self.assertNotIn(internal, search)
        first = search["results"][0]
        self.assertTrue(first["evidence_id"])
        self.assertTrue(first["legal_unit_id"])
        self.assertIn("viewer_ref", first)
        self.assertIn("source_role", first)
        self.assertNotIn("route_score", first)
        self.assertNotIn("source_sha256", first)
        self.assertNotIn("source_pdf_path", first)
        self.assertNotIn("source_sha256", first["viewer_ref"])
        self.assertNotIn("source_pdf_path", first["viewer_ref"])

        weak = self._post("/legal/uud/search", {"query": "aturan KUHP tentang pencurian"})
        self.assertEqual(weak["public_status"], "no_results")
        self.assertEqual(weak["results"], [])

        citation = self._post("/legal/uud/citation", {"query": "Pasal 1 ayat (3)"})
        self.assertEqual(citation["status"], "found")
        self.assertNotIn("matches", citation)
        self.assertNotIn("context_pack", citation)
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
        pdf_body, pdf_headers = self._get_bytes(viewer["pdf"]["access_url"])
        self.assertEqual(pdf_headers["Content-Type"], "application/pdf")
        self.assertTrue(pdf_body.startswith(b"%PDF"))

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
        ready = self._post("/legal/uud/ask", {"query": "Pasal 1 ayat (3)"})
        self.assertEqual(ready["status"], "answer_ready")
        self.assertEqual(ready["route"], "legal_reference")
        self.assertEqual(ready["intent"], "exact_citation")
        self.assertTrue(ready["citations"])
        self.assertTrue(ready["viewer_refs"])
        self.assertEqual(
            set(ready),
            {
                "status",
                "answer",
                "intent",
                "route",
                "citations",
                "viewer_refs",
                "metadata_facts",
                "legal_relations",
                "answer_scope",
                "warnings",
                "insufficient_reasons",
            },
        )
        for internal in ("matches", "context_pack", "evidence", "answer_type", "reason", "applied_filters"):
            self.assertNotIn(internal, ready)
        self.assertNotIn("source_sha256", ready["citations"][0])
        self.assertNotIn("source_pdf_path", ready["citations"][0])
        self.assertNotIn("source_sha256", ready["viewer_refs"][0])
        self.assertNotIn("source_pdf_path", ready["viewer_refs"][0])

        limited = self._post("/legal/uud/ask", {"query": "negara hukum"})
        self.assertEqual(limited["status"], "limited_answer")
        self.assertEqual(limited["route"], "lexical_fallback")
        self.assertTrue(limited["citations"])
        self.assertTrue(limited["viewer_refs"])

        metadata = self._post("/legal/uud/ask", {"query": "tanggal perubahan kedua UUD"})
        self.assertEqual(metadata["status"], "answer_ready")
        self.assertEqual(metadata["route"], "metadata_fact")
        self.assertEqual(metadata["metadata_facts"][0]["field"], "date")

        place = self._post("/legal/uud/ask", {"query": "tempat penetapan perubahan kedua UUD"})
        self.assertEqual(place["status"], "answer_ready")
        self.assertEqual(place["route"], "metadata_fact")
        self.assertEqual(place["metadata_facts"][0]["field"], "place")
        self.assertEqual(place["metadata_facts"][0]["answer"], "Jakarta")

        relation = self._post("/legal/uud/ask", {"query": "apa saja ayat dalam Pasal 1"})
        self.assertEqual(relation["status"], "answer_ready")
        self.assertEqual(relation["route"], "legal_relation")
        self.assertEqual({row["target_label"] for row in relation["legal_relations"]}, {"(1)", "(2)", "(3)"})

        unsupported_relation = self._post("/legal/uud/ask", {"query": "relasi amandemen Pasal 1"})
        self.assertEqual(unsupported_relation["status"], "insufficient_evidence")
        self.assertEqual(unsupported_relation["route"], "legal_relation")
        self.assertEqual(unsupported_relation["intent"], "legal_relation_lookup")
        self.assertEqual(unsupported_relation["citations"], [])
        self.assertEqual(unsupported_relation["viewer_refs"], [])
        self.assertEqual(unsupported_relation["insufficient_reasons"], ["insufficient_evidence"])

        weak = self._post("/legal/uud/ask", {"query": "aturan KUHP tentang pencurian"})
        self.assertEqual(weak["status"], "insufficient_evidence")
        self.assertEqual(weak["route"], "lexical_fallback")
        self.assertEqual(weak["citations"], [])
        self.assertEqual(weak["viewer_refs"], [])

        missing = self._post("/legal/uud/ask", {"query": "Pasal 999"})
        self.assertEqual(missing["status"], "citation_not_found")
        self.assertEqual(missing["route"], "legal_reference")
        self.assertEqual(missing["citations"], [])
        self.assertEqual(missing["viewer_refs"], [])

    def test_local_dev_cors_only(self) -> None:
        request = Request(
            self.base_url + "/legal/uud/ask",
            data=json.dumps({"query": "Pasal 1 ayat (3)"}).encode("utf-8"),
            headers={"Content-Type": "application/json", "Origin": "http://localhost:5173"},
            method="POST",
        )
        with urlopen(request, timeout=10) as response:
            self.assertEqual(response.headers["Access-Control-Allow-Origin"], "http://localhost:5173")
            self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")
            self.assertEqual(response.headers["Referrer-Policy"], "no-referrer")

        request = Request(
            self.base_url + "/legal/uud/ask",
            data=json.dumps({"query": "Pasal 1 ayat (3)"}).encode("utf-8"),
            headers={"Content-Type": "application/json", "Origin": "https://example.com"},
            method="POST",
        )
        with urlopen(request, timeout=10) as response:
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
            urlopen(request, timeout=10)
        self.assertEqual(error.exception.code, 400)
        self.assertEqual(json.loads(error.exception.read().decode("utf-8"))["reason"], "invalid_json")

        request = Request(
            self.base_url + "/legal/uud/ask",
            data=b"x" * (64 * 1024 + 1),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.assertRaises(HTTPError) as error:
            urlopen(request, timeout=10)
        self.assertEqual(error.exception.code, 413)
        self.assertEqual(json.loads(error.exception.read().decode("utf-8"))["reason"], "request_body_too_large")

        request = Request(
            self.base_url + "/legal/uud/search",
            data=json.dumps({"query": "x", "limit": 0}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.assertRaises(HTTPError) as error:
            urlopen(request, timeout=10)
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
        with urlopen(self.base_url + path, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))

    def _get_bytes(self, path: str) -> tuple[bytes, Any]:
        with urlopen(self.base_url + path, timeout=10) as response:
            return response.read(), response.headers

    def _post(self, path: str, payload: dict) -> dict:
        request = Request(
            self.base_url + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
