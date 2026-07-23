from __future__ import annotations

import json
from pathlib import Path
import threading
import unittest
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from tjipto.runtime.http import make_server


ROOT = Path(__file__).resolve().parents[1]
_FORBIDDEN = ("evidence_id", "legal_unit_id", "source_document_id", "bbox_id", "source_bbox_refs", "manifest_digest", "artifact_set_digest", "context_pack")


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

    def test_public_capabilities_and_search_are_closed(self) -> None:
        capabilities = self._get("/legal/uud/capabilities")
        self.assertEqual(capabilities, {"status": "ok", "capabilities": ["search", "ask", "citation", "viewer", "bookmarks"]})
        result = self._post("/legal/uud/search", {"query": "UUD 1945", "limit": 2})
        self.assertEqual(result["status"], "found")
        self.assertTrue(result["results"])
        self._assert_public(result)
        self.assertEqual(set(result["results"][0]), {"title", "label", "snippet", "source_role", "page_numbers", "viewer_target"})
        self.assertTrue(result["results"][0]["viewer_target"]["public_target_id"])

    def test_ask_citation_viewer_pdf_and_bookmark_use_only_public_targets(self) -> None:
        asked = self._post("/legal/uud/ask", {"query": "Pasal 16 UUD konsolidasi"})
        self.assertEqual(asked["status"], "answer_ready")
        self._assert_public(asked)
        support = asked["supports"][0]
        self.assertEqual(set(support), {
            "public_support_id", "support_kind", "panel_section", "fact_kind", "label", "role_label", "text", "layout_lines", "copy_text",
            "source_label", "source_role", "page_numbers", "legal_citation_available", "relevant_quote_eligible", "viewer_target",
        })
        self.assertEqual(support["panel_section"], "Kutipan Relevan")
        self.assertNotIn("\nPresiden\nmembentuk", support["copy_text"])
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
        self.assertEqual(set(saved["bookmark"]), {"public_bookmark_id", "public_target_id", "note", "created_at", "status"})
        bookmarks = self._get("/legal/uud/bookmarks")
        self._assert_public(bookmarks)
        self.assertTrue(bookmarks["bookmarks"])

    def test_citation_shares_the_support_contract(self) -> None:
        result = self._post("/legal/uud/citation", {"query": "Pasal 1 ayat (3)"})
        self.assertEqual(result["status"], "found")
        self.assertTrue(result["supports"])
        self.assertEqual(result["supports"][0]["panel_section"], "Kutipan Relevan")
        self._assert_public(result)

    def test_groups_preserve_members_and_keep_nonlegal_sections_out_of_quotes(self) -> None:
        result = self._post("/legal/uud/ask", {"query": "siapa wakil ketua yang tercantum dalam Perubahan Pertama?"})
        self.assertTrue(result["support_groups"])
        self._assert_public(result)
        for group in result["support_groups"]:
            self.assertEqual(group["member_count"], len(group["members"]))
            self.assertTrue(group["public_group_id"])
            for member in group["members"]:
                self.assertIn(member, result["supports"])
                if member["panel_section"] != "Kutipan Relevan":
                    self.assertFalse(member["legal_citation_available"])
                    self.assertFalse(member["relevant_quote_eligible"])

    def test_public_payload_rejects_unknown_and_legacy_fields(self) -> None:
        for path, payload in (
            ("/legal/uud/viewer", {"target": "bad", "evidence_id": "forged"}),
            ("/legal/uud/bookmarks", {"evidence_id": "forged"}),
            ("/legal/uud/ask", {"query": "Pasal 1", "context_pack": {}}),
        ):
            with self.assertRaises(HTTPError) as error:
                self._post(path, payload)
            self.assertEqual(error.exception.code, 400)
            self.assertEqual(json.loads(error.exception.read().decode("utf-8")), {"status": "bad_request", "reason": "invalid_request"})

    def test_invalid_target_and_pdf_query_do_not_leak(self) -> None:
        viewer = self._post("/legal/uud/viewer", {"target": "not-a-target"})
        self.assertEqual(viewer["status"], "not_found")
        self._assert_public(viewer)
        with self.assertRaises(HTTPError) as error:
            self._get("/legal/uud/pdf?evidence_id=forged")
        self.assertEqual(error.exception.code, 400)
        self._assert_public(json.loads(error.exception.read().decode("utf-8")))

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


if __name__ == "__main__":
    unittest.main()
