from __future__ import annotations

import json
import threading
import unittest
from pathlib import Path
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

    def test_uud_ask_examples(self) -> None:
        ready = self._post("/uud/ask", {"query": "Pasal 1 ayat (3)"})
        self.assertEqual(ready["status"], "answer_ready")
        self.assertTrue(ready["citations"])
        self.assertTrue(ready["viewer_refs"])

        limited = self._post("/uud/ask", {"query": "negara hukum"})
        self.assertEqual(limited["status"], "limited_answer")
        self.assertTrue(limited["citations"])
        self.assertTrue(limited["viewer_refs"])

        weak = self._post("/uud/ask", {"query": "aturan KUHP tentang pencurian"})
        self.assertEqual(weak["status"], "insufficient_evidence")
        self.assertEqual(weak["citations"], [])
        self.assertEqual(weak["viewer_refs"], [])

        missing = self._post("/uud/ask", {"query": "Pasal 999"})
        self.assertEqual(missing["status"], "citation_not_found")
        self.assertEqual(missing["citations"], [])
        self.assertEqual(missing["viewer_refs"], [])

    def test_local_dev_cors_only(self) -> None:
        request = Request(
            self.base_url + "/uud/ask",
            data=json.dumps({"query": "Pasal 1 ayat (3)"}).encode("utf-8"),
            headers={"Content-Type": "application/json", "Origin": "http://localhost:5173"},
            method="POST",
        )
        with urlopen(request, timeout=10) as response:
            self.assertEqual(response.headers["Access-Control-Allow-Origin"], "http://localhost:5173")

        request = Request(
            self.base_url + "/uud/ask",
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
            self.base_url + "/uud/ask",
            data=b"{",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.assertRaises(HTTPError) as error:
            urlopen(request, timeout=10)
        self.assertEqual(error.exception.code, 400)
        self.assertEqual(json.loads(error.exception.read().decode("utf-8"))["reason"], "invalid_json")

        request = Request(
            self.base_url + "/uud/ask",
            data=b"x" * (64 * 1024 + 1),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.assertRaises(HTTPError) as error:
            urlopen(request, timeout=10)
        self.assertEqual(error.exception.code, 413)
        self.assertEqual(json.loads(error.exception.read().decode("utf-8"))["reason"], "request_body_too_large")

    def _get(self, path: str) -> dict:
        with urlopen(self.base_url + path, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))

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
