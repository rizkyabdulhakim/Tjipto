from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from http.server import ThreadingHTTPServer
from pathlib import Path
import shutil
import socket
import tempfile
import threading
import unittest

from tjipto.runtime.http import make_server
from tjipto.runtime.service import LegalRuntimeService


ROOT = Path(__file__).resolve().parents[1]


class RuntimeTrustBoundaryTest(unittest.TestCase):
    server: ThreadingHTTPServer
    thread: threading.Thread

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = make_server(port=0, repo_root=ROOT)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def test_raw_request_framing_fails_closed(self) -> None:
        cases = (
            (b"Content-Length: -1\r\n", 400),
            (b"Content-Length: nope\r\n", 400),
            (b"Content-Length: 2\r\nContent-Length: 2\r\n", 400),
            (b"Content-Length: 65537\r\n", 413),
        )
        for headers, expected in cases:
            with self.subTest(headers=headers):
                self.assertEqual(self._raw_status(headers), expected)
        body = b"{}"
        self.assertEqual(self._raw_status(f"Content-Length: {len(body)}\r\n".encode(), body), 200)

    def test_incompatible_schema_cannot_answer(self) -> None:
        for schema in (1, "malformed"):
            with self.subTest(schema=schema), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / "data/final/uud").mkdir(parents=True)
                shutil.copy2(ROOT / "data/corpus_registry.json", root / "data/corpus_registry.json")
                manifest = json.loads((ROOT / "data/final/uud/manifest.json").read_text(encoding="utf-8"))
                manifest["schema_version"] = schema
                (root / "data/final/uud/manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                result = LegalRuntimeService(root).ask("uud", "Pasal 7A")
                self.assertEqual(result["status"], "unsupported_corpus")
                self.assertFalse(result.get("citations"))

    def test_bookmark_read_write_is_concurrency_safe_and_sorted(self) -> None:
        service = LegalRuntimeService(ROOT)
        evidence_id = service._store("uud").evidence[0]["evidence_id"]
        errors: list[BaseException] = []

        def write(index: int) -> None:
            try:
                service.bookmark("uud", evidence_id, note=str(index))
            except BaseException as error:  # pragma: no cover - assertion target
                errors.append(error)

        def read(_: int) -> None:
            try:
                rows = service.bookmarks("uud")["bookmarks"]
                self.assertEqual([row["bookmark_id"] for row in rows], sorted(row["bookmark_id"] for row in rows))
            except BaseException as error:  # pragma: no cover - assertion target
                errors.append(error)

        with ThreadPoolExecutor(max_workers=12) as pool:
            list(pool.map(write, range(20)))
            list(pool.map(read, range(20)))
        self.assertEqual(errors, [])

    def _raw_status(self, headers: bytes, body: bytes = b"") -> int:
        host = str(self.server.server_address[0])
        port = int(self.server.server_address[1])
        request = b"POST /legal/uud/ask HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n" + headers + b"\r\n" + body
        with socket.create_connection((host, port), timeout=5) as client:
            client.sendall(request)
            response = b""
            while chunk := client.recv(4096):
                response += chunk
        return int(response.split(b" ", 2)[1])


if __name__ == "__main__":
    unittest.main()
