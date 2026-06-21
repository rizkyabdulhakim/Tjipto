from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from tjipto.runtime.api import handle_request


LOCAL_ORIGINS = {"http://localhost:5173", "http://127.0.0.1:5173"}
MAX_REQUEST_BYTES = 64 * 1024


class PayloadTooLarge(ValueError):
    pass


def make_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    repo_root: Path | None = None,
) -> ThreadingHTTPServer:
    class Handler(TjiptoHttpHandler):
        root = repo_root

    return ThreadingHTTPServer((host, port), Handler)


class TjiptoHttpHandler(BaseHTTPRequestHandler):
    root: Path | None = None

    def do_OPTIONS(self) -> None:
        self._json(204, {})

    def do_GET(self) -> None:
        if self.path == "/health":
            self._json(200, {"status": "ok"})
            return
        self._json(404, {"status": "not_found"})

    def do_POST(self) -> None:
        if self.path != "/uud/ask":
            self._json(404, {"status": "not_found"})
            return
        try:
            payload = self._read_json()
            response = handle_request("uud", "ask", payload, self.root)
            self._json(200, response)
        except PayloadTooLarge:
            self._json(413, {"status": "payload_too_large", "reason": "request_body_too_large"})
        except json.JSONDecodeError:
            self._json(400, {"status": "bad_request", "reason": "invalid_json"})
        except ValueError:
            self._json(400, {"status": "bad_request", "reason": "invalid_content_length"})

    def _read_json(self) -> dict:
        size = int(self.headers.get("Content-Length", "0") or "0")
        if size > MAX_REQUEST_BYTES:
            raise PayloadTooLarge
        if size == 0:
            return {}
        data = self.rfile.read(size)
        payload = json.loads(data.decode("utf-8"))
        return payload if isinstance(payload, dict) else {}

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = b"" if status == 204 else json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors_headers()
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _cors_headers(self) -> None:
        origin = self.headers.get("Origin")
        if origin in LOCAL_ORIGINS:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def log_message(self, format: str, *args: Any) -> None:
        return


def main() -> None:
    server = make_server()
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
