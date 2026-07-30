from __future__ import annotations

import json
import os
import re
from time import perf_counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4

from tjipto.runtime.api import BadRequest, handle_catalog_pdf_request, handle_catalog_request, handle_pdf_request, handle_request
from tjipto.runtime.service import LegalRuntimeService
from tjipto.telemetry import DEFAULT_TELEMETRY, Telemetry


DEFAULT_LOCAL_ORIGINS = {"http://localhost:5173", "http://127.0.0.1:5173"}
DEFAULT_MAX_REQUEST_BYTES = 64 * 1024


class PayloadTooLarge(ValueError):
    def __init__(self, size: int):
        super().__init__("request_body_too_large")
        self.size = size


def make_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    repo_root: Path | None = None,
) -> ThreadingHTTPServer:
    service = LegalRuntimeService(repo_root)

    class Handler(TjiptoHttpHandler):
        root = repo_root
        runtime_service = service
        telemetry = service.telemetry

    return ThreadingHTTPServer((host, port), Handler)


class TjiptoHttpHandler(BaseHTTPRequestHandler):
    root: Path | None = None
    runtime_service: LegalRuntimeService
    telemetry: Telemetry = DEFAULT_TELEMETRY

    def handle_one_request(self) -> None:
        self._request_id = uuid4().hex
        self._request_started = perf_counter()
        self._request_recorded = False
        super().handle_one_request()

    def do_OPTIONS(self) -> None:
        self._json(204, {})

    def do_GET(self) -> None:
        if self.path == "/health":
            self._json(200, {"status": "ok"})
            return
        route = self._route()
        if route and len(route) == 2 and route[1] == "capabilities":
            self._json(200, handle_request(route[0], "capabilities", {}, self.root, self.runtime_service))
            return
        if route and len(route) == 2 and route[1] == "bookmarks":
            self._json(200, handle_request(route[0], "bookmarks", {}, self.root, self.runtime_service))
            return
        if route and len(route) == 2 and route[1] == "pdf":
            try:
                result = (
                    handle_catalog_pdf_request(self._query_payload(), self.root, self.runtime_service)
                    if route[0] == "catalog"
                    else handle_pdf_request(route[0], self._query_payload(), self.root, self.runtime_service)
                )
            except BadRequest:
                self._json(400, {"status": "bad_request"})
                return
            if result.get("status") != "pdf_access_ready":
                self._json(404, {"status": "not_found"})
                return
            self._pdf(result)
            return
        self._json(404, {"status": "not_found"})

    def do_POST(self) -> None:
        route = self._route()
        actions = {"ask", "search", "citation", "viewer", "bookmarks", "facets"}
        if not route or len(route) != 2 or route[1] not in actions:
            self._json(404, {"status": "not_found"})
            return
        try:
            payload = self._read_json()
            action = "bookmark" if route[1] == "bookmarks" else route[1]
            response = (
                handle_catalog_request(action, payload, self.root, self.runtime_service)
                if route[0] == "catalog"
                else handle_request(route[0], action, payload, self.root, self.runtime_service)
            )
            self._json(200, response)
        except PayloadTooLarge as error:
            self._discard_oversized_body(error.size)
            self._json(413, {"status": "payload_too_large"})
        except BadRequest:
            self._json(400, {"status": "bad_request"})
        except json.JSONDecodeError:
            self._json(400, {"status": "bad_request"})
        except ValueError:
            self._json(400, {"status": "bad_request"})

    def do_DELETE(self) -> None:
        route = self._route()
        if not route or len(route) != 2 or route[1] != "bookmarks" or route[0] == "catalog":
            self._json(404, {"status": "not_found"})
            return
        try:
            response = handle_request(
                route[0],
                "delete_bookmark",
                self._read_json(),
                self.root,
                self.runtime_service,
            )
            self._json(200, response)
        except (BadRequest, json.JSONDecodeError, ValueError):
            self._json(400, {"status": "bad_request"})

    def _read_json(self) -> dict:
        lengths = self.headers.get_all("Content-Length") or []
        if self.headers.get("Transfer-Encoding") or len(lengths) > 1:
            raise ValueError("invalid_content_length")
        raw_length = lengths[0].strip() if lengths else "0"
        if not re.fullmatch(r"0|[1-9][0-9]*", raw_length):
            raise ValueError("invalid_content_length")
        size = int(raw_length)
        if size > _max_request_bytes():
            raise PayloadTooLarge(size)
        if size == 0:
            return {}
        data = self.rfile.read(size)
        if len(data) != size:
            raise ValueError("invalid_content_length")
        payload = json.loads(data.decode("utf-8"))
        if not isinstance(payload, dict):
            raise BadRequest("invalid_json_object")
        return payload

    def _discard_oversized_body(self, declared_size: int) -> None:
        previous_timeout = self.connection.gettimeout()
        try:
            self.connection.settimeout(0.1)
            self.rfile.read(min(declared_size, _max_request_bytes() + 1))
        except (OSError, TimeoutError):
            pass
        finally:
            self.connection.settimeout(previous_timeout)

    def _route(self) -> list[str] | None:
        parts = [part for part in urlsplit(self.path).path.split("/") if part]
        if len(parts) == 3 and parts[0] == "legal":
            return parts[1:]
        if len(parts) == 2 and parts[0] == "uud" and _mode() == "development":
            return parts
        return None

    def _query_payload(self) -> dict:
        parsed = parse_qs(urlsplit(self.path).query, keep_blank_values=True)
        return {key: values[0] for key, values in parsed.items() if values}

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        self._record_http(status)
        body = b"" if status == 204 else json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self._cors_headers()
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _pdf(self, payload: dict[str, Any]) -> None:
        try:
            body = payload["path"].read_bytes()
        except OSError:
            self._json(404, {"status": "not_found"})
            return
        self.send_response(200)
        self._record_http(200)
        self.send_header("Content-Type", "application/pdf")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Disposition", "inline")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self._cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def _cors_headers(self) -> None:
        origin = self.headers.get("Origin")
        if origin in _allowed_origins():
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _record_http(self, status: int) -> None:
        if getattr(self, "_request_recorded", False):
            return
        self._request_recorded = True
        route = _telemetry_route(urlsplit(self.path).path)
        self.telemetry.emit(
            "http_request",
            request_id=getattr(self, "_request_id", uuid4().hex),
            method=getattr(self, "command", "UNKNOWN"),
            route=route,
            status_code=status,
            latency_ms=round((perf_counter() - getattr(self, "_request_started", perf_counter())) * 1000, 3),
        )


def _telemetry_route(path: str) -> str:
    if path == "/health":
        return "health"
    for action in ("ask", "search", "citation", "viewer", "pdf", "bookmarks", "capabilities", "facets"):
        if re.fullmatch(rf"/legal/[^/]+/{action}/?", path):
            return f"legal.{action}"
        if path in {f"/uud/{action}", f"/uud/{action}/"}:
            return f"legacy.{action}"
    return "not_found"


def _mode() -> str:
    return os.environ.get("TJIPTO_MODE", "development").casefold()


def _allowed_origins() -> set[str]:
    configured = os.environ.get("TJIPTO_CORS_ORIGINS")
    if configured:
        return {origin.strip() for origin in configured.split(",") if origin.strip()}
    return DEFAULT_LOCAL_ORIGINS if _mode() == "development" else set()


def _max_request_bytes() -> int:
    try:
        return int(os.environ.get("TJIPTO_MAX_REQUEST_BYTES", str(DEFAULT_MAX_REQUEST_BYTES)))
    except ValueError:
        return DEFAULT_MAX_REQUEST_BYTES


def main() -> None:
    server = make_server(host=os.environ.get("TJIPTO_HOST", "127.0.0.1"), port=int(os.environ.get("TJIPTO_PORT", "8000")))
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
