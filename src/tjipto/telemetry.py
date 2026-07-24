"""Small, local-only telemetry boundary for operational signals."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from typing import Any


EVENT_ATTRIBUTES = {
    "corpus_load": frozenset({"corpus_id", "status"}),
    "integrity_failure": frozenset({"corpus_id", "reason_code"}),
    "http_request": frozenset({"request_id", "method", "route", "status_code", "latency_ms"}),
    "retrieval_route": frozenset({"corpus_id", "route", "status"}),
    "ci_gate": frozenset({"gate", "status", "duration_ms"}),
    "release_validation": frozenset({"status", "forbidden_entry_count", "archive_sha256"}),
}
SENSITIVE_ATTRIBUTES = frozenset({"query", "query_text", "text", "quote", "quoted_text", "token", "tokens", "credential", "password", "secret", "path", "file_path", "email", "user_id"})
MAX_ATTRIBUTE_LENGTH = 96


def event_record(event: str, **attributes: Any) -> dict[str, Any]:
    """Return a bounded record containing only the event's approved fields."""
    allowed = EVENT_ATTRIBUTES[event]
    values = {
        key: value
        for key, value in attributes.items()
        if key in allowed and key not in SENSITIVE_ATTRIBUTES and _safe_value(value)
    }
    if set(values) != allowed:
        missing = sorted(allowed - set(values))
        raise ValueError(f"telemetry attributes missing or unsafe: {', '.join(missing)}")
    return {"event": event, "attributes": values}


def _safe_value(value: Any) -> bool:
    return isinstance(value, (bool, int, float)) or (isinstance(value, str) and len(value) <= MAX_ATTRIBUTE_LENGTH)


class Telemetry:
    """Disabled by default; when enabled, sends JSON-safe records to one local sink."""

    def __init__(self, sink: Callable[[dict[str, Any]], None] | None = None):
        self._sink = sink

    @classmethod
    def from_environment(cls) -> Telemetry:
        return cls(_stderr_sink if os.environ.get("TJIPTO_TELEMETRY") == "stderr" else None)

    def emit(self, event: str, **attributes: Any) -> None:
        if self._sink is not None:
            try:
                self._sink(event_record(event, **attributes))
            except (KeyError, ValueError):
                return


def _stderr_sink(record: dict[str, Any]) -> None:
    print(json.dumps(record, separators=(",", ":"), sort_keys=True), file=sys.stderr)


DEFAULT_TELEMETRY = Telemetry.from_environment()
