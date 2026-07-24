from __future__ import annotations

import unittest

from tjipto.telemetry import EVENT_ATTRIBUTES, Telemetry, event_record


class TelemetryContractTest(unittest.TestCase):
    def test_disabled_telemetry_emits_nothing(self) -> None:
        Telemetry().emit("corpus_load", corpus_id="demo", status="loaded")

    def test_event_names_and_required_attributes_are_bounded(self) -> None:
        self.assertEqual(
            set(EVENT_ATTRIBUTES),
            {"corpus_load", "integrity_failure", "http_request", "retrieval_route", "ci_gate", "release_validation"},
        )
        records: list[dict] = []
        Telemetry(records.append).emit("retrieval_route", corpus_id="demo", route="bm25", status="found")
        self.assertEqual(records, [{"event": "retrieval_route", "attributes": {"corpus_id": "demo", "route": "bm25", "status": "found"}}])

    def test_sensitive_or_unapproved_attributes_never_enter_a_record(self) -> None:
        record = event_record("retrieval_route", corpus_id="demo", route="bm25", status="found", query="secret legal text")
        self.assertNotIn("query", record["attributes"])
        self.assertNotIn("secret legal text", str(record))

    def test_missing_required_attributes_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            event_record("http_request", request_id="r", method="GET", route="health", status_code=200)

    def test_enabled_telemetry_drops_unsafe_events_without_affecting_callers(self) -> None:
        records: list[dict] = []
        Telemetry(records.append).emit("corpus_load", corpus_id="x" * 97, status="loaded")
        self.assertEqual(records, [])
