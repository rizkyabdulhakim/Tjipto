from __future__ import annotations

from contextlib import redirect_stderr
import io
import math
import json
import os
from pathlib import Path
import tempfile
import unittest

from tjipto.corpora.registry import CorpusRegistry
from tjipto.runtime.service import LegalRuntimeService
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
        Telemetry(records.append).emit("retrieval_route", corpus_id="uud", route="bm25", status="found")
        self.assertEqual(records, [{"event": "retrieval_route", "attributes": {"corpus_id": "uud", "route": "bm25", "status": "found"}}])

    def test_sensitive_or_unapproved_attributes_never_enter_a_record(self) -> None:
        record = event_record("retrieval_route", corpus_id="uud", route="bm25", status="found", query="secret legal text")
        self.assertNotIn("query", record["attributes"])
        self.assertNotIn("secret legal text", str(record))

    def test_missing_required_attributes_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            event_record("http_request", request_id="r", method="GET", route="health", status_code=200)

    def test_enabled_telemetry_drops_unsafe_events_without_affecting_callers(self) -> None:
        records: list[dict] = []
        Telemetry(records.append).emit("corpus_load", corpus_id="x" * 97, status="loaded")
        self.assertEqual(records, [])

    def test_closed_values_reject_user_dimensions_and_non_finite_numbers(self) -> None:
        with self.assertRaises(ValueError):
            event_record("http_request", request_id="a" * 32, method="GET", route="legal.attacker", status_code=200, latency_ms=1)
        with self.assertRaises(ValueError):
            event_record("retrieval_route", corpus_id="attacker", route="bm25", status="found")
        with self.assertRaises(ValueError):
            event_record("http_request", request_id="a" * 32, method="GET", route="health", status_code=200, latency_ms=math.nan)
        with self.assertRaises(ValueError):
            event_record("http_request", request_id="a" * 32, method="GET", route="health", status_code=200, latency_ms=float("inf"))

    def test_sink_failure_is_non_fatal_unless_explicitly_strict(self) -> None:
        def fail(_: dict) -> None:
            raise RuntimeError("sink unavailable")

        Telemetry(fail).emit("corpus_load", corpus_id="uud", status="loaded")
        with self.assertRaises(RuntimeError):
            Telemetry(fail, strict=True).emit("corpus_load", corpus_id="uud", status="loaded")

    def test_registered_custom_root_corpus_is_retained(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data").mkdir()
            (root / "data/corpus_registry.json").write_text(json.dumps({"demo": "data/demo/manifest.json"}), encoding="utf-8")
            records: list[dict] = []

            Telemetry(records.append, registry=CorpusRegistry(root)).emit("corpus_load", corpus_id="demo", status="loaded")

            self.assertEqual(records, [{"event": "corpus_load", "attributes": {"corpus_id": "demo", "status": "loaded"}}])

            previous = os.environ.get("TJIPTO_TELEMETRY")
            os.environ["TJIPTO_TELEMETRY"] = "stderr"
            try:
                output = io.StringIO()
                with redirect_stderr(output):
                    LegalRuntimeService(root).telemetry.emit("corpus_load", corpus_id="demo", status="loaded")
            finally:
                if previous is None:
                    os.environ.pop("TJIPTO_TELEMETRY", None)
                else:
                    os.environ["TJIPTO_TELEMETRY"] = previous
            self.assertEqual(json.loads(output.getvalue())["attributes"]["corpus_id"], "demo")
