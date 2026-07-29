from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tracemalloc


_REPORT = os.environ.get("TJIPTO_MEMORY_REPORT")
_TRACE = os.environ.get("TJIPTO_MEMORY_TRACEMALLOC") == "1"
_ROWS: list[dict] = []
_LAST_RSS = 0
_LAST_HWM = 0
_START_SNAPSHOT = None


def pytest_sessionstart(session) -> None:
    del session
    global _LAST_RSS, _LAST_HWM, _START_SNAPSHOT
    if _TRACE:
        tracemalloc.start(10)
        _START_SNAPSHOT = tracemalloc.take_snapshot()
    _LAST_RSS, _LAST_HWM = _linux_memory()


def pytest_runtest_logreport(report) -> None:
    if report.when != "teardown":
        return
    global _LAST_RSS, _LAST_HWM
    rss, hwm = _linux_memory()
    traced_current, traced_peak = tracemalloc.get_traced_memory() if _TRACE else (0, 0)
    _ROWS.append(
        {
            "nodeid": report.nodeid,
            "rss_bytes": rss,
            "rss_delta_bytes": rss - _LAST_RSS,
            "peak_rss_bytes": hwm,
            "peak_rss_delta_bytes": hwm - _LAST_HWM,
            "traced_current_bytes": traced_current,
            "traced_peak_bytes": traced_peak,
            "untraced_rss_bytes": max(0, rss - traced_current),
        }
    )
    _LAST_RSS, _LAST_HWM = rss, hwm


def pytest_sessionfinish(session, exitstatus) -> None:
    del session
    if not _REPORT:
        return
    report = {
        "exit_status": exitstatus,
        "tests": _ROWS,
        "largest_peak_deltas": sorted(_ROWS, key=lambda row: row["peak_rss_delta_bytes"], reverse=True)[:25],
        "largest_rss": sorted(_ROWS, key=lambda row: row["rss_bytes"], reverse=True)[:25],
    }
    if _TRACE and _START_SNAPSHOT is not None:
        snapshot = tracemalloc.take_snapshot()
        report["python_allocation_growth"] = [
            {"bytes": stat.size_diff, "count": stat.count_diff, "traceback": str(stat.traceback)}
            for stat in snapshot.compare_to(_START_SNAPSHOT, "lineno")[:50]
        ]
    path = Path(_REPORT)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _linux_memory() -> tuple[int, int]:
    if not sys.platform.startswith("linux"):
        return 0, 0
    values: dict[str, int] = {}
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        name, _, value = line.partition(":")
        if name in {"VmRSS", "VmHWM"}:
            values[name] = int(value.split()[0]) * 1024
    return values.get("VmRSS", 0), values.get("VmHWM", 0)
