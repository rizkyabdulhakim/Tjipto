from __future__ import annotations

import json
from pathlib import Path
import subprocess  # nosec B404
import sys
import time


def _rss_bytes(process: subprocess.Popen) -> int:
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]

        counters = Counters(ctypes.sizeof(Counters))
        memory_info = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
        memory_info.argtypes = (wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD)
        memory_info.restype = wintypes.BOOL
        if memory_info(wintypes.HANDLE(process._handle), ctypes.byref(counters), counters.cb):
            return int(counters.PeakWorkingSetSize)
    status = Path(f"/proc/{process.pid}/status")
    if status.exists():
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    return 0


def main(argv: list[str] | None = None) -> int:
    command = argv or sys.argv[1:]
    if not command:
        print("usage: python scripts/measure_command.py command [args...]", file=sys.stderr)
        return 2
    started = time.perf_counter()
    process = subprocess.Popen(command)  # nosec B603
    peak = 0
    while process.poll() is None:
        peak = max(peak, _rss_bytes(process))
        time.sleep(0.05)
    peak = max(peak, _rss_bytes(process))
    print(json.dumps({"command": command, "exit_code": process.returncode, "wall_seconds": round(time.perf_counter() - started, 3), "peak_rss_bytes": peak}, sort_keys=True))
    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
