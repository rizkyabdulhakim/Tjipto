from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess  # nosec B404
import sys
import time

from tjipto.telemetry import event_record


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
    import argparse

    parser = argparse.ArgumentParser(description="Run one CI gate and write bounded JSON evidence.")
    parser.add_argument("--gate")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--environment-report", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
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
    result = {"command": command, "exit_code": process.returncode, "wall_seconds": round(time.perf_counter() - started, 3), "peak_rss_bytes": peak}
    if args.gate:
        record = event_record("ci_gate", gate=args.gate, status="passed" if process.returncode == 0 else "failed", duration_ms=round(result["wall_seconds"] * 1000, 3))
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            with args.report.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
    if args.environment_report:
        args.environment_report.parent.mkdir(parents=True, exist_ok=True)
        args.environment_report.write_text(json.dumps(_environment_report(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return process.returncode


def _environment_report() -> dict:
    root = Path.cwd()
    manifest = root / "data" / "final" / "uud" / "manifest.json"
    files = {name: _sha256(root / name) for name in ("requirements.lock", "apps/web/package-lock.json", "data/final/uud/manifest.json")}
    return {
        "commit_sha": _command_output(root, "git", "rev-parse", "HEAD"),
        "tree_sha": _command_output(root, "git", "rev-parse", "HEAD^{tree}"),
        "lock_sha256": files["requirements.lock"],
        "node_version": _command_output(root, "node", "--version"),
        "npm_version": _command_output(root, "npm", "--version"),
        "package_lock_sha256": files["apps/web/package-lock.json"],
        "python_version": sys.version,
        "runner_image_os": os.environ.get("ImageOS"),
        "runner_image_version": os.environ.get("ImageVersion"),
        "artifact_manifest_sha256": files["data/final/uud/manifest.json"],
        "artifact_files": json.loads(manifest.read_text(encoding="utf-8")).get("files", {}),
    }


def _command_output(root: Path, *command: str) -> str:
    if sys.platform == "win32" and command[0] == "npm":
        command = ("cmd.exe", "/d", "/s", "/c", *command)
    return subprocess.check_output(command, cwd=root, text=True).strip()  # nosec B603 B607 - fixed CI commands.


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
