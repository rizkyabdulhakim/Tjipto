from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess  # nosec B404
import sys
import time

from tjipto.core.manifest import artifact_set_digest
from tjipto.corpora.registry import CorpusRegistry
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
    parser.add_argument("--result-report", type=Path)
    parser.add_argument("--environment-report", type=Path)
    parser.add_argument("--corpus", action="append", default=[])
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
    if args.result_report:
        args.result_report.parent.mkdir(parents=True, exist_ok=True)
        args.result_report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.environment_report:
        args.environment_report.parent.mkdir(parents=True, exist_ok=True)
        args.environment_report.write_text(json.dumps(_environment_report(args.corpus), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return process.returncode


def _environment_report(corpus_ids: list[str] | None = None) -> dict:
    root = Path(__file__).resolve().parents[1]
    registry = CorpusRegistry(root)
    selected = corpus_ids or list(registry.corpus_ids())
    corpora = {}
    for corpus_id in selected:
        config = registry.resolve(corpus_id)
        if config is None:
            raise ValueError(f"unavailable registered corpus: {corpus_id}")
        manifest = config.manifest
        corpora[config.corpus_id] = {
            "artifact_set_digest": artifact_set_digest(manifest),
            "extractor_fingerprint": manifest.get("extractor_fingerprint"),
            "manifest_sha256": _sha256(config.manifest_path),
            "schema_version": manifest.get("schema_version"),
        }
    files = {name: _sha256(root / name) for name in ("requirements.lock", "apps/web/package-lock.json", "data/corpus_registry.json")}
    return {
        "commit_sha": _command_output(root, "git", "rev-parse", "HEAD"),
        "tree_sha": _command_output(root, "git", "rev-parse", "HEAD^{tree}"),
        "parent_sha": _command_output(root, "git", "rev-parse", "HEAD^"),
        "branch": os.environ.get("GITHUB_HEAD_REF") or os.environ.get("GITHUB_REF_NAME") or _command_output(root, "git", "branch", "--show-current"),
        "workflow_ref": os.environ.get("GITHUB_WORKFLOW_REF"),
        "run_id": os.environ.get("GITHUB_RUN_ID"),
        "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "event": os.environ.get("GITHUB_EVENT_NAME"),
        "job_name": os.environ.get("GITHUB_JOB"),
        "python_lock_sha256": files["requirements.lock"],
        "node_version": _command_output(root, "node", "--version"),
        "npm_version": _command_output(root, "npm", "--version"),
        "package_lock_sha256": files["apps/web/package-lock.json"],
        "corpus_registry_sha256": files["data/corpus_registry.json"],
        "python_version": sys.version.split()[0],
        "runner_image_os": os.environ.get("ImageOS"),
        "runner_image_version": os.environ.get("ImageVersion"),
        "corpora": corpora,
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
