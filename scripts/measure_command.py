from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess  # nosec B404
import sys
import time
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import platform

from tjipto.core.manifest import artifact_set_digest
from tjipto.corpora.registry import CorpusRegistry
from tjipto.telemetry import event_record


RSS_LIMIT_BYTES = 721 * 1024 * 1024
RSS_RATIO_LIMIT = 1.05


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


def compare_pytest_resources(first: dict, second: dict, identity: dict) -> dict:
    """Apply the release RSS policy and retain wall time as diagnostic evidence."""
    rss_values = (first.get("peak_rss_bytes"), second.get("peak_rss_bytes"))
    rss_ratio = max(rss_values) / min(rss_values) if all(isinstance(value, int) and value > 0 for value in rss_values) else None
    wall_values = (first.get("wall_seconds"), second.get("wall_seconds"))
    wall_ratio = max(wall_values) / min(wall_values) if all(isinstance(value, (int, float)) and value > 0 for value in wall_values) else None
    return {
        "run_identity_id": identity["run_identity_id"],
        "both_pytest_runs_pass": first.get("exit_code") == 0 and second.get("exit_code") == 0,
        "run_1_peak_rss_bytes": rss_values[0],
        "run_2_peak_rss_bytes": rss_values[1],
        "rss_limit_bytes": RSS_LIMIT_BYTES,
        "rss_ratio": rss_ratio,
        "rss_limit_pass": rss_ratio is not None and max(rss_values) <= RSS_LIMIT_BYTES,
        "rss_stability_pass": rss_ratio is not None and rss_ratio <= RSS_RATIO_LIMIT,
        "run_1_wall_seconds": wall_values[0],
        "run_2_wall_seconds": wall_values[1],
        "wall_ratio": wall_ratio,
        "wall_status": "unavailable" if wall_ratio is None else ("stable" if wall_ratio <= 1.10 else "variable"),
        "wall_policy": "diagnostic_not_benchmark",
        "run_1_started_at_utc": first.get("started_at_utc"),
        "run_1_finished_at_utc": first.get("finished_at_utc"),
        "run_2_started_at_utc": second.get("started_at_utc"),
        "run_2_finished_at_utc": second.get("finished_at_utc"),
        "runner_identity": {key: identity.get(key) for key in (
            "runner_name", "runner_os", "runner_arch", "runner_environment",
            "runner_image_os", "runner_image_version", "platform",
        )},
    }


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
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    process = subprocess.Popen(command)  # nosec B603
    peak = 0
    while process.poll() is None:
        peak = max(peak, _rss_bytes(process))
        time.sleep(0.05)
    peak = max(peak, _rss_bytes(process))
    finished_at = datetime.now(timezone.utc)
    result = {
        "command": command,
        "exit_code": process.returncode,
        "wall_seconds": round(time.perf_counter() - started, 3),
        "peak_rss_bytes": peak,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
    }
    if args.gate:
        identity = _execution_identity()
        record = event_record(
            "ci_gate",
            gate=args.gate,
            status="passed" if process.returncode == 0 else "failed",
            duration_ms=round(result["wall_seconds"] * 1000, 3),
        ) | result | identity
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
    identity = _execution_identity(root)
    return identity | {
        "commit_sha": _command_output(root, "git", "rev-parse", "HEAD"),
        "tree_sha": _command_output(root, "git", "rev-parse", "HEAD^{tree}"),
        "parent_sha": _command_output(root, "git", "rev-parse", "HEAD^"),
        "branch": os.environ.get("GITHUB_HEAD_REF") or os.environ.get("GITHUB_REF_NAME") or _command_output(root, "git", "branch", "--show-current"),
        "workflow_ref": identity["workflow_ref"],
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
        "pytest_version": importlib.metadata.version("pytest"),
        "pymupdf_version": importlib.metadata.version("PyMuPDF"),
        "mupdf_version": _mupdf_version(),
        "runner_image_os": os.environ.get("ImageOS"),
        "runner_image_version": os.environ.get("ImageVersion"),
        "runner_os": os.environ.get("RUNNER_OS") or platform.system(),
        "runner_arch": os.environ.get("RUNNER_ARCH") or platform.machine(),
        "runner_environment": os.environ.get("RUNNER_ENVIRONMENT"),
        "runner_name": os.environ.get("RUNNER_NAME"),
        "platform": platform.platform(),
        "corpora": corpora,
    }


def _execution_identity(root: Path | None = None) -> dict:
    root = root or Path(__file__).resolve().parents[1]
    values = {
        "repository": os.environ.get("GITHUB_REPOSITORY") or _command_output(root, "git", "config", "--get", "remote.origin.url"),
        "workflow_name": os.environ.get("GITHUB_WORKFLOW"),
        "workflow_ref": os.environ.get("TJIPTO_JOB_WORKFLOW_REF") or os.environ.get("GITHUB_WORKFLOW_REF"),
        "workflow_sha": os.environ.get("TJIPTO_JOB_WORKFLOW_SHA") or os.environ.get("GITHUB_WORKFLOW_SHA"),
        "workflow_repository": os.environ.get("TJIPTO_JOB_WORKFLOW_REPOSITORY"),
        "workflow_file_path": os.environ.get("TJIPTO_JOB_WORKFLOW_FILE_PATH"),
        "commit_sha": os.environ.get("GITHUB_SHA") or _command_output(root, "git", "rev-parse", "HEAD"),
        "tree_sha": _command_output(root, "git", "rev-parse", "HEAD^{tree}"),
        "parent_sha": _command_output(root, "git", "rev-parse", "HEAD^"),
        "ref": os.environ.get("GITHUB_REF") or _command_output(root, "git", "symbolic-ref", "-q", "HEAD"),
        "branch": os.environ.get("GITHUB_HEAD_REF") or os.environ.get("GITHUB_REF_NAME") or _command_output(root, "git", "branch", "--show-current"),
        "run_id": os.environ.get("GITHUB_RUN_ID"),
        "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "job_key": os.environ.get("GITHUB_JOB"),
        "job_check_run_id": os.environ.get("TJIPTO_JOB_CHECK_RUN_ID"),
    }
    run_fields = {
        key: values[key]
        for key in (
            "repository",
            "workflow_ref",
            "workflow_sha",
            "workflow_repository",
            "workflow_file_path",
            "commit_sha",
            "tree_sha",
            "parent_sha",
            "ref",
            "run_id",
            "run_attempt",
        )
    }
    values["run_identity_id"] = sha256(json.dumps(run_fields, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    values["job_identity_id"] = _job_identity_id(values)
    if os.environ.get("GITHUB_ACTIONS") == "true":
        _validate_ci_identity(values)
    return values


def _job_identity_id(identity: dict) -> str:
    fields = {key: identity.get(key) for key in ("run_identity_id", "job_key", "job_check_run_id")}
    return sha256(json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _validate_ci_identity(identity: dict) -> None:
    required = (
        "repository",
        "workflow_ref",
        "workflow_sha",
        "workflow_repository",
        "workflow_file_path",
        "commit_sha",
        "tree_sha",
        "parent_sha",
        "ref",
        "run_id",
        "run_attempt",
        "job_key",
        "job_check_run_id",
    )
    missing = [key for key in required if not identity.get(key)]
    if missing:
        raise ValueError(f"missing typed CI identity: {', '.join(missing)}")
    if not str(identity["job_check_run_id"]).isdigit() or int(identity["job_check_run_id"]) <= 0:
        raise ValueError("invalid typed CI identity: job_check_run_id")
    if not str(identity["workflow_sha"]).isalnum() or len(str(identity["workflow_sha"])) != 40:
        raise ValueError("invalid typed CI identity: workflow_sha")
    if not str(identity["workflow_file_path"]).startswith(".github/workflows/"):
        raise ValueError("invalid typed CI identity: workflow_file_path")


def _mupdf_version() -> str:
    import pymupdf

    return str(getattr(pymupdf, "mupdf_version", None) or getattr(pymupdf, "VersionBind", "unknown"))


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
