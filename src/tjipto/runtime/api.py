from __future__ import annotations

from pathlib import Path

from tjipto.runtime.service import LegalRuntimeService


def handle_request(corpus_id: str, action: str, payload: dict, repo_root: Path | None = None) -> dict:
    service = LegalRuntimeService(repo_root)
    if action == "search":
        return service.search(corpus_id, str(payload.get("query", "")), int(payload.get("limit", 10)))
    if action == "citation":
        return service.citation(corpus_id, str(payload.get("query", "")), payload.get("source_role"))
    if action == "viewer":
        return service.viewer(corpus_id, str(payload.get("evidence_id", "")))
    return {"status": "unsupported_action"}
